"""
pinn.py
=======
Standard MLP Physics-Informed Neural Network (PINN) map representation for 2D advection-diffusion scalar fields.

Inputs [t, x, y] are normalized prior to passing into the MLP:
- Spatial (x, y) coordinates normalized to [-1, 1] based on x_bounds and y_bounds.
- Temporal t coordinate normalized to [0, 1] based on t_max.

PinnDomainConfig is passed when initializing PinnFieldMap instance (e.g. pinn_map = PinnFieldMap(config)),
binding domain bounds metadata to the map instance for fit() operations.
Physical parameters (flow velocity v_flow and diffusivity D) are included in PinnParams pytree
so they are jointly learned alongside neural network weights via JAX automatic differentiation.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax  # type: ignore[import-untyped]
from jax import Array


class PinnParams(NamedTuple):
    """Pytree holding ALL trainable parameters (MLP weights, biases, flow velocity, and diffusivity).

    Attributes
    ----------
    weights : list of Array
        Weight matrices for MLP layers.
    biases : list of Array
        Bias vectors for MLP layers.
    v_flow : Array
        Advection flow velocity vector [u_m_s, v_m_s], shape (2,).
    D : Array
        Diffusivity coefficient [m^2/s], scalar shape () or (1,).
    """

    weights: list[Array]
    biases: list[Array]
    v_flow: Array
    D: Array


class PinnDomainConfig(NamedTuple):
    """Static metadata configuration for domain bounds and normalization.

    Attributes
    ----------
    x_bounds : tuple of float
        Spatial x domain bounds [x_min, x_max].
    y_bounds : tuple of float
        Spatial y domain bounds [y_min, y_max].
    t_max : float
        Maximum time horizon [s].
    """

    x_bounds: tuple[float, float] = (-150.0, 150.0)
    y_bounds: tuple[float, float] = (-150.0, 150.0)
    t_max: float = 100.0


class PinnFieldMap:
    """Standard MLP Physics-Informed Neural Network scalar field map."""

    def __init__(self, config: PinnDomainConfig = PinnDomainConfig()) -> None:
        """Initializes PINN map instance with static domain bounds metadata config."""
        self.config = config

    @staticmethod
    def init_params(
        key: Array,
        v_flow_init: Array | None = None,
        D_init: float | Array = 0.5,
        in_dim: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 4,
    ) -> PinnParams:
        """Initializes trainable parameters pytree."""
        keys = jax.random.split(key, num_layers + 1)
        layer_dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [1]

        weights = []
        biases = []

        for i in range(len(layer_dims) - 1):
            fan_in, fan_out = layer_dims[i], layer_dims[i + 1]
            limit = jnp.sqrt(6.0 / (fan_in + fan_out))
            w = jax.random.uniform(keys[i], (fan_in, fan_out), minval=-limit, maxval=limit)
            b = jnp.zeros((fan_out,))
            weights.append(w)
            biases.append(b)

        if v_flow_init is None:
            v_flow_arr = jnp.zeros((2,), dtype=jnp.float64)
        else:
            v_flow_arr = jnp.asarray(v_flow_init, dtype=jnp.float64)

        D_arr = jnp.asarray(D_init, dtype=jnp.float64)

        return PinnParams(
            weights=weights,
            biases=biases,
            v_flow=v_flow_arr,
            D=D_arr,
        )

    @staticmethod
    def normalize_inputs(p: Array, config: PinnDomainConfig = PinnDomainConfig()) -> Array:
        """Normalizes spacetime input coordinates: (x, y) -> [-1, 1] and t -> [0, 1]."""
        t = p[..., 0]
        x = p[..., 1]
        y = p[..., 2]

        t_norm = jnp.clip(t / config.t_max, 0.0, 1.0)
        x_norm = 2.0 * (x - config.x_bounds[0]) / (config.x_bounds[1] - config.x_bounds[0]) - 1.0
        y_norm = 2.0 * (y - config.y_bounds[0]) / (config.y_bounds[1] - config.y_bounds[0]) - 1.0

        return jnp.stack([t_norm, x_norm, y_norm], axis=-1)

    @staticmethod
    def forward(params: PinnParams, p: Array, config: PinnDomainConfig = PinnDomainConfig()) -> Array:
        """Evaluates neural field map at unnormalized input coordinates."""
        h = PinnFieldMap.normalize_inputs(p, config)

        for w, b in zip(params.weights[:-1], params.biases[:-1], strict=True):
            h = jnp.tanh(jnp.dot(h, w) + b)

        out = jnp.dot(h, params.weights[-1]) + params.biases[-1]
        return jnp.squeeze(out, axis=-1)

    @staticmethod
    def pde_residual(
        params: PinnParams,
        config: PinnDomainConfig = PinnDomainConfig(),
        t: float | Array = 0.0,
        x: float | Array = 0.0,
        y: float | Array = 0.0,
    ) -> Array:
        """Computes physical advection-diffusion PDE residual using automatic differentiation."""
        def scalar_fn(t_val, x_val, y_val):
            p_val = jnp.stack([t_val, x_val, y_val], axis=-1)
            return PinnFieldMap.forward(params, p_val, config)

        dt = jax.grad(scalar_fn, argnums=0)(t, x, y)
        dx = jax.grad(scalar_fn, argnums=1)(t, x, y)
        dy = jax.grad(scalar_fn, argnums=2)(t, x, y)

        dx2 = jax.grad(lambda _t, _x, _y: jax.grad(scalar_fn, argnums=1)(_t, _x, _y), argnums=1)(t, x, y)
        dy2 = jax.grad(lambda _t, _x, _y: jax.grad(scalar_fn, argnums=2)(_t, _x, _y), argnums=2)(t, x, y)

        residual = dt + params.v_flow[0] * dx + params.v_flow[1] * dy - params.D * (dx2 + dy2)
        return residual

    @staticmethod
    def sample_collocation_points(
        trajectory_points: Array,
        t_curr: float | Array,
        num_colloc: int,
        key: Array,
        margin: float = 15.0,
        config: PinnDomainConfig | None = None,
    ) -> Array:
        """Generates random PDE collocation points bounded to current trajectory bounding box."""
        if trajectory_points.shape[1] == 3:
            x_coords = trajectory_points[:, 1]
            y_coords = trajectory_points[:, 2]
        else:
            x_coords = trajectory_points[:, 0]
            y_coords = trajectory_points[:, 1]

        x_min_box = jnp.min(x_coords) - margin
        x_max_box = jnp.max(x_coords) + margin
        y_min_box = jnp.min(y_coords) - margin
        y_max_box = jnp.max(y_coords) + margin

        if config is not None:
            x_min_box = jnp.clip(x_min_box, config.x_bounds[0], config.x_bounds[1])
            x_max_box = jnp.clip(x_max_box, config.x_bounds[0], config.x_bounds[1])
            y_min_box = jnp.clip(y_min_box, config.y_bounds[0], config.y_bounds[1])
            y_max_box = jnp.clip(y_max_box, config.y_bounds[0], config.y_bounds[1])

        k1, k2, k3 = jax.random.split(key, 3)

        colloc_t = jax.random.uniform(k1, (num_colloc,), minval=0.0, maxval=jnp.maximum(jnp.asarray(t_curr), 1e-3))
        colloc_x = jax.random.uniform(k2, (num_colloc,), minval=x_min_box, maxval=x_max_box)
        colloc_y = jax.random.uniform(k3, (num_colloc,), minval=y_min_box, maxval=y_max_box)

        return jnp.stack([colloc_t, colloc_x, colloc_y], axis=-1)

    def fit(
        self,
        params: PinnParams,
        opt_state: Any,
        optimizer: optax.GradientTransformation,
        buf_pts: Array,
        buf_vals: Array,
        t_curr: float | Array,
        key: Array,
        num_steps: int = 5,
        num_colloc: int = 40,
        margin: float = 15.0,
        w_pde: float = 0.05,
    ) -> tuple[PinnParams, Any, float]:
        """Fits PINN neural map and joint PDE parameters (v_flow, D) online using instance domain config."""
        colloc_pts = PinnFieldMap.sample_collocation_points(
            buf_pts, t_curr, num_colloc=num_colloc, key=key, margin=margin, config=self.config
        )

        curr_params = params
        curr_opt_state = opt_state
        last_loss = 0.0

        for _ in range(num_steps):
            loss_val, grads = jax.value_and_grad(pinn_loss_fn)(
                curr_params, self.config, buf_pts, buf_vals, colloc_pts, w_pde=w_pde
            )
            updates, curr_opt_state = optimizer.update(grads, curr_opt_state, curr_params)
            curr_params = optax.apply_updates(curr_params, updates)
            last_loss = float(loss_val)

        return curr_params, curr_opt_state, last_loss


def pinn_loss_fn(
    params: PinnParams,
    config: PinnDomainConfig,
    data_points: Array,  # shape (N, 3) [t, x, y]
    obs_vals: Array,     # shape (N,)
    colloc_points: Array,  # shape (K, 3) [t, x, y]
    w_pde: float = 0.05,
) -> Array:
    """Computes joint Data MSE Loss + PDE Physics Residual Loss using trainable v_flow and D from params.

    Parameters
    ----------
    params : PinnParams
        Trainable neural network parameters and physical constants v_flow, D.
    config : PinnDomainConfig
        Static domain metadata config.
    data_points : Array
        Consensus trajectory spacetime points (N, 3).
    obs_vals : Array
        Observed scalar values (N,).
    colloc_points : Array
        Spatial collocation points (K, 3) for PDE physics residual calculation.
    w_pde : float, default=0.05
        Physics loss weight.

    Returns
    -------
    total_loss : Array
        Weighted loss sum.
    """
    # 1. Data MSE Loss
    preds = PinnFieldMap.forward(params, data_points, config)
    data_loss = jnp.mean((preds - obs_vals) ** 2)

    # 2. Vectorized PDE Physics Residual Loss (uses params.v_flow and params.D internally)
    v_res = jax.vmap(
        lambda p: PinnFieldMap.pde_residual(params, config, p[0], p[1], p[2])
    )
    pde_residuals = v_res(colloc_points)

    return data_loss + w_pde * jnp.mean(pde_residuals ** 2)


sample_trajectory_collocation_points = PinnFieldMap.sample_collocation_points
