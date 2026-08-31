"""
pinn.py
=======
Physics-Informed Neural Network (PINN) map representation for 2D
advection-diffusion scalar fields (supports 1 or n simultaneous scalar fields).

Supports both standard sequential MLP architecture ('mlp') and Modified MLP
architecture ('modified_mlp' Wang & Perdikaris, 2021) with input coordinate
encoders and multiplicative gating across hidden layers.

Inputs [t, x, y] are normalized prior to passing into the network:
- Spatial (x, y) coordinates normalized to [-1, 1] based on x_bounds and y_bounds.
- Temporal t coordinate normalized to [0, 1] based on t_max.

PinnConfig is passed when initializing PinnFieldMap instance
(e.g. ``pinn_map = PinnFieldMap(config)``), binding domain bounds metadata,
architecture settings, initial parameters, optimizer, and training
hyperparameters to the map instance.

Physical parameters (flow velocity ``v_flow`` and diffusivity ``log_D``) are
included in the ``PinnParams`` pytree so they are jointly learned alongside
neural network weights via JAX automatic differentiation. Diffusivity is
stored in log-space (``log_D = log(D)``) so the optimizer sees an unconstrained
real; the PDE residual uses ``jnp.exp(log_D)`` to recover ``D``.
"""

from __future__ import annotations

from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax
from jax import Array

# ===========================================================================
# 1. Data Structures & Configuration
# ===========================================================================


class PinnParams(NamedTuple):
    """Pytree holding ALL trainable parameters.

    Parameters
    ----------
    weights : list of Array
        Weight matrices for MLP layers.
    biases : list of Array
        Bias vectors for MLP layers.
    v_flow : Array
        Shared advection flow velocity vector [u, v], shape (2,).
    log_D : Array
        Log-diffusivity log(D) [log(m²/s)]. Scalar shape () if n_fields=1,
        or shape (n_fields,) if n_fields > 1. Use jnp.exp(log_D) to recover
        the physical diffusivity coefficient(s).
    W_u : Array or None, optional
        First coordinate encoder weight matrix for Modified MLP architecture,
        shape (in_dim, hidden_dim). None for standard MLP.
    b_u : Array or None, optional
        First coordinate encoder bias vector for Modified MLP architecture,
        shape (hidden_dim,). None for standard MLP.
    W_v : Array or None, optional
        Second coordinate encoder weight matrix for Modified MLP architecture,
        shape (in_dim, hidden_dim). None for standard MLP.
    b_v : Array or None, optional
        Second coordinate encoder bias vector for Modified MLP architecture,
        shape (hidden_dim,). None for standard MLP.
    """

    weights: list[Array]
    biases: list[Array]
    v_flow: Array
    log_D: Array  # noqa: N815
    W_u: Array | None = None  # noqa: N815
    b_u: Array | None = None
    W_v: Array | None = None  # noqa: N815
    b_v: Array | None = None


class PinnConfig(NamedTuple):
    """Unified configuration for domain bounds, network architecture, and optimization.

    Parameters
    ----------
    x_bounds : tuple of float
        Spatial x domain bounds [x_min, x_max].
    y_bounds : tuple of float
        Spatial y domain bounds [y_min, y_max].
    t_max : float
        Maximum time horizon [s].
    in_dim : int
        Input feature dimension (default 3: [t, x, y]).
    n_fields : int
        Number of scalar fields estimated simultaneously (default 1).
    arch : str
        Neural network architecture: 'mlp' (standard sequential MLP) or
        'modified_mlp' (Wang & Perdikaris, 2021 gated coordinate encoder MLP).
    hidden_dim : int
        Width of each hidden layer in the MLP.
    num_layers : int
        Total number of layers in the MLP (including output layer).
    v_flow_init : tuple of float or None
        Initial flow velocity vector [u, v] (2,); defaults to zero.
    log_D_init : float or tuple of float
        Initial log-diffusivity log(D) per field; defaults to log(0.5) ≈ -0.693.
    learning_rate : float
        Learning rate for Adam optimizer.
    num_steps : int
        Number of gradient steps per fit iteration.
    num_colloc : int
        Number of PDE collocation points per fit step.
    margin : float
        Spatial margin [m] around trajectory bounding box for collocation sampling.
    w_pde : float
        Physics loss weight multiplier.
    batch_size : int
        Fixed batch size for observation sampling per fit step (prevents JAX
        recompilation).
    """

    x_bounds: tuple[float, float] = (-150.0, 150.0)
    y_bounds: tuple[float, float] = (-150.0, 150.0)
    t_max: float = 100.0
    in_dim: int = 3
    n_fields: int = 1
    arch: str = "mlp"
    hidden_dim: int = 64
    num_layers: int = 4
    v_flow_init: tuple[float, float] = (0.0, 0.0)
    log_D_init: float | tuple[float, ...] = -0.6931471805599453  # noqa: N803, N815
    learning_rate: float = 3e-3
    num_steps: int = 5
    num_colloc: int = 40
    margin: float = 15.0
    w_pde: float = 0.05
    batch_size: int = 64


def _sanitize_config(config: PinnConfig) -> PinnConfig:
    """Ensure all fields in PinnConfig are hashable Python primitives."""
    clean_v_flow: tuple[float, float] = (
        (float(config.v_flow_init[0]), float(config.v_flow_init[1]))
        if config.v_flow_init is not None
        else None
    )
    if isinstance(config.log_D_init, (tuple, list)):
        clean_log_d: float | tuple[float, ...] = tuple(
            float(x) for x in config.log_D_init
        )
    else:
        clean_log_d = float(config.log_D_init)

    clean_arch = str(config.arch).strip().lower()
    if clean_arch not in ("mlp", "standard", "modified_mlp", "wang", "perdikaris"):
        msg = (
            f"Unknown PINN architecture '{config.arch}'. "
            "Supported: 'mlp', 'modified_mlp'."
        )
        raise ValueError(msg)

    return config._replace(
        n_fields=int(config.n_fields),
        arch=clean_arch,
        v_flow_init=clean_v_flow,
        log_D_init=clean_log_d,
        t_max=float(config.t_max),
        learning_rate=float(config.learning_rate),
        num_steps=int(config.num_steps),
        num_colloc=int(config.num_colloc),
        margin=float(config.margin),
        w_pde=float(config.w_pde),
        batch_size=int(config.batch_size),
    )


# ===========================================================================
# 2. Pure JAX Functions (Stateless, JIT-Friendly, Zero Class Overhead)
# ===========================================================================


def normalize_inputs(p: Array, config: PinnConfig) -> Array:
    """Normalize spacetime input coordinates: (x, y) -> [-1, 1] and t -> [0, 1].

    Parameters
    ----------
    p : Array
        Spacetime coordinates [t, x, y], shape (..., 3).
    config : PinnConfig
        PINN configuration object.

    Returns
    -------
    Array
        Normalized coordinates, shape (..., 3).
    """
    t = p[..., 0]
    x = p[..., 1]
    y = p[..., 2]

    t_norm = jnp.clip(t / config.t_max, 0.0, 1.0)
    x_norm = (
        2.0 * (x - config.x_bounds[0]) / (config.x_bounds[1] - config.x_bounds[0]) - 1.0
    )
    y_norm = (
        2.0 * (y - config.y_bounds[0]) / (config.y_bounds[1] - config.y_bounds[0]) - 1.0
    )

    return jnp.stack([t_norm, x_norm, y_norm], axis=-1)


def pinn_forward_mlp(params: PinnParams, p: Array, config: PinnConfig) -> Array:
    """Evaluate standard sequential MLP forward pass.

    Parameters
    ----------
    params : PinnParams
        PINN parameter pytree.
    p : Array
        Spacetime coordinates [t, x, y], shape (..., 3).
    config : PinnConfig
        PINN configuration.

    Returns
    -------
    Array
        Predicted scalar value shape (...) if n_fields=1, or shape (..., n_fields)
        if n_fields > 1.
    """
    h = normalize_inputs(p, config)
    for w, b in zip(params.weights[:-1], params.biases[:-1], strict=True):
        h = jnp.tanh(jnp.dot(h, w) + b)
    out = jnp.dot(h, params.weights[-1]) + params.biases[-1]
    if config.n_fields == 1:
        return jnp.squeeze(out, axis=-1)
    return out


def pinn_forward_modified_mlp(
    params: PinnParams, p: Array, config: PinnConfig
) -> Array:
    """Evaluate Modified MLP (Wang & Perdikaris, 2021) forward pass.

    Uses coordinate gating to connect input spacetime coordinates directly
    to each intermediate hidden layer.

    Parameters
    ----------
    params : PinnParams
        PINN parameter pytree containing coordinate encoders (W_u, b_u, W_v, b_v).
    p : Array
        Spacetime coordinates [t, x, y], shape (..., 3).
    config : PinnConfig
        PINN configuration.

    Returns
    -------
    Array
        Predicted scalar value shape (...) if n_fields=1, or shape (..., n_fields)
        if n_fields > 1.
    """
    if (
        params.W_u is None
        or params.b_u is None
        or params.W_v is None
        or params.b_v is None
    ):
        msg = (
            "Modified MLP requires coordinate encoder weights (W_u, b_u, W_v, b_v) "
            "in PinnParams."
        )
        raise ValueError(msg)

    x = normalize_inputs(p, config)
    u = jnp.tanh(jnp.dot(x, params.W_u) + params.b_u)
    v = jnp.tanh(jnp.dot(x, params.W_v) + params.b_v)

    # First hidden layer
    h = jnp.tanh(jnp.dot(x, params.weights[0]) + params.biases[0])

    # Gated intermediate hidden layers
    for w, b in zip(params.weights[1:-1], params.biases[1:-1], strict=True):
        z = jnp.tanh(jnp.dot(h, w) + b)
        h = (1.0 - z) * u + z * v

    # Final linear output layer
    out = jnp.dot(h, params.weights[-1]) + params.biases[-1]
    if config.n_fields == 1:
        return jnp.squeeze(out, axis=-1)
    return out


def pinn_forward(params: PinnParams, p: Array, config: PinnConfig) -> Array:
    """Evaluate PINN forward pass dispatching to configured architecture.

    Parameters
    ----------
    params : PinnParams
        PINN parameter pytree.
    p : Array
        Spacetime coordinates [t, x, y], shape (..., 3).
    config : PinnConfig
        PINN configuration.

    Returns
    -------
    Array
        Predicted scalar value shape (...) if n_fields=1, or shape (..., n_fields)
        if n_fields > 1.
    """
    if config.arch in ("modified_mlp", "wang", "perdikaris"):
        return pinn_forward_modified_mlp(params, p, config)
    return pinn_forward_mlp(params, p, config)


def pinn_pde_residual(
    params: PinnParams,
    config: PinnConfig,
    t: float | Array = 0.0,
    x: float | Array = 0.0,
    y: float | Array = 0.0,
) -> Array:
    """Compute physical advection-diffusion PDE residual using autodiff.

    Parameters
    ----------
    params : PinnParams
        PINN parameters (weights, v_flow, log_D).
    config : PinnConfig
        PINN configuration object.
    t : float or Array
        Time coordinate.
    x : float or Array
        X spatial coordinate.
    y : float or Array
        Y spatial coordinate.

    Returns
    -------
    Array
        PDE residual scalar shape () if n_fields=1, or vector shape (n_fields,)
        if n_fields > 1.
    """
    if config.n_fields == 1:

        def scalar_fn(_t: Array, _x: Array, _y: Array) -> Array:
            p_val = jnp.stack([_t, _x, _y], axis=-1)
            return pinn_forward(params, p_val, config)

        dt = jax.grad(scalar_fn, argnums=0)(t, x, y)
        dx = jax.grad(scalar_fn, argnums=1)(t, x, y)
        dy = jax.grad(scalar_fn, argnums=2)(t, x, y)

        dx2 = jax.grad(
            lambda _t, _x, _y: jax.grad(scalar_fn, argnums=1)(_t, _x, _y), argnums=1
        )(t, x, y)
        dy2 = jax.grad(
            lambda _t, _x, _y: jax.grad(scalar_fn, argnums=2)(_t, _x, _y), argnums=2
        )(t, x, y)

        return (
            dt
            + params.v_flow[0] * dx
            + params.v_flow[1] * dy
            - jnp.exp(params.log_D) * (dx2 + dy2)
        )
    else:

        def vector_fn(_t: Array, _x: Array, _y: Array) -> Array:
            p_val = jnp.stack([_t, _x, _y], axis=-1)
            return pinn_forward(params, p_val, config)

        dt = jax.jacobian(vector_fn, argnums=0)(t, x, y)
        dx = jax.jacobian(vector_fn, argnums=1)(t, x, y)
        dy = jax.jacobian(vector_fn, argnums=2)(t, x, y)

        dx2 = jax.jacobian(
            lambda _t, _x, _y: jax.jacobian(vector_fn, argnums=1)(_t, _x, _y), argnums=1
        )(t, x, y)
        dy2 = jax.jacobian(
            lambda _t, _x, _y: jax.jacobian(vector_fn, argnums=2)(_t, _x, _y), argnums=2
        )(t, x, y)

        return (
            dt
            + params.v_flow[0] * dx
            + params.v_flow[1] * dy
            - jnp.exp(params.log_D) * (dx2 + dy2)
        )


def sample_collocation_points(
    trajectory_points: Array,
    t_curr: float | Array,
    key: Array,
    config: PinnConfig,
    num_colloc: int | None = None,
    margin: float | None = None,
) -> Array:
    """Generate random PDE collocation points bounded to trajectory bounding box.

    Parameters
    ----------
    trajectory_points : Array
        Sampled trajectory points, shape (N, 3) or (N, 2).
    t_curr : float or Array
        Current timestamp.
    key : Array
        JAX PRNG key.
    config : PinnConfig
        PINN configuration.
    num_colloc : int or None
        Optional override for collocation point count.
    margin : float or None
        Optional override for spatial margin.

    Returns
    -------
    Array
        Collocation points (t, x, y), shape (num_colloc, 3).
    """
    n_colloc = config.num_colloc if num_colloc is None else num_colloc
    m_margin = config.margin if margin is None else margin

    if trajectory_points.shape[1] == 3:
        x_coords = trajectory_points[:, 1]
        y_coords = trajectory_points[:, 2]
    else:
        x_coords = trajectory_points[:, 0]
        y_coords = trajectory_points[:, 1]

    x_min_box = jnp.clip(
        jnp.min(x_coords) - m_margin, config.x_bounds[0], config.x_bounds[1]
    )
    x_max_box = jnp.clip(
        jnp.max(x_coords) + m_margin, config.x_bounds[0], config.x_bounds[1]
    )
    y_min_box = jnp.clip(
        jnp.min(y_coords) - m_margin, config.y_bounds[0], config.y_bounds[1]
    )
    y_max_box = jnp.clip(
        jnp.max(y_coords) + m_margin, config.y_bounds[0], config.y_bounds[1]
    )

    k1, k2, k3 = jax.random.split(key, 3)

    colloc_t = jax.random.uniform(
        k1, (n_colloc,), minval=0.0, maxval=jnp.maximum(jnp.asarray(t_curr), 1e-3)
    )
    colloc_x = jax.random.uniform(k2, (n_colloc,), minval=x_min_box, maxval=x_max_box)
    colloc_y = jax.random.uniform(k3, (n_colloc,), minval=y_min_box, maxval=y_max_box)

    return jnp.stack([colloc_t, colloc_x, colloc_y], axis=-1)


@partial(jax.jit, static_argnames=("config", "w_pde"))
def pinn_loss_fn(
    params: PinnParams,
    config: PinnConfig,
    data_points: Array,
    obs_vals: Array,
    colloc_points: Array,
    w_pde: float = 0.05,
) -> Array:
    """Computes joint Data MSE Loss + PDE Physics Residual Loss.

    Parameters
    ----------
    params : PinnParams
        PINN parameters pytree.
    config : PinnConfig
        PINN configuration.
    data_points : Array
        Observation points (t, x, y), shape (N, 3).
    obs_vals : Array
        Observed values, shape (N,) if n_fields=1 or (N, n_fields) if n_fields > 1.
    colloc_points : Array
        Collocation points (t, x, y), shape (N_colloc, 3).
    w_pde : float, default=0.05
        PDE loss weight multiplier.

    Returns
    -------
    Array
        Total scalar loss.
    """
    preds = pinn_forward(params, data_points, config)
    pde_residuals = jax.vmap(
        lambda p: pinn_pde_residual(params, config, t=p[0], x=p[1], y=p[2])
    )(colloc_points)

    return jnp.mean((preds - obs_vals) ** 2) + w_pde * jnp.mean(pde_residuals**2)


@partial(jax.jit, static_argnames=("optimizer", "config", "w_pde"))
def pinn_step_fn(
    params: PinnParams,
    opt_state: Any,
    optimizer: optax.GradientTransformation,
    config: PinnConfig,
    buf_pts: Array,
    buf_vals: Array,
    colloc_pts: Array,
    w_pde: float = 0.05,
) -> tuple[PinnParams, Any, Array]:
    """Computes a single optimization step for PINN parameters."""
    loss_val, grads = jax.value_and_grad(pinn_loss_fn)(
        params, config, buf_pts, buf_vals, colloc_pts, w_pde=w_pde
    )
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss_val


# ===========================================================================
# 3. Stateful OO Class Interface
# ===========================================================================


class PinnFieldMap:
    """Physics-Informed Neural Network (PINN) scalar field map."""

    def __init__(self, config: PinnConfig, key: Array | None = None) -> None:
        """Initialize PINN map instance with required config and optional key.

        Parameters
        ----------
        config : PinnConfig
            PINN configuration.
        key : Array or None
            Optional JAX PRNG key to auto-initialize parameters and optimizer state.
        """
        self.config = _sanitize_config(config)
        self.optimizer: optax.GradientTransformation = optax.adam(
            learning_rate=self.config.learning_rate
        )
        self.params: PinnParams | None = None
        self.opt_state: Any | None = None

        if key is not None:
            self.init_params(key)

    @property
    def v_flow(self) -> Array | None:
        """Estimated flow velocity vector [u, v] (2,) if parameters initialized."""
        return self.params.v_flow if self.params is not None else None

    @property
    def log_D(self) -> Array | None:  # noqa: N802
        """Estimated log-diffusivity () or (n_fields,) if parameters initialized."""
        return self.params.log_D if self.params is not None else None

    @property
    def D(self) -> Array | None:  # noqa: N802
        """Estimated physical diffusivity coefficient exp(log_D) if initialized."""
        return jnp.exp(self.params.log_D) if self.params is not None else None

    def init_params(self, key: Array) -> PinnParams:
        """Initialize trainable parameters pytree and optimizer state using config."""
        cfg = self.config
        weights: list[Array] = []
        biases: list[Array] = []
        w_u: Array | None = None
        b_u: Array | None = None
        w_v: Array | None = None
        b_v: Array | None = None

        if cfg.arch in ("modified_mlp", "wang", "perdikaris"):
            k_u, k_v, k_layers = jax.random.split(key, 3)
            keys = jax.random.split(k_layers, cfg.num_layers)

            limit_in = jnp.sqrt(6.0 / (cfg.in_dim + cfg.hidden_dim))
            w_u = jax.random.uniform(
                k_u, (cfg.in_dim, cfg.hidden_dim), minval=-limit_in, maxval=limit_in
            )
            b_u = jnp.zeros((cfg.hidden_dim,))
            w_v = jax.random.uniform(
                k_v, (cfg.in_dim, cfg.hidden_dim), minval=-limit_in, maxval=limit_in
            )
            b_v = jnp.zeros((cfg.hidden_dim,))

            # Layer 0: in_dim -> hidden_dim
            weights.append(
                jax.random.uniform(
                    keys[0],
                    (cfg.in_dim, cfg.hidden_dim),
                    minval=-limit_in,
                    maxval=limit_in,
                )
            )
            biases.append(jnp.zeros((cfg.hidden_dim,)))

            # Intermediate layers 1 to num_layers - 2: hidden_dim -> hidden_dim
            limit_h = jnp.sqrt(6.0 / (2 * cfg.hidden_dim))
            for i in range(1, cfg.num_layers - 1):
                weights.append(
                    jax.random.uniform(
                        keys[i],
                        (cfg.hidden_dim, cfg.hidden_dim),
                        minval=-limit_h,
                        maxval=limit_h,
                    )
                )
                biases.append(jnp.zeros((cfg.hidden_dim,)))

            # Output layer: hidden_dim -> n_fields
            limit_out = jnp.sqrt(6.0 / (cfg.hidden_dim + cfg.n_fields))
            weights.append(
                jax.random.uniform(
                    keys[-1],
                    (cfg.hidden_dim, cfg.n_fields),
                    minval=-limit_out,
                    maxval=limit_out,
                )
            )
            biases.append(jnp.zeros((cfg.n_fields,)))
        else:
            keys = jax.random.split(key, cfg.num_layers + 1)
            layer_dims = (
                [cfg.in_dim] + [cfg.hidden_dim] * (cfg.num_layers - 1) + [cfg.n_fields]
            )

            for i in range(len(layer_dims) - 1):
                fan_in, fan_out = layer_dims[i], layer_dims[i + 1]
                limit = jnp.sqrt(6.0 / (fan_in + fan_out))
                w = jax.random.uniform(
                    keys[i], (fan_in, fan_out), minval=-limit, maxval=limit
                )
                b = jnp.zeros((fan_out,))
                weights.append(w)
                biases.append(b)

        v_flow_arr = (
            jnp.zeros((2,), dtype=jnp.float64)
            if cfg.v_flow_init is None
            else jnp.asarray(cfg.v_flow_init, dtype=jnp.float64)
        )

        if cfg.log_D_init is None:
            log_d_arr = jnp.zeros((cfg.n_fields,), dtype=jnp.float64)
        elif isinstance(cfg.log_D_init, (tuple, list)):
            log_d_arr = jnp.asarray(cfg.log_D_init, dtype=jnp.float64)
        else:
            log_d_arr = jnp.full(
                (cfg.n_fields,), float(cfg.log_D_init), dtype=jnp.float64
            )

        params = PinnParams(
            weights=weights,
            biases=biases,
            v_flow=v_flow_arr,
            log_D=log_d_arr,
            W_u=w_u,
            b_u=b_u,
            W_v=w_v,
            b_v=b_v,
        )
        self.params = params
        self.opt_state = self.optimizer.init(params)
        return params

    def normalize_inputs(self, p: Array) -> Array:
        """Normalize spacetime input coordinates."""
        return normalize_inputs(p, self.config)

    def forward(self, p: Array) -> Array:
        """Evaluate neural field map at unnormalized input coordinates."""
        if self.params is None:
            msg = (
                "PinnParams not set. Initialize PinnFieldMap with key or "
                "call init_params(key)."
            )
            raise ValueError(msg)
        return pinn_forward(self.params, p, self.config)

    def predict(self, t: float | Array, poses: Array) -> Array:
        """Predict scalar field values at timestamp t and 2D spatial poses.

        Parameters
        ----------
        t : float or Array
            Timestamp in seconds.
        poses : Array
            2D spatial positions [east, north], shape (N, 2) or (2,).

        Returns
        -------
        Array
            Predicted scalar values, shape (N,) or (N, n_fields).
        """
        poses_arr = jnp.atleast_2d(poses)
        t_val = float(t) if isinstance(t, (int, float)) else t
        t_col = jnp.full((poses_arr.shape[0], 1), t_val)
        query_pts = jnp.column_stack([t_col, poses_arr])
        preds = self.forward(query_pts)
        return preds[0] if jnp.ndim(poses) == 1 else preds

    def __call__(self, p: Array) -> Array:
        """Shortcut for self.forward(p)."""
        return self.forward(p)

    def pde_residual(
        self,
        t: float | Array = 0.0,
        x: float | Array = 0.0,
        y: float | Array = 0.0,
    ) -> Array:
        """Compute physical advection-diffusion PDE residual using autodiff."""
        if self.params is None:
            msg = (
                "PinnParams not set. Initialize PinnFieldMap with key or "
                "call init_params(key)."
            )
            raise ValueError(msg)
        return pinn_pde_residual(self.params, self.config, t, x, y)

    def sample_collocation_points(
        self,
        trajectory_points: Array,
        t_curr: float | Array,
        key: Array,
        num_colloc: int | None = None,
        margin: float | None = None,
    ) -> Array:
        """Generate random PDE collocation points bounded to trajectory bounding box."""
        return sample_collocation_points(
            trajectory_points,
            t_curr,
            key,
            self.config,
            num_colloc=num_colloc,
            margin=margin,
        )

    def fit(
        self,
        buf_pts: Array,
        buf_vals: Array,
        key: Array,
        shuffle: bool = False,
        batch_size: int | None = None,
    ) -> tuple[PinnParams, Any, float]:
        """Fit PINN neural map and joint PDE parameters online using instance state.

        Parameters
        ----------
        buf_pts : Array
            Buffered observation points (t, x, y), shape (N, 3).
        buf_vals : Array
            Buffered observed scalar values, shape (N,) if n_fields=1 or (N, n_fields)
            if n_fields > 1.
        key : Array
            JAX PRNG key.
        shuffle : bool, default=False
            If True, randomly shuffles buf_pts and buf_vals prior to training.
        batch_size : int or None, default=None
            Batch size for observation sampling. If None, defaults to config.batch_size.
            Passing a static batch size prevents JAX JIT recompilation.

        Returns
        -------
        params : PinnParams
            Updated pytree of PINN weights and physical parameters.
        opt_state : Any
            Updated Optax optimizer state.
        loss_val : float
            Final scalar loss value after training steps.
        """
        if self.params is None or self.opt_state is None:
            msg = (
                "PinnFieldMap params or opt_state not initialized. "
                "Initialize PinnFieldMap with key or call init_params(key)."
            )
            raise ValueError(msg)

        t_curr = float(jnp.max(buf_pts[:, 0]))
        k_colloc, k_batch = jax.random.split(key)

        bs = int(self.config.batch_size if batch_size is None else batch_size)
        n_obs = buf_pts.shape[0]

        # Sample a fixed static batch_size of observations to prevent JAX recompilation
        idx = jax.random.choice(k_batch, n_obs, shape=(bs,), replace=(n_obs < bs))
        batch_pts = buf_pts[idx]
        batch_vals = buf_vals[idx]

        colloc_pts = self.sample_collocation_points(buf_pts, t_curr, key=k_colloc)

        curr_params = self.params
        curr_opt_state = self.opt_state
        last_loss = 0.0

        for _ in range(self.config.num_steps):
            curr_params, curr_opt_state, loss_val = pinn_step_fn(
                curr_params,
                curr_opt_state,
                self.optimizer,
                self.config,
                batch_pts,
                batch_vals,
                colloc_pts,
                w_pde=self.config.w_pde,
            )
            last_loss = float(loss_val)

        self.params = curr_params
        self.opt_state = curr_opt_state
        return curr_params, curr_opt_state, last_loss
