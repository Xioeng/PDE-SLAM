"""
joint.py
========
Joint parameter identification for PDE advection-diffusion and kinematic trajectory corrections.

This module provides tools to estimate both PDE physical parameters (diffusivity, constant advection
flow velocity vector) and time-varying trajectory corrections (to correct estimated positions
from the dead-reckoned kinematic model) using scalar field observations and control inputs.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax  # type: ignore[import-untyped]
import scipy.optimize  # type: ignore[import-untyped]
from jax import Array

from pde_slam.interpolators import SpatialGrid, SpatiotemporalInterpolator
from pde_slam.kinematics import UnicycleKinematics
from pde_slam.optimization.kinematics import _pack_params, _unpack_params
from pde_slam.solver import AdvectionDiffusionSolver, PDEParams
from pde_slam.types import ObservationData, TrajectoryContext

# Re-export so existing code that imports from this module continues to work.
__all__ = [
    "JointSlamOptimizer",
    "MultiPdeSlamOptimizer",
    "unicycle_corrected_trajectory_fn",
]


def unicycle_corrected_trajectory_fn(
    x0: Array,
    thrusts: Array,
    headings: Array,
    dt: float | Array,
    k_thrust: float | Array,
    dx: Array,
) -> Array:
    """Predicts a 2-D trajectory with direct position corrections.

    Parameters
    ----------
    x0 : Array
        Initial state [east_m, north_m], shape (2,) or (3,). Only the first
        two elements are used as the starting position.
    thrusts : Array
        1-D array of thrust commands, length N.
    headings : Array
        1-D array of compass headings [rad, navigation convention], length N.
    dt : float or Array
        Time step [s] between commands. If an Array, must have shape (N,).
    k_thrust : float or Array
        Thrust-to-speed conversion factor.
    dx : Array
        Direct coordinate corrections of shape (N + 1, 2).

    Returns
    -------
    coords : Array
        Corrected trajectory coordinates [east_m, north_m], shape (N + 1, 2),
        including the initial corrected position.
    """
    coords_kin = UnicycleKinematics.integrate_trajectory(
        x0[:2], thrusts, headings, dt, k_thrust, include_initial=True
    )
    return coords_kin + dx


class JointSlamOptimizer:
    """Joint optimizer for identifying both PDE physical parameters and trajectory corrections.

    This class computes trajectory rollouts with parameterized corrections, solves the
    advection-diffusion PDE to predict the scalar field over time, and fits:
      - PDE diffusivity (scalar D)
      - PDE advection flow (2D vector v_flow)
      - Trajectory corrections (matrix dx of shape (N+1, 2))
      - Optionally, kinematic parameters (scalar k_thrust)

    It minimizes the Mean Squared Error (MSE) of scalar predictions against observed scalar
    features, plus L2 regularization on the trajectory corrections to keep them small.

    Parameters
    ----------
    grid : SpatialGrid
        The rectangular ENU computational spatial grid.
    pde_solver : AdvectionDiffusionSolver
        The differentiable PDE solver instance.
    corrected_trajectory_fn : Callable, optional
        A differentiable function that predicts the corrected trajectory.
        Must have the signature:
            `corrected_trajectory_fn(x0, thrusts, headings, dt, k_thrust, dx) -> coords`
        where:
            - x0: initial position, shape (2,)
            - thrusts: thrust commands, shape (N,)
            - headings: compass headings [rad], shape (N,)
            - dt: time step(s), float or shape (N,)
            - k_thrust: kinematic thrust parameter
            - dx: position corrections, shape (N + 1, 2)
            - returns: coordinates, shape (N + 1, 2)
        If None, defaults to `unicycle_corrected_trajectory_fn`.
    """

    def __init__(
        self,
        grid: SpatialGrid,
        pde_solver: AdvectionDiffusionSolver,
        corrected_trajectory_fn: Callable[
            [Array, Array, Array, float | Array, float | Array, Array], Array
        ]
        | None = None,
    ) -> None:
        self.grid = grid
        self.pde_solver = pde_solver
        self.corrected_trajectory_fn = (
            unicycle_corrected_trajectory_fn
            if corrected_trajectory_fn is None
            else corrected_trajectory_fn
        )

    # ------------------------------------------------------------------
    # Loss components (override in subclasses to customise behaviour)
    # ------------------------------------------------------------------

    def _data_loss(
        self,
        vals_pred: Array,
        obs_vals: Array,
        phi0: Array,
    ) -> Array:
        """Normalised data MSE: residuals scaled by each field's dynamic range.

        Dividing by ``max(phi0) - min(phi0)`` per field makes the loss
        dimensionless and field-scale-agnostic: a 1 °C error in a 4 °C
        temperature range contributes identically to a 1 PSU error in a
        4 PSU salinity range.

        Boundary condition assumption: zero-Neumann (default solver BC).

        Parameters
        ----------
        vals_pred : Array
            Predicted scalar values, shape ``(M,)`` or ``(M, K)``.
        obs_vals : Array
            Observed scalar values, same shape as ``vals_pred``.
        phi0 : Array
            Initial condition(s), shape ``(ny, nx)`` or ``(K, ny, nx)``.
            Only the spatial axes are used for scale computation.

        Returns
        -------
        loss : Array
            Scalar normalised MSE.
        """
        if phi0.ndim == 2:  # single field: shape (ny, nx)
            phi0_max = jnp.max(phi0)
            phi0_min = jnp.min(phi0)
            scale = jnp.maximum(phi0_max - phi0_min, 1e-6)
        else:  # batched: shape (K, ny, nx) — scale per field, kept for broadcasting
            phi0_max = jnp.max(phi0, axis=(-2, -1))  # (K,)
            phi0_min = jnp.min(phi0, axis=(-2, -1))  # (K,)
            scale = jnp.maximum(phi0_max - phi0_min, 1e-6)  # (K,)
        residuals = (vals_pred - obs_vals) / scale  # dimensionless
        return jnp.mean(residuals**2)

    def _reg_loss(self, dx: Array) -> Array:
        """Mean squared Euclidean norm of trajectory corrections per timestep.

        Penalises the 2-D displacement magnitude at each step jointly rather
        than treating east/north components as independent scalars.
        Smooth everywhere — no gradient singularity at ``dx = 0``.

        Parameters
        ----------
        dx : Array
            Position corrections, shape ``(N+1, 2)`` in [east_m, north_m].

        Returns
        -------
        loss : Array
            Scalar regularisation term.
        """
        return jnp.mean(jnp.sum(dx**2, axis=-1))

    # ------------------------------------------------------------------
    # Joint loss
    # ------------------------------------------------------------------

    def loss_fn(
        self,
        params: dict[str, Array],
        phi0: Array,
        obs: ObservationData,
        traj: TrajectoryContext,
        lambda_reg: float,
        k_thrust_fixed: float,
    ) -> Array:
        """Compute the joint optimization loss for a **single** PDE.

        Parameters
        ----------
        params : dict of str to Array
            Must contain ``'D'`` (scalar), ``'v_flow'`` (shape ``(2,)``), and
            ``'dx'`` (shape ``(N+1, 2)``). May optionally contain ``'k_thrust'``.
        phi0 : Array
            Initial scalar field, shape ``(ny, nx)``.
        obs : ObservationData
            Observation timestamps and scalar values.
        traj : TrajectoryContext
            Robot control inputs and time-grid metadata.
        lambda_reg : float
            Regularisation weight on trajectory-correction magnitude.
        k_thrust_fixed : float
            Fixed thrust-to-speed factor (used when ``'k_thrust'`` is not in
            ``params``).

        Returns
        -------
        loss : Array
            Scalar joint loss = normalised data MSE + λ · reg.
        """
        D = params["D"]  # noqa: N806
        v_flow = params["v_flow"]
        dx = params["dx"]
        k_thrust = params.get("k_thrust", k_thrust_fixed)

        x0 = jnp.zeros(2)

        # 1. Integrate corrected trajectory
        coords_pred = self.corrected_trajectory_fn(
            x0, traj.thrusts, traj.headings, traj.dt_arr, k_thrust, dx
        )

        # 2. Solve PDE with constant flow velocity field
        ny, nx = phi0.shape
        u_field = jnp.broadcast_to(v_flow, (ny, nx, 2))
        pde_params = PDEParams(u_field=u_field, D=D)
        snapshots = self.pde_solver.solve(
            phi0, pde_params, t0=traj.t0, t_end=traj.t_end, saveat=traj.t_traj
        )  # shape (T, ny, nx)

        # 3. Interpolate trajectory to observation times
        x_pred = jnp.interp(obs.ts, traj.t_traj, coords_pred[:, 0])
        y_pred = jnp.interp(obs.ts, traj.t_traj, coords_pred[:, 1])

        # 4. Spatiotemporal interpolation of scalar values at robot positions
        interp = SpatiotemporalInterpolator(self.grid, traj.t_traj, snapshots)
        vals_pred = interp(x_pred, y_pred, obs.ts)  # shape (M,)

        # 5. Compute and combine loss terms
        return self._data_loss(vals_pred, obs.vals, phi0) + lambda_reg * self._reg_loss(dx)

    def fit(
        self,
        phi0: Array | np.ndarray,
        obs_ts: Array | np.ndarray,
        obs_vals: Array | np.ndarray,
        thrusts: Array | np.ndarray,
        headings: Array | np.ndarray,
        dt: float | Array | np.ndarray,
        init_params: dict[str, float | Array | np.ndarray],
        bounds: dict[str, tuple[float | None, float | None]] | None = None,
        lambda_reg: float = 1e-3,
        k_thrust_fixed: float = 1.0,
        method: str = "l-bfgs-b",
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Array], dict[str, Any]]:
        """Fit the PDE parameters and trajectory corrections.

        Parameters
        ----------
        phi0 : Array or np.ndarray
            Initial scalar field on the grid, shape (ny, nx).
        obs_ts : Array or np.ndarray
            Observation timestamps, shape (M,).
        obs_vals : Array or np.ndarray
            Observed scalar values, shape (M,).
        thrusts : Array or np.ndarray
            Thrust commands, shape (N,).
        headings : Array or np.ndarray
            Compass headings, shape (N,).
        dt : float or Array or np.ndarray
            Simulation step time [s] or sequence of step times of shape (N,).
        init_params : dict of str to float or Array
            Initial guess for parameters. Must contain 'D', 'v_flow', and 'dx'.
            Can optionally contain 'k_thrust'.
        bounds : dict of str to tuple of (float or None, float or None), optional
            Bounds for each parameter name. Only supported with 'l-bfgs-b'.
        lambda_reg : float, default 1e-3
            Regularization weight on the magnitude of position corrections dx.
        k_thrust_fixed : float, default 1.0
            Fixed kinematic thrust parameter used if not present in `init_params`.
        method : {'l-bfgs-b', 'adam', 'sgd'}, default 'l-bfgs-b'
            Optimization method to use.
        options : dict, optional
            Optimization options (e.g. learning_rate, num_steps for optax).

        Returns
        -------
        best_params : dict of str to Array
            Optimized parameter JAX arrays (D, v_flow, dx, and optionally k_thrust).
        info : dict of str to Any
            Metadata and final loss.
        """
        # Convert inputs to JAX arrays
        phi0_jax = jnp.asarray(phi0, dtype=jnp.float64)
        obs_ts_jax = jnp.asarray(obs_ts, dtype=jnp.float64)
        obs_vals_jax = jnp.asarray(obs_vals, dtype=jnp.float64)
        thrusts_jax = jnp.asarray(thrusts, dtype=jnp.float64)
        headings_jax = jnp.asarray(headings, dtype=jnp.float64)

        dt_jax = jnp.asarray(dt, dtype=jnp.float64) if isinstance(dt, (np.ndarray, Array)) else dt

        n_steps = len(thrusts_jax)
        if isinstance(dt_jax, (Array, jax.Array)) and dt_jax.ndim > 0:
            dt_arr = dt_jax
        else:
            dt_arr = jnp.full(n_steps, dt_jax)

        t_traj = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(dt_arr)])

        # Compute static t0 and t_end as float values to avoid tracing problems
        if isinstance(dt, (np.ndarray, Array, jax.Array)):
            dt_np = np.asarray(dt)
        elif isinstance(dt, (list, tuple)):
            dt_np = np.array(dt)
        else:
            dt_np = np.array([dt])

        dt_np_full = np.full(n_steps, dt_np[0]) if len(dt_np) == 1 else dt_np

        t_traj_np = np.concatenate([[0.0], np.cumsum(dt_np_full)])
        t0 = float(t_traj_np[0])
        t_end = float(t_traj_np[-1])

        obs = ObservationData(ts=obs_ts_jax, vals=obs_vals_jax)
        traj = TrajectoryContext(
            thrusts=thrusts_jax,
            headings=headings_jax,
            dt_arr=dt_arr,
            t_traj=t_traj,
            t0=t0,
            t_end=t_end,
        )

        if method.lower() in ("adam", "sgd"):
            return self._fit_optax(
                phi0_jax=phi0_jax,
                obs=obs,
                traj=traj,
                init_params=init_params,
                bounds=bounds,
                lambda_reg=lambda_reg,
                k_thrust_fixed=k_thrust_fixed,
                method=method.lower(),
                options=options,
            )

        if method.lower() == "l-bfgs-b":
            return self._fit_scipy(
                phi0_jax=phi0_jax,
                obs=obs,
                traj=traj,
                init_params=init_params,
                bounds=bounds,
                lambda_reg=lambda_reg,
                k_thrust_fixed=k_thrust_fixed,
                options=options,
            )

        if method.lower() in ("bfgs", "cg", "jax-scipy"):
            return self._fit_jax_scipy(
                phi0_jax=phi0_jax,
                obs=obs,
                traj=traj,
                init_params=init_params,
                bounds=bounds,
                lambda_reg=lambda_reg,
                k_thrust_fixed=k_thrust_fixed,
                method=method.lower(),
                options=options,
            )

        raise ValueError(f"Unknown optimization method: {method}")

    def _fit_scipy(
        self,
        phi0_jax: Array,
        obs: ObservationData,
        traj: TrajectoryContext,
        init_params: dict[str, float | Array | np.ndarray],
        bounds: dict[str, tuple[float | None, float | None]] | None,
        lambda_reg: float,
        k_thrust_fixed: float,
        options: dict[str, Any] | None,
    ) -> tuple[dict[str, Array], dict[str, Any]]:
        # Flatten and pack initial guesses
        theta_init, shapes = _pack_params(init_params)  # type: ignore[arg-type]

        # Build bounds list matching the packed array structure
        bounds_list: list[tuple[float | None, float | None]] | None = None
        if bounds is not None:
            bounds_list = []
            for key, shape in shapes:
                size = int(np.prod(shape))
                key_bounds = bounds.get(key, (None, None))
                bounds_list.extend([key_bounds] * size)

        # Compile loss and gradient calculation using JAX JIT
        value_and_grad_fn = jax.jit(
            jax.value_and_grad(
                lambda p: self.loss_fn(
                    p,
                    phi0_jax,
                    obs,
                    traj,
                    lambda_reg,
                    k_thrust_fixed,
                )
            )
        )

        def objective_val_and_grad(theta: np.ndarray) -> tuple[float, np.ndarray]:
            params_dict = _unpack_params(theta, shapes)
            loss_val, grads_dict = value_and_grad_fn(params_dict)
            grad_list = []
            for key, _ in shapes:
                grad_list.append(np.array(grads_dict[key]).ravel())
            grad_flat = np.concatenate(grad_list)
            return float(loss_val), grad_flat.astype(np.float64)

        opt_options = options or {}
        res = scipy.optimize.minimize(
            fun=objective_val_and_grad,
            x0=theta_init,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds_list,
            options=opt_options,
        )

        best_params_dict = _unpack_params(res.x, shapes)
        best_params = {k: jnp.asarray(v) for k, v in best_params_dict.items()}

        info = {
            "success": res.success,
            "status": res.status,
            "message": res.message,
            "fun": float(res.fun),
            "nfev": res.nfev,
            "nit": res.nit,
        }

        return best_params, info

    def _fit_optax(
        self,
        phi0_jax: Array,
        obs: ObservationData,
        traj: TrajectoryContext,
        init_params: dict[str, float | Array | np.ndarray],
        bounds: dict[str, tuple[float | None, float | None]] | None,
        lambda_reg: float,
        k_thrust_fixed: float,
        method: str,
        options: dict[str, Any] | None,
    ) -> tuple[dict[str, Array], dict[str, Any]]:
        opt_options = options or {}
        lr = float(opt_options.get("learning_rate", 1e-2))
        num_steps = int(opt_options.get("num_steps", 200))

        opt = optax.adam(learning_rate=lr) if method == "adam" else optax.sgd(learning_rate=lr)

        params = {k: jnp.asarray(v, dtype=jnp.float64) for k, v in init_params.items()}
        opt_state = opt.init(params)

        @jax.jit
        def step_fn(
            p: dict[str, Array], state: optax.OptState
        ) -> tuple[dict[str, Array], optax.OptState, Array]:
            lossval, grads = jax.value_and_grad(self.loss_fn)(
                p,
                phi0_jax,
                obs,
                traj,
                lambda_reg,
                k_thrust_fixed,
            )
            updates, new_state = opt.update(grads, state, p)
            new_p = optax.apply_updates(p, updates)
            return new_p, new_state, lossval

        loss_history = []
        for _ in range(num_steps):
            params, opt_state, lossval = step_fn(params, opt_state)
            loss_history.append(float(lossval))

        best_params = {k: jnp.asarray(v) for k, v in params.items()}
        info = {
            "success": True,
            "message": "Optax optimization completed",
            "fun": float(lossval),
            "loss_history": loss_history,
        }

        return best_params, info

    def _fit_jax_scipy(
        self,
        phi0_jax: Array,
        obs: ObservationData,
        traj: TrajectoryContext,
        init_params: dict[str, float | Array | np.ndarray],
        bounds: dict[str, tuple[float | None, float | None]] | None,
        lambda_reg: float,
        k_thrust_fixed: float,
        method: str,
        options: dict[str, Any] | None,
    ) -> tuple[dict[str, Array], dict[str, Any]]:
        import jax.scipy.optimize

        # Flatten and pack initial guesses
        theta_init, shapes = _pack_params(init_params)  # type: ignore[arg-type]
        theta_init_jax = jnp.asarray(theta_init)

        # Build pure JAX objective
        @jax.jit
        def jax_objective(theta: Array) -> Array:
            params_dict = _unpack_params(theta, shapes)
            return self.loss_fn(
                params_dict,
                phi0_jax,
                obs,
                traj,
                lambda_reg,
                k_thrust_fixed,
            )

        opt_method = method.lower()
        if opt_method not in ("bfgs", "cg"):
            opt_method = "bfgs"

        opt_options = options or {}

        res = jax.scipy.optimize.minimize(
            fun=jax_objective,
            x0=theta_init_jax,
            method=opt_method,
            options=opt_options,
        )

        best_params_dict = _unpack_params(res.x, shapes)
        best_params = {k: jnp.asarray(v) for k, v in best_params_dict.items()}

        info = {
            "success": bool(res.success),
            "status": int(res.status),
            "fun": float(res.fun),
            "nit": int(res.nit),
        }

        return best_params, info


class MultiPdeSlamOptimizer(JointSlamOptimizer):
    """Joint optimizer for K concurrent advection-diffusion PDEs with a shared velocity field.

    Inherits the full optimizer loop (L-BFGS-B, Adam, JAX-SciPy) from
    :class:`JointSlamOptimizer`.  Only :meth:`loss_fn` is overridden to
    handle a batch of K initial conditions and K individual diffusivities,
    while ``v_flow`` remains a single shared 2-vector.

    Parameters
    ----------
    grid : SpatialGrid
        The rectangular ENU computational spatial grid.
    pde_solver : AdvectionDiffusionSolver
        The differentiable PDE solver instance.
    corrected_trajectory_fn : Callable, optional
        Same signature as in :class:`JointSlamOptimizer`.
        Defaults to :func:`unicycle_corrected_trajectory_fn`.

    Notes
    -----
    Expected parameter shapes inside ``params``:

    * ``'D'`` — shape ``(K,)`` — one diffusivity per field.
    * ``'v_flow'`` — shape ``(2,)`` — single flow shared by all fields.
    * ``'dx'`` — shape ``(N+1, 2)`` — trajectory corrections.

    ``phi0`` passed to :meth:`fit` must have shape ``(K, ny, nx)``.
    ``obs_vals`` must have shape ``(M, K)``.
    """

    def loss_fn(
        self,
        params: dict[str, Array],
        phi0: Array,
        obs: ObservationData,
        traj: TrajectoryContext,
        lambda_reg: float,
        k_thrust_fixed: float,
    ) -> Array:
        """Compute the joint optimization loss for **K concurrent** PDEs.

        Parameters
        ----------
        params : dict of str to Array
            Must contain ``'D'`` (shape ``(K,)``), ``'v_flow'`` (shape
            ``(2,)``), and ``'dx'`` (shape ``(N+1, 2)``).  May optionally
            contain ``'k_thrust'``.
        phi0 : Array
            Initial scalar fields, shape ``(K, ny, nx)``.
        obs : ObservationData
            Timestamps ``(M,)`` and values ``(M, K)`` for all K fields.
        traj : TrajectoryContext
            Robot control inputs and time-grid metadata.
        lambda_reg : float
            Regularisation weight on trajectory-correction magnitude.
        k_thrust_fixed : float
            Fixed thrust-to-speed factor (used when ``'k_thrust'`` not in
            ``params``).

        Returns
        -------
        loss : Array
            Scalar joint loss = normalised data MSE (per field) + λ · reg.
        """
        D = params["D"]  # noqa: N806  shape (K,)
        v_flow = params["v_flow"]  # shape (2,)
        dx = params["dx"]
        k_thrust = params.get("k_thrust", k_thrust_fixed)

        x0 = jnp.zeros(2)

        # 1. Integrate corrected trajectory
        coords_pred = self.corrected_trajectory_fn(
            x0, traj.thrusts, traj.headings, traj.dt_arr, k_thrust, dx
        )

        # 2. Batch-solve K PDEs — shared v_flow, unique D per field
        num_pdes, ny, nx = phi0.shape
        u_fields = jnp.broadcast_to(v_flow, (num_pdes, ny, nx, 2))
        pde_params = PDEParams(u_field=u_fields, D=D)

        solve_vmap = jax.vmap(
            lambda p0, p_params: self.pde_solver.solve(
                p0, p_params, t0=traj.t0, t_end=traj.t_end, saveat=traj.t_traj
            )
        )
        snapshots = solve_vmap(phi0, pde_params)  # (K, T, ny, nx)

        # 3. Interpolate trajectory to observation times
        x_pred = jnp.interp(obs.ts, traj.t_traj, coords_pred[:, 0])
        y_pred = jnp.interp(obs.ts, traj.t_traj, coords_pred[:, 1])

        # 4. Spatiotemporal interpolation — one interpolator per field
        def interp_single(snapshots_k: Array) -> Array:
            interp = SpatiotemporalInterpolator(self.grid, traj.t_traj, snapshots_k)
            return interp(x_pred, y_pred, obs.ts)  # (M,)

        vals_pred = jax.vmap(interp_single)(snapshots).T  # (M, K)

        # 5. Compute and combine loss terms (scale inherited from parent)
        return self._data_loss(vals_pred, obs.vals, phi0) + lambda_reg * self._reg_loss(dx)
