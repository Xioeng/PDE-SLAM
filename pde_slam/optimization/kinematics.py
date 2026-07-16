"""
kinematics.py
=============
JAX-based parameter identification for robot kinematic models.

This module provides tools to estimate kinematic parameters (e.g., thrust-to-speed
factor) from observed coordinates and control inputs (thrust, heading).
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

from pde_slam.kinematics import UnicycleKinematics


def unicycle_trajectory_fn(
    x0: Array,
    thrusts: Array,
    headings: Array,
    dt: float | Array,
    params: dict[str, Array],
) -> Array:
    """Predicts a 2-D trajectory using the unicycle kinematics model.

    Parameters
    ----------
    x0 : Array
        Initial state [east_m, north_m, heading0] or [east_m, north_m], shape (2,) or (3,).
        Only the first two coordinates (x0, y0) are used as the start point.
    thrusts : Array
        1-D array of thrust commands, length N-1.
    headings : Array
        1-D array of compass headings [rad, navigation convention], length N-1.
    dt : float or Array
        Time step [s] between commands. If an Array, must have shape (N-1,).
    params : dict of str to Array
        Parameters dictionary, must contain 'k_thrust'.

    Returns
    -------
    coords : Array
        Predicted trajectory coordinates [east_m, north_m], shape (N, 2),
        including the initial position as the first element.
    """
    return UnicycleKinematics.integrate_trajectory(
        x0[:2], thrusts, headings, dt, params["k_thrust"], include_initial=True
    )


def _pack_params(
    params: dict[str, Array | float],
) -> tuple[np.ndarray, list[tuple[str, tuple[int, ...]]]]:
    """Pack parameter dictionary into a flat 1-D NumPy array.

    Parameters
    ----------
    params : dict of str to Array or float
        Parameters to pack.

    Returns
    -------
    flat_array : np.ndarray
        Flat 1-D array of all parameter elements.
    shapes : list of tuple of (str, tuple of int)
        List of tuples matching parameter names to their original shapes.
    """
    flat_list = []
    shapes = []
    for key, val in sorted(params.items()):
        val_arr = np.asarray(val)
        flat_list.append(val_arr.ravel())
        shapes.append((key, val_arr.shape))
    return np.concatenate(flat_list), shapes


def _unpack_params(
    flat: Array | np.ndarray, shapes: list[tuple[str, tuple[int, ...]]]
) -> dict[str, Array]:
    """Unpack a flat array back into a parameter dictionary.

    Parameters
    ----------
    flat : Array or np.ndarray
        Flat 1-D array of parameter elements.
    shapes : list of tuple of (str, tuple of int)
        Layout of parameter names and shapes.

    Returns
    -------
    params : dict of str to Array
        Unpacked parameter dictionary.
    """
    params = {}
    idx = 0
    for key, shape in shapes:
        size = int(np.prod(shape))
        val_flat = flat[idx : idx + size]
        if len(shape) == 0:
            params[key] = jnp.asarray(val_flat.reshape(()))
        else:
            params[key] = jnp.asarray(val_flat.reshape(shape))
        idx += size
    return params


class KinematicsOptimizer:
    """Optimizer for identifying parameters of kinematic models from trajectory observations.

    This class computes trajectory rollouts using a differentiable model function and
    finds the parameters (such as `k_thrust`) that minimize the Mean Squared Error (MSE)
    against observed coordinates.

    Parameters
    ----------
    trajectory_fn : Callable[[Array, Array, Array, float | Array, dict], Array], optional
        A differentiable function that predicts the trajectory.
        Must have the signature:
            trajectory_fn(x0, thrusts, headings, dt, params) -> coords
        where:
            - x0: Initial position, shape (2,) or (3,)
            - thrusts: Thrust commands, shape (M,)
            - headings: Compass headings [rad], shape (M,)
            - dt: Time step(s), float or shape (M,)
            - params: dict of parameters to optimize
            - returns: coordinates, shape (M+1, 2)
        If None, defaults to the standard unicycle kinematics trajectory prediction.
    """

    def __init__(
        self,
        trajectory_fn: Callable[[Array, Array, Array, float | Array, dict[str, Array]], Array]
        | None = None,
    ) -> None:
        self.trajectory_fn: Callable[
            [Array, Array, Array, float | Array, dict[str, Array]], Array
        ] = unicycle_trajectory_fn if trajectory_fn is None else trajectory_fn

    def loss_fn(
        self,
        params: dict[str, Array],
        x0: Array,
        thrusts: Array,
        headings: Array,
        dt: float | Array,
        coords_obs: Array,
    ) -> Array:
        """Compute the mean squared error loss between predicted and observed coordinates.

        Parameters
        ----------
        params : dict of str to Array
            The dictionary of parameters to evaluate.
        x0 : Array
            Initial position, shape (2,) or (3,).
        thrusts : Array
            Thrust commands, shape (M,) or (M+1,).
        headings : Array
            Compass headings [rad], shape (M,) or (M+1,).
        dt : float or Array
            Time step [s] between commands.
        coords_obs : Array
            Observed coordinates [east_m, north_m], shape (N, 2).

        Returns
        -------
        loss : Array
            Scalar mean squared error loss.
        """
        num_steps = coords_obs.shape[0] - 1
        t_slice = thrusts[:num_steps]
        h_slice = headings[:num_steps]

        dt_slice = dt[:num_steps] if isinstance(dt, (Array, jax.Array)) and dt.ndim > 0 else dt

        coords_pred = self.trajectory_fn(x0, t_slice, h_slice, dt_slice, params)
        return jnp.mean((coords_pred - coords_obs) ** 2)

    def fit(
        self,
        coords_obs: Array | np.ndarray,
        thrusts: Array | np.ndarray,
        headings: Array | np.ndarray,
        dt: float | Array | np.ndarray,
        init_params: dict[str, float | Array],
        bounds: dict[str, tuple[float | None, float | None]] | None = None,
        method: str = "l-bfgs-b",
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """Fit the kinematic parameters to the observed coordinates.

        Parameters
        ----------
        coords_obs : Array or np.ndarray
            Observed coordinates [east_m, north_m], shape (N, 2).
        thrusts : Array or np.ndarray
            Thrust commands, shape (M,) or (M+1,).
        headings : Array or np.ndarray
            Compass headings [rad, navigation convention], shape (M,) or (M+1,).
        dt : float or Array or np.ndarray
            Time step [s] between commands.
        init_params : dict of str to float or Array
            Initial guess for the kinematic parameters to be optimized.
        bounds : dict of str to tuple of (float or None, float or None), optional
            Bounds for each parameter. Only supported when method is 'l-bfgs-b'.
        method : str, default 'l-bfgs-b'
            Optimization method to use:
            - 'l-bfgs-b': SciPy L-BFGS-B optimizer using JAX-compiled value and gradients.
            - 'adam': Optax Adam gradient descent.
            - 'sgd': Optax SGD gradient descent.
        options : dict, optional
            Method-specific optimization options:
            - For 'l-bfgs-b': passed directly to ``scipy.optimize.minimize``.
            - For 'adam' / 'sgd': supports:
                * 'learning_rate': learning rate [default 1e-2]
                * 'num_steps': number of iterations [default 200]

        Returns
        -------
        best_params : dict of str to float
            The optimized parameter dictionary containing scalar values.
        info : dict of str to Any
            Metadata and statistics about the optimization run (e.g. final loss).
        """
        # Convert inputs to JAX arrays
        coords_obs_jax = jnp.asarray(coords_obs, dtype=jnp.float64)
        thrusts_jax = jnp.asarray(thrusts, dtype=jnp.float64)
        headings_jax = jnp.asarray(headings, dtype=jnp.float64)
        dt_jax = jnp.asarray(dt, dtype=jnp.float64) if isinstance(dt, (np.ndarray, Array)) else dt

        # Initial state is the first observed coordinate
        x0_jax = coords_obs_jax[0]

        if method.lower() in ("adam", "sgd"):
            return self._fit_optax(
                x0_jax=x0_jax,
                thrusts_jax=thrusts_jax,
                headings_jax=headings_jax,
                dt_jax=dt_jax,
                coords_obs_jax=coords_obs_jax,
                init_params=init_params,
                method=method.lower(),
                options=options,
            )

        if method.lower() == "l-bfgs-b":
            return self._fit_scipy(
                x0_jax=x0_jax,
                thrusts_jax=thrusts_jax,
                headings_jax=headings_jax,
                dt_jax=dt_jax,
                coords_obs_jax=coords_obs_jax,
                init_params=init_params,
                bounds=bounds,
                options=options,
            )

        raise ValueError(f"Unknown optimization method: {method}")

    def _fit_scipy(
        self,
        x0_jax: Array,
        thrusts_jax: Array,
        headings_jax: Array,
        dt_jax: float | Array,
        coords_obs_jax: Array,
        init_params: dict[str, float | Array],
        bounds: dict[str, tuple[float | None, float | None]] | None,
        options: dict[str, Any] | None,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """Fit parameters using SciPy L-BFGS-B with JAX gradients."""
        theta_init, shapes = _pack_params(init_params)

        # Build bounds list matching the packed array structure
        bounds_list: list[tuple[float | None, float | None]] | None = None
        if bounds is not None:
            bounds_list = []
            for key, shape in shapes:
                size = int(np.prod(shape))
                key_bounds = bounds.get(key, (None, None))
                bounds_list.extend([key_bounds] * size)

        # Compile objective and gradient calculation
        value_and_grad_fn = jax.jit(jax.value_and_grad(self.loss_fn))

        def objective_val_and_grad(theta: np.ndarray) -> tuple[float, np.ndarray]:
            params_dict = _unpack_params(theta, shapes)
            loss_val, grads_dict = value_and_grad_fn(
                params_dict, x0_jax, thrusts_jax, headings_jax, dt_jax, coords_obs_jax
            )
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
        best_params = {k: float(v) for k, v in best_params_dict.items()}

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
        x0_jax: Array,
        thrusts_jax: Array,
        headings_jax: Array,
        dt_jax: float | Array,
        coords_obs_jax: Array,
        init_params: dict[str, float | Array],
        method: str,
        options: dict[str, Any] | None,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """Fit parameters using Optax gradient descent."""
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
            loss_val, grads = jax.value_and_grad(self.loss_fn)(
                p, x0_jax, thrusts_jax, headings_jax, dt_jax, coords_obs_jax
            )
            updates, new_state = opt.update(grads, state, p)
            new_p = optax.apply_updates(p, updates)
            return new_p, new_state, loss_val

        loss_history = []
        for _ in range(num_steps):
            params, opt_state, loss_val = step_fn(params, opt_state)
            loss_history.append(float(loss_val))

        best_params = {k: float(v) for k, v in params.items()}
        info = {
            "success": True,
            "message": "Optax optimization completed",
            "fun": float(loss_val),
            "loss_history": loss_history,
        }

        return best_params, info
