"""
pde_slam/optimization/rbpf.py
=============================
Rao-Blackwellized Particle Filter (RBPF) SLAM using JAX.
Implements a mutable-state OO design.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array


class RbpfState(NamedTuple):
    """Read-only snapshot of the RBPF SLAM state."""

    poses: Array
    headings: Array
    log_weights: Array
    trajectories: Array
    xl: Array
    P: Array
    speeds: Array


class RbpfSlam:
    """Mutable-state JAX Rao-Blackwellized Particle Filter for SLAM.

    Parameters
    ----------
    n_particles : int, default=100
        Number of particles.
    process_noise : Array or None, default=None
        2x2 covariance for (v, omega).
    measurement_noise : Array or float or None, default=None
        Per-field observation noise variance.
    lin_process_noise : Array or float or None, default=None
        Linear process noise variance.
    p0_lin : Array or float or None, default=None
        Initial linear state covariance.
    threshold_ratio : float, default=0.5
        Threshold for resampling.
    measurement_mode : str, default='oracle'
        Prediction mode.
    oracle_fn : Callable or None, default=None
        Ground truth function.
    interpolator : Any or None, default=None
        Interpolator for map.
    gp_map : Any or None, default=None
        Gaussian Process map.
    pinn_map : Any or None, default=None
        PINN map.
    map_fn : Callable or None, default=None
        Custom map function.
    field_map : Any or None, default=None
        Alias for gp_map.
    enable_neutral_correction : bool, default=True
        Whether to apply mixed-variance neutral correction.
    untrusted_var_thresh : float or None, default=None
        Threshold for map variance to trigger neutral correction.
    seed : int, default=0
        Seed for JAX PRNG key.
    """

    N: int
    key: Array
    Q_nl: Array
    r: Array
    q_lin: Array
    p0: Array
    threshold_ratio: float
    measurement_mode: str
    oracle_fn: Callable | None
    interpolator: Any | None
    gp_map: Any | None
    pinn_map: Any | None
    map_fn: Callable | None
    enable_neutral_correction: bool
    untrusted_var_thresh: float
    poses: Array
    headings: Array
    speeds: Array
    xl: Array
    P: Array
    log_weights: Array
    trajectories: Array

    def __init__(
        self,
        n_particles: int = 100,
        process_noise: Array | None = None,
        measurement_noise: Array | float | None = None,
        lin_process_noise: Array | float | None = None,
        p0_lin: Array | float | None = None,
        threshold_ratio: float = 0.5,
        measurement_mode: Literal[
            "oracle", "interpolator", "gp", "pinn", "custom"
        ] = "oracle",
        oracle_fn: Callable | None = None,
        interpolator: Any | None = None,
        gp_map: Any | None = None,
        pinn_map: Any | None = None,
        map_fn: Callable | None = None,
        field_map: Any | None = None,
        enable_neutral_correction: bool = True,
        untrusted_var_thresh: float | None = None,
        seed: int = 0,
    ) -> None:
        self.N = n_particles
        self.key = jax.random.PRNGKey(seed)

        if process_noise is not None:
            self.Q_nl = jnp.asarray(process_noise)
        else:
            self.Q_nl = jnp.diag(jnp.array([0.25, jnp.deg2rad(9.0)])) ** 2

        if measurement_noise is not None:
            self.r = jnp.atleast_1d(jnp.asarray(measurement_noise))
        else:
            self.r = jnp.array([0.1**2])

        if lin_process_noise is not None:
            self.q_lin = jnp.atleast_1d(jnp.asarray(lin_process_noise))
        else:
            self.q_lin = jnp.array([1e-4**2])

        if p0_lin is not None:
            self.p0 = jnp.atleast_1d(jnp.asarray(p0_lin))
        else:
            self.p0 = jnp.array([0.05**2])

        self.threshold_ratio = threshold_ratio
        self.measurement_mode = measurement_mode
        self.oracle_fn = oracle_fn
        self.interpolator = interpolator
        self.gp_map = gp_map if gp_map is not None else field_map
        self.pinn_map = pinn_map
        self.map_fn = map_fn
        self.enable_neutral_correction = enable_neutral_correction
        self.untrusted_var_thresh = (
            untrusted_var_thresh if untrusted_var_thresh is not None else 0.90
        )

        # Initialize attributes
        self.poses = jnp.empty((0, 2))
        self.headings = jnp.empty((0,))
        self.speeds = jnp.empty((0,))
        self.xl = jnp.empty((0, 1))
        self.P = jnp.empty((0, 1, 1))
        self.log_weights = jnp.empty((0,))
        self.trajectories = jnp.empty((0, 1, 2))

    def initialize(
        self, initial_state: Array, std_dev: Array, n_fields: int = 1
    ) -> None:
        """Initializes the particle filter.

        Parameters
        ----------
        initial_state : Array
            Initial state [x, y, heading] or [x, y, heading, v].
        std_dev : Array
            Standard deviations for initial state matching length.
        n_fields : int
            Number of linear states per particle.
        """
        initial_state = jnp.asarray(initial_state)
        std_dev = jnp.asarray(std_dev)

        self.key, k1, k2, k3 = jax.random.split(self.key, 4)

        pos_noise = std_dev[:2] * jax.random.normal(k1, (self.N, 2))
        self.poses = jnp.broadcast_to(initial_state[:2], (self.N, 2)) + pos_noise

        heading_noise = std_dev[2] * jax.random.normal(k2, (self.N,))
        self.headings = initial_state[2] + heading_noise

        if len(initial_state) > 3:
            speed_noise = std_dev[3] * jax.random.normal(k3, (self.N,))
            self.speeds = initial_state[3] + speed_noise
        else:
            self.speeds = jnp.zeros((self.N,))

        self.xl = jnp.zeros((self.N, n_fields))
        p0_matrix = jnp.diag(jnp.broadcast_to(self.p0, (n_fields,)))
        self.P = jnp.tile(p0_matrix, (self.N, 1, 1))

        self.log_weights = jnp.full((self.N,), -jnp.log(self.N))
        self.trajectories = self.poses[:, None, :]

    def predict(self, control: Array, dt: float) -> None:
        """Predict step using differential drive motion model.

        Parameters
        ----------
        control : Array
            Control inputs [v_cmd, omega_cmd].
        dt : float
            Time step.
        """
        self.key, k1, k2 = jax.random.split(self.key, 3)
        control = jnp.asarray(control)

        v_std = jnp.sqrt(self.Q_nl[0, 0])
        w_std = jnp.sqrt(self.Q_nl[1, 1])

        v_cmd = control[0] + v_std * jax.random.normal(k1, (self.N,))
        w_cmd = control[1] + w_std * jax.random.normal(k2, (self.N,))

        dx = v_cmd * jnp.cos(self.headings) * dt
        dy = v_cmd * jnp.sin(self.headings) * dt
        dth = w_cmd * dt

        self.poses = self.poses + jnp.stack([dx, dy], axis=1)
        self.headings = self.headings + dth
        self.speeds = v_cmd

        self.trajectories = jnp.concatenate(
            [self.trajectories, self.poses[:, None, :]], axis=1
        )

        # Propagate linear covariance P += Q_lin
        n_fields = self.P.shape[1]
        q_lin_matrix = jnp.diag(jnp.broadcast_to(self.q_lin, (n_fields,)))
        self.P = self.P + q_lin_matrix

        # Reset weights to uniform
        self.log_weights = jnp.full((self.N,), -jnp.log(self.N))

    def predict_measurement(self, t: float | None = None) -> Any:
        """Predicts map values based on measurement mode.

        Parameters
        ----------
        t : float or None
            Time.

        Returns
        -------
        Array
            Predicted measurements.
        """
        mode = self.measurement_mode.lower()
        if mode == "oracle":
            if self.oracle_fn is None:
                raise ValueError("oracle_fn must be provided for oracle mode.")
            res = self.oracle_fn(t, self.poses)
            return res[0] if isinstance(res, tuple) else res
        elif mode == "interpolator":
            if self.interpolator is None:
                raise ValueError("interpolator must be provided for interpolator mode.")
            return self.interpolator(self.poses)
        elif mode == "gp":
            if self.gp_map is None:
                raise ValueError("gp_map must be provided for gp mode.")
            res = self.gp_map.predict(self.poses)
            return res[0] if isinstance(res, tuple) else res
        elif mode == "pinn":
            if self.pinn_map is None:
                raise ValueError("pinn_map must be provided for pinn mode.")
            return self.pinn_map.predict(t, self.poses)
        elif mode == "custom":
            if self.map_fn is None:
                raise ValueError("map_fn must be provided for custom mode.")
            if t is not None:
                return self.map_fn(t, self.poses)
            return self.map_fn(self.poses)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def update(
        self,
        measurement: Array,
        t_now: float | None = None,
        predicted_vals: Array | None = None,
        predicted_variances: Array | None = None,
    ) -> None:
        """Update step of the RBPF.

        Parameters
        ----------
        measurement : Array
            Observation.
        t_now : float or None
            Current time.
        predicted_vals : Array or None
            Pre-computed predicted values.
        predicted_variances : Array or None
            Pre-computed predicted variances.
        """
        if predicted_vals is None:
            if self.measurement_mode.lower() == "gp" and self.gp_map is not None:
                predicted_vals, predicted_variances = self.gp_map.predict(self.poses)
            else:
                pred_res = self.predict_measurement(t_now)
                if isinstance(pred_res, tuple):
                    predicted_vals = pred_res[0]
                    if predicted_variances is None and len(pred_res) > 1:
                        predicted_variances = pred_res[1]
                else:
                    predicted_vals = pred_res

        h_base = jnp.asarray(predicted_vals)
        obs_arr = jnp.asarray(measurement).ravel()

        n_fields = self.xl.shape[-1]

        if h_base.ndim == 1 and n_fields == 1:
            h_base_2d = h_base[:, None]
        elif h_base.ndim == 1 and n_fields > 1:
            h_base_2d = h_base[None, :]
        else:
            h_base_2d = h_base

        obs_2d = jnp.broadcast_to(obs_arr, (self.N, n_fields))

        e = obs_2d - h_base_2d - self.xl
        e = jnp.nan_to_num(e, nan=1e2)

        if predicted_variances is not None:
            map_var = jnp.asarray(predicted_variances)
            if map_var.ndim == 1 and n_fields > 1:
                map_var = jnp.broadcast_to(map_var[:, None], (self.N, n_fields))
            elif map_var.ndim == 1 and n_fields == 1:
                map_var = map_var[:, None]
        else:
            map_var = jnp.zeros((self.N, n_fields))

        r_matrix = jnp.diag(jnp.broadcast_to(self.r, (n_fields,)))
        R_tot = jnp.tile(r_matrix, (self.N, 1, 1)) + jax.vmap(jnp.diag)(map_var)

        S = self.P + R_tot + jnp.eye(n_fields) * 1e-6

        K = jnp.linalg.solve(S, self.P).transpose(0, 2, 1)

        e_vec = e[..., None]
        self.xl = self.xl + (K @ e_vec)[..., 0]
        self.P = (jnp.eye(n_fields) - K) @ self.P

        _, log_det_S = jnp.linalg.slogdet(S)
        S_inv_e = jnp.linalg.solve(S, e_vec)
        mahalanobis = (jnp.swapaxes(e_vec, 1, 2) @ S_inv_e)[..., 0, 0]

        log_w = -0.5 * (log_det_S + mahalanobis + n_fields * jnp.log(2.0 * jnp.pi))
        log_w = jnp.nan_to_num(log_w, nan=-1e6)

        if self.enable_neutral_correction:
            untrusted = jnp.any(map_var >= self.untrusted_var_thresh, axis=-1)
            trusted = ~untrusted

            n_trusted = jnp.sum(trusted)
            n_untrusted = jnp.sum(untrusted)

            def _apply_neutral(lw: Array) -> Array:
                lw_masked = jnp.where(trusted, lw, -1e9)
                c0 = jnp.max(lw_masked)
                exp_diff = jnp.where(trusted, jnp.exp(lw - c0), 0.0)
                mean_exp = jnp.sum(exp_diff) / jnp.maximum(n_trusted, 1)
                neutral_val = c0 + jnp.log(mean_exp)
                return jnp.where(untrusted, neutral_val, lw)

            log_w = jax.lax.cond(
                (n_trusted > 0) & (n_untrusted > 0),
                _apply_neutral,
                lambda lw: jnp.where(n_trusted == 0, jnp.zeros_like(lw), lw),
                log_w,
            )

        unnorm_log_weights = self.log_weights + log_w
        unnorm_log_weights = jnp.nan_to_num(unnorm_log_weights, nan=-1e6)
        self.log_weights = unnorm_log_weights - jax.scipy.special.logsumexp(
            unnorm_log_weights
        )

    def resample(self, threshold_ratio: float | None = None) -> None:
        """Resamples particles if N_eff falls below threshold.

        Parameters
        ----------
        threshold_ratio : float or None
            Threshold ratio. Uses self.threshold_ratio if None.
        """
        ratio = threshold_ratio if threshold_ratio is not None else self.threshold_ratio
        n_eff = self.effective_particle_number()

        if n_eff < (ratio * self.N):
            weights = jnp.exp(self.log_weights)
            weights = weights / jnp.sum(weights)

            self.key, k = jax.random.split(self.key)
            u0 = jax.random.uniform(k, shape=(), minval=0.0, maxval=1.0 / self.N)
            positions = (jnp.arange(self.N) / self.N) + u0
            cumsum = jnp.cumsum(weights)
            indices = jnp.clip(jnp.searchsorted(cumsum, positions), 0, self.N - 1)

            self.poses = self.poses[indices]
            self.headings = self.headings[indices]
            self.speeds = self.speeds[indices]
            self.xl = self.xl[indices]
            self.P = self.P[indices]
            self.trajectories = self.trajectories[indices]
            self.log_weights = jnp.full((self.N,), -jnp.log(self.N))

    def estimate(self) -> tuple[Array, Array]:
        """Gets the weighted mean state and covariance.

        Returns
        -------
        mean_state : Array
            [x, y, heading].
        cov_matrix : Array
            2x2 covariance.
        """
        weights = jnp.exp(self.log_weights)
        weights = weights / jnp.sum(weights)

        mean_pose = jnp.sum(self.poses * weights[:, None], axis=0)
        mean_heading = jnp.arctan2(
            jnp.sum(jnp.sin(self.headings) * weights),
            jnp.sum(jnp.cos(self.headings) * weights),
        )
        mean_state = jnp.array([mean_pose[0], mean_pose[1], mean_heading])

        diff = self.poses - mean_pose[None, :]
        cov_matrix = jnp.dot((diff * weights[:, None]).T, diff)

        return mean_state, cov_matrix

    def get_best_estimate(self) -> tuple[Array, Array]:
        """Gets the weighted mean trajectory and current pose estimate.

        Returns
        -------
        mean_trajectory : Array
            Weighted average trajectory.
        mean_pose : Array
            Weighted average current position.
        """
        weights = jnp.exp(self.log_weights)
        weights = jnp.nan_to_num(weights, nan=1.0 / self.N)
        weights = weights / jnp.sum(weights)
        weights_3d = weights[:, None, None]
        mean_trajectory = jnp.sum(self.trajectories * weights_3d, axis=0)
        mean_pose = mean_trajectory[-1]
        return mean_trajectory, mean_pose

    def effective_particle_number(self) -> Array:
        """Gets the effective number of particles.

        Returns
        -------
        Array
            Effective number of particles.
        """
        return jnp.exp(-jax.scipy.special.logsumexp(2.0 * self.log_weights))

    @property
    def state(self) -> RbpfState:
        """Read-only snapshot of the current state."""
        return RbpfState(
            poses=self.poses,
            headings=self.headings,
            log_weights=self.log_weights,
            trajectories=self.trajectories,
            xl=self.xl,
            P=self.P,
            speeds=self.speeds,
        )

    @staticmethod
    def _motion_model(x: Array, u: Array, dt: float) -> Array:
        """Single-state differential drive step.

        Parameters
        ----------
        x : Array
            State [x, y, heading, v] or similar.
        u : Array
            Control [v, omega].
        dt : float
            Time step.

        Returns
        -------
        Array
            Next state.
        """
        v, omega = u[0], u[1]
        x_next = x[0] + v * jnp.cos(x[2]) * dt
        y_next = x[1] + v * jnp.sin(x[2]) * dt
        theta_next = x[2] + omega * dt

        if len(x) > 3:
            return jnp.array([x_next, y_next, theta_next, v])
        return jnp.array([x_next, y_next, theta_next])


RBPFSLAM = RbpfSlam
