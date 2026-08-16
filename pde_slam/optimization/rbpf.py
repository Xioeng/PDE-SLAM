"""
rbpf.py
=======
Rao-Blackwellized Particle Filter (RBPF) SLAM for aquatic robots.

This module provides a JAX-vectorized particle filter that estimates robot pose
trajectories under control actuation noise while conditioning scalar field map predictions
on candidate particle poses.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from pde_slam.kinematics import UnicycleKinematics


class RbpfState(NamedTuple):
    """Container for the Rao-Blackwellized Particle Filter state.

    Attributes
    ----------
    poses : Array
        Array of current 2-D particle positions [east_m, north_m], shape (num_particles, 2).
    headings : Array
        Array of current particle compass headings [rad], shape (num_particles,).
    log_weights : Array
        Normalized log-weights of particles, shape (num_particles,).
    trajectories : Array
        History of particle positions, shape (num_particles, T, 2).
    """

    poses: Array
    headings: Array
    log_weights: Array
    trajectories: Array


class RbpfSlam:
    """JAX-accelerated Rao-Blackwellized Particle Filter for pose trajectory estimation."""

    @staticmethod
    def init_state(
        x0: Array,
        heading0: float | Array,
        num_particles: int,
        key: Array,
        pos_init_std: float = 0.5,
        heading_init_std: float = 0.05,
    ) -> RbpfState:
        """Initializes particle filter state with a Gaussian cloud around starting pose.

        Parameters
        ----------
        x0 : Array
            Initial 2-D position [east_m, north_m], shape (2,).
        heading0 : float or Array
            Initial compass heading [rad].
        num_particles : int
            Number of particles M.
        key : Array
            JAX PRNG key.
        pos_init_std : float, default=0.5
            Standard deviation of initial position noise [m].
        heading_init_std : float, default=0.05
            Standard deviation of initial heading noise [rad].

        Returns
        -------
        state : RbpfState
            Initialized particle filter state.
        """
        k1, k2 = jax.random.split(key)
        pos_noise = pos_init_std * jax.random.normal(k1, shape=(num_particles, 2))
        heading_noise = heading_init_std * jax.random.normal(k2, shape=(num_particles,))

        poses = jnp.broadcast_to(x0[:2], (num_particles, 2)) + pos_noise
        headings = jnp.broadcast_to(jnp.asarray(heading0), (num_particles,)) + heading_noise
        log_weights = jnp.full((num_particles,), -jnp.log(num_particles))
        trajectories = poses[:, None, :]  # shape (num_particles, 1, 2)

        return RbpfState(
            poses=poses,
            headings=headings,
            log_weights=log_weights,
            trajectories=trajectories,
        )

    @staticmethod
    def predict(
        state: RbpfState,
        thrust: float | Array,
        heading_cmd: float | Array,
        dt: float | Array,
        k_thrust: float | Array,
        thrust_noise_std: float,
        heading_noise_std: float,
        key: Array,
    ) -> RbpfState:
        """Propagates particle states forward using the unicycle kinematic model.

        Parameters
        ----------
        state : RbpfState
            Current particle filter state.
        thrust : float or Array
            Commanded thrust input.
        heading_cmd : float or Array
            Commanded compass heading [rad].
        dt : float or Array
            Time step duration [s].
        k_thrust : float or Array
            Kinematic thrust conversion parameter.
        thrust_noise_std : float
            Standard deviation of thrust actuation noise.
        heading_noise_std : float
            Standard deviation of heading actuation noise [rad].
        key : Array
            JAX PRNG key.

        Returns
        -------
        new_state : RbpfState
            Propagated particle filter state.
        """
        num_particles = state.poses.shape[0]
        k1, k2 = jax.random.split(key)

        noisy_thrusts = thrust + thrust_noise_std * jax.random.normal(k1, (num_particles,))
        noisy_headings = heading_cmd + heading_noise_std * jax.random.normal(k2, (num_particles,))

        # Unicycle displacement step
        speeds = k_thrust * (noisy_thrusts / 100.0)
        dx = speeds * jnp.sin(noisy_headings) * dt
        dy = speeds * jnp.cos(noisy_headings) * dt

        new_poses = state.poses + jnp.stack([dx, dy], axis=1)
        new_headings = noisy_headings

        new_trajectories = jnp.concatenate(
            [state.trajectories, new_poses[:, None, :]], axis=1
        )

        return RbpfState(
            poses=new_poses,
            headings=new_headings,
            log_weights=state.log_weights,
            trajectories=new_trajectories,
        )

    @staticmethod
    def update_measurement(
        state: RbpfState,
        obs_val: float | Array,
        predicted_vals: Array,
        obs_std: float = 0.1,
    ) -> RbpfState:
        """Updates particle log-weights based on field observation likelihood.

        Parameters
        ----------
        state : RbpfState
            Current particle filter state.
        obs_val : float or Array
            Observed scalar feature value.
        predicted_vals : Array
            Predicted scalar values for each particle, shape (num_particles,).
        obs_std : float, default=0.1
            Observation noise standard deviation.

        Returns
        -------
        new_state : RbpfState
            Updated particle filter state with normalized log-weights.
        """
        residuals = obs_val - predicted_vals
        log_likelihoods = -0.5 * (residuals / obs_std) ** 2 - jnp.log(
            obs_std * jnp.sqrt(2.0 * jnp.pi)
        )

        unnorm_log_weights = state.log_weights + log_likelihoods
        norm_log_weights = unnorm_log_weights - jax.scipy.special.logsumexp(unnorm_log_weights)

        return RbpfState(
            poses=state.poses,
            headings=state.headings,
            log_weights=norm_log_weights,
            trajectories=state.trajectories,
        )

    @staticmethod
    def effective_particle_number(log_weights: Array) -> Array:
        """Computes the effective number of particles N_eff = 1 / sum(w_i^2).

        Parameters
        ----------
        log_weights : Array
            Normalized log-weights of particles, shape (num_particles,).

        Returns
        -------
        n_eff : Array
            Effective number of particles (scalar).
        """
        return jnp.exp(-jax.scipy.special.logsumexp(2.0 * log_weights))

    @staticmethod
    def resample_if_needed(state: RbpfState, key: Array, threshold_ratio: float = 0.5) -> RbpfState:
        """Resamples particles if N_eff falls below threshold_ratio * num_particles.

        Parameters
        ----------
        state : RbpfState
            Current particle filter state.
        key : Array
            JAX PRNG key.
        threshold_ratio : float, default=0.5
            Threshold ratio (e.g. 0.5 means N_eff < M / 2).

        Returns
        -------
        new_state : RbpfState
            Resampled state (or untouched state if N_eff is sufficient).
        """
        num_particles = state.poses.shape[0]
        n_eff = RbpfSlam.effective_particle_number(state.log_weights)

        def _do_resample(s: RbpfState) -> RbpfState:
            weights = jnp.exp(s.log_weights)
            indices = jax.random.choice(key, num_particles, shape=(num_particles,), p=weights)
            resampled_poses = s.poses[indices]
            resampled_headings = s.headings[indices]
            resampled_trajectories = s.trajectories[indices]
            reset_log_weights = jnp.full((num_particles,), -jnp.log(num_particles))

            return RbpfState(
                poses=resampled_poses,
                headings=resampled_headings,
                log_weights=reset_log_weights,
                trajectories=resampled_trajectories,
            )

        should_resample = n_eff < (threshold_ratio * num_particles)
        return jax.lax.cond(should_resample, _do_resample, lambda s: s, state)

    @staticmethod
    def get_best_estimate(state: RbpfState) -> tuple[Array, Array]:
        """Extracts the weighted mean trajectory and current pose estimate.

        Parameters
        ----------
        state : RbpfState
            Current particle filter state.

        Returns
        -------
        mean_trajectory : Array
            Weighted average estimated trajectory [east_m, north_m], shape (T, 2).
        mean_pose : Array
            Weighted average current position [east_m, north_m], shape (2,).
        """
        weights = jnp.exp(state.log_weights)[:, None, None]  # shape (num_particles, 1, 1)
        mean_trajectory = jnp.sum(state.trajectories * weights, axis=0)
        mean_pose = mean_trajectory[-1]
        return mean_trajectory, mean_pose
