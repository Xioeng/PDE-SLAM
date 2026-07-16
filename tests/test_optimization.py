"""Tests for pde_slam.optimization (KinematicsOptimizer class)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from pde_slam.kinematics import UnicycleKinematics
from pde_slam.optimization import KinematicsOptimizer


class TestKinematicsOptimizer:
    """Unit and integration tests for KinematicsOptimizer."""

    def test_unicycle_parameter_identification_lbfgs(self) -> None:
        """Verify L-BFGS-B identifies k_thrust from synthetic unicycle trajectory."""
        # 1. Generate ground truth trajectory
        k_thrust_true = 1.6
        dt = 0.5
        n_steps = 40

        robot = UnicycleKinematics(k_thrust=k_thrust_true, x0=10.0, y0=-5.0, heading0=0.0)

        # Create control inputs: time-varying thrust and heading
        np.random.seed(42)
        thrusts = np.random.uniform(30.0, 90.0, size=n_steps)
        headings = np.random.uniform(-np.pi, np.pi, size=n_steps)

        # Run trajectory to get coords_obs
        states = robot.trajectory(thrusts, headings, dt=dt, include_initial=True)
        coords_obs = states[:, :2]  # shape (n_steps + 1, 2)

        # 2. Setup Optimizer
        optimizer = KinematicsOptimizer()

        # Try to identify k_thrust starting from initial guess
        init_params = {"k_thrust": 1.0}
        bounds = {"k_thrust": (0.1, 5.0)}

        best_params, info = optimizer.fit(
            coords_obs=coords_obs,
            thrusts=thrusts,
            headings=headings,
            dt=dt,
            init_params=init_params,
            bounds=bounds,
            method="l-bfgs-b",
        )

        assert info["success"] is True
        assert best_params["k_thrust"] == pytest.approx(k_thrust_true, rel=1e-5)
        assert info["fun"] < 1e-10

    def test_unicycle_parameter_identification_optax(self) -> None:
        """Verify Optax Adam optimizer identifies k_thrust from synthetic unicycle trajectory."""
        k_thrust_true = 1.25
        dt = 1.0
        n_steps = 30

        robot = UnicycleKinematics(k_thrust=k_thrust_true, x0=0.0, y0=0.0, heading0=0.0)
        thrusts = np.full(n_steps, 80.0)
        headings = np.linspace(-np.pi / 2, np.pi / 2, n_steps)

        states = robot.trajectory(thrusts, headings, dt=dt, include_initial=True)
        coords_obs = states[:, :2]

        optimizer = KinematicsOptimizer()
        init_params = {"k_thrust": 0.5}

        # Fit with Adam
        best_params, info = optimizer.fit(
            coords_obs=coords_obs,
            thrusts=thrusts,
            headings=headings,
            dt=dt,
            init_params=init_params,
            method="adam",
            options={"learning_rate": 0.05, "num_steps": 150},
        )

        assert best_params["k_thrust"] == pytest.approx(k_thrust_true, rel=1e-2)
        assert info["fun"] < 1e-3

    def test_custom_trajectory_function(self) -> None:
        """Test optimizer with a custom kinematic trajectory function of multiple parameters.

        Model:
            x_{t+1} = x_t + (k_thrust * thrust_t - k_drag * speed_t**2) * sin(heading_t) * dt
            y_{t+1} = y_t + (k_thrust * thrust_t - k_drag * speed_t**2) * cos(heading_t) * dt
        """

        # Define the custom trajectory function
        def custom_trajectory(
            x0: Array,
            thrusts: Array,
            headings: Array,
            dt: float | Array,
            params: dict[str, Array],
        ) -> Array:
            k_thrust = params["k_thrust"]
            k_drag = params["k_drag"]

            # Compute displacements step by step
            # Note: JAX cumsum can be used if we build speeds sequentially.
            # For this simple model: speed = k_thrust * thrust - k_drag * (k_thrust * thrust)**2
            base_speed = k_thrust * thrusts
            net_speed = base_speed - k_drag * (base_speed**2)

            dx = net_speed * jnp.sin(headings) * dt
            dy = net_speed * jnp.cos(headings) * dt

            xs = x0[0] + jnp.cumsum(dx)
            ys = x0[1] + jnp.cumsum(dy)

            xs = jnp.concatenate([jnp.array([x0[0]]), xs])
            ys = jnp.concatenate([jnp.array([x0[1]]), ys])
            return jnp.stack([xs, ys], axis=-1)

        # Generate fake data
        x0 = jnp.array([0.0, 0.0])
        thrusts = jnp.array([0.5, 0.8, 0.6])
        headings = jnp.array([0.1, -0.2, 0.3])
        dt = 1.0

        true_params = {"k_thrust": 2.0, "k_drag": 0.15}
        coords_obs = custom_trajectory(x0, thrusts, headings, dt, true_params)

        # Run optimizer
        optimizer = KinematicsOptimizer(trajectory_fn=custom_trajectory)
        init_params = {"k_thrust": 1.0, "k_drag": 0.0}
        bounds = {"k_thrust": (0.1, 5.0), "k_drag": (0.0, 1.0)}

        best_params, info = optimizer.fit(
            coords_obs=coords_obs,
            thrusts=thrusts,
            headings=headings,
            dt=dt,
            init_params=init_params,
            bounds=bounds,
            method="l-bfgs-b",
        )

        assert info["success"] is True
        assert best_params["k_thrust"] == pytest.approx(2.0, rel=1e-4)
        assert best_params["k_drag"] == pytest.approx(0.15, rel=1e-4)
        assert info["fun"] < 1e-10

    def test_invalid_method_raises(self) -> None:
        """Verify fitting with an invalid method name raises ValueError."""
        optimizer = KinematicsOptimizer()
        with pytest.raises(ValueError, match="Unknown optimization method"):
            optimizer.fit(
                coords_obs=jnp.zeros((5, 2)),
                thrusts=jnp.zeros(4),
                headings=jnp.zeros(4),
                dt=1.0,
                init_params={"k_thrust": 1.0},
                method="invalid-method-name",
            )
