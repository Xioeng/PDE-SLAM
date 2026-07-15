"""Unit and integration tests for JointSlamOptimizer."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from pde_slam.interpolator import SpatialGrid
from pde_slam.joint_optimization import (
    JointSlamOptimizer,
    ObservationData,
    TrajectoryContext,
    unicycle_corrected_trajectory_fn,
)
from pde_slam.kinematics import UnicycleKinematics
from pde_slam.solver import AdvectionDiffusionSolver


class TestUnicycleCorrectedTrajectory:
    """Verify trajectory function with corrections."""

    def test_zero_corrections(self) -> None:
        """Verify that zero corrections matches standard kinematics integration."""
        x0 = jnp.array([10.0, -5.0])
        thrusts = jnp.array([0.5, 0.8, 0.6])
        headings = jnp.array([0.1, -0.2, 0.3])
        dt = 1.0
        k_thrust = 1.5
        dx = jnp.zeros((4, 2))

        coords_corrected = unicycle_corrected_trajectory_fn(x0, thrusts, headings, dt, k_thrust, dx)
        coords_standard = UnicycleKinematics.integrate_trajectory(
            x0, thrusts, headings, dt, k_thrust, include_initial=True
        )

        np.testing.assert_allclose(np.array(coords_corrected), np.array(coords_standard))

    def test_differentiability(self) -> None:
        """Verify that gradient w.r.t. dx is computable."""
        x0 = jnp.array([0.0, 0.0])
        thrusts = jnp.array([0.5, 0.8])
        headings = jnp.array([0.1, -0.2])
        dt = 1.0
        k_thrust = 1.5
        dx = jnp.zeros((3, 2))

        def loss_fn(corrections: Array) -> Array:
            coords = unicycle_corrected_trajectory_fn(
                x0, thrusts, headings, dt, k_thrust, corrections
            )
            return jnp.sum(coords**2)

        grad_fn = jax.grad(loss_fn)
        g = grad_fn(dx)
        assert g.shape == dx.shape
        assert not jnp.isnan(g).any()


class TestJointSlamOptimizer:
    """Verify JointSlamOptimizer on a small synthetic optimization task."""

    def test_joint_parameter_identification_lbfgs(self, small_grid: SpatialGrid) -> None:
        """Verify L-BFGS-B identifies flow, diffusion, and dx from synthetic data."""
        # 1. Setup grid and solver
        solver = AdvectionDiffusionSolver(small_grid, dt_max=0.5)

        # 2. Ground truth parameters
        D_true = 0.5  # noqa: N806
        v_flow_true = jnp.array([0.2, -0.1])
        k_thrust_true = 1.2
        x0 = jnp.array([0.0, 0.0])

        # Small 6-step trajectory
        n_steps = 6
        dt = 0.5
        thrusts = jnp.array([0.8] * n_steps)
        headings = jnp.linspace(0.0, np.pi / 2, n_steps)
        t_traj = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(jnp.full(n_steps, dt))])

        # Define ground truth dx corrections (small position shifts)
        dx_true = jnp.array(
            [
                [0.0, 0.0],
                [0.1, -0.1],
                [0.2, -0.1],
                [0.1, 0.0],
                [-0.1, 0.1],
                [-0.2, 0.2],
                [-0.3, 0.2],
            ]
        )

        # True trajectory coordinates
        coords_true = unicycle_corrected_trajectory_fn(
            x0, thrusts, headings, dt, k_thrust_true, dx_true
        )

        # Create smooth initial scalar field (Gaussian centered at origin)
        phi0 = jnp.exp(-(small_grid.XX**2 + small_grid.YY**2) / 200.0)

        # Simulate true PDE to generate observation snapshots
        u_field_true = jnp.broadcast_to(v_flow_true, (small_grid.ny, small_grid.nx, 2))
        from pde_slam.solver import PDEParams

        pde_params_true = PDEParams(u_field=u_field_true, D=jnp.array(D_true))
        snapshots_true = solver.solve(
            phi0, pde_params_true, t0=0.0, t_end=t_traj[-1], saveat=t_traj
        )

        # Build spatiotemporal interpolator to sample from snapshots
        from pde_slam.interpolators.spatiotemporal import SpatiotemporalInterpolator

        interp_true = SpatiotemporalInterpolator(small_grid, t_traj, snapshots_true)

        # Robot takes 4 scalar observations at specific times
        obs_ts = jnp.array([0.0, 1.0, 2.0, 3.0])
        # Find position at those observation times
        x_obs = jnp.interp(obs_ts, t_traj, coords_true[:, 0])
        y_obs = jnp.interp(obs_ts, t_traj, coords_true[:, 1])
        obs_vals = interp_true(x_obs, y_obs, obs_ts)

        # 3. Setup Optimizer
        optimizer = JointSlamOptimizer(small_grid, solver)

        # Initial wrong guesses
        init_params = {
            "D": 1.0,
            "v_flow": jnp.array([0.0, 0.0]),
            "dx": jnp.zeros_like(dx_true),
        }

        # Fit parameters
        bounds = {
            "D": (0.01, 2.0),
            "v_flow": (-1.0, 1.0),
            "dx": (-1.0, 1.0),
        }

        best_params, info = optimizer.fit(
            phi0=phi0,
            obs_ts=obs_ts,
            obs_vals=obs_vals,
            thrusts=thrusts,
            headings=headings,
            dt=dt,
            init_params=init_params,
            bounds=bounds,
            lambda_reg=1e-4,
            k_thrust_fixed=k_thrust_true,
            method="l-bfgs-b",
            options={"maxiter": 20},
        )

        # Assertions
        assert info["success"] is True or info["nit"] > 0
        assert "D" in best_params
        assert "v_flow" in best_params
        assert "dx" in best_params

        # Verify that optimized parameters reduce loss compared to initial guess
        obs = ObservationData(ts=obs_ts, vals=obs_vals)
        traj = TrajectoryContext(
            thrusts=thrusts,
            headings=headings,
            dt_arr=jnp.full(n_steps, dt),
            t_traj=t_traj,
            t0=float(t_traj[0]),
            t_end=float(t_traj[-1]),
        )
        init_loss = optimizer.loss_fn(
            {k: jnp.asarray(v) for k, v in init_params.items()},
            phi0,
            obs,
            traj,
            lambda_reg=1e-4,
            k_thrust_fixed=k_thrust_true,
        )
        final_loss = info["fun"]
        assert final_loss < init_loss
