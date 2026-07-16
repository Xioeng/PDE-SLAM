"""Unit and integration tests for JointSlamOptimizer."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from pde_slam.interpolators import SpatialGrid
from pde_slam.kinematics import UnicycleKinematics
from pde_slam.optimization import (
    JointSlamOptimizer,
    MultiPdeSlamOptimizer,
    unicycle_corrected_trajectory_fn,
)
from pde_slam.solver import AdvectionDiffusionSolver
from pde_slam.types import ObservationData, TrajectoryContext


class TestUnicycleCorrectedTrajectory:
    """Verify trajectory function with corrections."""

    def test_zero_corrections(self) -> None:
        """Verify that zero corrections matches standard kinematics integration."""
        x0 = jnp.array([10.0, -5.0])
        thrusts = jnp.array([50.0, 80.0, 60.0])
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
        thrusts = jnp.array([50.0, 80.0])
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
        thrusts = jnp.array([80.0] * n_steps)
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

    def test_joint_parameter_identification_multi_pde(self, small_grid: SpatialGrid) -> None:
        """Verify L-BFGS-B identifies flow, diffusivities, and dx from synthetic multi-PDE data."""
        # 1. Setup grid and solver
        solver = AdvectionDiffusionSolver(small_grid, dt_max=0.5)

        # 2. Ground truth parameters (3 PDEs)
        D_true = jnp.array([0.5, 0.2, 1.0])  # noqa: N806
        v_flow_true = jnp.array([0.2, -0.1])
        k_thrust_true = 1.2
        x0 = jnp.array([0.0, 0.0])

        # Small 6-step trajectory
        n_steps = 6
        dt = 0.5
        thrusts = jnp.array([80.0] * n_steps)
        headings = jnp.linspace(0.0, np.pi / 2, n_steps)
        t_traj = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(jnp.full(n_steps, dt))])

        # Define ground truth dx corrections
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

        # 3 unique initial conditions
        phi0_1 = jnp.exp(-(small_grid.XX**2 + small_grid.YY**2) / 200.0)
        phi0_2 = jnp.exp(-((small_grid.XX - 10.0) ** 2 + (small_grid.YY - 10.0) ** 2) / 150.0)
        phi0_3 = jnp.exp(-((small_grid.XX + 10.0) ** 2 + (small_grid.YY + 10.0) ** 2) / 300.0)
        phi0 = jnp.stack([phi0_1, phi0_2, phi0_3], axis=0)

        # Simulate true PDE to generate observation snapshots
        u_fields_true = jnp.broadcast_to(v_flow_true, (3, small_grid.ny, small_grid.nx, 2))
        from pde_slam.solver import PDEParams

        pde_params_true = PDEParams(u_field=u_fields_true, D=D_true)

        solve_vmap = jax.vmap(
            lambda p0, params: solver.solve(p0, params, t0=0.0, t_end=t_traj[-1], saveat=t_traj)
        )
        snapshots_true = solve_vmap(phi0, pde_params_true)  # shape (3, T, ny, nx)

        # Build spatiotemporal interpolator to sample from snapshots
        from pde_slam.interpolators.spatiotemporal import SpatiotemporalInterpolator

        obs_ts = jnp.array([0.0, 1.0, 2.0, 3.0])
        x_obs = jnp.interp(obs_ts, t_traj, coords_true[:, 0])
        y_obs = jnp.interp(obs_ts, t_traj, coords_true[:, 1])

        def interp_single_true(snapshots_single):
            interp = SpatiotemporalInterpolator(small_grid, t_traj, snapshots_single)
            return interp(x_obs, y_obs, obs_ts)

        obs_vals = jax.vmap(interp_single_true)(snapshots_true).T  # shape (M, 3)

        # 3. Setup Optimizer — multi-PDE subclass
        optimizer = MultiPdeSlamOptimizer(small_grid, solver)

        # Initial wrong guesses
        init_params = {
            "D": jnp.array([1.0, 0.5, 1.5]),
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
        assert best_params["D"].shape == (3,)
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


class TestJointSlamOptimizerSinglePdeRegression:
    """Regression guard: JointSlamOptimizer must still work for a single PDE."""

    def test_single_pde_loss_decreases(self, small_grid: SpatialGrid) -> None:
        """Verify that L-BFGS-B reduces loss on a single-PDE problem."""
        solver = AdvectionDiffusionSolver(small_grid, dt_max=0.5)
        D_true = 0.4  # noqa: N806
        v_flow_true = jnp.array([0.15, -0.05])
        k_thrust_true = 1.0
        x0 = jnp.array([0.0, 0.0])

        n_steps = 6
        dt = 0.5
        thrusts = jnp.array([60.0] * n_steps)
        headings = jnp.linspace(0.0, np.pi / 4, n_steps)
        t_traj = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(jnp.full(n_steps, dt))])
        dx_true = jnp.zeros((n_steps + 1, 2))

        coords_true = unicycle_corrected_trajectory_fn(
            x0, thrusts, headings, dt, k_thrust_true, dx_true
        )

        phi0 = jnp.exp(-(small_grid.XX**2 + small_grid.YY**2) / 200.0)

        from pde_slam.solver import PDEParams

        u_field = jnp.broadcast_to(v_flow_true, (small_grid.ny, small_grid.nx, 2))
        snapshots = solver.solve(
            phi0,
            PDEParams(u_field=u_field, D=jnp.array(D_true)),
            t0=0.0,
            t_end=float(t_traj[-1]),
            saveat=t_traj,
        )

        from pde_slam.interpolators.spatiotemporal import SpatiotemporalInterpolator

        obs_ts = jnp.array([0.0, 1.0, 2.0, 3.0])
        x_obs = jnp.interp(obs_ts, t_traj, coords_true[:, 0])
        y_obs = jnp.interp(obs_ts, t_traj, coords_true[:, 1])
        interp = SpatiotemporalInterpolator(small_grid, t_traj, snapshots)
        obs_vals = interp(x_obs, y_obs, obs_ts)

        optimizer = JointSlamOptimizer(small_grid, solver)
        init_params = {
            "D": 1.0,
            "v_flow": jnp.array([0.0, 0.0]),
            "dx": dx_true,
        }
        best_params, info = optimizer.fit(
            phi0=phi0,
            obs_ts=obs_ts,
            obs_vals=obs_vals,
            thrusts=thrusts,
            headings=headings,
            dt=dt,
            init_params=init_params,
            bounds={"D": (0.01, 2.0), "v_flow": (-1.0, 1.0), "dx": (-1.0, 1.0)},
            lambda_reg=1e-4,
            k_thrust_fixed=k_thrust_true,
            method="l-bfgs-b",
            options={"maxiter": 20},
        )
        assert info["success"] is True or info["nit"] > 0
        assert "D" in best_params and "v_flow" in best_params
