"""
test_scripts/demo_pipeline_two_paths.py
=======================================
End-to-end two-path calibration and SLAM pipeline demo:
- Loads pipeline specifications from configurations YAML.
- Generates simulated spatiotemporal passive scalar fields using PDE solver.
- Stage 1: Calibrates kinematic parameters (k_thrust) and reconstructs field (phi0).
- Stage 2: Performs joint SLAM optimization (trajectory corrections dx, diffusivity D,
  flow velocity v_flow) using the calibration from Stage 1 as a reference.

Run::

    uv run python test_scripts/demo_pipeline_two_paths.py
"""

from __future__ import annotations

import time
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from pde_slam.config import load_config
from pde_slam.interpolators import (
    FieldInterpolator,
    SpatialGrid,
    create_gaussian_plume,
    create_random_plumes,
    simulate_virtual_sensor,
)
from pde_slam.kinematics import UnicycleKinematics
from pde_slam.optimization import (
    KinematicsOptimizer,
    MultiPdeSlamOptimizer,
    unicycle_corrected_trajectory_fn,
)
from pde_slam.solver import AdvectionDiffusionSolver, PDEParams


def main() -> None:
    # ---------------------------------------------------------------------------
    # 1. Load Configurations and Initialize Environment
    # ---------------------------------------------------------------------------
    np.random.seed(42)

    config_path = Path("configs/pipeline_demo_config.yaml")
    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)

    # Spatial Grid
    grid = SpatialGrid(
        x_min=config.grid.x_min,
        x_max=config.grid.x_max,
        y_min=config.grid.y_min,
        y_max=config.grid.y_max,
        nx=config.grid.nx,
        ny=config.grid.ny,
    )
    solver = AdvectionDiffusionSolver(grid, dt_max=config.solver.dt_max)

    # Ground Truth PDE and Kinematics parameters
    D_true = jnp.array(config.pde_params.D)  # noqa: N806
    v_flow_true = jnp.array(config.pde_params.v_flow)
    k_thrust_true = config.pde_params.k_thrust

    # Define the initial condition fields
    if config.plumes.centers:
        plumes_list = []
        for center, width, amp in zip(
            config.plumes.centers, config.plumes.widths, config.plumes.amplitudes, strict=True
        ):
            c_arr = jnp.array(center)
            plumes_list.append(
                create_gaussian_plume(grid, center=c_arr, width=width, amplitude=amp)
            )
        phi0 = jnp.stack(plumes_list, axis=0)  # shape (K, ny, nx)
    else:
        phi0_single = create_random_plumes(
            grid, num_plumes=config.plumes.num_random, seed=config.plumes.seed
        )
        phi0 = jnp.expand_dims(phi0_single, axis=0)  # shape (1, ny, nx)

    print(f"Initial field condition defined with shape: {phi0.shape}")

    # ---------------------------------------------------------------------------
    # 2. Stage 1: Kinematic Calibration & Mapping using Path 1
    # ---------------------------------------------------------------------------
    print("\n--- STAGE 1: Kinematic Calibration & Initial Condition Estimation (Path 1) ---")
    # Mapping trajectory inputs (lawnmower pattern to cover the plumes)
    thrusts_list_1 = []
    headings_list_1 = []
    for i in range(5):
        h = np.pi / 2.0 if i % 2 == 0 else -np.pi / 2.0
        for _ in range(25):
            thrusts_list_1.append(80.0)
            headings_list_1.append(h)
        if i < 4:
            for _ in range(5):
                thrusts_list_1.append(80.0)
                headings_list_1.append(0.0)
    thrusts_1 = np.array(thrusts_list_1)
    headings_1 = np.array(headings_list_1)
    n_steps_1 = len(thrusts_1)
    dt_1 = 1.0
    t_traj_1 = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(jnp.full(n_steps_1, dt_1))])
    x0_1 = jnp.array([-60.0, -60.0])

    # Generate ground truth Path 1
    coords_true_1 = UnicycleKinematics.integrate_trajectory(
        x0_1, thrusts_1, headings_1, dt_1, k_thrust_true, include_initial=True
    )
    # Noisy coordinates (simulated GPS/odometry)
    coords_obs_1 = coords_true_1 + np.random.normal(0.0, 0.2, size=coords_true_1.shape)

    # Simulate Virtual Sensor spatiotemporal fields for Path 1
    u_field_true = jnp.broadcast_to(v_flow_true, (grid.ny, grid.nx, 2))
    pde_params_true = PDEParams(u_field=u_field_true, D=D_true)
    sensors_1 = simulate_virtual_sensor(grid, solver, phi0, pde_params_true, t_traj_1)

    # Query virtual sensor scalar readings along Path 1
    if not isinstance(sensors_1, list):
        sensors_1 = [sensors_1]
    obs_vals_1 = jnp.stack(
        [s(coords_true_1[:, 0], coords_true_1[:, 1], t_traj_1) for s in sensors_1], axis=-1
    )
    obs_vals_1 = obs_vals_1 + np.random.normal(0.0, 0.01, size=obs_vals_1.shape)

    # Calibrate k_thrust from coordinates and control inputs
    def unicycle_trajectory_fn(x0, thrusts, headings, dt, params):
        return UnicycleKinematics.integrate_trajectory(
            x0[:2], thrusts, headings, dt, params["k_thrust"], include_initial=True
        )

    kin_opt = KinematicsOptimizer(trajectory_fn=unicycle_trajectory_fn)
    k_thrust_guess = 3.0
    print(f"Initial k_thrust guess: {k_thrust_guess:.2f}")

    best_kin_params, kin_info = kin_opt.fit(
        coords_obs=coords_obs_1,
        thrusts=thrusts_1,
        headings=headings_1,
        dt=dt_1,
        init_params={"k_thrust": k_thrust_guess},
        bounds={"k_thrust": (0.1, 10.0)},
        method="l-bfgs-b",
    )
    k_thrust_est = float(best_kin_params["k_thrust"])
    print(f"Calibration complete. Estimated k_thrust: {k_thrust_est:.4f} (True: {k_thrust_true})")

    # Integrate the calibrated kinematics path to reconstruct coordinates
    coords_est_1 = UnicycleKinematics.integrate_trajectory(
        x0_1, thrusts_1, headings_1, dt_1, k_thrust_est, include_initial=True
    )

    # Back-advect the estimated coordinates to t = 0 to compensate for advection drift
    coords_back_advected = (
        np.array(coords_est_1) - np.array(v_flow_true) * np.array(t_traj_1)[:, None]
    )

    # Reconstruct initial condition phi0 from calibrated trajectory and observations
    print("Reconstructing initial passive scalar field conditions...")
    phi0_est_list = []
    kernel_args = {"kernel": "thin_plate_spline", "smoothing": 0.0}
    for k in range(phi0.shape[0]):
        field_interp = FieldInterpolator(grid, method="rbf", **kernel_args)
        phi0_est_k = field_interp.fit_predict(coords_back_advected, np.array(obs_vals_1[:, k]))
        phi0_est_list.append(phi0_est_k)
    phi0_est = jnp.stack(phi0_est_list, axis=0)

    # ---------------------------------------------------------------------------
    # 3. Stage 2: Joint SLAM Optimization on Path 2 (using Stage 1 results)
    # ---------------------------------------------------------------------------
    print("\n--- STAGE 2: Joint SLAM Optimization (Path 2) ---")
    n_steps_2 = 100
    dt_2 = 1.0
    times_2 = np.linspace(0.0, n_steps_2 * dt_2, n_steps_2)
    t_traj_2 = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(jnp.full(n_steps_2, dt_2))])

    # Path 2 commands (a separate loop pattern)
    thrusts_2 = 80.0 + 20.0 * np.sin(times_2 / 5.0)
    headings_2 = -times_2 / 10.0
    x0_2 = jnp.zeros(2)

    # Drifted trajectory for Path 2 (dead reckoning with estimated kinematics and drift corrections)
    dx_true_2 = 0.1 * np.column_stack([np.sin(t_traj_2 / 5.0), np.cos(t_traj_2 / 8.0)])
    dx_true_2 = dx_true_2 - dx_true_2[0]

    # Ground truth Path 2 (corrected trajectory)
    coords_true_2 = unicycle_corrected_trajectory_fn(
        x0_2, thrusts_2, headings_2, dt_2, k_thrust_true, dx_true_2
    )

    # Initial guess / dead-reckoned trajectory (without corrections)
    coords_drifted_2 = unicycle_corrected_trajectory_fn(
        x0_2, thrusts_2, headings_2, dt_2, k_thrust_est, jnp.zeros_like(dx_true_2)
    )

    # Simulate Virtual Sensor scalar readings along Path 2
    sensors_2 = simulate_virtual_sensor(grid, solver, phi0, pde_params_true, t_traj_2)
    if not isinstance(sensors_2, list):
        sensors_2 = [sensors_2]
    obs_vals_2 = jnp.stack(
        [s(coords_true_2[:, 0], coords_true_2[:, 1], t_traj_2) for s in sensors_2], axis=-1
    )
    obs_vals_2 = obs_vals_2 + np.random.normal(0.0, 0.01, size=obs_vals_2.shape)

    # Initialize parameters for joint SLAM optimizer
    D_init = jnp.full_like(D_true, 1.0)  # noqa: N806
    v_flow_init = jnp.array([0.0, 0.0])
    dx_init = jnp.zeros_like(dx_true_2)

    init_joint_params = {
        "D": D_init,
        "v_flow": v_flow_init,
        "dx": dx_init,
    }
    bounds = {
        "D": (0.01, 5.0),
        "v_flow": (-5.0, 5.0),
        "dx": (-30.0, 30.0),
    }

    joint_opt = MultiPdeSlamOptimizer(grid, solver)

    t_start = time.perf_counter()
    best_params_joint, joint_info = joint_opt.fit(
        phi0=phi0_est,
        obs_ts=t_traj_2,
        obs_vals=obs_vals_2,
        thrusts=thrusts_2,
        headings=headings_2,
        dt=dt_2,
        init_params=init_joint_params,
        bounds=bounds,
        lambda_reg=config.optimization.lambda_reg,
        k_thrust_fixed=k_thrust_est,
        method=config.optimization.method,
        options={"maxiter": config.optimization.maxiter, "disp": True},
    )
    t_elapsed = time.perf_counter() - t_start
    print(f"Joint optimization completed in {t_elapsed:.2f} seconds.")

    # Results extraction
    D_est = best_params_joint["D"]  # noqa: N806
    v_flow_est = best_params_joint["v_flow"]
    dx_est = best_params_joint["dx"]

    coords_opt_2 = unicycle_corrected_trajectory_fn(
        jnp.zeros(2), thrusts_2, headings_2, dt_2, k_thrust_est, dx_est
    )

    print("\nResults comparison:")
    print(f"  Diffusivity D: True = {D_true}, Est = {D_est}")
    print(f"  Flow Velocity: True = {v_flow_true}, Est = {v_flow_est}")
    print(f"  Optimization Success: {joint_info['success']}")
    print(f"  Final Loss: {joint_info['fun']:.6f}")

    # ---------------------------------------------------------------------------
    # 4. Visualisation and Plotting
    # ---------------------------------------------------------------------------
    print("\nPlotting results...")
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)
    fig.suptitle("PDE-SLAM Two-Path Calibration & SLAM Demo", fontsize=15, fontweight="bold")

    # Plot 1: Stage 1 Trajectory & Measurements Calibration
    ax = axes[0, 0]
    ax.plot(coords_true_1[:, 0], coords_true_1[:, 1], "g-", linewidth=2.0, label="GT Path 1")
    ax.scatter(coords_obs_1[:, 0], coords_obs_1[:, 1], c="red", alpha=0.4, s=12, label="GPS Obs 1")
    ax.plot(coords_est_1[:, 0], coords_est_1[:, 1], "b--", linewidth=1.5, label="Calib Kinematics")
    ax.set_title("Stage 1: Kinematic Calibration (Path 1)")
    ax.set_xlabel("East position [m]")
    ax.set_ylabel("North position [m]")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_aspect("equal")

    # Plot 2: Stage 2 SLAM Trajectory
    ax = axes[0, 1]
    ax.plot(coords_true_2[:, 0], coords_true_2[:, 1], "g-", linewidth=2.5, label="GT Path 2")
    ax.plot(
        coords_drifted_2[:, 0], coords_drifted_2[:, 1], "r-.", linewidth=1.5, label="Drifted Guess"
    )
    ax.plot(coords_opt_2[:, 0], coords_opt_2[:, 1], "b-", linewidth=2.0, label="Joint Optimized")
    ax.set_title("Stage 2: Joint SLAM Trajectory (Path 2)")
    ax.set_xlabel("East position [m]")
    ax.set_ylabel("North position [m]")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_aspect("equal")

    # Plot 3: Initial Condition Comparison (RGB or single channel)
    ax = axes[1, 0]
    num_fields = phi0.shape[0]
    if num_fields >= 3:
        # Stack first 3 fields as RGB channels (Red=P1, Green=P2, Blue=P3)
        rgb_true = jnp.clip(jnp.stack([phi0[0], phi0[1], phi0[2]], axis=-1), 0.0, 1.0)
        im = ax.imshow(
            np.array(rgb_true),
            origin="lower",
            extent=[grid.x_min, grid.x_max, grid.y_min, grid.y_max],
        )
        title_suffix = " (Red=P1, Green=P2, Blue=P3)"
    else:
        im = ax.imshow(
            np.array(phi0[0]),
            origin="lower",
            extent=[grid.x_min, grid.x_max, grid.y_min, grid.y_max],
            cmap="viridis",
        )
        fig.colorbar(im, ax=ax, shrink=0.8, label="Concentration")
        title_suffix = " (Field 1)"

    ax.scatter(coords_true_1[:, 0], coords_true_1[:, 1], c="white", s=2, alpha=0.5, label="Path 1")
    ax.set_title(f"Ground Truth Initial Condition{title_suffix}")
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.legend()

    # Plot 4: Reconstructed Initial Condition (RGB or single channel)
    ax = axes[1, 1]
    if num_fields >= 3:
        # Stack first 3 fields as RGB channels (Red=P1, Green=P2, Blue=P3)
        rgb_est = jnp.clip(jnp.stack([phi0_est[0], phi0_est[1], phi0_est[2]], axis=-1), 0.0, 1.0)
        im = ax.imshow(
            np.array(rgb_est),
            origin="lower",
            extent=[grid.x_min, grid.x_max, grid.y_min, grid.y_max],
        )
    else:
        im = ax.imshow(
            np.array(phi0_est[0]),
            origin="lower",
            extent=[grid.x_min, grid.x_max, grid.y_min, grid.y_max],
            cmap="viridis",
        )
        fig.colorbar(im, ax=ax, shrink=0.8, label="Concentration")

    ax.set_title(f"Stage 1 Reconstructed IC{title_suffix}")
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")

    out_path = output_dir / "demo_pipeline_two_paths.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved figure to: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
