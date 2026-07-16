"""
test_scripts/demo_joint_optimization.py
=======================================
Demonstrates joint identification of PDE physical parameters (diffusivity, flow velocity)
and trajectory position corrections using L-BFGS-B and Adam (Optax).

Run::

    uv run python test_scripts/demo_joint_optimization.py
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from pde_slam.interpolators import SpatialGrid, SpatiotemporalInterpolator
from pde_slam.optimization import (
    MultiPdeSlamOptimizer,
    unicycle_corrected_trajectory_fn,
)
from pde_slam.solver import AdvectionDiffusionSolver, PDEParams


def main() -> None:
    # ---------------------------------------------------------------------------
    # 1. Setup Environment, Grid, and Solver
    # ---------------------------------------------------------------------------
    np.random.seed(42)

    # Spatial Grid
    grid = SpatialGrid(
        x_min=-150.0, x_max=150.0, y_min=-150.0, y_max=150.0, nx=100, ny=100
    )
    solver = AdvectionDiffusionSolver(grid, dt_max=1.0)

    # Ground Truth PDE Parameters
    D_true = jnp.array([0.6, 0.2, 1.2])  # noqa: N806
    v_flow_true = 1 * jnp.array([0.5, -0.3])  # constant flow (East, North)
    k_thrust_true = 5.0

    # Three unique initial scalar fields: wider Gaussian plumes at different locations
    phi0_1 = jnp.exp(-((grid.XX + 30.0) ** 2 + (grid.YY - 80.0) ** 2) / 4000.0)
    phi0_2 = jnp.exp(-((grid.XX - 10.0) ** 2 + (grid.YY - 60.0) ** 2) / 3000.0)
    phi0_3 = jnp.exp(-((grid.XX + 50.0) ** 2 + (grid.YY - 100.0) ** 2) / 5000.0)
    phi0 = jnp.stack([phi0_1, phi0_2, phi0_3], axis=0)

    # ---------------------------------------------------------------------------
    # 2. Setup Trajectory and Control Inputs
    # ---------------------------------------------------------------------------
    n_steps = 200
    dt = 1.0
    times = np.linspace(0.0, n_steps * dt, n_steps)
    times_traj = np.linspace(0.0, n_steps * dt, n_steps + 1)

    # Spiral control inputs
    thrusts = 120.0 + 30.0 * np.sin(times / 6.0)
    headings = times / 8.0  # steering in a curve

    # True trajectory position corrections dx_true
    # (represents a wave drift or time-varying kinematics bias)
    dx_true = 0.1 * np.column_stack(
        [np.sin(times_traj / 5.0), np.cos(times_traj / 8.0)]
    )
    # Keep initial correction close to zero
    dx_true = dx_true - dx_true[0]

    # Ground truth trajectory
    x0 = jnp.array([0.0, 0.0])
    coords_true = unicycle_corrected_trajectory_fn(
        x0, thrusts, headings, dt, k_thrust_true, dx_true
    )

    # ---------------------------------------------------------------------------
    # 3. Simulate PDE & Generate Noisy Observations
    # ---------------------------------------------------------------------------
    u_field_true = jnp.broadcast_to(v_flow_true, (3, grid.ny, grid.nx, 2))
    pde_params_true = PDEParams(u_field=u_field_true, D=D_true)

    t_traj = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(jnp.full(n_steps, dt))])

    # Batch solve PDE for the 3 fields
    solve_vmap = jax.vmap(
        lambda p0, params: solver.solve(
            p0, params, t0=0.0, t_end=t_traj[-1], saveat=t_traj
        )
    )
    snapshots_true = solve_vmap(phi0, pde_params_true)  # shape (3, T, ny, nx)

    # Sample scalar observations at a subset of timestamps
    obs_ts = jnp.array(times_traj[:])
    x_obs = jnp.interp(obs_ts, t_traj, coords_true[:, 0])
    y_obs = jnp.interp(obs_ts, t_traj, coords_true[:, 1])

    def interp_single_true(snapshots_single):
        interp = SpatiotemporalInterpolator(grid, t_traj, snapshots_single)
        return interp(x_obs, y_obs, obs_ts)

    obs_vals_clean = jax.vmap(interp_single_true)(snapshots_true).T  # shape (M, 3)

    # Add noise to scalar observations
    obs_vals = obs_vals_clean + np.random.normal(0.0, 0.01, size=obs_vals_clean.shape)

    # Noisy coordinates measurements (GPS-like)
    coords_noisy = coords_true

    # Dead Reckoning trajectory (initial guess with zero corrections)
    dx_init = jnp.zeros_like(dx_true)
    coords_guess = unicycle_corrected_trajectory_fn(
        x0, thrusts, headings, dt, k_thrust_true, dx_init
    )

    # ---------------------------------------------------------------------------
    # 4. Joint SLAM Optimization
    # ---------------------------------------------------------------------------
    D_init = jnp.array([1.0, 0.5, 1.5])  # noqa: N806
    v_flow_init = jnp.array([0.0, 0.0])

    init_params = {
        "D": D_init,
        "v_flow": v_flow_init,
        "dx": dx_init,
    }

    bounds = {
        "D": (0.01, 3.0),
        "v_flow": (-5.0, 5.0),
        "dx": (-15.0, 15.0),
    }

    optimizer = MultiPdeSlamOptimizer(grid, solver)

    # A. Fit using L-BFGS-B (SciPy)
    print("Fitting using L-BFGS-B...")
    import time

    t_start = time.perf_counter()
    best_params_lbfgs, info_lbfgs = optimizer.fit(
        phi0=phi0,
        obs_ts=obs_ts,
        obs_vals=obs_vals,
        thrusts=thrusts,
        headings=headings,
        dt=dt,
        init_params=init_params,
        bounds=bounds,
        lambda_reg=1e-2,
        k_thrust_fixed=k_thrust_true,
        method="l-bfgs-b",
        options={"maxiter": 100, "disp": True},
    )
    best_params_lbfgs["D"].block_until_ready()
    t_lbfgs = time.perf_counter() - t_start

    print("\nL-BFGS-B Optimization Results:")
    print(f"  Success       : {info_lbfgs['success']}")
    print(
        f"  True D        : {np.array(D_true)}\n"
        f"  Estimated D   : {np.array(best_params_lbfgs['D'])}"
    )
    print(
        f"  True v_flow   : {np.array(v_flow_true)}\n"
        f"  Estimated flow: {np.array(best_params_lbfgs['v_flow'])}"
    )
    print(f"  Final Loss    : {info_lbfgs['fun']:.6f}")
    print(f"  Iterations    : {info_lbfgs['nit']}\n")

    # Predict L-BFGS-B trajectory and scalar values
    coords_lbfgs = unicycle_corrected_trajectory_fn(
        x0, thrusts, headings, dt, k_thrust_true, best_params_lbfgs["dx"]
    )
    u_field_lbfgs = jnp.broadcast_to(
        best_params_lbfgs["v_flow"], (3, grid.ny, grid.nx, 2)
    )
    pde_params_lbfgs = PDEParams(u_field=u_field_lbfgs, D=best_params_lbfgs["D"])
    snapshots_lbfgs = solve_vmap(phi0, pde_params_lbfgs)

    def interp_single_lbfgs(snapshots_single):
        interp = SpatiotemporalInterpolator(grid, t_traj, snapshots_single)
        return interp(coords_lbfgs[:, 0], coords_lbfgs[:, 1], t_traj)

    vals_lbfgs = jax.vmap(interp_single_lbfgs)(snapshots_lbfgs).T

    # B. Fit using Adam (Optax)
    print("Fitting using Adam...")
    t_start = time.perf_counter()
    best_params_adam, info_adam = optimizer.fit(
        phi0=phi0,
        obs_ts=obs_ts,
        obs_vals=obs_vals,
        thrusts=thrusts,
        headings=headings,
        dt=dt,
        init_params=init_params,
        bounds=bounds,
        lambda_reg=1e-3,
        k_thrust_fixed=k_thrust_true,
        method="adam",
        options={"learning_rate": 0.05, "num_steps": 120},
    )
    best_params_adam["D"].block_until_ready()
    t_adam = time.perf_counter() - t_start

    print("\nAdam Optimization Results:")
    print(
        f"  True D        : {np.array(D_true)}\n  Estimated D   : {np.array(best_params_adam['D'])}"
    )
    print(
        f"  True v_flow   : {np.array(v_flow_true)}\n"
        f"  Estimated flow: {np.array(best_params_adam['v_flow'])}"
    )
    print(f"  Final Loss    : {info_adam['fun']:.6f}\n")

    coords_adam = unicycle_corrected_trajectory_fn(
        x0, thrusts, headings, dt, k_thrust_true, best_params_adam["dx"]
    )
    u_field_adam = jnp.broadcast_to(
        best_params_adam["v_flow"], (3, grid.ny, grid.nx, 2)
    )
    pde_params_adam = PDEParams(u_field=u_field_adam, D=best_params_adam["D"])
    snapshots_adam = solve_vmap(phi0, pde_params_adam)

    def interp_single_adam(snapshots_single):
        interp = SpatiotemporalInterpolator(grid, t_traj, snapshots_single)
        return interp(coords_adam[:, 0], coords_adam[:, 1], t_traj)

    vals_adam = jax.vmap(interp_single_adam)(snapshots_adam).T

    print("==================================================")
    print("Execution Time Comparison (including JIT compile):")
    print(f"  L-BFGS-B (SciPy)  : {t_lbfgs:.4f} s")
    print(f"  Adam (Optax)      : {t_adam:.4f} s")
    print("==================================================")

    # Predict Dead Reckoning scalar values
    u_field_guess = jnp.broadcast_to(v_flow_init, (3, grid.ny, grid.nx, 2))
    pde_params_guess = PDEParams(u_field=u_field_guess, D=D_init)
    snapshots_guess = solve_vmap(phi0, pde_params_guess)

    def interp_single_guess(snapshots_single):
        interp = SpatiotemporalInterpolator(grid, t_traj, snapshots_single)
        return interp(coords_guess[:, 0], coords_guess[:, 1], t_traj)

    vals_guess = jax.vmap(interp_single_guess)(snapshots_guess).T

    # True scalar values along true trajectory
    def interp_single_true_traj(snapshots_single):
        interp = SpatiotemporalInterpolator(grid, t_traj, snapshots_single)
        return interp(coords_true[:, 0], coords_true[:, 1], t_traj)

    vals_true = jax.vmap(interp_single_true_traj)(snapshots_true).T

    # ---------------------------------------------------------------------------
    # 5. Visualisation and Plotting
    # ---------------------------------------------------------------------------
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    idx_0 = 0
    idx_mid = n_steps // 2
    idx_last = n_steps

    fig, axes = plt.subplots(4, 3, figsize=(18, 22), constrained_layout=True)
    fig.suptitle(
        "Joint PDE Parameters & Trajectory Correction Optimization",
        fontsize=16,
        fontweight="bold",
    )

    # Panel 1: Trajectory Comparison
    ax1 = axes[0, 0]
    ax1.plot(
        coords_true[:, 0], coords_true[:, 1], "g-", linewidth=3.0, label="Ground Truth"
    )
    ax1.scatter(
        coords_noisy[:, 0],
        coords_noisy[:, 1],
        color="red",
        alpha=0.3,
        s=15,
        label="Noisy GPS",
    )
    ax1.plot(
        coords_guess[:, 0],
        coords_guess[:, 1],
        "r--",
        linewidth=1.5,
        label="Dead Reckoning (Uncorrected)",
    )
    ax1.plot(
        coords_lbfgs[:, 0],
        coords_lbfgs[:, 1],
        "b-.",
        linewidth=2.0,
        label="Optimized (L-BFGS-B)",
    )
    ax1.plot(
        coords_adam[:, 0],
        coords_adam[:, 1],
        "m:",
        linewidth=2.0,
        label="Optimized (Adam)",
    )
    ax1.scatter([0.0], [0.0], color="black", marker="o", s=80, zorder=5, label="Start")
    ax1.set_title("Trajectory Comparison")
    ax1.set_xlabel("East position [m]")
    ax1.set_ylabel("North position [m]")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()
    ax1.set_aspect("equal")

    # Panel 2: Scalar Prediction Comparison
    ax2 = axes[0, 1]
    colors = ["g", "m", "c"]
    for i in range(3):
        ax2.plot(
            t_traj,
            vals_true[:, i],
            color=colors[i],
            linestyle="-",
            linewidth=2.0,
            label=f"GT Field {i + 1}" if i == 0 else "",
        )
        ax2.scatter(
            obs_ts,
            obs_vals[:, i],
            color=colors[i],
            alpha=0.3,
            s=15,
            label="Obs" if i == 0 else "",
        )
        ax2.plot(
            t_traj,
            vals_guess[:, i],
            color=colors[i],
            linestyle="--",
            linewidth=1.0,
            label="Dead Reckon" if i == 0 else "",
        )
        ax2.plot(
            t_traj,
            vals_lbfgs[:, i],
            color=colors[i],
            linestyle="-.",
            linewidth=1.5,
            label="L-BFGS-B" if i == 0 else "",
        )
        ax2.plot(
            t_traj,
            vals_adam[:, i],
            color=colors[i],
            linestyle=":",
            linewidth=1.5,
            label="Adam" if i == 0 else "",
        )
    ax2.set_title("Scalar Field Value along Trajectory")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Field value (e.g. salinity)")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    # Panel 3: Adam Training Loss
    ax3 = axes[0, 2]
    loss_history = info_adam["loss_history"]
    ax3.plot(loss_history, "m-", linewidth=2.0, label="Adam Loss")
    ax3.axhline(
        info_lbfgs["fun"], color="blue", linestyle="--", label="L-BFGS-B Final Loss"
    )
    ax3.set_title("Optimization Loss Curve")
    ax3.set_xlabel("Iteration / Steps")
    ax3.set_ylabel("Joint MSE Loss")
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.set_yscale("log")
    ax3.legend()

    # 2D Field comparisons row-by-row
    ext = [grid.x_min, grid.x_max, grid.y_min, grid.y_max]

    timestamps = [
        ("t = 0 s", idx_0),
        (f"t = {int(times_traj[idx_mid])} s (t_last//2)", idx_mid),
        (f"t = {int(times_traj[idx_last])} s (t_last)", idx_last),
    ]

    cols = [
        ("Ground Truth", snapshots_true, coords_true, "g-"),
        ("L-BFGS-B Estimated", snapshots_lbfgs, coords_lbfgs, "b-."),
        ("Adam Estimated", snapshots_adam, coords_adam, "m:"),
    ]

    for row_idx, (t_label, t_idx) in enumerate(timestamps, start=1):
        for col_idx, (name, snapshots, coords, style) in enumerate(cols):
            ax = axes[row_idx, col_idx]

            # Stack the 3 fields as RGB channels (Red=P1, Green=P2, Blue=P3)
            rgb_field = jnp.clip(
                jnp.stack(
                    [snapshots[0, t_idx], snapshots[1, t_idx], snapshots[2, t_idx]],
                    axis=-1,
                ),
                0.0,
                1.0,
            )

            ax.imshow(np.array(rgb_field), origin="lower", extent=ext)

            # Plot trajectory up to current time step
            ax.plot(
                coords[: t_idx + 1, 0], coords[: t_idx + 1, 1], style, linewidth=1.5
            )
            # Draw current position
            ax.scatter(
                coords[t_idx, 0],
                coords[t_idx, 1],
                color="red",
                marker="*",
                s=100,
                zorder=5,
            )

            # Labeling
            if row_idx == 1:
                ax.set_title(
                    f"{name}\n{t_label}\n(Red=P1, Green=P2, Blue=P3)",
                    fontsize=10,
                    fontweight="bold",
                )
            else:
                ax.set_title(t_label, fontsize=12)

            ax.set_xlabel("East position [m]")
            ax.set_ylabel("North position [m]")
            ax.grid(True, linestyle=":", alpha=0.5)
            ax.set_aspect("equal")

    out_path = output_dir / "demo_joint_optimization.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved visualization → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
