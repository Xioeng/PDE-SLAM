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

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import Array

from pde_slam.interpolator import SpatialGrid
from pde_slam.interpolators.spatiotemporal import SpatiotemporalInterpolator
from pde_slam.joint_optimization import (
    JointSlamOptimizer,
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
    D_true = 0.6
    v_flow_true = jnp.array([0.5, -0.3])  # constant flow (East, North)
    k_thrust_true = 1.0

    # Initial scalar field: Gaussian plume centered at (10.0, -10.0)
    phi0 = jnp.exp(-((grid.XX - 10.0) ** 2 + (grid.YY + 10.0) ** 2) / 800.0)

    # ---------------------------------------------------------------------------
    # 2. Setup Trajectory and Control Inputs
    # ---------------------------------------------------------------------------
    n_steps = 60
    dt = 1.0
    times = np.linspace(0.0, n_steps * dt, n_steps)
    times_traj = np.linspace(0.0, n_steps * dt, n_steps + 1)

    # Spiral control inputs
    thrusts = 1.2 + 0.3 * np.sin(times / 6.0)
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
    u_field_true = jnp.broadcast_to(v_flow_true, (grid.ny, grid.nx, 2))
    pde_params_true = PDEParams(u_field=u_field_true, D=jnp.array(D_true))

    t_traj = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(jnp.full(n_steps, dt))])
    snapshots_true = solver.solve(
        phi0, pde_params_true, t0=0.0, t_end=t_traj[-1], saveat=t_traj
    )

    # Create spatiotemporal interpolator to sample measurements along trajectory
    interp_true = SpatiotemporalInterpolator(grid, t_traj, snapshots_true)

    # Sample scalar observations at a subset of timestamps
    obs_ts = jnp.array(times_traj[:])  # every 2 seconds
    x_obs = jnp.interp(obs_ts, t_traj, coords_true[:, 0])
    y_obs = jnp.interp(obs_ts, t_traj, coords_true[:, 1])
    obs_vals_clean = interp_true(x_obs, y_obs, obs_ts)

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
    D_init = 1.2
    v_flow_init = jnp.array([0.0, 0.0])

    init_params = {
        "D": D_init,
        "v_flow": v_flow_init,
        "dx": dx_init,
    }

    bounds = {
        "D": (0.01, 3.0),
        "v_flow": (-2.0, 2.0),
        "dx": (-15.0, 15.0),
    }

    optimizer = JointSlamOptimizer(grid, solver)

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
        lambda_reg=1e-5,
        k_thrust_fixed=k_thrust_true,
        method="l-bfgs-b",
        options={"maxiter": 100, "disp": True},
    )
    best_params_lbfgs["D"].block_until_ready()
    t_lbfgs = time.perf_counter() - t_start

    print("\nL-BFGS-B Optimization Results:")
    print(f"  Success       : {info_lbfgs['success']}")
    print(
        f"  True D        : {D_true:.4f}  | Estimated: {float(best_params_lbfgs['D']):.4f}"
    )
    print(
        f"  True v_flow   : {np.array(v_flow_true)} | Estimated: {np.array(best_params_lbfgs['v_flow'])}"
    )
    print(f"  Final Loss    : {info_lbfgs['fun']:.6f}")
    print(f"  Iterations    : {info_lbfgs['nit']}\n")

    # Predict L-BFGS-B trajectory and scalar values
    coords_lbfgs = unicycle_corrected_trajectory_fn(
        x0, thrusts, headings, dt, k_thrust_true, best_params_lbfgs["dx"]
    )
    u_field_lbfgs = jnp.broadcast_to(best_params_lbfgs["v_flow"], (grid.ny, grid.nx, 2))
    pde_params_lbfgs = PDEParams(u_field=u_field_lbfgs, D=best_params_lbfgs["D"])
    snapshots_lbfgs = solver.solve(
        phi0, pde_params_lbfgs, t0=0.0, t_end=t_traj[-1], saveat=t_traj
    )
    interp_lbfgs = SpatiotemporalInterpolator(grid, t_traj, snapshots_lbfgs)
    vals_lbfgs = interp_lbfgs(coords_lbfgs[:, 0], coords_lbfgs[:, 1], t_traj)

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
        f"  True D        : {D_true:.4f}  | Estimated: {float(best_params_adam['D']):.4f}"
    )
    print(
        f"  True v_flow   : {np.array(v_flow_true)} | Estimated: {np.array(best_params_adam['v_flow'])}"
    )
    print(f"  Final Loss    : {info_adam['fun']:.6f}\n")

    coords_adam = unicycle_corrected_trajectory_fn(
        x0, thrusts, headings, dt, k_thrust_true, best_params_adam["dx"]
    )
    u_field_adam = jnp.broadcast_to(best_params_adam["v_flow"], (grid.ny, grid.nx, 2))
    pde_params_adam = PDEParams(u_field=u_field_adam, D=best_params_adam["D"])
    snapshots_adam = solver.solve(
        phi0, pde_params_adam, t0=0.0, t_end=t_traj[-1], saveat=t_traj
    )
    interp_adam = SpatiotemporalInterpolator(grid, t_traj, snapshots_adam)
    vals_adam = interp_adam(coords_adam[:, 0], coords_adam[:, 1], t_traj)

    # C. Fit using BFGS (JAX-SciPy)
    print("Fitting using JAX-SciPy (BFGS)...")
    t_start = time.perf_counter()
    best_params_jax, info_jax = optimizer.fit(
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
        method="bfgs",
        options={"maxiter": 100},
    )
    best_params_jax["D"].block_until_ready()
    t_jax = time.perf_counter() - t_start

    print("\nJAX-SciPy BFGS Optimization Results:")
    print(f"  Success       : {info_jax['success']}")
    print(
        f"  True D        : {D_true:.4f}  | Estimated: {float(best_params_jax['D']):.4f}"
    )
    print(
        f"  True v_flow   : {np.array(v_flow_true)} | Estimated: {np.array(best_params_jax['v_flow'])}"
    )
    print(f"  Final Loss    : {info_jax['fun']:.6f}")
    print(f"  Iterations    : {info_jax['nit']}\n")

    # Predict JAX-SciPy trajectory and scalar values
    coords_jax = unicycle_corrected_trajectory_fn(
        x0, thrusts, headings, dt, k_thrust_true, best_params_jax["dx"]
    )
    u_field_jax = jnp.broadcast_to(best_params_jax["v_flow"], (grid.ny, grid.nx, 2))
    pde_params_jax = PDEParams(u_field=u_field_jax, D=best_params_jax["D"])
    snapshots_jax = solver.solve(
        phi0, pde_params_jax, t0=0.0, t_end=t_traj[-1], saveat=t_traj
    )
    interp_jax = SpatiotemporalInterpolator(grid, t_traj, snapshots_jax)
    vals_jax = interp_jax(coords_jax[:, 0], coords_jax[:, 1], t_traj)

    print("==================================================")
    print("Execution Time Comparison (including JIT compile):")
    print(f"  L-BFGS-B (SciPy)  : {t_lbfgs:.4f} s")
    print(f"  Adam (Optax)      : {t_adam:.4f} s")
    print(f"  BFGS (JAX-SciPy)  : {t_jax:.4f} s")
    print("==================================================")

    # Predict Dead Reckoning scalar values
    u_field_guess = jnp.broadcast_to(v_flow_init, (grid.ny, grid.nx, 2))
    pde_params_guess = PDEParams(u_field=u_field_guess, D=jnp.array(D_init))
    snapshots_guess = solver.solve(
        phi0, pde_params_guess, t0=0.0, t_end=t_traj[-1], saveat=t_traj
    )
    interp_guess = SpatiotemporalInterpolator(grid, t_traj, snapshots_guess)
    vals_guess = interp_guess(coords_guess[:, 0], coords_guess[:, 1], t_traj)

    # True scalar values along true trajectory
    vals_true = interp_true(coords_true[:, 0], coords_true[:, 1], t_traj)

    # ---------------------------------------------------------------------------
    # 5. Visualisation and Plotting
    # ---------------------------------------------------------------------------
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)
    fig.suptitle(
        "Joint PDE Parameters & Trajectory Correction Optimization",
        fontsize=16,
        fontweight="bold",
    )

    # Panel 1: Trajectory Comparison
    ax1 = axes[0]
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
    ax1.plot(
        coords_jax[:, 0],
        coords_jax[:, 1],
        "c--",
        linewidth=1.5,
        label="Optimized (JAX-SciPy BFGS)",
    )
    ax1.scatter([0.0], [0.0], color="black", marker="o", s=80, zorder=5, label="Start")
    ax1.set_title("Trajectory Comparison")
    ax1.set_xlabel("East position [m]")
    ax1.set_ylabel("North position [m]")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()
    ax1.set_aspect("equal")

    # Panel 2: Scalar Prediction Comparison
    ax2 = axes[1]
    ax2.plot(t_traj, vals_true, "g-", linewidth=3.0, label="Ground Truth")
    ax2.scatter(
        obs_ts, obs_vals, color="red", alpha=0.5, s=25, label="Scalar Observations"
    )
    ax2.plot(t_traj, vals_guess, "r--", linewidth=1.5, label="Dead Reckoning Pred")
    ax2.plot(t_traj, vals_lbfgs, "b-.", linewidth=2.0, label="L-BFGS-B Pred")
    ax2.plot(t_traj, vals_adam, "m:", linewidth=2.0, label="Adam Pred")
    ax2.plot(t_traj, vals_jax, "c--", linewidth=1.5, label="JAX-SciPy Pred")
    ax2.set_title("Scalar Field Value along Trajectory")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Field value (e.g. salinity)")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    # Panel 3: Adam Training Loss
    ax3 = axes[2]
    loss_history = info_adam["loss_history"]
    ax3.plot(loss_history, "m-", linewidth=2.0, label="Adam Loss")
    ax3.axhline(
        info_lbfgs["fun"], color="blue", linestyle="--", label="L-BFGS-B Final Loss"
    )
    ax3.axhline(
        info_jax["fun"], color="cyan", linestyle=":", label="JAX-SciPy Final Loss"
    )
    ax3.set_title("Optimization Loss Curve")
    ax3.set_xlabel("Iteration / Steps")
    ax3.set_ylabel("Joint MSE Loss")
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.set_yscale("log")
    ax3.legend()

    out_path = output_dir / "demo_joint_optimization.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved visualization → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
