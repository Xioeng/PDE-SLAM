"""
test_scripts/demo_optimization.py
==================================
Demonstrates how to use the KinematicsOptimizer to identify the parameters
(e.g., k_thrust) of a kinematic model from noisy trajectory observations.

It compares:
1. Ground truth trajectory (known k_thrust)
2. Noisy coordinate measurements
3. Initial guess trajectory (wrong k_thrust)
4. Optimized trajectory using L-BFGS-B (SciPy)
5. Optimized trajectory using Adam (Optax)

Run::

    uv run python test_scripts/demo_optimization.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from jax import Array

from pde_slam.kinematics import UnicycleKinematics
from pde_slam.optimization import KinematicsOptimizer


def main() -> None:
    # ---------------------------------------------------------------------------
    # 1. Setup Ground Truth & Controls
    # ---------------------------------------------------------------------------
    k_thrust_true = 1.75
    dt = 0.5  # time step [s]
    t_max = 60.0  # total time [s]
    n_steps = int(t_max / dt)

    # Robot starting position & heading
    x0, y0 = 0.0, 0.0
    heading0 = 0.0

    robot = UnicycleKinematics(k_thrust=k_thrust_true, x0=x0, y0=y0, heading0=heading0)

    def unicycle_trajectory_fn(
        x0: Array,
        thrusts: Array,
        headings: Array,
        dts: float,
        params: dict[str, float],
    ) -> Array:
        return robot.integrate_trajectory(
            x0, thrusts, headings, dts, params["k_thrust"], include_initial=True
        )

    # Create smooth time-varying control inputs
    times = np.linspace(0.0, t_max, n_steps)
    thrusts = 50.0 + 30.0 * np.sin(2 * np.pi * times / 20.0)
    headings = 0.5 * np.cos(2 * np.pi * times / 30.0)

    # Generate ground truth trajectory
    states_true = robot.trajectory(thrusts, headings, dt=dt, include_initial=True)
    coords_true = states_true[:, :2]

    # Add Gaussian noise to coordinates to simulate measurement noise (e.g. GPS noise)
    np.random.seed(42)
    noise_std = 0.25  # [meters]
    coords_noisy = coords_true + np.random.normal(0.0, noise_std, size=coords_true.shape)

    # ---------------------------------------------------------------------------
    # 2. Setup initial guess & baseline trajectory
    # ---------------------------------------------------------------------------
    k_thrust_init = 4.0  # Wrong k_thrust parameter
    robot_guess = UnicycleKinematics(k_thrust=k_thrust_init, x0=x0, y0=y0, heading0=heading0)
    states_guess = robot_guess.trajectory(thrusts, headings, dt=dt, include_initial=True)
    coords_guess = states_guess[:, :2]

    # ---------------------------------------------------------------------------
    # 3. Fit parameters using KinematicsOptimizer (L-BFGS-B)
    # ---------------------------------------------------------------------------
    print("Fitting k_thrust using L-BFGS-B (SciPy + JAX gradients)...")
    optimizer = KinematicsOptimizer(trajectory_fn=unicycle_trajectory_fn)
    init_params = {"k_thrust": k_thrust_init}
    bounds = {"k_thrust": (0.1, 5.0)}

    best_params_lbfgs, info_lbfgs = optimizer.fit(
        coords_obs=coords_noisy,
        thrusts=thrusts,
        headings=headings,
        dt=dt,
        init_params=init_params,
        bounds=bounds,
        method="l-bfgs-b",
    )

    print("L-BFGS-B Results:")
    print(f"  Success      : {info_lbfgs['success']}")
    print(f"  True value   : {k_thrust_true:.4f}")
    print(f"  Estimated    : {best_params_lbfgs['k_thrust']:.4f}")
    print(f"  Final Loss   : {info_lbfgs['fun']:.6f}")
    print(f"  Iterations   : {info_lbfgs['nit']}")
    print("-" * 50)

    # Generate trajectory with L-BFGS-B optimized parameter
    robot_lbfgs = UnicycleKinematics(
        k_thrust=best_params_lbfgs["k_thrust"], x0=x0, y0=y0, heading0=heading0
    )
    states_lbfgs = robot_lbfgs.trajectory(thrusts, headings, dt=dt, include_initial=True)
    coords_lbfgs = states_lbfgs[:, :2]

    # ---------------------------------------------------------------------------
    # 4. Fit parameters using KinematicsOptimizer (Adam)
    # ---------------------------------------------------------------------------
    print("Fitting k_thrust using Adam (Optax + JAX gradients)...")
    best_params_adam, info_adam = optimizer.fit(
        coords_obs=coords_noisy,
        thrusts=thrusts,
        headings=headings,
        dt=dt,
        init_params=init_params,
        method="adam",
        options={"learning_rate": 0.05, "num_steps": 100},
    )

    print("Adam Results:")
    print(f"  True value   : {k_thrust_true:.4f}")
    print(f"  Estimated    : {best_params_adam['k_thrust']:.4f}")
    print(f"  Final Loss   : {info_adam['fun']:.6f}")
    print("-" * 50)

    # Generate trajectory with Adam optimized parameter
    robot_adam = UnicycleKinematics(
        k_thrust=best_params_adam["k_thrust"], x0=x0, y0=y0, heading0=heading0
    )
    states_adam = robot_adam.trajectory(thrusts, headings, dt=dt, include_initial=True)
    coords_adam = states_adam[:, :2]

    # ---------------------------------------------------------------------------
    # 5. Visualization & Plotting
    # ---------------------------------------------------------------------------
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    fig.suptitle(
        "Kinematics Parameter Optimization Demonstration",
        fontsize=14,
        fontweight="bold",
    )

    # Panel 1: Trajectory Comparison
    ax1 = axes[0]
    ax1.plot(coords_true[:, 0], coords_true[:, 1], "g-", linewidth=3.0, label="Ground Truth")
    ax1.scatter(
        coords_noisy[:, 0],
        coords_noisy[:, 1],
        color="red",
        alpha=0.3,
        s=15,
        label="Noisy Measurements",
    )
    ax1.plot(
        coords_guess[:, 0],
        coords_guess[:, 1],
        "r--",
        linewidth=1.5,
        label="Initial Guess",
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

    ax1.scatter([x0], [y0], color="black", marker="o", s=80, zorder=5, label="Start")
    ax1.set_title("Trajectory Comparison")
    ax1.set_xlabel("East position [m]")
    ax1.set_ylabel("North position [m]")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()
    ax1.set_aspect("equal")

    # Panel 2: Adam Training Loss History
    ax2 = axes[1]
    loss_history = info_adam["loss_history"]
    ax2.plot(loss_history, "m-", linewidth=2.0, label="Adam Loss")
    # Draw a horizontal line indicating final L-BFGS-B loss
    ax2.axhline(
        info_lbfgs["fun"],
        color="blue",
        linestyle="--",
        label="L-BFGS-B Final Loss",
    )
    ax2.set_title("Optimization Loss Curve")
    ax2.set_xlabel("Iteration / Steps")
    ax2.set_ylabel("Mean Squared Error (MSE)")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.set_yscale("log")
    ax2.legend()

    out_path = output_dir / "demo_optimization.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved visualization → {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
