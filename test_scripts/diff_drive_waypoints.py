"""
test_scripts/diff_drive_waypoints.py
====================================
Test waypoint-following path generation and control signal dynamics
(linear velocity v and angular velocity omega) using DiffDriveKinematics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path so pde_slam is importable without package installation
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from pde_slam.kinematics import DiffDriveKinematics
from pde_slam.viz import pick_waypoints_gui


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demo differential drive waypoint path generation."
    )
    parser.add_argument(
        "--speed-mps",
        type=float,
        default=2.0,
        help="Maximum linear speed [m/s] (default: 2.0)",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.5,
        help="Time step [s] (default: 0.5)",
    )
    parser.add_argument(
        "--acceptance-radius",
        type=float,
        default=3.0,
        help="Waypoint acceptance radius [m] (default: 3.0)",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Open interactive GUI to click and select waypoints on 2D map",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display plot windows interactively",
    )
    return parser.parse_args(args)


def main(args_list: list[str] | None = None) -> None:
    args = parse_args(args_list)

    print("==================================================")
    print("Testing Differential Drive Waypoint Path Generation")
    print("==================================================")
    print(f"Max Speed          : {args.speed_mps:.2f} m/s")
    print(f"Time Step (dt)     : {args.dt:.2f} s")
    print(f"Acceptance Radius  : {args.acceptance_radius:.2f} m")

    default_waypoints = np.array(
        [
            [0.0, 0.0],
            [30.0, 40.0],
            [70.0, 40.0],
            [70.0, -20.0],
            [30.0, -20.0],
            [10.0, -60.0],
            [80.0, -60.0],
        ]
    )

    if args.gui or not args.no_show:
        poly_bounds = np.array(
            [[-20.0, -80.0], [100.0, -80.0], [100.0, 60.0], [-20.0, 60.0]]
        )
        picked = pick_waypoints_gui(polygon_enu=poly_bounds)
        waypoints = picked if len(picked) >= 2 else default_waypoints

    else:
        waypoints = default_waypoints

    print(f"Waypoints Count    : {len(waypoints)}")
    for idx, wp in enumerate(waypoints):
        print(f"  WP {idx + 1}: East = {wp[0]:6.1f} m, North = {wp[1]:6.1f} m")

    # Instantiate robot kinematics
    robot = DiffDriveKinematics(
        x0=waypoints[0, 0],
        y0=waypoints[0, 1],
        heading0=0.0,
    )

    # Generate control signals from waypoint follower
    _, velocities, omegas = robot.drive_to_waypoints(
        waypoints,
        speed_mps=args.speed_mps,
        dt=args.dt,
        acceptance_radius=args.acceptance_radius,
    )

    # Obtain trajectory states by batch integration of control signals
    x0 = np.array([waypoints[0, 0], waypoints[0, 1], 0.0])
    states = np.array(
        DiffDriveKinematics.integrate_trajectory(
            x0, velocities, omegas, args.dt, include_initial=True
        )
    )

    n_steps = len(velocities)
    times = np.arange(n_steps + 1) * args.dt

    print(f"\nGenerated Path Steps: {n_steps} steps (Total Time: {times[-1]:.1f} s)")
    print(f"Trajectory Length   : {len(states)} poses")

    # Create diagnostic plots
    fig = plt.figure(figsize=(16, 7))
    fig.suptitle(
        "Differential Drive Waypoint Path & Control Signal Analysis\n"
        f"v_max = {args.speed_mps} m/s | dt = {args.dt}s | "
        f"acceptance radius = {args.acceptance_radius}m",
        fontsize=13,
        fontweight="bold",
    )

    # Subplot 1: 2D Trajectory Map
    ax_map = fig.add_subplot(1, 2, 1)
    ax_map.set_facecolor("#f8f9fa")

    # Draw waypoint acceptance circles
    for idx, wp in enumerate(waypoints):
        circle = plt.Circle(
            wp,
            args.acceptance_radius,
            color="orange",
            fill=True,
            alpha=0.15,
            linestyle="--",
        )
        ax_map.add_patch(circle)
        ax_map.text(
            wp[0] + 1.5,
            wp[1] + 1.5,
            f"WP {idx + 1}",
            fontsize=9,
            fontweight="bold",
            color="navy",
        )

    # Plot waypoints connectivity line
    ax_map.plot(
        waypoints[:, 0],
        waypoints[:, 1],
        "o--",
        color="darkorange",
        linewidth=1.5,
        markersize=6,
        label="Target Waypoints Path",
    )

    # Plot actual integrated robot path
    ax_map.plot(
        states[:, 0],
        states[:, 1],
        "b-",
        linewidth=2.2,
        label="Robot Path (Integrated)",
    )

    # Draw orientation heading arrows at sub-sampled steps
    arrow_subsample = max(1, n_steps // 20)
    for i in range(0, n_steps, arrow_subsample):
        x, y, heading = states[i, 0], states[i, 1], states[i, 2]
        dx_arrow = 3.0 * np.cos(heading)
        dy_arrow = 3.0 * np.sin(heading)
        ax_map.arrow(
            x,
            y,
            dx_arrow,
            dy_arrow,
            head_width=1.5,
            head_length=2.0,
            fc="red",
            ec="darkred",
            alpha=0.7,
            zorder=6,
        )

    # Mark Start and End
    ax_map.plot(states[0, 0], states[0, 1], "go", markersize=9, label="Start")
    ax_map.plot(states[-1, 0], states[-1, 1], "ro", markersize=9, label="End Position")

    ax_map.set_title(
        "2D Spatial Trajectory & Waypoint Tracking", fontsize=11, fontweight="bold"
    )
    ax_map.set_xlabel("East Position [m]", fontsize=10)
    ax_map.set_ylabel("North Position [m]", fontsize=10)
    ax_map.grid(True, linestyle=":", alpha=0.6)
    ax_map.legend(loc="best", fontsize=9)
    ax_map.set_aspect("equal", adjustable="datalim")

    # Subplot 2: Control Signals over Time (v and omega)
    ax_v = fig.add_subplot(2, 2, 2)
    ax_omega = fig.add_subplot(2, 2, 4)

    # Linear Velocity Plot
    ax_v.set_facecolor("#f8f9fa")
    ax_v.plot(times[:-1], velocities, "b-", linewidth=1.8, label="Linear Velocity v(t)")
    ax_v.axhline(
        args.speed_mps,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=f"Max Speed Cap ({args.speed_mps} m/s)",
    )
    ax_v.set_title(
        "Linear Velocity v(t) [Proportional to Distance, Capped]",
        fontsize=11,
        fontweight="bold",
    )
    ax_v.set_ylabel("Velocity [m/s]", fontsize=10)
    ax_v.grid(True, linestyle=":", alpha=0.6)
    ax_v.legend(loc="lower left", fontsize=8.5)

    # Angular Velocity Plot
    ax_omega.set_facecolor("#f8f9fa")
    ax_omega.plot(
        times[:-1], omegas, "m-", linewidth=1.8, label=r"Angular Velocity $\omega(t)$"
    )
    ax_omega.axhline(0.0, color="gray", linestyle=":", linewidth=1.0)
    ax_omega.set_title(
        r"Angular Velocity $\omega(t)$ [Smooth Heading Control]",
        fontsize=11,
        fontweight="bold",
    )
    ax_omega.set_xlabel("Time [s]", fontsize=10)
    ax_omega.set_ylabel("Angular Speed [rad/s]", fontsize=10)
    ax_omega.grid(True, linestyle=":", alpha=0.6)
    ax_omega.legend(loc="lower left", fontsize=8.5)

    plt.tight_layout()

    output_dir = Path("output/graphs")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "demo_diff_drive_waypoints.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved diagnostic figure to: {out_path}")

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1:])
