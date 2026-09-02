"""
test_dr_covariance_ellipses.py
==============================
Visual verification test script for DeadReckoningEstimator.
Simulates online robot motion with controls arriving one-by-one,
propagates dead reckoning covariance in real time, and renders the
spatial trajectory alongside 2-sigma and 3-sigma uncertainty ellipses
and largest eigenvalue (semi-major axis) evolution.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

# Add project root to sys.path so pde_slam is importable without package installation
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pde_slam.kinematics import DeadReckoningEstimator, DiffDriveKinematics
from pde_slam.viz.style import PLOT_RC_PARAMS


def get_covariance_ellipse_params(
    sigma_2d: np.ndarray, n_std: float = 2.0
) -> tuple[float, float, float]:
    """Compute width, height, and orientation angle of covariance ellipse.

    Parameters
    ----------
    sigma_2d : np.ndarray
        2x2 spatial covariance matrix.
    n_std : float, default=2.0
        Number of standard deviations (e.g. 2.0 for ~95% confidence).

    Returns
    -------
    width : float
        Full ellipse diameter along major axis (2 * n_std * sqrt(lambda_1)).
    height : float
        Full ellipse diameter along minor axis (2 * n_std * sqrt(lambda_2)).
    angle_deg : float
        Rotation angle of major axis in degrees.
    """
    eigvals, eigvecs = np.linalg.eigh(sigma_2d)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    lambda_1 = max(0.0, float(eigvals[0]))
    lambda_2 = max(0.0, float(eigvals[1]))

    width = 2.0 * n_std * np.sqrt(lambda_1)
    height = 2.0 * n_std * np.sqrt(lambda_2)

    angle_rad = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
    angle_deg = float(np.degrees(angle_rad))

    return width, height, angle_deg


def run_dr_ellipse_simulation(
    v_sigma: float = 0.02,
    w_sigma: float = 0.001,
    threshold_std: float = 1.0,
    dt: float = 1.0,
    save_path: Path | None = None,
    show_plot: bool = True,
) -> None:
    """Simulate online dead reckoning and plot covariance ellipses.

    Parameters
    ----------
    v_sigma : float
        Linear velocity noise standard deviation [m/s].
    w_sigma : float
        Angular velocity noise standard deviation [rad/s].
    threshold_std : float
        Uncertainty threshold on sqrt(lambda_max) [m].
    dt : float
        Time step [s].
    save_path : Path or None
        Path to save output figure.
    show_plot : bool
        Whether to call plt.show().
    """
    waypoints = jnp.array(
        [
            [0.0, 0.0],
            [25.0, 5.0],
            [45.0, 25.0],
            [55.0, 50.0],
            [35.0, 75.0],
            [10.0, 85.0],
            [-15.0, 70.0],
            [-25.0, 45.0],
        ]
    )

    controller = DiffDriveKinematics(x0=0.0, y0=0.0, heading0=0.0)
    _, v_cmds, w_cmds = controller.drive_to_waypoints(
        waypoints=waypoints[1:],
        speed_mps=1.8,
        dt=dt,
        acceptance_radius=1.5,
    )

    n_steps = len(v_cmds)
    times = np.arange(n_steps + 1) * dt

    np.random.seed(42)
    v_noise = np.random.normal(0.0, v_sigma, size=n_steps)
    w_noise = np.random.normal(0.0, w_sigma, size=n_steps)
    v_act = np.clip(np.array(v_cmds) + v_noise, 0.0, None)
    w_act = np.array(w_cmds) + w_noise

    gt_robot = DiffDriveKinematics(x0=0.0, y0=0.0, heading0=0.0)
    gt_poses = [gt_robot.state[:2].copy()]
    for i in range(n_steps):
        gt_robot.step(float(v_act[i]), float(w_act[i]), dt)
        gt_poses.append(gt_robot.state[:2].copy())
    gt_poses = np.array(gt_poses)

    q_u = jnp.diag(jnp.array([v_sigma**2, w_sigma**2]))
    estimator = DeadReckoningEstimator(x0=0.0, y0=0.0, heading0=0.0, q_u=q_u)

    dr_poses = [estimator.state.copy()]
    dr_covs = [estimator.covariance.copy()]
    max_stds = [estimator.max_std]
    pos_stds = [estimator.position_std]
    switch_step: int | None = None

    for i in range(n_steps):
        v_cmd = float(v_cmds[i])
        w_cmd = float(w_cmds[i])

        estimator.step(v=v_cmd, omega=w_cmd, dt=dt)

        dr_poses.append(estimator.state.copy())
        dr_covs.append(estimator.covariance.copy())
        curr_max_std = estimator.max_std
        max_stds.append(curr_max_std)
        pos_stds.append(estimator.position_std)

        if switch_step is None and curr_max_std >= threshold_std:
            switch_step = i + 1

    dr_poses = np.array(dr_poses)
    max_stds = np.array(max_stds)
    pos_stds = np.array(pos_stds)

    print("==================================================")
    print("Dead Reckoning Uncertainty Ellipse Verification")
    print("==================================================")
    print(f"Total Steps: {n_steps} ({times[-1]:.1f} s)")
    print(f"Noise: v_sigma = {v_sigma:.3f} m/s, w_sigma = {w_sigma:.3f} rad/s")
    print(f"Threshold: sqrt(lambda_max) = {threshold_std:.2f} m")
    if switch_step is not None:
        t_sw = times[switch_step]
        print(
            f"--> Switching threshold reached at step {switch_step}/{n_steps} "
            f"(t = {t_sw:.1f} s)"
        )
        print(
            f"    DR Pose at switch: x = {dr_poses[switch_step, 0]:.2f} m, "
            f"y = {dr_poses[switch_step, 1]:.2f} m"
        )
        print(f"    max_std at switch: {max_stds[switch_step]:.3f} m")
    else:
        print("--> Threshold was not crossed during this trajectory.")

    plt.rcParams.update(PLOT_RC_PARAMS)
    fig, (ax_map, ax_var) = plt.subplots(
        1, 2, figsize=(10.5, 4.8), gridspec_kw={"width_ratios": [1.4, 1.0]}
    )

    ax_map.set_title(
        "Dead Reckoning Trajectory & Uncertainty Ellipses ($2\\sigma$)",
        fontweight="bold",
    )
    ax_map.plot(
        gt_poses[:, 0], gt_poses[:, 1], "k-", linewidth=1.5, label="Ground Truth Path"
    )
    ax_map.plot(
        dr_poses[:, 0],
        dr_poses[:, 1],
        "#0088cc",
        linestyle="--",
        linewidth=1.4,
        label="Dead Reckoning Estimate",
    )
    ax_map.scatter(
        waypoints[:, 0],
        waypoints[:, 1],
        c="#ff7700",
        marker="x",
        s=40,
        zorder=5,
        label="Target Waypoints",
    )

    ellipse_interval = max(1, n_steps // 16)
    for k in range(0, n_steps + 1, ellipse_interval):
        pos_cov = np.array(dr_covs[k][:2, :2])
        w, h, angle = get_covariance_ellipse_params(pos_cov, n_std=2.0)
        x_k, y_k = float(dr_poses[k, 0]), float(dr_poses[k, 1])

        is_pre_switch = switch_step is None or k <= switch_step
        edge_c = "#00aa44" if is_pre_switch else "#cc0000"
        fill_c = "#00ee66" if is_pre_switch else "#ff5555"

        ell = Ellipse(
            xy=(x_k, y_k),
            width=w,
            height=h,
            angle=angle,
            edgecolor=edge_c,
            facecolor=fill_c,
            alpha=0.22,
            linewidth=1.0,
            linestyle="-",
            zorder=3,
        )
        ax_map.add_patch(ell)

    if switch_step is not None:
        x_sw, y_sw = float(dr_poses[switch_step, 0]), float(dr_poses[switch_step, 1])
        ax_map.scatter(
            [x_sw],
            [y_sw],
            c="#cc0000",
            s=90,
            marker="*",
            zorder=6,
            label=f"Switch Threshold ($t={times[switch_step]:.1f}$s)",
        )

    ax_map.set_xlabel("East Position $x$ [m]")
    ax_map.set_ylabel("North Position $y$ [m]")
    ax_map.legend(loc="upper right", framealpha=0.9)
    ax_map.grid(True, linestyle=":", alpha=0.6)
    ax_map.set_aspect("equal", "datalim")

    ax_var.set_title("Uncertainty Growth vs Switching Threshold", fontweight="bold")
    ax_var.plot(
        times,
        max_stds,
        color="#cc0000",
        linewidth=1.5,
        label="Worst-Case Std: $\\sqrt{\\lambda_{\\max}}$ [m]",
    )
    ax_var.plot(
        times,
        pos_stds,
        color="#0088cc",
        linestyle="--",
        linewidth=1.3,
        label="Total Pos Std: $\\sqrt{\\mathrm{Tr}(\\Sigma)}$ [m]",
    )
    ax_var.axhline(
        threshold_std,
        color="#ff5500",
        linestyle=":",
        linewidth=1.5,
        label=f"Threshold $\\tau = {threshold_std:.1f}$ m",
    )

    if switch_step is not None:
        t_sw = times[switch_step]
        ax_var.axvline(
            t_sw,
            color="#00aa44",
            linestyle="-.",
            linewidth=1.2,
            label=f"Switch Point ($t={t_sw:.1f}$ s)",
        )
        ax_var.axvspan(
            0, t_sw, color="#00aa44", alpha=0.08, label="Phase 1 (DR Buffer)"
        )
        ax_var.axvspan(
            t_sw, times[-1], color="#cc0000", alpha=0.08, label="Phase 2 (RBPF SLAM)"
        )

    ax_var.set_xlabel("Time $t$ [s]")
    ax_var.set_ylabel("Standard Deviation [m]")
    ax_var.legend(loc="upper left", framealpha=0.9)
    ax_var.grid(True, linestyle=":", alpha=0.6)
    ax_var.set_xlim(0, times[-1])

    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved ellipse verification plot to: {save_path}")

    if show_plot:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test Dead Reckoning Covariance Ellipses."
    )
    parser.add_argument(
        "--v-sigma",
        type=float,
        default=0.12,
        help="Linear velocity noise std [m/s]",
    )
    parser.add_argument(
        "--w-sigma",
        type=float,
        default=0.04,
        help="Angular velocity noise std [rad/s]",
    )
    parser.add_argument(
        "--threshold-std",
        type=float,
        default=2.5,
        help="Switching std threshold [m]",
    )
    parser.add_argument("--dt", type=float, default=0.1, help="Time step [s]")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/graphs/dr_covariance_ellipses.png"),
        help="Output image path",
    )
    parser.add_argument(
        "--no-show", action="store_true", help="Suppress interactive window"
    )
    args = parser.parse_args()

    run_dr_ellipse_simulation(
        v_sigma=args.v_sigma,
        w_sigma=args.w_sigma,
        threshold_std=args.threshold_std,
        dt=args.dt,
        save_path=args.output,
        show_plot=not args.no_show,
    )


if __name__ == "__main__":
    main()
