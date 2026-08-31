"""
examples/rbpf_slam_toy.py
=========================
Lightweight, instantaneous RBPF-SLAM demo using an exact analytical
advection-diffusion plume function (no numerical solver or external datasets needed).

Demonstrates:
1. Differential drive robot kinematics.
2. Exact continuous analytical advection-diffusion ground truth plume.
3. Rao-Blackwellized Particle Filter (RBPF) SLAM with oracle mode.
4. Real-time dead-reckoning vs. RBPF estimated trajectory tracking error.

Usage::

    python3 examples/rbpf_slam_toy.py
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from pde_slam.kinematics import DiffDriveKinematics
from pde_slam.slam import RbpfSlam
from pde_slam.viz import (
    render_field_panel,
    render_tracking_error_panel,
    render_trajectories_panel,
)


def analytical_plume(t: float | jnp.ndarray, poses: jnp.ndarray) -> jnp.ndarray:
    """Exact analytical advection-diffusion field (advecting Gaussian plume).

    Parameters
    ----------
    t : float or Array
        Time [s].
    poses : Array of shape (N, 2)
        Query coordinates [x, y].

    Returns
    -------
    Array of shape (N, 1)
        Scalar concentration field.
    """
    u_flow, v_flow = 0.3, -0.15
    D = 0.2
    sigma0_sq = 25.0
    ambient, peak = 35.0, 18.0

    sigma_t_sq = sigma0_sq + 2.0 * D * t
    cx_t = -20.0 + u_flow * t
    cy_t = -15.0 + v_flow * t

    dist_sq = (poses[:, 0] - cx_t) ** 2 + (poses[:, 1] - cy_t) ** 2
    amplitude_ratio = sigma0_sq / sigma_t_sq
    plume = (ambient - peak) * amplitude_ratio * jnp.exp(-dist_sq / (2.0 * sigma_t_sq))
    return (ambient - plume)[:, None]


def main(num_particles: int = 50, show: bool = True) -> None:
    """Run instantaneous analytical toy RBPF-SLAM demonstration."""
    print(f"Initializing RBPF-SLAM Toy Demo ({num_particles} particles)...")

    # 1. Kinematics & Waypoint Trajectory
    x0, y0, heading0 = -25.0, -25.0, 0.0
    waypoints = np.array(
        [
            [-25.0, -25.0],
            [25.0, -25.0],
            [25.0, 25.0],
            [-25.0, 25.0],
            [-25.0, -25.0],
        ],
        dtype=np.float32,
    )
    robot = DiffDriveKinematics(x0=x0, y0=y0, heading0=heading0)
    states, velocities_nom, omegas_nom = robot.drive_to_waypoints(
        waypoints, speed_mps=3.5, dt=1.0, acceptance_radius=3.0
    )
    n_steps = len(velocities_nom)
    dt = 1.0
    times = np.arange(n_steps + 1) * dt

    # Add actuator noise to true trajectory
    np.random.seed(42)
    v_noise = np.random.normal(0.0, 0.15, size=n_steps)
    omega_noise = np.random.normal(0.0, 0.03, size=n_steps)
    velocities_true = velocities_nom + v_noise
    omegas_true = omegas_nom + omega_noise

    coords_true = np.array(
        DiffDriveKinematics.integrate_trajectory(
            jnp.array([x0, y0, heading0]),
            jnp.array(velocities_true),
            jnp.array(omegas_true),
            dt,
        )
    )
    coords_dr = np.array(
        DiffDriveKinematics.integrate_trajectory(
            jnp.array([x0, y0, heading0]),
            jnp.array(velocities_nom),
            jnp.array(omegas_nom),
            dt,
        )
    )

    # 2. Setup RBPF-SLAM Filter
    rbpf = RbpfSlam(
        n_particles=num_particles,
        process_noise=jnp.diag(jnp.array([0.15**2, 0.03**2])),
        measurement_noise=jnp.array([0.05**2]),
        p0_lin=0.0025,
        threshold_ratio=0.5,
        measurement_mode="oracle",
        oracle_fn=analytical_plume,
        seed=42,
    )
    rbpf.initialize(
        initial_state=jnp.array([x0, y0, heading0]),
        std_dev=jnp.array([0.2, 0.2, 0.02]),
        n_fields=1,
    )

    estimated_traj = [coords_true[0, :2]]
    obs_noise_std = 0.05

    print(f"Running Online RBPF Filter over {n_steps} steps...")
    for k in range(n_steps):
        t_now = times[k + 1]
        u_cmd = jnp.array([velocities_nom[k], omegas_nom[k]])

        # Motion prediction
        rbpf.predict(control=u_cmd, dt=dt)

        # True observation with sensor noise
        true_pos = jnp.array([coords_true[k + 1, :2]])
        true_val = float(analytical_plume(t_now, true_pos)[0, 0])
        noisy_measurement = true_val + float(np.random.normal(0.0, obs_noise_std))

        # Filter update & resample
        rbpf.update(measurement=jnp.array([noisy_measurement]), t_now=t_now)
        rbpf.resample()

        # Best trajectory estimate
        best_traj, _ = rbpf.get_best_estimate()
        estimated_traj.append(np.array(best_traj[-1]))

    estimated_traj = np.array(estimated_traj)

    # Tracking error
    err_dr = np.linalg.norm(coords_dr[:, :2] - coords_true[:, :2], axis=-1)
    err_rbpf = np.linalg.norm(estimated_traj - coords_true[:, :2], axis=-1)

    print(f"Final Dead-Reckoning RMSE: {err_dr[-1]:.3f} m")
    print(f"Final RBPF Estimated RMSE: {err_rbpf[-1]:.3f} m")

    # 3. Visualization
    # 3. Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    xg = np.linspace(-35.0, 35.0, 70)
    yg = np.linspace(-35.0, 35.0, 70)
    XX, YY = np.meshgrid(xg, yg, indexing="ij")
    grid_pts = jnp.array(np.column_stack([XX.ravel(), YY.ravel()]))
    t_mid = float(times[len(times) // 2])
    field_mid = np.array(analytical_plume(t_mid, grid_pts)).reshape(70, 70)

    render_field_panel(
        ax=ax1,
        X=XX,
        Y=YY,
        field_data=field_mid,
        cmap="viridis",
        colorbar=True,
        colorbar_orientation="horizontal",
    )
    render_trajectories_panel(
        ax=ax1,
        coords_dict={
            "ground_truth": coords_true[:, :2],
            "dead_reckoning": coords_dr[:, :2],
            "online_rbpf": estimated_traj[:, :2],
        },
        legend=True,
    )
    ax1.set_title(
        f"Analytical Plume & Trajectories (t = {t_mid:.0f}s)",
        fontsize=11,
        fontweight="bold",
    )

    render_tracking_error_panel(
        ax=ax2,
        times=times,
        errors_dict={
            "dead_reckoning": err_dr,
            "online_rbpf": err_rbpf,
        },
        legend=True,
    )
    ax2.set_title("Tracking Error Comparison Over Time", fontsize=11, fontweight="bold")

    fig.tight_layout()
    out_file = Path("output/graphs") / "demo_rbpf_slam_toy.png"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=200)

    print(f"Saved figure to {out_file}")

    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
