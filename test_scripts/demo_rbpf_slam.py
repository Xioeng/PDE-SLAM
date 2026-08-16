"""
test_scripts/demo_rbpf_slam.py
==============================
Demonstrates true online Rao-Blackwellized Particle Filter (RBPF) SLAM with an online-trained
Physics-Informed Neural Network (PINN) map in a 300 × 300 m aquatic domain.

To prevent cold-start likelihood corruption from untrained random neural weights:
1. RBPF uses fast local spatial interpolation for particle measurement likelihoods.
2. The PINN map φ̂_θ(t, x, y) is trained online from the filtered consensus poses x̄_k
   using Data Loss + PDE Physics Residual Loss.

Run::

    python test_scripts/demo_rbpf_slam.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path so pde_slam is importable without package installation
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax  # type: ignore[import-untyped]

from pde_slam.interpolators import FieldInterpolator, SpatialGrid
from pde_slam.kinematics import UnicycleKinematics
from pde_slam.optimization.rbpf import RbpfSlam
from pde_slam.pinn import (
    PinnDomainConfig,
    PinnFieldMap,
    pinn_loss_fn,
    sample_trajectory_collocation_points,
)
from pde_slam.solver import AdvectionDiffusionSolver, PDEParams


def main() -> None:
    print("Initializing Online RBPF + PINN SLAM Simulation...")
    np.random.seed(42)
    key = jax.random.PRNGKey(42)

    # 1. Setup Spatial Grid & Ground Truth PDE Field
    grid = SpatialGrid(x_min=-150.0, x_max=150.0, y_min=-150.0, y_max=150.0, nx=80, ny=80)
    solver = AdvectionDiffusionSolver(grid, dt_max=1.0)

    phi0_true = 30.0 * jnp.exp(-((grid.XX + 20.0) ** 2 + (grid.YY - 40.0) ** 2) / 1500.0)
    v_flow_true = jnp.array([0.4, -0.2])  # East, North velocity [m/s]
    D_true = jnp.array(0.5)

    u_field_true = jnp.broadcast_to(v_flow_true, (grid.ny, grid.nx, 2))
    pde_params_true = PDEParams(u_field=u_field_true, D=D_true)

    n_steps = 60
    dt = 1.0
    times = np.linspace(0.0, n_steps * dt, n_steps + 1)

    print("Simulating ground truth environment...")
    snapshots_true = solver.solve(phi0_true, pde_params_true, t0=0.0, t_end=times[-1], saveat=jnp.array(times))

    # 2. Control Inputs & Ground Truth Trajectory with Actuation Noise
    x0_true = jnp.array([0.0, 0.0])
    k_thrust_true = 5.0

    thrusts_nominal = 100.0 + 20.0 * np.sin(times[:-1] / 5.0)
    headings_nominal = times[:-1] / 10.0

    thrust_noise = np.random.normal(0.0, 12.0, size=n_steps)
    heading_noise = np.random.normal(0.0, 0.08, size=n_steps)

    thrusts_actuated = thrusts_nominal + thrust_noise
    headings_actuated = headings_nominal + heading_noise

    coords_true = UnicycleKinematics.integrate_trajectory(
        x0_true, thrusts_actuated, headings_actuated, dt, k_thrust_true, include_initial=True
    )
    coords_dr = UnicycleKinematics.integrate_trajectory(
        x0_true, thrusts_nominal, headings_nominal, dt, k_thrust_true, include_initial=True
    )

    # 3. Initialize RBPF State & Online PINN Map
    num_particles = 100
    key, k_rbpf, k_pinn = jax.random.split(key, 3)

    state = RbpfSlam.init_state(
        x0=x0_true,
        heading0=headings_nominal[0],
        num_particles=num_particles,
        key=k_rbpf,
        pos_init_std=0.2,
        heading_init_std=0.02,
    )

    # Initialize PINN Domain Metadata Config & Trainable Parameters
    pinn_config = PinnDomainConfig(
        x_bounds=(grid.x_min, grid.x_max),
        y_bounds=(grid.y_min, grid.y_max),
        t_max=times[-1],
    )
    pinn_map = PinnFieldMap(config=pinn_config)
    pinn_params = pinn_map.init_params(
        k_pinn,
        v_flow_init=jnp.array([0.0, 0.0]),
        D_init=0.2,
        hidden_dim=64,
        num_layers=3,
    )
    optimizer = optax.adam(learning_rate=2e-3)
    opt_state = optimizer.init(pinn_params)

    # Online observations buffer
    obs_x_list = [float(x0_true[0])]
    obs_y_list = [float(x0_true[1])]
    obs_val_list = [float(snapshots_true[0, 40, 40])]

    data_buffer_points = [[0.0, float(x0_true[0]), float(x0_true[1])]]
    data_buffer_values = [float(snapshots_true[0, 40, 40])]

    def sample_sensor_value(snapshot_t: Array, pos: Array) -> Array:
        pos_arr = jnp.atleast_2d(pos)
        grid_x = (pos_arr[:, 0] - grid.x_min) / grid.dx
        grid_y = (pos_arr[:, 1] - grid.y_min) / grid.dy
        ix = jnp.clip(jnp.floor(grid_x).astype(int), 0, grid.nx - 2)
        iy = jnp.clip(jnp.floor(grid_y).astype(int), 0, grid.ny - 2)
        vals = snapshot_t[iy, ix]
        return vals[0] if jnp.ndim(pos) == 1 else vals

    print(f"Executing Online RBPF + PINN SLAM Loop ({n_steps} steps)...")
    particle_cloud_history = []

    for t_idx in range(n_steps):
        print(f"Step {t_idx + 1}/{n_steps}")
        t_curr = times[t_idx + 1]
        key, k_pred, k_resamp, k_colloc = jax.random.split(key, 4)

        # Step 1: Particle Motion Predict Step
        state = RbpfSlam.predict(
            state=state,
            thrust=thrusts_nominal[t_idx],
            heading_cmd=headings_nominal[t_idx],
            dt=dt,
            k_thrust=k_thrust_true,
            thrust_noise_std=12.0,
            heading_noise_std=0.08,
            key=k_pred,
        )

        # Step 2: Query Fast Field Interpolator for Particle Likelihoods
        # (Prevents cold-start divergence from uninitialized neural weights)
        true_pos = coords_true[t_idx + 1]
        obs_val = float(sample_sensor_value(snapshots_true[t_idx + 1], true_pos)) + float(np.random.normal(0.0, 0.1))

        # Update spatial interpolator with accumulated observations
        xy_obs = np.column_stack([obs_x_list, obs_y_list])
        vals_obs = np.array(obs_val_list)

        if len(vals_obs) >= 4:
            interpolator = FieldInterpolator(grid, method="spline")
            interpolator.fit(xy_obs, vals_obs)
            pred_grid = interpolator.predict()
            predicted_scalars = sample_sensor_value(pred_grid, state.poses)
        else:
            predicted_scalars = jnp.full((num_particles,), obs_val)

        # Step 3: Update Weights & Resample (obs_std=0.1 matches sensor noise)
        state = RbpfSlam.update_measurement(state, obs_val, predicted_scalars, obs_std=0.1)
        state = RbpfSlam.resample_if_needed(state, k_resamp, threshold_ratio=0.5)
        particle_cloud_history.append(np.array(state.poses))

        # Step 4: Extract Consensus Pose x̄_k
        _, consensus_pose = RbpfSlam.get_best_estimate(state)

        obs_x_list.append(float(consensus_pose[0]))
        obs_y_list.append(float(consensus_pose[1]))
        obs_val_list.append(obs_val)

        data_buffer_points.append([t_curr, float(consensus_pose[0]), float(consensus_pose[1])])
        data_buffer_values.append(obs_val)

        # Step 5: Train PINN Map & Learn Physical Constants (v_flow, D) Online
        buf_pts = jnp.array(data_buffer_points)
        buf_vals = jnp.array(data_buffer_values)

        pinn_params, opt_state, loss_val = pinn_map.fit(
            pinn_params,
            opt_state,
            optimizer,
            buf_pts,
            buf_vals,
            t_curr,
            k_colloc,
            num_steps=5,
            num_colloc=40,
            margin=15.0,
            w_pde=0.05,
        )

    # 4. Extract Final Trajectory & Metrics
    estimated_traj, _ = RbpfSlam.get_best_estimate(state)
    dr_rmse = np.sqrt(np.mean((np.array(coords_dr) - np.array(coords_true)) ** 2))
    rbpf_rmse = np.sqrt(np.mean((np.array(estimated_traj) - np.array(coords_true)) ** 2))

    print("--------------------------------------------------")
    print(f"Dead Reckoning Trajectory RMSE: {dr_rmse:.3f} m")
    print(f"Online RBPF + PINN SLAM Trajectory RMSE: {rbpf_rmse:.3f} m")
    print(f"Trajectory Error Reduction: {((dr_rmse - rbpf_rmse) / dr_rmse) * 100.0:.1f}%")
    print("--------------------------------------------------")

    # 5. Visualization
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    fig_path = output_dir / "demo_rbpf_pinn_slam.png"

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    final_t_grid = jnp.stack([
        jnp.full(grid.XX.shape, times[-1]),
        grid.XX,
        grid.YY
    ], axis=-1)
    pinn_map_eval = pinn_map.forward(pinn_params, final_t_grid)

    # Subplot 1: Ground Truth Field & Trajectory Tracking
    im1 = axes[0].pcolormesh(grid.XX, grid.YY, snapshots_true[-1], cmap="viridis", shading="auto", alpha=0.7)
    fig.colorbar(im1, ax=axes[0], label="Ground Truth Salinity [PSU]")
    axes[0].plot(coords_true[:, 0], coords_true[:, 1], "k-", linewidth=2.5, label="Ground Truth Trajectory")
    axes[0].plot(coords_dr[:, 0], coords_dr[:, 1], "r--", linewidth=1.8, label="Dead Reckoning")
    axes[0].plot(estimated_traj[:, 0], estimated_traj[:, 1], "b-", linewidth=2.0, label="RBPF Estimate")
    axes[0].set_title(f"Trajectory Tracking (RMSE: {rbpf_rmse:.2f} m vs DR: {dr_rmse:.2f} m)")
    axes[0].set_xlabel("East Position [m]")
    axes[0].set_ylabel("North Position [m]")
    axes[0].legend()
    axes[0].grid(True, linestyle=":", alpha=0.5)

    # Subplot 2: Online Learned PINN Map
    im2 = axes[1].pcolormesh(grid.XX, grid.YY, pinn_map_eval, cmap="viridis", shading="auto", alpha=0.7)
    fig.colorbar(im2, ax=axes[1], label="PINN Predicted Salinity [PSU]")
    axes[1].plot(estimated_traj[:, 0], estimated_traj[:, 1], "b.-", markersize=3, label="RBPF Consensus Poses")
    axes[1].set_title("Online Learned PINN Field Map φ̂_θ(t, x, y)")
    axes[1].set_xlabel("East Position [m]")
    axes[1].set_ylabel("North Position [m]")
    axes[1].legend()
    axes[1].grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    print(f"Saved Online PINN-SLAM visualization to {fig_path}")


if __name__ == "__main__":
    main()
