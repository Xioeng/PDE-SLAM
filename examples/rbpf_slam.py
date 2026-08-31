"""
examples/rbpf_slam.py
=====================
Multi-field Rao-Blackwellized Particle Filter (RBPF) SLAM and online Physics-
Informed Neural Network (PINN) map training on PDE simulation datasets, with
user-selected Initial Condition (IC) measurement points at t=0.

Run (interactive GUI)::

    JAX_PLATFORMS=cpu uv run python \
        examples/rbpf_slam.py \
        --config configs/biscayne_rbpf_simulation.yaml

Run (headless, pre-specified waypoints)::

    JAX_PLATFORMS=cpu uv run python \
        examples/rbpf_slam.py \
        --config configs/biscayne_rbpf_simulation.yaml \
        --waypoints "-200,50; 100,80; 300,-30" \
        --no-show
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from pde_slam.config import load_rbpf_experiment_config
from pde_slam.interpolators import SpatiotemporalInterpolator
from pde_slam.io import SlamExperimentData, save_experiment
from pde_slam.io.simulation import (
    generate_ic_anchors,
    load_simulation_dataset,
    sample_simulation_field,
)
from pde_slam.kinematics import DiffDriveKinematics
from pde_slam.pinn import PinnConfig, PinnFieldMap, PinnParams
from pde_slam.slam import RbpfSlam
from pde_slam.viz import (
    LiveSlamVisualizer,
    fetch_satellite_enu_backdrop,
    pick_waypoints_gui,
    plot_saved_experiment,
)


def parse_points_cli(pts_str: str) -> list[tuple[float, float]]:
    """Parse CLI ENU points string 'x,y; x,y; ...'.

    Parameters
    ----------
    pts_str : str
        Semicolon-separated ``x,y`` pairs.

    Returns
    -------
    list of tuple of float
        Parsed ``(east, north)`` pairs.
    """
    pts = []
    for pair in pts_str.split(";"):
        if not pair.strip():
            continue
        c0, c1 = pair.split(",")
        pts.append((float(c0.strip()), float(c1.strip())))
    return pts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv : list of str, optional
        Argument list; defaults to ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Multi-Field RBPF-SLAM on PDE Simulations with t=0 IC anchors.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/biscayne_rbpf_simulation.yaml"),
        help="Path to YAML experiment config.",
    )
    parser.add_argument(
        "--waypoints",
        type=str,
        default=None,
        help="ENU waypoints 'x,y; x,y; ...' (skips GUI). "
        "Required when --no-show is set.",
    )
    parser.add_argument(
        "--ic-points",
        type=str,
        default=None,
        help="ENU IC anchor points 'x,y; x,y; ...' (skips GUI/auto).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Suppress interactive GUI windows (requires --waypoints).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:  # noqa: C901
    """Run multi-field RBPF + PINN SLAM initialized with t=0 IC anchors.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments forwarded to :func:`parse_args`.
    """
    args = parse_args(argv)

    print("==================================================")
    print("Online RBPF-SLAM on PDE Simulations with Initial Condition Dataset (t=0)")
    print("==================================================")

    # ------------------------------------------------------------------
    # Configuration: Load from YAML config file
    # ------------------------------------------------------------------
    cfg = load_rbpf_experiment_config(args.config)

    sim_dir = Path(cfg.simulation.sim_dir)
    fields_req: list[str] | None = (
        cfg.simulation.fields if cfg.simulation.fields else None
    )
    pinn_arch: str = cfg.pinn.arch
    n_ic_points: int = cfg.ic_anchors.n_points
    n_ic_seed: int = cfg.ic_anchors.seed
    warmup_epochs: int = cfg.ic_anchors.epochs
    num_particles: int = cfg.rbpf.n_particles
    speed: float = cfg.robot.nominal_speed
    acceptance_radius: float = cfg.robot.acceptance_radius

    OBS_NOISE_STD: float = cfg.rbpf.measurement_noise_std
    VELOCITY_NOISE_STD: float = cfg.robot.v_noise_std
    OMEGA_NOISE_STD: float = cfg.robot.omega_noise_std
    pos_init_std: float = cfg.rbpf.pos_init_std
    heading_init_std: float = cfg.rbpf.heading_init_std
    rbpf_seed: int = cfg.rbpf.seed
    resample_threshold: float = cfg.rbpf.resample_threshold

    output_dir = Path(cfg.output.results_dir)
    graphs_dir = output_dir / "graphs"
    results_dir = output_dir / "results"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_pcts: list[float] = [float(p) for p in cfg.output.checkpoints]

    prng_key = jax.random.PRNGKey(rbpf_seed)
    np.random.seed(rbpf_seed)

    # ------------------------------------------------------------------
    # 1. Load simulation dataset
    # ------------------------------------------------------------------
    print(f"\n[1/5] Loading simulation dataset from: {sim_dir}")
    if not Path(sim_dir).exists():
        raise FileNotFoundError(f"Simulation directory '{sim_dir}' does not exist.")

    sim_data = load_simulation_dataset(sim_dir, requested_fields=fields_req)
    fields_to_use = sim_data.field_names
    n_fields = len(fields_to_use)
    grid = sim_data.grid
    polygon_enu = sim_data.polygon_enu
    mesh_mask = sim_data.mesh_mask
    sample_times_sim = sim_data.sample_times
    sim_t_max = float(sample_times_sim[-1])
    sim_n_times = len(sample_times_sim)
    enu_frame = sim_data.enu_frame

    print(f"  Loaded {n_fields} simulation fields: {fields_to_use}")
    print(f"  Time steps count: {sim_n_times} (t_max = {sim_t_max:.1f} s)")
    print(f"  Grid: {grid}")
    print(f"  Boundary polygon vertices: {len(polygon_enu)}")

    # Normalize solutions in-place (zero-mean / unit-variance)
    print("\nNormalizing simulation fields to zero-mean and unit-variance...")
    for f_name in fields_to_use:
        m = sim_data.field_means[f_name]
        s = sim_data.field_stds[f_name]
        sim_data.simulations[f_name]["solutions"] = (
            sim_data.simulations[f_name]["solutions"] - m
        ) / s
        print(f"  Field '{f_name}': mean = {m:.4f}, std = {s:.4f}")

    field_means_arr = np.array(
        [sim_data.field_means[f] for f in fields_to_use], dtype=np.float32
    )
    field_stds_arr = np.array(
        [sim_data.field_stds[f] for f in fields_to_use], dtype=np.float32
    )

    # Per-field metadata (name, initial snapshot, diffusivity)
    field_metadata: list[tuple[str, np.ndarray, float]] = []
    for f_name in fields_to_use:
        d_val = float(
            sim_data.simulations[f_name]
            .get("config", {})
            .get("diffusion_coefficient", 0.05)
        )
        init_snap = sim_data.simulations[f_name]["solutions"][0]
        field_metadata.append((f_name, init_snap, d_val))
        vmin, vmax = float(np.nanmin(init_snap)), float(np.nanmax(init_snap))
        print(
            f"  Field '{f_name}': Initial range = [{vmin:.2f}, {vmax:.2f}], "
            f"D = {d_val:.3f} m^2/s"
        )

    # Satellite backdrop (best-effort; margin_factor=1.05 keeps tiles close to polygon)
    sat_img, sat_extent = fetch_satellite_enu_backdrop(
        enu_frame=enu_frame,
        polygon_enu=polygon_enu,
        zoom=cfg.output.satellite_zoom,
        margin_factor=1.05,
    )

    # ------------------------------------------------------------------
    # 2. Initial Condition (t=0) anchor points
    # ------------------------------------------------------------------
    print("\n[2/5] Selecting Initial Condition (t=0) measurement points...")
    ic_points_enu: np.ndarray

    if args.ic_points is not None:
        print(f"  Using CLI IC points: {args.ic_points}")
        ic_points_enu = np.array(parse_points_cli(args.ic_points))
    elif not args.no_show:
        print("  Opening GUI for IC anchor placement (close to confirm)...")
        ic_pts_raw = pick_waypoints_gui(
            polygon_enu=polygon_enu,
            sat_img=sat_img,
            sat_extent=sat_extent,
        )
        if len(ic_pts_raw) >= 1:
            ic_points_enu = ic_pts_raw
        else:
            print(f"  No IC points selected — auto-generating {n_ic_points}...")
            ic_points_enu = generate_ic_anchors(
                polygon_enu, n_ic_points, seed=n_ic_seed, dataset=sim_data
            )
    else:
        print(f"  Generating {n_ic_points} automated IC anchor points...")
        ic_points_enu = generate_ic_anchors(
            polygon_enu, n_ic_points, seed=n_ic_seed, dataset=sim_data
        )

    print(f"  Initial Condition anchor count: {len(ic_points_enu)}")

    # ------------------------------------------------------------------
    # 3. Vehicle trajectory waypoints
    # ------------------------------------------------------------------
    print("\n[3/5] Selecting vehicle trajectory waypoints...")
    waypoints_enu: np.ndarray

    if args.waypoints is not None:
        print(f"  Using CLI waypoints: {args.waypoints}")
        waypoints_enu = np.array(parse_points_cli(args.waypoints))
    elif args.no_show:
        raise SystemExit(
            "ERROR: --no-show requires --waypoints to be specified. "
            "No lawnmower fallback is available."
        )
    else:
        print("  Opening GUI for waypoint selection (close to confirm)...")
        waypoints_enu = pick_waypoints_gui(
            polygon_enu=polygon_enu,
            sat_img=sat_img,
            sat_extent=sat_extent,
            ic_anchors=ic_points_enu,
        )
        if len(waypoints_enu) < 2:
            raise SystemExit(
                "ERROR: fewer than 2 waypoints selected. "
                "Please re-run and select at least 2."
            )

    print(f"  Waypoints count: {len(waypoints_enu)}")

    # ------------------------------------------------------------------
    # 4. Generate robot trajectories
    # ------------------------------------------------------------------
    dt: float = cfg.robot.dt
    x0_true = waypoints_enu[0]
    print(f"  Robot start position (ENU): x0 = {x0_true}")
    print(f"  Time step dt = {dt} s  |  Simulation t_max = {sim_t_max:.1f} s")

    controller = DiffDriveKinematics(
        x0=float(x0_true[0]), y0=float(x0_true[1]), heading0=0.0
    )
    _, velocities_nominal, omegas_nominal = controller.drive_to_waypoints(
        waypoints=waypoints_enu[1:],
        speed_mps=speed,
        dt=dt,
        acceptance_radius=acceptance_radius,
    )

    n_steps = len(velocities_nominal)
    traj_t_max = n_steps * dt
    if traj_t_max > sim_t_max:
        import warnings

        warnings.warn(
            f"Trajectory duration {traj_t_max:.1f} s (= {n_steps} steps × dt={dt} s) "
            f"exceeds simulation t_max = {sim_t_max:.1f} s. "
            "Simulation snapshots will be clamped to the last available timestamp.",
            stacklevel=2,
        )
    sample_times = np.arange(n_steps + 1, dtype=float) * dt
    velocities_nominal = np.array(velocities_nominal)
    omegas_nominal = np.array(omegas_nominal)

    print(
        f"  Trajectory: {n_steps} steps, "
        f"t_end = {traj_t_max:.1f} s "
        f"({'WARN: exceeds sim' if traj_t_max > sim_t_max else 'OK'})"
    )

    prng_key, k_vn, k_on = jax.random.split(prng_key, 3)
    v_noise = np.array(VELOCITY_NOISE_STD * jax.random.normal(k_vn, shape=(n_steps,)))
    w_noise = np.array(OMEGA_NOISE_STD * jax.random.normal(k_on, shape=(n_steps,)))
    v_act = np.clip(velocities_nominal + v_noise, 0.0, None)
    w_act = omegas_nominal + w_noise

    coords_true = DiffDriveKinematics.integrate_trajectory(
        x0_true, v_act, w_act, dt, include_initial=True
    )
    coords_dr = DiffDriveKinematics.integrate_trajectory(
        x0_true, velocities_nominal, omegas_nominal, dt, include_initial=True
    )

    # ------------------------------------------------------------------
    # 5. PINN init + IC warm-up
    # ------------------------------------------------------------------
    print("\n[4/5] Initializing PINN map & seeding dataset buffer with IC (t=0)...")
    prng_key, k_pinn = jax.random.split(prng_key)
    log_d_inits = tuple(float(jnp.log(0.1)) for _ in range(n_fields))

    config_pinn = PinnConfig(
        x_bounds=(grid.x_min, grid.x_max),
        y_bounds=(grid.y_min, grid.y_max),
        t_max=sim_t_max,
        n_fields=n_fields,
        arch=pinn_arch,
        v_flow_init=(0.0, 0.0),
        log_D_init=log_d_inits,
        hidden_dim=cfg.pinn.hidden_dim,
        num_layers=cfg.pinn.num_layers,
        learning_rate=cfg.pinn.learning_rate,
        num_steps=cfg.pinn.num_steps,
        num_colloc=cfg.pinn.num_colloc,
        margin=cfg.pinn.margin,
        w_pde=cfg.pinn.w_pde,
    )
    pinn_map = PinnFieldMap(config=config_pinn, key=k_pinn)
    assert pinn_map.params is not None
    pinn_params = pinn_map.params

    print(f"  Architecture: {pinn_arch}")
    print(f"  Target normalization means: {field_means_arr}")
    print(f"  Target normalization stds : {field_stds_arr}")

    # Seed buffer from IC anchors (t=0) + start position
    data_pts_buf: list[list[float]] = []
    obs_vals_buf: list[list[float]] = []

    print(f"  Sampling ground truth at t=0 for {len(ic_points_enu)} IC points...")
    for ic_pos in ic_points_enu:
        ic_vals = [
            sample_simulation_field(
                sim_data,
                f,
                0.0,
                float(ic_pos[0]),
                float(ic_pos[1]),
                normalized=False,
            )
            for f in fields_to_use
        ]
        noise = np.random.normal(0.0, OBS_NOISE_STD, size=n_fields)
        obs_val0 = [v + float(n) for v, n in zip(ic_vals, noise, strict=True)]
        data_pts_buf.append([0.0, float(ic_pos[0]), float(ic_pos[1])])
        obs_vals_buf.append(obs_val0)

    # Robot start position at t=0
    start_vals = [
        sample_simulation_field(
            sim_data,
            f,
            0.0,
            float(x0_true[0]),
            float(x0_true[1]),
            normalized=False,
        )
        for f in fields_to_use
    ]
    start_obs = [v + float(np.random.normal(0.0, OBS_NOISE_STD)) for v in start_vals]
    data_pts_buf.append([0.0, float(x0_true[0]), float(x0_true[1])])
    obs_vals_buf.append(start_obs)

    print(
        f"  Warm-up pre-training PINN on {len(obs_vals_buf)} IC points "
        f"({warmup_epochs} epochs)..."
    )
    buf_pts_init = jnp.array(data_pts_buf)
    buf_vals_init = jnp.array(obs_vals_buf)
    init_loss: float = float("inf")
    for _ in range(warmup_epochs):
        prng_key, k_warm = jax.random.split(prng_key)
        pinn_params, _, init_loss = pinn_map.fit(
            buf_pts_init, buf_vals_init, key=k_warm
        )
    print(f"  Warm-up completed. Initial Loss: {init_loss:.4f}")

    # ------------------------------------------------------------------
    # 5b. Initialize RBPF filters
    # ------------------------------------------------------------------
    print("\n[5/5] Initializing RBPF filters (online + oracle)...")

    # Continuous 3D spatiotemporal interpolators for each field (oracle baseline)
    oracle_interpolators: dict[str, SpatiotemporalInterpolator] = {}
    for fn in fields_to_use:
        raw_sols = sim_data.simulations[fn]["solutions"]
        if raw_sols.shape[1:] != grid.shape:
            raw_sols = np.transpose(raw_sols, (0, 2, 1))
        mean_val = float(np.nanmean(raw_sols))
        clean_sols = np.nan_to_num(raw_sols, nan=mean_val)
        oracle_interpolators[fn] = SpatiotemporalInterpolator(
            grid=grid,
            ts=sample_times_sim,
            snapshots=clean_sols,
        )

    def oracle_fn_multi(t: float | jax.Array, poses: jax.Array) -> jax.Array:
        """Oracle field lookup via continuous 3D SpatiotemporalInterpolator."""
        pos_arr = jnp.atleast_2d(poses)
        px = pos_arr[:, 0]
        py = pos_arr[:, 1]
        pt = jnp.full_like(px, float(t) if t is not None else 0.0)

        field_vals = [
            oracle_interpolators[fn](x=px, y=py, t=pt) for fn in fields_to_use
        ]
        stacked = jnp.column_stack(field_vals)
        return stacked[0] if jnp.ndim(poses) == 1 else stacked

    proc_noise = jnp.diag(jnp.array([VELOCITY_NOISE_STD, OMEGA_NOISE_STD])) ** 2
    meas_noise = jnp.array([OBS_NOISE_STD**2] * n_fields)
    rbpf_online = RbpfSlam(
        n_particles=num_particles,
        process_noise=proc_noise,
        measurement_noise=meas_noise,
        p0_lin=jnp.array([cfg.rbpf.p0_lin]),
        threshold_ratio=resample_threshold,
        measurement_mode="pinn",
        pinn_map=pinn_map,
        seed=rbpf_seed,
    )
    rbpf_oracle = RbpfSlam(
        n_particles=num_particles,
        process_noise=proc_noise,
        measurement_noise=meas_noise,
        p0_lin=jnp.array([cfg.rbpf.p0_lin]),
        threshold_ratio=resample_threshold,
        measurement_mode="oracle",
        oracle_fn=oracle_fn_multi,
        seed=rbpf_seed,
    )

    init_state = jnp.array([x0_true[0], x0_true[1], 0.0])
    init_std = jnp.array([pos_init_std, pos_init_std, heading_init_std])
    rbpf_online.initialize(init_state, init_std, n_fields=n_fields)
    rbpf_oracle.initialize(init_state, init_std, n_fields=n_fields)

    print(f"  RBPF particles: {num_particles}")
    print(
        f"  Noise: OBS={OBS_NOISE_STD:.4f}, V={VELOCITY_NOISE_STD:.4f}, "
        f"W={OMEGA_NOISE_STD:.4f}"
    )

    # Checkpoint steps mapped from config percentages (0%, 25%, 50%, ...)
    step_to_pct: dict[int, int] = {}
    for p in checkpoint_pcts:
        if float(p) > 0:
            step_val = max(1, min(int(round(float(p) / 100.0 * n_steps)), n_steps))
            step_to_pct[step_val] = int(round(float(p)))

    stage_params: dict[int, PinnParams] = {0: pinn_params}

    stage_trajs_true: dict[int, np.ndarray] = {0: np.array([coords_true[0]])}

    stage_trajs_dr: dict[int, np.ndarray] = {0: np.array([coords_dr[0]])}
    stage_trajs_rbpf: dict[int, np.ndarray] = {0: np.array([coords_true[0]])}
    stage_trajs_oracle: dict[int, np.ndarray] = {0: np.array([coords_true[0]])}
    stage_particles: dict[int, np.ndarray] = {0: np.array(rbpf_online.poses)}

    if any(float(p) == 0.0 for p in checkpoint_pcts):
        v_est0 = np.array(pinn_params.v_flow)
        d_est0 = np.array(jnp.exp(pinn_params.log_D))
        d_str0 = ", ".join(f"{d:.3f}" for d in d_est0)
        print(
            f"    Checkpoint   0% (Step   0/{n_steps}) | "
            f"Total Samples (IC + Path): {len(obs_vals_buf):3d} | "
            f"Loss: {init_loss:.4f} | "
            f"v_flow: [{v_est0[0]:.3f}, {v_est0[1]:.3f}] m/s | "
            f"D: [{d_str0}] m^2/s"
        )

    # ------------------------------------------------------------------
    # Live trajectory animation visualizer
    # ------------------------------------------------------------------
    print("\nInitializing Live Trajectory Animation...")
    viz = LiveSlamVisualizer(
        t_max=sim_t_max,
        polygon_enu=polygon_enu,
        ic_points_enu=ic_points_enu,
        sat_img=sat_img,
        sat_extent=sat_extent,
        coords_true=coords_true,
        initial_particles=rbpf_online.poses,
        interactive=not args.no_show,
    )

    # ------------------------------------------------------------------
    # Online RBPF + PINN SLAM loop
    # ------------------------------------------------------------------
    print(
        f"\nExecuting Online Multi-Field RBPF + PINN SLAM Loop ({n_steps} steps)...",
        flush=True,
    )
    loss_val: float = init_loss

    pbar = tqdm(
        range(n_steps), desc="  RBPF-SLAM Loop", unit="step", dynamic_ncols=True
    )
    for step_idx in pbar:
        t_curr = float(sample_times[step_idx + 1])
        true_pos = coords_true[step_idx + 1]

        true_vals = [
            sample_simulation_field(
                sim_data,
                f,
                t_curr,
                float(true_pos[0]),
                float(true_pos[1]),
                normalized=False,
            )
            for f in fields_to_use
        ]
        noise = np.random.normal(0.0, OBS_NOISE_STD, size=n_fields)
        obs_val = [t_v + float(n) for t_v, n in zip(true_vals, noise, strict=True)]

        prng_key, k_colloc = jax.random.split(prng_key)
        ctrl = jnp.array([velocities_nominal[step_idx], omegas_nominal[step_idx]])

        rbpf_oracle.predict(control=ctrl, dt=dt)
        rbpf_oracle.update(measurement=jnp.array(obs_val), t_now=t_curr)
        rbpf_oracle.resample()

        rbpf_online.predict(control=ctrl, dt=dt)
        rbpf_online.update(measurement=jnp.array(obs_val), t_now=t_curr)
        rbpf_online.resample()

        _, consensus_pose = rbpf_online.get_best_estimate()
        data_pts_buf.append(
            [t_curr, float(consensus_pose[0]), float(consensus_pose[1])]
        )
        obs_vals_buf.append(obs_val)

        if step_idx % 1 == 0:
            buf_pts = jnp.array(data_pts_buf)
            buf_vals = jnp.array(obs_vals_buf)
            pinn_params, _, loss_val = pinn_map.fit(buf_pts, buf_vals, key=k_colloc)

        v_est = np.array(pinn_params.v_flow)
        d_est = np.array(jnp.exp(pinn_params.log_D))
        pbar.set_postfix(
            loss=f"{loss_val:.4f}",
            v_x=f"{v_est[0]:.2f}",
            v_y=f"{v_est[1]:.2f}",
        )

        curr_rbpf_traj, curr_rbpf_pos = rbpf_online.get_best_estimate()
        curr_oracle_traj, curr_oracle_pos = rbpf_oracle.get_best_estimate()

        viz.update(
            step=step_idx + 1,
            t_curr=t_curr,
            true_pos=true_pos,
            rbpf_pos=curr_rbpf_pos,
            dr_pos=coords_dr[step_idx + 1],
            oracle_pos=curr_oracle_pos,
            particles=rbpf_online.poses,
            loss_val=loss_val,
        )

        step_num = step_idx + 1
        if step_num in step_to_pct:
            stage_params[step_num] = pinn_params
            stage_trajs_true[step_num] = np.array(coords_true[: step_num + 1])
            stage_trajs_dr[step_num] = np.array(coords_dr[: step_num + 1])
            stage_trajs_rbpf[step_num] = np.array(curr_rbpf_traj)
            stage_trajs_oracle[step_num] = np.array(curr_oracle_traj)
            stage_particles[step_num] = np.array(rbpf_online.poses)
            d_str = ", ".join(f"{d:.3f}" for d in d_est)
            pct = step_to_pct[step_num]
            tqdm.write(
                f"    Checkpoint {pct:3d}% (Step {step_num:3d}/{n_steps}) | "
                f"Total Samples (IC + Path): {len(obs_vals_buf):3d} | "
                f"Loss: {loss_val:.4f} | "
                f"v_flow: [{v_est[0]:.3f}, {v_est[1]:.3f}] m/s | "
                f"D: [{d_str}] m^2/s"
            )

    # ------------------------------------------------------------------
    # Trajectory metrics
    # ------------------------------------------------------------------
    estimated_traj, _ = rbpf_online.get_best_estimate()
    oracle_traj, _ = rbpf_oracle.get_best_estimate()

    dr_rmse = float(
        np.sqrt(np.mean((np.array(coords_dr) - np.array(coords_true)) ** 2))
    )
    oracle_rmse = float(
        np.sqrt(np.mean((np.array(oracle_traj) - np.array(coords_true)) ** 2))
    )
    rbpf_rmse = float(
        np.sqrt(np.mean((np.array(estimated_traj) - np.array(coords_true)) ** 2))
    )
    err_red = ((dr_rmse - rbpf_rmse) / dr_rmse) * 100.0

    print("\n--------------------------------------------------")
    print(f"1. Dead Reckoning Trajectory RMSE:       {dr_rmse:.3f} m")
    print(f"2. RBPF (Oracle Map) Trajectory RMSE:    {oracle_rmse:.3f} m")
    print(f"3. Online RBPF-SLAM (with IC) RMSE:      {rbpf_rmse:.3f} m")
    print(f"SLAM Error Reduction vs Dead Reckoning: {err_red:.1f}%")
    print("--------------------------------------------------")

    traj_plot_path = graphs_dir / "demo_rbpf_slam_trajectory_comparison.png"
    viz.finalize(
        output_path=traj_plot_path,
        rmse_dr=dr_rmse,
        rmse_rbpf=rbpf_rmse,
        rmse_oracle=oracle_rmse,
    )

    # ------------------------------------------------------------------
    # Save experiment dataset (.pkl)
    # ------------------------------------------------------------------
    sim_name = cfg.output.sim_name
    exp_file_path = results_dir / f"{sim_name}_rbpf_slam_experiment.pkl"

    print(
        "  Evaluating full 4D field estimations tensor across all timestamps "
        "(T, n_fields, nx, ny)..."
    )
    X_ij, Y_ij = np.meshgrid(
        np.linspace(grid.x_min, grid.x_max, grid.nx),
        np.linspace(grid.y_min, grid.y_max, grid.ny),
        indexing="ij",
    )
    pinn_map.params = pinn_params
    tensor_list = []
    for t_val in sample_times:
        qg = jnp.stack([jnp.full(X_ij.shape, float(t_val)), X_ij, Y_ij], axis=-1)
        pred_norm = np.array(pinn_map.forward(qg))
        pred_phys = (pred_norm * field_stds_arr) + field_means_arr
        tensor_list.append(np.moveaxis(pred_phys, -1, 0))

    field_estimations_tensor = np.stack(tensor_list, axis=0)

    d_true_arr = np.array([meta[2] for meta in field_metadata])
    enu_lat0 = float(enu_frame.lat0) if hasattr(enu_frame, "lat0") else 0.0
    enu_lon0 = float(enu_frame.lon0) if hasattr(enu_frame, "lon0") else 0.0
    exp_data = SlamExperimentData(
        sim_name=sim_name,
        grid_extent={
            "x_min": grid.x_min,
            "x_max": grid.x_max,
            "y_min": grid.y_min,
            "y_max": grid.y_max,
            "nx": grid.nx,
            "ny": grid.ny,
        },
        enu_origin=(enu_lat0, enu_lon0),
        polygon_enu=polygon_enu,
        mesh_mask=mesh_mask,
        sample_times=sample_times,
        field_names=fields_to_use,
        ground_truth_params={"v_flow": np.array([0.0, 0.0]), "D": d_true_arr},
        noise_params={
            "velocity_noise_std": VELOCITY_NOISE_STD,
            "omega_noise_std": OMEGA_NOISE_STD,
            "obs_noise_std": OBS_NOISE_STD,
        },
        gt_solutions={
            fn: sim_data.simulations[fn]["solutions"] for fn in fields_to_use
        },
        field_means=sim_data.field_means,
        field_stds=sim_data.field_stds,
        field_estimations=field_estimations_tensor,
        coords_true=coords_true,
        coords_dr=coords_dr,
        oracle_traj=np.array(oracle_traj),
        estimated_traj=np.array(estimated_traj),
        velocities=velocities_nominal,
        omegas=omegas_nominal,
        ic_points_enu=ic_points_enu,
        obs_pts=np.array(data_pts_buf),
        obs_vals=np.array(obs_vals_buf),
        pinn_config=config_pinn,
        pinn_checkpoints=stage_params,
    )
    if cfg.output.save_experiment:
        save_experiment(exp_data, exp_file_path)
        print(f"  Saved experiment dataset to: {exp_file_path}")

    # ------------------------------------------------------------------
    # Generate grid figures via pde_slam.viz module
    # ------------------------------------------------------------------
    print("\nGenerating grid figures ...")
    fig_paths = plot_saved_experiment(
        experiment=exp_data,
        output_dir=graphs_dir,
        sim_name=sim_name,
        save_grids=cfg.output.save_grids,
        save_individual=False,
        zoom=cfg.output.satellite_zoom,
    )
    for fig_name, fig_p in fig_paths.items():
        print(f"  Saved {fig_name} to: {fig_p}")

    print("\nExperiment run completed successfully!")
    print("==================================================")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
