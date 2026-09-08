"""
examples/graph_slam_pinn_survey.py
====================================
Pose-graph SLAM localized by scalar-field measurements, using PINN field maps
as the correction mechanism instead of a Rao-Blackwellized particle filter
(RBPF) -- no RBPF is used anywhere in this script.

Where the RBPF survey example weighs a particle cloud against a shared
multi-output PINN, this one drives a single pose-graph estimate (GraphSlam)
directly: one independent PINN per scalar field (GraphFieldMapper), each
queried via autodiff at a node's current estimated position to get both its
predicted value and its spatial gradient. The prediction-vs-observation
mismatch, weighted by the field's own training loss (an undertrained field
contributes a soft correction; a well-trained one, a sharp one), becomes a
unary "position factor" folded into the same Gauss-Newton pose-graph
optimization as the odometry edges. As more field data is collected and the
maps train further, these corrections sharpen -- the robot's position
estimate should visibly improve purely from acquiring field data, with no
notion of loop closure or revisit detection required.

Run (interactive)::

    uv run python examples/graph_slam_pinn_survey.py \
        --config configs/biscayne_survey_rbpf.yaml

Run (headless)::

    uv run python examples/graph_slam_pinn_survey.py \
        --config configs/biscayne_survey_rbpf.yaml \
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
from pde_slam.io import (
    generate_ic_anchors,
    load_simulation_dataset,
    load_survey_csv,
    sample_simulation_field,
)
from pde_slam.kinematics import DiffDriveKinematics
from pde_slam.pinn import PinnConfig
from pde_slam.slam import GraphFieldMapper, GraphSlam

_FIELD_TO_CSV_COLUMN = {
    "salinity": "Salinity (PPT)",
    "temperature": "Temperature (C)",
    "odo": "ODO (mg/L)",
    "chlorophyll": "Chlorophyll (ug/L)",
}


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
        description=(
            "Pose-graph SLAM localized by per-field PINN measurement factors "
            "(no RBPF) on real CSV survey trajectory with kinematic DR "
            "simulation."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/biscayne_survey_rbpf.yaml"),
        help="Path to YAML experiment config.",
    )
    parser.add_argument(
        "--sim-dir", type=str, default=None, help="Override for simulation directory."
    )
    parser.add_argument(
        "--csv-file", type=str, default=None, help="Override for survey CSV file path."
    )
    parser.add_argument(
        "--t-max", type=float, default=None, help="Optional duration cap [s]."
    )
    parser.add_argument(
        "--fit-interval",
        type=int,
        default=None,
        help="Refit every field's PINN every N steps.",
    )
    parser.add_argument(
        "--optimize-interval",
        type=int,
        default=None,
        help="Run graph optimization every N steps (dense Gauss-Newton solve "
        "cost grows with graph size, so this need not be every step).",
    )
    parser.add_argument(
        "--max-correction",
        type=float,
        default=None,
        help="Trust-region cap [m] on a single field-measurement position "
        "correction step.",
    )
    parser.add_argument(
        "--refresh-window",
        type=int,
        default=None,
        help="Re-linearize this many trailing nodes' field corrections against the "
        "current (better-trained) maps each optimization cycle. 0 disables.",
    )
    parser.add_argument(
        "--pinn-steps",
        type=int,
        default=None,
        help="Override config's pinn.num_steps (gradient steps per fit). More steps "
        "lower map error, which directly bounds achievable position accuracy.",
    )
    parser.add_argument(
        "--pinn-batch",
        type=int,
        default=None,
        help="Observations sampled per PINN gradient step. The default buffer grows "
        "to ~600 points, so too small a batch leaves the map noisily fit.",
    )
    parser.add_argument(
        "--map-error-frac",
        type=float,
        default=None,
        help="Assumed map predictive error as a fraction of each field's observed "
        "min-max range (stored per node). Sets how much a field correction is "
        "trusted; training loss is NOT used for this because it is in-sample and "
        "collapses toward zero, making corrections overconfident.",
    )
    parser.add_argument(
        "--measurement-loop-tol",
        type=float,
        default=None,
        help="Max distance between two nodes' normalized scalar signatures to call "
        "them the same place. 0 disables measurement-based loop closure. Field "
        "gradients are ~0.2 normalized units/m combined, so 0.06 ~ 0.3 m of "
        "position discrimination.",
    )
    parser.add_argument(
        "--measurement-loop-gap",
        type=int,
        default=None,
        help="Minimum node-index separation before a measurement match counts.",
    )
    parser.add_argument(
        "--measurement-loop-radius",
        type=float,
        default=None,
        help="Reject a signature match whose current position estimate is farther "
        "than this (guards against matching a distant look-alike).",
    )
    parser.add_argument(
        "--measurement-loop-info",
        type=float,
        default=None,
        help="Information weight for a measurement loop closure (4.0 ~ 0.5 m std).",
    )
    parser.add_argument(
        "--use-csv-measurements",
        action="store_true",
        help="Read the field readings from the CSV's sensor columns instead of "
        "resampling the simulation. Required for CSVs whose measurements encode "
        "something the simulation does not reproduce -- e.g. the out-and-back "
        "CSV, whose return leg replays the outbound readings in reverse. "
        "Resampling the time-evolving simulation instead gives a revisited "
        "place a completely different signature, so revisits become "
        "undetectable and loop closure never fires.",
    )
    parser.add_argument(
        "--no-loop-closure",
        action="store_true",
        help="Disable measurement-signature loop closure entirely (ablation).",
    )
    parser.add_argument(
        "--seed", type=int, default=43, help="PRNG seed for process/measurement noise."
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Suppress interactive plot window; only save the figure to disk.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run field-measurement pose-graph SLAM on a real survey CSV trajectory.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments forwarded to :func:`parse_args`.
    """
    args = parse_args(argv)

    print("==================================================")
    print("Pose-Graph SLAM with PINN Field-Measurement Correction (no RBPF)")
    print("==================================================")

    cfg = load_rbpf_experiment_config(args.config)

    # CLI flags override the config's graph_slam section; anything left unset
    # falls back to the config value.
    gs = cfg.graph_slam
    def _pick(cli, cfg_val):
        return cfg_val if cli is None else cli

    args.fit_interval = _pick(args.fit_interval, gs.fit_interval)
    args.optimize_interval = _pick(args.optimize_interval, gs.optimize_interval)
    args.max_correction = _pick(args.max_correction, gs.max_correction)
    args.refresh_window = _pick(args.refresh_window, gs.refresh_window)
    args.pinn_batch = _pick(args.pinn_batch, gs.pinn_batch)
    args.map_error_frac = _pick(args.map_error_frac, gs.map_error_frac)
    args.pinn_steps = _pick(args.pinn_steps, gs.pinn_steps)
    args.measurement_loop_tol = _pick(args.measurement_loop_tol, gs.loop_closure.tol)
    args.measurement_loop_gap = _pick(args.measurement_loop_gap, gs.loop_closure.min_gap)
    args.measurement_loop_radius = _pick(
        args.measurement_loop_radius, gs.loop_closure.max_radius
    )
    args.measurement_loop_info = _pick(
        args.measurement_loop_info, gs.loop_closure.information
    )
    if args.no_loop_closure or not gs.loop_closure.enabled:
        args.measurement_loop_tol = 0.0

    sim_dir = Path(args.sim_dir if args.sim_dir is not None else cfg.simulation.sim_dir)
    csv_path = Path(args.csv_file if args.csv_file is not None else cfg.survey.csv_path)
    t_max_cap = float(args.t_max if args.t_max is not None else cfg.survey.t_max)
    dt: float = cfg.survey.dt

    VELOCITY_NOISE_STD: float = cfg.robot.v_noise_std
    OMEGA_NOISE_STD: float = cfg.robot.omega_noise_std
    OBS_NOISE_STD: float = cfg.rbpf.measurement_noise_std

    output_dir = Path(cfg.output.results_dir)
    graphs_dir = output_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)

    prng_key = jax.random.PRNGKey(args.seed)
    np.random.seed(args.seed)

    # ------------------------------------------------------------------
    # 1. Load simulation dataset (scalar fields to be mapped/measured).
    # ------------------------------------------------------------------
    print(f"\n[1/4] Loading simulation dataset from: {sim_dir}")
    if not sim_dir.exists():
        raise FileNotFoundError(f"Simulation directory '{sim_dir}' does not exist.")
    sim_data = load_simulation_dataset(sim_dir, requested_fields=cfg.simulation.fields or None)
    # Only clamp the trajectory to the simulation's duration when we actually
    # sample the simulation per step. With --use-csv-measurements the readings
    # come from the CSV, and the simulation is needed only for the domain
    # bounds and IC anchors -- clamping there silently truncates the run (e.g.
    # cutting a 662 s out-and-back at 499 s, so the return leg never completes
    # and its revisits never happen).
    sim_t_max = float(sim_data.sample_times[-1])
    if not args.use_csv_measurements and t_max_cap > sim_t_max:
        print(
            f"  NOTE: capping t_max {t_max_cap:.0f}s -> {sim_t_max:.0f}s "
            "(simulation duration; pass --use-csv-measurements to run longer)"
        )
        t_max_cap = sim_t_max
    polygon_enu = sim_data.polygon_enu
    print(f"  Loaded {len(sim_data.field_names)} simulation fields: {sim_data.field_names}")

    # Observations are fed to the PINNs zero-mean / unit-variance normalized.
    # Raw physical units are a poor training target here: e.g. salinity sits at
    # ~30.04 PPT with a spatial std of only ~0.10, so a small-init MLP spends
    # its capacity learning the large DC offset and resolves the tiny spatial
    # variation -- the only part that actually carries position information --
    # very poorly. Position observability is (map error)/(spatial gradient), so
    # map error directly sets the achievable localization accuracy.
    field_norm_std = {f: float(sim_data.field_stds[f]) for f in sim_data.field_names}
    # Measurement noise expressed in the same normalized units per field.
    obs_noise_norm = {
        f: max(OBS_NOISE_STD / field_norm_std[f], 1e-6) for f in sim_data.field_names
    }
    print(
        "  Field normalization (mean/std): "
        + ", ".join(
            f"{f}={sim_data.field_means[f]:.3f}/{field_norm_std[f]:.4f}"
            for f in sim_data.field_names
        )
    )

    # ------------------------------------------------------------------
    # 2. Ingest and interpolate real CSV survey trajectory
    # ------------------------------------------------------------------
    print(f"\n[2/4] Loading survey trajectory from: {csv_path}")
    survey_traj = load_survey_csv(
        csv_path=csv_path,
        t_max=t_max_cap,
        dt=dt,
        enu_frame=sim_data.enu_frame,
        field_mappings={f: _FIELD_TO_CSV_COLUMN.get(f, f) for f in sim_data.field_names},
    )
    if args.use_csv_measurements:
        missing = [f for f in sim_data.field_names if f not in survey_traj.measurements]
        if missing:
            raise ValueError(
                f"--use-csv-measurements: CSV {csv_path} has no column for {missing}."
            )
        print(f"  Measurement source: CSV sensor columns ({len(sim_data.field_names)} fields)")
    else:
        print("  Measurement source: simulation resampling")
    coords_true = np.array(survey_traj.coords_enu)
    headings_true = np.array(survey_traj.headings)
    velocities_nominal = np.array(survey_traj.velocities[:-1])
    omegas_nominal = np.array(survey_traj.omegas[:-1])
    sample_times = np.array(survey_traj.timestamps)
    n_steps = survey_traj.n_steps
    print(f"  Survey duration capped to: {t_max_cap:.1f} s")
    print(f"  Kinematic trajectory steps: {n_steps} (dt = {dt} s)")

    print("\n  Simulating Dead Reckoning (DR) with process noise...")
    prng_key, k_vn, k_on = jax.random.split(prng_key, 3)
    v_noise = np.array(VELOCITY_NOISE_STD * jax.random.normal(k_vn, shape=(n_steps,)))
    w_noise = np.array(OMEGA_NOISE_STD * jax.random.normal(k_on, shape=(n_steps,)))
    v_act = np.clip(velocities_nominal + v_noise, 0.0, None)
    w_act = omegas_nominal + w_noise

    x0_state = jnp.array(
        [coords_true[0, 0], coords_true[0, 1], headings_true[0]], dtype=jnp.float64
    )
    dr_integrated = DiffDriveKinematics.integrate_trajectory(
        x0=x0_state,
        velocities=jnp.asarray(v_act),
        omegas=jnp.asarray(w_act),
        dt=dt,
        include_initial=True,
    )
    coords_dr = np.array(dr_integrated[:, :2])
    dr_rmse_final = float(np.sqrt(np.mean((coords_dr - coords_true) ** 2)))
    print(f"  Dead Reckoning drift RMSE vs GPS path: {dr_rmse_final:.3f} m")

    # ------------------------------------------------------------------
    # 3. Pose graph + per-field PINN maps, seeded with IC anchors (t=0),
    #    same convention as the RBPF example -- these are domain-coverage
    #    points independent of the robot's actual (future) path, not future
    #    ground truth leaking into the estimator.
    # ------------------------------------------------------------------
    print("\n[3/4] Initializing pose graph and per-field PINN maps...")
    init_pose = np.array([coords_true[0, 0], coords_true[0, 1], headings_true[0]])
    graph_slam = GraphSlam(process_noise=np.diag([VELOCITY_NOISE_STD, OMEGA_NOISE_STD]) ** 2)
    graph_slam.initialize(init_pose)

    prng_key, k_pinn = jax.random.split(prng_key)
    field_pinn_config = PinnConfig(
        x_bounds=(sim_data.grid.x_min, sim_data.grid.x_max),
        y_bounds=(sim_data.grid.y_min, sim_data.grid.y_max),
        t_max=t_max_cap,
        hidden_dim=cfg.pinn.hidden_dim,
        num_layers=cfg.pinn.num_layers,
        learning_rate=cfg.pinn.learning_rate,
        num_steps=(args.pinn_steps if args.pinn_steps is not None else cfg.pinn.num_steps),
        batch_size=args.pinn_batch,
        num_colloc=cfg.pinn.num_colloc,
        margin=cfg.pinn.margin,
        w_pde=cfg.pinn.w_pde,
    )
    field_mapper = GraphFieldMapper(
        field_names=sim_data.field_names, pinn_config=field_pinn_config, key=k_pinn
    )

    ic_points_enu = generate_ic_anchors(
        polygon_enu, cfg.ic_anchors.n_points, seed=cfg.ic_anchors.seed, dataset=sim_data
    )
    print(f"  Seeding {len(ic_points_enu)} IC anchor points (t=0)...")
    for ic_pos in ic_points_enu:
        ic_vals = {
            f: sample_simulation_field(
                sim_data, f, 0.0, float(ic_pos[0]), float(ic_pos[1]), normalized=True
            )
            + float(np.random.normal(0.0, obs_noise_norm[f]))
            for f in sim_data.field_names
        }
        field_mapper.add_fixed_observation(np.asarray(ic_pos), 0.0, ic_vals)

    print(f"  Warm-up pre-training PINNs ({cfg.ic_anchors.epochs} epochs)...")
    warmup_losses: dict[str, float] = {}
    for _ in range(cfg.ic_anchors.epochs):
        prng_key, k_warm = jax.random.split(prng_key)
        warmup_losses = field_mapper.fit(graph_slam, key=k_warm)
    print(
        "  Warm-up losses: "
        + ", ".join(f"{k}={v:.4f}" for k, v in warmup_losses.items())
    )

    # ------------------------------------------------------------------
    # 4. Main loop: odometry + field-measurement correction, with a live
    #    plot (trajectory panel + position-error-over-time panel) showing
    #    the estimate improving as more field data is acquired.
    # ------------------------------------------------------------------
    print("\n[4/4] Running pose-graph SLAM with PINN field-measurement correction...")
    interactive = not args.no_show
    if interactive:
        plt.ion()
    fig, (ax_map, ax_err) = plt.subplots(1, 2, figsize=(14, 7))

    ax_map.plot(coords_true[:, 0], coords_true[:, 1], "k-", lw=2, label="Ground Truth (GPS)")
    ax_map.plot(coords_dr[:, 0], coords_dr[:, 1], "r--", lw=1.5, label="Dead Reckoning")
    (line_estimate,) = ax_map.plot(
        [init_pose[0]], [init_pose[1]], "b-", lw=1.5, label="Graph-SLAM (PINN-corrected)"
    )
    (marker_curr,) = ax_map.plot([init_pose[0]], [init_pose[1]], "bo", ms=6, label="Current pose")
    ax_map.scatter(
        ic_points_enu[:, 0], ic_points_enu[:, 1], c="orange", s=12, marker="x", label="IC anchors"
    )
    ax_map.set_xlabel("East [m]")
    ax_map.set_ylabel("North [m]")
    ax_map.legend(loc="best", fontsize=8)
    ax_map.set_aspect("equal", adjustable="datalim")
    ax_map.grid(True, alpha=0.3)

    (line_dr_err,) = ax_err.plot([], [], "r--", lw=1.5, label="Dead Reckoning error")
    (line_est_err,) = ax_err.plot([], [], "b-", lw=1.5, label="Graph-SLAM error")
    ax_err.set_xlabel("Step")
    ax_err.set_ylabel("Position error vs GPS [m]")
    ax_err.legend(loc="best")
    ax_err.grid(True, alpha=0.3)

    checkpoints_dir = graphs_dir / "graph_slam_pinn_checkpoints"
    dr_err_hist: list[float] = []
    est_err_hist: list[float] = []
    field_losses: dict[str, float] = dict(warmup_losses)
    last_cost = 0.0
    n_meas_closures = 0

    def redraw(step_num: int) -> None:
        """Refresh both panels with the current estimate and error history."""
        traj = graph_slam.get_trajectory()
        line_estimate.set_data(traj[:, 0], traj[:, 1])
        marker_curr.set_data([traj[-1, 0]], [traj[-1, 1]])

        steps = np.arange(1, len(dr_err_hist) + 1)
        line_dr_err.set_data(steps, dr_err_hist)
        line_est_err.set_data(steps, est_err_hist)
        ax_err.relim()
        ax_err.autoscale_view()

        loss_str = ", ".join(f"{k}={v:.3f}" for k, v in field_losses.items())
        fig.suptitle(
            f"Field-Measurement Pose-Graph SLAM (Step {step_num}/{n_steps})\n"
            f"cost={last_cost:.2e}  |  field losses: {loss_str}"
        )
        ax_map.relim()
        ax_map.autoscale_view()

        if interactive:
            fig.canvas.draw_idle()
            plt.pause(0.001)
        else:
            checkpoints_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                checkpoints_dir / f"step_{step_num:04d}.png", dpi=100, bbox_inches="tight"
            )

    pbar = tqdm(range(n_steps), desc="  Graph-SLAM+PINN Survey", unit="step", dynamic_ncols=True)
    for step_idx in pbar:
        t_curr = float(sample_times[step_idx + 1])
        true_pos = coords_true[step_idx + 1]
        ctrl = np.array([v_act[step_idx], w_act[step_idx]])
        node_idx = graph_slam.predict(control=ctrl, dt=dt)

        if args.use_csv_measurements:
            # CSV values are raw physical units; normalize with the same
            # per-field statistics used for the simulation-sampled path so both
            # sources land in identical units.
            obs_vals = {
                f: (
                    float(survey_traj.measurements[f][step_idx + 1])
                    - float(sim_data.field_means[f])
                )
                / field_norm_std[f]
                for f in sim_data.field_names
            }
        else:
            obs_vals = {
                f: sample_simulation_field(
                    sim_data, f, t_curr, float(true_pos[0]), float(true_pos[1]), normalized=True
                )
                + float(np.random.normal(0.0, obs_noise_norm[f]))
                for f in sim_data.field_names
            }
        field_mapper.add_observation(node_idx, t_curr, obs_vals)
        # Keep the scalar readings on the graph node itself, so nodes can be
        # associated by what was measured there rather than by where the
        # (drifting) estimate currently places them.
        graph_slam.set_node_measurement(
            node_idx, np.array([obs_vals[f] for f in sim_data.field_names])
        )
        if args.measurement_loop_tol > 0.0:
            newly = graph_slam.detect_measurement_loop_closures(
                current_idx=node_idx,
                tol=args.measurement_loop_tol,
                min_gap=args.measurement_loop_gap,
                max_radius=args.measurement_loop_radius,
                max_matches=1,
                information=args.measurement_loop_info,
            )
            n_meas_closures += len(newly)

        step_num = step_idx + 1

        # Correct this node BEFORE refitting, so the map used here is the one
        # from the previous fit -- which has never seen this node's own
        # reading. Correcting after the fit leaks: the map memorizes the new
        # observation at the node's (drifted) position, so it predicts the
        # observed value there exactly, the residual collapses to ~0, and the
        # correction says "you're already right" no matter how far off you are.
        field_mapper.apply_field_correction(
            node_idx,
            graph_slam,
            measurement_noise_std=obs_noise_norm,
            field_losses=field_losses,
            max_correction=args.max_correction,
            map_error_frac=args.map_error_frac,
        )

        if step_num % args.fit_interval == 0 or step_num == n_steps:
            prng_key, k_fit = jax.random.split(prng_key)
            field_losses = field_mapper.fit(graph_slam, key=k_fit)

        # Re-linearize recent nodes' factors against the *current* (better
        # trained) maps and their updated position estimates. Without this,
        # every node keeps the one-shot correction computed when it was
        # created -- i.e. from the least-trained map it ever saw -- so the
        # maps improving with more data never propagates back to earlier
        # poses. Bounded to a trailing window to keep the cost flat.
        if args.refresh_window > 0 and step_num % args.optimize_interval == 0:
            for refresh_idx in range(max(1, node_idx - args.refresh_window + 1), node_idx):
                field_mapper.apply_field_correction(
                    refresh_idx,
                    graph_slam,
                    measurement_noise_std=obs_noise_norm,
                    field_losses=field_losses,
                    max_correction=args.max_correction,
                    map_error_frac=args.map_error_frac,
                )

        did_optimize = False
        if step_num % args.optimize_interval == 0 or step_num == n_steps:
            last_cost = graph_slam.optimize(max_iters=5)
            did_optimize = True

        est_pos = graph_slam.nodes[node_idx, :2]
        dr_err_hist.append(float(np.linalg.norm(coords_dr[step_num] - true_pos)))
        est_err_hist.append(float(np.linalg.norm(est_pos - true_pos)))

        if did_optimize:
            redraw(step_num)

        pbar.set_postfix(
            pos_err=f"{est_err_hist[-1]:.2f}m",
            mean_loss=f"{np.mean(list(field_losses.values())):.4f}",
            cost=f"{last_cost:.2e}",
        )

    estimated_traj = graph_slam.get_trajectory()

    # ------------------------------------------------------------------
    # Metrics and final report
    # ------------------------------------------------------------------
    est_rmse = float(np.sqrt(np.mean((estimated_traj - coords_true) ** 2)))
    err_red = ((dr_rmse_final - est_rmse) / dr_rmse_final) * 100.0

    print("\n--------------------------------------------------")
    print(f"1. Dead Reckoning Trajectory RMSE:       {dr_rmse_final:.3f} m")
    print(f"2. Graph-SLAM (PINN-corrected) RMSE:      {est_rmse:.3f} m")
    print(f"   Final optimization cost:              {last_cost:.4e}")
    print(f"   Measurement-matched loop closures:    {n_meas_closures}")
    print(f"SLAM Error Reduction vs Dead Reckoning:  {err_red:.1f}%")
    print(f"   Per-field PINN networks:              {len(field_mapper.field_maps)}")
    for name, fm in field_mapper.field_maps.items():
        v_est = np.array(fm.params.v_flow)
        d_est = np.array(fm.D)
        print(
            f"     {name:>12s}: loss={field_losses.get(name, float('nan')):.4f}, "
            f"v_flow=[{v_est[0]:.3f}, {v_est[1]:.3f}] m/s, D={d_est[0]:.4f} m^2/s"
        )
    print("--------------------------------------------------")

    redraw(n_steps)
    fig_path = graphs_dir / "graph_slam_pinn_survey_trajectory_comparison.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\n  Saved final trajectory figure to: {fig_path}")
    if not interactive and checkpoints_dir.exists():
        print(f"  Saved per-optimization snapshots to: {checkpoints_dir}")

    if interactive:
        plt.ioff()
        plt.show()
    else:
        plt.close(fig)

    print("\nExperiment run completed successfully!")
    print("==================================================")


if __name__ == "__main__":
    main()
