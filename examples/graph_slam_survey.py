"""
examples/graph_slam_survey.py
==============================
Pose-graph SLAM (GraphSlam) with loop-closure detection and Gauss-Newton
optimization, driven by a real field survey CSV trajectory (Latitude,
Longitude, Date/Time) interpolated via differential drive kinematics.

Dead reckoning is simulated by corrupting nominal kinematic controls with
calibrated process noise (same convention as examples/rbpf_slam_survey.py),
and periodically checked for revisits to close loops and correct drift.

Unlike the RBPF example (one shared multi-output PINN), scalar-field mapping
here (enabled by default; disable with --no-field-map) uses a
GraphFieldMapper: one independent single-output PINN per field, each tied to
the pose graph by node index rather than a baked-in coordinate. Every time a
loop closure corrects past nodes, refitting automatically trains against the
corrected positions -- there is no separate relabeling step.

Run (interactive)::

    uv run python examples/graph_slam_survey.py \
        --config configs/biscayne_survey_rbpf.yaml

Run (headless)::

    uv run python examples/graph_slam_survey.py \
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
from pde_slam.io import load_simulation_dataset, load_survey_csv, sample_simulation_field
from pde_slam.kinematics import DiffDriveKinematics
from pde_slam.pinn import PinnConfig
from pde_slam.slam import GraphFieldMapper, GraphSlam


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
            "Pose-graph SLAM (odometry + loop closure) on real CSV survey "
            "trajectory with kinematic DR simulation."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/biscayne_survey_rbpf.yaml"),
        help="Path to YAML experiment config (only survey/robot/output used).",
    )
    parser.add_argument(
        "--csv-file",
        type=str,
        default=None,
        help="Optional override for survey CSV file path.",
    )
    parser.add_argument(
        "--t-max",
        type=float,
        default=None,
        help="Optional duration cap [s] (defaults to config's survey.t_max).",
    )
    parser.add_argument(
        "--loop-closure-radius",
        type=float,
        default=3.0,
        help="Max distance [m] between current and a past node to close a loop.",
    )
    parser.add_argument(
        "--min-loop-gap",
        type=int,
        default=30,
        help="Minimum node-index separation before a loop closure is proposed.",
    )
    parser.add_argument(
        "--check-interval",
        type=int,
        default=10,
        help="Run loop-closure detection + graph optimization every N steps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=43,
        help="PRNG seed for dead-reckoning process noise.",
    )
    parser.add_argument(
        "--sim-dir",
        type=str,
        default=None,
        help="Optional override for hydrodynamic simulation directory (field maps).",
    )
    parser.add_argument(
        "--no-field-map",
        action="store_true",
        help=(
            "Disable multi-field PINN mapping (GraphFieldMapper); track pose "
            "only, same as the original pose-only GraphSlam demo."
        ),
    )
    parser.add_argument(
        "--pinn-fit-interval",
        type=int,
        default=10,
        help="Refit every field's PINN every N steps (in addition to after each "
        "loop closure).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Suppress interactive plot window; only save the figure to disk.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run pose-graph SLAM with loop closure on a real survey CSV trajectory.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments forwarded to :func:`parse_args`.
    """
    args = parse_args(argv)

    print("==================================================")
    print("Pose-Graph SLAM (GraphSlam) on Real Field Survey Trajectory")
    print("==================================================")

    cfg = load_rbpf_experiment_config(args.config)

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
    use_field_map = not args.no_field_map

    # ------------------------------------------------------------------
    # 0. (Optional) Load hydrodynamic simulation dataset -- only needed for
    #    the per-field PINN maps, so the survey CSV is anchored to its ENU
    #    frame (matches examples/rbpf_slam_survey.py's convention).
    # ------------------------------------------------------------------
    sim_data = None
    enu_frame = None
    if use_field_map:
        sim_dir = Path(args.sim_dir if args.sim_dir is not None else cfg.simulation.sim_dir)
        print(f"\n[0/3] Loading simulation dataset from: {sim_dir}")
        if not sim_dir.exists():
            raise FileNotFoundError(f"Simulation directory '{sim_dir}' does not exist.")
        sim_data = load_simulation_dataset(
            sim_dir, requested_fields=cfg.simulation.fields or None
        )
        enu_frame = sim_data.enu_frame
        t_max_cap = min(t_max_cap, float(sim_data.sample_times[-1]))

        print(f"  Loaded {len(sim_data.field_names)} simulation fields: {sim_data.field_names}")
        print("  Normalizing simulation fields to zero-mean and unit-variance...")
        for f_name in sim_data.field_names:
            m = sim_data.field_means[f_name]
            s = sim_data.field_stds[f_name]
            sim_data.simulations[f_name]["solutions"] = (
                sim_data.simulations[f_name]["solutions"] - m
            ) / s

    # ------------------------------------------------------------------
    # 1. Ingest and interpolate real CSV survey trajectory
    # ------------------------------------------------------------------
    print(f"\n[1/3] Loading survey trajectory from: {csv_path}")
    survey_traj = load_survey_csv(
        csv_path=csv_path, t_max=t_max_cap, dt=dt, enu_frame=enu_frame
    )

    coords_true = np.array(survey_traj.coords_enu)
    headings_true = np.array(survey_traj.headings)
    velocities_nominal = np.array(survey_traj.velocities[:-1])
    omegas_nominal = np.array(survey_traj.omegas[:-1])
    sample_times = np.array(survey_traj.timestamps)
    n_steps = survey_traj.n_steps

    print(f"  Survey duration capped to: {t_max_cap:.1f} s")
    print(f"  Kinematic trajectory steps: {n_steps} (dt = {dt} s)")

    # ------------------------------------------------------------------
    # 2. Dead Reckoning simulation with noise (same convention as the RBPF
    #    survey example) -- used both to drive GraphSlam and as a DR-only
    #    baseline for comparison.
    # ------------------------------------------------------------------
    print("\n[2/3] Simulating Dead Reckoning (DR) with process noise...")
    prng_key, k_vn, k_on = jax.random.split(prng_key, 3)
    v_noise = np.array(VELOCITY_NOISE_STD * jax.random.normal(k_vn, shape=(n_steps,)))
    w_noise = np.array(OMEGA_NOISE_STD * jax.random.normal(k_on, shape=(n_steps,)))
    v_act = np.clip(velocities_nominal + v_noise, 0.0, None)
    w_act = omegas_nominal + w_noise

    x0_state = jnp.array(
        [coords_true[0, 0], coords_true[0, 1], headings_true[0]],
        dtype=jnp.float64,
    )
    dr_integrated = DiffDriveKinematics.integrate_trajectory(
        x0=x0_state,
        velocities=jnp.asarray(v_act),
        omegas=jnp.asarray(w_act),
        dt=dt,
        include_initial=True,
    )
    coords_dr = np.array(dr_integrated[:, :2])
    dr_rmse = float(np.sqrt(np.mean((coords_dr - coords_true) ** 2)))
    print(
        f"  Process noise: sigma_v = {VELOCITY_NOISE_STD:.4f} m/s, "
        f"sigma_omega = {OMEGA_NOISE_STD:.4f} rad/s"
    )
    print(f"  Dead Reckoning drift RMSE vs GPS path: {dr_rmse:.3f} m")

    # ------------------------------------------------------------------
    # 3. Pose-graph SLAM: odometry tracking + periodic loop-closure
    #    detection and optimization, with a live plot that redraws the
    #    estimated trajectory every time a loop closure actually corrects it.
    # ------------------------------------------------------------------
    print("\n[3/3] Running pose-graph SLAM with loop closure...")
    graph_slam = GraphSlam(
        process_noise=np.diag([VELOCITY_NOISE_STD, OMEGA_NOISE_STD]) ** 2,
        loop_closure_radius=args.loop_closure_radius,
        min_loop_gap=args.min_loop_gap,
    )
    init_pose = np.array([coords_true[0, 0], coords_true[0, 1], headings_true[0]])
    graph_slam.initialize(init_pose)

    field_mapper: GraphFieldMapper | None = None
    if use_field_map:
        assert sim_data is not None
        prng_key, k_pinn = jax.random.split(prng_key)
        field_pinn_config = PinnConfig(
            x_bounds=(sim_data.grid.x_min, sim_data.grid.x_max),
            y_bounds=(sim_data.grid.y_min, sim_data.grid.y_max),
            t_max=t_max_cap,
            hidden_dim=cfg.pinn.hidden_dim,
            num_layers=cfg.pinn.num_layers,
            learning_rate=cfg.pinn.learning_rate,
            num_steps=cfg.pinn.num_steps,
            num_colloc=cfg.pinn.num_colloc,
            margin=cfg.pinn.margin,
            w_pde=cfg.pinn.w_pde,
        )
        field_mapper = GraphFieldMapper(
            field_names=sim_data.field_names, pinn_config=field_pinn_config, key=k_pinn
        )
        # Root node (t=0) observation, same as the RBPF example's IC anchor.
        root_vals = {
            f: sample_simulation_field(
                sim_data, f, 0.0, float(init_pose[0]), float(init_pose[1]), normalized=False
            )
            + float(np.random.normal(0.0, OBS_NOISE_STD))
            for f in sim_data.field_names
        }
        field_mapper.add_observation(0, 0.0, root_vals)
        print(
            f"  Multi-field PINN mapping ENABLED: one network per field "
            f"({', '.join(sim_data.field_names)})"
        )
    else:
        print("  Multi-field PINN mapping disabled (--no-field-map); pose only.")

    interactive = not args.no_show
    if interactive:
        plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(coords_true[:, 0], coords_true[:, 1], "k-", lw=2, label="Ground Truth (GPS)")
    ax.plot(coords_dr[:, 0], coords_dr[:, 1], "r--", lw=1.5, label="Dead Reckoning")
    (line_graph_slam,) = ax.plot(
        [init_pose[0]], [init_pose[1]], "b-", lw=1.5, label="Graph-SLAM (optimized)"
    )
    (marker_curr,) = ax.plot(
        [init_pose[0]], [init_pose[1]], "bo", ms=6, label="Current pose"
    )
    loop_closure_lines: list[plt.Line2D] = []
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.legend(loc="best")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)

    n_loop_closures_total = 0
    last_cost = 0.0
    field_losses: dict[str, float] = {}
    checkpoints_dir = graphs_dir / "graph_slam_checkpoints"

    def redraw(step_num: int, newly_closed: list[int]) -> None:
        """Refresh the plot with the current (possibly just-corrected)
        estimated trajectory and draw any newly found loop-closure edges.
        """
        traj = graph_slam.get_trajectory()
        line_graph_slam.set_data(traj[:, 0], traj[:, 1])
        marker_curr.set_data([traj[-1, 0]], [traj[-1, 1]])

        for j in newly_closed:
            (edge_line,) = ax.plot(
                [traj[j, 0], traj[-1, 0]],
                [traj[j, 1], traj[-1, 1]],
                color="green",
                linestyle=":",
                linewidth=0.8,
                alpha=0.6,
            )
            loop_closure_lines.append(edge_line)

        running_rmse = float(
            np.sqrt(np.mean((traj - coords_true[: step_num + 1]) ** 2))
        )
        loss_str = ""
        if field_losses:
            parts = ", ".join(f"{k}={v:.4f}" for k, v in field_losses.items())
            loss_str = f", field losses: {parts}"
        ax.set_title(
            f"Pose-Graph SLAM with Loop Closure (Step {step_num}/{n_steps})\n"
            f"RMSE so far={running_rmse:.2f}m, cost={last_cost:.2e}, "
            f"{n_loop_closures_total} closures{loss_str}"
        )
        ax.relim()
        ax.autoscale_view()

        if interactive:
            fig.canvas.draw_idle()
            plt.pause(0.001)
        else:
            checkpoints_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                checkpoints_dir / f"step_{step_num:04d}.png", dpi=100, bbox_inches="tight"
            )

    pbar = tqdm(
        range(n_steps), desc="  Graph-SLAM Survey", unit="step", dynamic_ncols=True
    )
    for step_idx in pbar:
        t_curr = float(sample_times[step_idx + 1])
        ctrl = np.array([v_act[step_idx], w_act[step_idx]])
        node_idx = graph_slam.predict(control=ctrl, dt=dt)

        if field_mapper is not None:
            assert sim_data is not None
            true_pos = coords_true[step_idx + 1]
            obs_vals = {
                f: sample_simulation_field(
                    sim_data, f, t_curr, float(true_pos[0]), float(true_pos[1]), normalized=False
                )
                + float(np.random.normal(0.0, OBS_NOISE_STD))
                for f in sim_data.field_names
            }
            field_mapper.add_observation(node_idx, t_curr, obs_vals)

        step_num = step_idx + 1
        did_close_loop = False
        if step_num % args.check_interval == 0 or step_num == n_steps:
            closed = graph_slam.detect_loop_closures()
            if closed:
                n_loop_closures_total += len(closed)
                last_cost = graph_slam.optimize()
                did_close_loop = True

        if field_mapper is not None and (
            did_close_loop
            or step_num % args.pinn_fit_interval == 0
            or step_num == n_steps
        ):
            # Refitting here after a loop closure is what makes the maps
            # consistent with the just-corrected geometry: GraphFieldMapper
            # looks up each observation's position live from graph_slam.nodes,
            # so this call automatically trains against the corrected
            # positions with no separate relabeling step.
            prng_key, k_fit = jax.random.split(prng_key)
            field_losses = field_mapper.fit(graph_slam, key=k_fit)

        if did_close_loop:
            redraw(step_num, closed)

        postfix = {"loop_closures": n_loop_closures_total, "cost": f"{last_cost:.2e}"}
        if field_losses:
            postfix["mean_field_loss"] = f"{np.mean(list(field_losses.values())):.4f}"
        pbar.set_postfix(**postfix)

    estimated_traj = graph_slam.get_trajectory()

    # ------------------------------------------------------------------
    # Metrics and final report
    # ------------------------------------------------------------------
    graph_slam_rmse = float(np.sqrt(np.mean((estimated_traj - coords_true) ** 2)))
    err_red = ((dr_rmse - graph_slam_rmse) / dr_rmse) * 100.0

    print("\n--------------------------------------------------")
    print(f"1. Dead Reckoning Trajectory RMSE:      {dr_rmse:.3f} m")
    print(f"2. Graph-SLAM (loop closure) RMSE:       {graph_slam_rmse:.3f} m")
    print(f"   Total loop closures found:            {n_loop_closures_total}")
    print(f"   Final optimization cost:              {last_cost:.4e}")
    print(f"SLAM Error Reduction vs Dead Reckoning:  {err_red:.1f}%")
    if field_mapper is not None:
        print(f"   Per-field PINN networks:              {len(field_mapper.field_maps)}")
        for name, fm in field_mapper.field_maps.items():
            v_est = np.array(fm.params.v_flow)
            d_est = np.array(fm.D)
            print(
                f"     {name:>12s}: loss={field_losses.get(name, float('nan')):.4f}, "
                f"v_flow=[{v_est[0]:.3f}, {v_est[1]:.3f}] m/s, D={d_est[0]:.4f} m^2/s"
            )
    print("--------------------------------------------------")

    redraw(n_steps, [])
    fig_path = graphs_dir / "graph_slam_survey_trajectory_comparison.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\n  Saved final trajectory figure to: {fig_path}")
    if not interactive and checkpoints_dir.exists():
        print(f"  Saved per-loop-closure snapshots to: {checkpoints_dir}")

    if interactive:
        plt.ioff()
        plt.show()
    else:
        plt.close(fig)

    print("\nExperiment run completed successfully!")
    print("==================================================")


if __name__ == "__main__":
    main()
