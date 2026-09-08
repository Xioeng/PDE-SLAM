"""
examples/graph_slam_out_and_back_survey.py
=============================================
Synthetic out-and-back survey path for stress-testing GraphSlam + PINN
field-measurement correction (see examples/graph_slam_pinn_survey.py; no
RBPF is used here either).

The real survey CSV's outbound zigzag leg (from the start out to its point
of maximum distance) is kept as-is; whatever path the boat actually took
back (straight line, different zigzag, doesn't matter) is pruned and
replaced with an exact mirror of the outbound leg, so the robot returns to
its initial position by retracing the same zigzag pattern in reverse.
Because the return leg revisits the exact same physical points as the
outbound leg, its field measurements are taken to be the outbound leg's own
recorded values played back in reverse order -- not freshly resampled from
the (time-evolving) simulation -- giving a controlled, unambiguous test case
for whether the field-measurement correction actually recognizes and
exploits the revisit.

Run (interactive)::

    uv run python examples/graph_slam_out_and_back_survey.py \
        --config configs/biscayne_survey_rbpf.yaml

Run (headless)::

    uv run python examples/graph_slam_out_and_back_survey.py \
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
import pandas as pd
from tqdm import tqdm

from pde_slam.config import load_rbpf_experiment_config
from pde_slam.coords import ENUFrame
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


def wrap_angle(theta: np.ndarray) -> np.ndarray:
    """Wraps angle(s) to ``(-pi, pi]``."""
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


def build_out_and_back_path(
    coords: np.ndarray, turnaround_idx: int | None = None, turn_dwell: int = 8
) -> tuple[np.ndarray, np.ndarray, int]:
    """Prunes whatever return leg a recorded path took and replaces it with
    an exact mirror of its outbound leg.

    The turnaround point defaults to the recorded path's point of maximum
    distance from the start -- everything after that (however the vehicle
    actually got back) is discarded and replaced by retracing the outbound
    leg (0..turnaround) in reverse, so the synthetic path returns exactly to
    its starting position.

    Parameters
    ----------
    coords : np.ndarray
        Recorded ``(x, y)`` positions, shape ``(N, 2)``.
    turnaround_idx : int or None, default=None
        Index to treat as the turnaround point; defaults to
        ``argmax(||coords - coords[0]||)``.

    Returns
    -------
    new_coords : np.ndarray
        Out-and-back path, shape ``(2*turnaround_idx + 1, 2)``.
    new_headings : np.ndarray
        Motion heading at each point (direction of travel into it, last
        point repeats the final motion heading), same length as
        ``new_coords``.
    turnaround_idx : int
        The turnaround index used (echoed back for reuse by the caller,
        e.g. to slice the matching outbound measurement array).
    """
    if turnaround_idx is None:
        dist_from_start = np.linalg.norm(coords - coords[0], axis=1)
        turnaround_idx = int(np.argmax(dist_from_start))

    outbound = coords[: turnaround_idx + 1]
    return_leg = outbound[-2::-1]  # reverse, excluding the turnaround point (already included once)

    # Dwell at the apex before reversing. The consumer of this path
    # (load_survey_csv -> interpolate_kinematic_trajectory) refits a
    # kinematically feasible differential-drive path through these points, and
    # an instantaneous 180-degree reversal is not feasible: the smoother cuts
    # the corner, the path loses length, and the vehicle ends up short of its
    # start. Holding position for a few steps gives it time to turn in place.
    if turn_dwell > 0:
        apex = np.repeat(outbound[-1][None, :], turn_dwell, axis=0)
        new_coords = np.concatenate([outbound, apex, return_leg], axis=0)
    else:
        new_coords = np.concatenate([outbound, return_leg], axis=0)

    deltas = np.diff(new_coords, axis=0)
    motion_headings = np.arctan2(deltas[:, 1], deltas[:, 0])
    # Stationary dwell steps have a zero delta, whose atan2 is a meaningless 0;
    # carry the previous heading forward through them instead.
    # Stationary dwell steps have a zero delta, whose atan2 is a meaningless 0.
    # Ramp the heading across them instead of carrying it forward: the loader
    # feeds this Heading column to the kinematic refit, and an instantaneous
    # 180-degree flip there is not realizable, so the refit cuts the corner and
    # the vehicle lands short of its start. A gradual turn-in-place is.
    moving = np.linalg.norm(deltas, axis=1) > 1e-9
    idx = np.arange(len(motion_headings))
    stationary = idx[~moving]
    if stationary.size:
        before = stationary[0] - 1
        after = stationary[-1] + 1
        h_start = motion_headings[before] if before >= 0 else motion_headings[after]
        h_end = motion_headings[after] if after < len(motion_headings) else h_start
        # Shortest-arc interpolation across the reversal.
        delta_h = (h_end - h_start + np.pi) % (2.0 * np.pi) - np.pi
        for k, i in enumerate(stationary, start=1):
            motion_headings[i] = h_start + delta_h * k / (stationary.size + 1)
    new_headings = np.concatenate([motion_headings, motion_headings[-1:]])

    return new_coords, new_headings, turnaround_idx


def save_out_and_back_csv(
    out_path: Path,
    coords: np.ndarray,
    headings: np.ndarray,
    sample_times: np.ndarray,
    obs_vals_all: list[dict[str, float] | None],
    enu_frame: ENUFrame,
    base_date: int,
    base_time: int,
) -> None:
    """Writes the out-and-back path to a CSV in the same schema
    :func:`~pde_slam.io.load_survey_csv` expects (Date, Time, Latitude,
    Longitude, sensor columns), so it's a drop-in survey file for
    ``examples/rbpf_slam_survey.py`` or any other consumer of that loader.

    Parameters
    ----------
    out_path : Path
        Destination CSV path.
    coords : np.ndarray
        Out-and-back ``(x, y)`` ENU positions, shape ``(N, 2)``.
    headings : np.ndarray
        Motion heading at each point, shape ``(N,)``.
    sample_times : np.ndarray
        Elapsed time [s] at each point, shape ``(N,)``.
    obs_vals_all : list of (dict of str to float) or None
        Per-step field observations (length ``N - 1``, one per motion step);
        ``None`` for the one step with no available measurement (see
        :func:`build_out_and_back_path`'s caller).
    enu_frame : ENUFrame
        Frame used to convert back to geodetic (Latitude/Longitude).
    base_date : int
        Start date as ``YYYYMMDD``, matching the original CSV's convention.
    base_time : int
        Start time as ``HHMMSS``, matching the original CSV's convention.
    """
    n = len(coords)
    geo = enu_frame.enu_to_geodetic(coords)

    base_dt = pd.to_datetime(f"{base_date}{base_time:06d}", format="%Y%m%d%H%M%S")
    timestamps = base_dt + pd.to_timedelta(sample_times, unit="s")

    data: dict[str, np.ndarray] = {
        "Date": np.asarray(timestamps.strftime("%Y%m%d"), dtype=np.int64),
        "Time": np.asarray(timestamps.strftime("%H%M%S"), dtype=np.int64),
        "Latitude": geo[:, 0],
        "Longitude": geo[:, 1],
        "Heading (degrees Magnetic)": np.degrees(headings) % 360.0,
    }

    field_names = sorted({k for entry in obs_vals_all if entry is not None for k in entry})
    for f in field_names:
        col = _FIELD_TO_CSV_COLUMN.get(f, f)
        vals = np.full(n, np.nan)
        for i, entry in enumerate(obs_vals_all):
            if entry is not None:
                vals[i + 1] = entry.get(f, np.nan)
        data[col] = vals

    pd.DataFrame(data).to_csv(out_path, index=False)


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
            "Out-and-back synthetic survey (pruned real return leg, mirrored "
            "zigzag retrace, reversed measurements) for testing GraphSlam + "
            "PINN field-measurement correction."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=Path("configs/biscayne_survey_rbpf.yaml"))
    parser.add_argument("--sim-dir", type=str, default=None)
    parser.add_argument("--csv-file", type=str, default=None)
    parser.add_argument(
        "--outbound-t-max",
        type=float,
        default=None,
        help="Cap [s] on the recorded path before finding the turnaround point "
        "(defaults to config's survey.t_max).",
    )
    parser.add_argument(
        "--turnaround-step",
        type=int,
        default=None,
        help="Explicit turnaround step index; defaults to the point of maximum "
        "distance from the start.",
    )
    parser.add_argument("--fit-interval", type=int, default=1)
    parser.add_argument("--optimize-interval", type=int, default=5)
    parser.add_argument("--max-correction", type=float, default=2.0)
    parser.add_argument(
        "--known-loop-closures",
        action="store_true",
        default=True,
        help="Add exact loop-closure edges between each return-leg node and its "
        "known outbound mirror (this path's revisit correspondence is exact by "
        "construction, unlike a real deployment's detected ones).",
    )
    parser.add_argument(
        "--no-known-loop-closures",
        dest="known_loop_closures",
        action="store_false",
        help="Disable known-correspondence loop closures; use field-measurement "
        "correction only.",
    )
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument(
        "--save-csv",
        type=Path,
        default=None,
        help="Path to save the out-and-back path as a survey CSV usable by "
        "examples/rbpf_slam_survey.py (default: "
        "data/csv/<original csv stem>_out_and_back.csv).",
    )
    parser.add_argument(
        "--no-save-csv", action="store_true", help="Skip saving the out-and-back CSV."
    )
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run field-measurement pose-graph SLAM on a synthetic out-and-back path.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments forwarded to :func:`parse_args`.
    """
    args = parse_args(argv)

    print("==================================================")
    print("Out-and-Back Graph-SLAM + PINN Field-Measurement Test (no RBPF)")
    print("==================================================")

    cfg = load_rbpf_experiment_config(args.config)

    sim_dir = Path(args.sim_dir if args.sim_dir is not None else cfg.simulation.sim_dir)
    csv_path = Path(args.csv_file if args.csv_file is not None else cfg.survey.csv_path)
    outbound_t_max_cap = float(
        args.outbound_t_max if args.outbound_t_max is not None else cfg.survey.t_max
    )
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
    # 1. Load simulation dataset + raw survey CSV, then build the synthetic
    #    out-and-back path from it.
    # ------------------------------------------------------------------
    print(f"\n[1/4] Loading simulation dataset from: {sim_dir}")
    if not sim_dir.exists():
        raise FileNotFoundError(f"Simulation directory '{sim_dir}' does not exist.")
    sim_data = load_simulation_dataset(sim_dir, requested_fields=cfg.simulation.fields or None)
    outbound_t_max_cap = min(outbound_t_max_cap, float(sim_data.sample_times[-1]))
    polygon_enu = sim_data.polygon_enu

    # Same normalization the single-pass script uses: raw units make a poor
    # PINN target (salinity ~30.04 with a spatial std of only ~0.10), so the
    # network spends its capacity on the DC offset rather than on the spatial
    # structure that actually carries position information.
    field_norm_std = {f: float(sim_data.field_stds[f]) for f in sim_data.field_names}
    obs_noise_norm = {
        f: max(OBS_NOISE_STD / field_norm_std[f], 1e-6) for f in sim_data.field_names
    }

    print(f"\n  Loading raw survey trajectory from: {csv_path}")
    survey_traj = load_survey_csv(
        csv_path=csv_path, t_max=outbound_t_max_cap, dt=dt, enu_frame=sim_data.enu_frame
    )
    raw_coords = np.array(survey_traj.coords_enu)

    coords_true, headings_true, turnaround_idx = build_out_and_back_path(
        raw_coords, turnaround_idx=args.turnaround_step
    )
    n_steps = len(coords_true) - 1
    print(
        f"  Recorded path had {len(raw_coords)} points capped to "
        f"{outbound_t_max_cap:.0f}s; pruning return leg after turnaround step "
        f"{turnaround_idx} (max dist {np.linalg.norm(raw_coords[turnaround_idx] - raw_coords[0]):.1f}m "
        f"from start)."
    )
    print(
        f"  Out-and-back path: {len(coords_true)} points / {n_steps} steps "
        f"(outbound {turnaround_idx}, return {n_steps - turnaround_idx})."
    )
    print(
        f"  Start-to-end distance: "
        f"{np.linalg.norm(coords_true[-1] - coords_true[0]):.3f} m (should be ~0)."
    )

    deltas = np.diff(coords_true, axis=0)
    velocities_nominal = np.linalg.norm(deltas, axis=1) / dt
    omegas_nominal = wrap_angle(np.diff(headings_true)) / dt
    sample_times = np.arange(len(coords_true)) * dt

    print("\n[2/4] Simulating Dead Reckoning (DR) with process noise...")
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
    dr_return_err = float(np.linalg.norm(coords_dr[-1] - coords_true[0]))
    print(f"  Dead Reckoning drift RMSE vs true path: {dr_rmse_final:.3f} m")
    print(f"  Dead Reckoning final return-to-start error: {dr_return_err:.3f} m")

    # ------------------------------------------------------------------
    # Ground-truth field measurements: sampled fresh for the outbound leg;
    # the return leg reuses the outbound leg's own recorded values in
    # reverse order (same physical points, so this is what a perfectly
    # static/reproducible field would give anyway -- here made exact and
    # explicit per the test's design, rather than resampling the
    # time-evolving simulation at each return step's later true time).
    # ------------------------------------------------------------------
    print("\n  Sampling ground-truth field measurements (outbound fresh, return leg mirrored)...")
    obs_vals_outbound: list[dict[str, float]] = []
    for step in range(turnaround_idx):
        t_curr = float(sample_times[step + 1])
        pos = coords_true[step + 1]
        vals = {
            f: sample_simulation_field(
                sim_data, f, t_curr, float(pos[0]), float(pos[1]), normalized=True
            )
            + float(np.random.normal(0.0, obs_noise_norm[f]))
            for f in sim_data.field_names
        }
        obs_vals_outbound.append(vals)

    # Return leg (steps turnaround_idx .. n_steps-1) revisits outbound points
    # turnaround_idx-1, turnaround_idx-2, ..., 1 in order -> reuse those exact
    # recorded values, reversed. We only have forward measurements, so the
    # return leg's observations are strictly the forward leg's own recorded
    # values played back in reverse -- nothing is freshly sampled. The one
    # exception is unavoidable, not a choice: the very last return step
    # re-arrives at the original start (point 0), and the outbound loop above
    # never measured point 0 itself (it only records steps 1..turnaround_idx,
    # i.e. points 1..turnaround_idx) -- there is no forward measurement for
    # that point to reverse, so that final step simply gets no field
    # observation at all (odometry-only), rather than fabricating one.
    obs_vals_return: list[dict[str, float] | None] = (
        list(reversed(obs_vals_outbound[:-1])) if turnaround_idx > 0 else []
    )
    obs_vals_return.append(None)
    obs_vals_all: list[dict[str, float] | None] = obs_vals_outbound + obs_vals_return
    assert len(obs_vals_all) == n_steps, (len(obs_vals_all), n_steps)

    if not args.no_save_csv:
        save_csv_path = (
            args.save_csv
            if args.save_csv is not None
            else Path("data/csv") / f"{csv_path.stem}_out_and_back.csv"
        )
        raw_first_row = pd.read_csv(csv_path, nrows=1)
        save_out_and_back_csv(
            save_csv_path,
            coords_true,
            headings_true,
            sample_times,
            obs_vals_all,
            sim_data.enu_frame,
            base_date=int(raw_first_row["Date"].iloc[0]),
            base_time=int(raw_first_row["Time"].iloc[0]),
        )
        print(f"  Saved out-and-back path as survey CSV to: {save_csv_path}")

    # ------------------------------------------------------------------
    # 3. Pose graph + per-field PINN maps, seeded with IC anchors (t=0).
    # ------------------------------------------------------------------
    print("\n[3/4] Initializing pose graph and per-field PINN maps...")
    init_pose = np.array([coords_true[0, 0], coords_true[0, 1], headings_true[0]])
    graph_slam = GraphSlam(process_noise=np.diag([VELOCITY_NOISE_STD, OMEGA_NOISE_STD]) ** 2)
    graph_slam.initialize(init_pose)

    prng_key, k_pinn = jax.random.split(prng_key)
    field_pinn_config = PinnConfig(
        x_bounds=(sim_data.grid.x_min, sim_data.grid.x_max),
        y_bounds=(sim_data.grid.y_min, sim_data.grid.y_max),
        t_max=float(sample_times[-1]),
        hidden_dim=cfg.pinn.hidden_dim,
        num_layers=cfg.pinn.num_layers,
        learning_rate=cfg.pinn.learning_rate,
        num_steps=(
            cfg.graph_slam.pinn_steps
            if cfg.graph_slam.pinn_steps is not None
            else cfg.pinn.num_steps
        ),
        batch_size=cfg.graph_slam.pinn_batch,
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
    print("  Warm-up losses: " + ", ".join(f"{k}={v:.4f}" for k, v in warmup_losses.items()))

    # ------------------------------------------------------------------
    # 4. Main loop with live plot.
    # ------------------------------------------------------------------
    print("\n[4/4] Running pose-graph SLAM over the out-and-back path...")
    interactive = not args.no_show
    if interactive:
        plt.ion()
    fig, (ax_map, ax_err) = plt.subplots(1, 2, figsize=(14, 7))

    ax_map.plot(
        coords_true[: turnaround_idx + 1, 0],
        coords_true[: turnaround_idx + 1, 1],
        "k-",
        lw=2,
        label="True outbound leg",
    )
    ax_map.plot(
        coords_true[turnaround_idx:, 0],
        coords_true[turnaround_idx:, 1],
        "k:",
        lw=2,
        label="True return leg (mirrored)",
    )
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
    ax_err.axvline(turnaround_idx, color="gray", linestyle=":", label="Turnaround")
    ax_err.set_xlabel("Step")
    ax_err.set_ylabel("Position error vs true path [m]")
    ax_err.legend(loc="best")
    ax_err.grid(True, alpha=0.3)

    checkpoints_dir = graphs_dir / "graph_slam_out_and_back_checkpoints"
    dr_err_hist: list[float] = []
    est_err_hist: list[float] = []
    field_losses: dict[str, float] = dict(warmup_losses)
    last_cost = 0.0

    def redraw(step_num: int) -> None:
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
            f"Out-and-Back Pose-Graph SLAM (Step {step_num}/{n_steps})\n"
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

    pbar = tqdm(range(n_steps), desc="  Out-and-Back Survey", unit="step", dynamic_ncols=True)
    for step_idx in pbar:
        t_curr = float(sample_times[step_idx + 1])
        true_pos = coords_true[step_idx + 1]
        ctrl = np.array([v_act[step_idx], w_act[step_idx]])
        node_idx = graph_slam.predict(control=ctrl, dt=dt)

        step_obs = obs_vals_all[step_idx]
        if step_obs is not None:
            field_mapper.add_observation(node_idx, t_curr, step_obs)

        step_num = step_idx + 1

        # Correct BEFORE refitting so the map used has not yet seen this node's
        # own reading (leave-one-out). Correcting after the fit lets the map
        # memorize the observation at the drifted position, collapsing the
        # residual to ~0 -- the correction then reports "already correct" no
        # matter how wrong the pose is.
        # No-op (returns False) for the one step with no observation (see the
        # comment above obs_vals_return).
        field_mapper.apply_field_correction(
            node_idx,
            graph_slam,
            measurement_noise_std=obs_noise_norm,
            field_losses=field_losses,
            max_correction=args.max_correction,
            map_error_frac=cfg.graph_slam.map_error_frac,
        )

        if step_num % args.fit_interval == 0 or step_num == n_steps:
            prng_key, k_fit = jax.random.split(prng_key)
            field_losses = field_mapper.fit(graph_slam, key=k_fit)

        if args.known_loop_closures and node_idx > turnaround_idx:
            # This path was engineered so the return leg exactly revisits the
            # outbound leg in reverse -- unlike a real deployment, we KNOW the
            # correspondence exactly (no detection needed): return node
            # node_idx and outbound node (2*turnaround_idx - node_idx) are the
            # same physical location by construction. Add that directly as a
            # tight loop-closure edge instead of relying only on the softer,
            # field-derived position factor.
            #
            # Constrain POSITION ONLY, leaving headings free. A full SE(2)
            # edge additionally pins the relative heading, and here that
            # heading is only approximately pi: the mirrored path's headings
            # are recomputed from consecutive reversed positions, so they are
            # not an exact 180-degree offset of the outbound ones. Asserting an
            # exact pi offset injects error the optimizer then spreads through
            # the whole graph, distorting interior nodes -- which is why the
            # earlier SE(2) version improved return-to-start closure while
            # making overall trajectory RMSE worse.
            mirror_idx = 2 * turnaround_idx - node_idx
            graph_slam.add_position_loop_closure(
                mirror_idx, node_idx, information=cfg.graph_slam.loop_closure.information
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
    est_return_err = float(np.linalg.norm(estimated_traj[-1] - coords_true[0]))
    err_red = ((dr_rmse_final - est_rmse) / dr_rmse_final) * 100.0

    print("\n--------------------------------------------------")
    print(f"1. Dead Reckoning Trajectory RMSE:       {dr_rmse_final:.3f} m")
    print(f"2. Graph-SLAM (PINN-corrected) RMSE:      {est_rmse:.3f} m")
    print(f"   DR final return-to-start error:        {dr_return_err:.3f} m")
    print(f"   Graph-SLAM final return-to-start error: {est_return_err:.3f} m")
    print(f"   Final optimization cost:               {last_cost:.4e}")
    print(f"SLAM Error Reduction vs Dead Reckoning:   {err_red:.1f}%")
    for name, fm in field_mapper.field_maps.items():
        v_est = np.array(fm.params.v_flow)
        d_est = np.array(fm.D)
        print(
            f"     {name:>12s}: loss={field_losses.get(name, float('nan')):.4f}, "
            f"v_flow=[{v_est[0]:.3f}, {v_est[1]:.3f}] m/s, D={d_est[0]:.4f} m^2/s"
        )
    print("--------------------------------------------------")

    redraw(n_steps)
    fig_path = graphs_dir / "graph_slam_out_and_back_trajectory_comparison.png"
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
