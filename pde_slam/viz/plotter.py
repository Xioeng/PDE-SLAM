"""
plotter.py
==========
High-level experiment visualizer and batch plot generator for PDE-SLAM experiments.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from pde_slam.io.experiment import SlamExperimentData, load_experiment
from pde_slam.viz.grids import compose_evolution_grid, compose_residuals_grid
from pde_slam.viz.panels import (
    render_field_panel,
    render_residual_panel,
    render_tracking_error_panel,
    render_trajectories_panel,
)
from pde_slam.viz.satellite import fetch_satellite_enu_backdrop
from pde_slam.viz.style import (
    FIG1_WSPACE,
    FIGSIZE_FIELD_PANEL,
    FIGSIZE_PATHS,
    FIGSIZE_RMSE,
    FIGSIZE_TRAJECTORY_COMBINED,
    PLOT_RC_PARAMS,
    get_feature_cmap,
)
from pde_slam.viz.utils import ensure_closed_polygon, mask_field_grid


def plot_saved_experiment(
    experiment: SlamExperimentData | str | Path,
    output_dir: str | Path | None = None,
    sim_name: str | None = None,
    save_grids: bool = True,
    save_individual: bool = True,
    zoom: int = 18,
    figsize_paths: tuple[float, float] = FIGSIZE_PATHS,
    figsize_rmse: tuple[float, float] = FIGSIZE_RMSE,
    figsize_panel: tuple[float, float] = FIGSIZE_FIELD_PANEL,
    figsize_grid_width: float = 14.0,
    figsize_grid_row_height: float = 2.4,
) -> dict[str, Path]:
    """Generate and save all publication visualizations for a saved experiment.

    Parameters
    ----------
    experiment : SlamExperimentData or str or Path
        Loaded SlamExperimentData object or path to a .pkl experiment file.
    output_dir : str or Path, optional
        Base output figures directory (default: 'output/figures/{sim_name}/').
    sim_name : str, optional
        Simulation experiment name identifier.
    save_grids : bool, default=True
        Whether to generate and save composite multi-stage grids.
    save_individual : bool, default=True
        Whether to save individual standalone field, residual, path, and error figures.
    zoom : int, default=18
        Map tile zoom level.
    figsize_paths : tuple of float
        Dimensions for standalone path plot.
    figsize_rmse : tuple of float
        Dimensions for standalone RMSE plot.
    figsize_panel : tuple of float
        Dimensions for individual field and residual panels.
    figsize_grid_width : float
        Width for composite evolution grids.
    figsize_grid_row_height : float
        Row height for composite evolution grids.

    Returns
    -------
    dict of str to Path
        Dictionary mapping figure names to saved file paths.
    """
    for k, v in PLOT_RC_PARAMS.items():
        plt.rcParams[k] = v

    if isinstance(experiment, (str, Path)):
        data = load_experiment(experiment)
        if sim_name is None:
            sim_name = (
                Path(experiment)
                .stem.replace("_experiment", "")
                .replace("_rbpf_slam", "")
            )
    else:
        data = experiment
        if sim_name is None:
            sim_name = data.sim_name

    sim_name = sim_name or "simulation"
    save_folder = (
        Path(output_dir)
        if output_dir is not None
        else Path("output/figures") / sim_name
    )
    save_folder.mkdir(parents=True, exist_ok=True)

    saved_files: dict[str, Path] = {}

    # 1. Fetch Satellite Backdrop
    sat_img, sat_extent_enu = fetch_satellite_enu_backdrop(
        enu_frame=data.enu_frame,
        grid_extent=data.grid_extent,
        zoom=zoom,
    )

    poly_closed = ensure_closed_polygon(data.polygon_enu)
    grid_ext = data.grid_extent
    xlim = (grid_ext.get("x_min", -50.0), grid_ext.get("x_max", 50.0))
    ylim = (grid_ext.get("y_min", -50.0), grid_ext.get("y_max", 50.0))

    coords_true_2d = (
        data.coords_true[:, :2] if data.coords_true.ndim == 2 else data.coords_true
    )
    coords_dr_2d = data.coords_dr[:, :2] if data.coords_dr.ndim == 2 else data.coords_dr
    oracle_traj_2d = (
        data.oracle_traj[:, :2] if data.oracle_traj.ndim == 2 else data.oracle_traj
    )
    estimated_traj_2d = (
        data.estimated_traj[:, :2]
        if data.estimated_traj.ndim == 2
        else data.estimated_traj
    )

    # 2. Standalone Trajectories Comparison
    if save_individual:
        fig_path, ax_path = plt.subplots(figsize=figsize_paths)
        render_trajectories_panel(
            ax=ax_path,
            coords_dict={
                "ground_truth": coords_true_2d,
                "dead_reckoning": coords_dr_2d,
                "oracle_rbpf": oracle_traj_2d,
                "online_rbpf": estimated_traj_2d,
            },
            sat_img=sat_img,
            sat_extent=sat_extent_enu,
            poly_closed=poly_closed,
            ic_anchors=data.ic_points_enu,
            xlim=xlim,
            ylim=ylim,
            legend=True,
            frameon=True,
        )
        fig_path.tight_layout()
        path_file = save_folder / f"{sim_name}_trajectories.png"
        fig_path.savefig(
            path_file, dpi=PLOT_RC_PARAMS.get("savefig.dpi", 300), bbox_inches="tight"
        )
        plt.close(fig_path)
        saved_files["trajectories"] = path_file

        # 3. Standalone Tracking Error
        fig_rmse, ax_rmse = plt.subplots(figsize=figsize_rmse)
        err_dr = np.linalg.norm(coords_dr_2d - coords_true_2d, axis=-1)
        err_oracle = np.linalg.norm(oracle_traj_2d - coords_true_2d, axis=-1)
        err_rbpf = np.linalg.norm(estimated_traj_2d - coords_true_2d, axis=-1)
        render_tracking_error_panel(
            ax=ax_rmse,
            times=data.sample_times,
            errors_dict={
                "dead_reckoning": err_dr,
                "oracle_rbpf": err_oracle,
                "online_rbpf": err_rbpf,
            },
            legend=True,
            frameon=False,
        )
        fig_rmse.tight_layout()
        rmse_file = save_folder / f"{sim_name}_tracking_error.png"
        fig_rmse.savefig(
            rmse_file, dpi=PLOT_RC_PARAMS.get("savefig.dpi", 300), bbox_inches="tight"
        )
        plt.close(fig_rmse)
        saved_files["tracking_error"] = rmse_file

    # 4. Combined Trajectories & Tracking Error Grid
    if save_grids:
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_TRAJECTORY_COMBINED)
        render_trajectories_panel(
            ax=ax1,
            coords_dict={
                "ground_truth": coords_true_2d,
                "dead_reckoning": coords_dr_2d,
                "oracle_rbpf": oracle_traj_2d,
                "online_rbpf": estimated_traj_2d,
            },
            sat_img=sat_img,
            sat_extent=sat_extent_enu,
            poly_closed=poly_closed,
            ic_anchors=data.ic_points_enu,
            xlim=xlim,
            ylim=ylim,
            legend=True,
            frameon=True,
        )
        err_dr = np.linalg.norm(coords_dr_2d - coords_true_2d, axis=-1)
        err_oracle = np.linalg.norm(oracle_traj_2d - coords_true_2d, axis=-1)
        err_rbpf = np.linalg.norm(estimated_traj_2d - coords_true_2d, axis=-1)
        render_tracking_error_panel(
            ax=ax2,
            times=data.sample_times,
            errors_dict={
                "dead_reckoning": err_dr,
                "oracle_rbpf": err_oracle,
                "online_rbpf": err_rbpf,
            },
            legend=True,
            frameon=False,
        )
        fig1.subplots_adjust(
            wspace=FIG1_WSPACE, left=0.08, right=0.96, top=0.95, bottom=0.12
        )
        fig1_file = save_folder / f"{sim_name}_trajectories_and_error.png"
        fig1.savefig(
            fig1_file, dpi=PLOT_RC_PARAMS.get("savefig.dpi", 300), bbox_inches="tight"
        )
        saved_files["trajectories_and_error"] = fig1_file

    # 5. PINN Multi-Stage Evolution and PDE Physics Residuals
    check_steps = sorted(data.pinn_checkpoints.keys())
    if len(check_steps) > 0 and data.pinn_config is not None:
        nx = int(grid_ext.get("nx", 50))
        ny = int(grid_ext.get("ny", 50))
        xs = np.linspace(grid_ext.get("x_min", -50.0), grid_ext.get("x_max", 50.0), nx)
        ys = np.linspace(grid_ext.get("y_min", -50.0), grid_ext.get("y_max", 50.0), ny)
        XX, YY = np.meshgrid(xs, ys, indexing="ij")
        poses_2d = jnp.array(np.column_stack([XX.ravel(), YY.ravel()]))

        mask = data.mesh_mask
        sim_t_max = (
            float(data.sample_times[-1]) if len(data.sample_times) > 0 else 100.0
        )
        n_steps = len(data.sample_times) - 1
        path_steps = [int(s) for s in check_steps if int(s) > 0]
        eval_timestamps = [0.0] + [
            float(data.sample_times[min(s, n_steps)]) for s in path_steps
        ]

        stage_trajectories: dict[int | str, np.ndarray | None] = {}
        for s in check_steps:
            k_key: int | str = (
                int(s) if isinstance(s, (int, float, np.number)) else str(s)
            )
            s_val = int(s) if isinstance(s, (int, float, np.number)) else s
            if isinstance(s_val, int) and s_val > 0 and len(coords_true_2d) > s_val:
                stage_trajectories[k_key] = coords_true_2d[: s_val + 1]
            else:
                stage_trajectories[k_key] = None

        for field_idx, field_name in enumerate(data.field_names):
            cmap_curr = get_feature_cmap(field_name)

            # Ground truth snapshots
            gt_snapshots = []
            if field_name in data.gt_solutions:
                gt_sol_arr = data.gt_solutions[field_name]
                sim_n_times = len(gt_sol_arr)
                for col_idx, t_eval in enumerate(eval_timestamps):
                    t_idx = int(
                        np.clip(
                            int((t_eval / sim_t_max) * (sim_n_times - 1)),
                            0,
                            sim_n_times - 1,
                        )
                    )
                    gt_snap = gt_sol_arr[t_idx]
                    gt_plot = mask_field_grid(gt_snap, mask=mask, target_shape=(nx, ny))
                    gt_snapshots.append(gt_plot)

                    if save_individual:
                        fig_ind, ax_ind = plt.subplots(figsize=figsize_panel)
                        render_field_panel(
                            ax=ax_ind,
                            X=XX,
                            Y=YY,
                            field_data=gt_plot,
                            cmap=cmap_curr,
                            sat_img=sat_img,
                            sat_extent=sat_extent_enu,
                            poly_closed=poly_closed,
                            colorbar=True,
                            colorbar_orientation="horizontal",
                            xlim=xlim,
                            ylim=ylim,
                        )
                        fname = (
                            f"{sim_name}_{field_name}_gt_t{col_idx}_{int(t_eval)}s.png"
                        )
                        ind_file = save_folder / fname
                        fig_ind.savefig(
                            ind_file,
                            dpi=PLOT_RC_PARAMS.get("savefig.dpi", 300),
                            bbox_inches="tight",
                        )
                        plt.close(fig_ind)

            # PINN Stage Predictions
            pinn_stage_snaps: dict[int | str, list[np.ndarray]] = {}
            for s in check_steps:
                stage_key: int | str = (
                    int(s) if isinstance(s, (int, float, np.number)) else str(s)
                )
                pinn_map = data.reconstruct_pinn_map(stage_key)

                stage_snaps = []
                for col_idx, t_eval in enumerate(eval_timestamps):
                    pred_vals = np.array(
                        pinn_map.predict(t=t_eval, poses=poses_2d)
                    ).reshape(nx, ny, -1)
                    pred_field = pred_vals[:, :, field_idx]
                    pred_plot = mask_field_grid(
                        pred_field, mask=mask, target_shape=(nx, ny)
                    )

                    stage_snaps.append(pred_plot)

                    if save_individual:
                        fig_p_ind, ax_p_ind = plt.subplots(figsize=figsize_panel)
                        render_field_panel(
                            ax=ax_p_ind,
                            X=XX,
                            Y=YY,
                            field_data=pred_plot,
                            cmap=cmap_curr,
                            sat_img=sat_img,
                            sat_extent=sat_extent_enu,
                            poly_closed=poly_closed,
                            path=stage_trajectories[stage_key],
                            colorbar=True,
                            colorbar_orientation="horizontal",
                            xlim=xlim,
                            ylim=ylim,
                        )
                        p_name = (
                            f"{sim_name}_{field_name}_pinn_step{stage_key}"
                            f"_t{col_idx}_{int(t_eval)}s.png"
                        )
                        ind_p_file = save_folder / p_name
                        fig_p_ind.savefig(
                            ind_p_file,
                            dpi=PLOT_RC_PARAMS.get("savefig.dpi", 300),
                            bbox_inches="tight",
                        )
                        plt.close(fig_p_ind)

                pinn_stage_snaps[stage_key] = stage_snaps

            if save_grids and len(gt_snapshots) > 0:
                fig_grid = compose_evolution_grid(
                    field_name=field_name,
                    gt_snapshots=gt_snapshots,
                    pinn_stage_snapshots=pinn_stage_snaps,
                    timestamps=eval_timestamps,
                    X=XX,
                    Y=YY,
                    sat_img=sat_img,
                    sat_extent=sat_extent_enu,
                    poly_closed=poly_closed,
                    stage_trajectories=stage_trajectories,
                    cmap=cmap_curr,
                    figsize=(
                        figsize_grid_width,
                        figsize_grid_row_height * (len(check_steps) + 1),
                    ),
                    xlim=xlim,
                    ylim=ylim,
                )
                grid_file = save_folder / f"{sim_name}_{field_name}_evolution_grid.png"
                fig_grid.savefig(
                    grid_file,
                    dpi=PLOT_RC_PARAMS.get("savefig.dpi", 300),
                    bbox_inches="tight",
                )
                saved_files[f"{field_name}_grid"] = grid_file

        last_step = check_steps[-1]
        last_step_key: int | str = (
            int(last_step)
            if isinstance(last_step, (int, float, np.number))
            else str(last_step)
        )
        pinn_map_final = data.reconstruct_pinn_map(last_step_key)

        vmap_residual = jax.vmap(
            lambda pos, t_val: pinn_map_final.pde_residual(t=t_val, x=pos[0], y=pos[1])
        )
        residual_snaps = []
        for col_idx, t_eval in enumerate(eval_timestamps):
            t_arr = jnp.full((len(poses_2d),), t_eval)
            pde_res_vals = np.array(vmap_residual(poses_2d, t_arr)).reshape(nx, ny, -1)
            res_field = np.linalg.norm(pde_res_vals, axis=-1)
            res_plot = mask_field_grid(res_field, mask=mask, target_shape=(nx, ny))
            residual_snaps.append(res_plot)

            if save_individual:
                fig_res_ind, ax_res_ind = plt.subplots(figsize=figsize_panel)
                render_residual_panel(
                    ax=ax_res_ind,
                    X=XX,
                    Y=YY,
                    residual_data=res_plot,
                    sat_img=sat_img,
                    sat_extent=sat_extent_enu,
                    poly_closed=poly_closed,
                    true_path=coords_true_2d,
                    colorbar=True,
                    max_ticks=3,
                    xlim=xlim,
                    ylim=ylim,
                )
                ind_res_file = (
                    save_folder
                    / f"{sim_name}_pde_residual_t{col_idx}_{int(t_eval)}s.png"
                )
                fig_res_ind.savefig(
                    ind_res_file,
                    dpi=PLOT_RC_PARAMS.get("savefig.dpi", 300),
                    bbox_inches="tight",
                )
                plt.close(fig_res_ind)

        if save_grids:
            fig_res = compose_residuals_grid(
                residual_snapshots=residual_snaps,
                timestamps=eval_timestamps,
                X=XX,
                Y=YY,
                sat_img=sat_img,
                sat_extent=sat_extent_enu,
                poly_closed=poly_closed,
                coords_true=coords_true_2d,
                xlim=xlim,
                ylim=ylim,
            )
            res_file = save_folder / f"{sim_name}_pde_residuals_grid.png"
            fig_res.savefig(
                res_file,
                dpi=PLOT_RC_PARAMS.get("savefig.dpi", 300),
                bbox_inches="tight",
            )
            saved_files["residuals_grid"] = res_file

    return saved_files
