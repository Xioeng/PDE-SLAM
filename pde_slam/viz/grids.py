"""
grids.py
========
High-level composite multi-panel evolution and residual grid composers.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from pde_slam.viz.panels import render_field_panel, render_residual_panel
from pde_slam.viz.style import (
    CMAP_RESIDUALS,
    FIGSIZE_GRID_PER_ROW,
    FIGSIZE_GRID_WIDTH,
    FIGSIZE_RESIDUALS_GRID,
    GRID_HSPACE,
    GRID_WSPACE,
    get_feature_cmap,
)


def compose_evolution_grid(
    field_name: str,
    gt_snapshots: list[np.ndarray],
    pinn_stage_snapshots: dict[int | str, list[np.ndarray]],
    timestamps: list[float],
    X: np.ndarray,
    Y: np.ndarray,
    sat_img: np.ndarray | None = None,
    sat_extent: list[float] | None = None,
    poly_closed: np.ndarray | None = None,
    stage_trajectories: dict[int | str, np.ndarray | None] | None = None,
    cmap: str | None = None,
    figsize: tuple[float, float] | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> Figure:
    """Compose a publication-grade (N_stages+1) x N_timestamps evolution grid.

    Row 0 contains the Ground Truth field across timestamps.
    Rows 1..K contain the PINN continuous field reconstructed at sequential SLAM stages.

    Parameters
    ----------
    field_name : str
        Physical water feature identifier (e.g. salinity, temperature).
    gt_snapshots : list of np.ndarray
        List of 2D Ground Truth arrays for each timestamp.
    pinn_stage_snapshots : dict
        Mapping from stage checkpoint key to list of 2D predicted field arrays.
    timestamps : list of float
        Evaluation timestamps in seconds.
    X : np.ndarray
        2D meshgrid X coordinates.
    Y : np.ndarray
        2D meshgrid Y coordinates.
    sat_img : np.ndarray, optional
        Satellite RGB imagery backdrop.
    sat_extent : list of float, optional
        Satellite ENU bounding extent.
    poly_closed : np.ndarray, optional
        Closed boundary polygon.
    stage_trajectories : dict, optional
        Mapping from stage key to true trajectory segment.
    cmap : str, optional
        Colormap (defaults to feature-matched colormap).
    figsize : tuple of float, optional
        Figure size in inches.
    xlim : tuple of float, optional
        (x_min, x_max) limits.
    ylim : tuple of float, optional
        (y_min, y_max) limits.

    Returns
    -------
    Figure
        The constructed composite figure.
    """
    cmap_curr = cmap or get_feature_cmap(field_name)
    n_stages = len(pinn_stage_snapshots)
    n_rows = n_stages + 1
    n_cols = len(timestamps)

    if figsize is None:
        figsize = (FIGSIZE_GRID_WIDTH, FIGSIZE_GRID_PER_ROW * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if n_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    # Row 0: Ground Truth Field Snapshots
    for col_idx, gt_snap in enumerate(gt_snapshots):
        ax = axes[0, col_idx]
        render_field_panel(
            ax=ax,
            X=X,
            Y=Y,
            field_data=gt_snap,
            cmap=cmap_curr,
            sat_img=sat_img,
            sat_extent=sat_extent,
            poly_closed=poly_closed,
            colorbar=True,
            colorbar_orientation="vertical",
            xlim=xlim,
            ylim=ylim,
        )

    # Rows 1..K: PINN Stages
    for row_idx, (stage_key, stage_snaps) in enumerate(
        pinn_stage_snapshots.items(), start=1
    ):
        traj_curr = None
        if stage_trajectories is not None:
            traj_curr = stage_trajectories.get(stage_key, None)

        for col_idx, pred_snap in enumerate(stage_snaps):
            ax = axes[row_idx, col_idx]
            render_field_panel(
                ax=ax,
                X=X,
                Y=Y,
                field_data=pred_snap,
                cmap=cmap_curr,
                sat_img=sat_img,
                sat_extent=sat_extent,
                poly_closed=poly_closed,
                path=traj_curr,
                colorbar=True,
                colorbar_orientation="vertical",
                xlim=xlim,
                ylim=ylim,
            )

    fig.subplots_adjust(
        hspace=GRID_HSPACE,
        wspace=GRID_WSPACE,
        left=0.01,
        right=0.95,
        top=0.985,
        bottom=0.015,
    )
    return fig


def compose_residuals_grid(
    residual_snapshots: list[np.ndarray],
    timestamps: list[float],
    X: np.ndarray,
    Y: np.ndarray,
    sat_img: np.ndarray | None = None,
    sat_extent: list[float] | None = None,
    poly_closed: np.ndarray | None = None,
    coords_true: np.ndarray | None = None,
    cmap: str = CMAP_RESIDUALS,
    figsize: tuple[float, float] | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> Figure:
    """Compose a 1 x N_timestamps space-time PDE physics residuals grid.

    Parameters
    ----------
    residual_snapshots : list of np.ndarray
        2D PDE physics residual arrays for each evaluation timestamp.
    timestamps : list of float
        Evaluation timestamps in seconds.
    X : np.ndarray
        2D meshgrid X coordinates.
    Y : np.ndarray
        2D meshgrid Y coordinates.
    sat_img : np.ndarray, optional
        Satellite RGB imagery backdrop.
    sat_extent : list of float, optional
        Satellite ENU bounding extent.
    poly_closed : np.ndarray, optional
        Closed boundary polygon.
    coords_true : np.ndarray, optional
        Ground truth trajectory path.
    cmap : str, default='inferno'
        Colormap for physics residuals.
    figsize : tuple of float, optional
        Figure size in inches.
    xlim : tuple of float, optional
        (x_min, x_max) limits.
    ylim : tuple of float, optional
        (y_min, y_max) limits.

    Returns
    -------
    Figure
        The constructed composite figure.
    """
    n_cols = len(timestamps)
    if figsize is None:
        figsize = FIGSIZE_RESIDUALS_GRID

    fig, axes = plt.subplots(1, n_cols, figsize=figsize)
    if n_cols == 1:
        axes = [axes]

    for col_idx, (_, res_snap) in enumerate(
        zip(timestamps, residual_snapshots, strict=True)
    ):
        ax = axes[col_idx]
        render_residual_panel(
            ax=ax,
            X=X,
            Y=Y,
            residual_data=res_snap,
            cmap=cmap,
            sat_img=sat_img,
            sat_extent=sat_extent,
            poly_closed=poly_closed,
            true_path=coords_true,
            colorbar=True,
            max_ticks=3,
            xlim=xlim,
            ylim=ylim,
        )

    fig.subplots_adjust(
        wspace=GRID_WSPACE,
        left=0.01,
        right=0.98,
        top=0.98,
        bottom=0.08,
    )
    return fig
