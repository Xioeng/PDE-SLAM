"""
panels.py
=========
Modular, reusable single-panel drawing routines for field snapshots, PDE residuals,
trajectories, and error analysis curves.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator

from pde_slam.viz.style import (
    CMAP_RESIDUALS,
    FIELD_ALPHA,
    RESIDUAL_ALPHA,
    SAT_ALPHA,
    TRAJECTORY_COLORS,
)
from pde_slam.viz.utils import setup_spatial_axes


def render_field_panel(
    ax: Axes,
    X: np.ndarray,
    Y: np.ndarray,
    field_data: np.ndarray,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    sat_img: np.ndarray | None = None,
    sat_extent: list[float] | None = None,
    poly_closed: np.ndarray | None = None,
    path: np.ndarray | None = None,
    path_style: dict[str, Any] | None = None,
    ic_anchors: np.ndarray | None = None,
    colorbar: bool = True,
    colorbar_orientation: str = "horizontal",
    max_ticks: int = 3,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    alpha: float = FIELD_ALPHA,
    sat_alpha: float = SAT_ALPHA,
    aspect: str | float | None = "equal",
) -> Any:
    """Render a single 2D continuous water feature field snapshot onto an axis.

    Parameters
    ----------
    ax : Axes
        Matplotlib axis to render on.
    X : np.ndarray
        2D meshgrid X coordinates.
    Y : np.ndarray
        2D meshgrid Y coordinates.
    field_data : np.ndarray
        2D scalar array of field values (can contain NaNs for masking).
    cmap : str, default='viridis'
        Matplotlib colormap.
    vmin : float, optional
        Minimum value for colormap scaling.
    vmax : float, optional
        Maximum value for colormap scaling.
    sat_img : np.ndarray, optional
        Satellite RGB imagery backdrop.
    sat_extent : list of float, optional
        [east_min, east_max, north_min, north_max] bounding extent for satellite image.
    poly_closed : np.ndarray, optional
        (K, 2) closed polygon boundary points.
    path : np.ndarray, optional
        (N, 2) robot trajectory path to overlay.
    path_style : dict, optional
        Custom styling dict for trajectory path line.
    ic_anchors : np.ndarray, optional
        (M, 2) initial condition anchor locations at t=0.
    colorbar : bool, default=True
        Whether to draw a colorbar.
    colorbar_orientation : str, default='horizontal'
        'horizontal' or 'vertical'.
    max_ticks : int, default=3
        Maximum tick markers on colorbar.
    xlim : tuple of float, optional
        (x_min, x_max) limits.
    ylim : tuple of float, optional
        (y_min, y_max) limits.
    alpha : float, default=FIELD_ALPHA
        Transparency of the pcolormesh layer.
    sat_alpha : float, default=SAT_ALPHA
        Transparency of satellite image.
    aspect : str or float or None, default='equal'
        Axes aspect ratio ('equal' ensures 1 unit East = 1 unit North).

    Returns
    -------
    QuadMesh
        The Matplotlib pcolormesh artist.
    """
    # 1. Setup spatial backdrop, polygon, IC anchors, and axes limits
    setup_spatial_axes(
        ax=ax,
        sat_img=sat_img,
        sat_extent=sat_extent,
        poly_closed=poly_closed,
        ic_anchors=ic_anchors,
        xlim=xlim,
        ylim=ylim,
        sat_alpha=sat_alpha,
        aspect=aspect,
    )

    # 2. Continuous Scalar Field
    im = ax.pcolormesh(
        X,
        Y,
        field_data,
        cmap=cmap,
        alpha=alpha,
        vmin=vmin,
        vmax=vmax,
        shading="auto",
        zorder=1,
    )

    # 3. Colorbar (if requested)
    if colorbar:
        fig = ax.figure
        if colorbar_orientation == "horizontal":
            cbar = fig.colorbar(
                im,
                ax=ax,
                orientation="horizontal",
                fraction=0.048,
                pad=0.06,
                aspect=20,
            )
            cbar.locator = MaxNLocator(nbins=max_ticks)
            cbar.update_ticks()
            cbar.set_label("")
            cbar.ax.tick_params(labelsize=5.5)
        else:
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.locator = MaxNLocator(nbins=max_ticks)
            cbar.update_ticks()
            cbar.set_label("")
            cbar.ax.tick_params(labelsize=7)

    # 4. Trajectory Path
    if path is not None and len(path) > 0:
        style = path_style or TRAJECTORY_COLORS["ground_truth"]
        ax.plot(
            path[:, 0],
            path[:, 1],
            color=style.get("color", "#000000"),
            linestyle=style.get("linestyle", "-"),
            linewidth=style.get("linewidth", 1.4),
            zorder=3,
        )

    ax.set_title("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    return im


def render_residual_panel(
    ax: Axes,
    X: np.ndarray,
    Y: np.ndarray,
    residual_data: np.ndarray,
    cmap: str = CMAP_RESIDUALS,
    vmin: float | None = None,
    vmax: float | None = None,
    sat_img: np.ndarray | None = None,
    sat_extent: list[float] | None = None,
    poly_closed: np.ndarray | None = None,
    true_path: np.ndarray | None = None,
    colorbar: bool = True,
    max_ticks: int = 3,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    alpha: float = RESIDUAL_ALPHA,
    sat_alpha: float = SAT_ALPHA,
    aspect: str | float | None = "equal",
) -> Any:
    """Render a single Space-Time PDE Physics Residual panel onto an axis."""
    return render_field_panel(
        ax=ax,
        X=X,
        Y=Y,
        field_data=residual_data,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        sat_img=sat_img,
        sat_extent=sat_extent,
        poly_closed=poly_closed,
        path=true_path,
        path_style={"color": "#ffffff", "linestyle": "-", "linewidth": 1.1},
        colorbar=colorbar,
        colorbar_orientation="horizontal",
        max_ticks=max_ticks,
        xlim=xlim,
        ylim=ylim,
        alpha=alpha,
        sat_alpha=sat_alpha,
        aspect=aspect,
    )


def render_trajectories_panel(
    ax: Axes,
    coords_dict: dict[str, np.ndarray],
    sat_img: np.ndarray | None = None,
    sat_extent: list[float] | None = None,
    poly_closed: np.ndarray | None = None,
    ic_anchors: np.ndarray | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    legend: bool = True,
    frameon: bool = True,
    sat_alpha: float = SAT_ALPHA,
    aspect: str | float | None = "equal",
) -> None:
    """Render 2D path comparisons on an axis with optional satellite backdrop."""
    setup_spatial_axes(
        ax=ax,
        sat_img=sat_img,
        sat_extent=sat_extent,
        poly_closed=poly_closed,
        ic_anchors=ic_anchors,
        xlim=xlim,
        ylim=ylim,
        sat_alpha=sat_alpha,
        aspect=aspect,
    )

    for key, arr in coords_dict.items():
        if len(arr) > 0 and key in TRAJECTORY_COLORS:
            cfg = TRAJECTORY_COLORS[key]
            ax.plot(
                arr[:, 0],
                arr[:, 1],
                color=cfg["color"],
                linestyle=cfg["linestyle"],
                linewidth=cfg["linewidth"],
                label=cfg["label"],
                zorder=3,
            )

    ax.set_title("")
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.grid(True, linestyle=":", alpha=0.5)
    if legend:
        ax.legend(loc="best", frameon=frameon)


def render_tracking_error_panel(
    ax: Axes,
    times: np.ndarray,
    errors_dict: dict[str, np.ndarray],
    legend: bool = True,
    frameon: bool = False,
) -> None:
    """Render Euclidean position tracking error curves over time."""
    ax.set_facecolor("#ffffff")
    for key, err in errors_dict.items():
        if len(err) > 0 and key in TRAJECTORY_COLORS:
            cfg = TRAJECTORY_COLORS[key]
            label = f"{cfg['label'].replace(' Path', '')} (Mean: {np.mean(err):.2f}m)"
            ax.plot(
                times,
                err,
                color=cfg["color"],
                linestyle=cfg["linestyle"],
                linewidth=cfg["linewidth"],
                label=label,
            )

    ax.set_title("")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position Error [m]")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if legend:
        ax.legend(loc="upper left", frameon=frameon)
