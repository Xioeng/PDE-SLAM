"""
utils.py
========
Shared geometry, axes setup, and array masking utilities for visualization.
"""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes

from pde_slam.viz.style import SAT_ALPHA, TRAJECTORY_COLORS


def ensure_closed_polygon(polygon: np.ndarray | None) -> np.ndarray | None:
    """Ensure that a 2D boundary polygon is closed by appending its first vertex.

    Parameters
    ----------
    polygon : np.ndarray or None
        (N, 2) boundary polygon array or None.

    Returns
    -------
    np.ndarray or None
        (N+1, 2) closed polygon array or None.
    """
    if polygon is None or len(polygon) == 0:
        return None
    pts = np.asarray(polygon, dtype=np.float64)
    if not np.array_equal(pts[0], pts[-1]):
        return np.vstack([pts, pts[0]])
    return pts


def mask_field_grid(
    field: np.ndarray,
    mask: np.ndarray | None = None,
    target_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Orient 2D field grid to target shape and apply domain masking with NaNs.

    Parameters
    ----------
    field : np.ndarray
        2D scalar field array of shape (nx, ny) or (ny, nx).
    mask : np.ndarray, optional
        2D boolean mesh mask where True indicates valid domain points.
    target_shape : tuple of int, optional
        (nx, ny) desired spatial grid dimensions.

    Returns
    -------
    np.ndarray
        Masked 2D field array with NaNs outside the valid domain.
    """
    arr = np.asarray(field, dtype=np.float64)
    if target_shape is not None and arr.shape != target_shape:
        arr = arr.T

    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if target_shape is not None and m.shape != target_shape:
            m = m.T
        return np.where(m, arr, np.nan)
    return arr


def setup_spatial_axes(
    ax: Axes,
    sat_img: np.ndarray | None = None,
    sat_extent: list[float] | None = None,
    poly_closed: np.ndarray | None = None,
    ic_anchors: np.ndarray | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    sat_alpha: float = SAT_ALPHA,
    polygon_label: str = "Boundary Polygon",
    aspect: str | float | None = "equal",
) -> None:
    """Configure spatial axes with satellite backdrop, boundary polygon, and IC anchors.

    Parameters
    ----------
    ax : Axes
        Matplotlib axis to configure.
    sat_img : np.ndarray, optional
        Satellite RGB imagery backdrop.
    sat_extent : list of float, optional
        [east_min, east_max, north_min, north_max] bounding extent in meters.
    poly_closed : np.ndarray, optional
        (K, 2) closed polygon boundary points.
    ic_anchors : np.ndarray, optional
        (M, 2) initial condition anchor locations at t=0.
    xlim : tuple of float, optional
        (x_min, x_max) limits. Clamps to sat_extent if None.
    ylim : tuple of float, optional
        (y_min, y_max) limits. Clamps to sat_extent if None.
    sat_alpha : float, default=SAT_ALPHA
        Transparency of satellite image.
    polygon_label : str, default='Boundary Polygon'
        Legend label for domain boundary polygon.
    aspect : str or float or None, default='equal'
        Axes aspect ratio ('equal' ensures 1 unit East = 1 unit North).
    """
    # 1. Aspect Ratio
    if aspect is not None:
        ax.set_aspect(aspect)

    # 2. Satellite Backdrop
    if sat_img is not None and sat_extent is not None:
        ax.imshow(
            sat_img,
            extent=sat_extent,
            origin="upper",
            aspect=aspect if aspect is not None else "auto",
            alpha=sat_alpha,
            zorder=0,
        )
    else:
        ax.set_facecolor("#e0e0e0")

    # 3. Polygon Domain Boundary
    if poly_closed is not None and len(poly_closed) > 0:
        p_cfg = TRAJECTORY_COLORS["polygon"]
        ax.plot(
            poly_closed[:, 0],
            poly_closed[:, 1],
            color=p_cfg["color"],
            linestyle=p_cfg["linestyle"],
            linewidth=1.2,
            alpha=p_cfg["alpha"],
            label=polygon_label,
            zorder=1,
        )

    # 4. Initial Condition Anchors (t=0)
    if ic_anchors is not None and len(ic_anchors) > 0:
        ic_cfg = TRAJECTORY_COLORS["ic_anchors"]
        ax.plot(
            ic_anchors[:, 0],
            ic_anchors[:, 1],
            marker=ic_cfg["marker"],
            color=ic_cfg["color"],
            markersize=ic_cfg["markersize"],
            markeredgecolor=ic_cfg["markeredgecolor"],
            markeredgewidth=ic_cfg["markeredgewidth"],
            linestyle="None",
            label=f"IC Anchors (t=0, N={len(ic_anchors)})",
            zorder=2,
        )

    # 5. Axes View Limits
    if xlim is not None:
        ax.set_xlim(xlim)
    elif sat_extent is not None:
        ax.set_xlim(sat_extent[0], sat_extent[1])

    if ylim is not None:
        ax.set_ylim(ylim)
    elif sat_extent is not None:
        ax.set_ylim(sat_extent[2], sat_extent[3])
