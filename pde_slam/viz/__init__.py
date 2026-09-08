"""
viz
===
Reusable publication visualization engine for PDE-SLAM experiments and maps.
"""

from __future__ import annotations

from pde_slam.viz.grids import compose_evolution_grid, compose_residuals_grid
from pde_slam.viz.interactive import InteractiveWaypointPicker, pick_waypoints_gui
from pde_slam.viz.live import LiveSlamVisualizer
from pde_slam.viz.panels import (
    render_field_panel,
    render_residual_panel,
    render_tracking_error_panel,
    render_trajectories_panel,
)
from pde_slam.viz.plotter import plot_saved_experiment
from pde_slam.viz.satellite import (
    fetch_satellite_enu_backdrop,
    fetch_satellite_image,
    latlon_to_tile,
    tile_to_latlon,
)
from pde_slam.viz.style import (
    CMAP_PER_FEATURE,
    CMAP_RESIDUALS,
    FIGSIZE_FIELD_PANEL,
    FIGSIZE_GRID_PER_ROW,
    FIGSIZE_GRID_WIDTH,
    FIGSIZE_PATHS,
    FIGSIZE_RESIDUAL_PANEL,
    FIGSIZE_RESIDUALS_GRID,
    FIGSIZE_RMSE,
    FIGSIZE_TRAJECTORY_COMBINED,
    PLOT_RC_PARAMS,
    TRAJECTORY_COLORS,
    get_feature_cmap,
)
from pde_slam.viz.utils import (
    ensure_closed_polygon,
    mask_field_grid,
    setup_spatial_axes,
)

__all__ = [
    "PLOT_RC_PARAMS",
    "TRAJECTORY_COLORS",
    "CMAP_PER_FEATURE",
    "CMAP_RESIDUALS",
    "get_feature_cmap",
    "FIGSIZE_PATHS",
    "FIGSIZE_RMSE",
    "FIGSIZE_FIELD_PANEL",
    "FIGSIZE_RESIDUAL_PANEL",
    "FIGSIZE_TRAJECTORY_COMBINED",
    "FIGSIZE_GRID_WIDTH",
    "FIGSIZE_GRID_PER_ROW",
    "FIGSIZE_RESIDUALS_GRID",
    "fetch_satellite_image",
    "fetch_satellite_enu_backdrop",
    "latlon_to_tile",
    "tile_to_latlon",
    "render_field_panel",
    "render_residual_panel",
    "render_trajectories_panel",
    "render_tracking_error_panel",
    "compose_evolution_grid",
    "compose_residuals_grid",
    "plot_saved_experiment",
    "InteractiveWaypointPicker",
    "pick_waypoints_gui",
    "LiveSlamVisualizer",
    "ensure_closed_polygon",
    "mask_field_grid",
    "setup_spatial_axes",
]
