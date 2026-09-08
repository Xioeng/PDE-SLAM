"""
interactive.py
==============
Interactive graphical waypoint selection GUI using Matplotlib event loops
over local ENU coordinate frames, domain boundaries, and satellite imagery backdrops.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from pde_slam.viz.style import SAT_ALPHA
from pde_slam.viz.utils import ensure_closed_polygon, setup_spatial_axes


class InteractiveWaypointPicker:
    """Interactive GUI for picking robot waypoints on an ENU satellite plot.

    Parameters
    ----------
    polygon_enu : np.ndarray
        (K, 2) boundary polygon array in ENU coordinates.
    sat_img : np.ndarray, optional
        Satellite RGB image backdrop.
    sat_extent : list of float, optional
        [east_min, east_max, north_min, north_max] bounding extent in meters.
    ic_anchors : np.ndarray, optional
        (M, 2) initial condition anchor coordinates.
    xlim : tuple of float, optional
        X-axis (East) limits.
    ylim : tuple of float, optional
        Y-axis (North) limits.
    default_waypoints : np.ndarray, optional
        Pre-populated waypoints.
    """

    def __init__(
        self,
        polygon_enu: np.ndarray,
        sat_img: np.ndarray | None = None,
        sat_extent: list[float] | None = None,
        ic_anchors: np.ndarray | None = None,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        default_waypoints: np.ndarray | None = None,
    ) -> None:
        self.polygon_enu = polygon_enu
        self.sat_img = sat_img
        self.sat_extent = sat_extent
        self.ic_anchors = ic_anchors
        self.xlim = xlim
        self.ylim = ylim

        self.waypoints: list[list[float]] = []
        if default_waypoints is not None and len(default_waypoints) > 0:
            self.waypoints = [list(pt) for pt in default_waypoints]

        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.line_artist: Any = None
        self.points_artist: Any = None
        self.confirmed = False

        self._setup_plot()
        self._connect_events()

    def _setup_plot(self) -> None:
        """Initialize domain, satellite imagery, and instructions."""
        self.ax.clear()
        poly_closed = ensure_closed_polygon(self.polygon_enu)
        setup_spatial_axes(
            ax=self.ax,
            sat_img=self.sat_img,
            sat_extent=self.sat_extent,
            poly_closed=poly_closed,
            ic_anchors=self.ic_anchors,
            xlim=self.xlim,
            ylim=self.ylim,
            sat_alpha=SAT_ALPHA,
            polygon_label="Boundary Polygon",
        )

        # Waypoint artists
        (self.line_artist,) = self.ax.plot(
            [], [], "r--", linewidth=1.8, label="Selected Waypoint Track"
        )
        (self.points_artist,) = self.ax.plot(
            [],
            [],
            "ro",
            markersize=6,
            markeredgecolor="black",
            markeredgewidth=1.2,
        )

        self.ax.set_xlabel("East [m]")
        self.ax.set_ylabel("North [m]")
        self._update_title()
        self.ax.grid(True, linestyle=":", alpha=0.6)
        self.ax.legend(loc="upper right", framealpha=0.85)
        self._redraw_waypoints()

    def _update_title(self) -> None:
        """Update figure title with instructions and count."""
        count = len(self.waypoints)
        self.ax.set_title(
            f"Interactive Waypoint Selection ({count} waypoints)\n"
            r"$\bf{Left-Click}$: Add | $\bf{Right-Click}$: Remove | "
            r"$\bf{Key\ 'c'}$: Clear | $\bf{Enter\ /\ Close}$: Confirm",
            fontsize=10.5,
            pad=10,
        )

    def _connect_events(self) -> None:
        """Connect mouse and keyboard callbacks."""
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _redraw_waypoints(self) -> None:
        """Refresh lines and markers on canvas."""
        if len(self.waypoints) > 0:
            arr = np.array(self.waypoints)
            self.line_artist.set_data(arr[:, 0], arr[:, 1])
            self.points_artist.set_data(arr[:, 0], arr[:, 1])
        else:
            self.line_artist.set_data([], [])
            self.points_artist.set_data([], [])

        self._update_title()
        self.fig.canvas.draw_idle()

    def _on_click(self, event: Any) -> None:
        """Handle mouse click events."""
        if event.inaxes != self.ax:
            return

        if event.button == 1:  # Left click: Add waypoint
            self.waypoints.append([float(event.xdata), float(event.ydata)])
            self._redraw_waypoints()
        elif event.button == 3:  # Right click: Remove last waypoint
            if len(self.waypoints) > 0:
                self.waypoints.pop()
                self._redraw_waypoints()

    def _on_key(self, event: Any) -> None:
        """Handle keyboard shortcuts."""
        if event.key in ("enter", "return"):
            self.confirmed = True
            plt.close(self.fig)
        elif event.key in ("c", "C"):
            self.waypoints.clear()
            self._redraw_waypoints()
        elif event.key in ("escape", "q", "Q"):
            plt.close(self.fig)

    def run(self) -> np.ndarray:
        """Launch interactive event loop and return selected waypoints."""
        plt.show()
        if len(self.waypoints) < 2:
            print("No/insufficient waypoints selected; using fallback.")
            return (
                np.array(self.waypoints)
                if len(self.waypoints) > 0
                else np.empty((0, 2))
            )
        return np.array(self.waypoints)


def pick_waypoints_gui(
    polygon_enu: np.ndarray,
    sat_img: np.ndarray | None = None,
    sat_extent: list[float] | None = None,
    ic_anchors: np.ndarray | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    default_waypoints: np.ndarray | None = None,
) -> np.ndarray:
    """Convenience helper to pick waypoints using an interactive GUI.

    Parameters
    ----------
    polygon_enu : np.ndarray
        (K, 2) boundary polygon in ENU coordinates.
    sat_img : np.ndarray, optional
        Satellite image backdrop.
    sat_extent : list of float, optional
        [east_min, east_max, north_min, north_max] in meters.
    ic_anchors : np.ndarray, optional
        Initial condition anchor coordinates.
    xlim : tuple of float, optional
        X-axis limits.
    ylim : tuple of float, optional
        Y-axis limits.
    default_waypoints : np.ndarray, optional
        Default waypoints to pre-populate.

    Returns
    -------
    np.ndarray
        (N, 2) selected waypoint array.
    """
    picker = InteractiveWaypointPicker(
        polygon_enu=polygon_enu,
        sat_img=sat_img,
        sat_extent=sat_extent,
        ic_anchors=ic_anchors,
        xlim=xlim,
        ylim=ylim,
        default_waypoints=default_waypoints,
    )
    pts = picker.run()
    if len(pts) < 2 and default_waypoints is not None and len(default_waypoints) >= 2:
        return default_waypoints
    return pts
