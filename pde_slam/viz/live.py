"""
live.py
=======
Live 2D trajectory tracking and error animation visualizer for online SLAM runs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pde_slam.viz.style import SAT_ALPHA, TRAJECTORY_COLORS
from pde_slam.viz.utils import ensure_closed_polygon, setup_spatial_axes


class LiveSlamVisualizer:
    """Live 2D Trajectory and Error animation visualizer for online SLAM runs.

    Encapsulates real-time Matplotlib subplots for concurrent display of spatial
    particle clouds, estimated trajectories, ground truth paths, and tracking errors.

    Parameters
    ----------
    t_max : float
        Maximum duration of the simulation in seconds.
    polygon_enu : np.ndarray, optional
        (N, 2) boundary polygon in local ENU coordinates.
    ic_points_enu : np.ndarray, optional
        (M, 2) initial condition anchor locations at t=0.
    sat_img : np.ndarray, optional
        Satellite RGB imagery backdrop.
    sat_extent : list of float, optional
        [east_min, east_max, north_min, north_max] in meters.
    coords_true : np.ndarray, optional
        (T, 2) complete true ground-truth trajectory.
    initial_particles : np.ndarray, optional
        (N, 2) initial particle coordinates at t=0.
    interactive : bool, default=True
        Whether to display interactive live window via `plt.ion()`.
    figsize : tuple of float, default=(18, 8)
        Dimensions of the visualization figure in inches.
    """

    def __init__(
        self,
        t_max: float,
        polygon_enu: np.ndarray | None = None,
        ic_points_enu: np.ndarray | None = None,
        sat_img: np.ndarray | None = None,
        sat_extent: list[float] | None = None,
        coords_true: np.ndarray | None = None,
        initial_particles: np.ndarray | None = None,
        interactive: bool = True,
        figsize: tuple[float, float] = (18, 8),
    ) -> None:
        self.t_max = float(t_max)
        self.interactive = bool(interactive)
        self.sat_extent = sat_extent

        self.fig: Figure
        self.ax_map: Axes
        self.ax_err: Axes
        self.fig, (self.ax_map, self.ax_err) = plt.subplots(1, 2, figsize=figsize)

        if self.interactive:
            plt.ion()
            self.fig.show()

        p0 = coords_true[0] if coords_true is not None else np.array([0.0, 0.0])
        self._init_histories(p0)
        self._init_map_layers(
            sat_img, sat_extent, polygon_enu, ic_points_enu, coords_true
        )
        self._init_trajectory_artists(p0, initial_particles)
        self._init_error_panel()
        self.fig.tight_layout()

    def _init_histories(self, p0: np.ndarray) -> None:
        """Initialize state histories and tracking error logs."""
        self.time_history: list[float] = [0.0]
        self.err_history_dr: list[float] = [0.0]
        self.err_history_oracle: list[float] = [0.0]
        self.err_history_rbpf: list[float] = [0.0]

        self.traj_true: list[np.ndarray] = [p0]
        self.traj_dr: list[np.ndarray] = [p0]
        self.traj_oracle: list[np.ndarray] = [p0]
        self.traj_rbpf: list[np.ndarray] = [p0]

    def _init_map_layers(
        self,
        sat_img: np.ndarray | None,
        sat_extent: list[float] | None,
        polygon_enu: np.ndarray | None,
        ic_points_enu: np.ndarray | None,
        coords_true: np.ndarray | None,
    ) -> None:
        """Draw static map backdrop, boundary polygon, and IC anchor markers."""
        poly_closed = ensure_closed_polygon(polygon_enu)
        setup_spatial_axes(
            ax=self.ax_map,
            sat_img=sat_img,
            sat_extent=sat_extent,
            poly_closed=poly_closed,
            ic_anchors=ic_points_enu,
            sat_alpha=SAT_ALPHA,
            polygon_label="Boundary Polygon",
        )

        if coords_true is not None and len(coords_true) > 0:
            self.ax_map.plot(
                coords_true[:, 0],
                coords_true[:, 1],
                color="gray",
                linestyle="-",
                linewidth=1.2,
                alpha=0.5,
                label="True Path (Ground Truth)",
                zorder=2,
            )
            self.ax_map.plot(
                coords_true[0, 0],
                coords_true[0, 1],
                "go",
                markersize=8,
                label="Start Position (t=0)",
                zorder=4,
            )

    def _init_trajectory_artists(
        self, p0: np.ndarray, initial_particles: np.ndarray | None
    ) -> None:
        """Initialize animated line artists and particle scatter overlays."""
        c_gt = TRAJECTORY_COLORS["ground_truth"]
        c_dr = TRAJECTORY_COLORS["dead_reckoning"]
        c_ora = TRAJECTORY_COLORS["oracle_rbpf"]
        c_rbpf = TRAJECTORY_COLORS["online_rbpf"]

        (self.line_true_map,) = self.ax_map.plot(
            [p0[0]], [p0[1]], color=c_gt["color"], linestyle="-", linewidth=2.2
        )
        (self.line_dr_map,) = self.ax_map.plot(
            [p0[0]],
            [p0[1]],
            color=c_dr["color"],
            linestyle=c_dr["linestyle"],
            linewidth=1.8,
            label=c_dr["label"],
        )
        (self.line_oracle_map,) = self.ax_map.plot(
            [p0[0]],
            [p0[1]],
            color=c_ora["color"],
            linestyle=c_ora["linestyle"],
            linewidth=2.0,
            label=c_ora["label"],
        )
        (self.line_rbpf_map,) = self.ax_map.plot(
            [p0[0]],
            [p0[1]],
            color=c_rbpf["color"],
            linestyle=c_rbpf["linestyle"],
            linewidth=2.0,
            label=c_rbpf["label"],
        )

        pts_init = (
            initial_particles
            if initial_particles is not None
            else np.repeat([p0], 100, axis=0)
        )
        self.scatter_particles = self.ax_map.scatter(
            pts_init[:, 0],
            pts_init[:, 1],
            c="lime",
            s=15,
            alpha=0.6,
            edgecolors="none",
            zorder=5,
            label="Particle Cloud",
        )
        (self.marker_curr_pose,) = self.ax_map.plot(
            p0[0], p0[1], "ro", markersize=8, zorder=6
        )

        self.ax_map.set_title(
            "Live 2D Spatial Trajectory Tracking (Step 0)",
            fontsize=12,
            fontweight="bold",
        )
        self.ax_map.set_xlabel("East Position [m]", fontsize=10)
        self.ax_map.set_ylabel("North Position [m]", fontsize=10)
        self.ax_map.grid(True, linestyle=":", alpha=0.6)
        self.ax_map.legend(loc="best", fontsize=9)

    def _init_error_panel(self) -> None:
        """Initialize tracking error axis and curves."""
        c_dr = TRAJECTORY_COLORS["dead_reckoning"]
        c_ora = TRAJECTORY_COLORS["oracle_rbpf"]
        c_rbpf = TRAJECTORY_COLORS["online_rbpf"]

        self.ax_err.set_facecolor("#f8f9fa")
        (self.line_dr_err,) = self.ax_err.plot(
            [],
            [],
            color=c_dr["color"],
            linestyle=c_dr["linestyle"],
            linewidth=2.0,
            label="Dead Reckoning Error",
        )
        (self.line_oracle_err,) = self.ax_err.plot(
            [],
            [],
            color=c_ora["color"],
            linestyle=c_ora["linestyle"],
            linewidth=2.0,
            label="Oracle RBPF Error",
        )
        (self.line_rbpf_err,) = self.ax_err.plot(
            [],
            [],
            color=c_rbpf["color"],
            linestyle=c_rbpf["linestyle"],
            linewidth=2.2,
            label="Online RBPF-SLAM Error",
        )
        self.ax_err.set_title(
            "Live Tracking Error Over Time", fontsize=12, fontweight="bold"
        )
        self.ax_err.set_xlabel("Time [s]", fontsize=10)
        self.ax_err.set_ylabel("Euclidean Position Error [m]", fontsize=10)
        self.ax_err.set_xlim(0.0, self.t_max)
        self.ax_err.set_ylim(0.0, 5.0)
        self.ax_err.grid(True, linestyle=":", alpha=0.6)
        self.ax_err.legend(loc="upper left", fontsize=9)

        self.fig.tight_layout()

    def update(
        self,
        step: int,
        t_curr: float,
        true_pos: np.ndarray,
        rbpf_pos: np.ndarray,
        dr_pos: np.ndarray,
        oracle_pos: np.ndarray,
        particles: np.ndarray | None = None,
        loss_val: float | None = None,
        redraw_every: int = 1,
    ) -> None:
        """Update live trajectory plots with new state estimates.

        Parameters
        ----------
        step : int
            Current simulation step index.
        t_curr : float
            Current simulation timestamp in seconds.
        true_pos : np.ndarray
            (2,) ground-truth ENU position [east, north].
        rbpf_pos : np.ndarray
            (2,) RBPF-SLAM consensus estimated pose [east, north].
        dr_pos : np.ndarray
            (2,) Dead reckoning estimated pose [east, north].
        oracle_pos : np.ndarray
            (2,) Oracle filter estimated pose [east, north].
        particles : np.ndarray, optional
            (N, 2) current particle cloud positions.
        loss_val : float, optional
            Latest PINN loss value.
        redraw_every : int, default=1
            Frequency of interactive GUI draw updates.
        """
        err_dr = float(np.linalg.norm(dr_pos - true_pos))
        err_ora = float(np.linalg.norm(oracle_pos - true_pos))
        err_rbpf = float(np.linalg.norm(rbpf_pos - true_pos))

        self.time_history.append(float(t_curr))
        self.err_history_dr.append(err_dr)
        self.err_history_oracle.append(err_ora)
        self.err_history_rbpf.append(err_rbpf)

        self.traj_true.append(true_pos)
        self.traj_dr.append(dr_pos)
        self.traj_oracle.append(oracle_pos)
        self.traj_rbpf.append(rbpf_pos)

        # Update Spatial Map Lines
        t_arr = np.array(self.traj_true)
        d_arr = np.array(self.traj_dr)
        o_arr = np.array(self.traj_oracle)
        r_arr = np.array(self.traj_rbpf)

        self.line_true_map.set_data(t_arr[:, 0], t_arr[:, 1])
        self.line_dr_map.set_data(d_arr[:, 0], d_arr[:, 1])
        self.line_oracle_map.set_data(o_arr[:, 0], o_arr[:, 1])
        self.line_rbpf_map.set_data(r_arr[:, 0], r_arr[:, 1])

        if particles is not None and len(particles) > 0:
            self.scatter_particles.set_offsets(particles[:, :2])
        self.marker_curr_pose.set_data([rbpf_pos[0]], [rbpf_pos[1]])

        loss_str = f" | PINN Loss: {loss_val:.4f}" if loss_val is not None else ""
        self.ax_map.set_title(
            f"Live 2D Spatial Trajectory Tracking\n"
            f"(Step {step}, t = {t_curr:.1f}s{loss_str})",
            fontsize=11,
            fontweight="bold",
        )

        # Update Error Lines
        times = np.array(self.time_history)
        self.line_dr_err.set_data(times, self.err_history_dr)
        self.line_oracle_err.set_data(times, self.err_history_oracle)
        self.line_rbpf_err.set_data(times, self.err_history_rbpf)

        max_err = max(
            5.0,
            max(self.err_history_dr),
            max(self.err_history_oracle),
            max(self.err_history_rbpf),
        )
        self.ax_err.set_ylim(0.0, max_err * 1.15)

        if self.interactive and step % redraw_every == 0:
            self.fig.canvas.draw_idle()
            plt.pause(0.001)

    def finalize(
        self,
        output_path: str | Path | None = None,
        rmse_dr: float | None = None,
        rmse_rbpf: float | None = None,
        rmse_oracle: float | None = None,
        dpi: int = 150,
    ) -> None:
        """Finalize figure with summary metrics and save to file.

        Parameters
        ----------
        output_path : str or Path, optional
            File destination to save the trajectory comparison figure.
        rmse_dr : float, optional
            Final Dead Reckoning RMSE in meters.
        rmse_rbpf : float, optional
            Final RBPF-SLAM RMSE in meters.
        rmse_oracle : float, optional
            Final Oracle filter RMSE in meters.
        dpi : int, default=150
            Resolution for saved image.
        """
        mean_dr = (
            rmse_dr
            if rmse_dr is not None
            else float(np.mean(self.err_history_dr[1:] or [0.0]))
        )
        mean_ora = (
            rmse_oracle
            if rmse_oracle is not None
            else float(np.mean(self.err_history_oracle[1:] or [0.0]))
        )
        mean_rbpf = (
            rmse_rbpf
            if rmse_rbpf is not None
            else float(np.mean(self.err_history_rbpf[1:] or [0.0]))
        )

        self.line_dr_err.set_label(f"Dead Reckoning (RMSE: {mean_dr:.2f} m)")
        self.line_oracle_err.set_label(f"Oracle RBPF (RMSE: {mean_ora:.2f} m)")
        self.line_rbpf_err.set_label(f"Online RBPF-SLAM (RMSE: {mean_rbpf:.2f} m)")
        self.ax_err.legend(loc="upper left", fontsize=9)
        self.fig.tight_layout()

        if output_path is not None:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            self.fig.savefig(out_p, dpi=dpi, bbox_inches="tight")
            print(f"  Saved live trajectory figure to: {out_p}")

        if not self.interactive:
            plt.close(self.fig)
