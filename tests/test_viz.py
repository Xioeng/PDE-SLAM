"""
test_viz.py
===========
Unit tests for the publication visualization engine (pde_slam.viz).
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from pde_slam.viz.grids import compose_evolution_grid, compose_residuals_grid
from pde_slam.viz.live import LiveSlamVisualizer
from pde_slam.viz.panels import (
    render_field_panel,
    render_residual_panel,
    render_tracking_error_panel,
    render_trajectories_panel,
)
from pde_slam.viz.satellite import latlon_to_tile, tile_to_latlon, zoom_to_pixel_size
from pde_slam.viz.style import get_feature_cmap


def test_tile_math() -> None:
    """Verify forward and inverse tile coordinates and zoom sizing math."""
    lat, lon, zoom = 25.7617, -80.1918, 18
    tx, ty = latlon_to_tile(lat, lon, zoom)
    lat_rec, lon_rec = tile_to_latlon(tx, ty, zoom)

    assert isinstance(tx, int)
    assert isinstance(ty, int)
    assert abs(lat_rec - lat) < 0.05
    assert abs(lon_rec - lon) < 0.05

    w, h = zoom_to_pixel_size(25.75, 25.77, -80.20, -80.18, zoom=18)
    assert isinstance(w, int)
    assert isinstance(h, int)
    assert 256 <= w <= 2048
    assert 256 <= h <= 2048


def test_feature_colormaps() -> None:
    """Verify distinct colormaps mapped per physical feature."""
    assert get_feature_cmap("salinity") == "viridis"
    assert get_feature_cmap("Salinity (PPT)") == "viridis"
    assert get_feature_cmap("temperature") == "coolwarm"
    assert get_feature_cmap("Temperature (C)") == "coolwarm"
    assert get_feature_cmap("chlorophyll") == "YlGn"
    assert get_feature_cmap("odo") == "cividis"


def test_render_panels() -> None:
    """Verify single-panel rendering routines execute without error."""
    fig, ax = plt.subplots()
    xs = np.linspace(-10, 10, 20)
    ys = np.linspace(-10, 10, 20)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    data_2d = np.sin(XX) * np.cos(YY)

    # 1. Field panel
    im = render_field_panel(ax, XX, YY, data_2d, cmap="viridis", colorbar=False)
    assert im is not None
    plt.close(fig)

    # 2. Residual panel
    fig, ax = plt.subplots()
    im_res = render_residual_panel(ax, XX, YY, np.abs(data_2d), colorbar=False)
    assert im_res is not None
    plt.close(fig)

    # 3. Trajectories panel
    fig, ax = plt.subplots()
    render_trajectories_panel(
        ax,
        coords_dict={
            "ground_truth": np.array([[0, 0], [1, 1], [2, 2]]),
            "dead_reckoning": np.array([[0, 0], [1.1, 0.9], [2.1, 1.8]]),
        },
        legend=False,
    )
    plt.close(fig)

    # 4. Tracking error panel
    fig, ax = plt.subplots()
    render_tracking_error_panel(
        ax,
        times=np.array([0, 1, 2]),
        errors_dict={"dead_reckoning": np.array([0.0, 0.2, 0.4])},
        legend=False,
    )
    plt.close(fig)


def test_compose_grids() -> None:
    """Verify evolution and residual grid composers create valid Figures."""
    xs = np.linspace(-5, 5, 10)
    ys = np.linspace(-5, 5, 10)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")

    gt_snaps = [np.zeros((10, 10)), np.ones((10, 10))]
    pinn_snaps = {0: gt_snaps, 25: gt_snaps}
    timestamps = [0.0, 50.0]

    fig_grid = compose_evolution_grid(
        field_name="salinity",
        gt_snapshots=gt_snaps,
        pinn_stage_snapshots=pinn_snaps,
        timestamps=timestamps,
        X=XX,
        Y=YY,
    )
    assert fig_grid is not None
    plt.close(fig_grid)

    fig_res = compose_residuals_grid(
        residual_snapshots=gt_snaps,
        timestamps=timestamps,
        X=XX,
        Y=YY,
    )
    assert fig_res is not None
    plt.close(fig_res)


def test_live_slam_visualizer(tmp_path: Any) -> None:
    """Verify LiveSlamVisualizer initialization, step updates, and finalization."""
    poly = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])

    coords_true = np.array([[0, 0], [2, 2], [4, 4]])

    viz = LiveSlamVisualizer(
        t_max=10.0,
        polygon_enu=poly,
        coords_true=coords_true,
        interactive=False,
    )
    assert viz.fig is not None
    assert viz.ax_map is not None
    assert viz.ax_err is not None

    viz.update(
        step=1,
        t_curr=5.0,
        true_pos=np.array([2.0, 2.0]),
        rbpf_pos=np.array([2.1, 1.9]),
        dr_pos=np.array([1.8, 2.3]),
        oracle_pos=np.array([2.05, 1.95]),
        particles=np.random.normal(size=(20, 2)),
        loss_val=0.0123,
    )

    assert len(viz.time_history) == 2
    assert len(viz.err_history_rbpf) == 2

    out_file = tmp_path / "live_traj_test.png"
    viz.finalize(output_path=out_file, rmse_dr=0.5, rmse_rbpf=0.2, rmse_oracle=0.1)
    assert out_file.exists()


def test_viz_utils() -> None:
    """Verify shared viz geometry, masking, and spatial axes setup utilities."""
    from pde_slam.viz.utils import (
        ensure_closed_polygon,
        mask_field_grid,
        setup_spatial_axes,
    )

    # 1. Closed polygon utility
    assert ensure_closed_polygon(None) is None
    assert ensure_closed_polygon(np.zeros((0, 2))) is None
    open_poly = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    closed_poly = ensure_closed_polygon(open_poly)
    assert closed_poly is not None
    assert len(closed_poly) == 5
    assert np.array_equal(closed_poly[0], closed_poly[-1])
    assert len(ensure_closed_polygon(closed_poly)) == 5

    # 2. Mask field grid utility
    field = np.ones((5, 10))
    mask = np.zeros((5, 10), dtype=bool)
    mask[2, 2] = True
    masked = mask_field_grid(field, mask=mask, target_shape=(5, 10))
    assert np.isnan(masked[0, 0])
    assert masked[2, 2] == 1.0

    # 3. Setup spatial axes utility
    fig, ax = plt.subplots()
    setup_spatial_axes(
        ax=ax,
        poly_closed=closed_poly,
        ic_anchors=np.array([[0.5, 0.5]]),
        xlim=(-10.0, 10.0),
        ylim=(-10.0, 10.0),
    )
    assert ax.get_xlim() == (-10.0, 10.0)
    plt.close(fig)
