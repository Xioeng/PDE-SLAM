"""
satellite.py
============
High-resolution satellite imagery fetching and geodetic ENU backdrop rendering
via the ArcGIS REST Export Image API.
"""

from __future__ import annotations

import contextlib
import math
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from pde_slam.coords import ENUFrame


def latlon_to_tile(lat: float, lon: float, zoom: int) -> tuple[int, int]:
    """Convert WGS84 lat/lon to OpenStreetMap/ESRI tile coordinates."""
    lat_rad = math.radians(lat)
    n = 2.0**zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile


def tile_to_latlon(xtile: int, ytile: int, zoom: int) -> tuple[float, float]:
    """Convert tile coordinates back to top-left WGS84 lat/lon."""
    n = 2.0**zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg


def zoom_to_pixel_size(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    zoom: int = 18,
    max_dim: int = 2048,
) -> tuple[int, int]:
    """Calculate pixel dimensions matching a Web Mercator zoom level.

    Parameters
    ----------
    lat_min, lat_max : float
        Latitude bounds in decimal degrees.
    lon_min, lon_max : float
        Longitude bounds in decimal degrees.
    zoom : int, default=18
        Target slippy map zoom level.
    max_dim : int, default=2048
        Maximum allowable dimension in pixels.

    Returns
    -------
    tuple of int
        (width, height) in pixels.
    """
    scale = (256.0 * (2.0**zoom)) / 360.0
    lat_mid = math.radians(0.5 * (lat_min + lat_max))
    # Correct longitude scaling by cosine of mean latitude
    width = int(abs(lon_max - lon_min) * scale * math.cos(lat_mid))
    height = int(abs(lat_max - lat_min) * scale)

    width = min(max_dim, max(256, width))
    height = min(max_dim, max(256, height))
    return width, height


def fetch_satellite_image(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    zoom: int = 18,
    size: tuple[int, int] | None = None,
    cache_dir: str | Path | None = None,
) -> tuple[np.ndarray | None, list[float] | None]:
    """Fetch high-resolution satellite imagery covering geographic bounding box.

    Uses a single HTTP request to the ArcGIS World Imagery REST Export API.

    Parameters
    ----------
    lat_min, lat_max : float
        Latitude bounds in decimal degrees.
    lon_min, lon_max : float
        Longitude bounds in decimal degrees.
    zoom : int, default=18
        Target zoom level for automatic pixel resolution scaling.
    size : tuple of int, optional
        Explicit (width, height) in pixels. Overrides zoom calculation.
    cache_dir : str or Path, optional
        Local directory for caching downloaded images.

    Returns
    -------
    image : np.ndarray or None
        RGB image array of shape (H, W, 3) or None if request fails.
    extent : list of float or None
        [lon_min, lon_max, lat_min, lat_max] geographic extent or None.
    """
    if size is not None:
        width, height = size
    else:
        width, height = zoom_to_pixel_size(
            lat_min, lat_max, lon_min, lon_max, zoom=zoom
        )

    if cache_dir is not None:
        cpath = Path(cache_dir)
    else:
        cpath = Path.home() / ".cache" / "pde_slam" / "satellite_exports"

    try:
        cpath.mkdir(parents=True, exist_ok=True)
    except Exception:
        cpath = Path("/tmp") / "pde_slam_satellite_exports"
        with contextlib.suppress(Exception):
            cpath.mkdir(parents=True, exist_ok=True)

    cache_name = (
        f"arcgis_z{zoom}_{width}x{height}_"
        f"{lat_min:.4f}_{lat_max:.4f}_{lon_min:.4f}_{lon_max:.4f}.jpg"
    )
    cache_file = cpath / cache_name

    if not cache_file.exists():
        url = (
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/export"
            f"?bbox={lon_min:.7f},{lat_min:.7f},{lon_max:.7f},{lat_max:.7f}"
            f"&bboxSR=4326&imageSR=4326&size={width},{height}&format=jpg&f=image"
        )
        headers = {"User-Agent": "PDE-SLAM/1.0 (Aquatic-SLAM-Research)"}
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=8) as response:
                img_data = response.read()
            with open(cache_file, "wb") as f:
                f.write(img_data)
        except Exception:
            return None, None

    try:
        img = Image.open(cache_file).convert("RGB")
        return np.array(img), [lon_min, lon_max, lat_min, lat_max]
    except Exception:
        return None, None


def fetch_satellite_enu_backdrop(
    enu_frame: ENUFrame,
    grid_extent: dict[str, float] | Any | None = None,
    polygon_enu: np.ndarray | None = None,
    zoom: int = 18,
    margin_factor: float = 1.3,
    cache_dir: str | Path | None = None,
) -> tuple[np.ndarray | None, list[float] | None]:
    """Fetch satellite backdrop projected to local 2D ENU metric coordinates.

    Parameters
    ----------
    enu_frame : ENUFrame
        Local ENU reference origin frame.
    grid_extent : dict of str to float or SpatialGrid, optional
        Extent containing 'x_min', 'x_max', 'y_min', 'y_max' or SpatialGrid.
    polygon_enu : np.ndarray, optional
        (N, 2) boundary polygon in local ENU coordinates.
    zoom : int, default=18
        Map zoom level for image resolution.
    margin_factor : float, default=1.3
        Padding factor around grid extent to ensure full coverage.
    cache_dir : str or Path, optional
        Directory for caching downloaded imagery.

    Returns
    -------
    image : np.ndarray or None
        RGB satellite image array.
    extent_enu : list of float or None
        [east_min, east_max, north_min, north_max] in meters.
    """
    if polygon_enu is not None and len(polygon_enu) > 0:
        x_min_raw = float(np.min(polygon_enu[:, 0]))
        x_max_raw = float(np.max(polygon_enu[:, 0]))
        y_min_raw = float(np.min(polygon_enu[:, 1]))
        y_max_raw = float(np.max(polygon_enu[:, 1]))
    elif grid_extent is not None:
        if hasattr(grid_extent, "x_min"):
            x_min_raw = float(grid_extent.x_min)
            x_max_raw = float(grid_extent.x_max)
            y_min_raw = float(grid_extent.y_min)
            y_max_raw = float(grid_extent.y_max)
        else:
            x_min_raw = float(grid_extent.get("x_min", -50.0))
            x_max_raw = float(grid_extent.get("x_max", 50.0))
            y_min_raw = float(grid_extent.get("y_min", -50.0))
            y_max_raw = float(grid_extent.get("y_max", 50.0))
    else:
        x_min_raw, x_max_raw = -50.0, 50.0
        y_min_raw, y_max_raw = -50.0, 50.0

    # Expand symmetrically from the center
    x_center = 0.5 * (x_min_raw + x_max_raw)
    x_half = max(5.0, 0.5 * (x_max_raw - x_min_raw) * margin_factor)
    x_min, x_max = x_center - x_half, x_center + x_half

    y_center = 0.5 * (y_min_raw + y_max_raw)
    y_half = max(5.0, 0.5 * (y_max_raw - y_min_raw) * margin_factor)
    y_min, y_max = y_center - y_half, y_center + y_half

    corners_enu = np.array(
        [[x_min, y_min], [x_max, y_min], [x_min, y_max], [x_max, y_max]]
    )
    corners_geo = enu_frame.enu_to_geodetic(corners_enu)

    lat_min = float(np.min(corners_geo[:, 0]))
    lat_max = float(np.max(corners_geo[:, 0]))
    lon_min = float(np.min(corners_geo[:, 1]))
    lon_max = float(np.max(corners_geo[:, 1]))

    img, extent_geo = fetch_satellite_image(
        lat_min, lat_max, lon_min, lon_max, zoom=zoom, cache_dir=cache_dir
    )
    if img is None or extent_geo is None:
        return None, None

    # Project satellite bounding corners back to ENU
    sat_corners_geo = np.array(
        [
            [extent_geo[2], extent_geo[0]],  # SW: (lat_min, lon_min)
            [extent_geo[3], extent_geo[1]],  # NE: (lat_max, lon_max)
        ]
    )
    sat_corners_enu = enu_frame.geodetic_to_enu(sat_corners_geo)

    extent_enu = [
        float(sat_corners_enu[0, 0]),
        float(sat_corners_enu[1, 0]),
        float(sat_corners_enu[0, 1]),
        float(sat_corners_enu[1, 1]),
    ]
    return img, extent_enu
