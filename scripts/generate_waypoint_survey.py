"""
scripts/generate_waypoint_survey.py
===================================
Generate a synthetic survey CSV by clicking waypoints on an interactive map.

This script loads a boundary polygon (default or from a CSV), opens a Matplotlib
window to let the user select waypoints in GPS coordinates (lat/lon), and then
drives the kinematic robot model through those waypoints to generate a survey log.

Outputs identical columns to generate_synthetic_survey.py:
``t_s, lat_deg, lon_deg, x_m, y_m, heading_rad, thrust,
salinity_psu, temperature_c, chlorophyll_ug_l``

Usage::

    python scripts/generate_waypoint_survey.py
    python scripts/generate_waypoint_survey.py --output data/raw/waypoint_survey.csv
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

# Add project root to sys.path so pde_slam is importable without package installation
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

# Use interactive GUI backend if possible
with contextlib.suppress(Exception):
    matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pde_slam.coords import ENUFrame
from pde_slam.kinematics import UnicycleKinematics

# ---------------------------------------------------------------------------
# Default survey origin — Biscayne Bay, Miami FL
# ---------------------------------------------------------------------------
DEFAULT_LAT0 = 25.909619867836824
DEFAULT_LON0 = -80.13657451246902

# Tidal mean current [m/s]
TIDAL_U = np.array([0.03, 0.05])

# Biscayne Bay physical field parameters
AMBIENT_SALINITY = 36.0
FRESH_SALINITY = 12.0
PLUME_WIDTH = 50.0
PLUME_DECAY = 0.004

AMBIENT_TEMP = 30.0
INTRUSION_TEMP = 32.5
FRONT_X = 80.0
FRONT_WIDTH = 40.0

CHL_AMBIENT = 0.5
CHL_PLUME_PEAK = 5.0


def _plume_salinity(x: np.ndarray, y: np.ndarray, plume_sources: np.ndarray) -> np.ndarray:
    total_freshness = np.zeros_like(x)
    for px, py in plume_sources:
        plume_dir = np.array([1.0, 1.0]) / np.sqrt(2.0)
        dx = x - px
        dy = y - py
        along = dx * plume_dir[0] + dy * plume_dir[1]
        cross = -dx * plume_dir[1] + dy * plume_dir[0]
        along_clamped = np.maximum(along, 0.0)
        width = PLUME_WIDTH + 0.15 * along_clamped
        freshness = np.exp(-0.5 * (cross / width) ** 2) * np.exp(-PLUME_DECAY * along_clamped)
        total_freshness = np.maximum(total_freshness, freshness)

    sal = AMBIENT_SALINITY - (AMBIENT_SALINITY - FRESH_SALINITY) * total_freshness

    # Add localized peaks (positive and negative salinity anomalies)
    for i, (px, py) in enumerate(plume_sources):
        amp = 4.0 if i % 2 == 0 else -4.0
        sal += amp * np.exp(-0.5 * (((x - px) ** 2 + (y - py) ** 2) / (50.0**2)))

    return sal


def _front_temperature(x: np.ndarray, y: np.ndarray, plume_sources: np.ndarray) -> np.ndarray:
    # Base temperature from front
    temp = AMBIENT_TEMP + (INTRUSION_TEMP - AMBIENT_TEMP) / (
        1.0 + np.exp((x - FRONT_X) / FRONT_WIDTH)
    )
    # Add thermal peaks at the plume sources
    for i, (px, py) in enumerate(plume_sources):
        amp = 2.0 if i % 2 == 0 else -1.5
        temp += amp * np.exp(-0.5 * (((x - px) ** 2 + (y - py) ** 2) / (40.0**2)))
    return temp


def _plume_chlorophyll(x: np.ndarray, y: np.ndarray, plume_sources: np.ndarray) -> np.ndarray:
    total_bloom = np.zeros_like(x)
    for px, py in plume_sources:
        plume_dir = np.array([1.0, 1.0]) / np.sqrt(2.0)
        dx = x - px
        dy = y - py
        along = dx * plume_dir[0] + dy * plume_dir[1]
        cross = -dx * plume_dir[1] + dy * plume_dir[0]
        along_clamped = np.maximum(along, 0.0)
        width = PLUME_WIDTH * 1.2 + 0.2 * along_clamped
        bloom = np.exp(-0.5 * (cross / width) ** 2) * np.exp(
            -PLUME_DECAY * 0.5 * np.maximum(along_clamped - 50.0, 0.0)
        )
        total_bloom = np.maximum(total_bloom, bloom)

    chl = CHL_AMBIENT + (CHL_PLUME_PEAK - CHL_AMBIENT) * total_bloom

    # Add sharp chlorophyll peaks at the sources
    for px, py in plume_sources:
        chl += 2.0 * np.exp(-0.5 * (((x - px) ** 2 + (y - py) ** 2) / (30.0**2)))

    return chl


def get_random_points_in_polygon(
    x_poly: np.ndarray, y_poly: np.ndarray, num_points: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample random points inside a 2D polygon in ENU coordinates.

    Parameters
    ----------
    x_poly : np.ndarray
        X coordinates of the polygon vertices in meters.
    y_poly : np.ndarray
        Y coordinates of the polygon vertices in meters.
    num_points : int
        Number of points to sample.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Array of shape (num_points, 2) containing the sampled points.
    """
    from matplotlib.path import Path as MPath

    path = MPath(np.stack([x_poly, y_poly], axis=1))
    x_min, x_max = x_poly.min(), x_poly.max()
    y_min, y_max = y_poly.min(), y_poly.max()

    points = []
    for _ in range(10000):
        xs = rng.uniform(x_min, x_max, size=num_points * 2)
        ys = rng.uniform(y_min, y_max, size=num_points * 2)
        candidates = np.stack([xs, ys], axis=1)
        inside = path.contains_points(candidates)
        valid = candidates[inside]
        for pt in valid:
            points.append(pt)
            if len(points) == num_points:
                return np.array(points)

    # Fallback to mean of polygon if sampling fails
    mean_pt = np.array([x_poly.mean(), y_poly.mean()])
    while len(points) < num_points:
        points.append(mean_pt)
    return np.array(points)


def load_polygon(polygon_path: str | None, frame: ENUFrame) -> tuple[np.ndarray, np.ndarray]:
    """Load boundary polygon vertices in GPS coordinates."""
    if polygon_path is not None:
        path = Path(polygon_path)
        if not path.exists():
            raise FileNotFoundError(f"Polygon CSV not found: {path}")
        df = pd.read_csv(path)
        if "latitude" in df.columns and "longitude" in df.columns:
            lats = df["latitude"].to_numpy(dtype=np.float64)
            lons = df["longitude"].to_numpy(dtype=np.float64)
            return lats, lons
        elif "x_m" in df.columns and "y_m" in df.columns:
            xs = df["x_m"].to_numpy(dtype=np.float64)
            ys = df["y_m"].to_numpy(dtype=np.float64)
            lats, lons = frame.from_enu(xs, ys)
            return lats, lons
        else:
            raise KeyError("Polygon CSV must contain ['latitude', 'longitude'] or ['x_m', 'y_m']")

    # Default square boundary: 500m x 500m centered around origin
    x_corners = np.array([-250.0, -250.0, 250.0, 250.0, -250.0])
    y_corners = np.array([-250.0, 250.0, 250.0, -250.0, -250.0])
    lats, lons = frame.from_enu(x_corners, y_corners)
    return lats, lons


def select_waypoints_gui(lats: np.ndarray, lons: np.ndarray) -> list[tuple[float, float]]:
    """Open GUI window to select waypoints by clicking."""
    print("\n------------------------------------------------------------")
    print("Opening interactive map window...")
    print("Instructions:")
    print("  1. Click to add waypoints inside/near the boundary.")
    print("  2. Press ENTER or Middle-Click to finish and start simulation.")
    print("------------------------------------------------------------")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(lons, lats, "r--", linewidth=1.5, label="Survey Boundary")
    ax.fill(lons, lats, "r", alpha=0.05)
    ax.plot(lons[0], lats[0], "ro", markersize=6, label="Boundary Origin")

    ax.set_title(
        "Click to place waypoints. Press ENTER when done.",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right")

    # Center view on boundary
    padding_lon = (lons.max() - lons.min()) * 0.1
    padding_lat = (lats.max() - lats.min()) * 0.1
    ax.set_xlim(lons.min() - padding_lon, lons.max() + padding_lon)
    ax.set_ylim(lats.min() - padding_lat, lats.max() + padding_lat)

    # Bring window to front
    plt.ion()
    plt.show()
    plt.pause(0.1)

    pts = plt.ginput(n=-1, timeout=0, show_clicks=True)
    plt.close(fig)

    # Return list of (lat, lon)
    return [(lat, lon) for lon, lat in pts]


def parse_waypoints_cli(wp_str: str) -> list[tuple[float, float]]:
    """Parse waypoints string: 'lat1,lon1; lat2,lon2; ...'"""
    pts = []
    for pair in wp_str.split(";"):
        if not pair.strip():
            continue
        lat, lon = pair.split(",")
        pts.append((float(lat.strip()), float(lon.strip())))
    return pts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate survey log by defining waypoints interactively.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output", default="data/raw/waypoint_survey.csv", help="Output CSV path.")
    parser.add_argument("--lat0", type=float, default=DEFAULT_LAT0, help="ENU origin latitude.")
    parser.add_argument("--lon0", type=float, default=DEFAULT_LON0, help="ENU origin longitude.")
    parser.add_argument("--speed", type=float, default=1.0, help="Robot speed [m/s].")
    parser.add_argument("--k-thrust", type=float, default=1.0, help="Thrust-to-speed factor.")
    parser.add_argument("--dt", type=float, default=1.0, help="Time step [s].")
    parser.add_argument(
        "--acceptance-radius", type=float, default=10.0, help="Waypoint acceptance [m]."
    )
    parser.add_argument("--polygon", default=None, help="Path to boundary polygon CSV.")
    parser.add_argument(
        "--waypoints",
        default=None,
        help="CLI waypoints 'lat,lon;lat,lon;...' (bypasses GUI).",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    parser.add_argument(
        "--num-sources",
        type=int,
        default=3,
        help="Number of random plume sources to generate.",
    )
    args = parser.parse_args()

    frame = ENUFrame(lat0=args.lat0, lon0=args.lon0)
    lats_poly, lons_poly = load_polygon(args.polygon, frame)

    # Convert polygon boundary to ENU coordinates and choose random plume sources
    x_poly, y_poly = frame.to_enu(lats_poly, lons_poly)
    rng = np.random.default_rng(args.seed)
    plume_sources = get_random_points_in_polygon(x_poly, y_poly, args.num_sources, rng)

    print("\nGenerated Plume Sources (ENU):")
    for idx, (px, py) in enumerate(plume_sources):
        print(f"  Source {idx + 1}: x={px:.2f} m, y={py:.2f} m")

    # Get waypoints
    if args.waypoints is not None:
        pts = parse_waypoints_cli(args.waypoints)
    else:
        try:
            pts = select_waypoints_gui(lats_poly, lons_poly)
        except Exception as e:
            print(f"\nWarning: Could not open GUI window due to: {e}")
            print("Falling back to headless input. Please enter waypoints at the prompt.")
            print("Format: lat,lon; lat,lon; ... (e.g. 25.909,-80.136; 25.910,-80.135)")
            user_input = input("Enter waypoints: ")
            pts = parse_waypoints_cli(user_input)

    if not pts:
        print("No waypoints selected. Exiting.")
        sys.exit(0)

    print("\nCaptured Waypoints (GPS):")
    for idx, (lat, lon) in enumerate(pts):
        print(f"  WP {idx + 1}: {lat:.8f}°N, {lon:.8f}°E")

    # Convert waypoints to ENU
    wp_lats = np.array([p[0] for p in pts])
    wp_lons = np.array([p[1] for p in pts])
    wp_east, wp_north = frame.to_enu(wp_lats, wp_lons)
    wp_enu = np.stack([wp_east, wp_north], axis=1)

    # Start kinematic simulation
    robot = UnicycleKinematics(
        k_thrust=args.k_thrust,
        x0=wp_enu[0, 0],
        y0=wp_enu[0, 1],
        heading0=0.0,
    )

    states, thrusts, headings = robot.drive_to_waypoints(
        wp_enu,
        speed_mps=args.speed,
        dt=args.dt,
        acceptance_radius=args.acceptance_radius,
    )

    # Generate synthetic observations along the trajectory
    rows = []
    t = 0.0

    for idx, (x, y, heading) in enumerate(states):
        # Sample tidal-drift position
        x_eff = x + TIDAL_U[0] * t
        y_eff = y + TIDAL_U[1] * t

        sal = _plume_salinity(np.array([x_eff]), np.array([y_eff]), plume_sources)[0]
        tmp = _front_temperature(np.array([x_eff]), np.array([y_eff]), plume_sources)[0]
        chl = _plume_chlorophyll(np.array([x_eff]), np.array([y_eff]), plume_sources)[0]

        lat, lon = frame.from_enu(np.array([x]), np.array([y]))

        # Get thrust for this step (last step has no active commands, we use final controls)
        cmd_idx = min(idx, len(thrusts) - 1)
        thrust_val = float(thrusts[cmd_idx]) if len(thrusts) > 0 else 0.0
        heading_val = float(headings[cmd_idx]) if len(headings) > 0 else float(heading)

        rows.append(
            {
                "Time": round(t, 2),
                "Latitude": round(float(lat[0]), 8),
                "Longitude": round(float(lon[0]), 8),
                "x_m": round(float(x), 3),
                "y_m": round(float(y), 3),
                "Heading (degrees Magnetic)": round(float(np.degrees(heading_val)), 4),
                "Thrust (% Thrust)": round(float(thrust_val), 2),
                "Salinity (PPT)": round(float(sal) + rng.normal(0.0, 0.05), 4),
                "Temperature (C)": round(float(tmp) + rng.normal(0.0, 0.05), 4),
                "Chlorophyll (ug/L)": round(max(0.0, float(chl) + rng.normal(0.0, 0.03)), 4),
            }
        )
        t += args.dt

    df = pd.DataFrame(rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print("\nSimulation complete!")
    print(f"Wrote {len(df)} rows → {out_path}")
    print(f"  Duration  : {df['Time'].max():.1f} s")
    print(f"  Lat range : [{df['Latitude'].min():.6f}, {df['Latitude'].max():.6f}] °")
    print(f"  Lon range : [{df['Longitude'].min():.6f}, {df['Longitude'].max():.6f}] °")
    print(f"  Salinity  : [{df['Salinity (PPT)'].min():.2f}, {df['Salinity (PPT)'].max():.2f}] PPT")
    print(
        f"  Temp      : [{df['Temperature (C)'].min():.2f}, {df['Temperature (C)'].max():.2f}] °C"
    )
    print(
        f"  Chlorophyll: [{df['Chlorophyll (ug/L)'].min():.2f}, "
        f"{df['Chlorophyll (ug/L)'].max():.2f}] µg/L"
    )


if __name__ == "__main__":
    main()
