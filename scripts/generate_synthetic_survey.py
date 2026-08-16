"""
scripts/generate_synthetic_survey.py
=====================================
Generate a synthetic survey CSV driven by the :class:`~pde_slam.kinematics.UnicycleKinematics`
robot model, with GPS coordinates and realistic Biscayne Bay (Miami, FL) scalar fields.

Robot model
-----------
The vehicle is a thrust-controlled, compass-heading robot.  The thrust command
is derived from the desired survey speed and the ``k_thrust`` parameter
(m s⁻¹ per unit thrust).  Heading is set by a perfect compass; position is
integrated by :class:`~pde_slam.kinematics.UnicycleKinematics`.

Survey track
------------
Lawnmower pattern: parallel east-west passes connected by northward turns.
Each row in the output is a sensor sample recorded at the robot's ENU position
(before the position is advanced by one integration step).

Scalar fields (Biscayne Bay, July)
------------------------------------
* **Salinity** [PSU] — canal/Everglades freshwater plume (SW origin),
  ambient 36 PSU, fresh end-member 12 PSU.
* **Temperature** [°C] — warm-shallow-flats intrusion from the east,
  ambient 30.0 °C, peak 32.5 °C.
* **Chlorophyll-a** [µg/L] — bloom correlated with the salinity plume,
  ambient 0.5 µg/L, plume peak 5.0 µg/L.

GPS output
----------
ENU (x_m, y_m) positions are converted to geodetic (lat_deg, lon_deg) via
:class:`~pde_slam.coords.ENUFrame` with a user-configurable survey origin
(default: Biscayne Bay centre, 25.9096°N 80.1366°W).

Output columns
--------------
``t_s, lat_deg, lon_deg, x_m, y_m, heading_rad, thrust,
salinity_psu, temperature_c, chlorophyll_ug_l``

Usage::

    python scripts/generate_synthetic_survey.py
    python scripts/generate_synthetic_survey.py \\
        --lat0 25.909619867836824 --lon0 -80.13657451246902 \\
        --k-thrust 1.5 --speed 1.0 --output data/raw/survey.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path so pde_slam is importable without package installation
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from pde_slam.coords import ENUFrame
from pde_slam.kinematics import UnicycleKinematics

# ---------------------------------------------------------------------------
# Default survey origin — Biscayne Bay, Miami FL
# ---------------------------------------------------------------------------
DEFAULT_LAT0 = 25.909619867836824
DEFAULT_LON0 = -80.13657451246902

# ---------------------------------------------------------------------------
# Physical parameters — Biscayne Bay, Miami FL (July)
# ---------------------------------------------------------------------------

# Tidal mean current [m/s] — flood tide, broadly northward in the bay
TIDAL_U = np.array([0.03, 0.05])  # (east, north) m/s

# Canal / Everglades freshwater plume — SW origin
PLUME_SOURCE = np.array([-220.0, -220.0])
AMBIENT_SALINITY = 36.0  # PSU — slightly hypersaline in the dry season
FRESH_SALINITY = 12.0  # PSU — Everglades / canal freshwater end-member
PLUME_WIDTH = 80.0  # m,  Gaussian half-width at source
PLUME_DECAY = 0.004  # m⁻¹, along-plume exponential mixing decay

# Temperature: warm shallow flats intrusion from the east
AMBIENT_TEMP = 30.0  # °C — mean bay surface, July
INTRUSION_TEMP = 32.5  # °C — very shallow eastern flats
FRONT_X = 80.0  # m,  position of thermal front
FRONT_WIDTH = 40.0  # m,  sigmoid transition half-width

# Chlorophyll-a bloom correlated with the freshwater plume
CHL_AMBIENT = 0.5  # µg/L — open bay background
CHL_PLUME_PEAK = 5.0  # µg/L — at the plume core near source


# ---------------------------------------------------------------------------
# Synthetic scalar field functions
# ---------------------------------------------------------------------------


def _plume_salinity(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Salinity field [PSU] — Everglades/canal freshwater plume.

    Parameters
    ----------
    x, y :
        ENU east and north positions [m].

    Returns
    -------
    salinity :
        Salinity values at each (x, y) point [PSU].
    """
    plume_dir = np.array([1.0, 1.0]) / np.sqrt(2.0)
    dx = x - PLUME_SOURCE[0]
    dy = y - PLUME_SOURCE[1]

    along = dx * plume_dir[0] + dy * plume_dir[1]  # distance along jet axis
    cross = -dx * plume_dir[1] + dy * plume_dir[0]  # cross-jet distance

    along_clamped = np.maximum(along, 0.0)
    width = PLUME_WIDTH + 0.15 * along_clamped
    freshness = np.exp(-0.5 * (cross / width) ** 2) * np.exp(-PLUME_DECAY * along_clamped)
    return AMBIENT_SALINITY - (AMBIENT_SALINITY - FRESH_SALINITY) * freshness


def _front_temperature(x: np.ndarray, _y: np.ndarray) -> np.ndarray:
    """Temperature field [°C] — warm shallow-flats intrusion from the east.

    Parameters
    ----------
    x :
        ENU east position [m].
    _y :
        ENU north position [m] (unused; retained for API uniformity).

    Returns
    -------
    temperature :
        Temperature values at each x position [°C].
    """
    return AMBIENT_TEMP + (INTRUSION_TEMP - AMBIENT_TEMP) / (
        1.0 + np.exp((x - FRONT_X) / FRONT_WIDTH)
    )


def _plume_chlorophyll(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Chlorophyll-a field [µg/L] — bloom correlated with freshwater plume.

    The bloom peaks slightly *downstream* of the salinity minimum to mimic
    the biological lag between nutrient injection and phytoplankton growth.

    Parameters
    ----------
    x, y :
        ENU east and north positions [m].

    Returns
    -------
    chl :
        Chlorophyll-a concentrations at each (x, y) point [µg/L].
    """
    plume_dir = np.array([1.0, 1.0]) / np.sqrt(2.0)
    dx = x - PLUME_SOURCE[0]
    dy = y - PLUME_SOURCE[1]

    along = dx * plume_dir[0] + dy * plume_dir[1]
    cross = -dx * plume_dir[1] + dy * plume_dir[0]

    along_clamped = np.maximum(along, 0.0)
    # Wider Gaussian + delayed decay to place bloom slightly downstream
    width = PLUME_WIDTH * 1.2 + 0.2 * along_clamped
    bloom = np.exp(-0.5 * (cross / width) ** 2) * np.exp(
        -PLUME_DECAY * 0.5 * np.maximum(along_clamped - 50.0, 0.0)
    )
    return CHL_AMBIENT + (CHL_PLUME_PEAK - CHL_AMBIENT) * bloom


# ---------------------------------------------------------------------------
# Survey generation
# ---------------------------------------------------------------------------


def generate_survey(
    lat0: float = DEFAULT_LAT0,
    lon0: float = DEFAULT_LON0,
    n_passes: int = 8,
    pass_spacing_m: float = 60.0,
    domain_half: float = 240.0,
    speed_mps: float = 1.0,
    dt_s: float = 1.0,
    k_thrust: float = 1.0,
    noise_salinity: float = 0.05,
    noise_temp: float = 0.05,
    noise_chl: float = 0.03,
    rng_seed: int = 0,
) -> pd.DataFrame:
    """Generate a lawnmower survey driven by :class:`~pde_slam.kinematics.UnicycleKinematics`.

    The robot executes east-west passes connected by northward turns.
    Position is integrated by the kinematic model at every ``dt_s`` step;
    GPS coordinates are derived from the ENU frame.

    Parameters
    ----------
    lat0, lon0 :
        Geodetic origin of the ENU frame (survey centre) [degrees].
        Defaults to Biscayne Bay, Miami FL.
    n_passes :
        Number of parallel east-west survey lines.
    pass_spacing_m :
        North spacing between consecutive survey lines [m].
    domain_half :
        Half-extent of the domain in the east direction [m].
        Lines span ``[-domain_half, +domain_half]``.
    speed_mps :
        Desired robot speed along survey lines [m s⁻¹].
    dt_s :
        Sampling / integration interval [s].
    k_thrust :
        Thrust-to-speed factor [m s⁻¹ per unit thrust].  The constant thrust
        command is ``(speed_mps / k_thrust) * 100.0`` and must be ≤ 100.
    noise_salinity :
        Gaussian sensor noise standard deviation for salinity [PSU].
    noise_temp :
        Gaussian sensor noise standard deviation for temperature [°C].
    noise_chl :
        Gaussian sensor noise standard deviation for chlorophyll [µg/L].
    rng_seed :
        RNG seed for reproducibility.

    Returns
    -------
    df :
        DataFrame with columns:
        ``t_s, lat_deg, lon_deg, x_m, y_m, heading_rad, thrust,
        salinity_psu, temperature_c, chlorophyll_ug_l``.

    Raises
    ------
    ValueError
        If ``speed_mps / k_thrust > 1.0``.
    """
    thrust_cmd = (speed_mps / k_thrust) * 100.0
    if thrust_cmd > 100.0:
        raise ValueError(
            f"speed_mps / k_thrust = {speed_mps / k_thrust:.3f} > 1.0 — "
            "reduce speed_mps or increase k_thrust."
        )

    frame = ENUFrame(lat0=lat0, lon0=lon0)
    rng = np.random.default_rng(rng_seed)

    # Navigation headings (convention: 0=North, π/2=East, CW positive)
    HDG_EAST = np.pi / 2.0  # 90°
    HDG_WEST = 3.0 * np.pi / 2.0  # 270°
    HDG_NORTH = 0.0  # 0°

    # Y-positions of each pass (centred around y=0)
    y_passes = [-domain_half + i * pass_spacing_m for i in range(n_passes)]
    y_passes = [y for y in y_passes if y <= domain_half]

    # Number of integration steps per segment
    n_steps_pass = max(1, round(2 * domain_half / (speed_mps * dt_s)))
    n_steps_turn = max(1, round(pass_spacing_m / (speed_mps * dt_s)))

    # Initialise robot at SW corner, heading east
    robot = UnicycleKinematics(
        k_thrust=k_thrust,
        x0=-domain_half,
        y0=y_passes[0],
        heading0=HDG_EAST,
    )

    rows: list[dict] = []
    t = 0.0

    for pass_idx, _y_track in enumerate(y_passes):
        heading = HDG_EAST if pass_idx % 2 == 0 else HDG_WEST

        for _step in range(n_steps_pass):
            x = robot.x_m
            y = robot.y_m

            # Sample scalar fields at tidally-advected position
            x_eff = x + TIDAL_U[0] * t
            y_eff = y + TIDAL_U[1] * t

            sal = _plume_salinity(np.array([x_eff]), np.array([y_eff]))[0]
            tmp = _front_temperature(np.array([x_eff]), np.array([y_eff]))[0]
            chl = _plume_chlorophyll(np.array([x_eff]), np.array([y_eff]))[0]

            lat, lon = frame.from_enu(np.array([x]), np.array([y]))

            rows.append(
                {
                    "Time": round(t, 2),
                    "Latitude": round(float(lat[0]), 8),
                    "Longitude": round(float(lon[0]), 8),
                    "x_m": round(x, 3),
                    "y_m": round(y, 3),
                    "Heading (degrees Magnetic)": round(float(np.degrees(robot.heading_rad)), 4),
                    "Thrust (% Thrust)": round(float(thrust_cmd), 2),
                    "Salinity (PPT)": round(float(sal) + rng.normal(0.0, noise_salinity), 4),
                    "Temperature (C)": round(float(tmp) + rng.normal(0.0, noise_temp), 4),
                    "Chlorophyll (ug/L)": round(
                        max(0.0, float(chl) + rng.normal(0.0, noise_chl)), 4
                    ),
                }
            )
            t += dt_s
            robot.step(thrust_cmd, heading, dt_s)

        # Execute northward turn to next pass (no measurements during turn)
        if pass_idx < len(y_passes) - 1:
            for _step in range(n_steps_turn):
                robot.step(thrust_cmd, HDG_NORTH, dt_s)
                t += dt_s

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry-point for the survey generator script."""
    parser = argparse.ArgumentParser(
        description="Generate a synthetic Biscayne Bay survey log with GPS coordinates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output", default="data/raw/survey.csv", help="Output CSV path.")
    parser.add_argument(
        "--lat0",
        type=float,
        default=DEFAULT_LAT0,
        help="Survey origin latitude [deg].",
    )
    parser.add_argument(
        "--lon0",
        type=float,
        default=DEFAULT_LON0,
        help="Survey origin longitude [deg].",
    )
    parser.add_argument("--n-passes", type=int, default=8, help="Number of survey lines.")
    parser.add_argument("--speed", type=float, default=1.0, help="Robot speed [m/s].")
    parser.add_argument(
        "--k-thrust",
        type=float,
        default=1.0,
        help="Thrust-to-speed factor [m/s per unit thrust].",
    )
    parser.add_argument("--dt", type=float, default=1.0, help="Sampling interval [s].")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = generate_survey(
        lat0=args.lat0,
        lon0=args.lon0,
        n_passes=args.n_passes,
        speed_mps=args.speed,
        k_thrust=args.k_thrust,
        dt_s=args.dt,
        rng_seed=args.seed,
    )
    df.to_csv(out, index=False)

    print(f"Wrote {len(df)} rows → {out}")
    print(f"  Origin    : ({args.lat0:.6f}°N, {args.lon0:.6f}°E)")
    print(f"  Duration  : {df['Time'].max():.1f} s")
    print(f"  Lat range : [{df['Latitude'].min():.6f}, {df['Latitude'].max():.6f}] °")
    print(f"  Lon range : [{df['Longitude'].min():.6f}, {df['Longitude'].max():.6f}] °")
    print(f"  Salinity  : [{df['Salinity (PPT)'].min():.2f}, {df['Salinity (PPT)'].max():.2f}] PPT")
    print(
        f"  Temp      : [{df['Temperature (C)'].min():.2f}, {df['Temperature (C)'].max():.2f}] °C"
    )
    print(
        f"  Chlorophyll: [{df['Chlorophyll (ug/L)'].min():.2f},"
        f" {df['Chlorophyll (ug/L)'].max():.2f}] µg/L"
    )


if __name__ == "__main__":
    main()
