"""
scripts/generate_synthetic_survey.py
=====================================
Generate a synthetic Phase 1 survey CSV with realistic water features.

The domain is a 500 × 500 m metric patch (ENU coordinates, origin at
centre).  The synthetic scalar fields mimic common coastal/estuarine
water features:

Salinity
    A **river plume** emanating from the south-west corner, modelled as
    a Gaussian jet that decays exponentially with distance from the
    plume centreline.  Salinity is low near the source (fresh water)
    and rises toward the ambient ocean value away from it.

Temperature
    A **warm-water intrusion** from the east boundary, implemented as a
    sigmoid front, plus a gentle large-scale gradient mimicking solar
    heating of shallow near-shore water.

Advection
    The underlying flow field is a mean tidal current directed north-east
    (0.08 m s⁻¹) plus a weak anti-clockwise eddy centred at (100, 50) m.
    The vehicle samples the field at its instantaneous position as it
    moves through this evolving flow.

Survey track
    Lawnmower pattern: 8 parallel east-west passes spaced 60 m apart,
    at 0.4 m s⁻¹, sampled every 1 s.

Output
------
CSV with columns: ``t_s, x_m, y_m, salinity_psu, temperature_c``

Usage::

    python scripts/generate_synthetic_survey.py
    python scripts/generate_synthetic_survey.py --output data/raw/survey.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# True physical parameters
# ---------------------------------------------------------------------------

# Tidal mean current [m/s]
TIDAL_U = np.array([0.06, 0.05])

# River plume source position [m] and fresh-water salinity
PLUME_SOURCE = np.array([-220.0, -220.0])
AMBIENT_SALINITY = 34.5   # PSU (ocean background)
FRESH_SALINITY   = 18.0   # PSU (river end-member)
PLUME_WIDTH      = 80.0   # m, Gaussian half-width of plume cross-section
PLUME_DECAY      = 0.004  # m⁻¹, along-plume exponential decay

# Warm intrusion
AMBIENT_TEMP    = 16.0  # °C
INTRUSION_TEMP  = 22.0  # °C
FRONT_X         = 80.0  # m, position of thermal front (east of centre)
FRONT_WIDTH     = 40.0  # m, sigmoid transition width


# ---------------------------------------------------------------------------
# Synthetic field functions
# ---------------------------------------------------------------------------


def _plume_salinity(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """River plume salinity field [PSU].

    Low-salinity jet originates at PLUME_SOURCE, travels north-east,
    widening and mixing with ambient water along the way.
    """
    plume_dir = np.array([1.0, 1.0]) / np.sqrt(2.0)
    dx = x - PLUME_SOURCE[0]
    dy = y - PLUME_SOURCE[1]

    along = dx * plume_dir[0] + dy * plume_dir[1]   # distance along jet axis
    cross = -dx * plume_dir[1] + dy * plume_dir[0]  # cross-jet distance

    # Plume only exists downstream (along > 0)
    along_clamped = np.maximum(along, 0.0)

    # Cross-jet Gaussian profile widens with distance
    width = PLUME_WIDTH + 0.15 * along_clamped
    freshness = np.exp(-0.5 * (cross / width) ** 2) * np.exp(-PLUME_DECAY * along_clamped)

    return AMBIENT_SALINITY - (AMBIENT_SALINITY - FRESH_SALINITY) * freshness


def _front_temperature(x: np.ndarray, _y: np.ndarray) -> np.ndarray:
    """Warm-water intrusion from the east [°C], sigmoid front."""
    return AMBIENT_TEMP + (INTRUSION_TEMP - AMBIENT_TEMP) / (
        1.0 + np.exp((x - FRONT_X) / FRONT_WIDTH)
    )


# ---------------------------------------------------------------------------
# Survey generation
# ---------------------------------------------------------------------------


def generate_survey(
    n_passes: int = 8,
    pass_spacing_m: float = 60.0,
    domain_half: float = 240.0,
    speed_mps: float = 0.4,
    dt_s: float = 1.0,
    noise_salinity: float = 0.05,
    noise_temp: float = 0.02,
    rng_seed: int = 0,
) -> pd.DataFrame:
    """Generate a lawnmower survey DataFrame.

    Parameters
    ----------
    n_passes :
        Number of parallel survey lines.
    pass_spacing_m :
        North spacing between survey lines [m].
    domain_half :
        Half-width of the domain [m]; lines span ±domain_half in x.
    speed_mps :
        Vehicle speed [m s⁻¹].
    dt_s :
        Sampling interval [s].
    noise_salinity, noise_temp :
        Gaussian sensor noise standard deviations.
    rng_seed :
        RNG seed for reproducibility.

    Returns
    -------
    df :
        DataFrame with columns ``t_s, x_m, y_m, salinity_psu, temperature_c``.
    """
    rng = np.random.default_rng(rng_seed)
    rows: list[dict] = []
    t = 0.0

    for i in range(n_passes):
        y_track = -domain_half + i * pass_spacing_m
        if y_track > domain_half:
            break

        x_start = -domain_half if i % 2 == 0 else domain_half
        x_end   =  domain_half if i % 2 == 0 else -domain_half
        n_steps = max(1, int(2 * domain_half / (speed_mps * dt_s)))

        for step in range(n_steps):
            alpha = step / max(n_steps - 1, 1)
            x = x_start + (x_end - x_start) * alpha

            # Current position after tidal drift from t=0
            x_eff = x + TIDAL_U[0] * t
            y_eff = y_track + TIDAL_U[1] * t

            sal = _plume_salinity(np.array([x_eff]), np.array([y_eff]))[0]
            tmp = _front_temperature(np.array([x_eff]), np.array([y_eff]))[0]

            rows.append({
                "t_s":           round(t, 2),
                "x_m":           round(float(x), 3),
                "y_m":           round(float(y_track), 3),
                "salinity_psu":  round(float(sal) + rng.normal(0.0, noise_salinity), 4),
                "temperature_c": round(float(tmp) + rng.normal(0.0, noise_temp), 4),
            })
            t += dt_s

        t += 8.0  # turning time between passes

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic water-feature survey log (metric units)."
    )
    parser.add_argument("--output", default="data/raw/survey.csv")
    parser.add_argument("--n_passes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = generate_survey(n_passes=args.n_passes, rng_seed=args.seed)
    df.to_csv(out, index=False)

    print(f"Wrote {len(df)} rows → {out}")
    print(f"  Duration  : {df['t_s'].max():.1f} s")
    print(f"  Salinity  : [{df['salinity_psu'].min():.2f}, {df['salinity_psu'].max():.2f}] PSU")
    print(f"  Temp      : [{df['temperature_c'].min():.2f}, {df['temperature_c'].max():.2f}] °C")


if __name__ == "__main__":
    main()
