"""
test_scripts/spatiotemporal_interpolator.py
===========================================
Demonstrates the SpatiotemporalInterpolator by generating space-time
snapshots from an exact analytical advection-diffusion plume and sampling
continuous virtual sensor observations along a robot trajectory.

Usage::

    python3 test_scripts/spatiotemporal_interpolator.py --no-show
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from pde_slam.interpolators import SpatialGrid, SpatiotemporalInterpolator


def analytical_plume_field(
    grid: SpatialGrid,
    t: float,
    v_flow: tuple[float, float] = (0.4, 0.2),
    D: float = 0.25,
    center0: tuple[float, float] = (-80.0, -60.0),
    sigma0: float = 45.0,
    ambient: float = 34.5,
    peak: float = 20.0,
) -> np.ndarray:
    """Evaluate exact analytical advection-diffusion plume on a spatial grid."""
    vx, vy = v_flow
    x0, y0 = center0
    sigma_t_sq = sigma0**2 + 2.0 * D * t

    cx_t = x0 + vx * t
    cy_t = y0 + vy * t

    r2 = (grid.XX - cx_t) ** 2 + (grid.YY - cy_t) ** 2
    amplitude_ratio = (sigma0**2) / sigma_t_sq
    plume = (ambient - peak) * amplitude_ratio * np.exp(-r2 / (2.0 * sigma_t_sq))
    return ambient - plume


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SpatiotemporalInterpolator Analytical Plume Demo"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        default=False,
        help="Run non-interactively without displaying GUI plot",
    )
    args = parser.parse_args()

    # 1. Setup Grid & Analytical Snapshots
    domain = 200.0
    nx = ny = 60
    grid = SpatialGrid(
        x_min=-domain, x_max=domain, y_min=-domain, y_max=domain, nx=nx, ny=ny
    )

    t_max = 200.0
    t_snapshots = jnp.linspace(0.0, t_max, 21)

    print(f"Generating {len(t_snapshots)} analytical space-time snapshots...")
    snapshots_list = [
        analytical_plume_field(grid, float(t_val)) for t_val in t_snapshots
    ]
    snapshots = jnp.array(snapshots_list)  # (n_time, ny, nx)

    # 2. Instantiate SpatiotemporalInterpolator
    interpolator = SpatiotemporalInterpolator(
        grid=grid,
        ts=t_snapshots,
        snapshots=snapshots,
    )
    print("SpatiotemporalInterpolator instantiated successfully.")

    # 3. Simulate Robot Trajectory and Query Continuous Sensor Values
    t_robot = jnp.linspace(0.0, t_max, 100)
    # Circle/spiral survey path
    r_path = 80.0
    omega = 2.0 * np.pi / t_max
    x_robot = r_path * jnp.cos(omega * t_robot)
    y_robot = r_path * jnp.sin(omega * t_robot)

    sampled_values = interpolator(x_robot, y_robot, t_robot)
    print(f"Sampled {len(sampled_values)} continuous trajectory observations.")

    # 4. Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Spatial Snapshot with Path
    im = ax1.pcolormesh(
        grid.XX, grid.YY, snapshots[len(snapshots) // 2], cmap="viridis", shading="auto"
    )
    ax1.plot(x_robot, y_robot, "r--", linewidth=1.5, label="Robot Track")
    ax1.set_title(
        f"Salinity Plume (t = {float(t_snapshots[len(snapshots) // 2]):.0f}s)"
    )
    ax1.set_xlabel("East [m]")
    ax1.set_ylabel("North [m]")
    ax1.legend(loc="upper left")
    plt.colorbar(im, ax=ax1, label="Salinity [PSU]")

    # Time-series along track
    ax2.plot(t_robot, sampled_values, "b-", linewidth=2.0, label="Sampled Salinity")
    ax2.set_title("Continuous Virtual Sensor Measurements along Track")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Salinity [PSU]")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    fig.tight_layout()
    out_file = Path("output/graphs") / "spatiotemporal_interpolation_demo.png"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=200)
    print(f"Saved demo figure to {out_file}")

    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
