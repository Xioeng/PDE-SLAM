"""
test_scripts/demo_pipeline.py
==============================
End-to-end pipeline demo: CSV survey → ENU projection → field interpolation
→ PDE advection-diffusion solve → visualisation.

Usage::

    python test_scripts/demo_pipeline.py data/raw/survey.csv \\
        --lat0 36.7996 --lon0 -76.0 \\
        --field salinity_psu \\
        --nx 64 --ny 64 \\
        --t-end 300 \\
        --method rbf

Arguments
---------
csv_path    Path to the survey CSV file (required, positional).
--lat0      Latitude of the ENU origin [deg] (required).
--lon0      Longitude of the ENU origin [deg] (required).
--field     Scalar field column to use (default: salinity_psu).
--nx        Grid points in East direction (default: 64).
--ny        Grid points in North direction (default: 64).
--t-end     PDE integration end time in seconds (default: 300).
--method    Interpolation backend: rbf or spline (default: rbf).
--no-show   Save figure to outputs/ without opening a window.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from pde_slam.coords import ENUFrame
from pde_slam.interpolators import FieldInterpolator, SpatialGrid
from pde_slam.solver import AdvectionDiffusionSolver, PDEParams
from pde_slam.survey_loader import SurveyLoader

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PDE-SLAM end-to-end pipeline: CSV → ENU → interpolation → PDE solve.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("csv_path", type=Path, help="Path to the survey CSV file.")
    parser.add_argument("--lat0", type=float, required=True, help="ENU origin latitude [deg].")
    parser.add_argument("--lon0", type=float, required=True, help="ENU origin longitude [deg].")
    parser.add_argument("--field", default="Salinity (PPT)", help="Scalar field column to use.")
    parser.add_argument("--nx", type=int, default=64, help="Grid points in East direction.")
    parser.add_argument("--ny", type=int, default=64, help="Grid points in North direction.")
    parser.add_argument("--t-end", type=float, default=300.0, help="PDE solve end time [s].")
    parser.add_argument(
        "--method",
        choices=["rbf", "spline"],
        default="rbf",
        help="Interpolation backend.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save figure without opening an interactive window.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------


def step_load(args: argparse.Namespace) -> tuple[SurveyLoader, np.ndarray, np.ndarray]:
    """Step 1 – Load CSV and project to ENU metres."""
    print("\n[1/4] Loading survey data …")
    frame = ENUFrame(lat0=args.lat0, lon0=args.lon0)
    loader = SurveyLoader(frame, field=args.field).load(args.csv_path)

    xy = loader.xy_obs  # (N, 2) float64
    vals = loader.values  # (N,)   float64

    east_range = (xy[:, 0].min(), xy[:, 0].max())
    north_range = (xy[:, 1].min(), xy[:, 1].max())

    print(f"  CSV       : {args.csv_path}")
    print(f"  Field     : {args.field}")
    print(f"  N obs     : {loader.n_obs}")
    print(f"  ENU frame : {frame}")
    print(f"  East  [m] : [{east_range[0]:+.1f}, {east_range[1]:+.1f}]")
    print(f"  North [m] : [{north_range[0]:+.1f}, {north_range[1]:+.1f}]")
    print(f"  {args.field} : [{vals.min():.4f}, {vals.max():.4f}]")

    available = loader.available_fields()
    if len(available) > 1:
        others = [f for f in available if f != args.field]
        print(f"  Other fields available: {others}")

    return loader, xy, vals


def step_interpolate(
    args: argparse.Namespace,
    xy: np.ndarray,
    vals: np.ndarray,
) -> tuple[SpatialGrid, jnp.ndarray]:
    """Step 2 – Build grid and interpolate observations onto it."""
    print("\n[2/4] Interpolating observations onto grid …")

    # Auto-derive grid extent from observation bounding box with 5 % margin
    margin = 0.05
    e_min, e_max = xy[:, 0].min(), xy[:, 0].max()
    n_min, n_max = xy[:, 1].min(), xy[:, 1].max()
    e_pad = (e_max - e_min) * margin
    n_pad = (n_max - n_min) * margin

    grid = SpatialGrid(
        x_min=e_min - e_pad,
        x_max=e_max + e_pad,
        y_min=n_min - n_pad,
        y_max=n_max + n_pad,
        nx=args.nx,
        ny=args.ny,
    )

    print(f"  Grid      : {grid}")
    print(f"  Method    : {args.method}")

    kernel_args = {"kernel": "thin_plate_spline", "epsilon": 1.0, "smoothing": 1.0}
    interp = FieldInterpolator(grid, method=args.method, **kernel_args)
    phi0 = interp.fit_predict(xy, vals)

    print(f"  phi0 range: [{float(phi0.min()):.4f}, {float(phi0.max()):.4f}]")
    return grid, phi0


def step_solve(
    args: argparse.Namespace,
    grid: SpatialGrid,
    phi0: jnp.ndarray,
) -> tuple[AdvectionDiffusionSolver, jnp.ndarray, list[float]]:
    """Step 3 – Run the PDE solver forward in time."""
    print("\n[3/4] Running PDE solver …")

    # Zero-flow, mild diffusion (sensible default without velocity data)
    U_dir = 0.1 * jnp.array([1.0, -1.0])
    u_field = U_dir * np.ones((grid.ny, grid.nx, 2))
    diffusivity = jnp.array(0.1, dtype=jnp.float32)  # m² s⁻¹

    params = PDEParams(u_field=u_field, D=diffusivity)

    # Choose dt_max so diffusion number < 0.5
    solver = AdvectionDiffusionSolver(grid, dt_max=1.0)
    diff_n = solver.diffusion_number(diffusivity, 1.0)
    courant_n = solver.courant_number(u_field, 1.0)
    print(f"  Courant number : {float(courant_n):.4f}  (≤ 1 required)")
    print(f"  Diffusion number: {float(diff_n):.4f}  (≤ 0.5 required)")

    if float(diff_n) > 0.5:
        print("  ⚠  Diffusion number > 0.5 — reducing dt_max to 0.1 s")
        solver = AdvectionDiffusionSolver(grid, dt_max=0.1)

    t_snaps = [0.0, args.t_end / 3, 2 * args.t_end / 3, args.t_end]
    print(f"  Snapshots at  : {[f'{t:.0f} s' for t in t_snaps]}")

    snapshots = solver.solve(phi0, params, t0=0.0, t_end=args.t_end, saveat=t_snaps)
    print("  Solve complete.")
    return solver, snapshots, t_snaps


def step_plot(
    args: argparse.Namespace,
    grid: SpatialGrid,
    phi0: jnp.ndarray,
    snapshots: jnp.ndarray,
    xy: np.ndarray,
    vals: np.ndarray,
    t_snaps: list[float],
) -> None:
    """Step 4 – Visualise initial condition + PDE evolution."""
    print("\n[4/4] Plotting …")

    vmin = float(jnp.min(snapshots))
    vmax = float(jnp.max(snapshots))
    cmap = "RdYlBu_r"
    ext = [grid.x_min, grid.x_max, grid.y_min, grid.y_max]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)
    fig.suptitle(
        f"PDE-SLAM pipeline — {args.field} from {args.csv_path.name}",
        fontsize=13,
        fontweight="bold",
    )

    for i, (ax, snap, t) in enumerate(zip(axes, snapshots, t_snaps, strict=True)):
        im = ax.imshow(np.array(snap), origin="lower", extent=ext, vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_title(f"t = {int(t)} s")
        ax.set_xlabel("East [m]")
        if i == 0:
            ax.set_ylabel("North [m]")
            # Overlay raw observations on the initial condition panel
            ax.scatter(
                xy[:, 0],
                xy[:, 1],
                c=vals,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                s=6,
                edgecolors="k",
                linewidths=0.3,
                zorder=5,
                label="Observations",
            )
            ax.legend(loc="upper right", fontsize=8)
        plt.colorbar(im, ax=ax, shrink=0.85, label=args.field)

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    stem = args.csv_path.stem
    out_path = output_dir / f"demo_pipeline_{stem}_{args.field}.png"
    fig.savefig(out_path, dpi=150)
    print(f"  Saved → {out_path}")

    if not args.no_show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    print("=" * 60)
    print("PDE-SLAM — End-to-End Pipeline Demo")
    print("=" * 60)

    loader, xy, vals = step_load(args)
    grid, phi0 = step_interpolate(args, xy, vals)
    solver, snapshots, t_snaps = step_solve(args, grid, phi0)
    step_plot(args, grid, phi0, snapshots, xy, vals, t_snaps)

    print("\nDone.")


if __name__ == "__main__":
    main(sys.argv[1:])
