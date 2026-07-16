"""
test_scripts/demo_interpolator_csv.py
=====================================
Demonstrates how FieldInterpolator reconstructs a scalar field from
scattered observations loaded from a CSV survey file.

Run::

    python test_scripts/demo_interpolator_csv.py data/raw/data.csv \
        --lat0 25.9095516 --lon0 -80.1373136 --field 'Temperature (C)'
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pde_slam.coords import ENUFrame
from pde_slam.interpolators import FieldInterpolator, SpatialGrid
from pde_slam.survey_loader import SurveyLoader

# ---------------------------------------------------------------------------
# Configuration & CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interpolate scattered survey data from CSV onto a regular grid.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("csv_path", type=Path, help="Path to the survey CSV file.")
    parser.add_argument("--lat0", type=float, required=True, help="ENU origin latitude [deg].")
    parser.add_argument("--lon0", type=float, required=True, help="ENU origin longitude [deg].")
    parser.add_argument(
        "--field",
        type=str,
        default="Temperature (C)",
        help="Scalar field column to interpolate.",
    )
    parser.add_argument("--nx", type=int, default=80, help="Grid points in East direction.")
    parser.add_argument("--ny", type=int, default=80, help="Grid points in North direction.")
    parser.add_argument(
        "--method",
        type=str,
        choices=["rbf", "spline"],
        default="rbf",
        help="Interpolation backend.",
    )
    parser.add_argument(
        "--kernel",
        type=str,
        default="",
        help="RBF kernel (e.g., thin_plate_spline, gaussian, multiquadric).",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.0,
        help="RBF smoothing factor (set > 0 if duplicate coordinates exist).",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="RBF epsilon parameter.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save figure without opening an interactive window.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    print("============================================================")
    print("PDE-SLAM — CSV Interpolator Demo")
    print("============================================================")

    # 1. Load data
    print("\n[1/3] Loading survey data …")
    frame = ENUFrame(lat0=args.lat0, lon0=args.lon0)
    loader = SurveyLoader(frame, field=args.field).load(args.csv_path)

    xy = loader.xy_obs
    vals = loader.values

    e_min, e_max = xy[:, 0].min(), xy[:, 0].max()
    n_min, n_max = xy[:, 1].min(), xy[:, 1].max()

    print(f"  CSV       : {args.csv_path}")
    print(f"  Field     : {args.field}")
    print(f"  N obs     : {loader.n_obs}")
    print(f"  East  [m] : [{e_min:+.1f}, {e_max:+.1f}]")
    print(f"  North [m] : [{n_min:+.1f}, {n_max:+.1f}]")
    print(f"  Values    : [{vals.min():.4f}, {vals.max():.4f}]")

    # 2. Setup grid and interpolate
    print("\n[2/3] Interpolating observations onto grid …")
    margin = 0.05
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

    kwargs = {}
    if args.method == "rbf":
        kwargs["kernel"] = args.kernel
        kwargs["smoothing"] = args.smoothing
        if args.epsilon is not None:
            kwargs["epsilon"] = args.epsilon

    interp = FieldInterpolator(grid, method=args.method, **kwargs)
    grid_field = interp.fit_predict(xy, vals)

    print(f"  Grid field range: [{float(grid_field.min()):.4f}, {float(grid_field.max()):.4f}]")

    # 3. Plotting
    print("\n[3/3] Plotting results …")
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    vmin, vmax = vals.min(), vals.max()
    cmap = "RdYlBu_r"

    ext = [grid.x_min, grid.x_max, grid.y_min, grid.y_max]
    im = ax.imshow(
        np.array(grid_field),
        origin="lower",
        extent=ext,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )
    # Overlay observation points
    sc = ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=vals,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        s=20,
        edgecolors="k",
        linewidths=0.4,
        alpha=0.8,
        label="Observations",
    )

    ax.set_title(f"{args.method.upper()} Interpolation: {args.field}")
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.legend(loc="upper right")
    plt.colorbar(im, ax=ax, label=args.field)

    out_name = f"demo_interpolator_csv_{args.method}_{args.field.replace(' ', '_').replace('(', '').replace(')', '')}.png"
    out_path = output_dir / out_name
    fig.savefig(out_path, dpi=150)
    print(f"  Saved → {out_path}")

    if not args.no_show:
        plt.show()

    print("\nDone.")


if __name__ == "__main__":
    main(sys.argv[1:])
