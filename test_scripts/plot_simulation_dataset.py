"""
test_scripts/plot_simulation_dataset.py
=======================================
Demonstration script that loads advection-diffusion PDE simulation datasets from
`data/adv_diff_simulations` and plots water features (e.g., ODO, Salinity,
Temperature, Chlorophyll) across different timestamps starting from t=0.

Key Features:
- Supports all simulation folders in `data/adv_diff_simulations/`.
- Evaluates multiple temporal snapshots starting from t=0s up to t_max.
- Renders each subplot with its own independent, nicely-scaled colorbar.
- Overlays the physical domain boundary polygon.

Usage::

    # Run on Biscayne simulation (headless)
    uv run python test_scripts/plot_simulation_dataset.py \
        --sim-dir data/adv_diff_simulations/biscayne_simulation --no-show

    # Run on Miami Canal simulation
    uv run python test_scripts/plot_simulation_dataset.py \
        --sim-dir data/adv_diff_simulations/miami_canal --no-show
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from pde_slam.io.simulation import load_simulation_dataset
from pde_slam.viz import (
    ensure_closed_polygon,
    get_feature_cmap,
    mask_field_grid,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot PDE simulation water features across timestamps."
    )
    parser.add_argument(
        "--sim-dir",
        type=Path,
        default=Path("data/adv_diff_simulations/biscayne_simulation"),
        help="Path to simulation directory containing .npz files.",
    )
    parser.add_argument(
        "--num-timestamps",
        type=int,
        default=5,
        help="Number of timestamps to plot across simulation duration.",
    )
    parser.add_argument(
        "--timestamps",
        type=float,
        nargs="+",
        default=None,
        help="Explicit list of timestamps in seconds to plot (e.g. --timestamps 0 50).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/graphs"),
        help="Directory to save generated figures (default: output/graphs/).",
    )

    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save figures to output-dir without opening interactive windows.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Main execution function."""
    args = parse_args(argv)

    print("==================================================")
    print("Advection-Diffusion Simulation Water Features Plotter")
    print(f"  Simulation Directory : {args.sim_dir}")
    print("==================================================")

    sim_data = load_simulation_dataset(args.sim_dir)
    avail_fields = sim_data.field_names
    print(f"Loaded {len(avail_fields)} water features: {avail_fields}")

    time_steps = sim_data.sample_times
    t_min = float(time_steps[0])
    t_max = float(time_steps[-1])
    n_total_steps = len(time_steps)
    print(f"Time range: {t_min:.1f} s -> {t_max:.1f} s ({n_total_steps} time steps)")

    # 1. Determine Timestamps to Plot (Starting from t=0)
    if args.timestamps is not None:
        eval_timestamps = [float(t) for t in args.timestamps]
    else:
        eval_timestamps = list(np.linspace(t_min, t_max, args.num_timestamps))

    if 0.0 not in eval_timestamps and t_min == 0.0:
        eval_timestamps = [0.0] + [t for t in eval_timestamps if t > 0.0]

    print(f"Selected timestamps to plot ({len(eval_timestamps)}):")
    for idx, t in enumerate(eval_timestamps):
        print(f"  [{idx + 1}] t = {t:.1f} s")

    poly_closed = ensure_closed_polygon(sim_data.polygon_enu)
    grid = sim_data.grid
    x_mesh, y_mesh = np.meshgrid(
        np.linspace(grid.x_min, grid.x_max, grid.nx),
        np.linspace(grid.y_min, grid.y_max, grid.ny),
        indexing="ij",
    )

    output_dir = args.output_dir
    output_dir.mkdir(exist_ok=True, parents=True)
    sim_name_clean = args.sim_dir.name

    # 2. Multi-Row Overview Figure: (N_fields rows x N_timestamps cols)
    n_rows = len(avail_fields)
    n_cols = len(eval_timestamps)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.0 * n_cols, 4.5 * n_rows),
        squeeze=False,
    )
    fig.suptitle(
        f"Advection-Diffusion Simulation Water Features ({sim_name_clean})\n"
        f"Temporal Evolution across {n_cols} Timestamps",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    for row_idx, f_name in enumerate(avail_fields):
        sols = sim_data.simulations[f_name]["solutions"]
        cmap_name = get_feature_cmap(f_name)

        for col_idx, t_eval in enumerate(eval_timestamps):
            ax = axes[row_idx, col_idx]
            t_span = t_max - t_min if t_max > t_min else 1.0
            step_idx = int(
                np.clip(
                    int((t_eval / t_span) * (n_total_steps - 1)),
                    0,
                    n_total_steps - 1,
                )
            )
            actual_t = float(time_steps[step_idx])
            snapshot = sols[step_idx]
            snapshot_masked = mask_field_grid(
                snapshot, mask=sim_data.mesh_mask, target_shape=(grid.nx, grid.ny)
            )

            ax.set_facecolor("#e9ecef")
            im = ax.pcolormesh(
                x_mesh,
                y_mesh,
                snapshot_masked,
                cmap=cmap_name,
                shading="auto",
            )

            if poly_closed is not None:
                lbl = "Domain Polygon" if (row_idx == 0 and col_idx == 0) else None
                ax.plot(
                    poly_closed[:, 0],
                    poly_closed[:, 1],
                    "r--",
                    linewidth=1.2,
                    alpha=0.75,
                    label=lbl,
                )

            t_label = (
                "t = 0.0 s (Initial)" if actual_t == 0.0 else f"t = {actual_t:.1f} s"
            )
            ax.set_title(
                f"{f_name.upper()} | {t_label}",
                fontsize=11,
                fontweight="bold",
            )

            ax.set_xlabel("East Position [m]", fontsize=9)
            ax.set_ylabel("North Position [m]", fontsize=9)
            ax.grid(True, linestyle=":", alpha=0.5)

            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(f_name, fontsize=9)
            cbar.ax.tick_params(labelsize=8)

            if row_idx == 0 and col_idx == 0 and poly_closed is not None:
                ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    overview_path = output_dir / f"adv_diff_{sim_name_clean}_overview.png"
    fig.savefig(overview_path, dpi=160, bbox_inches="tight")
    print(f"\n[+] Saved multi-feature overview figure to: {overview_path}")
    if not args.no_show:
        plt.show()
    plt.close(fig)

    # 3. Individual Figures for Each Water Feature
    print("\nGenerating individual high-resolution figures per water feature...")
    for f_name in avail_fields:
        sols = sim_data.simulations[f_name]["solutions"]
        cmap_name = get_feature_cmap(f_name)

        fig_field, ax_row = plt.subplots(
            1,
            n_cols,
            figsize=(5.2 * n_cols, 4.8),
            squeeze=False,
        )
        fig_field.suptitle(
            f"Advection-Diffusion Simulation: {f_name.upper()} ({sim_name_clean})\n"
            f"Evolution across Timestamps",
            fontsize=13,
            fontweight="bold",
            y=1.02,
        )

        for col_idx, t_eval in enumerate(eval_timestamps):
            ax = ax_row[0, col_idx]
            t_span = t_max - t_min if t_max > t_min else 1.0
            step_idx = int(
                np.clip(
                    int((t_eval / t_span) * (n_total_steps - 1)),
                    0,
                    n_total_steps - 1,
                )
            )
            actual_t = float(time_steps[step_idx])
            snapshot = sols[step_idx]
            snapshot_masked = mask_field_grid(
                snapshot, mask=sim_data.mesh_mask, target_shape=(grid.nx, grid.ny)
            )

            ax.set_facecolor("#e9ecef")
            im = ax.pcolormesh(
                x_mesh,
                y_mesh,
                snapshot_masked,
                cmap=cmap_name,
                shading="auto",
            )

            if poly_closed is not None:
                ax.plot(
                    poly_closed[:, 0],
                    poly_closed[:, 1],
                    "r--",
                    linewidth=1.2,
                    alpha=0.75,
                )

            t_label = (
                "t = 0.0 s (Initial)" if actual_t == 0.0 else f"t = {actual_t:.1f} s"
            )
            ax.set_title(
                f"{f_name.upper()} ({t_label})",
                fontsize=11,
                fontweight="bold",
            )

            ax.set_xlabel("East Position [m]", fontsize=9)
            ax.set_ylabel("North Position [m]", fontsize=9)
            ax.grid(True, linestyle=":", alpha=0.5)

            cbar = fig_field.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(f_name, fontsize=9)
            cbar.ax.tick_params(labelsize=8)

        plt.tight_layout()
        field_path = output_dir / f"adv_diff_{sim_name_clean}_{f_name}.png"
        fig_field.savefig(field_path, dpi=160, bbox_inches="tight")
        print(f"  [+] Saved {f_name} plot to: {field_path}")
        if not args.no_show:
            plt.show()
        plt.close(fig_field)

    print("\nAll plots generated successfully!")
    print("==================================================")


if __name__ == "__main__":
    main(sys.argv[1:])
