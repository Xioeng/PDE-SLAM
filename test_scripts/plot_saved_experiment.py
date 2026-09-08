"""
test_scripts/plot_saved_experiment.py
=====================================
Publication-grade visualizer for saved PDE-SLAM experiment PKL datasets.

Renders:
1. Standalone / Combined Trajectory Paths & RMSE Tracking Error curves.
2. Ground Truth vs Online PINN Multi-Stage Evolution Grids (with satellite backdrops).
3. Space-Time PDE Physics Residual Grids (with Euclidean multi-field L2 norm).
4. Individual standalone high-resolution figure panels.

Usage::

    # Default run with auto-detected sample experiment
    python3 test_scripts/plot_saved_experiment.py

    # Specific experiment file
    python3 test_scripts/plot_saved_experiment.py \
        --file output/results/biscayne_simulation_rbpf_slam_experiment.pkl

    # Save to custom folder without opening GUI windows
    python3 test_scripts/plot_saved_experiment.py \
        --file output/results/biscayne_simulation_rbpf_slam_experiment.pkl --no-show
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt

from pde_slam.io.experiment import load_experiment
from pde_slam.viz.plotter import plot_saved_experiment
from pde_slam.viz.style import (
    FIGSIZE_FIELD_PANEL,
    FIGSIZE_GRID_PER_ROW,
    FIGSIZE_GRID_WIDTH,
    FIGSIZE_PATHS,
    FIGSIZE_RMSE,
)


def find_default_experiment_file(base_dir: Path) -> Path | None:
    """Auto-discover a saved experiment .pkl file if none is specified."""
    res_dir = base_dir / "output" / "results"
    preferred = [
        res_dir / "biscayne_simulation_rbpf_slam_experiment.pkl",
        res_dir / "miami_canal_rbpf_slam_experiment.pkl",
        res_dir / "toy_simulation_rbpf_slam_experiment.pkl",
        res_dir / "sample_experiment.pkl",
    ]

    for p in preferred:
        if p.is_file():
            return p

    results_dir = base_dir / "output" / "results"
    pkl_files = list(results_dir.glob("*.pkl")) if results_dir.exists() else []
    if not pkl_files:
        output_dir = base_dir / "output"
        pkl_files = list(output_dir.glob("*.pkl")) if output_dir.exists() else []
    return pkl_files[0] if pkl_files else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publication visualizer for saved PDE-SLAM experiments"
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to saved .pkl experiment data file",
    )
    parser.add_argument(
        "--sim-name",
        type=str,
        default=None,
        help="Custom simulation name for output folder (default: infer from pkl)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output figures directory (default: figures/{sim_name}/)",
    )
    parser.add_argument(
        "--zoom",
        type=int,
        default=18,
        help="Satellite imagery zoom level (default: 18)",
    )
    parser.add_argument(
        "--figsize-path",
        type=float,
        nargs=2,
        default=list(FIGSIZE_PATHS),
        help="Width and height for standalone path plot",
    )
    parser.add_argument(
        "--figsize-rmse",
        type=float,
        nargs=2,
        default=list(FIGSIZE_RMSE),
        help="Width and height for standalone RMSE tracking error plot",
    )
    parser.add_argument(
        "--figsize-panel",
        type=float,
        nargs=2,
        default=list(FIGSIZE_FIELD_PANEL),
        help="Width and height for individual field & residual panels",
    )
    parser.add_argument(
        "--figsize-grid-width",
        type=float,
        default=FIGSIZE_GRID_WIDTH,
        help="Width for multi-stage evolution grids",
    )
    parser.add_argument(
        "--figsize-grid-row-height",
        type=float,
        default=FIGSIZE_GRID_PER_ROW,
        help="Height per row for evolution grids",
    )
    parser.add_argument(
        "--save-grids",
        action="store_true",
        default=True,
        help="Save composite multi-panel grids (default: True)",
    )
    parser.add_argument(
        "--no-grids",
        action="store_false",
        dest="save_grids",
        help="Disable saving composite grids",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        default=False,
        help="Run non-interactively without displaying GUI plot windows",
    )

    args = parser.parse_args()
    root_dir = Path(__file__).resolve().parents[1]

    exp_path = Path(args.file) if args.file else find_default_experiment_file(root_dir)

    if exp_path is None or not exp_path.is_file():
        print(
            f"Error: Experiment file not found. Specified: '{args.file}'",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading experiment data from {exp_path}...")
    data = load_experiment(exp_path)
    sim_name = (
        args.sim_name
        or data.sim_name
        or exp_path.stem.replace("_experiment", "").replace("_rbpf_slam", "")
    )

    out_folder = (
        Path(args.output_dir)
        if args.output_dir
        else root_dir / "output" / "figures" / sim_name
    )
    print(f"Output figures directory: {out_folder}/\n")

    saved_files = plot_saved_experiment(
        experiment=data,
        output_dir=out_folder,
        sim_name=sim_name,
        save_grids=args.save_grids,
        save_individual=True,
        zoom=args.zoom,
        figsize_paths=tuple(args.figsize_path),
        figsize_rmse=tuple(args.figsize_rmse),
        figsize_panel=tuple(args.figsize_panel),
        figsize_grid_width=args.figsize_grid_width,
        figsize_grid_row_height=args.figsize_grid_row_height,
    )

    print("\n" + "=" * 50)
    print("Visualizations Generated Successfully!")
    print(f"Total Figures Generated: {len(saved_files)}")
    for name, fpath in saved_files.items():
        print(f"  - {name}: {fpath}")
    print("=" * 50)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
