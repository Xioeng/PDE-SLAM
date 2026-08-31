"""
style.py
========
Publication-quality typography, color palettes, and colormap configurations.
"""

from __future__ import annotations

from typing import Any

# Matplotlib global RC parameters for publication-ready LaTeX typography
PLOT_RC_PARAMS: dict[str, Any] = {
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "cm",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
    "lines.linewidth": 1.0,
    "lines.markersize": 3.0,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
}

# High-contrast trajectory and geometry color palette
TRAJECTORY_COLORS: dict[str, dict[str, Any]] = {
    "ground_truth": {
        "color": "#000000",
        "linestyle": "-",
        "linewidth": 1.4,
        "label": "Ground Truth Robot Path",
    },
    "dead_reckoning": {
        "color": "#00B0FF",
        "linestyle": "--",
        "linewidth": 1.1,
        "label": "Dead Reckoning Path",
    },
    "oracle_rbpf": {
        "color": "#8E24AA",
        "linestyle": ":",
        "linewidth": 1.2,
        "label": "Oracle RBPF Path",
    },
    "online_rbpf": {
        "color": "#D50000",
        "linestyle": "-",
        "linewidth": 1.3,
        "label": "Online RBPF-SLAM Path (Proposed)",
    },
    "polygon": {
        "color": "#D50000",
        "linestyle": "--",
        "linewidth": 1.1,
        "alpha": 0.85,
        "label": "PDE Domain Boundary",
    },
    "ic_anchors": {
        "marker": "D",
        "color": "#FFD600",
        "markersize": 4.5,
        "markeredgecolor": "#3E2723",
        "markeredgewidth": 0.5,
        "label": "Initial Condition Anchors (t=0)",
    },
}

# Dedicated colormaps for each aquatic physical feature
CMAP_PER_FEATURE: dict[str, str] = {
    "salinity": "viridis",
    "temperature": "coolwarm",
    "chlorophyll": "YlGn",
    "odo": "cividis",
    "residuals": "inferno",
}

CMAP_RESIDUALS: str = "inferno"

# Standard figure aspect ratios and layout spacing (in inches)
FIGSIZE_PATHS: tuple[float, float] = (6.0, 5.0)
FIGSIZE_RMSE: tuple[float, float] = (6.0, 3.8)
FIGSIZE_FIELD_PANEL: tuple[float, float] = (3.5, 3.2)
FIGSIZE_RESIDUAL_PANEL: tuple[float, float] = (3.5, 3.2)
FIGSIZE_TRAJECTORY_COMBINED: tuple[float, float] = (12.0, 4.8)
FIGSIZE_GRID_WIDTH: float = 14.0
FIGSIZE_GRID_PER_ROW: float = 2.4
FIGSIZE_RESIDUALS_GRID: tuple[float, float] = (14.0, 2.8)

FIG1_WSPACE: float = 0.22
GRID_HSPACE: float = 0.12
GRID_WSPACE: float = 0.08

SAT_ALPHA: float = 0.70
FIELD_ALPHA: float = 0.72
RESIDUAL_ALPHA: float = 0.85


def get_feature_cmap(field_name: str) -> str:
    """Return the assigned colormap for a physical water feature.

    Parameters
    ----------
    field_name : str
        Name of the physical variable (e.g. salinity, temperature, chlorophyll, odo).

    Returns
    -------
    str
        Matplotlib colormap identifier.
    """
    clean = field_name.lower().replace(" ", "").replace("_", "")
    for key, cmap in CMAP_PER_FEATURE.items():
        if key in clean:
            return cmap
    return "viridis"
