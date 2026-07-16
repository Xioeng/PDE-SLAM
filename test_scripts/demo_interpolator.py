"""
test_scripts/demo_interpolator.py
==================================
Demonstrates how FieldInterpolator reconstructs a scalar field from
scattered observations in a 500 × 500 m aquatic survey domain.

The "true" field is a river plume salinity pattern.  Scattered pseudo-
observations are sampled from it with mild sensor noise, then fed into
both the RBF and Spline backends.  The resulting grids are plotted
side-by-side against the true field and saved to outputs/.

Run::

    python test_scripts/demo_interpolator.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pde_slam.interpolators import FieldInterpolator, SpatialGrid

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DOMAIN = 250.0  # half-width [m]; domain is [-250, 250]²
N_OBS = 120  # number of scattered observations
SEED = 42

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# True salinity field (river plume, same model as the survey generator)
# ---------------------------------------------------------------------------

PLUME_SOURCE = np.array([-180.0, -180.0])
AMBIENT_SALINITY = 34.5
FRESH_SALINITY = 18.0
PLUME_WIDTH = 70.0
PLUME_DECAY = 0.004


def _true_salinity(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """River plume salinity [PSU] at arbitrary (x, y) positions."""
    plume_dir = np.array([1.0, 1.0]) / np.sqrt(2.0)
    dx = x - PLUME_SOURCE[0]
    dy = y - PLUME_SOURCE[1]
    along = dx * plume_dir[0] + dy * plume_dir[1]
    cross = -dx * plume_dir[1] + dy * plume_dir[0]
    along = np.maximum(along, 0.0)
    width = PLUME_WIDTH + 0.15 * along
    freshness = np.exp(-0.5 * (cross / width) ** 2) * np.exp(-PLUME_DECAY * along)
    return AMBIENT_SALINITY - (AMBIENT_SALINITY - FRESH_SALINITY) * freshness


# ---------------------------------------------------------------------------
# Build grid and observations
# ---------------------------------------------------------------------------

grid = SpatialGrid(
    x_min=-DOMAIN,
    x_max=DOMAIN,
    y_min=-DOMAIN,
    y_max=DOMAIN,
    nx=80,
    ny=80,
)

rng = np.random.default_rng(SEED)
xy_obs = rng.uniform(-DOMAIN * 0.92, DOMAIN * 0.92, size=(N_OBS, 2))
values_true = _true_salinity(xy_obs[:, 0], xy_obs[:, 1])
values_noisy = values_true + rng.normal(0.0, 0.08, size=N_OBS)

# True reference field on the grid
true_grid = _true_salinity(grid.XX.ravel(), grid.YY.ravel()).reshape(grid.shape)

# ---------------------------------------------------------------------------
# Interpolate – RBF and Spline backends
# ---------------------------------------------------------------------------

rbf_field = FieldInterpolator(grid, method="rbf").fit_predict(xy_obs, values_noisy)
spl_field = FieldInterpolator(grid, method="spline", spline_s=0.5).fit_predict(xy_obs, values_noisy)

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

VMIN, VMAX = FRESH_SALINITY - 1, AMBIENT_SALINITY + 0.5
CMAP = "RdYlBu_r"

fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
fig.suptitle(
    "FieldInterpolator demo — River plume salinity [PSU]",
    fontsize=13,
    fontweight="bold",
)

ext = [-DOMAIN, DOMAIN, -DOMAIN, DOMAIN]

# Panel 1 – true field
ax = axes[0]
im = ax.imshow(true_grid, origin="lower", extent=ext, vmin=VMIN, vmax=VMAX, cmap=CMAP)
ax.scatter(
    xy_obs[:, 0],
    xy_obs[:, 1],
    c=values_noisy,
    cmap=CMAP,
    vmin=VMIN,
    vmax=VMAX,
    s=30,
    edgecolors="k",
    linewidths=0.4,
    label="observations",
)
ax.set_title("True field + observations")
ax.set_xlabel("East [m]")
ax.set_ylabel("North [m]")
ax.legend(loc="upper right", fontsize=8)
plt.colorbar(im, ax=ax, shrink=0.85)

# Panel 2 – RBF reconstruction
ax = axes[1]
im = ax.imshow(np.array(rbf_field), origin="lower", extent=ext, vmin=VMIN, vmax=VMAX, cmap=CMAP)
ax.scatter(xy_obs[:, 0], xy_obs[:, 1], c="white", s=12, edgecolors="k", linewidths=0.4, alpha=0.6)
ax.set_title("RBF interpolation (thin-plate-spline)")
ax.set_xlabel("East [m]")
plt.colorbar(im, ax=ax, shrink=0.85)

# Panel 3 – Spline reconstruction
ax = axes[2]
im = ax.imshow(np.array(spl_field), origin="lower", extent=ext, vmin=VMIN, vmax=VMAX, cmap=CMAP)
ax.scatter(xy_obs[:, 0], xy_obs[:, 1], c="white", s=12, edgecolors="k", linewidths=0.4, alpha=0.6)
ax.set_title("Spline interpolation (bivariate cubic)")
ax.set_xlabel("East [m]")
plt.colorbar(im, ax=ax, shrink=0.85)

out_path = OUTPUT_DIR / "demo_interpolator.png"
fig.savefig(out_path, dpi=150)
print(f"Saved → {out_path}")
plt.show()
