"""
test_scripts/demo_solver.py
============================
Demonstrates how AdvectionDiffusionSolver evolves a scalar field
(salinity plume) under an advection-diffusion PDE in a 500 × 500 m domain.

The initial condition is a salinity plume (high-salinity patch near the
centre, mimicking a well-mixed tidal intrusion).  The solver advances
this field under a river-like outflow velocity field and mild diffusion,
visualising snapshots at t = 0, 60, 120, 180 s.

Run::

    python test_scripts/demo_solver.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path so pde_slam is importable without package installation
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from pde_slam.interpolators import SpatialGrid
from pde_slam.solver import AdvectionDiffusionSolver, PDEParams

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DOMAIN = 250.0  # half-width [m]
NX = NY = 80
DT_MAX = 2.0  # solver max step [s]

T_SNAPSHOTS = [0.0, 60.0, 120.0, 500.0]  # times to capture [s]

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

grid = SpatialGrid(
    x_min=-DOMAIN,
    x_max=DOMAIN,
    y_min=-DOMAIN,
    y_max=DOMAIN,
    nx=NX,
    ny=NY,
)

# ---------------------------------------------------------------------------
# Initial condition: Gaussian salinity plume
# ---------------------------------------------------------------------------

AMBIENT_SAL = 34.5  # PSU
PLUME_SAL = 20.0  # PSU (fresh water intrusion)
PLUME_CX = -80.0  # plume centre east [m]
PLUME_CY = -60.0  # plume centre north [m]
PLUME_SIG = 55.0  # Gaussian half-width [m]

r2 = (grid.XX - PLUME_CX) ** 2 + (grid.YY - PLUME_CY) ** 2
phi0 = jnp.array(
    AMBIENT_SAL - (AMBIENT_SAL - PLUME_SAL) * np.exp(-r2 / (2 * PLUME_SIG**2)),
    dtype=jnp.float32,
)

# ---------------------------------------------------------------------------
# PDE parameters: tidal outflow + weak eddy
# ---------------------------------------------------------------------------

# Mean outflow: south-west to north-east
u_mean = np.array([0.15, 0.10])  # m/s

# Weak anti-clockwise eddy centred at (80, 40) m
eddy_cx, eddy_cy = -80.0, -60.0
eddy_r = 60.0  # eddy radius [m]
eddy_mag = 0.5  # peak eddy speed [m/s]

dx_eddy = grid.XX - eddy_cx
dy_eddy = grid.YY - eddy_cy
r_eddy = np.sqrt(dx_eddy**2 + dy_eddy**2) + 1e-6
eddy_u = -eddy_mag * (dy_eddy / r_eddy) * np.exp(-(r_eddy**2) / (2 * eddy_r**2))
eddy_v = eddy_mag * (dx_eddy / r_eddy) * np.exp(-(r_eddy**2) / (2 * eddy_r**2))

u_field = jnp.array(
    np.stack([u_mean[0] + eddy_u, u_mean[1] + eddy_v], axis=-1),
    dtype=jnp.float32,
)
print(jnp.sqrt(jnp.square(u_field[..., 0]) + jnp.square(u_field[..., 1])).mean())
D = jnp.array(0.8, dtype=jnp.float32)  # m²/s diffusivity

params = PDEParams(u_field=u_field, D=D)

# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

solver = AdvectionDiffusionSolver(grid, dt_max=DT_MAX)

# Courant-number sanity check
C = solver.courant_number(u_field, DT_MAX)
d = solver.diffusion_number(D, DT_MAX)
print(f"Courant number C = {float(C):.3f}  (target ≤ 1)")
print(f"Diffusion number d = {float(d):.4f}  (target ≤ 0.5)")

# Collect snapshots
snapshots = solver.solve(phi0, params, t0=T_SNAPSHOTS[0], t_end=T_SNAPSHOTS[-1], saveat=T_SNAPSHOTS)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

VMIN = float(phi0.min()) - 0.2
VMAX = float(phi0.max()) + 0.2
CMAP = "RdYlBu_r"
EXT = [-DOMAIN, DOMAIN, -DOMAIN, DOMAIN]

fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), constrained_layout=True)
fig.suptitle(
    "AdvectionDiffusionSolver — Salinity plume evolution [PSU]",
    fontsize=13,
    fontweight="bold",
)

for ax, snap, t in zip(axes, snapshots, T_SNAPSHOTS):
    im = ax.imshow(snap, origin="lower", extent=EXT, vmin=VMIN, vmax=VMAX, cmap=CMAP)
    ax.set_title(f"t = {int(t)} s")
    ax.set_xlabel("East [m]")
    if ax is axes[0]:
        ax.set_ylabel("North [m]")
    plt.colorbar(im, ax=ax, shrink=0.85)

# Overlay velocity arrows on the last panel (subsample for clarity)
step = 8
ax = axes[-1]
ax.quiver(
    grid.XX[::step, ::step],
    grid.YY[::step, ::step],
    np.array(u_field[::step, ::step, 0]),
    np.array(u_field[::step, ::step, 1]),
    color="k",
    scale=4.5,
    alpha=0.55,
    width=0.003,
)

out_path = OUTPUT_DIR / "demo_solver.png"
fig.savefig(out_path, dpi=150)
print(f"Saved → {out_path}")
plt.show()
