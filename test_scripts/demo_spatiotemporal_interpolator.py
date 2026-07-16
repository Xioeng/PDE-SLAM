"""
test_scripts/demo_spatiotemporal_interpolator.py
================================================
Demonstrates the SpatiotemporalInterpolator by querying a virtual sensor
along a trajectory and generating space-time slices of a simulated salinity plume.

Run::

    uv run python test_scripts/demo_spatiotemporal_interpolator.py
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from pde_slam.interpolators import SpatialGrid, SpatiotemporalInterpolator
from pde_slam.solver import AdvectionDiffusionSolver, PDEParams

# ---------------------------------------------------------------------------
# Setup and Simulation
# ---------------------------------------------------------------------------

DOMAIN = 250.0  # half-width [m]
NX = NY = 80
DT_MAX = 2.0  # solver max step [s]

# Generate snapshots every 10 seconds
T_MAX = 300.0
T_SNAPSHOTS = jnp.linspace(0.0, T_MAX, 31)

grid = SpatialGrid(
    x_min=-DOMAIN,
    x_max=DOMAIN,
    y_min=-DOMAIN,
    y_max=DOMAIN,
    nx=NX,
    ny=NY,
)

# Initial condition: Gaussian salinity plume
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

# PDE parameters: outflow
u_mean = np.array([0.15, 0.10])  # m/s
u_field = jnp.array(
    np.stack([np.full_like(grid.XX, u_mean[0]), np.full_like(grid.YY, u_mean[1])], axis=-1),
    dtype=jnp.float32,
)
D = jnp.array(0.8, dtype=jnp.float32)  # m²/s diffusivity
params = PDEParams(u_field=u_field, D=D)

# Run solver
solver = AdvectionDiffusionSolver(grid, dt_max=DT_MAX)
print("Running forward solver to collect snapshots...")
snapshots = solver.solve(phi0, params, t0=0.0, t_end=T_MAX, saveat=list(T_SNAPSHOTS))

# ---------------------------------------------------------------------------
# Spatiotemporal Interpolation
# ---------------------------------------------------------------------------

print("Initializing SpatiotemporalInterpolator...")
interp = SpatiotemporalInterpolator(grid, T_SNAPSHOTS, snapshots)

# 1. Trajectory of a moving virtual sensor
# The robot starts at (-150, -150) and drives to (150, 150) over T_MAX seconds
t_traj = jnp.linspace(0.0, T_MAX, 300)
x_traj = -150.0 + 1.0 * t_traj
y_traj = -150.0 + 1.0 * t_traj

# Query virtual sensor readings
print("Querying virtual sensor along trajectory...")
sal_sensor = interp(x_traj, y_traj, t_traj)

# 2. Slice in space-time along the diagonal (x = y)
diag_coords = jnp.linspace(-DOMAIN, DOMAIN, 100)
diag_t, diag_s = jnp.meshgrid(T_SNAPSHOTS, diag_coords, indexing="ij")
# Query space-time grid
diag_sal = interp(diag_s, diag_s, diag_t)  # shape (len(T_SNAPSHOTS), 100)

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
fig.suptitle(
    "SpatiotemporalInterpolator — Virtual Sensor & Space-Time Slices",
    fontsize=14,
    fontweight="bold",
)

# Panel 1: Trajectory on Initial & Final Plumes
ax1 = axes[0]
im1 = ax1.imshow(
    snapshots[0],
    origin="lower",
    extent=[-DOMAIN, DOMAIN, -DOMAIN, DOMAIN],
    cmap="RdYlBu_r",
    vmin=20,
    vmax=35,
)
ax1.plot(x_traj, y_traj, "k--", label="Sensor Trajectory", linewidth=2.0)
ax1.scatter([x_traj[0]], [y_traj[0]], color="green", marker="o", s=80, label="Start")
ax1.scatter([x_traj[-1]], [y_traj[-1]], color="red", marker="x", s=80, label="End")
ax1.set_title("Trajectory overlaid on t=0 Plume")
ax1.set_xlabel("East [m]")
ax1.set_ylabel("North [m]")
ax1.legend(loc="upper left")
plt.colorbar(im1, ax=ax1, label="Salinity [PSU]")

# Panel 2: Virtual Sensor Measurements vs Time
ax2 = axes[1]
ax2.plot(t_traj, sal_sensor, "b-", linewidth=2.5, label="Interpolated Sensor")
# Sample discrete snapshot values near trajectory for validation
snap_indices = [0, 5, 10, 15, 20, 25, 30]
for idx in snap_indices:
    t_snap = float(T_SNAPSHOTS[idx])
    x_snap = -150.0 + 1.0 * t_snap
    y_snap = -150.0 + 1.0 * t_snap
    val = float(interp(x_snap, y_snap, t_snap))
    ax2.scatter(
        t_snap,
        val,
        color="red",
        zorder=5,
        s=40,
        label="Exact Node" if idx == 0 else "",
    )

ax2.set_title("Virtual Sensor Salinity vs Time")
ax2.set_xlabel("Time [s]")
ax2.set_ylabel("Salinity [PSU]")
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.legend()

# Panel 3: Space-time slice along diagonal x = y
ax3 = axes[2]
im3 = ax3.imshow(
    diag_sal.T,
    origin="lower",
    extent=[0, T_MAX, -DOMAIN, DOMAIN],
    aspect="auto",
    cmap="RdYlBu_r",
    vmin=20,
    vmax=35,
)
ax3.set_title("Space-Time Slice along Diagonal (x=y)")
ax3.set_xlabel("Time [s]")
ax3.set_ylabel("Position along Diagonal [m]")
plt.colorbar(im3, ax=ax3, label="Salinity [PSU]")

out_path = OUTPUT_DIR / "demo_spatiotemporal_interpolator.png"
fig.savefig(out_path, dpi=150)
print(f"Saved visualization → {out_path}")
plt.show()
