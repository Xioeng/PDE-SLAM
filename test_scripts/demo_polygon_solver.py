"""
test_scripts/demo_polygon_solver.py
===================================
Prototype demonstrating how to solve the advection-diffusion PDE inside an
arbitrary polygonal boundary on a regular Cartesian grid.

The domain boundary is defined by a polygon. A binary mask is computed for the
grid, and the finite difference stencils (advection/diffusion) are adapted to
enforce zero-flux (zero-Neumann) boundary conditions on the polygonal boundary
by copying cell values to outside neighbors.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path so pde_slam is importable without package installation
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
import numpy as np

from pde_slam.interpolators import SpatialGrid
from pde_slam.solver import PDEParams
import diffrax

# ---------------------------------------------------------------------------
# Custom Solver for Polygonal Domains
# ---------------------------------------------------------------------------

class PolygonalAdvectionDiffusionSolver:
    """Solver that enforces zero-flux boundaries on an arbitrary binary mask."""

    def __init__(
        self,
        grid: SpatialGrid,
        mask: jnp.ndarray,
        dt_max: float = 1.0,
        diffrax_solver: diffrax.AbstractSolver | None = None,
        adjoint: diffrax.AbstractAdjoint | None = None,
    ) -> None:
        self.grid = grid
        self.mask = mask  # Boolean array of shape (ny, nx)
        self.dt_max = dt_max
        self._solver = diffrax_solver if diffrax_solver is not None else diffrax.Heun()
        self._adjoint = adjoint if adjoint is not None else diffrax.RecursiveCheckpointAdjoint()

    def solve(
        self,
        phi0: jnp.ndarray,
        pde_params: PDEParams,
        t0: float,
        t_end: float,
        saveat: list[float] | jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        dx, dy = self.grid.dx, self.grid.dy
        mask = self.mask

        def _pde_rhs_masked(t, phi, _args):
            phi_padded = jnp.pad(phi, pad_width=1, mode="edge")
            
            # Active faces (both cells must be inside the mask)
            mask_padded_x = jnp.pad(mask, pad_width=((0, 0), (0, 1)), mode="constant", constant_values=False)
            face_x_active = mask & mask_padded_x[:, 1:]

            mask_padded_y = jnp.pad(mask, pad_width=((0, 1), (0, 0)), mode="constant", constant_values=False)
            face_y_active = mask & mask_padded_y[1:, :]

            # Diffusive fluxes (D * dphi/dx)
            flux_x_diff = jnp.where(face_x_active, pde_params.D * (phi_padded[1:-1, 2:] - phi_padded[1:-1, 1:-1]) / dx, 0.0)
            flux_y_diff = jnp.where(face_y_active, pde_params.D * (phi_padded[2:, 1:-1] - phi_padded[1:-1, 1:-1]) / dy, 0.0)

            # Advective velocities at faces
            ux = pde_params.u_field[..., 0]
            uy = pde_params.u_field[..., 1]

            ux_padded = jnp.pad(ux, pad_width=((0, 0), (0, 1)), mode="constant", constant_values=0.0)
            uy_padded = jnp.pad(uy, pad_width=((0, 1), (0, 0)), mode="constant", constant_values=0.0)
            
            ux_face = 0.5 * (ux + ux_padded[:, 1:])
            uy_face = 0.5 * (uy + uy_padded[1:, :])

            # Zero out velocities at boundary faces
            ux_face = jnp.where(face_x_active, ux_face, 0.0)
            uy_face = jnp.where(face_y_active, uy_face, 0.0)

            # Advective fluxes (ux * phi_upwind)
            phi_upwind_x = jnp.where(ux_face >= 0.0, phi_padded[1:-1, 1:-1], phi_padded[1:-1, 2:])
            phi_upwind_y = jnp.where(uy_face >= 0.0, phi_padded[1:-1, 1:-1], phi_padded[2:, 1:-1])

            flux_x_adv = ux_face * phi_upwind_x
            flux_y_adv = uy_face * phi_upwind_y

            # Total fluxes (diffusion - advection)
            flux_x = flux_x_diff - flux_x_adv
            flux_y = flux_y_diff - flux_y_adv

            # Left/down incoming fluxes
            flux_x_left = jnp.pad(flux_x[:, :-1], pad_width=((0, 0), (1, 0)), mode="constant", constant_values=0.0)
            flux_y_down = jnp.pad(flux_y[:-1, :], pad_width=((1, 0), (0, 0)), mode="constant", constant_values=0.0)

            # Divergence of the total flux
            rhs = (flux_x - flux_x_left) / dx + (flux_y - flux_y_down) / dy
            
            return jnp.where(mask, rhs, 0.0)
            # return rhs
        term = diffrax.ODETerm(_pde_rhs_masked)
        sol = diffrax.diffeqsolve(
            terms=term,
            solver=self._solver,
            t0=t0,
            t1=t_end,
            dt0=min(self.dt_max, t_end - t0),
            y0=phi0,
            stepsize_controller=diffrax.ConstantStepSize(),
            adjoint=self._adjoint,
            max_steps=int((t_end - t0) / self.dt_max) + 2,
            saveat=diffrax.SaveAt(ts=saveat) if saveat is not None else diffrax.SaveAt(t1=True),
        )
        if saveat is None:
            return sol.ys[-1]
        else:
            return sol.ys

# ---------------------------------------------------------------------------
# Setup & Execution
# ---------------------------------------------------------------------------

def run_demo():
    # Grid
    DOMAIN = 150.0
    NX = NY = 100
    grid = SpatialGrid(
        x_min=-DOMAIN,
        x_max=DOMAIN,
        y_min=-DOMAIN,
        y_max=DOMAIN,
        nx=NX,
        ny=NY,
    )

    # 1. Define a hexagonal polygon boundary centered at (0, 0)
    theta = np.linspace(0, 2 * np.pi, 7)[:-1]
    radius = 120.0
    poly_x = radius * np.cos(theta)
    poly_y = radius * np.sin(theta)
    polygon_vertices = np.column_stack([poly_x, poly_y])

    # 2. Build the binary mask using matplotlib's Path
    path = MplPath(polygon_vertices)
    mask = jnp.array(
        path.contains_points(grid.query_points).reshape(grid.shape),
        dtype=jnp.bool_
    )

    # 3. Initial condition: Gaussian plume inside the hexagon, background value outside
    AMBIENT_SAL = 34.5
    PLUME_SAL = 20.0
    # Center plume near the bottom-left boundary of the hexagon
    PLUME_CX = -50.0
    PLUME_CY = -30.0
    PLUME_SIG = 25.0

    r2 = (grid.XX - PLUME_CX) ** 2 + (grid.YY - PLUME_CY) ** 2
    phi0_raw = AMBIENT_SAL + (AMBIENT_SAL - PLUME_SAL) * np.exp(-r2 / (2 * PLUME_SIG**2))
    
    # Mask out-of-boundary region to ambient salinity
    phi0 = jnp.where(mask, phi0_raw, AMBIENT_SAL)

    # 4. Velocity field: constant velocity field with boundary flipping to point inwards
    ux = jnp.ones(grid.shape) * 0.7
    uy = jnp.ones(grid.shape) * 0.7
    
    # Flip normal velocity at boundaries to point inward
    mask_padded = jnp.pad(mask, pad_width=1, mode="constant", constant_values=False)
    
    right_outside = ~mask_padded[1:-1, 2:]
    ux = jnp.where(right_outside & (ux > 0.0), -ux, ux)
    
    left_outside = ~mask_padded[1:-1, :-2]
    ux = jnp.where(left_outside & (ux < 0.0), -ux, ux)
    
    up_outside = ~mask_padded[2:, 1:-1]
    uy = jnp.where(up_outside & (uy > 0.0), -uy, uy)
    
    down_outside = ~mask_padded[:-2, 1:-1]
    uy = jnp.where(down_outside & (uy < 0.0), -uy, uy)
    
    u_field = jnp.array(np.stack([ux, uy], axis=-1), dtype=jnp.float32)
    u_field = jnp.where(mask[..., None], u_field, 0.0)

    D = jnp.array(0.0, dtype=jnp.float32)  # m²/s diffusivity
    params = PDEParams(u_field=u_field, D=D)

    # 5. Solve using the Polygonal Solver
    DT_MAX = 1.0
    T_SNAPSHOTS = [0.0, 50.0, 100.0, 200.0, 300.0]
    solver = PolygonalAdvectionDiffusionSolver(grid, mask, dt_max=DT_MAX)
    
    print("Running solver...")
    snapshots = solver.solve(phi0, params, t0=T_SNAPSHOTS[0], t_end=T_SNAPSHOTS[-1], saveat=T_SNAPSHOTS)
    print("Solver finished successfully.")
    for t, snap in zip(T_SNAPSHOTS, snapshots):
        mass = float(jnp.sum((snap - AMBIENT_SAL) * mask))
        max_val = float(jnp.max((snap - AMBIENT_SAL) * mask))
        print(f"t = {t:5.1f} s | Total excess salinity mass: {mass:.6f} | Max excess: {max_val:.6f}")

    # 6. Plotting
    VMIN = float(phi0[mask].min()) - 0.2
    VMAX = float(phi0[mask].max()) + 0.2
    CMAP = "RdYlBu_r"
    EXT = [-DOMAIN, DOMAIN, -DOMAIN, DOMAIN]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), constrained_layout=True)
    fig.suptitle(
        "Polygonal Solver Prototype — Salinity Plume in a Hexagonal Domain",
        fontsize=14,
        fontweight="bold",
    )

    # Add the polygon patch coords to close it for plotting
    plot_poly = np.vstack([polygon_vertices, polygon_vertices[0]])

    for ax, snap, t in zip(axes, snapshots, T_SNAPSHOTS):
        # Mask the output field for clean plotting (show outside as grey)
        snap_plot = np.where(np.array(mask), np.array(snap), np.nan)
        
        # Draw background in grey
        ax.set_facecolor("#e0e0e0")
        
        im = ax.imshow(snap_plot, origin="lower", extent=EXT, vmin=VMIN, vmax=VMAX, cmap=CMAP)
        
        # Plot boundary
        ax.plot(plot_poly[:, 0], plot_poly[:, 1], "r-", linewidth=2.0, label="Boundary")
        
        # Overlay velocity quiver
        step = 6
        ax.quiver(
            grid.XX[::step, ::step],
            grid.YY[::step, ::step],
            np.array(u_field[::step, ::step, 0]),
            np.array(u_field[::step, ::step, 1]),
            color="black",
            scale=15.0,
            alpha=0.4,
            width=0.002,
        )
        
        ax.set_title(f"t = {int(t)} s")
        ax.set_xlabel("East [m]")
        if ax is axes[0]:
            ax.set_ylabel("North [m]")
            ax.legend(loc="upper right")
        plt.colorbar(im, ax=ax, shrink=0.85)

    # Save to outputs
    OUTPUT_DIR = Path("outputs")
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "demo_polygon_solver.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved visualization to {out_path}")

if __name__ == "__main__":
    run_demo()
