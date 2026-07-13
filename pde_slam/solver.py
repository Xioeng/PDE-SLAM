"""
solver.py
=========
Differentiable 2-D finite-difference advection-diffusion PDE solver.

The solved PDE is the scalar advection-diffusion equation:

.. math::

    \\frac{\\partial \\phi}{\\partial t} + \\mathbf{u} \\cdot \\nabla \\phi
    = D \\, \\nabla^2 \\phi

where

* ``φ(x, y, t)``  – passive scalar field (e.g. salinity, temperature).
* ``u(x, y)``     – 2-D advection velocity field, shape ``(ny, nx, 2)`` [m s⁻¹].
* ``D``           – scalar isotropic diffusivity [m² s⁻¹].

Spatial discretisation
-----------------------
* **Diffusion** – 2nd-order central differences on a uniform Cartesian stencil.
* **Advection** – 1st-order upwind scheme (donor-cell).

Usage
-----
::

    from pde_slam.interpolator import SpatialGrid
    from pde_slam.solver import AdvectionDiffusionSolver, PDEParams

    grid  = SpatialGrid(0, 500, 0, 500, nx=64, ny=64)
    params = PDEParams(u_field=..., D=...)
    solver = AdvectionDiffusionSolver(grid)
    phi_end = solver.solve(phi0, params, t0=0.0, t_end=60.0)
"""

from __future__ import annotations

from typing import NamedTuple

import diffrax
import jax
import jax.numpy as jnp
from jax import Array

from pde_slam.interpolator import SpatialGrid

# ---------------------------------------------------------------------------
# PDE parameter container
# ---------------------------------------------------------------------------


class PDEParams(NamedTuple):
    """Learnable PDE parameters.

    Attributes
    ----------
    u_field :
        Advection velocity field of shape ``(ny, nx, 2)`` – ``[u_east, u_north]``
        in m s⁻¹.
    D :
        Isotropic diffusivity [m² s⁻¹].
    """

    u_field: Array  # shape (ny, nx, 2)
    D: Array  # scalar


# ---------------------------------------------------------------------------
# Private stencil helpers
# ---------------------------------------------------------------------------


def _laplacian_cd2(phi: Array, dx: float, dy: float) -> Array:
    """5-point central-difference Laplacian ∇²φ with zero-Neumann BCs."""
    phi_p = jnp.pad(phi, pad_width=1, mode="edge")
    d2_dx2 = (phi_p[1:-1, 2:] - 2.0 * phi_p[1:-1, 1:-1] + phi_p[1:-1, :-2]) / dx**2
    d2_dy2 = (phi_p[2:, 1:-1] - 2.0 * phi_p[1:-1, 1:-1] + phi_p[:-2, 1:-1]) / dy**2
    return d2_dx2 + d2_dy2


def _advection_central_diff(phi: Array, u_field: Array, dx: float, dy: float) -> Array:
    """2nd-order central-difference advection flux divergence u · ∇φ."""
    ux = u_field[..., 0]
    uy = u_field[..., 1]
    phi_p = jnp.pad(phi, pad_width=1, mode="edge")

    dphi_dx_fwd = (phi_p[1:-1, 2:] - phi_p[1:-1, 1:-1]) / dx
    dphi_dx_bwd = (phi_p[1:-1, 1:-1] - phi_p[1:-1, :-2]) / dx
    dphi_dx = (dphi_dx_fwd + dphi_dx_bwd) / 2.0

    dphi_dy_fwd = (phi_p[2:, 1:-1] - phi_p[1:-1, 1:-1]) / dy
    dphi_dy_bwd = (phi_p[1:-1, 1:-1] - phi_p[:-2, 1:-1]) / dy
    dphi_dy = (dphi_dy_fwd + dphi_dy_bwd) / 2.0

    return ux * dphi_dx + uy * dphi_dy


def _advection_upwind(phi: Array, u_field: Array, dx: float, dy: float) -> Array:
    """1st-order upwind advection flux divergence u · ∇φ."""
    ux = u_field[..., 0]
    uy = u_field[..., 1]
    phi_p = jnp.pad(phi, pad_width=1, mode="edge")

    dphi_dx_fwd = (phi_p[1:-1, 2:] - phi_p[1:-1, 1:-1]) / dx
    dphi_dx_bwd = (phi_p[1:-1, 1:-1] - phi_p[1:-1, :-2]) / dx
    dphi_dx = jnp.where(ux >= 0.0, dphi_dx_bwd, dphi_dx_fwd)

    dphi_dy_fwd = (phi_p[2:, 1:-1] - phi_p[1:-1, 1:-1]) / dy
    dphi_dy_bwd = (phi_p[1:-1, 1:-1] - phi_p[:-2, 1:-1]) / dy
    dphi_dy = jnp.where(uy >= 0.0, dphi_dy_bwd, dphi_dy_fwd)

    return ux * dphi_dx + uy * dphi_dy


def _pde_rhs(
    t: Array,
    phi: Array,
    pde_params: PDEParams,
    dx: float,
    dy: float,
) -> Array:
    """RHS of the advection-diffusion PDE: dφ/dt = D∇²φ − u·∇φ."""
    diffusion = pde_params.D * _laplacian_cd2(phi, dx, dy)
    advection = _advection_central_diff(phi, pde_params.u_field, dx, dy)
    return diffusion - advection


# ---------------------------------------------------------------------------
# Solver class
# ---------------------------------------------------------------------------


class AdvectionDiffusionSolver:
    """Differentiable forward solver for the 2-D advection-diffusion PDE.

    Parameters
    ----------
    grid :
        :class:`~pde_slam.interpolator.SpatialGrid` that defines the spatial
        domain and grid spacings ``dx``, ``dy``.
    dt_max :
        Maximum integration step size [s].
    diffrax_solver :
        A ``diffrax`` solver instance.  Defaults to :class:`diffrax.Heun`
        (2nd-order explicit, stable for parabolic PDEs).
    adjoint :
        Adjoint method for gradient computation.  Defaults to
        :class:`diffrax.RecursiveCheckpointAdjoint`.
    """

    def __init__(
        self,
        grid: SpatialGrid,
        *,
        dt_max: float = 1.0,
        diffrax_solver: diffrax.AbstractSolver | None = None,
        adjoint: diffrax.AbstractAdjoint | None = None,
    ) -> None:
        self.grid = grid
        self.dt_max = dt_max
        self._solver = diffrax_solver if diffrax_solver is not None else diffrax.Heun()
        self._adjoint = (
            adjoint if adjoint is not None else diffrax.RecursiveCheckpointAdjoint()
        )

    def solve(
        self,
        phi0: Array,
        pde_params: PDEParams,
        t0: float,
        t_end: float,
        saveat: list[float] | None = None,
    ) -> Array:
        """Integrate the PDE forward from ``t0`` to ``t_end``.

        The result is fully differentiable with respect to *pde_params* via
        JAX automatic differentiation.

        Parameters
        ----------
        phi0 :
            Initial condition of shape ``(ny, nx)``.
        pde_params :
            :class:`PDEParams` holding the advection field and diffusivity.
        t0 :
            Start time [s].
        t_end :
            End time [s].

        Returns
        -------
        phi_end :
            PDE solution at *t_end*, shape ``(ny, nx)``.
        """
        dx, dy = self.grid.dx, self.grid.dy
        term = diffrax.ODETerm(lambda t, y, _args: _pde_rhs(t, y, pde_params, dx, dy))
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
            saveat=diffrax.SaveAt(ts=saveat)
            if saveat is not None
            else diffrax.SaveAt(t1=True),
        )
        if saveat is None:
            return sol.ys[-1]
        else:
            return sol.ys

    def courant_number(self, u_field: Array, dt: float) -> Array:
        """Maximum advective Courant number over the grid (should stay ≤ 1)."""
        cx = jnp.max(jnp.abs(u_field[..., 0])) * dt / self.grid.dx
        cy = jnp.max(jnp.abs(u_field[..., 1])) * dt / self.grid.dy
        return jnp.maximum(cx, cy)

    def diffusion_number(self, D: Array, dt: float) -> Array:
        """Diffusion number (should stay ≤ 0.5 for explicit schemes)."""
        return D * dt * (1.0 / self.grid.dx**2 + 1.0 / self.grid.dy**2)
