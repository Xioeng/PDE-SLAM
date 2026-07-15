"""Tests for pde_slam.solver (AdvectionDiffusionSolver class API)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pde_slam.interpolator import SpatialGrid
from pde_slam.solver import (
    AdvectionDiffusionSolver,
    PDEParams,
    _advection_upwind,
    _laplacian_cd2,
)

NY, NX = 32, 32
GRID = SpatialGrid(x_min=0.0, x_max=31.0, y_min=0.0, y_max=31.0, nx=NX, ny=NY)
DX, DY = GRID.dx, GRID.dy


@pytest.fixture()
def solver() -> AdvectionDiffusionSolver:
    return AdvectionDiffusionSolver(GRID, dt_max=0.1)


@pytest.fixture()
def flat_params() -> PDEParams:
    """Zero velocity, small diffusivity – pure diffusion regime."""
    return PDEParams(u_field=jnp.zeros((NY, NX, 2)), D=jnp.array(0.1))


@pytest.fixture()
def gaussian_field() -> jnp.ndarray:
    """2-D Gaussian blob centred on the grid."""
    xs = jnp.linspace(-16, 16, NX)
    ys = jnp.linspace(-16, 16, NY)
    XX, YY = jnp.meshgrid(xs, ys)
    return jnp.exp(-(XX**2 + YY**2) / 8.0)


# ---------------------------------------------------------------------------
# Stencil helpers
# ---------------------------------------------------------------------------


class TestLaplacian:
    def test_uniform_field_has_zero_laplacian(self) -> None:
        phi = jnp.ones((NY, NX))
        lap = _laplacian_cd2(phi, DX, DY)
        np.testing.assert_allclose(np.array(lap), 0.0, atol=1e-6)

    def test_laplacian_shape(self, gaussian_field: jnp.ndarray) -> None:
        lap = _laplacian_cd2(gaussian_field, DX, DY)
        assert lap.shape == (NY, NX)


class TestAdvection:
    def test_zero_velocity_no_advection(self, gaussian_field: jnp.ndarray) -> None:
        u_field = jnp.zeros((NY, NX, 2))
        adv = _advection_upwind(gaussian_field, u_field, DX, DY)
        np.testing.assert_allclose(np.array(adv), 0.0, atol=1e-6)

    def test_advection_shape(self, gaussian_field: jnp.ndarray) -> None:
        u_field = jnp.ones((NY, NX, 2)) * 0.1
        adv = _advection_upwind(gaussian_field, u_field, DX, DY)
        assert adv.shape == (NY, NX)


# ---------------------------------------------------------------------------
# Solver class
# ---------------------------------------------------------------------------


class TestSolverSolve:
    def test_output_shape(
        self,
        solver: AdvectionDiffusionSolver,
        flat_params: PDEParams,
        gaussian_field: jnp.ndarray,
    ) -> None:
        phi_end = solver.solve(gaussian_field, flat_params, t0=0.0, t_end=0.5)
        assert phi_end.shape == (NY, NX)

    def test_diffusion_reduces_peak(
        self,
        solver: AdvectionDiffusionSolver,
        flat_params: PDEParams,
        gaussian_field: jnp.ndarray,
    ) -> None:
        """Pure diffusion should reduce the peak of a Gaussian blob."""
        phi_end = solver.solve(gaussian_field, flat_params, t0=0.0, t_end=1.0)
        assert float(jnp.max(phi_end)) < float(jnp.max(gaussian_field))


class TestStabilityMetrics:
    def test_courant_zero_velocity(self, solver: AdvectionDiffusionSolver) -> None:
        u_field = jnp.zeros((NY, NX, 2))
        C = solver.courant_number(u_field, dt=0.5)
        assert float(C) == pytest.approx(0.0, abs=1e-7)

    def test_diffusion_number_positive(self, solver: AdvectionDiffusionSolver) -> None:
        D = jnp.array(0.1)
        d = solver.diffusion_number(D, dt=0.5)
        assert float(d) > 0.0


class TestDifferentiability:
    def test_solve_is_differentiable(
        self,
        solver: AdvectionDiffusionSolver,
        flat_params: PDEParams,
        gaussian_field: jnp.ndarray,
    ) -> None:
        """JAX must be able to differentiate through solver.solve()."""

        def loss(p: PDEParams) -> jnp.ndarray:
            phi = solver.solve(gaussian_field, p, t0=0.0, t_end=0.5)
            return jnp.mean(phi**2)

        grads = jax.grad(loss)(flat_params)
        assert jnp.isfinite(grads.D)
