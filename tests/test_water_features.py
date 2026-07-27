"""Unit tests for the water features generator and simulator."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from pde_slam.interpolators import (
    SpatialGrid,
    SpatiotemporalInterpolator,
    create_gaussian_plume,
    create_random_plumes,
    simulate_virtual_sensor,
)
from pde_slam.solver import AdvectionDiffusionSolver, PDEParams


def test_create_gaussian_plume(small_grid: SpatialGrid) -> None:
    """Test Gaussian plume creation, shapes, values, and decay characteristics."""
    center = (10.0, -10.0)
    width = 15.0
    amp = 2.0
    field = create_gaussian_plume(small_grid, center=center, width=width, amplitude=amp)

    assert isinstance(field, Array)
    assert field.shape == (small_grid.ny, small_grid.nx)


    # Maximum should be close to center (allow minor shift due to grid discretisation)
    max_val = float(jnp.max(field))
    assert max_val <= amp
    assert max_val >= 0.9 * amp

    # Corner values should be significantly lower (decay check)
    assert float(field[0, 0]) < max_val
    assert float(field[-1, -1]) < max_val


def test_create_random_plumes(small_grid: SpatialGrid) -> None:
    """Test random plumes generation, reproducibility, and shapes."""
    field_1 = create_random_plumes(small_grid, num_plumes=3, seed=123)
    field_2 = create_random_plumes(small_grid, num_plumes=3, seed=123)
    field_diff = create_random_plumes(small_grid, num_plumes=3, seed=456)

    assert field_1.shape == (small_grid.ny, small_grid.nx)
    assert field_2.shape == (small_grid.ny, small_grid.nx)
    assert field_diff.shape == (small_grid.ny, small_grid.nx)

    # Determinism check
    assert jnp.allclose(field_1, field_2)

    # Seed variations check
    assert not jnp.allclose(field_1, field_diff)


def test_simulate_virtual_sensor(small_grid: SpatialGrid) -> None:
    """Test PDE-based virtual sensor simulation for single and batched initial fields."""
    solver = AdvectionDiffusionSolver(small_grid, dt_max=0.5)

    u_field = jnp.broadcast_to(jnp.array([0.5, -0.2]), (small_grid.ny, small_grid.nx, 2))
    pde_params = PDEParams(u_field=u_field, D=jnp.array(0.5))

    ts = jnp.array([0.0, 1.0, 2.0])

    # 1. Single Field
    phi0_single = create_gaussian_plume(small_grid, center=(0.0, 0.0), width=10.0)
    sensor = simulate_virtual_sensor(small_grid, solver, phi0_single, pde_params, ts)

    assert isinstance(sensor, SpatiotemporalInterpolator)
    val = sensor(jnp.array([0.0]), jnp.array([0.0]), jnp.array([1.0]))
    assert val.shape == (1,)

    # 2. Batched Fields
    phi0_batched = jnp.stack(
        [
            create_gaussian_plume(small_grid, center=(-10.0, 0.0), width=8.0),
            create_gaussian_plume(small_grid, center=(10.0, 5.0), width=12.0),
        ],
        axis=0,
    )
    sensors = simulate_virtual_sensor(small_grid, solver, phi0_batched, pde_params, ts)

    assert isinstance(sensors, list)
    assert len(sensors) == 2
    for s in sensors:
        assert isinstance(s, SpatiotemporalInterpolator)
        val = s(jnp.array([5.0]), jnp.array([-5.0]), jnp.array([1.5]))
        assert val.shape == (1,)
