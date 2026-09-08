"""Unit tests for the water features generator and simulator."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from pde_slam.interpolators import (
    SpatialGrid,
    create_gaussian_plume,
    create_random_plumes,
)


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
