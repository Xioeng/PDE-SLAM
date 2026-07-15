"""Shared test fixtures for pde_slam test suite."""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import pytest

from pde_slam.interpolator import SpatialGrid


@pytest.fixture(scope="session")
def small_grid() -> SpatialGrid:
    """A small 16×16 grid for fast unit tests."""
    return SpatialGrid(x_min=-50.0, x_max=50.0, y_min=-50.0, y_max=50.0, nx=16, ny=16)


@pytest.fixture(scope="session")
def medium_grid() -> SpatialGrid:
    """A medium 64×64 grid for integration tests."""
    return SpatialGrid(x_min=-500.0, x_max=500.0, y_min=-500.0, y_max=500.0, nx=64, ny=64)
