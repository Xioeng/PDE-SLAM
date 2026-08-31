"""
water_features.py
Utilities for generating spatial initial conditions (Gaussian plumes,
multiple/random plumes) and constructing virtual sensor interpolators.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from pde_slam.interpolators.grid import SpatialGrid


def create_gaussian_plume(
    grid: SpatialGrid,
    center: tuple[float, float] | Array = (0.0, 0.0),
    width: float | Array = 10.0,
    amplitude: float | Array = 1.0,
) -> Array:
    """Evaluate a single 2-D Gaussian plume on the spatial grid.

    Parameters
    ----------
    grid : SpatialGrid
        The spatial grid on which to evaluate the plume.
    center : tuple of float or Array
        The (east, north) center coordinates of the plume [m].
    width : float or Array
        The standard deviation/width of the plume [m].
    amplitude : float or Array
        The peak amplitude/value of the plume.

    Returns
    -------
    field : Array
        2-D array of shape ``(ny, nx)`` containing the Gaussian plume.
    """
    x0, y0 = center
    XX = jnp.asarray(grid.XX)  # noqa: N806
    YY = jnp.asarray(grid.YY)  # noqa: N806

    return amplitude * jnp.exp(-((XX - x0) ** 2 + (YY - y0) ** 2) / (2.0 * width**2))


def create_random_plumes(
    grid: SpatialGrid,
    num_plumes: int = 3,
    seed: int | None = None,
) -> Array:
    """Generate a 2-D field containing multiple randomized Gaussian plumes.

    Parameters
    ----------
    grid : SpatialGrid
        The spatial grid on which to evaluate the plumes.
    num_plumes : int, optional
        The number of randomized plumes to generate (default is 3).
    seed : int, optional
        Random seed for reproducible plume parameters.

    Returns
    -------
    field : Array
        2-D array of shape ``(ny, nx)`` containing the sum of randomized plumes.
    """
    key = jax.random.PRNGKey(seed if seed is not None else 42)
    key_x, key_y, key_w, key_a = jax.random.split(key, 4)

    # Place centers within the inner 70% of the grid to prevent boundary clipping
    x_min, x_max = grid.x_min, grid.x_max
    y_min, y_max = grid.y_min, grid.y_max

    centers_x = jax.random.uniform(
        key_x, (num_plumes,), minval=x_min * 0.7, maxval=x_max * 0.7
    )
    centers_y = jax.random.uniform(
        key_y, (num_plumes,), minval=y_min * 0.7, maxval=y_max * 0.7
    )

    # Randomized width between 5% and 15% of the average grid span
    grid_span = 0.5 * ((x_max - x_min) + (y_max - y_min))
    widths = jax.random.uniform(
        key_w, (num_plumes,), minval=grid_span * 0.05, maxval=grid_span * 0.15
    )

    # Randomized amplitude between 0.5 and 1.5
    amplitudes = jax.random.uniform(key_a, (num_plumes,), minval=0.5, maxval=1.5)

    XX = jnp.asarray(grid.XX)  # noqa: N806
    YY = jnp.asarray(grid.YY)  # noqa: N806

    def single_plume(cx: Array, cy: Array, w: Array, amp: Array) -> Array:
        return amp * jnp.exp(-((XX - cx) ** 2 + (YY - cy) ** 2) / (2.0 * w**2))

    plumes_field = jax.vmap(single_plume)(centers_x, centers_y, widths, amplitudes)
    return jnp.sum(plumes_field, axis=0)
