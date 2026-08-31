from __future__ import annotations

import jax.numpy as jnp
from jax import Array


class SpatialGrid:
    """Rectangular ENU-frame computational grid (all dimensions in metres).

    Parameters
    ----------
    x_min, x_max : float
        East extent [m].
    y_min, y_max : float
        North extent [m].
    nx : int
        Number of grid points in the East direction.
    ny : int
        Number of grid points in the North direction.
    """

    def __init__(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        nx: int,
        ny: int,
    ) -> None:
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.nx = nx
        self.ny = ny

        self.dx: float = (x_max - x_min) / (nx - 1)
        self.dy: float = (y_max - y_min) / (ny - 1)

        xs = jnp.linspace(x_min, x_max, nx)
        ys = jnp.linspace(y_min, y_max, ny)
        self.XX, self.YY = jnp.meshgrid(xs, ys, indexing="ij")  # shape (nx, ny)

    @property
    def query_points(self) -> Array:
        """Flat ``(N_grid, 2)`` array of grid query points."""
        return jnp.column_stack([self.XX.ravel(), self.YY.ravel()])

    @property
    def shape(self) -> tuple[int, int]:
        """``(nx, ny)`` shape tuple."""
        return (self.nx, self.ny)

    def __repr__(self) -> str:
        return (
            f"SpatialGrid(x=[{self.x_min}, {self.x_max}], "
            f"y=[{self.y_min}, {self.y_max}], "
            f"nx={self.nx}, ny={self.ny}, "
            f"dx={self.dx:.3f} m, dy={self.dy:.3f} m)"
        )
