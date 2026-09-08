from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.scipy.ndimage import map_coordinates

from pde_slam.interpolators.grid import SpatialGrid


class SpatiotemporalInterpolator:
    """JAX-compatible, differentiable spatiotemporal interpolator for uniform grids.

    This class performs trilinear interpolation (1D in time, 2D in space) over
    dense PDE solver snapshots of shape ``(nt, nx, ny)`` corresponding to
    timestamps ``ts``.

    Parameters
    ----------
    grid : SpatialGrid
        The spatial grid on which the snapshots are defined.
    ts : Array
        Sorted 1D array of shape ``(nt,)`` containing the snapshot timestamps.
    snapshots : Array
        3D array of shape ``(nt, nx, ny)`` containing the solver solution snapshots.
    """

    grid: SpatialGrid
    ts: Array
    snapshots: Array

    def __init__(self, grid: SpatialGrid, ts: Array, snapshots: Array) -> None:

        self.grid = grid
        self.ts = jnp.asarray(ts, dtype=jnp.float32)
        self.snapshots = jnp.asarray(snapshots, dtype=jnp.float32)

        if self.ts.ndim != 1:
            raise ValueError(f"ts must be 1-D; got shape {self.ts.shape}")
        if self.snapshots.ndim != 3:
            raise ValueError(f"snapshots must be 3-D; got shape {self.snapshots.shape}")
        if len(self.ts) != self.snapshots.shape[0]:
            raise ValueError(
                f"Mismatch: ts length is {len(self.ts)} but snapshots axis 0 "
                f"has size {self.snapshots.shape[0]}"
            )
        if self.snapshots.shape[1:] != self.grid.shape:
            raise ValueError(
                f"Mismatch: snapshots spatial shape is {self.snapshots.shape[1:]} "
                f"but grid shape is {self.grid.shape}"
            )

    def __call__(self, x: Array, y: Array, t: Array) -> Array:
        """Query the interpolated field at coordinate(s) (x, y, t).

        Parameters
        ----------
        x : Array
            East coordinate(s) [m].
        y : Array
            North coordinate(s) [m].
        t : Array
            Time coordinate(s) [s].

        Returns
        -------
        Array
            Interpolated value(s) matching the broadcasted shape of x, y, t.
        """
        x = jnp.asarray(x, dtype=jnp.float32)
        y = jnp.asarray(y, dtype=jnp.float32)
        t = jnp.asarray(t, dtype=jnp.float32)

        # Map physical space coordinates to continuous grid indices
        x_idx = (
            (x - self.grid.x_min)
            / (self.grid.x_max - self.grid.x_min)
            * (self.grid.nx - 1)
        )
        y_idx = (
            (y - self.grid.y_min)
            / (self.grid.y_max - self.grid.y_min)
            * (self.grid.ny - 1)
        )

        # Map physical time to continuous index via linear interpolation of indices
        t_xp = self.ts
        t_fp = jnp.arange(len(self.ts), dtype=jnp.float32)
        t_idx = jnp.interp(t, t_xp, t_fp)

        # Clamp to bounds to prevent out-of-bounds/NaN issues
        x_idx = jnp.clip(x_idx, 0.0, self.grid.nx - 1.0)
        y_idx = jnp.clip(y_idx, 0.0, self.grid.ny - 1.0)
        t_idx = jnp.clip(t_idx, 0.0, len(self.ts) - 1.0)

        # Stack coordinates (axis 0 is the dimension of the grid: t, x, y)
        coords = jnp.stack([t_idx, x_idx, y_idx], axis=0)

        # Perform trilinear interpolation
        return map_coordinates(self.snapshots, coords, order=1, mode="nearest")
