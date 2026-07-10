"""
interpolator.py
===============
Scattered sensor data → structured grid interpolation.

After a survey pass, the vehicle holds a cloud of scattered (x, y, value)
triples for each measured scalar field.  The :class:`FieldInterpolator`
converts this unstructured cloud into a dense array on a fixed
``(ny, nx)`` computational grid, ready to be used as an initial condition
for the PDE solver.

Two backends are supported:

``"rbf"``     – Thin-plate-spline Radial Basis Function interpolation via
                :class:`scipy.interpolate.RBFInterpolator`.  Accurate for
                smooth fields; O(N³) build cost.

``"spline"``  – Bivariate cubic B-spline via
                :class:`scipy.interpolate.SmoothBivariateSpline`.
                Faster build; requires ≥ 16 observations.

Usage
-----
::

    grid = SpatialGrid(0, 500, 0, 500, nx=64, ny=64)
    interp = FieldInterpolator(grid, method="rbf")
    phi0 = interp.fit_predict(xy_obs, values)
"""

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp
import numpy as np
from jax import Array
from scipy.interpolate import RBFInterpolator, griddata

# ---------------------------------------------------------------------------
# Grid specification
# ---------------------------------------------------------------------------


class SpatialGrid:
    """Rectangular ENU-frame computational grid (all dimensions in metres).

    Parameters
    ----------
    x_min, x_max :
        East extent [m].
    y_min, y_max :
        North extent [m].
    nx :
        Number of grid points in the East direction.
    ny :
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

        xs = np.linspace(x_min, x_max, nx)
        ys = np.linspace(y_min, y_max, ny)
        self.XX, self.YY = np.meshgrid(xs, ys)  # shape (ny, nx)

    @property
    def query_points(self) -> np.ndarray:
        """Flat ``(N_grid, 2)`` array of grid query points."""
        return np.column_stack([self.XX.ravel(), self.YY.ravel()])

    @property
    def shape(self) -> tuple[int, int]:
        """``(ny, nx)`` shape tuple."""
        return (self.ny, self.nx)

    def __repr__(self) -> str:
        return (
            f"SpatialGrid(x=[{self.x_min}, {self.x_max}], "
            f"y=[{self.y_min}, {self.y_max}], "
            f"nx={self.nx}, ny={self.ny}, "
            f"dx={self.dx:.3f} m, dy={self.dy:.3f} m)"
        )


# ---------------------------------------------------------------------------
# Interpolator class
# ---------------------------------------------------------------------------


class FieldInterpolator:
    """Interpolates scattered scalar observations onto a regular grid.

    Parameters
    ----------
    grid :
        Target :class:`SpatialGrid`.
    method :
        ``"rbf"`` (default) or ``"spline"``.

        * ``"rbf"`` – Thin-plate-spline Radial Basis Function via
          :class:`scipy.interpolate.RBFInterpolator`.  Smooth and accurate;
          controlled by *rbf_kernel* and *rbf_smoothing*.
        * ``"spline"`` – Piecewise cubic interpolation on the Delaunay
          triangulation of the observations via
          :func:`scipy.interpolate.griddata` (``method="cubic"``).
          Extrapolation outside the convex hull falls back to
          nearest-neighbour.  Requires ≥ 4 observations.
    rbf_kernel :
        Kernel for the RBF backend (e.g. ``"thin_plate_spline"``,
        ``"multiquadric"``, ``"gaussian"``).
    rbf_smoothing :
        RBF smoothing factor (0 = exact interpolation).
    """

    def __init__(
        self,
        grid: SpatialGrid,
        method: Literal["rbf", "spline"] = "rbf",
        *,
        rbf_kernel: str = "thin_plate_spline",
        rbf_smoothing: float = 0.0,
        fill_value: float = 0.0,
    ) -> None:
        if method not in ("rbf", "spline"):
            raise ValueError(f"method must be 'rbf' or 'spline'; got {method!r}")
        self.grid = grid
        self.method = method
        self._rbf_kernel = rbf_kernel
        self._rbf_smoothing = rbf_smoothing
        self._fill_value = fill_value

        self._xy_obs: np.ndarray | None = None
        self._values: np.ndarray | None = None

    def fit(self, xy_obs: np.ndarray, values: np.ndarray) -> "FieldInterpolator":
        """Store observations.  Returns ``self`` for chaining.

        Parameters
        ----------
        xy_obs :
            Observed positions of shape ``(N, 2)`` – ``[east_m, north_m]``.
        values :
            Observed scalar values of shape ``(N,)``.
        """
        self._validate(xy_obs, values)
        self._xy_obs = np.asarray(xy_obs, dtype=np.float64)
        self._values = np.asarray(values, dtype=np.float64)
        return self

    def predict(self) -> Array:
        """Interpolate stored observations onto the grid.

        Returns
        -------
        field :
            JAX array of shape ``(ny, nx)``.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called first.
        """
        if self._xy_obs is None or self._values is None:
            raise RuntimeError("Call fit() before predict().")
        return self._interpolate(self._xy_obs, self._values)

    def fit_predict(self, xy_obs: np.ndarray, values: np.ndarray) -> Array:
        """Convenience method combining :meth:`fit` and :meth:`predict`.

        Parameters
        ----------
        xy_obs :
            Observed positions of shape ``(N, 2)``.
        values :
            Observed scalar values of shape ``(N,)``.

        Returns
        -------
        field :
            JAX array of shape ``(ny, nx)``.
        """
        return self.fit(xy_obs, values).predict()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(xy_obs: np.ndarray, values: np.ndarray) -> None:
        if np.asarray(xy_obs).ndim != 2 or np.asarray(xy_obs).shape[1] != 2:
            raise ValueError(f"xy_obs must have shape (N, 2); got {xy_obs.shape}")
        if np.asarray(values).ndim != 1 or len(values) != len(xy_obs):
            raise ValueError("values must be 1-D with the same length as xy_obs rows.")

    def _interpolate(self, xy_obs: np.ndarray, values: np.ndarray) -> Array:
        if self.method == "rbf":
            field_flat = self._rbf(xy_obs, values)
        else:
            if len(values) < 4:
                raise ValueError(
                    f"Spline backend requires ≥ 4 observations; got {len(values)}."
                    "  Use method='rbf' instead."
                )
            field_flat = self._spline(xy_obs, values)
        return jnp.array(field_flat.reshape(self.grid.shape), dtype=jnp.float32)

    def _rbf(self, xy_obs: np.ndarray, values: np.ndarray) -> np.ndarray:
        rbf = RBFInterpolator(
            xy_obs,
            values,
            kernel=self._rbf_kernel,
            smoothing=self._rbf_smoothing,
        )
        return rbf(self.grid.query_points)

    def _spline(self, xy_obs: np.ndarray, values: np.ndarray) -> np.ndarray:
        # Use scipy griddata with cubic method — robust for arbitrary scattered
        # data with no domain/knot constraints.  Regions outside the convex
        # hull of the observations fall back to nearest-neighbour.
        z_cubic = griddata(xy_obs, values, self.grid.query_points, method="cubic")
        outside = np.isnan(z_cubic)
        if outside.any():
            z_nn = griddata(xy_obs, values, self.grid.query_points, method="nearest")
            z_cubic[outside] = z_nn[outside]
        return z_cubic

