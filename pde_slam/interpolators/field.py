from __future__ import annotations

from typing import Any, Literal

import jax.numpy as jnp
import numpy as np
from jax import Array
from scipy.interpolate import RBFInterpolator, griddata

from pde_slam.interpolators.grid import SpatialGrid


class FieldInterpolator:
    """Interpolates scattered scalar observations onto a regular grid.

    Parameters
    ----------
    grid : SpatialGrid
        Target :class:`SpatialGrid`.
    method : {"rbf", "spline"}, default "rbf"
        * ``"rbf"`` – Thin-plate-spline Radial Basis Function via
          :class:`scipy.interpolate.RBFInterpolator`.  Smooth and accurate;
          controlled by *rbf_kernel* and *rbf_smoothing*.
        * ``"spline"`` – Piecewise cubic interpolation on the Delaunay
          triangulation of the observations via
          :func:`scipy.interpolate.griddata` (``method="cubic"``).
          Extrapolation outside the convex hull falls back to
          nearest-neighbour.  Requires ≥ 4 observations.
    rbf_kernel : str
        Kernel for the RBF backend (e.g. ``"thin_plate_spline"``,
        ``"multiquadric"``, ``"gaussian"``).
    rbf_smoothing : float
        RBF smoothing factor (0 = exact interpolation).
    """

    grid: SpatialGrid
    method: Literal["rbf", "spline"]
    _kernel_args: dict[str, Any]
    _xy_obs: np.ndarray | None
    _values: np.ndarray | None

    def __init__(
        self,
        grid: SpatialGrid,
        method: Literal["rbf", "spline"] = "rbf",
        **kernel_args: Any,
    ) -> None:

        if method not in ("rbf", "spline"):
            raise ValueError(f"method must be 'rbf' or 'spline'; got {method!r}")
        self.grid = grid
        self.method = method

        self._kernel_args = kernel_args

        self._xy_obs: np.ndarray | None = None
        self._values: np.ndarray | None = None

    def fit(self, xy_obs: np.ndarray, values: np.ndarray) -> FieldInterpolator:
        """Store observations.  Returns ``self`` for chaining.

        Parameters
        ----------
        xy_obs : np.ndarray
            Observed positions of shape ``(N, 2)`` – ``[east_m, north_m]``.
        values : np.ndarray
            Observed scalar values of shape ``(N,)``.

        Returns
        -------
        FieldInterpolator
            This instance.
        """
        self._validate(xy_obs, values)
        self._xy_obs = np.asarray(xy_obs, dtype=np.float64)
        self._values = np.asarray(values, dtype=np.float64)
        return self

    def predict(self) -> Array:
        """Interpolate stored observations onto the grid.

        Returns
        -------
        field : Array
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
        xy_obs : np.ndarray
            Observed positions of shape ``(N, 2)``.
        values : np.ndarray
            Observed scalar values of shape ``(N,)``.

        Returns
        -------
        field : Array
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
            **self._kernel_args,
        )
        # Convert grid query points to NumPy array (required by SciPy RBFInterpolator)
        query_pts_np = np.asarray(self.grid.query_points, dtype=np.float64)
        return rbf(query_pts_np)

    def _spline(self, xy_obs: np.ndarray, values: np.ndarray) -> np.ndarray:
        # Use scipy griddata with cubic method — robust for arbitrary scattered
        # data with no domain/knot constraints.  Regions outside the convex
        # hull of the observations fall back to nearest-neighbour.
        query_pts_np = np.asarray(self.grid.query_points, dtype=np.float64)
        z_cubic = griddata(xy_obs, values, query_pts_np, method="cubic")
        outside = np.isnan(z_cubic)
        if outside.any():
            z_nn = griddata(xy_obs, values, query_pts_np, method="nearest")
            z_cubic[outside] = z_nn[outside]
        return z_cubic
