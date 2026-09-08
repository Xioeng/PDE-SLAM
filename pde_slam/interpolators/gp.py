"""
pde_slam/interpolators/gp.py
============================
Gaussian Process (GP) spatial regression mapping for 2D scalar fields based on
scikit-learn's GaussianProcessRegressor backend with exact Bayesian predictive
mean and variance calculation.

Supports:
- Single-field and multi-field spatial data (F >= 1).
- Kernel choices: RBF (Squared Exponential), Matern (nu=1.5), Matern (nu=2.5).
- Hyperparameter tuning or fixed hyperparameter inference.
- Seamless JAX array returns for compatibility with JAX pipelines.
"""

from __future__ import annotations

from typing import Any, Literal

import jax.numpy as jnp
import numpy as np
from jax import Array
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    Matern,
    WhiteKernel,
)

from pde_slam.interpolators.grid import SpatialGrid


class GaussianProcessField:
    """2D Gaussian Process Spatial Regression Field powered by Scikit-Learn.

    Parameters
    ----------
    grid : SpatialGrid or None, default=None
        Optional target SpatialGrid definition for grid predictions.
    lengthscale : float, default=20.0
        Spatial correlation length scale [m].
    signal_variance : float, default=1.0
        Prior signal variance sigma_f^2 of the field.
    noise_variance : float, default=0.01
        Measurement noise variance sigma_n^2.
    kernel : {"rbf", "matern32", "matern52"}, default="rbf"
        Stationary covariance kernel choice.
    optimize_hyperparams : bool, default=False
        Whether to optimize kernel hyperparameters via marginal likelihood
        during fit (default False uses fixed parameters for maximum online speed).
    """

    grid: SpatialGrid | None
    lengthscale: float
    signal_variance: float
    noise_variance: float
    kernel_name: str
    optimize_hyperparams: bool
    _models: list[GaussianProcessRegressor]
    _x_train: np.ndarray | None
    _n_fields: int

    def __init__(
        self,
        grid: SpatialGrid | None = None,
        lengthscale: float = 20.0,
        signal_variance: float = 1.0,
        noise_variance: float = 0.01,
        kernel: Literal["rbf", "matern32", "matern52"] = "rbf",
        optimize_hyperparams: bool = False,
    ) -> None:
        if kernel not in ("rbf", "matern32", "matern52"):
            msg = (
                f"Unknown kernel '{kernel}'. Supported: 'rbf', 'matern32', 'matern52'."
            )
            raise ValueError(msg)

        self.grid = grid
        self.lengthscale = float(lengthscale)
        self.signal_variance = float(signal_variance)
        self.noise_variance = float(noise_variance)
        self.kernel_name = kernel
        self.optimize_hyperparams = optimize_hyperparams

        self._models: list[GaussianProcessRegressor] = []
        self._x_train: np.ndarray | None = None
        self._n_fields: int = 1

    @property
    def is_fitted(self) -> bool:
        """Whether the Gaussian Process has been fitted to data."""
        return len(self._models) > 0 and self._x_train is not None

    def _build_kernel(self) -> Any:
        """Build Scikit-Learn kernel object based on configuration."""
        c_bounds = "fixed" if not self.optimize_hyperparams else (1e-3, 1e3)
        c_k = ConstantKernel(
            constant_value=self.signal_variance,
            constant_value_bounds=c_bounds,
        )
        l_bounds = "fixed" if not self.optimize_hyperparams else (1.0, 500.0)

        if self.kernel_name == "rbf":
            spatial_k = RBF(length_scale=self.lengthscale, length_scale_bounds=l_bounds)
        elif self.kernel_name == "matern32":
            spatial_k = Matern(
                length_scale=self.lengthscale, length_scale_bounds=l_bounds, nu=1.5
            )
        else:
            spatial_k = Matern(
                length_scale=self.lengthscale, length_scale_bounds=l_bounds, nu=2.5
            )

        w_bounds = "fixed" if not self.optimize_hyperparams else (1e-5, 1.0)
        w_k = WhiteKernel(
            noise_level=self.noise_variance,
            noise_level_bounds=w_bounds,
        )
        return c_k * spatial_k + w_k

    def fit(
        self, x_obs: Array | np.ndarray, values: Array | np.ndarray
    ) -> GaussianProcessField:
        """Fit Gaussian Process models to observations across all scalar fields.

        Parameters
        ----------
        x_obs : Array or np.ndarray
            Observed 2D spatial positions, shape (N, 2).
        values : Array or np.ndarray
            Observed scalar values, shape (N,) if single field or (N, F) for
            F simultaneous fields.

        Returns
        -------
        GaussianProcessField
            This instance.
        """
        x_np = np.asarray(x_obs, dtype=np.float64)
        y_np = np.asarray(values, dtype=np.float64)

        if x_np.ndim != 2 or x_np.shape[1] != 2:
            msg = f"x_obs must have shape (N, 2), got {x_np.shape}."
            raise ValueError(msg)

        if y_np.ndim == 1:
            self._n_fields = 1
            y_cols = [y_np]
        elif y_np.ndim == 2:
            self._n_fields = y_np.shape[1]
            y_cols = [y_np[:, f] for f in range(self._n_fields)]
        else:
            msg = f"values must have shape (N,) or (N, F), got {y_np.shape}."
            raise ValueError(msg)

        optimizer = "fmin_l_bfgs_b" if self.optimize_hyperparams else None
        self._models = []

        for y_f in y_cols:
            kernel_obj = self._build_kernel()
            gpr = GaussianProcessRegressor(
                kernel=kernel_obj,
                alpha=1e-8,
                optimizer=optimizer,
                normalize_y=False,
            )
            gpr.fit(x_np, y_f)
            self._models.append(gpr)

        self._x_train = x_np
        return self

    def predict(self, x_query: Array | np.ndarray) -> tuple[Array, Array]:
        """Compute predictive mean and variance at query positions via scikit-learn.

        Parameters
        ----------
        x_query : Array or np.ndarray
            Query 2D positions, shape (M, 2) or (2,).

        Returns
        -------
        mean : Array
            Predictive mean as JAX array, shape (M,) or (M, F).
        variance : Array
            Predictive variance as JAX array, shape (M,) or (M, F).
        """
        if not self.is_fitted:
            msg = "GaussianProcessField is not fitted. Call fit() before predict()."
            raise RuntimeError(msg)

        x_q = np.atleast_2d(np.asarray(x_query, dtype=np.float64))
        means_list: list[np.ndarray] = []
        vars_list: list[np.ndarray] = []

        for gpr in self._models:
            mu, std = gpr.predict(x_q, return_std=True)
            means_list.append(mu)
            vars_list.append(std**2)

        if self._n_fields == 1:
            mean_arr = jnp.asarray(means_list[0])
            var_arr = jnp.asarray(vars_list[0])
        else:
            mean_arr = jnp.column_stack([jnp.asarray(m) for m in means_list])
            var_arr = jnp.column_stack([jnp.asarray(v) for v in vars_list])

        if np.ndim(x_query) == 1:
            return mean_arr[0], var_arr[0]
        return mean_arr, var_arr

    def predict_grid(self, grid: SpatialGrid | None = None) -> tuple[Array, Array]:
        """Evaluate GP predictive mean and variance across 2D spatial grid.

        Parameters
        ----------
        grid : SpatialGrid or None, default=None
            Optional override for spatial grid definition. If None, uses self.grid.

        Returns
        -------
        grid_mean : Array
            Predictive mean map, shape (ny, nx) if n_fields=1 or (ny, nx, F) if F > 1.
        grid_var : Array
            Predictive variance map, shape (ny, nx) if n_fields=1 or (ny, nx, F).
        """
        target_grid = self.grid if grid is None else grid
        if target_grid is None:
            msg = "No SpatialGrid provided. Pass grid to constructor or predict_grid()."
            raise ValueError(msg)

        flat_x = np.ravel(np.asarray(target_grid.XX))
        flat_y = np.ravel(np.asarray(target_grid.YY))
        x_mesh = np.column_stack([flat_x, flat_y])

        mean_flat, var_flat = self.predict(x_mesh)

        if self._n_fields == 1:
            grid_mean = jnp.reshape(mean_flat, target_grid.XX.shape)
            grid_var = jnp.reshape(var_flat, target_grid.XX.shape)
        else:
            grid_mean = jnp.reshape(
                mean_flat, (target_grid.ny, target_grid.nx, self._n_fields)
            )
            grid_var = jnp.reshape(
                var_flat, (target_grid.ny, target_grid.nx, self._n_fields)
            )

        return grid_mean, grid_var
