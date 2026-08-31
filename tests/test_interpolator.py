"""Tests for pde_slam.interpolators (FieldInterpolator class API)."""

from __future__ import annotations

import numpy as np
import pytest

from pde_slam.interpolators import FieldInterpolator, SpatialGrid

GRID = SpatialGrid(x_min=-50.0, x_max=50.0, y_min=-50.0, y_max=50.0, nx=20, ny=20)


def _make_scattered(n: int = 80, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Generate scattered observations on a smooth test function."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-45.0, 45.0, size=(n, 2))
    values = np.sin(xy[:, 0] / 20.0) * np.cos(xy[:, 1] / 20.0)
    return xy, values


# ---------------------------------------------------------------------------
# SpatialGrid
# ---------------------------------------------------------------------------


class TestSpatialGrid:
    def test_shape(self) -> None:
        assert GRID.shape == (20, 20)

    def test_spacing(self) -> None:
        assert GRID.dx == pytest.approx(100.0 / 19.0, rel=1e-6)

    def test_query_points_count(self) -> None:
        assert GRID.query_points.shape == (20 * 20, 2)

    def test_repr(self) -> None:
        assert "SpatialGrid" in repr(GRID)


# ---------------------------------------------------------------------------
# FieldInterpolator – RBF backend
# ---------------------------------------------------------------------------


class TestRBFInterpolation:
    def test_output_shape(self) -> None:
        xy, vals = _make_scattered()
        field = FieldInterpolator(GRID, method="rbf").fit_predict(xy, vals)
        assert field.shape == (20, 20)

    def test_output_finite(self) -> None:
        import jax.numpy as jnp

        xy, vals = _make_scattered()
        field = FieldInterpolator(GRID, method="rbf").fit_predict(xy, vals)
        assert bool(jnp.all(jnp.isfinite(field)))

    def test_accuracy_at_observations(self) -> None:
        """RBF with zero smoothing should near-exactly fit observation values."""
        xy, vals = _make_scattered(n=50)
        field = FieldInterpolator(
            GRID, method="rbf", kernel="thin_plate_spline", smoothing=0.0
        ).fit_predict(xy, vals)
        field_np = np.array(field)
        assert field_np.min() >= vals.min() - 0.2
        assert field_np.max() <= vals.max() + 0.2

    def test_fit_then_predict(self) -> None:
        """fit() and predict() can be called separately."""
        xy, vals = _make_scattered()
        interp = FieldInterpolator(GRID, method="rbf")
        interp.fit(xy, vals)
        field = interp.predict()
        assert field.shape == (20, 20)

    def test_predict_before_fit_raises(self) -> None:
        interp = FieldInterpolator(GRID, method="rbf")
        with pytest.raises(RuntimeError, match="fit()"):
            interp.predict()


# ---------------------------------------------------------------------------
# FieldInterpolator – Spline backend
# ---------------------------------------------------------------------------


class TestSplineInterpolation:
    def test_output_shape(self) -> None:
        xy, vals = _make_scattered(n=80)
        field = FieldInterpolator(GRID, method="spline").fit_predict(xy, vals)
        assert field.shape == (20, 20)

    def test_too_few_points_raises(self) -> None:
        xy, vals = _make_scattered(n=3)
        with pytest.raises(ValueError, match="4"):
            FieldInterpolator(GRID, method="spline").fit_predict(xy, vals)


# ---------------------------------------------------------------------------
# Invalid constructor
# ---------------------------------------------------------------------------


class TestConstructorValidation:
    def test_invalid_method_raises(self) -> None:
        with pytest.raises(ValueError, match="method must be"):
            FieldInterpolator(GRID, method="kriging")  # type: ignore[arg-type]

    def test_bad_xy_shape_raises(self) -> None:
        xy = np.random.randn(10, 3)  # wrong second dimension
        vals = np.ones(10)
        with pytest.raises(ValueError, match="shape"):
            FieldInterpolator(GRID).fit(xy, vals)


# ---------------------------------------------------------------------------
# SpatiotemporalInterpolator
# ---------------------------------------------------------------------------


class TestSpatiotemporalInterpolator:
    def test_spatiotemporal_interpolation(self) -> None:
        import jax
        import jax.numpy as jnp

        from pde_slam.interpolators import SpatiotemporalInterpolator

        # Create an asymmetric 4x3 grid
        grid = SpatialGrid(x_min=0.0, x_max=3.0, y_min=0.0, y_max=2.0, nx=4, ny=3)
        ts = jnp.array([0.0, 1.0, 2.0])

        # Create snapshots with 'ij' indexing: f(x, y, t) = x + 2*y + 3*t
        t_grid, x_grid, y_grid = jnp.meshgrid(
            ts, grid.XX[:, 0], grid.YY[0, :], indexing="ij"
        )
        snapshots = x_grid + 2.0 * y_grid + 3.0 * t_grid  # shape (3, 4, 3)

        interp = SpatiotemporalInterpolator(grid, ts, snapshots)

        # 1. Test exact values at grid nodes
        val = interp(1.0, 1.0, 1.0)
        assert float(val) == pytest.approx(6.0, abs=1e-5)

        # 2. Test intermediate coordinates (trilinear interpolation)
        # (x, y, t) = (0.5, 1.5, 0.5) -> 0.5 + 2(1.5) + 3(0.5) = 5.0
        val_interp = interp(0.5, 1.5, 0.5)
        assert float(val_interp) == pytest.approx(5.0, abs=1e-5)

        # 3. Test differentiability w.r.t coordinates
        grad_fn = jax.grad(lambda x: interp(x, 1.0, 1.0))
        assert float(grad_fn(1.0)) == pytest.approx(1.0, abs=1e-3)

        # 4. Test differentiability w.r.t snapshots
        def loss(snaps):
            model = SpatiotemporalInterpolator(grid, ts, snaps)
            return model(0.5, 0.5, 0.5)

        grad_snaps = jax.grad(loss)(snapshots)
        assert grad_snaps.shape == snapshots.shape
        assert float(jnp.sum(grad_snaps)) == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# GaussianProcessField
# ---------------------------------------------------------------------------


class TestGaussianProcessField:
    def test_gp_single_field_fit_predict(self) -> None:
        import jax.numpy as jnp

        from pde_slam.interpolators.gp import GaussianProcessField

        xy, vals = _make_scattered(n=30, seed=1)
        gp = GaussianProcessField(
            grid=GRID, lengthscale=20.0, signal_variance=1.0, noise_variance=1e-4
        )
        gp.fit(jnp.array(xy), jnp.array(vals))

        # 1. Prediction at observations should be close to ground truth
        mu_obs, var_obs = gp.predict(jnp.array(xy[:5]))
        assert mu_obs.shape == (5,)
        assert var_obs.shape == (5,)
        assert jnp.allclose(mu_obs, vals[:5], atol=0.05)
        assert bool(jnp.all(var_obs < 0.05))

        # 2. Prediction far away should have variance approaching signal_variance
        far_point = jnp.array([[200.0, 200.0]])
        _, var_far = gp.predict(far_point)
        assert float(var_far[0]) == pytest.approx(1.0, abs=0.05)

    def test_gp_multi_field_and_grid_predict(self) -> None:
        import jax.numpy as jnp

        from pde_slam.interpolators.gp import GaussianProcessField

        xy, vals1 = _make_scattered(n=25, seed=2)
        _, vals2 = _make_scattered(n=25, seed=3)
        multi_vals = jnp.column_stack([vals1, vals2])  # (25, 2)

        gp = GaussianProcessField(grid=GRID, lengthscale=15.0, kernel="matern32")
        gp.fit(jnp.array(xy), multi_vals)

        # Multi-field query shape
        query_pts = jnp.array([[0.0, 0.0], [10.0, -10.0]])
        mu, var = gp.predict(query_pts)
        assert mu.shape == (2, 2)
        assert var.shape == (2, 2)

        # Grid prediction
        grid_mu, grid_var = gp.predict_grid(GRID)
        assert grid_mu.shape == (20, 20, 2)
        assert grid_var.shape == (20, 20, 2)
        assert bool(jnp.all(jnp.isfinite(grid_mu)))
        assert bool(jnp.all(grid_var >= 0.0))
