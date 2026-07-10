"""Tests for pde_slam.interpolator (FieldInterpolator class API)."""
from __future__ import annotations

import numpy as np
import pytest

from pde_slam.interpolator import FieldInterpolator, SpatialGrid

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
            GRID, method="rbf", rbf_kernel="thin_plate_spline", rbf_smoothing=0.0
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
        xy = np.random.randn(10, 3)   # wrong second dimension
        vals = np.ones(10)
        with pytest.raises(ValueError, match="shape"):
            FieldInterpolator(GRID).fit(xy, vals)

