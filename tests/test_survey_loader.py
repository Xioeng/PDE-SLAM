"""Tests for pde_slam.survey_loader (SurveyLoader class)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest

from pde_slam.coords import ENUFrame
from pde_slam.interpolators import SpatialGrid
from pde_slam.survey_loader import SurveyLoader

# ---------------------------------------------------------------------------
# Shared fixtures & helpers
# ---------------------------------------------------------------------------

# Origin matching the synthetic survey CSV area (Norfolk, VA)
FRAME = ENUFrame(lat0=36.7996, lon0=-76.0000)
GRID = SpatialGrid(x_min=-500.0, x_max=500.0, y_min=-500.0, y_max=500.0, nx=16, ny=16)

# Number of synthetic rows — small enough for fast tests
_N = 50


def _make_csv(tmp_path: Path, rows: int = _N, extra_col: bool = True) -> Path:
    """Write a minimal survey CSV with realistic columns."""
    rng = np.random.default_rng(0)
    lats = 36.7996 + rng.uniform(-0.003, 0.003, rows)
    lons = -76.0000 + rng.uniform(-0.003, 0.003, rows)
    sal = 30.0 + rng.normal(0, 0.1, rows)
    tmp = 20.0 + rng.normal(0, 0.05, rows)

    lines = ["Time,Latitude,Longitude,Salinity (PPT),Temperature (C)"]
    if extra_col:
        lines[0] += ",backscatter_db"
    for i in range(rows):
        row = f"{i},{lats[i]:.7f},{lons[i]:.7f},{sal[i]:.4f},{tmp[i]:.4f}"
        if extra_col:
            row += f",{rng.uniform(-60, -40):.4f}"
        lines.append(row)

    csv = tmp_path / "survey.csv"
    csv.write_text("\n".join(lines))
    return csv


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestSurveyLoaderConstruction:
    def test_defaults(self) -> None:
        loader = SurveyLoader(FRAME)
        assert loader.field == "Salinity (PPT)"
        assert loader.frame == FRAME

    def test_custom_field(self) -> None:
        loader = SurveyLoader(FRAME, field="Temperature (C)")
        assert loader.field == "Temperature (C)"

    def test_repr_before_load(self) -> None:
        loader = SurveyLoader(FRAME)
        assert "not loaded" in repr(loader)

    def test_access_before_load_raises(self) -> None:
        loader = SurveyLoader(FRAME)
        with pytest.raises(RuntimeError, match="load()"):
            _ = loader.xy_obs


# ---------------------------------------------------------------------------
# Loading & validation
# ---------------------------------------------------------------------------


class TestLoad:
    def test_loads_successfully(self, tmp_path: Path) -> None:
        csv = _make_csv(tmp_path)
        loader = SurveyLoader(FRAME).load(csv)
        assert loader.n_obs == _N

    def test_xy_obs_shape(self, tmp_path: Path) -> None:
        csv = _make_csv(tmp_path)
        loader = SurveyLoader(FRAME).load(csv)
        assert loader.xy_obs.shape == (_N, 2)

    def test_values_shape(self, tmp_path: Path) -> None:
        csv = _make_csv(tmp_path)
        loader = SurveyLoader(FRAME).load(csv)
        assert loader.values.shape == (_N,)

    def test_xy_obs_dtype_float64(self, tmp_path: Path) -> None:
        csv = _make_csv(tmp_path)
        loader = SurveyLoader(FRAME).load(csv)
        assert loader.xy_obs.dtype == np.float64

    def test_values_dtype_float64(self, tmp_path: Path) -> None:
        csv = _make_csv(tmp_path)
        loader = SurveyLoader(FRAME).load(csv)
        assert loader.values.dtype == np.float64

    def test_temperature_field(self, tmp_path: Path) -> None:
        csv = _make_csv(tmp_path)
        loader = SurveyLoader(FRAME, field="Temperature (C)").load(csv)
        assert loader.n_obs == _N

    def test_chaining(self, tmp_path: Path) -> None:
        """load() returns self for chaining."""
        csv = _make_csv(tmp_path)
        loader = SurveyLoader(FRAME)
        result = loader.load(csv)
        assert result is loader

    def test_repr_after_load(self, tmp_path: Path) -> None:
        csv = _make_csv(tmp_path)
        loader = SurveyLoader(FRAME).load(csv)
        assert "n_obs=" in repr(loader)

    def test_file_not_found_raises(self) -> None:
        loader = SurveyLoader(FRAME)
        with pytest.raises(FileNotFoundError):
            loader.load("nonexistent/path/survey.csv")

    def test_missing_field_raises(self, tmp_path: Path) -> None:
        csv = _make_csv(tmp_path)
        loader = SurveyLoader(FRAME, field="Oxygen (ml)")
        with pytest.raises(KeyError, match="Oxygen \\(ml\\)"):
            loader.load(csv)

    def test_missing_lat_col_raises(self, tmp_path: Path) -> None:
        csv = _make_csv(tmp_path)
        loader = SurveyLoader(FRAME, lat_col="Latitude_deg")
        with pytest.raises(KeyError, match="Latitude_deg"):
            loader.load(csv)


# ---------------------------------------------------------------------------
# ENU coordinate correctness
# ---------------------------------------------------------------------------


class TestENUConversion:
    def test_origin_row_near_zero(self, tmp_path: Path) -> None:
        """A row placed exactly at the ENUFrame origin should have ~0 easting/northing."""
        lat0, lon0 = 36.7996, -76.0000
        frame = ENUFrame(lat0=lat0, lon0=lon0)

        content = textwrap.dedent(f"""\
            Time,Latitude,Longitude,Salinity (PPT),Temperature (C)
            0,{lat0},{lon0},30.0,20.0
            1,{lat0 + 0.001},{lon0 + 0.001},30.1,20.1
            2,{lat0 - 0.001},{lon0 - 0.001},29.9,19.9
            3,{lat0 + 0.002},{lon0 - 0.001},30.2,20.2
            4,{lat0 - 0.002},{lon0 + 0.002},29.8,19.8
        """)
        csv = tmp_path / "origin.csv"
        csv.write_text(content)

        loader = SurveyLoader(frame).load(csv)
        # First row is at the origin — should be (0, 0)
        np.testing.assert_allclose(loader.xy_obs[0], [0.0, 0.0], atol=1e-6)

    def test_north_displacement_positive(self, tmp_path: Path) -> None:
        """A point north of origin has positive northing."""
        frame = ENUFrame(lat0=36.7996, lon0=-76.0000)
        content = textwrap.dedent("""\
            Time,Latitude,Longitude,Salinity (PPT),Temperature (C)
            0,36.7996,-76.0000,30.0,20.0
            1,36.8006,-76.0000,30.1,20.1
            2,36.7986,-76.0000,29.9,19.9
            3,36.7996,-75.9990,30.2,20.2
            4,36.7996,-76.0010,29.8,19.8
        """)
        csv = tmp_path / "dir.csv"
        csv.write_text(content)
        loader = SurveyLoader(frame).load(csv)
        assert loader.xy_obs[1, 1] > 0  # north of origin → positive northing
        assert loader.xy_obs[2, 1] < 0  # south → negative


# ---------------------------------------------------------------------------
# available_fields
# ---------------------------------------------------------------------------


class TestAvailableFields:
    def test_excludes_nav_cols(self, tmp_path: Path) -> None:
        csv = _make_csv(tmp_path, extra_col=True)
        loader = SurveyLoader(FRAME).load(csv)
        fields = loader.available_fields()
        assert "Latitude" not in fields
        assert "Longitude" not in fields
        assert "Time" not in fields

    def test_includes_scalar_cols(self, tmp_path: Path) -> None:
        csv = _make_csv(tmp_path, extra_col=True)
        loader = SurveyLoader(FRAME).load(csv)
        fields = loader.available_fields()
        assert "Salinity (PPT)" in fields
        assert "Temperature (C)" in fields
        assert "backscatter_db" in fields


# ---------------------------------------------------------------------------
# to_initial_condition
# ---------------------------------------------------------------------------


class TestToInitialCondition:
    def test_output_shape(self, tmp_path: Path) -> None:
        csv = _make_csv(tmp_path, rows=80)
        loader = SurveyLoader(FRAME).load(csv)
        phi0 = loader.to_initial_condition(GRID, method="rbf")
        assert phi0.shape == GRID.shape

    def test_output_finite(self, tmp_path: Path) -> None:
        import jax.numpy as jnp

        csv = _make_csv(tmp_path, rows=80)
        loader = SurveyLoader(FRAME).load(csv)
        phi0 = loader.to_initial_condition(GRID)
        assert bool(jnp.all(jnp.isfinite(phi0)))

    def test_spline_backend(self, tmp_path: Path) -> None:
        csv = _make_csv(tmp_path, rows=80)
        loader = SurveyLoader(FRAME).load(csv)
        phi0 = loader.to_initial_condition(GRID, method="spline")
        assert phi0.shape == GRID.shape

    def test_not_loaded_raises(self) -> None:
        loader = SurveyLoader(FRAME)
        with pytest.raises(RuntimeError, match="load()"):
            loader.to_initial_condition(GRID)


# ---------------------------------------------------------------------------
# Real CSV integration (skipped if file absent)
# ---------------------------------------------------------------------------


REAL_CSV = Path("data/raw/survey.csv")


@pytest.mark.skipif(not REAL_CSV.exists(), reason="data/raw/survey.csv not present")
class TestRealCSV:
    def test_loads_real_csv(self) -> None:
        frame = ENUFrame(lat0=36.7996, lon0=-76.0000)
        loader = SurveyLoader(frame, field="Salinity (PPT)").load(REAL_CSV)
        assert loader.n_obs > 100

    def test_real_csv_initial_condition(self) -> None:
        frame = ENUFrame(lat0=36.7996, lon0=-76.0000)
        grid = SpatialGrid(x_min=-500, x_max=500, y_min=-500, y_max=500, nx=32, ny=32)
        loader = SurveyLoader(frame).load(REAL_CSV)
        phi0 = loader.to_initial_condition(grid, method="rbf")
        assert phi0.shape == (32, 32)
