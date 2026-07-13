"""Tests for pde_slam.coords (ENUFrame class)."""

from __future__ import annotations

import numpy as np
import pytest

from pde_slam.coords import ENUFrame

# A realistic survey origin (Norfolk, VA area — matches the synthetic survey CSV)
LAT0 = 36.7996
LON0 = -76.0000


# ---------------------------------------------------------------------------
# Construction & repr
# ---------------------------------------------------------------------------


class TestENUFrameConstruction:
    def test_attributes_stored(self) -> None:
        frame = ENUFrame(lat0=LAT0, lon0=LON0)
        assert frame.lat0 == LAT0
        assert frame.lon0 == LON0

    def test_repr(self) -> None:
        frame = ENUFrame(lat0=LAT0, lon0=LON0)
        assert "ENUFrame" in repr(frame)
        assert str(LAT0) in repr(frame)

    def test_equality(self) -> None:
        a = ENUFrame(lat0=LAT0, lon0=LON0)
        b = ENUFrame(lat0=LAT0, lon0=LON0)
        assert a == b

    def test_inequality_different_origin(self) -> None:
        a = ENUFrame(lat0=LAT0, lon0=LON0)
        b = ENUFrame(lat0=LAT0 + 0.001, lon0=LON0)
        assert a != b


# ---------------------------------------------------------------------------
# to_enu — geodetic → ENU
# ---------------------------------------------------------------------------


class TestToENU:
    def test_origin_maps_to_zero(self) -> None:
        frame = ENUFrame(lat0=LAT0, lon0=LON0)
        e, n = frame.to_enu(LAT0, LON0)
        assert float(e) == pytest.approx(0.0, abs=1e-9)
        assert float(n) == pytest.approx(0.0, abs=1e-9)

    def test_north_displacement(self) -> None:
        """1° north ≈ 111 195 m (R_earth * π/180)."""
        frame = ENUFrame(lat0=0.0, lon0=0.0)
        _, n = frame.to_enu(1.0, 0.0)
        assert float(n) == pytest.approx(6_378_137.0 * np.pi / 180.0, rel=1e-6)

    def test_east_displacement_at_equator(self) -> None:
        """1° east at equator (lat0=0) ≈ same as north."""
        frame = ENUFrame(lat0=0.0, lon0=0.0)
        e, _ = frame.to_enu(0.0, 1.0)
        assert float(e) == pytest.approx(6_378_137.0 * np.pi / 180.0, rel=1e-6)

    def test_east_shrinks_with_latitude(self) -> None:
        """Metres-per-degree east decreases poleward."""
        frame_eq = ENUFrame(lat0=0.0, lon0=0.0)
        frame_hi = ENUFrame(lat0=60.0, lon0=0.0)
        e_eq, _ = frame_eq.to_enu(0.0, 1.0)
        e_hi, _ = frame_hi.to_enu(60.0, 1.0)
        assert float(e_hi) < float(e_eq)
        # cos(60°) = 0.5
        assert float(e_hi) == pytest.approx(float(e_eq) * 0.5, rel=1e-4)

    def test_array_input(self) -> None:
        frame = ENUFrame(lat0=LAT0, lon0=LON0)
        lats = np.array([LAT0, LAT0 + 0.001, LAT0 - 0.001])
        lons = np.array([LON0, LON0 + 0.001, LON0 - 0.001])
        e, n = frame.to_enu(lats, lons)
        assert e.shape == (3,)
        assert n.shape == (3,)

    def test_numpy_input_accepted(self) -> None:
        frame = ENUFrame(lat0=LAT0, lon0=LON0)
        lats = np.array([LAT0, LAT0 + 0.001])
        lons = np.array([LON0, LON0 + 0.001])
        e, n = frame.to_enu(lats, lons)
        assert e.shape == (2,)


# ---------------------------------------------------------------------------
# to_enu_xy — stacked output for FieldInterpolator
# ---------------------------------------------------------------------------


class TestToENUXY:
    def test_shape(self) -> None:
        frame = ENUFrame(lat0=LAT0, lon0=LON0)
        lats = np.linspace(LAT0 - 0.001, LAT0 + 0.001, 50)
        lons = np.linspace(LON0 - 0.001, LON0 + 0.001, 50)
        xy = frame.to_enu_xy(lats, lons)
        assert xy.shape == (50, 2)

    def test_columns_match_to_enu(self) -> None:
        frame = ENUFrame(lat0=LAT0, lon0=LON0)
        lats = np.array([LAT0 + 0.002, LAT0 - 0.001])
        lons = np.array([LON0 - 0.001, LON0 + 0.003])
        e, n = frame.to_enu(lats, lons)
        xy = frame.to_enu_xy(lats, lons)
        np.testing.assert_allclose(xy[:, 0], e)
        np.testing.assert_allclose(xy[:, 1], n)


# ---------------------------------------------------------------------------
# from_enu — ENU → geodetic (inverse)
# ---------------------------------------------------------------------------


class TestFromENU:
    def test_origin_round_trip(self) -> None:
        frame = ENUFrame(lat0=LAT0, lon0=LON0)
        lat, lon = frame.from_enu(0.0, 0.0)
        assert float(lat) == pytest.approx(LAT0, rel=1e-9)
        assert float(lon) == pytest.approx(LON0, rel=1e-9)

    def test_round_trip_scalars(self) -> None:
        """to_enu → from_enu should recover the original geodetic point."""
        frame = ENUFrame(lat0=LAT0, lon0=LON0)
        lat_in, lon_in = LAT0 + 0.0018, LON0 - 0.0023
        e, n = frame.to_enu(lat_in, lon_in)
        lat_out, lon_out = frame.from_enu(e, n)
        assert float(lat_out) == pytest.approx(lat_in, rel=1e-12)
        assert float(lon_out) == pytest.approx(lon_in, rel=1e-12)

    def test_round_trip_arrays(self) -> None:
        frame = ENUFrame(lat0=LAT0, lon0=LON0)
        rng = np.random.default_rng(42)
        lats = LAT0 + rng.uniform(-0.005, 0.005, 100)
        lons = LON0 + rng.uniform(-0.005, 0.005, 100)
        e, n = frame.to_enu(lats, lons)
        lat_out, lon_out = frame.from_enu(e, n)
        np.testing.assert_allclose(np.array(lat_out), lats, rtol=1e-12)
        np.testing.assert_allclose(np.array(lon_out), lons, rtol=1e-12)

    def test_known_distance_north(self) -> None:
        """100 m north should increase latitude by ~100 / R_earth * 180/π degrees."""
        frame = ENUFrame(lat0=0.0, lon0=0.0)
        lat, _ = frame.from_enu(0.0, 100.0)
        expected = 100.0 / (6_378_137.0 * np.pi / 180.0)
        assert float(lat) == pytest.approx(float(expected), rel=1e-6)
