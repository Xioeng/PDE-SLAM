"""Tests for pde_slam.kinematics."""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pde_slam.kinematics import (
    GeoOrigin,
    KinematicParams,
    apply_drift_correction,
    dead_reckon_step,
    enu_to_latlon,
    integrate_trajectory,
    latlon_to_enu,
)


# ---------------------------------------------------------------------------
# WGS-84 projection tests
# ---------------------------------------------------------------------------


class TestWGS84Projection:
    """Round-trip and sanity checks for WGS-84 ↔ ENU conversion."""

    ORIGIN = GeoOrigin(lat_deg=36.8, lon_deg=-76.0, alt_m=0.0)

    def test_origin_maps_to_zero(self) -> None:
        """The reference origin must project to (0, 0)."""
        east, north = latlon_to_enu(
            jnp.array(self.ORIGIN.lat_deg),
            jnp.array(self.ORIGIN.lon_deg),
            self.ORIGIN,
        )
        assert float(east) == pytest.approx(0.0, abs=1e-6)
        assert float(north) == pytest.approx(0.0, abs=1e-6)

    def test_round_trip_latlon_enu_latlon(self) -> None:
        """latlon → ENU → latlon should recover the original coordinates."""
        lat_in = jnp.array(36.9)
        lon_in = jnp.array(-75.9)
        east, north = latlon_to_enu(lat_in, lon_in, self.ORIGIN)
        lat_out, lon_out = enu_to_latlon(east, north, self.ORIGIN)

        assert float(lat_out) == pytest.approx(float(lat_in), rel=1e-6)
        assert float(lon_out) == pytest.approx(float(lon_in), rel=1e-6)

    def test_northward_displacement(self) -> None:
        """Moving 1° north ≈ 111 km at any longitude."""
        _, north = latlon_to_enu(
            jnp.array(self.ORIGIN.lat_deg + 1.0),
            jnp.array(self.ORIGIN.lon_deg),
            self.ORIGIN,
        )
        # Should be close to the meridional circumference / 360
        assert 110_000 < float(north) < 112_000


# ---------------------------------------------------------------------------
# Dead-reckoning tests
# ---------------------------------------------------------------------------


class TestDeadReckoning:
    """Unit tests for the kinematic integrator."""

    PARAMS = KinematicParams(c_v=1.0, c_omega=1.0)
    DT = 1.0

    def test_stationary_zero_control(self) -> None:
        """Zero thrust and zero rudder should leave the pose unchanged."""
        pose = jnp.array([1.0, 2.0, 0.5])
        ctrl = jnp.array([0.0, 0.0])
        next_pose = dead_reckon_step(pose, ctrl, self.DT, self.PARAMS)
        np.testing.assert_allclose(np.array(next_pose), np.array(pose), atol=1e-7)

    def test_forward_motion_east(self) -> None:
        """Heading = 0 rad (East) + full thrust should move +x only."""
        pose = jnp.zeros(3)  # [0, 0, 0 rad]
        ctrl = jnp.array([1.0, 0.0])  # full thrust, no rudder
        next_pose = dead_reckon_step(pose, ctrl, self.DT, self.PARAMS)

        assert float(next_pose[0]) == pytest.approx(1.0, rel=1e-5)  # moved east
        assert float(next_pose[1]) == pytest.approx(0.0, abs=1e-7)  # no north
        assert float(next_pose[2]) == pytest.approx(0.0, abs=1e-7)  # heading unchanged

    def test_trajectory_length(self) -> None:
        """integrate_trajectory output shape must be (N+1, 3)."""
        N = 20
        controls = jnp.ones((N, 2)) * jnp.array([0.5, 0.1])
        poses = integrate_trajectory(jnp.zeros(3), controls, self.DT, self.PARAMS)
        assert poses.shape == (N + 1, 3)

    def test_drift_correction_additive(self) -> None:
        """Drift correction must be purely additive."""
        N = 10
        controls = jnp.zeros((N, 2))
        poses = integrate_trajectory(jnp.zeros(3), controls, self.DT, self.PARAMS)
        deltax = jnp.ones_like(poses) * 0.5
        corrected = apply_drift_correction(poses, deltax)
        np.testing.assert_allclose(
            np.array(corrected), np.array(poses + deltax), atol=1e-7
        )
