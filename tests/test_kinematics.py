"""Tests for pde_slam.kinematics (DiffDriveKinematics and BaseKinematics)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from pde_slam.kinematics import BaseKinematics, DiffDriveKinematics

# ---------------------------------------------------------------------------
# Single step — cardinal directions
# ---------------------------------------------------------------------------


class TestStep:
    """Unit tests for DiffDriveKinematics.step()."""

    def test_zero_velocity_stays_put(self) -> None:
        """Robot with zero velocity and zero omega does not move."""
        robot = DiffDriveKinematics(x0=5.0, y0=3.0, heading0=0.0)
        state = robot.step(v=0.0, omega=0.0, dt=1.0)
        assert state[0] == pytest.approx(5.0)
        assert state[1] == pytest.approx(3.0)
        assert state[2] == pytest.approx(0.0)

    def test_east_advances_x_only(self) -> None:
        """Heading = 0.0 (East) advances x; y is unchanged."""
        robot = DiffDriveKinematics(x0=0.0, y0=0.0, heading0=0.0)
        state = robot.step(v=2.0, omega=0.0, dt=3.0)
        assert state[0] == pytest.approx(6.0)
        assert state[1] == pytest.approx(0.0, abs=1e-12)
        assert state[2] == pytest.approx(0.0)

    def test_north_advances_y_only(self) -> None:
        """Heading = π/2 (North) advances y; x is unchanged."""
        robot = DiffDriveKinematics(x0=0.0, y0=0.0, heading0=np.pi / 2)
        state = robot.step(v=1.5, omega=0.0, dt=2.0)
        assert state[0] == pytest.approx(0.0, abs=1e-12)
        assert state[1] == pytest.approx(3.0)
        assert state[2] == pytest.approx(np.pi / 2)

    def test_west_decreases_x(self) -> None:
        """Heading = π (West) decreases x."""
        robot = DiffDriveKinematics(x0=10.0, y0=0.0, heading0=np.pi)
        state = robot.step(v=2.0, omega=0.0, dt=1.0)
        assert state[0] == pytest.approx(8.0)
        assert state[1] == pytest.approx(0.0, abs=1e-12)

    def test_south_decreases_y(self) -> None:
        """Heading = -π/2 (South) decreases y."""
        robot = DiffDriveKinematics(x0=0.0, y0=5.0, heading0=-np.pi / 2)
        state = robot.step(v=2.0, omega=0.0, dt=1.0)
        assert state[0] == pytest.approx(0.0, abs=1e-12)
        assert state[1] == pytest.approx(3.0)

    def test_omega_turns_heading(self) -> None:
        """Omega updates heading linearly by omega * dt."""
        robot = DiffDriveKinematics(x0=0.0, y0=0.0, heading0=0.0)
        state = robot.step(v=0.0, omega=0.5, dt=2.0)
        assert state[2] == pytest.approx(1.0)
        assert robot.heading_rad == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# State accessors & reset
# ---------------------------------------------------------------------------


class TestStateAccessorsAndReset:
    """Unit tests for state properties and reset()."""

    def test_state_properties(self) -> None:
        """Accessors reflect current position and heading."""
        robot = DiffDriveKinematics(x0=1.0, y0=2.0, heading0=0.5)
        assert robot.x_m == pytest.approx(1.0)
        assert robot.y_m == pytest.approx(2.0)
        assert robot.heading_rad == pytest.approx(0.5)
        assert robot.state.shape == (3,)

    def test_reset_restores_state(self) -> None:
        """reset() returns robot to the specified state."""
        robot = DiffDriveKinematics(x0=0.0, y0=0.0, heading0=0.0)
        robot.step(v=5.0, omega=0.2, dt=2.0)
        robot.reset(x0=10.0, y0=-5.0, heading0=np.pi / 4)
        assert robot.x_m == pytest.approx(10.0)
        assert robot.y_m == pytest.approx(-5.0)
        assert robot.heading_rad == pytest.approx(np.pi / 4)


# ---------------------------------------------------------------------------
# integrate_trajectory & _motion_model
# ---------------------------------------------------------------------------


class TestTrajectoryIntegration:
    """Unit tests for static integration methods."""

    def test_integrate_trajectory_shape(self) -> None:
        """integrate_trajectory produces expected array shape."""
        x0 = jnp.array([0.0, 0.0, 0.0])
        vels = jnp.array([1.0, 1.0, 1.0])
        omegas = jnp.array([0.0, 0.1, -0.1])
        traj = DiffDriveKinematics.integrate_trajectory(x0, vels, omegas, dt=1.0)
        assert traj.shape == (4, 3)
        assert traj[0, 0] == pytest.approx(0.0)

    def test_motion_model_single_step(self) -> None:
        """_motion_model executes pure differential drive step."""
        x = jnp.array([0.0, 0.0, 0.0])
        u = jnp.array([2.0, 0.5])
        x_next = DiffDriveKinematics._motion_model(x, u, dt=1.0)
        assert x_next[0] == pytest.approx(2.0)
        assert x_next[1] == pytest.approx(0.0, abs=1e-12)
        assert x_next[2] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Waypoint Following
# ---------------------------------------------------------------------------


class TestWaypointFollowing:
    """Unit tests for drive_to_waypoints()."""

    def test_empty_waypoints_raises(self) -> None:
        """Passing empty waypoints raises ValueError."""
        robot = DiffDriveKinematics()
        with pytest.raises(ValueError, match="empty"):
            robot.drive_to_waypoints(np.empty((0, 2)), speed_mps=1.5, dt=1.0)

    def test_single_waypoint_reached(self) -> None:
        """Robot successfully navigates towards and reaches single waypoint."""
        robot = DiffDriveKinematics(x0=0.0, y0=0.0, heading0=0.0)
        waypoints = np.array([[10.0, 0.0]])
        states, vels, omegas = robot.drive_to_waypoints(
            waypoints, speed_mps=2.0, dt=1.0, acceptance_radius=3.0
        )
        assert len(states) > 1
        assert len(vels) == len(omegas)
        final_pos = states[-1, :2]
        dist_to_wp = np.linalg.norm(final_pos - waypoints[0])
        assert dist_to_wp <= 3.0

    def test_multiple_waypoints_followed(self) -> None:
        """Robot navigates through multiple waypoints in sequence."""
        robot = DiffDriveKinematics(x0=0.0, y0=0.0, heading0=0.0)
        waypoints = np.array([[10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
        states, vels, omegas = robot.drive_to_waypoints(
            waypoints, speed_mps=2.0, dt=0.5, acceptance_radius=2.0
        )
        assert len(states) > len(waypoints)
        final_pos = states[-1, :2]
        assert np.linalg.norm(final_pos - waypoints[-1]) <= 2.5


# ---------------------------------------------------------------------------
# BaseKinematics
# ---------------------------------------------------------------------------


class TestBaseKinematics:
    """Tests for the abstract BaseKinematics contract."""

    def test_cannot_instantiate_base_class(self) -> None:
        """BaseKinematics cannot be directly instantiated."""
        with pytest.raises(TypeError):
            BaseKinematics()  # type: ignore[abstract]

    def test_diff_drive_inherits_base(self) -> None:
        """DiffDriveKinematics inherits from BaseKinematics."""
        robot = DiffDriveKinematics()
        assert isinstance(robot, BaseKinematics)
