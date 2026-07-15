"""Tests for pde_slam.kinematics (UnicycleKinematics class)."""

from __future__ import annotations

import numpy as np
import pytest

from pde_slam.kinematics import BaseKinematics, UnicycleKinematics

# ---------------------------------------------------------------------------
# Single step — cardinal directions
# ---------------------------------------------------------------------------


class TestStep:
    """Unit tests for UnicycleKinematics.step()."""

    def test_zero_thrust_stays_put(self) -> None:
        """Robot with zero thrust does not move."""
        robot = UnicycleKinematics(k_thrust=1.0, x0=5.0, y0=3.0)
        state = robot.step(thrust=0.0, compass_heading_rad=0.0, dt=1.0)
        assert state[0] == pytest.approx(5.0)
        assert state[1] == pytest.approx(3.0)

    def test_east_advances_x_only(self) -> None:
        """Heading = π/2 (East) advances x; y is unchanged."""
        robot = UnicycleKinematics(k_thrust=2.0, x0=0.0, y0=0.0)
        state = robot.step(thrust=1.0, compass_heading_rad=np.pi / 2, dt=3.0)
        assert state[0] == pytest.approx(2.0 * 1.0 * 3.0)  # k_thrust * T * dt
        assert state[1] == pytest.approx(0.0, abs=1e-12)

    def test_north_advances_y_only(self) -> None:
        """Heading = 0 (North) advances y; x is unchanged."""
        robot = UnicycleKinematics(k_thrust=1.0, x0=0.0, y0=0.0)
        state = robot.step(thrust=1.0, compass_heading_rad=0.0, dt=2.0)
        assert state[0] == pytest.approx(0.0, abs=1e-12)
        assert state[1] == pytest.approx(2.0)

    def test_west_decreases_x(self) -> None:
        """Heading = 3π/2 (West) decreases x."""
        robot = UnicycleKinematics(k_thrust=1.0, x0=10.0, y0=0.0)
        state = robot.step(thrust=1.0, compass_heading_rad=3 * np.pi / 2, dt=1.0)
        assert state[0] == pytest.approx(9.0)
        assert state[1] == pytest.approx(0.0, abs=1e-12)

    def test_south_decreases_y(self) -> None:
        """Heading = π (South) decreases y."""
        robot = UnicycleKinematics(k_thrust=1.0, x0=0.0, y0=5.0)
        state = robot.step(thrust=1.0, compass_heading_rad=np.pi, dt=1.0)
        assert state[0] == pytest.approx(0.0, abs=1e-12)
        assert state[1] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# State accessors
# ---------------------------------------------------------------------------


class TestStateAccessors:
    """Unit tests for state properties and their invariants."""

    def test_heading_stored_after_step(self) -> None:
        """After step(), state[2] and heading_rad reflect the compass input."""
        robot = UnicycleKinematics(k_thrust=1.0)
        heading = np.pi / 4
        state = robot.step(thrust=0.5, compass_heading_rad=heading, dt=1.0)
        assert state[2] == pytest.approx(heading)
        assert robot.heading_rad == pytest.approx(heading)

    def test_state_property_is_copy(self) -> None:
        """state property returns a JAX array, which is immutable."""
        robot = UnicycleKinematics(k_thrust=1.0, x0=1.0, y0=2.0)
        s = robot.state
        with pytest.raises(TypeError):
            s[0] = 999.0
        assert robot.x_m == pytest.approx(1.0)

    def test_k_thrust_doubles_displacement(self) -> None:
        """Doubling k_thrust exactly doubles the displacement for the same command."""
        robot1 = UnicycleKinematics(k_thrust=1.0, x0=0.0, y0=0.0)
        robot2 = UnicycleKinematics(k_thrust=2.0, x0=0.0, y0=0.0)
        s1 = robot1.step(thrust=0.5, compass_heading_rad=np.pi / 2, dt=2.0)
        s2 = robot2.step(thrust=0.5, compass_heading_rad=np.pi / 2, dt=2.0)
        assert s2[0] == pytest.approx(2.0 * s1[0])


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------


class TestReset:
    """Unit tests for UnicycleKinematics.reset()."""

    def test_restores_state(self) -> None:
        """reset() returns robot to the specified initial condition."""
        robot = UnicycleKinematics(k_thrust=1.0)
        robot.step(thrust=1.0, compass_heading_rad=np.pi / 2, dt=10.0)
        robot.reset(x0=3.0, y0=4.0, heading0=np.pi)
        assert robot.x_m == pytest.approx(3.0)
        assert robot.y_m == pytest.approx(4.0)
        assert robot.heading_rad == pytest.approx(np.pi)


# ---------------------------------------------------------------------------
# trajectory()
# ---------------------------------------------------------------------------


class TestTrajectory:
    """Unit tests for UnicycleKinematics.trajectory()."""

    def test_shape_with_initial(self) -> None:
        """trajectory() returns (N+1, 3) when include_initial=True (default)."""
        robot = UnicycleKinematics(k_thrust=1.0)
        n = 10
        states = robot.trajectory(np.ones(n) * 0.5, np.zeros(n), dt=1.0)
        assert states.shape == (n + 1, 3)

    def test_shape_without_initial(self) -> None:
        """trajectory() returns (N, 3) when include_initial=False."""
        robot = UnicycleKinematics(k_thrust=1.0)
        n = 5
        states = robot.trajectory(np.ones(n), np.zeros(n), dt=1.0, include_initial=False)
        assert states.shape == (n, 3)

    def test_matches_repeated_step(self) -> None:
        """trajectory() and sequential step() produce identical state sequences."""
        thrusts = np.array([0.5, 0.8, 0.3])
        headings = np.array([0.0, np.pi / 2, np.pi])
        dt = 0.5

        robot_a = UnicycleKinematics(k_thrust=1.5, x0=1.0, y0=2.0)
        states_traj = robot_a.trajectory(thrusts, headings, dt=dt)

        robot_b = UnicycleKinematics(k_thrust=1.5, x0=1.0, y0=2.0)
        manual = [robot_b.state]
        for T, psi in zip(thrusts, headings, strict=True):
            manual.append(robot_b.step(T, psi, dt))

        np.testing.assert_allclose(states_traj, np.array(manual), atol=1e-12)

    def test_updates_internal_state(self) -> None:
        """After trajectory(), robot.state reflects the final integrated position."""
        robot = UnicycleKinematics(k_thrust=1.0, x0=0.0, y0=0.0)
        n = 5
        states = robot.trajectory(np.ones(n), np.full(n, np.pi / 2), dt=1.0)
        assert robot.x_m == pytest.approx(states[-1, 0])
        assert robot.y_m == pytest.approx(states[-1, 1])

    def test_mismatched_lengths_raises(self) -> None:
        """trajectory() raises ValueError when thrusts and headings differ in length."""
        robot = UnicycleKinematics(k_thrust=1.0)
        with pytest.raises(ValueError, match="same length"):
            robot.trajectory(np.ones(3), np.zeros(5), dt=1.0)


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------


class TestRepr:
    """Unit tests for UnicycleKinematics.__repr__()."""

    def test_contains_k_thrust(self) -> None:
        """__repr__ includes the k_thrust value."""
        robot = UnicycleKinematics(k_thrust=2.5)
        assert "2.5" in repr(robot)


# ---------------------------------------------------------------------------
# drive_to_waypoints()
# ---------------------------------------------------------------------------


class TestWaypointFollowing:
    """Unit tests for UnicycleKinematics.drive_to_waypoints()."""

    def test_empty_waypoints_raises(self) -> None:
        """Empty waypoints raises ValueError."""
        robot = UnicycleKinematics(k_thrust=1.0)
        with pytest.raises(ValueError):
            robot.drive_to_waypoints(np.empty((0, 2)), speed_mps=1.0, dt=1.0)

    def test_single_waypoint_reached(self) -> None:
        """Robot drives towards a single waypoint and stops when reached."""
        robot = UnicycleKinematics(k_thrust=2.0, x0=0.0, y0=0.0)
        # Waypoint is at (10, 0)
        waypoints = np.array([[10.0, 0.0]])
        # Driving at 2.0 m/s (thrust_cmd = 1.0) with dt=1.0 s, acceptance_radius=5.0 m.
        # Step 0: pos=(0,0). Heading is pointing East (π/2). Next pos is (2,0).
        # Step 1: pos=(2,0). Distance to (10,0) is 8.0 m > 5.0 m. Heading East. Next pos is (4,0).
        # Step 2: pos=(4,0). Distance to (10,0) is 6.0 m > 5.0 m. Heading East. Next pos is (6,0).
        # Step 3: pos=(6,0). Distance to (10,0) is 4.0 m < 5.0 m. Waypoint reached, loop ends.
        states, thrusts, headings = robot.drive_to_waypoints(
            waypoints, speed_mps=2.0, dt=1.0, acceptance_radius=5.0
        )
        assert len(states) == 4
        assert float(states[-1, 0]) == pytest.approx(6.0)
        assert float(states[-1, 1]) == pytest.approx(0.0)
        assert len(thrusts) == 3
        assert len(headings) == 3
        for h in headings:
            assert float(h) == pytest.approx(np.pi / 2.0)

    def test_invalid_waypoint_shape_raises(self) -> None:
        """Invalid waypoint arrays raise ValueError."""
        robot = UnicycleKinematics(k_thrust=1.0)
        with pytest.raises(ValueError):
            robot.drive_to_waypoints(np.array([1, 2, 3]), speed_mps=1.0, dt=1.0)

    def test_excessive_speed_raises(self) -> None:
        """If speed_mps / k_thrust > 1.0, raise ValueError."""
        robot = UnicycleKinematics(k_thrust=1.0)
        with pytest.raises(ValueError, match="speed_mps / k_thrust"):
            robot.drive_to_waypoints(np.array([[10.0, 10.0]]), speed_mps=2.0, dt=1.0)


# ---------------------------------------------------------------------------
# BaseKinematics behaviour
# ---------------------------------------------------------------------------


class TestBaseKinematics:
    """Unit tests for the BaseKinematics abstract class behavior."""

    def test_cannot_instantiate_base_class(self) -> None:
        """Attempting to instantiate BaseKinematics directly raises TypeError."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseKinematics()  # type: ignore[abstract]

    def test_unicycle_inherits_base_kinematics(self) -> None:
        """UnicycleKinematics correctly inherits from BaseKinematics."""
        robot = UnicycleKinematics(k_thrust=1.0)
        assert isinstance(robot, BaseKinematics)
