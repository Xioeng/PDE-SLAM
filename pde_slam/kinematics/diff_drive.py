"""
diff_drive.py
=============
Differential drive kinematic model.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from pde_slam.kinematics.base import BaseKinematics


class DiffDriveKinematics(BaseKinematics):
    """Kinematic model for a differential drive robot.

    Parameters
    ----------
    x0 :
        Initial x position [m].
    y0 :
        Initial y position [m].
    heading0 :
        Initial heading [rad].
    """

    def __init__(
        self,
        x0: float = 0.0,
        y0: float = 0.0,
        heading0: float = 0.0,
    ) -> None:
        super().__init__()
        self._state: Array = jnp.array(
            [float(x0), float(y0), float(heading0)], dtype=jnp.float64
        )

    @property
    def x_m(self) -> float:
        """Current x position [m]."""
        return float(self._state[0])

    @property
    def y_m(self) -> float:
        """Current y position [m]."""
        return float(self._state[1])

    @property
    def heading_rad(self) -> float:
        """Current heading [rad]."""
        return float(self._state[2])

    @property
    def state(self) -> Array:
        """Current state vector ``[x_m, y_m, heading_rad]``."""
        return self._state.copy()

    def step(self, v: float, omega: float, dt: float) -> Array:
        """Integrate one time step.

        Parameters
        ----------
        v :
            Linear velocity [m/s].
        omega :
            Angular velocity [rad/s].
        dt :
            Time step [s].

        Returns
        -------
        state :
            Updated state ``[x_m, y_m, heading_rad]``.
        """
        psi = self.heading_rad
        dx = float(v) * jnp.cos(psi) * float(dt)
        dy = float(v) * jnp.sin(psi) * float(dt)
        dpsi = float(omega) * float(dt)
        self._state = self._state.at[0].add(dx).at[1].add(dy).at[2].add(dpsi)
        return self._state.copy()

    def reset(
        self,
        x0: float = 0.0,
        y0: float = 0.0,
        heading0: float = 0.0,
    ) -> None:
        """Reset the robot state."""
        self._state = jnp.array(
            [float(x0), float(y0), float(heading0)], dtype=jnp.float64
        )

    @staticmethod
    def _motion_model(x: Array, u: Array, dt: float | Array) -> Array:
        """Pure differential drive motion model for a single state vector.

        Parameters
        ----------
        x : Array
            State vector ``[x, y, heading]``.
        u : Array
            Control vector ``[v, omega]``.
        dt : float or Array
            Integration time step [s].

        Returns
        -------
        Array
            Updated state vector ``[x', y', heading']``.
        """
        v, omega = u[0], u[1]
        theta = x[2]
        dx = v * jnp.cos(theta) * dt
        dy = v * jnp.sin(theta) * dt
        dtheta = omega * dt
        return jnp.array([x[0] + dx, x[1] + dy, x[2] + dtheta], dtype=jnp.float64)

    @staticmethod
    def integrate_trajectory(
        x0: Array,
        velocities: Array,
        omegas: Array,
        dt: float | Array,
        *,
        include_initial: bool = True,
    ) -> Array:
        """Differentiable integration of a diff drive trajectory.

        Parameters
        ----------
        x0 : Array
            Initial state vector, shape (2,) or (3,).
        velocities : Array
            Linear velocities [m/s].
        omegas : Array
            Angular velocities [rad/s].
        dt : float | Array
            Time step [s].
        include_initial : bool
            Whether to prepend the initial state.

        Returns
        -------
        states : Array
            Integrated states.
        """
        velocities_arr = jnp.asarray(velocities, dtype=jnp.float64).ravel()
        omegas_arr = jnp.asarray(omegas, dtype=jnp.float64).ravel()

        n = len(velocities_arr)
        has_heading = x0.shape[0] == 3
        num_cols = 3 if has_heading else 2
        offset = 1 if include_initial else 0

        states = jnp.empty((n + offset, num_cols), dtype=jnp.float64)

        if include_initial:
            states = states.at[0].set(x0)

        initial_heading = x0[2] if has_heading else 0.0

        # Start headings for Euler integration
        start_headings = initial_heading + jnp.concatenate(
            [jnp.array([0.0]), jnp.cumsum(omegas_arr[:-1] * dt)]
        )

        dx = velocities_arr * jnp.cos(start_headings) * dt
        dy = velocities_arr * jnp.sin(start_headings) * dt

        xs = x0[0] + jnp.cumsum(dx)
        ys = x0[1] + jnp.cumsum(dy)

        states = states.at[offset:, 0].set(xs)
        states = states.at[offset:, 1].set(ys)

        if has_heading:
            headings = initial_heading + jnp.cumsum(omegas_arr * dt)
            states = states.at[offset:, 2].set(headings)

        return states

    def trajectory(
        self,
        velocities: Array,
        omegas: Array,
        dt: float,
        *,
        include_initial: bool = True,
    ) -> Array:
        """Integrate a full sequence of commands.

        Parameters
        ----------
        velocities :
            Array of linear velocities.
        omegas :
            Array of angular velocities.
        dt :
            Time step.
        include_initial :
            Whether to prepend the state before the first command.

        Returns
        -------
        states :
            Array of integrated states.
        """
        velocities_arr = jnp.asarray(velocities, dtype=jnp.float64).ravel()
        omegas_arr = jnp.asarray(omegas, dtype=jnp.float64).ravel()
        if velocities_arr.shape != omegas_arr.shape:
            raise ValueError("velocities and omegas must have the same length")

        states = self.integrate_trajectory(
            self._state,
            velocities_arr,
            omegas_arr,
            dt,
            include_initial=include_initial,
        )

        self._state = states[-1]
        return states

    def drive_to_waypoints(
        self,
        waypoints: Array,
        speed_mps: float,
        dt: float,
        *,
        acceptance_radius: float = 1.0,
    ) -> tuple[Array, Array, Array]:
        """Drive the robot through a sequence of waypoints.

        Parameters
        ----------
        waypoints :
            Array of shape (M, 2) representing target waypoints.
        speed_mps :
            Desired robot speed [m/s].
        dt :
            Simulation time step [s].
        acceptance_radius :
            Distance threshold [m] to switch to the next waypoint.

        Returns
        -------
        states :
            Integrated states at each step.
        velocities :
            Linear velocities applied at each step.
        omegas :
            Angular velocities applied at each step.
        """
        waypoints = jnp.asarray(waypoints, dtype=jnp.float64)
        if waypoints.ndim != 2 or waypoints.shape[1] != 2:
            raise ValueError(
                f"waypoints must be of shape (M, 2); got {waypoints.shape}"
            )
        if len(waypoints) == 0:
            raise ValueError("waypoints must not be empty")

        states_list = [self._state.copy()]
        velocities_list = []
        omegas_list = []

        wp_idx = 0
        num_wps = len(waypoints)
        K_p = 0.5
        K_v = 1.0
        print(f"Starting drive to waypoints: {num_wps} waypoints {dt}")
        while wp_idx < num_wps:
            curr_pos = self._state[:2]
            target_pos = waypoints[wp_idx]

            dx = target_pos[0] - curr_pos[0]
            dy = target_pos[1] - curr_pos[1]
            dist = jnp.hypot(dx, dy)

            if dist < acceptance_radius:
                wp_idx += 1
                if wp_idx >= num_wps:
                    break
                target_pos = waypoints[wp_idx]
                dx = target_pos[0] - curr_pos[0]
                dy = target_pos[1] - curr_pos[1]
                dist = jnp.hypot(dx, dy)

            target_heading = jnp.arctan2(dy, dx)
            angle_error = target_heading - self.heading_rad
            angle_error = (angle_error + jnp.pi) % (2 * jnp.pi) - jnp.pi

            v = min(float(speed_mps), K_v * float(dist))
            omega = K_p * float(angle_error)

            velocities_list.append(v)
            omegas_list.append(omega)

            self.step(v, omega, dt)
            states_list.append(self._state.copy())

        return (
            jnp.stack(states_list),
            jnp.array(velocities_list),
            jnp.array(omegas_list),
        )

    def __repr__(self) -> str:
        return (
            f"DiffDriveKinematics("
            f"x={self.x_m:.3f} m, y={self.y_m:.3f} m, "
            f"heading={float(jnp.degrees(self.heading_rad)):.1f}°)"
        )
