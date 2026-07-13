"""
kinematics.py
=============
Simple unicycle kinematic model for a heading-controlled, thrust-driven robot.

Heading convention
------------------
The compass heading follows the **navigation convention**:

* ``0 rad`` → North (positive ENU y-axis)
* ``π/2 rad`` → East  (positive ENU x-axis)
* Clockwise positive when viewed from above.

The kinematic equations are:

.. math::

    \\dot{x} = k_{\\text{thrust}} \\cdot T \\cdot \\sin(\\psi)

    \\dot{y} = k_{\\text{thrust}} \\cdot T \\cdot \\cos(\\psi)

where :math:`T \\in [0, 1]` is the (dimensionless) thrust command, :math:`\\psi`
is the compass heading [rad], and :math:`k_{\\text{thrust}}` [m s⁻¹ per unit thrust]
is the conversion factor intended for future JAX-autodiff optimisation.

Position is integrated with Euler-forward; the heading is read directly from
the compass at each step (no rate-of-turn integration).

Typical usage
-------------
::

    from pde_slam.kinematics import UnicycleKinematics
    import numpy as np

    robot = UnicycleKinematics(k_thrust=1.5, x0=-240.0, y0=-240.0)

    # single step: heading east (π/2 rad), full thrust, dt=1 s
    state = robot.step(thrust=1.0, compass_heading_rad=np.pi / 2, dt=1.0)

    # batch trajectory
    thrusts  = np.ones(100) * 0.8
    headings = np.full(100, np.pi / 2)
    states   = robot.trajectory(thrusts, headings, dt=1.0)   # (101, 3)
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


class UnicycleKinematics:
    """Kinematic model for a compass- and thrust-controlled surface robot.

    The heading is **read directly from the compass** at each step rather than
    integrated from a turn-rate command.  Only the (x, y) position is
    numerically integrated (Euler-forward).

    Heading convention: **navigation** (0 = North, π/2 = East, clockwise
    positive when viewed from above).  The corresponding ENU velocity is::

        v_east  = k_thrust * thrust * sin(heading_rad)
        v_north = k_thrust * thrust * cos(heading_rad)

    The ``k_thrust`` parameter is intentionally kept as a plain Python
    ``float`` here so that the class works without JAX.  When the parameter
    is to be optimised, the caller can wrap the integration in a JAX-traced
    function and pass ``k_thrust`` as a JAX scalar.

    Parameters
    ----------
    k_thrust :
        Thrust-to-speed conversion factor [m s⁻¹ per unit thrust].  A value
        of ``1.0`` means a thrust command of ``0.5`` produces ``0.5 m s⁻¹``.
        This is the optimisable parameter.
    x0 :
        Initial ENU east position [m].
    y0 :
        Initial ENU north position [m].
    heading0 :
        Initial compass heading [rad, navigation convention].
    """

    def __init__(
        self,
        k_thrust: float = 1.0,
        x0: float = 0.0,
        y0: float = 0.0,
        heading0: float = 0.0,
    ) -> None:
        self.k_thrust: float = float(k_thrust)
        self._state: Array = jnp.array(
            [float(x0), float(y0), float(heading0)], dtype=jnp.float64
        )

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    @property
    def x_m(self) -> float:
        """Current ENU east position [m]."""
        return float(self._state[0])

    @property
    def y_m(self) -> float:
        """Current ENU north position [m]."""
        return float(self._state[1])

    @property
    def heading_rad(self) -> float:
        """Current compass heading [rad, navigation convention]."""
        return float(self._state[2])

    @property
    def state(self) -> Array:
        """Current state vector ``[x_m, y_m, heading_rad]``, shape ``(3,)``."""
        return self._state.copy()

    # ------------------------------------------------------------------
    # Integration
    # ------------------------------------------------------------------

    def step(
        self,
        thrust: float,
        compass_heading_rad: float,
        dt: float,
    ) -> Array:
        """Integrate one time step using Euler-forward integration.

        Parameters
        ----------
        thrust :
            Dimensionless thrust command in ``[0, 1]``.
        compass_heading_rad :
            Compass heading reading [rad, navigation convention:
            0 = North, π/2 = East, clockwise positive].
        dt :
            Time step [s].

        Returns
        -------
        state :
            Updated state ``[x_m, y_m, heading_rad]``, shape ``(3,)``.
        """
        speed = self.k_thrust * float(thrust)
        psi = float(compass_heading_rad)
        dx = speed * jnp.sin(psi) * float(dt)
        dy = speed * jnp.cos(psi) * float(dt)
        self._state = self._state.at[0].add(dx).at[1].add(dy).at[2].set(psi)
        return self._state.copy()

    def reset(
        self,
        x0: float = 0.0,
        y0: float = 0.0,
        heading0: float = 0.0,
    ) -> None:
        """Reset the robot state to a new initial condition.

        Parameters
        ----------
        x0 :
            Initial ENU east position [m].
        y0 :
            Initial ENU north position [m].
        heading0 :
            Initial compass heading [rad, navigation convention].
        """
        self._state = jnp.array([float(x0), float(y0), float(heading0)], dtype=jnp.float64)

    def trajectory(
        self,
        thrusts: Array,
        headings: Array,
        dt: float,
        *,
        include_initial: bool = True,
    ) -> Array:
        """Integrate a full sequence of thrust + compass commands.

        Parameters
        ----------
        thrusts :
            1-D array of thrust commands, length N.
        headings :
            1-D array of compass headings [rad, navigation convention], length N.
        dt :
            Fixed time step [s] applied at each command.
        include_initial :
            If ``True`` (default), the state *before* the first command is
            prepended, giving shape ``(N + 1, 3)``.  Set to ``False`` to
            obtain shape ``(N, 3)``.

        Returns
        -------
        states :
            Array of shape ``(N + 1, 3)`` or ``(N, 3)`` with columns
            ``[x_m, y_m, heading_rad]``.

        Raises
        ------
        ValueError
            If *thrusts* and *headings* have different lengths.
        """
        thrusts = jnp.asarray(thrusts, dtype=jnp.float64).ravel()
        headings = jnp.asarray(headings, dtype=jnp.float64).ravel()
        if thrusts.shape != headings.shape:
            raise ValueError(
                "thrusts and headings must have the same length; "
                f"got {len(thrusts)} vs {len(headings)}"
            )

        n = len(thrusts)
        offset = 1 if include_initial else 0
        states = jnp.empty((n + offset, 3), dtype=jnp.float64)

        if include_initial:
            states = states.at[0].set(self._state)

        # Vectorised position update: compute speeds and displacements in bulk,
        # then accumulate with cumsum so we avoid a Python loop over elements.
        speeds = self.k_thrust * thrusts
        dx = speeds * jnp.sin(headings) * dt
        dy = speeds * jnp.cos(headings) * dt

        xs = self._state[0] + jnp.cumsum(dx)
        ys = self._state[1] + jnp.cumsum(dy)

        states = states.at[offset:, 0].set(xs)
        states = states.at[offset:, 1].set(ys)
        states = states.at[offset:, 2].set(headings)

        # Update internal state to end of trajectory
        self._state = jnp.array([float(xs[-1]), float(ys[-1]), float(headings[-1])],
                                dtype=jnp.float64)

        return states

    def drive_to_waypoints(
        self,
        waypoints: Array,
        speed_mps: float,
        dt: float,
        *,
        acceptance_radius: float = 5.0,
    ) -> tuple[Array, Array, Array]:
        """Drive the robot through a sequence of waypoints.

        At each time step, the robot steers directly toward the active waypoint,
        applies constant thrust, and integrates its position. Once within
        `acceptance_radius` of the active waypoint, it switches to the next.

        Parameters
        ----------
        waypoints :
            Array of shape ``(M, 2)`` representing target ``[east_m, north_m]`` waypoints.
        speed_mps :
            Desired robot speed [m/s].
        dt :
            Simulation time step [s].
        acceptance_radius :
            Distance threshold [m] to switch to the next waypoint.

        Returns
        -------
        states :
            Integrated states ``[x_m, y_m, heading_rad]`` at each step (including initial),
            shape ``(N + 1, 3)``.
        thrusts :
            Thrust commands applied at each step, shape ``(N,)``.
        headings :
            Compass headings applied at each step [rad], shape ``(N,)``.
        """
        waypoints = jnp.asarray(waypoints, dtype=jnp.float64)
        if waypoints.ndim != 2 or waypoints.shape[1] != 2:
            raise ValueError(f"waypoints must be of shape (M, 2); got {waypoints.shape}")
        if len(waypoints) == 0:
            raise ValueError("waypoints must not be empty")

        thrust_cmd = speed_mps / self.k_thrust
        if thrust_cmd > 1.0:
            raise ValueError(
                f"speed_mps / k_thrust = {thrust_cmd:.3f} > 1.0 — "
                "increase k_thrust or reduce speed_mps."
            )

        states_list = [self._state.copy()]
        thrusts_list = []
        headings_list = []

        wp_idx = 0
        num_wps = len(waypoints)

        # Python control flow loop (not over array elements) to handle waypoint sequence switching
        while wp_idx < num_wps:
            curr_pos = self._state[:2]
            target_pos = waypoints[wp_idx]

            dx = target_pos[0] - curr_pos[0]
            dy = target_pos[1] - curr_pos[1]
            dist = jnp.hypot(dx, dy)

            # Check if waypoint is reached
            if dist < acceptance_radius:
                wp_idx += 1
                if wp_idx >= num_wps:
                    break
                # Re-calculate with the new waypoint
                target_pos = waypoints[wp_idx]
                dx = target_pos[0] - curr_pos[0]
                dy = target_pos[1] - curr_pos[1]
                dist = jnp.hypot(dx, dy)

            # Steer directly to waypoint (navigation convention: 0 = North, π/2 = East)
            heading = jnp.arctan2(dx, dy)

            # Record commands
            thrusts_list.append(thrust_cmd)
            headings_list.append(heading)

            # Integrate one step
            self.step(thrust_cmd, heading, dt)
            states_list.append(self._state.copy())

        return (
            jnp.stack(states_list),
            jnp.array(thrusts_list),
            jnp.array(headings_list),
        )

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"UnicycleKinematics(k_thrust={self.k_thrust}, "
            f"x={self.x_m:.3f} m, y={self.y_m:.3f} m, "
            f"heading={float(jnp.degrees(self.heading_rad)):.1f}°)"
        )

