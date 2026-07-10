"""
kinematics.py
=============
Dead-reckoning kinematics for an aquatic vehicle.

This module owns **all** coordinate-frame bookkeeping:

* Forward/angular velocity integration via tunable kinematic constants
  ``c_v`` (thrust-to-speed) and ``c_omega`` (rudder-to-yaw-rate).
* Batched trajectory prediction expressed as a sequence of SE(2) poses.
* Differentiable *trajectory drift correction*: given a vector of additive
  pose latents ``deltax ∈ ℝ^{N×3}`` (δx, δy, δθ per time-step), the module
  returns corrected world-frame positions used downstream by the optimizer.
* WGS-84 **global-to-local** conversion: GPS fixes are projected onto a
  tangent-plane metric grid anchored at a configurable origin.

Coordinate conventions
-----------------------
* World frame  – right-handed East-North-Up (ENU), metres.
* Body frame   – x forward, y left, z up.
* Angles       – radians, measured CCW from East.

Dependencies: jax, jaxlib, pywmm (optional, for magnetic declination).
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

# ---------------------------------------------------------------------------
# Physical / kinematic constants
# ---------------------------------------------------------------------------

#: Thrust-coefficient – maps normalised thruster command [0, 1] → m s⁻¹.
C_V_DEFAULT: float = 0.85

#: Rudder-coefficient – maps normalised rudder command [-1, 1] → rad s⁻¹.
C_OMEGA_DEFAULT: float = 0.40

# WGS-84 ellipsoid parameters
_WGS84_A: float = 6_378_137.0          # semi-major axis [m]
_WGS84_E2: float = 6.694_379_990_14e-3  # first eccentricity squared


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


class KinematicParams(NamedTuple):
    """Learnable / configurable kinematic parameters.

    Attributes
    ----------
    c_v :
        Thrust-to-speed coefficient [m s⁻¹ / unit].
    c_omega :
        Rudder-to-yaw-rate coefficient [rad s⁻¹ / unit].
    """

    c_v: float = C_V_DEFAULT
    c_omega: float = C_OMEGA_DEFAULT


class GeoOrigin(NamedTuple):
    """Geodetic anchor point for the local ENU tangent plane.

    Attributes
    ----------
    lat_deg :
        Reference latitude in decimal degrees (positive = North).
    lon_deg :
        Reference longitude in decimal degrees (positive = East).
    alt_m :
        Reference altitude above WGS-84 ellipsoid [m].
    """

    lat_deg: float = 0.0
    lon_deg: float = 0.0
    alt_m: float = 0.0


# ---------------------------------------------------------------------------
# WGS-84 utilities
# ---------------------------------------------------------------------------


def wgs84_radii(lat_deg: float) -> tuple[float, float]:
    """Return the meridional (M) and normal (N) radii of curvature [m].

    Parameters
    ----------
    lat_deg :
        Geodetic latitude in decimal degrees.

    Returns
    -------
    M :
        Radius of curvature in the meridian plane [m].
    N :
        Radius of curvature in the prime vertical plane [m].
    """
    lat_rad = math.radians(lat_deg)
    sin_lat = math.sin(lat_rad)
    denom = math.sqrt(1.0 - _WGS84_E2 * sin_lat**2)
    N = _WGS84_A / denom
    M = _WGS84_A * (1.0 - _WGS84_E2) / denom**3
    return M, N


def latlon_to_enu(
    lat_deg: Array,
    lon_deg: Array,
    origin: GeoOrigin,
) -> tuple[Array, Array]:
    """Project geodetic coordinates onto the local ENU tangent plane.

    Uses a first-order (flat-earth) approximation valid within ~100 km of the
    origin, sufficient for nearshore / littoral SLAM applications.

    Parameters
    ----------
    lat_deg :
        Geodetic latitude of the query point(s) [decimal degrees].
    lon_deg :
        Geodetic longitude of the query point(s) [decimal degrees].
    origin :
        Anchor point defining the origin of the ENU frame.

    Returns
    -------
    east_m, north_m :
        ENU east and north displacements from the origin [m].
    """
    M, N = wgs84_radii(origin.lat_deg)
    lat_rad_origin = math.radians(origin.lat_deg)

    d_lat = jnp.radians(lat_deg - origin.lat_deg)
    d_lon = jnp.radians(lon_deg - origin.lon_deg)

    north_m = M * d_lat
    east_m = N * jnp.cos(lat_rad_origin) * d_lon
    return east_m, north_m


def enu_to_latlon(
    east_m: Array,
    north_m: Array,
    origin: GeoOrigin,
) -> tuple[Array, Array]:
    """Inverse of :func:`latlon_to_enu`.

    Parameters
    ----------
    east_m, north_m :
        ENU displacements from *origin* [m].
    origin :
        Anchor point defining the ENU frame origin.

    Returns
    -------
    lat_deg, lon_deg :
        Geodetic coordinates [decimal degrees].
    """
    M, N = wgs84_radii(origin.lat_deg)
    lat_rad_origin = math.radians(origin.lat_deg)

    d_lat = north_m / M
    d_lon = east_m / (N * jnp.cos(lat_rad_origin))

    lat_deg = origin.lat_deg + jnp.degrees(d_lat)
    lon_deg = origin.lon_deg + jnp.degrees(d_lon)
    return lat_deg, lon_deg


# ---------------------------------------------------------------------------
# Dead-reckoning integrator
# ---------------------------------------------------------------------------


def dead_reckon_step(
    pose: Array,
    control: Array,
    dt: float,
    params: KinematicParams,
) -> Array:
    """Integrate one time step of unicycle dead-reckoning kinematics.

    State update equations (discrete Euler):

    .. math::

        x_{t+1}   &= x_t + c_v \\, u_t \\cos(\\theta_t) \\, \\Delta t \\\\
        y_{t+1}   &= y_t + c_v \\, u_t \\sin(\\theta_t) \\, \\Delta t \\\\
        \\theta_{t+1} &= \\theta_t + c_\\omega \\, \\omega_t \\, \\Delta t

    Parameters
    ----------
    pose :
        Current SE(2) pose ``[x_m, y_m, theta_rad]``.
    control :
        Control input ``[u_thrust ∈ [0,1], u_rudder ∈ [-1,1]]``.
    dt :
        Integration step size [s].
    params :
        Kinematic coefficients.

    Returns
    -------
    next_pose :
        Updated SE(2) pose after one step.
    """
    x, y, theta = pose[0], pose[1], pose[2]
    u_thrust, u_rudder = control[0], control[1]

    speed = params.c_v * u_thrust
    x_new = x + speed * jnp.cos(theta) * dt
    y_new = y + speed * jnp.sin(theta) * dt
    theta_new = theta + params.c_omega * u_rudder * dt

    return jnp.stack([x_new, y_new, theta_new])


def integrate_trajectory(
    pose0: Array,
    controls: Array,
    dt: float,
    params: KinematicParams,
) -> Array:
    """Integrate the full dead-reckoning trajectory from an initial pose.

    Parameters
    ----------
    pose0 :
        Initial SE(2) pose ``[x0, y0, theta0]``.
    controls :
        Array of shape ``(N, 2)`` containing per-step control inputs.
    dt :
        Fixed integration step [s].
    params :
        Kinematic coefficients.

    Returns
    -------
    poses :
        Array of shape ``(N+1, 3)`` – trajectory including the initial pose.
    """

    def _step(carry: Array, ctrl: Array) -> tuple[Array, Array]:
        next_pose = dead_reckon_step(carry, ctrl, dt, params)
        return next_pose, next_pose

    _, poses_rest = jax.lax.scan(_step, pose0, controls)
    poses = jnp.concatenate([pose0[None], poses_rest], axis=0)
    return poses


# ---------------------------------------------------------------------------
# Drift-corrected trajectory
# ---------------------------------------------------------------------------


def apply_drift_correction(
    nominal_poses: Array,
    deltax: Array,
) -> Array:
    """Add additive drift latents to a nominal trajectory.

    In Phase 2 of the SLAM loop, ``deltax`` is a learnable parameter vector
    that absorbs slow kinematic model errors and unmodelled currents.

    Parameters
    ----------
    nominal_poses :
        Nominal SE(2) trajectory of shape ``(N+1, 3)``.
    deltax :
        Drift correction latents of shape ``(N+1, 3)``; should be
        initialised to zero and regularised in the optimizer loss.

    Returns
    -------
    corrected_poses :
        Drift-corrected trajectory of the same shape as *nominal_poses*.
    """
    return nominal_poses + deltax
