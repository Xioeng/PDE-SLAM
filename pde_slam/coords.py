"""
coords.py
=========
Geodetic ↔ local East-North-Up (ENU) coordinate frame conversions.

This module provides :class:`ENUFrame`, a lightweight class that anchors a
local Cartesian (east, north) reference frame to an explicit geodetic origin
``(lat0, lon0)``.  All conversions use an equirectangular (flat-earth)
projection, which is accurate to better than 0.1 m over survey domains up to
~10 km across.

The origin **must** be supplied explicitly by the caller so that downstream
components (grid layout, multi-session comparisons) are always unambiguous.

Usage
-----
::

    from pde_slam.coords import ENUFrame

    frame = ENUFrame(lat0=36.7996, lon0=-76.0000)

    # geodetic → ENU
    east_m, north_m = frame.to_enu(lat_array, lon_array)

    # ENU → geodetic
    lat, lon = frame.from_enu(east_m, north_m)

    # Convenience: stacked (N, 2) array ready for FieldInterpolator
    xy = frame.to_enu_xy(lat_array, lon_array)
"""

from __future__ import annotations

import numpy as np
from jax import Array

# WGS-84 semi-major axis [m]
_R_EARTH: float = 6_378_137.0

# Degrees → radians conversion factor
_DEG2RAD: float = np.pi / 180.0


class ENUFrame:
    """Local East-North-Up Cartesian frame anchored to an explicit geodetic origin.

    All coordinate conversions use an equirectangular (flat-earth) projection:

    .. math::

        e = (\\lambda - \\lambda_0) \\cos(\\phi_0) \\, R \\, \\frac{\\pi}{180}

        n = (\\phi - \\phi_0) \\, R \\, \\frac{\\pi}{180}

    where :math:`\\phi_0, \\lambda_0` are the origin latitude and longitude in
    degrees and :math:`R` is the WGS-84 semi-major axis (6 378 137 m).

    This approximation is accurate to < 0.1 m over domains up to roughly 10 km.

    Parameters
    ----------
    lat0 :
        Geodetic latitude of the ENU origin [degrees, WGS-84].
    lon0 :
        Geodetic longitude of the ENU origin [degrees, WGS-84].
    """

    def __init__(self, lat0: float, lon0: float) -> None:
        self.lat0: float = float(lat0)
        self.lon0: float = float(lon0)

        # Pre-compute the scale factors (constant for a fixed origin)
        lat0_rad = self.lat0 * _DEG2RAD
        self._m_per_deg_north: float = float(_R_EARTH * _DEG2RAD)
        self._m_per_deg_east: float = float(_R_EARTH * _DEG2RAD * np.cos(lat0_rad))

    # ------------------------------------------------------------------
    # Forward: geodetic → ENU
    # ------------------------------------------------------------------

    def to_enu(
        self, lat: np.ndarray | Array, lon: np.ndarray | Array
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert geodetic coordinates to local ENU metres.

        Parameters
        ----------
        lat :
            Geodetic latitude(s) [degrees]. Scalar or array of any shape.
        lon :
            Geodetic longitude(s) [degrees]. Same shape as *lat*.

        Returns
        -------
        east_m :
            Easting(s) relative to the frame origin [m]. Same shape as *lat*.
        north_m :
            Northing(s) relative to the frame origin [m]. Same shape as *lat*.
        """
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        east_m = (lon - self.lon0) * self._m_per_deg_east
        north_m = (lat - self.lat0) * self._m_per_deg_north
        return east_m, north_m

    def to_enu_xy(
        self, lat: np.ndarray | Array, lon: np.ndarray | Array
    ) -> np.ndarray:
        """Convert geodetic coordinates to a stacked ``(N, 2)`` ENU array.

        Convenience wrapper around :meth:`to_enu` that produces the column
        layout expected by :class:`~pde_slam.interpolator.FieldInterpolator`.

        Parameters
        ----------
        lat :
            Geodetic latitude(s) [degrees]. 1-D array of length N.
        lon :
            Geodetic longitude(s) [degrees]. 1-D array of length N.

        Returns
        -------
        xy :
            Array of shape ``(N, 2)`` with columns ``[east_m, north_m]``.
        """
        east_m, north_m = self.to_enu(lat, lon)
        return np.stack([east_m, north_m], axis=-1)

    # ------------------------------------------------------------------
    # Inverse: ENU → geodetic
    # ------------------------------------------------------------------

    def from_enu(
        self, east_m: np.ndarray | Array, north_m: np.ndarray | Array
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert local ENU metres back to geodetic coordinates.

        Parameters
        ----------
        east_m :
            Easting(s) relative to the frame origin [m]. Scalar or array.
        north_m :
            Northing(s) relative to the frame origin [m]. Same shape as *east_m*.

        Returns
        -------
        lat :
            Geodetic latitude(s) [degrees]. Same shape as *east_m*.
        lon :
            Geodetic longitude(s) [degrees]. Same shape as *east_m*.
        """
        east_m = np.asarray(east_m, dtype=np.float64)
        north_m = np.asarray(north_m, dtype=np.float64)
        lat = self.lat0 + north_m / self._m_per_deg_north
        lon = self.lon0 + east_m / self._m_per_deg_east
        return lat, lon

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"ENUFrame(lat0={self.lat0}, lon0={self.lon0})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ENUFrame):
            return NotImplemented
        return self.lat0 == other.lat0 and self.lon0 == other.lon0
