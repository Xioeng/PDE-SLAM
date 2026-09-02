"""
survey.py
=========
Ingestion and kinematic interpolation for field survey datasets.
Loads CSV records containing geodetic coordinates (Latitude, Longitude),
timestamps, and in-situ water quality sensor observations, projecting them into
local ENU coordinates and interpolating them into kinematically compliant
differential drive trajectories capped to a specified duration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import Array

from pde_slam.coords import ENUFrame
from pde_slam.kinematics.diff_drive import DiffDriveKinematics


@dataclass
class SurveyTrajectory:
    """Container for kinematically interpolated survey trajectory and measurements.

    Parameters
    ----------
    timestamps : np.ndarray
        Regularly spaced simulation timestamps [s], shape (N,).
    coords_enu : np.ndarray
        Local ENU coordinates [m], shape (N, 2).
    headings : np.ndarray
        Vehicle heading angles [rad], shape (N,).
    velocities : np.ndarray
        Linear forward velocities [m/s], shape (N,).
    omegas : np.ndarray
        Angular yaw rates [rad/s], shape (N,).
    measurements : dict[str, np.ndarray]
        Sensor field measurements interpolated onto timestamps.
    enu_frame : ENUFrame
        Coordinate frame used for geodetic to ENU projection.
    raw_timestamps : np.ndarray
        Raw cleaned timestamps [s] before interpolation, shape (M,).
    raw_coords_enu : np.ndarray
        Raw cleaned ENU coordinates [m], shape (M, 2).
    raw_measurements : dict[str, np.ndarray]
        Raw sensor field measurements, shape (M,).
    """

    timestamps: np.ndarray
    coords_enu: np.ndarray
    headings: np.ndarray
    velocities: np.ndarray
    omegas: np.ndarray
    measurements: dict[str, np.ndarray] = field(default_factory=dict)
    enu_frame: ENUFrame | None = None
    raw_timestamps: np.ndarray = field(default_factory=lambda: np.empty(0))
    raw_coords_enu: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    raw_measurements: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n_steps(self) -> int:
        """Number of motion steps (N - 1)."""
        return max(0, len(self.timestamps) - 1)

    @property
    def dt(self) -> float:
        """Time step between consecutive trajectory points [s]."""
        if len(self.timestamps) < 2:
            return 1.0
        return float(self.timestamps[1] - self.timestamps[0])

    @property
    def duration(self) -> float:
        """Total duration of the trajectory [s]."""
        if len(self.timestamps) == 0:
            return 0.0
        return float(self.timestamps[-1] - self.timestamps[0])


def _clean_survey_dataframe(
    df: pd.DataFrame,
    lat_col: str = "Latitude",
    lon_col: str = "Longitude",
    date_col: str = "Date",
    time_col: str = "Time",
) -> pd.DataFrame:
    """Filter invalid GPS fixes, null island coordinates, and parse timestamps.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input dataframe.
    lat_col : str, default='Latitude'
        Column name for latitude.
    lon_col : str, default='Longitude'
        Column name for longitude.
    date_col : str, default='Date'
        Column name for date.
    time_col : str, default='Time'
        Column name for time.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe sorted by timestamp with non-zero coordinates.
    """
    valid = df.copy()

    # Drop null coordinates or Null Island glitches (0.0, 0.0)
    valid = valid[valid[lat_col].notna() & valid[lon_col].notna()]
    valid = valid[(valid[lat_col].abs() > 1e-4) & (valid[lon_col].abs() > 1e-4)]

    # Parse date and time to seconds
    times_str = valid[date_col].astype(str) + valid[time_col].astype(str).str.zfill(6)
    dt_series = pd.to_datetime(times_str, format="%Y%m%d%H%M%S", errors="coerce")
    valid = valid[dt_series.notna()].copy()
    dt_series = dt_series[dt_series.notna()]

    elapsed = (dt_series - dt_series.iloc[0]).dt.total_seconds().to_numpy()
    valid["_t_sec"] = elapsed

    # Deduplicate timestamps keeping the first record
    _, unique_idx = np.unique(elapsed, return_index=True)
    valid = valid.iloc[unique_idx].sort_values("_t_sec").reset_index(drop=True)
    return valid


def interpolate_kinematic_trajectory(
    t_raw: np.ndarray,
    xy_raw: np.ndarray,
    t_max: float = 500.0,
    dt: float = 1.0,
    compass_headings: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate a 2D trajectory onto a uniform grid using diff-drive kinematics.


    Parameters
    ----------
    t_raw : np.ndarray
        Raw timestamps [s], strictly monotonic, shape (M,).
    xy_raw : np.ndarray
        Raw 2D positions [m], shape (M, 2).
    t_max : float, default=500.0
        Maximum duration to cap the trajectory [s].
    dt : float, default=1.0
        Sampling interval for uniform grid [s].
    compass_headings : np.ndarray or None, optional
        Optional raw compass headings [rad] for stationary periods.

    Returns
    -------
    t_grid : np.ndarray
        Uniform timestamps [s], shape (N,).
    coords : np.ndarray
        Kinematically integrated positions [m], shape (N, 2).
    headings : np.ndarray
        Vehicle headings [rad], shape (N,).
    velocities : np.ndarray
        Forward velocities [m/s], shape (N,).
    omegas : np.ndarray
        Angular velocities [rad/s], shape (N,).
    """
    t_grid = np.arange(0.0, float(t_max) + float(dt) * 0.5, float(dt))
    n_pts = len(t_grid)

    # 1. Resample positions onto uniform time grid
    x_interp = np.interp(t_grid, t_raw, xy_raw[:, 0])
    y_interp = np.interp(t_grid, t_raw, xy_raw[:, 1])
    coords_interp = np.column_stack([x_interp, y_interp])

    # 2. Derive segment kinematics
    dx = np.diff(x_interp)
    dy = np.diff(y_interp)
    dists = np.hypot(dx, dy)
    v_segments = dists / dt

    headings_seg = np.zeros(n_pts - 1, dtype=np.float64)
    last_heading = 0.0
    if compass_headings is not None and len(compass_headings) > 0:
        last_heading = float(compass_headings[0])

    for i in range(n_pts - 1):
        if dists[i] > 1e-3:
            headings_seg[i] = np.arctan2(dy[i], dx[i])
            last_heading = headings_seg[i]
        else:
            headings_seg[i] = last_heading

    # Align initial heading
    init_heading = headings_seg[0] if n_pts > 1 else 0.0
    headings = np.concatenate([[init_heading], headings_seg])

    # 3. Derive angular velocity (omega) with angular unwrap
    dtheta = np.diff(np.unwrap(headings))
    omegas_seg = dtheta / dt
    omegas = np.concatenate([omegas_seg, [0.0]])
    velocities = np.concatenate([v_segments, [0.0]])

    # 4. Integrate trajectory strictly with DiffDriveKinematics
    x0 = jnp.array([coords_interp[0, 0], coords_interp[0, 1], init_heading])
    v_jax = jnp.asarray(v_segments, dtype=jnp.float64)
    w_jax = jnp.asarray(omegas_seg, dtype=jnp.float64)

    integrated_states: Array = DiffDriveKinematics.integrate_trajectory(
        x0=x0,
        velocities=v_jax,
        omegas=w_jax,
        dt=float(dt),
        include_initial=True,
    )
    coords_kinematic = np.array(integrated_states[:, :2])
    headings_kinematic = np.array(integrated_states[:, 2])

    return t_grid, coords_kinematic, headings_kinematic, velocities, omegas


def load_survey_csv(
    csv_path: str | Path,
    t_max: float = 500.0,
    dt: float = 1.0,
    enu_frame: ENUFrame | None = None,
    field_mappings: dict[str, str] | None = None,
) -> SurveyTrajectory:
    """Load, clean, project, and interpolate a field survey CSV trajectory.

    Parameters
    ----------
    csv_path : str or Path
        Path to CSV file containing survey data.
    t_max : float, default=500.0
        Maximum trajectory time cap [s].
    dt : float, default=1.0
        Resampling time step [s].
    enu_frame : ENUFrame or None, optional
        Reference coordinate frame. If None, anchored to the first valid GPS fix.
    field_mappings : dict of str to str, optional
        Mapping from standardized field name to CSV column name.
        Defaults to standard aquatic sensor fields (Salinity, Temperature, ODO).

    Returns
    -------
    SurveyTrajectory
        Container with kinematically compliant trajectory and sensor measurements.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Survey CSV file not found: {path}")

    raw_df = pd.read_csv(path)
    clean_df = _clean_survey_dataframe(raw_df)

    if len(clean_df) == 0:
        raise ValueError(f"No valid GPS records found in {path}")

    # Cap to t_max
    clean_df = clean_df[clean_df["_t_sec"] <= float(t_max)].copy()
    if len(clean_df) < 2:
        raise ValueError(f"Fewer than 2 survey points within t_max={t_max}s in {path}")

    t_raw = clean_df["_t_sec"].to_numpy()
    lats = clean_df["Latitude"].to_numpy()
    lons = clean_df["Longitude"].to_numpy()

    # Determine ENU frame
    if enu_frame is None:
        enu_frame = ENUFrame(lat0=float(lats[0]), lon0=float(lons[0]))

    east_raw, north_raw = enu_frame.to_enu(lats, lons)
    xy_raw = np.column_stack([east_raw, north_raw])

    # Optional compass headings from CSV
    compass_rad = None
    for h_col in ["Heading (degrees Magnetic)", "Heading", "heading"]:
        if h_col in clean_df.columns:
            compass_rad = np.radians(clean_df[h_col].to_numpy())
            break

    # Kinematic interpolation
    t_grid, coords, headings, velocities, omegas = interpolate_kinematic_trajectory(
        t_raw=t_raw,
        xy_raw=xy_raw,
        t_max=t_max,
        dt=dt,
        compass_headings=compass_rad,
    )

    # Standard sensor field mapping
    default_mappings = {
        "salinity": "Salinity (PPT)",
        "temperature": "Temperature (C)",
        "odo": "ODO (mg/L)",
    }
    if field_mappings is not None:
        default_mappings.update(field_mappings)

    raw_measurements: dict[str, np.ndarray] = {}
    interp_measurements: dict[str, np.ndarray] = {}

    for field_name, col_name in default_mappings.items():
        matched_col: str | None = None
        for c in clean_df.columns:
            if c.strip().lower() == col_name.strip().lower() or (
                field_name.lower() in c.lower()
            ):
                matched_col = c
                break

        if matched_col and matched_col in clean_df.columns:
            vals = pd.to_numeric(clean_df[matched_col], errors="coerce").to_numpy()
            mean_val = float(np.nanmean(vals) if not np.all(np.isnan(vals)) else 0.0)
            vals = np.nan_to_num(vals, nan=mean_val)
            raw_measurements[field_name] = vals
            interp_measurements[field_name] = np.interp(t_grid, t_raw, vals)

    return SurveyTrajectory(
        timestamps=t_grid,
        coords_enu=coords,
        headings=headings,
        velocities=velocities,
        omegas=omegas,
        measurements=interp_measurements,
        enu_frame=enu_frame,
        raw_timestamps=t_raw,
        raw_coords_enu=xy_raw,
        raw_measurements=raw_measurements,
    )
