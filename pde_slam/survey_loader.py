"""
survey_loader.py
================
Load a survey CSV, project observations to local ENU metres, and prepare
(xy, values) arrays ready for :class:`~pde_slam.interpolator.FieldInterpolator`.

The loader is intentionally thin: it handles I/O and unit conversion only.
Interpolation and solving remain the responsibility of their own classes so
that each step stays independently testable and composable.

The loader supports two coordinate modes:

* **Lat/lon mode** (default): CSV has ``lat_deg`` / ``lon_deg`` columns; the
  loader projects them to ENU metres via :class:`~pde_slam.coords.ENUFrame`.
* **Pre-projected mode**: CSV already contains ``x_m`` / ``y_m`` columns in
  local ENU metres (e.g. from ``scripts/generate_synthetic_survey.py``). In
  this case the ``frame`` is still accepted for metadata but the projection
  step is skipped.

Typical usage — lat/lon CSV
---------------------------
::

    from pde_slam.coords import ENUFrame
    from pde_slam.interpolator import FieldInterpolator, SpatialGrid
    from pde_slam.survey_loader import SurveyLoader

    frame  = ENUFrame(lat0=36.7996, lon0=-76.0000)
    loader = SurveyLoader(frame, field="salinity_psu")
    loader.load("data/raw/survey.csv")

    # Access prepared arrays
    xy     = loader.xy_obs      # (N, 2) float64 numpy – east_m, north_m
    values = loader.values      # (N,)   float64 numpy

Typical usage — pre-projected CSV (x_m / y_m)
---------------------------------------------
::

    loader = SurveyLoader(frame, field="salinity_psu",
                          x_col="x_m", y_col="y_m")
    loader.load("data/raw/survey.csv")

    # One-shot: load + interpolate → JAX array (ny, nx)
    grid = SpatialGrid(x_min=-500, x_max=500, y_min=-500, y_max=500, nx=64, ny=64)
    phi0 = loader.to_initial_condition(grid, method="rbf")
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from jax import Array

from pde_slam.coords import ENUFrame
from pde_slam.interpolator import FieldInterpolator, SpatialGrid

# Columns that identify *navigation* data — excluded from the list of
# available scalar fields.
_NAV_COLS: frozenset[str] = frozenset(
    {
        "timestamp_s",
        "t_s",
        "Time",
        "Date",
        "Date (MMDDYY)",
        "Time (HHMMSS)",
        "lat_deg",
        "lon_deg",
        "Latitude",
        "Longitude",
        "x_m",
        "y_m",
        "depth_m",
        "Depth (m)",
        "heading_rad",
        "Heading (degrees Magnetic)",
        "roll_rad",
        "Roll (degrees)",
        "pitch_rad",
        "Pitch (degrees)",
        "thrust",
        "Thrust (% Thrust)",
        "Thrust difference (% Thrust)",
        "Control Mode",
        "Heave",
        "Temperature in electronics box (degrees C)",
        "Acceleration x, forward (G)",
        "Acceleration y, starboard (G)",
        "Acceleration z, down (G)",
        "Yaw rate [degrees/s]",
        "Pressure (psia)",
        "ODO (%Sat)",
        "ODO (mg/L)",
        "Turbidity (FNU)",
        "Wiper Position (V)",
        "rudder",
        "speed_mps",
    }
)


class SurveyLoader:
    """Load a survey CSV and convert observations to local ENU metres.

    The loader auto-detects the coordinate mode at :meth:`load` time:

    * **Pre-projected mode**: when the CSV contains the columns named by
      *x_col* and *y_col* (default ``"x_m"`` / ``"y_m"``), those columns are
      used directly as ENU east/north metres and the ENU projection is skipped.
    * **Lat/lon mode**: otherwise the columns named by *lat_col* / *lon_col*
      are projected through :attr:`frame` to obtain ENU metres.

    Parameters
    ----------
    frame :
        :class:`~pde_slam.coords.ENUFrame` that defines the local coordinate
        origin.  Required for lat/lon mode; accepted (but unused for
        projection) in pre-projected mode.
    field :
        Name of the scalar column to use as the PDE initial condition
        (e.g. ``"salinity_psu"``, ``"temperature_c"``).
    lat_col :
        Name of the latitude column in the CSV [degrees].  Only used in
        lat/lon mode.
    lon_col :
        Name of the longitude column in the CSV [degrees].  Only used in
        lat/lon mode.
    x_col :
        Name of the pre-projected east column in the CSV [metres].  When this
        column is present the loader operates in pre-projected mode.
    y_col :
        Name of the pre-projected north column in the CSV [metres].  When this
        column is present the loader operates in pre-projected mode.
    """

    def __init__(
        self,
        frame: ENUFrame,
        field: str = "Salinity (PPT)",
        *,
        lat_col: str = "Latitude",
        lon_col: str = "Longitude",
        x_col: str = "x_m",
        y_col: str = "y_m",
    ) -> None:
        self.frame = frame
        self.field = field
        self._lat_col = lat_col
        self._lon_col = lon_col
        self._x_col = x_col
        self._y_col = y_col

        self._xy_obs: np.ndarray | None = None
        self._values: np.ndarray | None = None
        self._df: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, csv_path: str | Path, max_rows: int | None = None) -> SurveyLoader:
        """Read *csv_path*, validate columns, and project to ENU metres.

        Parameters
        ----------
        csv_path :
            Path to the survey CSV file.

        Returns
        -------
        self :
            Returns ``self`` for method chaining.

        Raises
        ------
        FileNotFoundError
            If *csv_path* does not exist.
        KeyError
            If required columns are missing from the CSV.
        ValueError
            If the selected *field* column contains non-finite values after
            dropping NaNs, or if fewer than 4 rows remain.
        """
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Survey CSV not found: {path}")

        if max_rows is not None:
            df = pd.read_csv(path, nrows=max_rows)
        else:
            df = pd.read_csv(path)

        pre_projected = self._is_pre_projected(df)
        self._validate_columns(df, pre_projected=pre_projected)

        # Drop rows where the coordinate or field columns are NaN
        if pre_projected:
            coord_cols = [self._x_col, self._y_col]
        else:
            coord_cols = [self._lat_col, self._lon_col]
            df = df[(df[self._lat_col] != 0.0) & (df[self._lon_col] != 0.0)]

        df = df.dropna(subset=[*coord_cols, self.field]).reset_index(drop=True)

        if len(df) < 4:
            raise ValueError(
                f"Only {len(df)} valid rows after dropping NaNs — need ≥ 4."
            )

        vals = df[self.field].to_numpy(dtype=np.float64)

        if not np.all(np.isfinite(vals)):
            raise ValueError(
                f"Column '{self.field}' contains non-finite values after dropping NaNs."
            )

        if pre_projected:
            east = df[self._x_col].to_numpy(dtype=np.float64)
            north = df[self._y_col].to_numpy(dtype=np.float64)
            self._xy_obs = np.stack([east, north], axis=1)  # (N, 2) float64
        else:
            lats = df[self._lat_col].to_numpy(dtype=np.float64)
            lons = df[self._lon_col].to_numpy(dtype=np.float64)
            self._xy_obs = self.frame.to_enu_xy(lats, lons)  # (N, 2) float64

        self._values = vals
        self._df = df

        return self

    # ------------------------------------------------------------------
    # Accessors (available after load())
    # ------------------------------------------------------------------

    @property
    def xy_obs(self) -> np.ndarray:
        """Observation positions in ENU metres, shape ``(N, 2)`` — ``[east, north]``.

        Raises
        ------
        RuntimeError
            If :meth:`load` has not been called.
        """
        self._check_loaded()
        return self._xy_obs  # type: ignore[return-value]

    @property
    def values(self) -> np.ndarray:
        """Scalar values for the selected field, shape ``(N,)``.

        Raises
        ------
        RuntimeError
            If :meth:`load` has not been called.
        """
        self._check_loaded()
        return self._values  # type: ignore[return-value]

    @property
    def n_obs(self) -> int:
        """Number of valid observations loaded."""
        self._check_loaded()
        return len(self._values)  # type: ignore[arg-type]

    def available_fields(self) -> list[str]:
        """Return the scalar field columns present in the loaded CSV.

        Returns
        -------
        fields :
            Column names excluding navigation columns.

        Raises
        ------
        RuntimeError
            If :meth:`load` has not been called.
        """
        self._check_loaded()
        assert self._df is not None
        return [c for c in self._df.columns if c not in _NAV_COLS]

    # ------------------------------------------------------------------
    # Convenience: full pipeline to initial condition
    # ------------------------------------------------------------------

    def to_initial_condition(
        self,
        grid: SpatialGrid,
        method: str = "rbf",
        *,
        rbf_kernel: str = "thin_plate_spline",
        rbf_smoothing: float = 0.0,
    ) -> Array:
        """Interpolate the loaded observations onto *grid*.

        Combines :meth:`load` data with :class:`~pde_slam.interpolator.FieldInterpolator`
        to produce a JAX array suitable as ``phi0`` for the PDE solver.

        Parameters
        ----------
        grid :
            Target :class:`~pde_slam.interpolator.SpatialGrid`.
        method :
            Interpolation backend: ``"rbf"`` (default) or ``"spline"``.
        rbf_kernel :
            RBF kernel (only used when *method* is ``"rbf"``).
        rbf_smoothing :
            RBF smoothing factor (0 = exact interpolation).

        Returns
        -------
        phi0 :
            JAX array of shape ``(ny, nx)`` ready for the PDE solver.

        Raises
        ------
        RuntimeError
            If :meth:`load` has not been called first.
        """
        self._check_loaded()
        interp = FieldInterpolator(
            grid,
            method=method,  # type: ignore[arg-type]
            kernel=rbf_kernel,
            smoothing=rbf_smoothing,
        )
        return interp.fit_predict(self.xy_obs, self.values)

    def load_fields(
        self,
        csv_path: str | Path,
        fields: Sequence[str],
        *,
        method: str = "rbf",
        grid: SpatialGrid | None = None,
    ) -> dict[str, np.ndarray]:
        """Load multiple scalar fields from *csv_path* into ENU-projected arrays.

        This does **not** call :class:`~pde_slam.interpolator.FieldInterpolator`;
        it returns raw ``(N,)`` value arrays so the caller controls interpolation.

        Parameters
        ----------
        csv_path :
            Path to the CSV file.
        fields :
            Sequence of column names to load (e.g. ``["salinity_psu", "temperature_c"]``).

        Returns
        -------
        data :
            Mapping ``{field_name: values_array}`` where each array has shape ``(N,)``.
            The same ``xy_obs`` applies to all fields (access via ``self.xy_obs``).
        """
        # Load the first field to set xy_obs and validate the CSV
        original_field = self.field
        self.field = fields[0]
        self.load(csv_path)

        result: dict[str, np.ndarray] = {fields[0]: self._values.copy()}  # type: ignore[union-attr]

        assert self._df is not None
        for f in fields[1:]:
            if f not in self._df.columns:
                raise KeyError(
                    f"Field '{f}' not found in CSV columns: {list(self._df.columns)}"
                )
            vals = self._df[f].to_numpy(dtype=np.float64)
            result[f] = vals

        self.field = original_field
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _is_pre_projected(self, df: pd.DataFrame) -> bool:
        """Return ``True`` when *df* contains pre-projected ENU metre columns."""
        return self._x_col in df.columns and self._y_col in df.columns

    def _validate_columns(self, df: pd.DataFrame, *, pre_projected: bool) -> None:
        """Raise :exc:`KeyError` if required columns are absent.

        Parameters
        ----------
        df :
            DataFrame to validate.
        pre_projected :
            If ``True``, validate *x_col* / *y_col*; otherwise *lat_col* / *lon_col*.
        """
        if pre_projected:
            coord_cols = [self._x_col, self._y_col]
        else:
            coord_cols = [self._lat_col, self._lon_col]
        missing = [col for col in (*coord_cols, self.field) if col not in df.columns]
        if missing:
            raise KeyError(
                f"Required column(s) not found in CSV: {missing}. "
                f"Available: {list(df.columns)}"
            )

    def _check_loaded(self) -> None:
        if self._xy_obs is None:
            raise RuntimeError("Call load() before accessing survey data.")

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        loaded = f"n_obs={self.n_obs}" if self._xy_obs is not None else "not loaded"
        return f"SurveyLoader(frame={self.frame!r}, field={self.field!r}, {loaded})"
