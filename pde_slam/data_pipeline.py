"""
data_pipeline.py
================
Data ingestion, parsing, and spatial replay buffer management.

Responsibilities
----------------
1. **Log parsing** – convert raw comma-separated / JSON log strings into typed
   ``pandas`` DataFrames with physical units.
2. **Sensor Measurement Pool** – a structured store of ``(t, x_enu, y_enu,
   field_value)`` tuples for each scalar observable.
3. **PDE Collocation Pool** – a spatial random-access buffer that draws
   ``(x, y)`` query points for evaluating PDE residuals during Phase 2.
4. **Spatial Replay Buffer** – maintains a rolling window of recent
   observations for online loss computation, supporting reservoir sampling
   to avoid catastrophic forgetting.

Data flow
---------
::

    Raw log strings
         │
         ▼  parse_log_strings()
    Raw DataFrame
         │
         ▼  build_measurement_pool()
    Sensor Measurement Pool  ──►  interpolator.build_initial_condition()
         │
         ▼  build_collocation_pool()
    PDE Collocation Pool  ──►  optimizer (Phase 2 PDE residual loss)
         │
         ▼  SpatialReplayBuffer.add() / sample()
    Online mini-batches  ──►  optimizer (Phase 2 data alignment loss)

Interaction with other modules
------------------------------
* Produces data consumed by :mod:`pde_slam.interpolator` (Phase 1) and
  :mod:`pde_slam.optimizer` (Phase 2).
* Uses corrected poses from :mod:`pde_slam.kinematics`.
"""

from __future__ import annotations

import io
import json
import random
from collections import deque
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

#: Canonical column names expected in raw CSV log files.
RAW_LOG_COLUMNS: list[str] = [
    "timestamp_s",   # POSIX epoch seconds
    "lat_deg",       # WGS-84 latitude
    "lon_deg",       # WGS-84 longitude
    "depth_m",       # depth below surface (positive down)
    "heading_rad",   # vehicle heading (ENU convention)
    "thrust",        # normalised thruster command [0, 1]
    "rudder",        # normalised rudder command [-1, 1]
    "salinity_psu",  # practical salinity units
    "temperature_c", # Celsius
    "backscatter_db",# acoustic backscatter [dB re 1 m⁻¹]
]

#: Fields that represent measurable scalar PDE quantities.
SCALAR_FIELDS: list[str] = ["salinity_psu", "temperature_c", "backscatter_db"]


def parse_log_strings(
    log_lines: list[str],
    *,
    delimiter: str = ",",
    comment_char: str = "#",
) -> pd.DataFrame:
    """Parse raw comma-separated log strings into a typed DataFrame.

    Lines beginning with *comment_char* (default ``"#"``) are skipped.
    The function accepts either a header-less log (columns inferred from
    :data:`RAW_LOG_COLUMNS`) or a log whose first non-comment line is a
    header.

    Parameters
    ----------
    log_lines :
        List of raw log strings.  Each string represents one measurement
        epoch.  May be read directly from a file via ``f.readlines()``.
    delimiter :
        Field separator character (default ``","``).
    comment_char :
        Lines whose first non-whitespace character matches this string are
        ignored.

    Returns
    -------
    df :
        DataFrame with columns matching :data:`RAW_LOG_COLUMNS` and
        ``dtype=float64`` for all numeric columns.  The index is the
        original line order (integer).

    Raises
    ------
    ValueError
        If the number of parsed columns does not match
        :data:`RAW_LOG_COLUMNS`.
    """
    cleaned = [
        line.strip()
        for line in log_lines
        if line.strip() and not line.strip().startswith(comment_char)
    ]
    if not cleaned:
        return pd.DataFrame(columns=RAW_LOG_COLUMNS)

    # Detect header
    first = cleaned[0]
    if all(col in first for col in ["timestamp_s", "lat_deg"]):
        text_block = "\n".join(cleaned)
        df = pd.read_csv(io.StringIO(text_block), sep=delimiter)
    else:
        text_block = "\n".join(cleaned)
        df = pd.read_csv(
            io.StringIO(text_block),
            sep=delimiter,
            header=None,
            names=RAW_LOG_COLUMNS,
        )

    if len(df.columns) != len(RAW_LOG_COLUMNS):
        raise ValueError(
            f"Expected {len(RAW_LOG_COLUMNS)} columns; got {len(df.columns)}.\n"
            f"  Expected: {RAW_LOG_COLUMNS}\n"
            f"  Got:      {list(df.columns)}"
        )
    return df.astype(float)


def parse_json_log(path: str) -> pd.DataFrame:
    """Parse a newline-delimited JSON log file.

    Each line must be a JSON object with keys matching :data:`RAW_LOG_COLUMNS`.

    Parameters
    ----------
    path :
        Filesystem path to the JSON log.

    Returns
    -------
    df :
        Parsed DataFrame.
    """
    records: list[dict[str, Any]] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records, columns=RAW_LOG_COLUMNS)


# ---------------------------------------------------------------------------
# Sensor Measurement Pool
# ---------------------------------------------------------------------------


def build_measurement_pool(
    df: pd.DataFrame,
    xy_enu: np.ndarray,
    field_keys: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Assemble the Sensor Measurement Pool from a parsed log DataFrame.

    The pool is a simple dictionary of NumPy arrays keyed by field name,
    with the spatial positions stored under ``"xy_enu"``.

    Parameters
    ----------
    df :
        Parsed log DataFrame (output of :func:`parse_log_strings`).
    xy_enu :
        ENU positions array of shape ``(N, 2)`` corresponding to ``df``
        rows, computed by :func:`pde_slam.kinematics.latlon_to_enu`.
    field_keys :
        Scalar field column names to include.  Defaults to
        :data:`SCALAR_FIELDS`.

    Returns
    -------
    pool :
        Dictionary with keys:

        * ``"timestamp_s"``  – shape ``(N,)``
        * ``"xy_enu"``       – shape ``(N, 2)``
        * *<field_key>*      – shape ``(N,)`` for each requested field

    Raises
    ------
    KeyError
        If any requested *field_keys* are not present in *df*.
    """
    if field_keys is None:
        field_keys = SCALAR_FIELDS

    missing = [k for k in field_keys if k not in df.columns]
    if missing:
        raise KeyError(f"Missing field columns in DataFrame: {missing}")

    pool: dict[str, np.ndarray] = {
        "timestamp_s": df["timestamp_s"].to_numpy(dtype=np.float64),
        "xy_enu": xy_enu.astype(np.float64),
    }
    for key in field_keys:
        pool[key] = df[key].to_numpy(dtype=np.float64)

    return pool


# ---------------------------------------------------------------------------
# PDE Collocation Pool
# ---------------------------------------------------------------------------


def build_collocation_pool(
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    n_points: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a uniform random collocation point set for PDE residual evaluation.

    Collocation points are sampled uniformly within the spatial domain and
    held fixed throughout Phase 2 to provide stable gradient estimates of
    the PDE residual loss.

    Parameters
    ----------
    x_range :
        ``(x_min, x_max)`` East extent [m].
    y_range :
        ``(y_min, y_max)`` North extent [m].
    n_points :
        Number of collocation points to generate.
    rng :
        Optional :class:`numpy.random.Generator` for reproducibility.
        If ``None``, a fresh default generator is used.

    Returns
    -------
    colloc_pts :
        Array of shape ``(n_points, 2)`` – ``[east_m, north_m]``.
    """
    if rng is None:
        rng = np.random.default_rng()
    xs = rng.uniform(x_range[0], x_range[1], size=n_points)
    ys = rng.uniform(y_range[0], y_range[1], size=n_points)
    return np.column_stack([xs, ys])


# ---------------------------------------------------------------------------
# Spatial Replay Buffer
# ---------------------------------------------------------------------------


class SpatialReplayBuffer:
    """Fixed-capacity replay buffer for online Phase 2 SLAM updates.

    Uses reservoir sampling to maintain a statistically representative
    subset of all past observations, preventing catastrophic forgetting
    of early-trajectory constraints.

    Parameters
    ----------
    capacity :
        Maximum number of samples retained.
    seed :
        Random seed for reproducible sampling.
    """

    def __init__(self, capacity: int = 10_000, seed: int = 0) -> None:
        self.capacity = capacity
        self._rng = random.Random(seed)
        self._buffer: deque[dict[str, float]] = deque(maxlen=capacity)
        self._total_seen: int = 0

    def add(
        self,
        timestamp: float,
        xy_enu: np.ndarray,
        field_values: dict[str, float],
    ) -> None:
        """Add a single measurement epoch to the buffer.

        Uses reservoir sampling when the buffer is at capacity so that each
        past observation has an equal probability of being retained.

        Parameters
        ----------
        timestamp :
            POSIX timestamp [s] of the observation.
        xy_enu :
            ENU position ``[east_m, north_m]``.
        field_values :
            Dictionary mapping field names to scalar measurements.
        """
        record: dict[str, float] = {
            "t": timestamp,
            "x": float(xy_enu[0]),
            "y": float(xy_enu[1]),
            **{k: float(v) for k, v in field_values.items()},
        }
        self._total_seen += 1
        if len(self._buffer) < self.capacity:
            self._buffer.append(record)
        else:
            # Reservoir sampling: replace a random element
            idx = self._rng.randint(0, self._total_seen - 1)
            if idx < self.capacity:
                buf_list = list(self._buffer)
                buf_list[idx] = record
                self._buffer = deque(buf_list, maxlen=self.capacity)

    def add_batch(
        self,
        timestamps: np.ndarray,
        xy_enus: np.ndarray,
        field_arrays: dict[str, np.ndarray],
    ) -> None:
        """Batch-insert multiple observations.

        Parameters
        ----------
        timestamps :
            Array of shape ``(N,)`` of POSIX timestamps.
        xy_enus :
            Array of shape ``(N, 2)`` of ENU positions.
        field_arrays :
            Dictionary mapping field names to shape-``(N,)`` arrays.
        """
        N = len(timestamps)
        for i in range(N):
            fv = {k: float(v[i]) for k, v in field_arrays.items()}
            self.add(float(timestamps[i]), xy_enus[i], fv)

    def sample(
        self,
        batch_size: int,
        field_keys: list[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """Draw a random mini-batch from the buffer.

        Parameters
        ----------
        batch_size :
            Number of samples to return.  Clamped to ``len(buffer)`` if
            the buffer has fewer entries than requested.
        field_keys :
            Scalar field names to include.  If ``None``, all stored field
            keys are returned.

        Returns
        -------
        batch :
            Dictionary with keys ``"t"``, ``"xy_enu"``, and each *field_key*.
            Arrays are of shape ``(batch_size, ...)`` with consistent row
            ordering.
        """
        actual_size = min(batch_size, len(self._buffer))
        samples = self._rng.sample(list(self._buffer), actual_size)

        ts = np.array([s["t"] for s in samples])
        xy = np.stack([[s["x"], s["y"]] for s in samples])

        if field_keys is None:
            stored_keys = [k for k in samples[0] if k not in ("t", "x", "y")]
            field_keys = stored_keys

        batch: dict[str, np.ndarray] = {"t": ts, "xy_enu": xy}
        for key in field_keys:
            batch[key] = np.array([s[key] for s in samples])
        return batch

    def __len__(self) -> int:
        return len(self._buffer)

    def __repr__(self) -> str:
        return (
            f"SpatialReplayBuffer(capacity={self.capacity}, "
            f"stored={len(self._buffer)}, total_seen={self._total_seen})"
        )
