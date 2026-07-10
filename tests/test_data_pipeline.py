"""Tests for pde_slam.data_pipeline."""
from __future__ import annotations

import numpy as np
import pytest

from pde_slam.data_pipeline import (
    RAW_LOG_COLUMNS,
    SpatialReplayBuffer,
    build_collocation_pool,
    build_measurement_pool,
    parse_log_strings,
)

# Minimal valid CSV header + 3 rows
_VALID_CSV_LINES = [
    ",".join(RAW_LOG_COLUMNS),
    "0.0,36.8,-76.0,0.5,0.0,0.5,-0.1,33.5,20.0,-40.0",
    "1.0,36.801,-76.001,0.5,0.01,0.5,-0.1,33.6,20.1,-40.1",
    "2.0,36.802,-76.002,0.5,0.02,0.5,-0.1,33.7,20.2,-40.2",
]


class TestParseLogs:
    def test_parse_valid_csv(self) -> None:
        df = parse_log_strings(_VALID_CSV_LINES)
        assert len(df) == 3
        assert list(df.columns) == RAW_LOG_COLUMNS

    def test_comment_lines_skipped(self) -> None:
        lines = ["# this is a comment\n"] + _VALID_CSV_LINES
        df = parse_log_strings(lines)
        assert len(df) == 3

    def test_empty_input_returns_empty_df(self) -> None:
        df = parse_log_strings([])
        assert len(df) == 0

    def test_headerless_parsing(self) -> None:
        """Header-less CSV (no column names) should parse via positional mapping."""
        headerless = _VALID_CSV_LINES[1:]  # skip header row
        df = parse_log_strings(headerless)
        assert len(df) == 3


class TestMeasurementPool:
    def test_pool_contains_required_keys(self) -> None:
        df = parse_log_strings(_VALID_CSV_LINES)
        xy_enu = np.zeros((len(df), 2))
        pool = build_measurement_pool(df, xy_enu)
        assert "timestamp_s" in pool
        assert "xy_enu" in pool

    def test_pool_rejects_missing_fields(self) -> None:
        df = parse_log_strings(_VALID_CSV_LINES)
        xy_enu = np.zeros((len(df), 2))
        with pytest.raises(KeyError):
            build_measurement_pool(df, xy_enu, field_keys=["nonexistent_field"])


class TestCollocationPool:
    def test_shape(self) -> None:
        pts = build_collocation_pool((-100, 100), (-100, 100), n_points=500)
        assert pts.shape == (500, 2)

    def test_within_bounds(self) -> None:
        pts = build_collocation_pool((0.0, 50.0), (10.0, 60.0), n_points=200)
        assert np.all(pts[:, 0] >= 0.0) and np.all(pts[:, 0] <= 50.0)
        assert np.all(pts[:, 1] >= 10.0) and np.all(pts[:, 1] <= 60.0)


class TestReplayBuffer:
    def test_add_and_sample(self) -> None:
        buf = SpatialReplayBuffer(capacity=100)
        for i in range(50):
            buf.add(float(i), np.array([float(i), float(i)]), {"salinity_psu": float(i)})
        assert len(buf) == 50
        batch = buf.sample(10, field_keys=["salinity_psu"])
        assert batch["xy_enu"].shape == (10, 2)

    def test_reservoir_sampling_caps_capacity(self) -> None:
        buf = SpatialReplayBuffer(capacity=10)
        for i in range(100):
            buf.add(float(i), np.zeros(2), {"salinity_psu": 0.0})
        assert len(buf) <= 10

    def test_batch_add(self) -> None:
        buf = SpatialReplayBuffer(capacity=500)
        timestamps = np.arange(30, dtype=float)
        xy = np.random.randn(30, 2)
        fields = {"salinity_psu": np.random.randn(30)}
        buf.add_batch(timestamps, xy, fields)
        assert len(buf) == 30
