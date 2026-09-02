"""
test_survey.py
==============
Unit tests for field survey CSV ingestion, kinematic differential drive
interpolation, and dead reckoning noise simulation.
"""

from __future__ import annotations

import io
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from pde_slam.coords import ENUFrame
from pde_slam.io.survey import (
    SurveyTrajectory,
    _clean_survey_dataframe,
    interpolate_kinematic_trajectory,
    load_survey_csv,
)
from pde_slam.kinematics.diff_drive import DiffDriveKinematics


class TestSurveyDataframeCleaning:
    """Test GPS outlier filtering and timestamp parsing."""

    def test_clean_filters_null_island_and_duplicates(self) -> None:
        csv_text = (
            "Date,Time,Latitude,Longitude,Heading\n"
            "20250609,100000,25.912,-80.137,10.0\n"
            "20250609,100001,0.0,0.0,12.0\n"  # null island dropout
            "20250609,100002,25.913,-80.138,14.0\n"
            "20250609,100002,25.913,-80.138,14.0\n"  # duplicate timestamp
            "20250609,100005,25.914,-80.139,15.0\n"
        )
        df = pd.read_csv(io.StringIO(csv_text))
        cleaned = _clean_survey_dataframe(df)

        assert len(cleaned) == 3
        assert np.all(cleaned["Latitude"] != 0.0)
        assert np.all(cleaned["Longitude"] != 0.0)
        # Check monotonic timestamps
        t = cleaned["_t_sec"].to_numpy()
        assert t[0] == 0.0
        assert t[1] == 2.0
        assert t[2] == 5.0


class TestKinematicInterpolation:
    """Test differential drive kinematic trajectory interpolation."""

    def test_kinematic_trajectory_reconstruction(self) -> None:
        # Synthetic straight line path: 100m in 100s at 1 m/s along x-axis
        t_raw = np.array([0.0, 20.0, 50.0, 100.0, 200.0])
        xy_raw = np.column_stack([t_raw * 1.0, np.zeros_like(t_raw)])

        t_grid, coords, headings, v, w = interpolate_kinematic_trajectory(
            t_raw=t_raw,
            xy_raw=xy_raw,
            t_max=100.0,
            dt=1.0,
        )

        assert len(t_grid) == 101
        assert coords.shape == (101, 2)
        assert np.isclose(coords[0, 0], 0.0, atol=1e-3)
        assert np.isclose(coords[-1, 0], 100.0, atol=1e-3)
        assert np.allclose(coords[:, 1], 0.0, atol=1e-3)
        # Speed should be ~1 m/s
        assert np.allclose(v[:-1], 1.0, atol=1e-2)
        assert np.allclose(w[:-1], 0.0, atol=1e-2)

    def test_cap_duration_at_specified_t_max(self) -> None:
        t_raw = np.linspace(0, 1000, 101)
        xy_raw = np.column_stack([t_raw, t_raw])

        t_grid, coords, _, _, _ = interpolate_kinematic_trajectory(
            t_raw=t_raw,
            xy_raw=xy_raw,
            t_max=500.0,
            dt=1.0,
        )

        assert t_grid[-1] == 500.0
        assert len(t_grid) == 501
        assert len(coords) == 501


class TestSurveyCsvLoader:
    """Test end-to-end loading from real CSV survey file."""

    def test_load_real_survey_csv(self) -> None:
        csv_path = Path("data/csv/data.csv")
        if not csv_path.exists():
            pytest.skip("data/csv/data.csv not found.")

        frame = ENUFrame(lat0=25.912777, lon0=-80.13774)
        traj = load_survey_csv(
            csv_path=csv_path,
            t_max=500.0,
            dt=1.0,
            enu_frame=frame,
        )

        assert isinstance(traj, SurveyTrajectory)
        assert traj.duration == 500.0
        assert traj.n_steps == 500
        assert traj.coords_enu.shape == (501, 2)
        assert traj.velocities.shape == (501,)
        assert traj.omegas.shape == (501,)
        assert "salinity" in traj.measurements
        assert "temperature" in traj.measurements
        assert "odo" in traj.measurements
        assert len(traj.measurements["salinity"]) == 501

    def test_dead_reckoning_noise_simulation(self) -> None:
        csv_path = Path("data/csv/data.csv")
        if not csv_path.exists():
            pytest.skip("data/csv/data.csv not found.")

        traj = load_survey_csv(csv_path, t_max=100.0, dt=1.0)
        n_steps = traj.n_steps
        v_nom = traj.velocities[:-1]
        w_nom = traj.omegas[:-1]

        # Simulate DR noise
        key = jax.random.PRNGKey(42)
        k_v, k_w = jax.random.split(key)
        v_noise = 0.01 * np.array(jax.random.normal(k_v, shape=(n_steps,)))
        w_noise = 0.005 * np.array(jax.random.normal(k_w, shape=(n_steps,)))

        v_act = np.clip(v_nom + v_noise, 0.0, None)
        w_act = w_nom + w_noise

        x0 = jnp.array([traj.coords_enu[0, 0], traj.coords_enu[0, 1], traj.headings[0]])
        coords_dr = np.array(
            DiffDriveKinematics.integrate_trajectory(
                x0=x0,
                velocities=jnp.asarray(v_act),
                omegas=jnp.asarray(w_act),
                dt=1.0,
                include_initial=True,
            )[:, :2]
        )

        assert coords_dr.shape == traj.coords_enu.shape
        # Dead reckoning should drift slightly from the true trajectory due to noise
        dr_rmse = np.sqrt(np.mean((coords_dr - traj.coords_enu) ** 2))
        assert dr_rmse > 0.0
        assert dr_rmse < 10.0  # reasonable drift within 100s
