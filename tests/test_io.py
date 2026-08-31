"""
test_io.py
==========
Unit tests for the experiment serialization module pde_slam.io.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from pde_slam.io import SlamExperimentData, load_experiment, save_experiment
from pde_slam.pinn import PinnConfig, PinnFieldMap


def test_save_and_load_experiment(tmp_path):
    """Test saving and loading a SlamExperimentData object."""
    key = jax.random.PRNGKey(0)
    config = PinnConfig(
        x_bounds=(-50.0, 50.0),
        y_bounds=(-50.0, 50.0),
        t_max=100.0,
        n_fields=2,
    )
    pinn_map = PinnFieldMap(config=config, key=key)

    data = SlamExperimentData(
        grid_extent={
            "x_min": -50.0,
            "x_max": 50.0,
            "y_min": -50.0,
            "y_max": 50.0,
            "nx": 50,
            "ny": 50,
        },
        enu_origin=(25.9, -80.1),
        polygon_enu=np.array([[0, 0], [10, 0], [10, 10], [0, 10]]),
        sample_times=np.linspace(0, 100, 10),
        field_names=["salinity", "temperature"],
        ground_truth_params={
            "v_flow": np.array([0.5, -0.2]),
            "D": np.array([0.1, 0.05]),
        },
        noise_params={
            "velocity_noise_std": 0.15,
            "omega_noise_std": 0.02,
            "obs_noise_std": 0.1,
            "q_lin": 1e-8,
            "p0_lin": 0.0025,
        },
        gt_solutions={"salinity": np.zeros((10, 50, 50))},
        field_means={"salinity": 35.0, "temperature": 25.0},
        field_stds={"salinity": 2.0, "temperature": 1.5},
        field_estimations=np.zeros((10, 2, 50, 50)),
        coords_true=np.zeros((10, 2)),
        coords_dr=np.zeros((10, 2)),
        oracle_traj=np.zeros((10, 2)),
        estimated_traj=np.zeros((10, 2)),
        velocities=np.ones(9),
        omegas=np.zeros(9),
        particle_poses_history=np.zeros((10, 20, 2)),
        particle_weights_history=np.zeros((10, 20)),
        ic_points_enu=np.array([[1.0, 2.0], [3.0, 4.0]]),
        obs_pts=np.zeros((50, 3)),
        obs_vals=np.zeros((50, 2)),
        pinn_config=config,
        pinn_checkpoints={0: pinn_map.params},
        loss_history=[0.5, 0.2, 0.1],
        v_flow_history=[np.array([0.1, 0.2])],
        D_history=[np.array([0.1, 0.1])],
    )

    file_path = tmp_path / "test_run.pkl"
    save_experiment(data, file_path)

    assert file_path.exists()

    loaded_data = load_experiment(file_path)

    assert loaded_data.field_names == ["salinity", "temperature"]
    assert loaded_data.field_means["salinity"] == 35.0
    assert loaded_data.field_stds["temperature"] == 1.5
    assert loaded_data.polygon_enu.shape == (4, 2)
    assert loaded_data.gt_solutions["salinity"].shape == (10, 50, 50)
    assert loaded_data.field_estimations.shape == (10, 2, 50, 50)
    assert loaded_data.noise_params["velocity_noise_std"] == 0.15
    assert loaded_data.pinn_config.n_fields == 2

    # Verify PINN map reconstruction
    reconstructed_pinn = loaded_data.reconstruct_pinn_map(0)
    assert reconstructed_pinn.params is not None

    # Verify ENU Frame and coordinate transformations
    assert loaded_data.enu_frame.lat0 == 25.9
    assert loaded_data.enu_frame.lon0 == -80.1
    lat, lon = loaded_data.to_geodetic(100.0, 200.0)
    e, n = loaded_data.to_enu(lat, lon)
    np.testing.assert_allclose([e, n], [100.0, 200.0], rtol=1e-5)

    query_poses = jnp.array([[5.0, -5.0]])
    orig_pred = pinn_map.predict(t=10.0, poses=query_poses)
    recon_pred = reconstructed_pinn.predict(t=10.0, poses=query_poses)
    np.testing.assert_allclose(np.array(orig_pred), np.array(recon_pred), rtol=1e-5)


def test_simulation_dataset_loader():
    """Test loading NPZ simulation files from sample directory."""
    from pde_slam.io.simulation import (
        generate_ic_anchors,
        load_simulation_dataset,
        sample_simulation_field,
    )

    sim_dir = "data/adv_diff_simulations/biscayne_simulation"
    dataset = load_simulation_dataset(sim_dir)

    assert "salinity" in dataset.field_names
    assert dataset.grid.nx == 100
    assert dataset.grid.ny == 100
    assert len(dataset.polygon_enu) > 2

    # Test sampling continuous field
    val = sample_simulation_field(dataset, "salinity", t=5.0, x=0.0, y=0.0)
    assert isinstance(val, float)

    # Test generating IC anchors inside polygon
    anchors = generate_ic_anchors(dataset.polygon_enu, n_points=15, seed=42)
    assert len(anchors) <= 15
    assert anchors.shape[1] == 2
