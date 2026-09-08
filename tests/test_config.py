"""Unit tests for the YAML configuration loader."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def temp_yaml_file(tmp_path: Path) -> Path:
    """Fixture that generates a temporary valid YAML config file."""
    yaml_content = """
grid:
  x_min: -100.0
  x_max: 100.0
  y_min: -100.0
  y_max: 100.0
  nx: 50
  ny: 50

solver:
  dt_max: 0.5

plumes:
  centers:
    - [0.0, 0.0]
  widths:
    - 10.0
  amplitudes:
    - 1.5
  num_random: 2
  seed: 99

pde_params:
  D: [0.5, 0.8]
  v_flow: [0.2, -0.4]
  k_thrust: 2.5

optimization:
  lambda_reg: 0.05
  method: "adam"
  maxiter: 50
  learning_rate: 0.02
  num_steps: 120
"""
    file_path = tmp_path / "test_config.yaml"
    file_path.write_text(yaml_content)
    return file_path


def test_load_rbpf_experiment_config(tmp_path: Path) -> None:
    """Verify parsing of RBPF experiment configuration YAML."""
    from pde_slam.config import load_rbpf_experiment_config

    yaml_content = """
simulation:
  sim_dir: "data/adv_diff_simulations/biscayne_simulation"
  fields:
    - "Salinity"
    - "Temperature"
  mask_outside: true

ic_anchors:
  mode: "auto"
  n_points: 20
  seed: 123
  epochs: 25

robot:
  kinematics: "diff_drive"
  nominal_speed: 1.8
  dt: 0.5
  v_noise_std: 0.04
  omega_noise_std: 0.01
  acceptance_radius: 4.0

rbpf:
  n_particles: 150
  pos_init_std: 0.3
  heading_init_std: 0.04
  measurement_noise_std: 0.08
  resample_threshold: 0.6
  seed: 99

pinn:
  arch: "modified_mlp"
  hidden_dim: 32
  num_layers: 4
  learning_rate: 0.002
  num_steps: 20
  num_colloc: 64
  margin: 15.0
  w_pde: 0.8

output:
  sim_name: "test_biscayne"
  results_dir: "output"
  figures_dir: "figures"
  checkpoints: [0, 50, 100]

"""
    file_path = tmp_path / "rbpf_exp.yaml"
    file_path.write_text(yaml_content)

    cfg = load_rbpf_experiment_config(file_path)
    assert cfg.simulation.sim_dir == "data/adv_diff_simulations/biscayne_simulation"
    assert cfg.simulation.fields == ["Salinity", "Temperature"]
    assert cfg.ic_anchors.n_points == 20
    assert cfg.ic_anchors.epochs == 25
    assert cfg.robot.nominal_speed == 1.8
    assert cfg.rbpf.n_particles == 150
    assert cfg.pinn.arch == "modified_mlp"
    assert cfg.output.sim_name == "test_biscayne"


def test_load_existing_configs() -> None:
    """Verify all repository config files load without error."""
    from pde_slam.config import load_rbpf_experiment_config

    for cfg_name in [
        "biscayne_rbpf_simulation.yaml",
        "miami_canal_rbpf_simulation.yaml",
    ]:
        p = Path("configs") / cfg_name
        if p.exists():
            cfg = load_rbpf_experiment_config(p)
            assert len(cfg.simulation.fields) > 0
