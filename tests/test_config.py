"""Unit tests for the YAML configuration loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from pde_slam.config import load_config


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


def test_load_config_valid(temp_yaml_file: Path) -> None:
    """Verify parsing of a valid YAML configuration."""
    config = load_config(temp_yaml_file)

    assert config.grid.x_min == -100.0
    assert config.grid.nx == 50
    assert config.solver.dt_max == 0.5
    assert config.plumes.centers == [[0.0, 0.0]]
    assert config.plumes.num_random == 2
    assert config.plumes.seed == 99
    assert config.pde_params.D == [0.5, 0.8]
    assert config.pde_params.v_flow == [0.2, -0.4]
    assert config.pde_params.k_thrust == 2.5
    assert config.optimization.lambda_reg == 0.05
    assert config.optimization.method == "adam"
    assert config.optimization.maxiter == 50
    assert config.optimization.learning_rate == 0.02
    assert config.optimization.num_steps == 120


def test_load_config_file_not_found() -> None:
    """Verify loader raises FileNotFoundError for missing path."""
    with pytest.raises(FileNotFoundError):
        load_config("non_existent_file_path.yaml")


def test_load_config_invalid_keys(tmp_path: Path) -> None:
    """Verify loader raises ValueError for missing required keys."""
    invalid_content = """
grid:
  x_min: -100.0
  x_max: 100.0
# Missing y_min, y_max, nx, ny
pde_params:
  D: [0.5]
  v_flow: [0.1, 0.1]
  k_thrust: 1.0
"""
    file_path = tmp_path / "invalid_config.yaml"
    file_path.write_text(invalid_content)

    with pytest.raises(ValueError):
        load_config(file_path)
