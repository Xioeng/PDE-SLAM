"""
config.py
=========
YAML parser and typed configuration classes for PDE-SLAM pipeline and experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import yaml  # type: ignore[import-untyped]


class GridConfig(NamedTuple):
    """Configuration parameters for the SpatialGrid."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    nx: int
    ny: int


class PlumeConfig(NamedTuple):
    """Configuration parameters for generating water feature plume(s)."""

    centers: list[list[float]] = []
    widths: list[float] = []
    amplitudes: list[float] = []
    num_random: int = 0
    seed: int | None = None


class PipelineConfig(NamedTuple):
    """Top-level pipeline configuration container."""

    grid: GridConfig
    plumes: list[PlumeConfig]


@dataclass
class SimulationConfig:
    """Simulation dataset ingestion parameters."""

    sim_dir: str
    fields: list[str] = field(default_factory=lambda: ["salinity", "temperature"])
    mask_outside: bool = True


@dataclass
class IcAnchorsConfig:
    """Initial Condition (t=0) spatial measurement anchors parameters."""

    mode: str = "auto"
    n_points: int = 30
    seed: int = 42
    epochs: int = 20


@dataclass
class RobotConfig:
    """Robot motion model, control, and noise parameters."""

    kinematics: str = "diff_drive"
    nominal_speed: float = 1.5
    dt: float = 1.0
    v_noise_std: float = 0.05
    omega_noise_std: float = 0.02
    acceptance_radius: float = 5.0
    waypoints_mode: str = "auto"
    waypoints: list[list[float]] = field(default_factory=list)
    interactive: bool = False


@dataclass
class RbpfFilterConfig:
    """Rao-Blackwellized Particle Filter (RBPF) SLAM parameters."""

    n_particles: int = 200
    pos_init_std: float = 0.5
    heading_init_std: float = 0.05
    measurement_noise_std: float = 0.1
    lin_process_noise: float = 1e-4
    p0_lin: float = 0.0025
    resample_threshold: float = 0.5
    seed: int = 42


@dataclass
class PinnMapConfig:
    """Physics-Informed Neural Network (PINN) map parameters."""

    arch: str = "modified_mlp"
    hidden_dim: int = 64
    num_layers: int = 3
    learning_rate: float = 0.003
    num_steps: int = 15
    num_colloc: int = 128
    margin: float = 20.0
    w_pde: float = 0.5


@dataclass
class OutputConfig:
    """Output directories and checkpoint settings."""

    sim_name: str = "simulation"
    results_dir: str = "output"
    figures_dir: str = "output/figures"
    # Checkpoint percentages along trajectory (e.g. [0, 25, 50, 75, 100] for 0% to 100%)
    checkpoints: list[int | float] = field(default_factory=lambda: [0, 25, 50, 75, 100])
    satellite_zoom: int = 17
    save_experiment: bool = True
    save_grids: bool = True


@dataclass
class SurveyConfig:
    """CSV survey trajectory ingestion parameters."""

    csv_path: str = "data/csv/data.csv"
    t_max: float = 500.0
    dt: float = 1.0
    use_csv_measurements: bool = False


@dataclass
class RbpfExperimentConfig:
    """Comprehensive experiment configuration for RBPF simulation runs."""

    simulation: SimulationConfig
    ic_anchors: IcAnchorsConfig = field(default_factory=IcAnchorsConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    rbpf: RbpfFilterConfig = field(default_factory=RbpfFilterConfig)
    pinn: PinnMapConfig = field(default_factory=PinnMapConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    survey: SurveyConfig = field(default_factory=SurveyConfig)


def load_rbpf_experiment_config(yaml_path: str | Path) -> RbpfExperimentConfig:
    """Parse a YAML experiment configuration file into a typed RbpfExperimentConfig.

    Parameters
    ----------
    yaml_path : str or Path
        Path to the YAML configuration file.

    Returns
    -------
    RbpfExperimentConfig
        Typed experiment configuration object.
    """
    path = Path(yaml_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    sim_raw = raw.get("simulation", {})
    if "sim_dir" not in sim_raw and "data" in raw:
        # Fallback / compatibility with older schema
        data_raw = raw.get("data", {})
        sim_raw = {
            "sim_dir": data_raw.get(
                "sim_dir", "data/adv_diff_simulations/biscayne_simulation"
            ),
            "fields": data_raw.get("fields", ["salinity", "temperature"]),
            "mask_outside": raw.get("polygon", {}).get("mask_outside", True),
        }

    simulation_cfg = SimulationConfig(
        sim_dir=sim_raw.get("sim_dir", sim_raw.get("dataset_dir", "")),
        fields=sim_raw.get("fields", ["salinity", "temperature"]),
        mask_outside=sim_raw.get("mask_outside", True),
    )

    ic_raw = raw.get("ic_anchors", {})
    ic_cfg = IcAnchorsConfig(
        mode=ic_raw.get("mode", "auto"),
        n_points=int(ic_raw.get("n_points", ic_raw.get("num_points", 30))),
        seed=int(ic_raw.get("seed", 42)),
        epochs=int(
            ic_raw.get(
                "epochs",
                ic_raw.get(
                    "warmup_steps",
                    raw.get("pinn", {}).get("warmup_steps", 20),
                ),
            )
        ),
    )

    robot_raw = raw.get("robot", raw.get("rbpf", {}))
    robot_cfg = RobotConfig(
        kinematics=robot_raw.get("kinematics", "diff_drive"),
        nominal_speed=float(
            robot_raw.get("nominal_speed", robot_raw.get("speed_mps", 1.5))
        ),
        dt=float(robot_raw.get("dt", 1.0)),
        v_noise_std=float(robot_raw.get("v_noise_std", 0.05)),
        omega_noise_std=float(robot_raw.get("omega_noise_std", 0.02)),
        acceptance_radius=float(robot_raw.get("acceptance_radius", 5.0)),
        waypoints_mode=robot_raw.get("waypoints_mode", "auto"),
        waypoints=robot_raw.get("waypoints", []),
        interactive=bool(robot_raw.get("interactive", raw.get("interactive", False))),
    )

    rbpf_raw = raw.get("rbpf", {})
    rbpf_cfg = RbpfFilterConfig(
        n_particles=int(
            rbpf_raw.get("n_particles", rbpf_raw.get("num_particles", 200))
        ),
        pos_init_std=float(rbpf_raw.get("pos_init_std", 0.5)),
        heading_init_std=float(rbpf_raw.get("heading_init_std", 0.05)),
        measurement_noise_std=float(
            rbpf_raw.get("measurement_noise_std", rbpf_raw.get("obs_std", 0.1))
        ),
        lin_process_noise=float(rbpf_raw.get("lin_process_noise", 1e-4)),
        p0_lin=float(rbpf_raw.get("p0_lin", 0.0025)),
        resample_threshold=float(rbpf_raw.get("resample_threshold", 0.5)),
        seed=int(rbpf_raw.get("seed", 42)),
    )

    pinn_raw = raw.get("pinn", {})
    pinn_cfg = PinnMapConfig(
        arch=pinn_raw.get("arch", "modified_mlp"),
        hidden_dim=int(pinn_raw.get("hidden_dim", 64)),
        num_layers=int(pinn_raw.get("num_layers", 3)),
        learning_rate=float(pinn_raw.get("learning_rate", 0.003)),
        num_steps=int(pinn_raw.get("num_steps", 15)),
        num_colloc=int(pinn_raw.get("num_colloc", 128)),
        margin=float(pinn_raw.get("margin", 20.0)),
        w_pde=float(pinn_raw.get("w_pde", 0.5)),
    )

    out_raw = raw.get("output", {})
    out_cfg = OutputConfig(
        sim_name=out_raw.get("sim_name", Path(yaml_path).stem.replace("_config", "")),
        results_dir=out_raw.get("results_dir", "output"),
        figures_dir=out_raw.get("figures_dir", "output/figures"),
        checkpoints=out_raw.get("checkpoints", [0, 25, 50, 75, 100]),
        satellite_zoom=int(out_raw.get("satellite_zoom", 17)),
        save_experiment=out_raw.get("save_experiment", True),
        save_grids=out_raw.get("save_grids", True),
    )

    survey_raw = raw.get("survey", {})
    survey_cfg = SurveyConfig(
        csv_path=str(survey_raw.get("csv_path", "data/csv/data.csv")),
        t_max=float(survey_raw.get("t_max", 500.0)),
        dt=float(survey_raw.get("dt", 1.0)),
        use_csv_measurements=bool(survey_raw.get("use_csv_measurements", False)),
    )

    return RbpfExperimentConfig(
        simulation=simulation_cfg,
        ic_anchors=ic_cfg,
        robot=robot_cfg,
        rbpf=rbpf_cfg,
        pinn=pinn_cfg,
        output=out_cfg,
        survey=survey_cfg,
    )


def load_config(yaml_path: str | Path) -> PipelineConfig:
    """Parse a legacy YAML configuration file into a PipelineConfig object."""
    path = Path(yaml_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    grid_raw = raw.get("grid", {})
    grid_cfg = GridConfig(
        x_min=float(grid_raw.get("x_min", 0.0)),
        x_max=float(grid_raw.get("x_max", 500.0)),
        y_min=float(grid_raw.get("y_min", 0.0)),
        y_max=float(grid_raw.get("y_max", 500.0)),
        nx=int(grid_raw.get("nx", 64)),
        ny=int(grid_raw.get("ny", 64)),
    )

    plumes_list: list[PlumeConfig] = []
    for p_raw in raw.get("plumes", []):
        plumes_list.append(
            PlumeConfig(
                centers=p_raw.get("centers", []),
                widths=[float(w) for w in p_raw.get("widths", [])],
                amplitudes=[float(a) for a in p_raw.get("amplitudes", [])],
                num_random=int(p_raw.get("num_random", 0)),
                seed=p_raw.get("seed", None),
            )
        )

    return PipelineConfig(grid=grid_cfg, plumes=plumes_list)
