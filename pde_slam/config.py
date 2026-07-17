"""
config.py
=========
YAML parser and configuration classes for PDE-SLAM pipeline parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml  # type: ignore[import-untyped]


@dataclass
class GridConfig:
    """Configuration parameters for the SpatialGrid.

    Parameters
    ----------
    x_min : float
        Minimum east coordinate [m].
    x_max : float
        Maximum east coordinate [m].
    y_min : float
        Minimum north coordinate [m].
    y_max : float
        Maximum north coordinate [m].
    nx : int
        Number of grid intervals in east direction.
    ny : int
        Number of grid intervals in north direction.
    """

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    nx: int
    ny: int


@dataclass
class SolverConfig:
    """Configuration parameters for the PDE solver.

    Parameters
    ----------
    dt_max : float
        Maximum integration step size [s].
    """

    dt_max: float = 1.0


@dataclass
class PlumeConfig:
    """Configuration parameters for generating water feature plume(s).

    Parameters
    ----------
    centers : list of list of float
        A list of [x, y] coordinates for each plume.
    widths : list of float
        A list of standard deviation/width for each plume.
    amplitudes : list of float
        A list of amplitudes/strengths for each plume.
    num_random : int
        Number of random plumes to generate if centers are empty.
    seed : int or None
        Random seed for generating random plumes.
    """

    centers: list[list[float]] = field(default_factory=list)
    widths: list[float] = field(default_factory=list)
    amplitudes: list[float] = field(default_factory=list)
    num_random: int = 0
    seed: int | None = None


@dataclass
class PdeParamsConfig:
    """Ground truth PDE physical and kinematic parameters.

    Parameters
    ----------
    D : list of float
        Diffusivity coefficient for each passive scalar field [m^2 s^-1].
    v_flow : list of float
        Constant fluid velocity vector [u_east, u_north] in m s^-1.
    k_thrust : float
        Thrust-to-speed conversion factor.
    """

    D: list[float]  # diffusivity
    v_flow: list[float]  # velocity field [u, v]
    k_thrust: float


@dataclass
class OptimizationConfig:
    """Parameters for the kinematics and joint SLAM optimization.

    Parameters
    ----------
    lambda_reg : float
        Regularization penalty on trajectory corrections.
    method : str
        Optimization method ('l-bfgs-b' or 'adam').
    maxiter : int
        Maximum number of iterations for L-BFGS-B.
    learning_rate : float
        Learning rate for Adam optimizer.
    num_steps : int
        Number of training steps for Adam optimizer.
    """

    lambda_reg: float = 1e-2
    method: str = "l-bfgs-b"
    maxiter: int = 100
    learning_rate: float = 0.05
    num_steps: int = 100


@dataclass
class PipelineConfig:
    """Root configuration object mapping the entire pipeline structure."""

    grid: GridConfig
    solver: SolverConfig
    plumes: PlumeConfig
    pde_params: PdeParamsConfig
    optimization: OptimizationConfig


def load_config(path: str | Path) -> PipelineConfig:
    """Load and parse a YAML file into a PipelineConfig object.

    Parameters
    ----------
    path : str or Path
        Path to the configuration YAML file.

    Returns
    -------
    config : PipelineConfig
        The parsed strongly-typed configuration.

    Raises
    ------
    ValueError
        If there are missing or invalid keys in the configuration.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Configuration file not found: {path_obj}")

    with open(path_obj) as f:
        try:
            raw = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse YAML file: {e}") from e

    if raw is None or not isinstance(raw, dict):
        raise ValueError("Configuration YAML must resolve to a dictionary.")

    try:
        grid_data = raw["grid"]
        grid = GridConfig(
            x_min=float(grid_data["x_min"]),
            x_max=float(grid_data["x_max"]),
            y_min=float(grid_data["y_min"]),
            y_max=float(grid_data["y_max"]),
            nx=int(grid_data["nx"]),
            ny=int(grid_data["ny"]),
        )

        solver_data = raw.get("solver", {})
        solver = SolverConfig(
            dt_max=float(solver_data.get("dt_max", 1.0)),
        )

        plumes_data = raw.get("plumes", {})
        plumes = PlumeConfig(
            centers=plumes_data.get("centers", []),
            widths=plumes_data.get("widths", []),
            amplitudes=plumes_data.get("amplitudes", []),
            num_random=int(plumes_data.get("num_random", 0)),
            seed=plumes_data.get("seed")
            if plumes_data.get("seed") is None
            else int(plumes_data["seed"]),
        )

        pde_data = raw["pde_params"]
        pde_params = PdeParamsConfig(
            D=[float(d) for d in pde_data["D"]],
            v_flow=[float(v) for v in pde_data["v_flow"]],
            k_thrust=float(pde_data["k_thrust"]),
        )

        opt_data = raw.get("optimization", {})
        optimization = OptimizationConfig(
            lambda_reg=float(opt_data.get("lambda_reg", 1e-2)),
            method=str(opt_data.get("method", "l-bfgs-b")),
            maxiter=int(opt_data.get("maxiter", 100)),
            learning_rate=float(opt_data.get("learning_rate", 0.05)),
            num_steps=int(opt_data.get("num_steps", 100)),
        )

        return PipelineConfig(
            grid=grid,
            solver=solver,
            plumes=plumes,
            pde_params=pde_params,
            optimization=optimization,
        )

    except KeyError as e:
        raise ValueError(f"Missing required configuration key: {e}") from e
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid configuration value type: {e}") from e
