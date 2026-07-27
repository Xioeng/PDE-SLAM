"""
pde_slam
========
Aquatic SLAM constrained by an advection-diffusion PDE.

Public API
----------
"""

from importlib.metadata import PackageNotFoundError, version

from pde_slam.coords import ENUFrame

try:
    __version__: str = version("pde-slam")
except PackageNotFoundError:
    __version__ = "0.0.0.dev"

from pde_slam.interpolators import FieldInterpolator, SpatialGrid
from pde_slam.kinematics import BaseKinematics, UnicycleKinematics
from pde_slam.optimization import (
    JointSlamOptimizer,
    KinematicsOptimizer,
    MultiPdeSlamOptimizer,
)
from pde_slam.solver import AdvectionDiffusionSolver, PDEParams
from pde_slam.survey_loader import SurveyLoader
from pde_slam.types import ObservationData, TrajectoryContext
from pde_slam.config import (
    GridConfig,
    SolverConfig,
    PlumeConfig,
    PdeParamsConfig,
    OptimizationConfig,
    PipelineConfig,
    load_config,
)

__all__: list[str] = [
    "__version__",
    "AdvectionDiffusionSolver",
    "BaseKinematics",
    "ENUFrame",
    "FieldInterpolator",
    "JointSlamOptimizer",
    "KinematicsOptimizer",
    "MultiPdeSlamOptimizer",
    "ObservationData",
    "PDEParams",
    "SpatialGrid",
    "SurveyLoader",
    "TrajectoryContext",
    "UnicycleKinematics",
    "GridConfig",
    "SolverConfig",
    "PlumeConfig",
    "PdeParamsConfig",
    "OptimizationConfig",
    "PipelineConfig",
    "load_config",
]
