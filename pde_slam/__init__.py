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

from pde_slam.interpolator import FieldInterpolator, SpatialGrid
from pde_slam.joint_optimization import JointSlamOptimizer
from pde_slam.kinematics import BaseKinematics, UnicycleKinematics
from pde_slam.optimization import KinematicsOptimizer
from pde_slam.solver import AdvectionDiffusionSolver, PDEParams
from pde_slam.survey_loader import SurveyLoader

__all__: list[str] = [
    "__version__",
    "AdvectionDiffusionSolver",
    "BaseKinematics",
    "ENUFrame",
    "FieldInterpolator",
    "PDEParams",
    "SpatialGrid",
    "SurveyLoader",
    "UnicycleKinematics",
    "KinematicsOptimizer",
    "JointSlamOptimizer",
]
