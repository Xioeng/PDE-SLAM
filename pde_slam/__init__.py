"""
pde_slam
========
Aquatic SLAM constrained by an advection-diffusion PDE.

Public API
----------
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version("pde-slam")
except PackageNotFoundError:
    __version__ = "0.0.0.dev"

from pde_slam.interpolator import FieldInterpolator, SpatialGrid
from pde_slam.solver import AdvectionDiffusionSolver, PDEParams

__all__: list[str] = [
    "__version__",
    "AdvectionDiffusionSolver",
    "FieldInterpolator",
    "PDEParams",
    "SpatialGrid",
]
