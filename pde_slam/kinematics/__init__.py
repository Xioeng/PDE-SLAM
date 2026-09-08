"""
kinematics
==========
Robot kinematic models for differential drive trajectory integration.
"""

from __future__ import annotations

import jax

from pde_slam.kinematics.base import BaseKinematics
from pde_slam.kinematics.dead_reckoning import (
    DeadReckoningEstimator,
    compute_position_eigenvalues,
)
from pde_slam.kinematics.diff_drive import DiffDriveKinematics

jax.config.update("jax_enable_x64", True)

__all__: list[str] = [
    "BaseKinematics",
    "DeadReckoningEstimator",
    "DiffDriveKinematics",
    "compute_position_eigenvalues",
]
