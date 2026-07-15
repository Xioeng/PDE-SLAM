"""
kinematics
==========
Robot kinematic models for trajectory integration and parameters identification.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)


from pde_slam.kinematics.base import BaseKinematics
from pde_slam.kinematics.unicycle import UnicycleKinematics

__all__: list[str] = [
    "BaseKinematics",
    "UnicycleKinematics",
]
