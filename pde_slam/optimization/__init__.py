"""
optimization
============
JAX-based parameter identification and SLAM optimization modules.
"""

from __future__ import annotations

from pde_slam.optimization.kinematics import (
    KinematicsOptimizer,
    _pack_params,
    _unpack_params,
    unicycle_trajectory_fn,
)
from pde_slam.optimization.rbpf import RbpfSlam, RbpfState

__all__ = [
    "KinematicsOptimizer",
    "RbpfSlam",
    "RbpfState",
    "unicycle_trajectory_fn",
    "_pack_params",
    "_unpack_params",
]
