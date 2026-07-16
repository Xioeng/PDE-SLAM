"""
types.py
========
Shared data-contract types for PDE-SLAM optimizers.

These NamedTuples decouple data layout from any specific optimizer class
and can be reused across kinematics, joint, and future SLAM back-ends.
"""

from __future__ import annotations

from typing import NamedTuple

from jax import Array


class ObservationData(NamedTuple):
    """Container for experimental or simulated scalar field observations.

    Parameters
    ----------
    ts : Array
        Timestamps of the scalar observations, shape (M,).
    vals : Array
        Observed scalar values at the robot position for each timestamp.
        Shape (M,) for a single field or (M, K) for K concurrent fields.
    """

    ts: Array
    vals: Array


class TrajectoryContext(NamedTuple):
    """Container for robot control inputs and integrated time trajectory metadata.

    Parameters
    ----------
    thrusts : Array
        Dimensionless thrust commands in ``[0, 100]``, shape (N,).
    headings : Array
        Compass headings [rad, navigation convention], shape (N,).
    dt_arr : Array
        Per-step time increments [s], shape (N,).
    t_traj : Array
        Cumulative trajectory timestamps including t=0, shape (N+1,).
    t0 : float
        Start time of simulation [s].
    t_end : float
        End time of simulation [s].
    """

    thrusts: Array
    headings: Array
    dt_arr: Array
    t_traj: Array
    t0: float
    t_end: float
