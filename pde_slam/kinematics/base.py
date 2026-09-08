"""
base.py
=======
Abstract base class for robot kinematic models in PDE-SLAM.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from jax import Array


class BaseKinematics(ABC):
    """Abstract base class defining the standard interface for kinematics in PDE-SLAM.

    This ensures that any custom kinematics model is drop-in compatible with the rest
    of the codebase (such as the optimization module and pipeline).
    """

    @property
    @abstractmethod
    def state(self) -> Array:
        """Current state vector.

        Returns
        -------
        state : Array
            State vector of shape (D,).
        """
        pass

    @property
    @abstractmethod
    def x_m(self) -> float:
        """Current East position [m] in ENU frame.

        Returns
        -------
        x_m : float
            East coordinate.
        """
        pass

    @property
    @abstractmethod
    def y_m(self) -> float:
        """Current North position [m] in ENU frame.

        Returns
        -------
        y_m : float
            North coordinate.
        """
        pass

    @abstractmethod
    def step(self, *args, **kwargs) -> Array:
        """Integrate one time step.

        Returns
        -------
        state : Array
            Updated state vector.
        """
        pass

    @abstractmethod
    def reset(self, *args, **kwargs) -> None:
        """Reset the robot state to a new initial condition."""
        pass

    @abstractmethod
    def trajectory(self, *args, **kwargs) -> Array:
        """Integrate a full sequence of control commands.

        Returns
        -------
        states : Array
            Array of integrated states.
        """
        pass

    @staticmethod
    @abstractmethod
    def integrate_trajectory(
        x0: Array,
        *args,
        **kwargs,
    ) -> Array:
        """Differentiable integration of a trajectory.

        Parameters
        ----------
        x0 : Array
            Initial state/position vector.

        Returns
        -------
        states : Array
            Integrated states/positions array.
        """
        pass
