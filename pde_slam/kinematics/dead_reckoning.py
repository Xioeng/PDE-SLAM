"""
dead_reckoning.py
=================
Dead reckoning state and covariance estimator for mobile robot kinematics.
Tracks pose uncertainty via first-order error propagation (EKF covariance
propagation) with analytical and JAX autodiff Jacobians.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array


@jax.jit
def _diff_drive_step(
    x: Array,
    u: Array,
    dt: float | Array,
) -> Array:
    """Pure differential drive state update.

    Parameters
    ----------
    x : Array
        Current state vector ``[x, y, heading]``, shape (3,).
    u : Array
        Control command vector ``[v, omega]``, shape (2,).
    dt : float or Array
        Time step [s].

    Returns
    -------
    Array
        Next state vector ``[x', y', heading']``, shape (3,).
    """
    v, omega = u[0], u[1]
    theta = x[2]
    dx = v * jnp.cos(theta) * dt
    dy = v * jnp.sin(theta) * dt
    dtheta = omega * dt
    return jnp.array([x[0] + dx, x[1] + dy, x[2] + dtheta], dtype=jnp.float64)


@jax.jit
def _diff_drive_jacobians(
    x: Array,
    u: Array,
    dt: float | Array,
) -> tuple[Array, Array]:
    """Analytical Jacobians F (d_f/d_x) and G (d_f/d_u) for differential drive.

    Parameters
    ----------
    x : Array
        State vector ``[x, y, theta]``.
    u : Array
        Control vector ``[v, omega]``.
    dt : float or Array
        Time step [s].

    Returns
    -------
    F : Array
        State transition Jacobian, shape (3, 3).
    G : Array
        Control input Jacobian, shape (3, 2).
    """
    v = u[0]
    theta = x[2]

    f_mat = jnp.array(
        [
            [1.0, 0.0, -v * jnp.sin(theta) * dt],
            [0.0, 1.0, v * jnp.cos(theta) * dt],
            [0.0, 0.0, 1.0],
        ],
        dtype=jnp.float64,
    )

    g_mat = jnp.array(
        [
            [jnp.cos(theta) * dt, 0.0],
            [jnp.sin(theta) * dt, 0.0],
            [0.0, dt],
        ],
        dtype=jnp.float64,
    )

    return f_mat, g_mat


@jax.jit
def _propagate_covariance_step(
    x: Array,
    sigma: Array,
    u: Array,
    q_u: Array,
    dt: float | Array,
) -> tuple[Array, Array]:
    """Pure JIT single-step propagation of state and error covariance.

    Parameters
    ----------
    x : Array
        Current state ``[x, y, theta]``, shape (3,).
    sigma : Array
        Current covariance matrix, shape (3, 3).
    u : Array
        Control vector ``[v, omega]``, shape (2,).
    q_u : Array
        Control noise covariance matrix, shape (2, 2).
    dt : float or Array
        Time step [s].

    Returns
    -------
    next_x : Array
        Updated state ``[x', y', theta']``, shape (3,).
    next_sigma : Array
        Updated covariance matrix, shape (3, 3).
    """
    f_mat, g_mat = _diff_drive_jacobians(x, u, dt)
    next_x = _diff_drive_step(x, u, dt)
    next_sigma = f_mat @ sigma @ f_mat.T + g_mat @ q_u @ g_mat.T
    # Ensure numerical symmetry and positive semi-definiteness
    next_sigma = 0.5 * (next_sigma + next_sigma.T)
    return next_x, next_sigma


@jax.jit
def compute_position_eigenvalues(sigma_2d: Array) -> tuple[Array, Array]:
    """Calculate the eigenvalues (lambda_min, lambda_max) of a 2x2 covariance.

    Parameters
    ----------
    sigma_2d : Array
        2x2 symmetric spatial covariance matrix.

    Returns
    -------
    lambda_min : Array
        Smallest eigenvalue.
    lambda_max : Array
        Largest eigenvalue (semi-major axis squared).
    """
    s00 = sigma_2d[0, 0]
    s11 = sigma_2d[1, 1]
    s01 = sigma_2d[0, 1]

    trace = s00 + s11
    disc = jnp.sqrt(jnp.maximum(0.0, (s00 - s11) ** 2 + 4.0 * (s01**2)))

    lambda_max = 0.5 * (trace + disc)
    lambda_min = 0.5 * (trace - disc)
    return lambda_min, lambda_max


class DeadReckoningEstimator:
    """Dead reckoning state and uncertainty covariance estimator.

    Parameters
    ----------
    x0 : float, default=0.0
        Initial x position [m].
    y0 : float, default=0.0
        Initial y position [m].
    heading0 : float, default=0.0
        Initial heading [rad].
    sigma0 : Array or None, default=None
        Initial 3x3 error covariance. Defaults to zeros.
    q_u : Array or None, default=None
        2x2 control noise covariance matrix diag(sigma_v^2, sigma_omega^2).
    """

    def __init__(
        self,
        x0: float = 0.0,
        y0: float = 0.0,
        heading0: float = 0.0,
        sigma0: Array | None = None,
        q_u: Array | None = None,
    ) -> None:
        super().__init__()
        self._state: Array = jnp.array(
            [float(x0), float(y0), float(heading0)], dtype=jnp.float64
        )

        if sigma0 is not None:
            self._sigma: Array = jnp.asarray(sigma0, dtype=jnp.float64)
            if self._sigma.shape != (3, 3):
                raise ValueError(
                    f"sigma0 must have shape (3, 3), got {self._sigma.shape}"
                )
        else:
            self._sigma = jnp.zeros((3, 3), dtype=jnp.float64)

        if q_u is not None:
            self._q_u: Array = jnp.asarray(q_u, dtype=jnp.float64)
            if self._q_u.shape != (2, 2):
                raise ValueError(f"q_u must have shape (2, 2), got {self._q_u.shape}")
        else:
            self._q_u = jnp.zeros((2, 2), dtype=jnp.float64)

    @property
    def x_m(self) -> float:
        """Current x position [m]."""
        return float(self._state[0])

    @property
    def y_m(self) -> float:
        """Current y position [m]."""
        return float(self._state[1])

    @property
    def heading_rad(self) -> float:
        """Current heading [rad]."""
        return float(self._state[2])

    @property
    def state(self) -> Array:
        """Current state vector ``[x_m, y_m, heading_rad]``."""
        return self._state.copy()

    @property
    def covariance(self) -> Array:
        """Current 3x3 error covariance matrix."""
        return self._sigma.copy()

    @property
    def position_covariance(self) -> Array:
        """Current 2x2 position covariance matrix."""
        return self._sigma[:2, :2].copy()

    @property
    def position_variance(self) -> float:
        """Trace of the 2D position covariance: Var(x) + Var(y) [m^2]."""
        return float(self._sigma[0, 0] + self._sigma[1, 1])

    @property
    def position_std(self) -> float:
        """Standard deviation of position error: sqrt(Var(x) + Var(y)) [m]."""
        return float(jnp.sqrt(jnp.maximum(0.0, self._sigma[0, 0] + self._sigma[1, 1])))

    @property
    def max_eigenvalue(self) -> float:
        """Largest eigenvalue of the 2D spatial covariance matrix [m^2].

        Corresponds to the squared semi-major axis of the 2D uncertainty ellipse.
        """
        _, lambda_max = compute_position_eigenvalues(self._sigma[:2, :2])
        return float(lambda_max)

    @property
    def max_std(self) -> float:
        """Maximum directional standard deviation (sqrt(lambda_max)) [m]."""
        return float(jnp.sqrt(jnp.maximum(0.0, self.max_eigenvalue)))

    @property
    def heading_variance(self) -> float:
        """Variance of heading error Var(theta) [rad^2]."""
        return float(self._sigma[2, 2])

    @property
    def heading_std(self) -> float:
        """Standard deviation of heading error [rad]."""
        return float(jnp.sqrt(jnp.maximum(0.0, self._sigma[2, 2])))

    @property
    def q_u(self) -> Array:
        """Control noise covariance matrix (2, 2)."""
        return self._q_u.copy()

    @q_u.setter
    def q_u(self, value: Array) -> None:
        val = jnp.asarray(value, dtype=jnp.float64)
        if val.shape != (2, 2):
            raise ValueError(f"q_u must have shape (2, 2), got {val.shape}")
        self._q_u = val

    def step(self, v: float, omega: float, dt: float) -> Array:
        """Integrate one time step and propagate covariance.

        Parameters
        ----------
        v : float
            Linear velocity [m/s].
        omega : float
            Angular velocity [rad/s].
        dt : float
            Time step [s].

        Returns
        -------
        state : Array
            Updated state vector ``[x_m, y_m, heading_rad]``.
        """
        u = jnp.array([float(v), float(omega)], dtype=jnp.float64)
        self._state, self._sigma = _propagate_covariance_step(
            self._state, self._sigma, u, self._q_u, float(dt)
        )
        return self._state.copy()

    def reset(
        self,
        x0: float = 0.0,
        y0: float = 0.0,
        heading0: float = 0.0,
        sigma0: Array | None = None,
    ) -> None:
        """Reset state and covariance."""
        self._state = jnp.array(
            [float(x0), float(y0), float(heading0)], dtype=jnp.float64
        )
        if sigma0 is not None:
            self._sigma = jnp.asarray(sigma0, dtype=jnp.float64)
        else:
            self._sigma = jnp.zeros((3, 3), dtype=jnp.float64)

    def __repr__(self) -> str:
        return (
            f"DeadReckoningEstimator("
            f"x={self.x_m:.3f} m, y={self.y_m:.3f} m, "
            f"heading={float(jnp.degrees(self.heading_rad)):.1f}°, "
            f"max_std={self.max_std:.3f} m, "
            f"pos_std={self.position_std:.3f} m)"
        )
