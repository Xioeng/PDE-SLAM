"""
optimizer.py
============
Multi-objective joint loss assembly and Phase 2 SLAM parameter update loop.

Loss architecture
-----------------
The total loss is a weighted sum of three objectives:

.. math::

    \\mathcal{L}_{\\text{total}} = w_d \\mathcal{L}_{\\text{data}}
        + w_p \\mathcal{L}_{\\text{pde}}
        + w_r \\mathcal{L}_{\\text{reg}}

Where:

``L_data`` – **Data alignment loss** (Phase 2a – trajectory correction)
    Mean squared error between the field value predicted at the
    drift-corrected sensor positions and the actual sensor readings.
    Encourages the trajectory latent ``deltax`` to place measurements
    at self-consistent field locations.

``L_pde`` – **PDE verification loss** (Phase 2b – PDE parameter update)
    MSE between the PDE-evolved field ``φ(t + Δt)`` and the measured
    field values at collocation points.  Penalises PDE parameter sets
    whose predicted dynamics disagree with observations.

``L_reg`` – **Motion regularisation loss** (Phase 2a + 2b)
    L2 (Tikhonov) penalty on the trajectory drift latents ``deltax``
    and on the magnitude of the advection velocity field ``u``.
    Prevents degenerate solutions.

Decoupled optimisation phases
------------------------------
Following the SLAM literature (see iSAM2, ElasticFusion), the optimiser
alternates:

1. **Trajectory step** (fixed PDE params) – update ``deltax`` using
   ``L_data + L_reg`` gradient.
2. **PDE step** (fixed trajectory) – update ``(u_field, D)`` using
   ``L_pde + L_reg`` gradient.

Both sub-optimisers use ``optax`` gradient transformations (default: Adam).

Interaction with other modules
------------------------------
* Receives ``corrected_poses`` from :mod:`pde_slam.kinematics`.
* Calls :func:`pde_slam.solver.solve_pde` for the PDE loss forward pass.
* Reads mini-batches from :class:`pde_slam.data_pipeline.SpatialReplayBuffer`.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax
from jax import Array

from pde_slam.solver import PDEParams, solve_pde


# ---------------------------------------------------------------------------
# Loss weight container
# ---------------------------------------------------------------------------


class LossWeights(NamedTuple):
    """Hyper-parameters controlling the multi-objective loss balance.

    Attributes
    ----------
    data :
        Weight for the sensor data alignment term.
    pde :
        Weight for the PDE residual verification term.
    reg_trajectory :
        L2 regularisation weight on the trajectory drift latents.
    reg_velocity :
        L2 regularisation weight on the advection velocity field magnitude.
    """

    data: float = 1.0
    pde: float = 1.0
    reg_trajectory: float = 1e-3
    reg_velocity: float = 1e-4


# ---------------------------------------------------------------------------
# Optimiser state containers
# ---------------------------------------------------------------------------


class TrajectoryState(NamedTuple):
    """State for Phase 2a trajectory optimiser.

    Attributes
    ----------
    deltax :
        Drift correction latents of shape ``(N_poses, 3)`` – ``[δx, δy, δθ]``.
    opt_state :
        ``optax`` optimiser state for *deltax*.
    """

    deltax: Array
    opt_state: Any


class PDEState(NamedTuple):
    """State for Phase 2b PDE parameter optimiser.

    Attributes
    ----------
    pde_params :
        :class:`~pde_slam.solver.PDEParams` – learnable ``u_field`` and ``D``.
    opt_state :
        ``optax`` optimiser state for *pde_params*.
    """

    pde_params: PDEParams
    opt_state: Any


# ---------------------------------------------------------------------------
# Loss components
# ---------------------------------------------------------------------------


def data_alignment_loss(
    deltax: Array,
    nominal_poses: Array,
    phi_grid: Array,
    grid_x_range: tuple[float, float],
    grid_y_range: tuple[float, float],
    obs_xy_enu: Array,
    obs_values: Array,
) -> Array:
    """Sensor data alignment loss (Phase 2a).

    Bilinearly interpolates the current field estimate ``phi_grid`` at the
    drift-corrected sensor positions and computes MSE against the raw
    sensor readings.

    Parameters
    ----------
    deltax :
        Trajectory drift latents of shape ``(N_poses, 3)``.
    nominal_poses :
        Nominal (dead-reckoned) poses of shape ``(N_poses, 3)``.
    phi_grid :
        Current scalar field estimate of shape ``(ny, nx)``.
    grid_x_range, grid_y_range :
        Domain extents ``(min, max)`` [m] used for bilinear lookup.
    obs_xy_enu :
        Sensor positions (ENU) of shape ``(N_obs, 2)`` in the *nominal*
        frame.  Drift correction offsets are applied internally.
    obs_values :
        Observed scalar field values of shape ``(N_obs,)``.

    Returns
    -------
    loss :
        Scalar data alignment MSE.
    """
    # Corrected positions (only x, y channels; theta not needed here)
    corrected_poses = nominal_poses + deltax
    # Broadcast drift correction to observations (simplified: assume one pose
    # per observation; a full implementation would index per timestamp)
    # For the skeleton we add the mean trajectory correction to all obs.
    mean_correction = jnp.mean(corrected_poses[:, :2] - nominal_poses[:, :2], axis=0)
    corrected_xy = obs_xy_enu + mean_correction  # (N_obs, 2)

    predicted = _bilinear_sample(phi_grid, corrected_xy, grid_x_range, grid_y_range)
    return jnp.mean((predicted - obs_values) ** 2)


def pde_verification_loss(
    pde_params: PDEParams,
    phi_current: Array,
    phi_observed_next: Array,
    dx: float,
    dy: float,
    dt: float,
) -> Array:
    """PDE residual verification loss (Phase 2b).

    Evolves the current field one step forward using the candidate PDE
    parameters and measures MSE against the next observed field snapshot.

    Parameters
    ----------
    pde_params :
        Candidate :class:`~pde_slam.solver.PDEParams`.
    phi_current :
        Current observed/interpolated field of shape ``(ny, nx)``.
    phi_observed_next :
        Field values observed at time ``t + dt``, shape ``(ny, nx)``.
    dx, dy :
        Grid spacings [m].
    dt :
        Time step between the two field snapshots [s].

    Returns
    -------
    loss :
        Scalar PDE verification MSE.
    """
    phi_predicted = solve_pde(
        phi_current, pde_params, dx, dy, t0=0.0, t_end=dt, dt_max=dt
    )
    return jnp.mean((phi_predicted - phi_observed_next) ** 2)


def motion_regularisation_loss(
    deltax: Array,
    pde_params: PDEParams,
    weights: LossWeights,
) -> Array:
    """Motion and parameter regularisation loss.

    Parameters
    ----------
    deltax :
        Trajectory drift latents of shape ``(N_poses, 3)``.
    pde_params :
        Current PDE parameters.
    weights :
        :class:`LossWeights` specifying regularisation coefficients.

    Returns
    -------
    loss :
        Scalar regularisation loss.
    """
    traj_reg = weights.reg_trajectory * jnp.mean(deltax**2)
    vel_reg = weights.reg_velocity * jnp.mean(pde_params.u_field**2)
    return traj_reg + vel_reg


def total_loss(
    deltax: Array,
    pde_params: PDEParams,
    nominal_poses: Array,
    phi_current: Array,
    phi_next: Array,
    obs_xy_enu: Array,
    obs_values: Array,
    dx: float,
    dy: float,
    dt: float,
    grid_x_range: tuple[float, float],
    grid_y_range: tuple[float, float],
    weights: LossWeights,
) -> Array:
    """Compute the full multi-objective loss.

    This function is the primary target for ``jax.grad`` in both Phase 2
    sub-steps.

    Returns
    -------
    loss_total :
        Scalar total loss value.
    """
    l_data = data_alignment_loss(
        deltax, nominal_poses, phi_current,
        grid_x_range, grid_y_range, obs_xy_enu, obs_values,
    )
    l_pde = pde_verification_loss(pde_params, phi_current, phi_next, dx, dy, dt)
    l_reg = motion_regularisation_loss(deltax, pde_params, weights)

    return weights.data * l_data + weights.pde * l_pde + l_reg


# ---------------------------------------------------------------------------
# Optimiser initialisation
# ---------------------------------------------------------------------------


def make_trajectory_optimiser(
    deltax_init: Array,
    learning_rate: float = 1e-3,
    b1: float = 0.9,
    b2: float = 0.999,
) -> tuple[TrajectoryState, optax.GradientTransformation]:
    """Initialise the Phase 2a trajectory optimiser.

    Parameters
    ----------
    deltax_init :
        Initial drift latents (typically all-zeros).
    learning_rate :
        Adam step size.
    b1, b2 :
        Adam moment decay rates.

    Returns
    -------
    state :
        :class:`TrajectoryState` with zeroed optimiser state.
    tx :
        ``optax`` gradient transformation (Adam + gradient clipping).
    """
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate, b1=b1, b2=b2),
    )
    opt_state = tx.init(deltax_init)
    return TrajectoryState(deltax=deltax_init, opt_state=opt_state), tx


def make_pde_optimiser(
    pde_params_init: PDEParams,
    learning_rate: float = 1e-3,
    b1: float = 0.9,
    b2: float = 0.999,
) -> tuple[PDEState, optax.GradientTransformation]:
    """Initialise the Phase 2b PDE parameter optimiser.

    Parameters
    ----------
    pde_params_init :
        Initial PDE parameters.
    learning_rate :
        Adam step size.
    b1, b2 :
        Adam moment decay rates.

    Returns
    -------
    state :
        :class:`PDEState` with zeroed optimiser state.
    tx :
        ``optax`` gradient transformation.
    """
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate, b1=b1, b2=b2),
    )
    opt_state = tx.init(pde_params_init)
    return PDEState(pde_params=pde_params_init, opt_state=opt_state), tx


# ---------------------------------------------------------------------------
# Single gradient-update steps
# ---------------------------------------------------------------------------


def trajectory_step(
    traj_state: TrajectoryState,
    tx: optax.GradientTransformation,
    grad_fn: Any,
    *grad_args: Any,
) -> tuple[TrajectoryState, Array]:
    """Apply one gradient step to the trajectory drift latents.

    Parameters
    ----------
    traj_state :
        Current :class:`TrajectoryState`.
    tx :
        ``optax`` gradient transformation.
    grad_fn :
        Function that returns ``(loss, grad_deltax)`` given
        ``(deltax, *grad_args)``.
    *grad_args :
        Additional positional arguments forwarded to *grad_fn*.

    Returns
    -------
    new_state :
        Updated :class:`TrajectoryState`.
    loss :
        Scalar loss value before the update.
    """
    loss, grads = grad_fn(traj_state.deltax, *grad_args)
    updates, new_opt_state = tx.update(grads, traj_state.opt_state, traj_state.deltax)
    new_deltax = optax.apply_updates(traj_state.deltax, updates)
    return TrajectoryState(deltax=new_deltax, opt_state=new_opt_state), loss


def pde_step(
    pde_state: PDEState,
    tx: optax.GradientTransformation,
    grad_fn: Any,
    *grad_args: Any,
) -> tuple[PDEState, Array]:
    """Apply one gradient step to the PDE parameters.

    Parameters
    ----------
    pde_state :
        Current :class:`PDEState`.
    tx :
        ``optax`` gradient transformation.
    grad_fn :
        Function that returns ``(loss, grad_pde_params)`` given
        ``(pde_params, *grad_args)``.
    *grad_args :
        Additional positional arguments forwarded to *grad_fn*.

    Returns
    -------
    new_state :
        Updated :class:`PDEState`.
    loss :
        Scalar loss value before the update.
    """
    loss, grads = grad_fn(pde_state.pde_params, *grad_args)
    updates, new_opt_state = tx.update(
        grads, pde_state.opt_state, pde_state.pde_params
    )
    new_params = optax.apply_updates(pde_state.pde_params, updates)
    return PDEState(pde_params=new_params, opt_state=new_opt_state), loss


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------


def _bilinear_sample(
    field: Array,
    xy_enu: Array,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> Array:
    """Differentiable bilinear interpolation of *field* at *xy_enu* points.

    Parameters
    ----------
    field :
        Scalar field of shape ``(ny, nx)``.
    xy_enu :
        Query ENU positions of shape ``(N, 2)``.
    x_range, y_range :
        Domain extents ``(min, max)`` [m].

    Returns
    -------
    values :
        Interpolated values of shape ``(N,)``.
    """
    ny, nx = field.shape
    x_min, x_max = x_range
    y_min, y_max = y_range

    # Normalise to [0, nx-1] and [0, ny-1]
    ix = (xy_enu[:, 0] - x_min) / (x_max - x_min) * (nx - 1)
    iy = (xy_enu[:, 1] - y_min) / (y_max - y_min) * (ny - 1)

    ix = jnp.clip(ix, 0, nx - 1 - 1e-6)
    iy = jnp.clip(iy, 0, ny - 1 - 1e-6)

    ix0 = jnp.floor(ix).astype(jnp.int32)
    iy0 = jnp.floor(iy).astype(jnp.int32)
    ix1 = ix0 + 1
    iy1 = iy0 + 1

    dx = ix - ix0
    dy = iy - iy0

    f00 = field[iy0, ix0]
    f10 = field[iy0, ix1]
    f01 = field[iy1, ix0]
    f11 = field[iy1, ix1]

    return (
        f00 * (1 - dx) * (1 - dy)
        + f10 * dx * (1 - dy)
        + f01 * (1 - dx) * dy
        + f11 * dx * dy
    )
