"""
main.py
=======
Top-level orchestrator for the PDE-SLAM two-phase pipeline.

Phase 1 – Survey & Initialization
----------------------------------
1. Load raw sensor log from ``data/raw/``.
2. Parse log → DataFrame → ENU positions via kinematic dead-reckoning.
3. Build the Sensor Measurement Pool and the PDE Collocation Pool.
4. Interpolate scattered measurements onto the computational grid to
   produce the initial condition ``u0`` for the PDE.
5. Initialise learnable parameters: ``deltax``, ``u_field``, ``D``.

Phase 2 – Decoupled Online SLAM Optimization Loop
---------------------------------------------------
Alternating gradient descent sub-steps:

    for epoch in range(n_epochs):
        # 2a  Trajectory correction (fixed PDE params)
        for k in range(n_traj_steps):
            sample mini-batch from SpatialReplayBuffer
            compute grad(L_data + L_reg) w.r.t. deltax
            update deltax via Adam

        # 2b  PDE parameter update (fixed trajectory)
        for k in range(n_pde_steps):
            compute grad(L_pde + L_reg) w.r.t. (u_field, D)
            update (u_field, D) via Adam

        log metrics / checkpoint

Entry points
------------
* ``run(cfg)`` – programmatic entry point.
* ``cli_entry()`` – console-script entry point (installed as ``pde-slam``).

Configuration
-------------
All hyper-parameters are managed through an :class:`omegaconf.DictConfig`
object.  See ``configs/default.yaml`` for the canonical parameter file.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from pde_slam.data_pipeline import (
    SpatialReplayBuffer,
    build_collocation_pool,
    build_measurement_pool,
    parse_log_strings,
    SCALAR_FIELDS,
)
from pde_slam.interpolator import (
    InterpolationMethod,
    SpatialGrid,
    build_initial_condition,
)
from pde_slam.kinematics import (
    GeoOrigin,
    KinematicParams,
    apply_drift_correction,
    integrate_trajectory,
    latlon_to_enu,
)
from pde_slam.optimizer import (
    LossWeights,
    PDEState,
    TrajectoryState,
    make_pde_optimiser,
    make_trajectory_optimiser,
    pde_verification_loss,
    data_alignment_loss,
    motion_regularisation_loss,
)
from pde_slam.solver import PDEParams, solve_pde

logger = logging.getLogger("pde_slam.main")


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_CFG: dict[str, Any] = {
    "data": {
        "raw_log": "data/raw/survey.csv",
        "field_key": "salinity_psu",
    },
    "grid": {
        "x_min": -500.0,  # [m] ENU East
        "x_max":  500.0,
        "y_min": -500.0,  # [m] ENU North
        "y_max":  500.0,
        "nx": 64,
        "ny": 64,
    },
    "kinematics": {
        "c_v": 0.85,
        "c_omega": 0.40,
        "dt": 0.5,          # integration step [s]
        "origin_lat": 36.8, # WGS-84 reference point
        "origin_lon": -76.0,
    },
    "phase1": {
        "interp_method": "rbf",  # "rbf" or "spline"
    },
    "phase2": {
        "n_epochs": 100,
        "n_traj_steps": 5,   # trajectory sub-steps per epoch
        "n_pde_steps": 5,    # PDE sub-steps per epoch
        "batch_size": 128,
        "replay_capacity": 10_000,
        "pde_dt": 5.0,       # prediction horizon for PDE loss [s]
        "lr_traj": 1e-3,
        "lr_pde": 1e-4,
        "loss_weights": {
            "data": 1.0,
            "pde": 1.0,
            "reg_trajectory": 1e-3,
            "reg_velocity": 1e-4,
        },
    },
    "output_dir": "outputs/",
    "seed": 42,
    "log_level": "INFO",
}


# ---------------------------------------------------------------------------
# Phase 1 helpers
# ---------------------------------------------------------------------------


def _load_and_parse(cfg: DictConfig) -> tuple[Any, np.ndarray]:
    """Load raw log and compute ENU trajectory.

    Returns
    -------
    df :
        Parsed log DataFrame.
    xy_enu :
        ENU positions of shape ``(N, 2)`` corresponding to log rows.
    """
    log_path = Path(cfg.data.raw_log)
    logger.info("Loading raw log from %s", log_path)
    with open(log_path) as fh:
        lines = fh.readlines()
    df = parse_log_strings(lines)

    origin = GeoOrigin(
        lat_deg=cfg.kinematics.origin_lat,
        lon_deg=cfg.kinematics.origin_lon,
    )
    east_m, north_m = latlon_to_enu(
        jnp.array(df["lat_deg"].to_numpy()),
        jnp.array(df["lon_deg"].to_numpy()),
        origin,
    )
    xy_enu = np.column_stack([np.array(east_m), np.array(north_m)])
    logger.info("Parsed %d measurement epochs.", len(df))
    return df, xy_enu


def _run_phase1(
    df: Any,
    xy_enu: np.ndarray,
    cfg: DictConfig,
) -> tuple[jnp.ndarray, SpatialGrid, SpatialReplayBuffer]:
    """Execute Phase 1: interpolate initial condition, seed replay buffer.

    Returns
    -------
    phi0 :
        Initial condition array of shape ``(ny, nx)``.
    grid :
        Computational :class:`~pde_slam.interpolator.SpatialGrid`.
    replay_buffer :
        Pre-populated :class:`~pde_slam.data_pipeline.SpatialReplayBuffer`.
    """
    logger.info("=== Phase 1: Survey & Initialization ===")

    grid = SpatialGrid(
        x_min=cfg.grid.x_min,
        x_max=cfg.grid.x_max,
        y_min=cfg.grid.y_min,
        y_max=cfg.grid.y_max,
        nx=cfg.grid.nx,
        ny=cfg.grid.ny,
    )
    logger.info("Grid: %s", grid)

    pool = build_measurement_pool(df, xy_enu, field_keys=SCALAR_FIELDS)

    method = InterpolationMethod[cfg.phase1.interp_method.upper()]
    phi0 = build_initial_condition(pool, cfg.data.field_key, grid, method=method)
    logger.info(
        "Initial condition interpolated: shape=%s, range=[%.3f, %.3f]",
        phi0.shape, float(phi0.min()), float(phi0.max()),
    )

    # Seed the replay buffer with Phase 1 observations
    replay = SpatialReplayBuffer(
        capacity=cfg.phase2.replay_capacity,
        seed=cfg.seed,
    )
    field_arrays = {k: pool[k] for k in SCALAR_FIELDS if k in pool}
    replay.add_batch(pool["timestamp_s"], pool["xy_enu"], field_arrays)
    logger.info("Replay buffer seeded: %d entries.", len(replay))

    return phi0, grid, replay


# ---------------------------------------------------------------------------
# Phase 2 helpers
# ---------------------------------------------------------------------------


def _initialise_phase2_params(
    df: Any,
    cfg: DictConfig,
    grid: SpatialGrid,
    key: jax.Array,
) -> tuple[jnp.ndarray, jnp.ndarray, PDEParams, jnp.ndarray, KinematicParams]:
    """Initialise all learnable parameters for Phase 2.

    Returns
    -------
    nominal_poses :
        Dead-reckoned trajectory ``(N+1, 3)``.
    deltax :
        Zero-initialised drift latents ``(N+1, 3)``.
    pde_params :
        Initial :class:`~pde_slam.solver.PDEParams` (small random velocity).
    controls :
        Control inputs array ``(N, 2)``.
    kin_params :
        :class:`~pde_slam.kinematics.KinematicParams`.
    """
    kin_params = KinematicParams(
        c_v=cfg.kinematics.c_v,
        c_omega=cfg.kinematics.c_omega,
    )
    controls = jnp.array(
        df[["thrust", "rudder"]].to_numpy(), dtype=jnp.float32
    )
    pose0 = jnp.zeros(3, dtype=jnp.float32)
    nominal_poses = integrate_trajectory(pose0, controls, cfg.kinematics.dt, kin_params)
    deltax = jnp.zeros_like(nominal_poses)

    # Small random initial velocity field, zero diffusivity
    key_u, _ = jax.random.split(key)
    u_field_init = jax.random.normal(key_u, shape=(grid.ny, grid.nx, 2)) * 0.01
    D_init = jnp.array(1e-3, dtype=jnp.float32)
    pde_params = PDEParams(u_field=u_field_init, D=D_init)

    return nominal_poses, deltax, pde_params, controls, kin_params


def _run_phase2(
    phi0: jnp.ndarray,
    nominal_poses: jnp.ndarray,
    deltax_init: jnp.ndarray,
    pde_params_init: PDEParams,
    replay: SpatialReplayBuffer,
    grid: SpatialGrid,
    cfg: DictConfig,
) -> tuple[TrajectoryState, PDEState]:
    """Execute Phase 2: decoupled SLAM optimisation loop.

    Returns
    -------
    traj_state, pde_state :
        Final optimiser states after all epochs.
    """
    logger.info("=== Phase 2: Decoupled Online SLAM Optimization ===")

    weights = LossWeights(**cfg.phase2.loss_weights)
    x_range = (cfg.grid.x_min, cfg.grid.x_max)
    y_range = (cfg.grid.y_min, cfg.grid.y_max)
    dt_pde = cfg.phase2.pde_dt

    traj_state, tx_traj = make_trajectory_optimiser(
        deltax_init, learning_rate=cfg.phase2.lr_traj
    )
    pde_state, tx_pde = make_pde_optimiser(
        pde_params_init, learning_rate=cfg.phase2.lr_pde
    )

    # JIT-compile gradient functions for efficiency
    @jax.jit
    def _grad_traj(deltax: jnp.ndarray, obs_xy: jnp.ndarray, obs_vals: jnp.ndarray) -> tuple:
        def loss_fn(dx: jnp.ndarray) -> jnp.ndarray:
            l_d = data_alignment_loss(
                dx, nominal_poses, phi0, x_range, y_range, obs_xy, obs_vals
            )
            l_r = motion_regularisation_loss(dx, pde_state.pde_params, weights)
            return weights.data * l_d + l_r
        return jax.value_and_grad(loss_fn)(deltax)

    @jax.jit
    def _grad_pde(params: PDEParams) -> tuple:
        def loss_fn(p: PDEParams) -> jnp.ndarray:
            corrected_poses = apply_drift_correction(nominal_poses, traj_state.deltax)
            # For brevity: use phi0 as both current and next snapshot.
            # In production, phi_next would be the interpolated field at t+dt.
            l_p = pde_verification_loss(p, phi0, phi0, grid.dx, grid.dy, dt_pde)
            l_r = motion_regularisation_loss(traj_state.deltax, p, weights)
            return weights.pde * l_p + l_r
        return jax.value_and_grad(loss_fn)(params)

    history: list[dict[str, float]] = []

    for epoch in tqdm(range(cfg.phase2.n_epochs), desc="Phase 2 Epochs"):
        # ---- 2a: Trajectory correction ----
        traj_losses: list[float] = []
        for _ in range(cfg.phase2.n_traj_steps):
            batch = replay.sample(cfg.phase2.batch_size, field_keys=[cfg.data.field_key])
            obs_xy = jnp.array(batch["xy_enu"], dtype=jnp.float32)
            obs_vals = jnp.array(batch[cfg.data.field_key], dtype=jnp.float32)
            loss_val, grads = _grad_traj(traj_state.deltax, obs_xy, obs_vals)
            updates, new_opt_state = tx_traj.update(
                grads, traj_state.opt_state, traj_state.deltax
            )
            import optax
            new_deltax = optax.apply_updates(traj_state.deltax, updates)
            traj_state = TrajectoryState(deltax=new_deltax, opt_state=new_opt_state)
            traj_losses.append(float(loss_val))

        # ---- 2b: PDE parameter update ----
        pde_losses: list[float] = []
        for _ in range(cfg.phase2.n_pde_steps):
            loss_val, grads = _grad_pde(pde_state.pde_params)
            updates, new_opt_state = tx_pde.update(
                grads, pde_state.opt_state, pde_state.pde_params
            )
            import optax
            new_params = optax.apply_updates(pde_state.pde_params, updates)
            pde_state = PDEState(pde_params=new_params, opt_state=new_opt_state)
            pde_losses.append(float(loss_val))

        epoch_log = {
            "epoch": epoch,
            "traj_loss": float(np.mean(traj_losses)),
            "pde_loss": float(np.mean(pde_losses)),
        }
        history.append(epoch_log)

        if epoch % 10 == 0:
            logger.info(
                "Epoch %d | traj_loss=%.4e | pde_loss=%.4e",
                epoch, epoch_log["traj_loss"], epoch_log["pde_loss"],
            )

    logger.info("Phase 2 complete after %d epochs.", cfg.phase2.n_epochs)
    return traj_state, pde_state


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run(cfg: DictConfig) -> dict[str, Any]:
    """Execute the full two-phase PDE-SLAM pipeline.

    Parameters
    ----------
    cfg :
        ``OmegaConf`` configuration.  Typically loaded from
        ``configs/default.yaml`` and merged with CLI overrides.

    Returns
    -------
    results :
        Dictionary containing final ``traj_state``, ``pde_state``,
        ``phi0`` initial condition, and the computational ``grid``.
    """
    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("PDE-SLAM starting.  JAX backend: %s", jax.default_backend())

    rng = jax.random.PRNGKey(cfg.seed)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Phase 1 ---
    df, xy_enu = _load_and_parse(cfg)
    phi0, grid, replay = _run_phase1(df, xy_enu, cfg)

    # --- Phase 2 initialisation ---
    nominal_poses, deltax_init, pde_params_init, _, _ = _initialise_phase2_params(
        df, cfg, grid, rng
    )

    # --- Phase 2 optimisation ---
    t0 = time.perf_counter()
    traj_state, pde_state = _run_phase2(
        phi0, nominal_poses, deltax_init, pde_params_init, replay, grid, cfg
    )
    elapsed = time.perf_counter() - t0
    logger.info("Total Phase 2 wall time: %.2f s", elapsed)

    results = {
        "traj_state": traj_state,
        "pde_state": pde_state,
        "phi0": phi0,
        "grid": grid,
        "nominal_poses": nominal_poses,
    }
    return results


def cli_entry() -> None:
    """Console-script entry point installed as ``pde-slam``.

    Usage::

        pde-slam [config_path] [key=value ...]

    Loads ``configs/default.yaml`` if no path is given, then applies any
    dot-notation overrides from the remaining arguments.

    Example::

        pde-slam configs/default.yaml phase2.n_epochs=200 phase2.lr_traj=5e-4
    """
    import sys

    base_cfg = OmegaConf.create(DEFAULT_CFG)

    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        config_path = Path(sys.argv[1])
        if config_path.exists():
            file_cfg = OmegaConf.load(config_path)
            base_cfg = OmegaConf.merge(base_cfg, file_cfg)
            override_args = sys.argv[2:]
        else:
            override_args = sys.argv[1:]
    else:
        override_args = sys.argv[1:]

    if override_args:
        cli_overrides = OmegaConf.from_dotlist(override_args)
        base_cfg = OmegaConf.merge(base_cfg, cli_overrides)

    run(base_cfg)


if __name__ == "__main__":
    cli_entry()
