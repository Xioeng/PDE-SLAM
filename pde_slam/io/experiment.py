"""
experiment.py
Serialization and IO submodule for recording, saving, and loading complete
PDE-SLAM experiment runs (trajectories, particle histories, PINN checkpoints).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from pde_slam.coords import ENUFrame
from pde_slam.pinn import PinnConfig, PinnFieldMap, PinnParams


def _serialize_pytree(obj: Any) -> Any:
    """Recursively convert JAX arrays to NumPy arrays for serialization."""
    if isinstance(obj, PinnParams):
        return {
            "weights": [np.array(w) for w in obj.weights],
            "biases": [np.array(b) for b in obj.biases],
            "v_flow": np.array(obj.v_flow),
            "log_D": np.array(obj.log_D),
            "W_u": np.array(obj.W_u) if obj.W_u is not None else None,
            "b_u": np.array(obj.b_u) if obj.b_u is not None else None,
            "W_v": np.array(obj.W_v) if obj.W_v is not None else None,
            "b_v": np.array(obj.b_v) if obj.b_v is not None else None,
        }
    elif isinstance(obj, (np.ndarray, np.generic)):
        return obj
    elif hasattr(obj, "__array__"):
        return np.array(obj)
    elif isinstance(obj, dict):
        return {k: _serialize_pytree(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_serialize_pytree(v) for v in obj]
    return obj


def _deserialize_pinn_params(d: dict[str, Any]) -> PinnParams:
    """Restore PinnParams JAX PyTree from a dictionary."""
    return PinnParams(
        weights=[jnp.array(w) for w in d["weights"]],
        biases=[jnp.array(b) for b in d["biases"]],
        v_flow=jnp.array(d["v_flow"]),
        log_D=jnp.array(d["log_D"]),
        W_u=jnp.array(d["W_u"]) if d.get("W_u") is not None else None,
        b_u=jnp.array(d["b_u"]) if d.get("b_u") is not None else None,
        W_v=jnp.array(d["W_v"]) if d.get("W_v") is not None else None,
        b_v=jnp.array(d["b_v"]) if d.get("b_v") is not None else None,
    )


@dataclass
class SlamExperimentData:
    """Dataclass holding full PDE-SLAM experiment history for offline analysis."""

    # 1. Environment & Domain Metadata
    sim_name: str = "simulation"
    grid_extent: dict[str, Any] = field(default_factory=dict)
    enu_origin: tuple[float, float] = (0.0, 0.0)
    polygon_enu: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    mesh_mask: np.ndarray | None = None
    sample_times: np.ndarray = field(default_factory=lambda: np.zeros(0))
    field_names: list[str] = field(default_factory=list)
    ground_truth_params: dict[str, Any] = field(default_factory=dict)
    noise_params: dict[str, Any] = field(default_factory=dict)

    # 2. Ground Truth Field Solutions (shape per field: T_sim, nx, ny)
    gt_solutions: dict[str, np.ndarray] = field(default_factory=dict)
    field_means: dict[str, float] = field(default_factory=dict)
    field_stds: dict[str, float] = field(default_factory=dict)

    # 2b. Current Field Estimations Tensor (shape: n_fields, nx, ny)
    field_estimations: np.ndarray | None = None

    # 3. Trajectories & Controls
    coords_true: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    coords_dr: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    oracle_traj: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    estimated_traj: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    velocities: np.ndarray = field(default_factory=lambda: np.zeros(0))
    omegas: np.ndarray = field(default_factory=lambda: np.zeros(0))

    # 4. Particle Filter State History
    particle_poses_history: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0, 2))
    )
    particle_weights_history: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0))
    )
    particle_xl_history: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))
    particle_P_history: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0, 0, 0))
    )

    # 5. Measurements & Initial Conditions
    ic_points_enu: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    obs_pts: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    obs_vals: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))

    # 6. PINN Model Checkpoints & Parameter Evolution
    pinn_config: PinnConfig | None = None
    pinn_checkpoints: dict[int | float | str, Any] = field(default_factory=dict)
    loss_history: list[float] = field(default_factory=list)
    v_flow_history: list[np.ndarray] = field(default_factory=list)
    D_history: list[np.ndarray] = field(default_factory=list)

    def reconstruct_pinn_map(self, checkpoint_key: int | float | str) -> PinnFieldMap:
        """Reconstruct a PinnFieldMap at a specific checkpoint step or timestamp.

        Parameters
        ----------
        checkpoint_key : int or float
            Key corresponding to a saved PINN checkpoint.

        Returns
        -------
        PinnFieldMap
            Reconstructed PINN map instance with parameters loaded.
        """
        if self.pinn_config is None:
            raise ValueError("pinn_config is not set in SlamExperimentData")
        if checkpoint_key not in self.pinn_checkpoints:
            valid_keys = list(self.pinn_checkpoints.keys())
            raise KeyError(
                f"Checkpoint {checkpoint_key} not found. Available keys: {valid_keys}"
            )

        pinn_map = PinnFieldMap(config=self.pinn_config)
        ckpt = self.pinn_checkpoints[checkpoint_key]
        if isinstance(ckpt, PinnParams):
            pinn_map.params = ckpt
        elif isinstance(ckpt, dict):
            pinn_map.params = _deserialize_pinn_params(ckpt)
        else:
            pinn_map.params = ckpt
        return pinn_map

    @property
    def enu_frame(self) -> ENUFrame:
        """Construct the ENUFrame object using the stored origin (lat0, lon0)."""
        return ENUFrame(lat0=self.enu_origin[0], lon0=self.enu_origin[1])

    def to_geodetic(
        self, east_m: np.ndarray | float, north_m: np.ndarray | float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert local ENU coordinates [m] back to geodetic (lat, lon) [deg]."""
        return self.enu_frame.from_enu(east_m, north_m)

    def to_enu(
        self, lat: np.ndarray | float, lon: np.ndarray | float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert geodetic (lat, lon) [deg] to local ENU coordinates [m]."""
        return self.enu_frame.to_enu(lat, lon)


def save_experiment(data: SlamExperimentData, filepath: str | Path) -> Path:
    """Save SlamExperimentData object to disk.

    Parameters
    ----------
    data : SlamExperimentData
        Experiment dataset to serialize.
    filepath : str or Path
        Target destination path (.pkl or .npz).

    Returns
    -------
    Path
        Absolute path to the saved file.
    """
    path = Path(filepath).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    serialized_dict = {
        "sim_name": getattr(data, "sim_name", "simulation"),
        "grid_extent": data.grid_extent,
        "enu_origin": data.enu_origin,
        "polygon_enu": _serialize_pytree(data.polygon_enu),
        "mesh_mask": _serialize_pytree(data.mesh_mask)
        if data.mesh_mask is not None
        else None,
        "sample_times": _serialize_pytree(data.sample_times),
        "field_names": data.field_names,
        "ground_truth_params": _serialize_pytree(data.ground_truth_params),
        "noise_params": _serialize_pytree(data.noise_params),
        "gt_solutions": _serialize_pytree(data.gt_solutions),
        "field_means": data.field_means,
        "field_stds": data.field_stds,
        "field_estimations": _serialize_pytree(data.field_estimations)
        if data.field_estimations is not None
        else None,
        "coords_true": _serialize_pytree(data.coords_true),
        "coords_dr": _serialize_pytree(data.coords_dr),
        "oracle_traj": _serialize_pytree(data.oracle_traj),
        "estimated_traj": _serialize_pytree(data.estimated_traj),
        "velocities": _serialize_pytree(data.velocities),
        "omegas": _serialize_pytree(data.omegas),
        "particle_poses_history": _serialize_pytree(data.particle_poses_history),
        "particle_weights_history": _serialize_pytree(data.particle_weights_history),
        "particle_xl_history": _serialize_pytree(data.particle_xl_history),
        "particle_P_history": _serialize_pytree(data.particle_P_history),
        "ic_points_enu": _serialize_pytree(data.ic_points_enu),
        "obs_pts": _serialize_pytree(data.obs_pts),
        "obs_vals": _serialize_pytree(data.obs_vals),
        "pinn_config": data.pinn_config,
        "pinn_checkpoints": _serialize_pytree(data.pinn_checkpoints),
        "loss_history": data.loss_history,
        "v_flow_history": _serialize_pytree(data.v_flow_history),
        "D_history": _serialize_pytree(data.D_history),
    }

    with open(path, "wb") as f:
        pickle.dump(serialized_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Successfully saved SlamExperimentData to {path}")
    return path


def load_experiment(filepath: str | Path) -> SlamExperimentData:
    """Load SlamExperimentData object from disk.

    Parameters
    ----------
    filepath : str or Path
        Path to saved experiment file (.pkl).

    Returns
    -------
    SlamExperimentData
        Loaded experiment dataset.
    """
    path = Path(filepath).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Experiment file not found: {path}")

    with open(path, "rb") as f:
        raw_dict = pickle.load(f)

    return SlamExperimentData(
        sim_name=raw_dict.get(
            "sim_name", path.stem.replace("_experiment", "").replace("_rbpf_slam", "")
        ),
        grid_extent=raw_dict.get("grid_extent", {}),
        enu_origin=raw_dict.get("enu_origin", (0.0, 0.0)),
        polygon_enu=np.array(raw_dict.get("polygon_enu", [])),
        mesh_mask=np.array(raw_dict["mesh_mask"])
        if raw_dict.get("mesh_mask") is not None
        else None,
        sample_times=np.array(raw_dict.get("sample_times", [])),
        field_names=raw_dict.get("field_names", []),
        ground_truth_params=raw_dict.get("ground_truth_params", {}),
        noise_params=raw_dict.get("noise_params", {}),
        gt_solutions=raw_dict.get("gt_solutions", {}),
        field_means=raw_dict.get("field_means", {}),
        field_stds=raw_dict.get("field_stds", {}),
        field_estimations=np.array(raw_dict["field_estimations"])
        if raw_dict.get("field_estimations") is not None
        else None,
        coords_true=np.array(raw_dict.get("coords_true", [])),
        coords_dr=np.array(raw_dict.get("coords_dr", [])),
        oracle_traj=np.array(raw_dict.get("oracle_traj", [])),
        estimated_traj=np.array(raw_dict.get("estimated_traj", [])),
        velocities=np.array(raw_dict.get("velocities", [])),
        omegas=np.array(raw_dict.get("omegas", [])),
        particle_poses_history=np.array(raw_dict.get("particle_poses_history", [])),
        particle_weights_history=np.array(raw_dict.get("particle_weights_history", [])),
        particle_xl_history=np.array(raw_dict.get("particle_xl_history", [])),
        particle_P_history=np.array(raw_dict.get("particle_P_history", [])),
        ic_points_enu=np.array(raw_dict.get("ic_points_enu", [])),
        obs_pts=np.array(raw_dict.get("obs_pts", [])),
        obs_vals=np.array(raw_dict.get("obs_vals", [])),
        pinn_config=raw_dict.get("pinn_config"),
        pinn_checkpoints=raw_dict.get("pinn_checkpoints", {}),
        loss_history=raw_dict.get("loss_history", []),
        v_flow_history=raw_dict.get("v_flow_history", []),
        D_history=raw_dict.get("D_history", []),
    )
