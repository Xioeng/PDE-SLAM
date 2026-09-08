"""
simulation.py
=============
Loader and continuous evaluation interface for multi-field PDE simulation datasets.
"""

from __future__ import annotations

import glob
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.path as mpath
import numpy as np

from pde_slam.coords import ENUFrame
from pde_slam.interpolators.grid import SpatialGrid


@dataclass
class SimulationDataset:
    """Multi-field hydrodynamic / PDE simulation dataset container.

    Attributes
    ----------
    sim_dir : Path
        Directory where NPZ files are located.
    field_names : list of str
        List of loaded field names (e.g. ['salinity', 'temperature']).
    simulations : dict of str to dict
        Raw arrays per field ('X', 'Y', 'solutions', 'time_steps', 'mesh_mask').
    enu_frame : ENUFrame
        Geodetic reference frame origin.
    polygon_enu : np.ndarray
        (K, 2) boundary polygon in local ENU coordinates [m].
    grid : SpatialGrid
        Grid specification for the spatial domain.
    mesh_mask : np.ndarray
        2D boolean mask (True inside PDE fluid domain, False on land/outside).
    sample_times : np.ndarray
        1D array of simulation timestamps [s].
    field_means : dict of str to float
        Normalization mean per field.
    field_stds : dict of str to float
        Normalization standard deviation per field.
    """

    sim_dir: Path
    field_names: list[str]
    simulations: dict[str, dict[str, Any]]
    enu_frame: ENUFrame
    polygon_enu: np.ndarray
    grid: SpatialGrid
    mesh_mask: np.ndarray
    sample_times: np.ndarray
    field_means: dict[str, float]
    field_stds: dict[str, float]


def match_field_name(requested: str, available_fields: list[str]) -> str | None:
    """Flexibly match requested field name against available simulation fields.

    Parameters
    ----------
    requested : str
        Requested field name (e.g. 'Salinity', 'temp', 'odo').
    available_fields : list of str
        Available field names in the simulation dataset.

    Returns
    -------
    str or None
        Matched field name or None.
    """
    req_clean = (
        requested.lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("(", "")
        .replace(")", "")
    )
    for avail in available_fields:
        avail_clean = (
            avail.lower()
            .replace(" ", "")
            .replace("_", "")
            .replace("(", "")
            .replace(")", "")
        )
        if req_clean in avail_clean or avail_clean in req_clean:
            return avail
    return None


def load_simulation_dataset(
    sim_dir: str | Path,
    requested_fields: list[str] | None = None,
) -> SimulationDataset:
    """Load multi-field NPZ PDE simulation files from a directory.

    Parameters
    ----------
    sim_dir : str or Path
        Directory containing .npz simulation files.
    requested_fields : list of str, optional
        Specific fields to load. If None, loads all available NPZ files.

    Returns
    -------
    SimulationDataset
        Structured dataset object ready for simulation runs.
    """
    sim_path = Path(sim_dir)
    npz_files = sorted(glob.glob(str(sim_path / "*.npz")))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {sim_path}")

    simulations: dict[str, dict[str, Any]] = {}
    for npz_path in npz_files:
        field_key = Path(npz_path).stem.lower()
        with np.load(npz_path, allow_pickle=True) as data:
            sim_dict: dict[str, Any] = {
                "X": data["X"],
                "Y": data["Y"],
                "solutions": data["solutions"],
                "time_steps": data["time_steps"],
                "mesh_mask": data["mesh_mask"],
            }
            if "config_json" in data:
                try:
                    sim_dict["config"] = json.loads(str(data["config_json"]))
                except Exception:
                    sim_dict["config"] = {}
            else:
                sim_dict["config"] = {}
            simulations[field_key] = sim_dict

    available_fields = list(simulations.keys())
    if requested_fields is not None:
        active_fields = []
        for req in requested_fields:
            matched = match_field_name(req, available_fields)
            if matched and matched not in active_fields:
                active_fields.append(matched)
        if not active_fields:
            active_fields = available_fields
    else:
        active_fields = available_fields

    first_field = active_fields[0]
    primary_sim = simulations[first_field]
    cfg = primary_sim.get("config", {})

    are_coords_lonlat = bool(cfg.get("are_coordinates_lonlat", False))
    raw_polygon_pts = np.array(
        cfg.get("polygon_points", cfg.get("polygon_enu", [])), dtype=np.float64
    )

    X_mat = primary_sim["X"]
    Y_mat = primary_sim["Y"]

    if not are_coords_lonlat and "are_coordinates_lonlat" not in cfg:
        x_min_val, x_max_val = float(np.nanmin(X_mat)), float(np.nanmax(X_mat))
        y_min_val, y_max_val = float(np.nanmin(Y_mat)), float(np.nanmax(Y_mat))
        if (
            -180.0 <= x_min_val <= 180.0
            and -180.0 <= x_max_val <= 180.0
            and -90.0 <= y_min_val <= 90.0
            and -90.0 <= y_max_val <= 90.0
            and (abs(x_max_val - x_min_val) < 1.0)
            and (abs(y_max_val - y_min_val) < 1.0)
        ):
            are_coords_lonlat = True

    if are_coords_lonlat:
        if len(raw_polygon_pts) > 0:
            lat0 = float(cfg.get("origin_lat", np.mean(raw_polygon_pts[:, 1])))
            lon0 = float(cfg.get("origin_lon", np.mean(raw_polygon_pts[:, 0])))
        else:
            lat0 = float(cfg.get("origin_lat", np.nanmean(Y_mat)))
            lon0 = float(cfg.get("origin_lon", np.nanmean(X_mat)))

        enu_frame = ENUFrame(lat0=lat0, lon0=lon0)

        if len(raw_polygon_pts) > 0:
            poly_e, poly_n = enu_frame.to_enu(
                lat=raw_polygon_pts[:, 1], lon=raw_polygon_pts[:, 0]
            )
            polygon_enu = np.column_stack([poly_e, poly_n])
        else:
            grid_e, grid_n = enu_frame.to_enu(lat=Y_mat, lon=X_mat)
            polygon_enu = np.array(
                [
                    [grid_e[0, 0], grid_n[0, 0]],
                    [grid_e[-1, 0], grid_n[-1, 0]],
                    [grid_e[-1, -1], grid_n[-1, -1]],
                    [grid_e[0, -1], grid_n[0, -1]],
                ]
            )

        grid_e, grid_n = enu_frame.to_enu(lat=Y_mat, lon=X_mat)
        x_min, x_max = float(np.nanmin(grid_e)), float(np.nanmax(grid_e))
        y_min, y_max = float(np.nanmin(grid_n)), float(np.nanmax(grid_n))
        nx, ny = X_mat.shape[1], X_mat.shape[0]
        grid = SpatialGrid(
            x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, nx=nx, ny=ny
        )

        for f in simulations:
            simulations[f]["X_geo"] = simulations[f]["X"]
            simulations[f]["Y_geo"] = simulations[f]["Y"]
            simulations[f]["X"] = grid_e
            simulations[f]["Y"] = grid_n
    else:
        lat0 = float(cfg.get("origin_lat", 25.7617))
        lon0 = float(cfg.get("origin_lon", -80.1918))
        enu_frame = ENUFrame(lat0=lat0, lon0=lon0)

        if len(raw_polygon_pts) > 0:
            polygon_enu = raw_polygon_pts
        else:
            polygon_enu = np.array(
                [
                    [X_mat[0, 0], Y_mat[0, 0]],
                    [X_mat[-1, 0], Y_mat[-1, 0]],
                    [X_mat[-1, -1], Y_mat[-1, -1]],
                    [X_mat[0, -1], Y_mat[0, -1]],
                ]
            )

        x_min, x_max = float(np.nanmin(X_mat)), float(np.nanmax(X_mat))
        y_min, y_max = float(np.nanmin(Y_mat)), float(np.nanmax(Y_mat))
        nx, ny = X_mat.shape[1], X_mat.shape[0]
        grid = SpatialGrid(
            x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, nx=nx, ny=ny
        )

    mesh_mask = np.asarray(primary_sim["mesh_mask"], dtype=bool)
    time_steps = np.asarray(primary_sim["time_steps"], dtype=float)

    # Compute per-field mean and std for normalization using nanmean / nanstd
    field_means = {}
    field_stds = {}
    for f in active_fields:
        sols = simulations[f]["solutions"]
        mean_val = float(np.nanmean(sols))
        std_val = float(np.nanstd(sols))
        field_means[f] = mean_val if np.isfinite(mean_val) else 0.0
        field_stds[f] = std_val if np.isfinite(std_val) and std_val > 1e-6 else 1.0

    return SimulationDataset(
        sim_dir=sim_path,
        field_names=active_fields,
        simulations=simulations,
        enu_frame=enu_frame,
        polygon_enu=polygon_enu,
        grid=grid,
        mesh_mask=mesh_mask,
        sample_times=time_steps,
        field_means=field_means,
        field_stds=field_stds,
    )


def sample_simulation_field(
    sim_data: SimulationDataset,
    field_name: str,
    t: float,
    x: float,
    y: float,
    normalized: bool = False,
) -> float:
    """Sample ground-truth simulation scalar field value at continuous (t, x, y).

    Parameters
    ----------
    sim_data : SimulationDataset
        Loaded simulation dataset.
    field_name : str
        Name of the physical field (e.g. salinity).
    t : float
        Query timestamp [s].
    x : float
        Query East coordinate [m].
    y : float
        Query North coordinate [m].
    normalized : bool, default=False
        If True, returns zero-mean unit-variance normalized value.

    Returns
    -------
    float
        Sampled field value.
    """
    field_key = field_name
    if field_key not in sim_data.simulations:
        for k in sim_data.simulations:
            if k.lower() == field_name.lower():
                field_key = k
                break
    sim = sim_data.simulations[field_key]
    sols = sim["solutions"]  # shape (T, nx, ny) or (T, ny, nx)
    n_times = len(sols)
    t_max = float(sim_data.sample_times[-1]) if len(sim_data.sample_times) > 0 else 1.0

    t_idx = int(np.clip(int((t / t_max) * (n_times - 1)), 0, n_times - 1))
    grid = sim_data.grid

    # Bilinear interpolation on grid
    ix = np.clip(
        (x - grid.x_min) / (grid.x_max - grid.x_min) * (grid.nx - 1), 0, grid.nx - 1
    )
    iy = np.clip(
        (y - grid.y_min) / (grid.y_max - grid.y_min) * (grid.ny - 1), 0, grid.ny - 1
    )

    ix0, iy0 = int(np.floor(ix)), int(np.floor(iy))
    ix1, iy1 = min(ix0 + 1, grid.nx - 1), min(iy0 + 1, grid.ny - 1)
    fx, fy = ix - ix0, iy - iy0

    snap = sols[t_idx]
    # Handle indexing (ny, nx) vs (nx, ny)
    if snap.shape == (grid.nx, grid.ny):
        vals = np.array(
            [
                snap[ix0, iy0],
                snap[ix1, iy0],
                snap[ix0, iy1],
                snap[ix1, iy1],
            ],
            dtype=np.float64,
        )
    else:
        vals = np.array(
            [
                snap[iy0, ix0],
                snap[iy0, ix1],
                snap[iy1, ix0],
                snap[iy1, ix1],
            ],
            dtype=np.float64,
        )

    weights = np.array(
        [
            (1.0 - fx) * (1.0 - fy),
            fx * (1.0 - fy),
            (1.0 - fx) * fy,
            fx * fy,
        ],
        dtype=np.float64,
    )

    valid_mask = np.isfinite(vals)
    w_valid_sum = np.sum(weights[valid_mask])
    if np.any(valid_mask) and w_valid_sum > 1e-9:
        val_f = float(np.sum(weights[valid_mask] * vals[valid_mask]) / w_valid_sum)
    else:
        val_f = float(sim_data.field_means.get(field_key, 0.0))

    if normalized:
        mean_f = sim_data.field_means.get(field_key, 0.0)
        std_f = sim_data.field_stds.get(field_key, 1.0)
        return float((val_f - mean_f) / std_f)
    return val_f


def generate_ic_anchors(
    polygon_enu: np.ndarray,
    n_points: int = 30,
    mode: str = "auto",
    seed: int = 42,
    dataset: SimulationDataset | None = None,
) -> np.ndarray:
    """Generate Initial Condition (t=0) spatial measurement anchor coordinates.

    Parameters
    ----------
    polygon_enu : np.ndarray
        (K, 2) boundary polygon in local ENU coordinates.
    n_points : int, default=30
        Number of anchor points to generate.
    mode : str, default='auto'
        'auto' for uniform grid sampling within polygon / fluid mesh.
    seed : int, default=42
        Random seed.
    dataset : SimulationDataset, optional
        Simulation dataset to ensure anchors are strictly within valid fluid mesh.

    Returns
    -------
    np.ndarray
        (M, 2) array of anchor coordinates inside fluid polygon.
    """
    poly_path = mpath.Path(polygon_enu)
    np.random.seed(seed)

    if dataset is not None and dataset.mesh_mask is not None:
        first_field = dataset.field_names[0]
        X = dataset.simulations[first_field]["X"]
        Y = dataset.simulations[first_field]["Y"]
        sols0 = dataset.simulations[first_field]["solutions"][0]
        valid_mask = dataset.mesh_mask & np.isfinite(sols0)

        valid_x = X[valid_mask]
        valid_y = Y[valid_mask]
        all_fluid_pts = np.column_stack([valid_x, valid_y])

        inside_poly = poly_path.contains_points(all_fluid_pts)
        candidate_pts = (
            all_fluid_pts[inside_poly] if np.any(inside_poly) else all_fluid_pts
        )

        if len(candidate_pts) >= n_points:
            indices = np.random.choice(len(candidate_pts), size=n_points, replace=False)
            return candidate_pts[indices]
        return candidate_pts

    x_min, y_min = np.min(polygon_enu, axis=0)
    x_max, y_max = np.max(polygon_enu, axis=0)

    if mode in ("auto", "uniform", "grid"):
        grid_density = int(np.ceil(np.sqrt(n_points * 2.5)))
        xs = np.linspace(x_min, x_max, grid_density)
        ys = np.linspace(y_min, y_max, grid_density)
        xx, yy = np.meshgrid(xs, ys, indexing="ij")
        pts = np.column_stack([xx.ravel(), yy.ravel()])
        inside = poly_path.contains_points(pts)
        valid_pts = pts[inside]

        if len(valid_pts) >= n_points:
            indices = np.random.choice(len(valid_pts), size=n_points, replace=False)
            return valid_pts[indices]
        return valid_pts

    return np.mean(polygon_enu, axis=0, keepdims=True)
