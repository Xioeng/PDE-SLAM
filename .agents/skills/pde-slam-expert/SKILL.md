---
name: pde-slam-expert
description: >
  Expert in the PDE-SLAM project: JAX-based continuous Physics-Informed Neural Network
  (PINN) mapping, Rao-Blackwellized Particle Filter (RBPF) SLAM, and spatio-temporal
  field estimation for aquatic SLAM. Triggered for tasks involving SLAM filter design,
  kinematics, PINN architecture, autodiff PDE residuals, visualization, or extending pde_slam.
---

# PDE-SLAM Expert Skill

You are an expert in the **PDE-SLAM** codebase — an aquatic SLAM system integrating
Rao-Blackwellized Particle Filtering (RBPF), continuous Physics-Informed Neural Network
(PINN) mapping with automatic differentiation PDE residuals, and robot kinematics.

## Project Layout

```
pde_slam/
├── __init__.py          # Public API exports
├── coords.py            # Geodetic ENUFrame & coordinates
├── pinn.py              # PINN mapping, Wang & Perdikaris arch, autodiff PDE residuals
├── types.py             # Type aliases & protocols
├── config.py            # YAML configuration dataclasses & loaders
├── slam/                # Online State Estimation
│   └── rbpf.py          # Rao-Blackwellized Particle Filter (mutable OO, JAX-accelerated)
├── kinematics/          # Forward Kinematics & Calibration
│   ├── base.py          # Kinematics base class
│   ├── diff_drive.py    # Differential drive model & trajectory integration
│   ├── unicycle.py      # Unicycle kinematic model
│   └── calibration.py   # Parameter identification (Optax / Scipy)
├── interpolators/       # Continuous & Discrete Spatial Evaluators
│   ├── grid.py          # SpatialGrid domain mesh container
│   ├── field.py         # 2D bilinear lookup-table interpolator
│   ├── spatiotemporal.py# 3D (t, x, y) spatio-temporal lookup-table interpolator
│   └── gp.py            # Gaussian Process regression map (benchmarking baseline)
├── io/                  # Data Ingestion & Serialization
│   ├── experiment.py    # SlamExperimentData serialization (.pkl)
│   ├── simulation.py    # NPZ multi-field simulation dataset loader & interpolator
│   └── survey.py        # CSV field survey loader
└── viz/                 # Reusable Publication Visualization Engine
    ├── style.py         # LaTeX typography, palettes, trajectory colors, feature cmaps
    ├── satellite.py     # Slippy tile map downloader & ENU geo-referencing
    ├── panels.py        # Modular single-panel renderers (field, residual, path, rmse)
    ├── grids.py         # Composite multi-stage evolution & residual grid composers
    └── plotter.py       # High-level one-line plotting API for saved experiments
```

## Core Abstractions

### `RbpfSlam` (pde_slam.slam.rbpf)
- Mutable-state OO design managing particles (`poses`, `headings`, `speeds`, `xl`, `P`, `log_weights`, `trajectories`).
- Predictive motion updates via `DiffDriveKinematics`.
- Multi-field Kalman observation update with map variance inflation.
- Systematic resampling conditioned on $N_{\text{eff}} < \text{threshold} \times N$.

### `PinnFieldMap` & `pinn_pde_residual` (pde_slam.pinn)
- Meshless continuous representation $u_\theta(t, x, y)$ for multi-feature fields (salinity, temperature, chlorophyll, ODO).
- Architecture: Standard MLP or Wang & Perdikaris Modified MLP with Fourier spatial embeddings.
- Evaluates the physics constraint via exact JAX automatic differentiation:
  $$\mathcal{R}(t, x, y) = \frac{\partial u}{\partial t} + \mathbf{v}_{\text{flow}} \cdot \nabla u - D \nabla^2 u$$

### `SimulationDataset` & `SpatiotemporalInterpolator` (pde_slam.io & pde_slam.interpolators)
- Ingests offline pre-computed hydrodynamic simulation datasets (e.g. Biscayne Bay, Miami Canal).
- Provides continuous $(t, x, y)$ lookup tables for ground truth querying.

## Coding Rules & Architecture Guidelines

1. **Object-Oriented with JAX under the hood**: Maintain clear, stateful class interfaces (`RbpfSlam`, `DiffDriveKinematics`, `PinnFieldMap`), while using `jnp.*` for all array operations.
2. **Selective JIT Compilation**: Apply `@jax.jit` only to pure, statically-shaped mathematical steps (e.g. motion propagation, Kalman updates, PINN forward pass, train steps). Avoid blanket `@jax.jit` on stateful classes or high-level orchestration methods.
3. **No Python loops over arrays**: Use `jnp.where`, `jax.lax.scan`, or `jax.vmap`.
4. **Concise Functions**: Keep functions small, modular, and single-purpose; avoid long monolithic methods.
5. **Linting & Formatting**: Follow `ruff` with 88-character line length (`ruff check pde_slam/ tests/` and `ruff format pde_slam/ tests/`).
6. **Testing**: Add pytest test cases in `tests/` using fixtures from `conftest.py`. Run `pytest -v` to verify.
