# PDE-SLAM

> **Aquatic SLAM constrained by an advection-diffusion PDE**  
> A modular JAX/diffrax framework for optimising water-feature maps and vehicle
> trajectories jointly in a decoupled two-phase loop.

---

## Repository Layout

```
PDE-SLAM/
├── src/
│   └── pde_slam/
│       ├── __init__.py          # package version & re-exports
│       ├── kinematics.py        # dead-reckoning, WGS-84 projection, drift latents
│       ├── interpolator.py      # scattered → grid field initialisation (RBF / spline)
│       ├── solver.py            # differentiable 2D advection-diffusion FD solver
│       ├── data_pipeline.py     # log parsing, measurement pool, replay buffer
│       ├── optimizer.py         # multi-objective loss + optax gradient updates
│       └── main.py              # Phase 1 → Phase 2 orchestrator & CLI
├── configs/
│   └── default.yaml             # all hyper-parameters (OmegaConf)
├── data/
│   ├── raw/                     # raw sensor logs (CSV / NDJSON)
│   └── processed/               # pre-processed arrays
├── tests/
│   ├── test_kinematics.py
│   ├── test_solver.py
│   ├── test_data_pipeline.py
│   └── test_interpolator.py
├── scripts/
│   ├── setup_env.sh             # uv environment bootstrap
│   └── generate_synthetic_survey.py
├── outputs/                     # checkpoints & plots (git-ignored)
├── pyproject.toml
└── README.md
```

---

## Algorithm Overview

### Phase 1 – Survey & Initialization

```
Raw GPS + sensor log
       │
       ▼  kinematics.latlon_to_enu()
ENU trajectory (dead-reckoned)
       │
       ▼  data_pipeline.build_measurement_pool()
Scattered (x, y, φ) observations
       │
       ▼  interpolator.build_initial_condition()
Dense initial condition φ₀(x,y)  →  solver.solve_pde() input
```

### Phase 2 – Decoupled Online SLAM Loop

```
for epoch in 1..N:
    ┌── Phase 2a: Trajectory Correction ──────────────────────────────────┐
    │   sample mini-batch from SpatialReplayBuffer                        │
    │   loss = w_d · L_data(δx) + w_r · L_reg(δx)                        │
    │   δx ← δx - α∇_{δx} loss     [Adam, fixed PDE params]              │
    └──────────────────────────────────────────────────────────────────────┘
    ┌── Phase 2b: PDE Parameter Update ───────────────────────────────────┐
    │   loss = w_p · L_pde(u, D) + w_r · L_reg(u)                        │
    │   (u, D) ← (u, D) - β∇_{u,D} loss   [Adam, fixed trajectory]       │
    └──────────────────────────────────────────────────────────────────────┘
```

### Loss Terms

| Symbol | Name | Optimises |
|---|---|---|
| `L_data` | Data alignment MSE | trajectory drift `δx` |
| `L_pde` | PDE residual MSE | advection `u`, diffusivity `D` |
| `L_reg` | L2 Tikhonov | `δx` magnitude + `u` magnitude |

---

## Quick Start

### 1. Environment Setup (uv)

```bash
chmod +x scripts/setup_env.sh
./scripts/setup_env.sh          # CPU JAX
# ./scripts/setup_env.sh --gpu  # CUDA 12 JAX

source .venv/bin/activate
```

### 2. Generate Synthetic Survey Data

```bash
python scripts/generate_synthetic_survey.py --output data/raw/survey.csv
```

### 3. Run the Pipeline

```bash
pde-slam configs/default.yaml
# Override any parameter on the fly:
pde-slam configs/default.yaml phase2.n_epochs=200 grid.nx=128
```

### 4. Run Tests

```bash
pytest
```

---

## Key Dependencies

| Package | Role |
|---|---|
| `jax` / `jaxlib` | Autodiff, JIT compilation, vectorisation |
| `diffrax` | Differentiable ODE/PDE time integration |
| `optax` | Gradient transformations (Adam, clipping) |
| `torch` | Neural-field baseline / GPU kernel fallback |
| `scipy` | RBF / spline interpolation (Phase 1) |
| `pywmm` | WGS-84 magnetic declination lookup |
| `omegaconf` | Hierarchical YAML configuration |

---

## Configuration Reference

See [`configs/default.yaml`](configs/default.yaml) for the full annotated
parameter set.  All keys support dot-notation CLI overrides.

---

## Module Interaction Diagram

```
                     ┌──────────────┐
                     │ data_pipeline│◄── raw CSV / NDJSON logs
                     └──────┬───────┘
                            │ measurement_pool, replay_buffer
              ┌─────────────┼──────────────────┐
              ▼             ▼                  ▼
      ┌────────────┐ ┌─────────────┐   ┌───────────────┐
      │kinematics  │ │ interpolator│   │    solver     │
      │ (ENU proj, │ │ (RBF/spline │   │ (FD advection-│
      │dead-reckon)│ │   → φ₀)     │   │  diffusion)   │
      └─────┬──────┘ └──────┬──────┘   └──────┬────────┘
            │               │ φ₀              │ φ(t+dt)
            │ δx, poses     └──────┬──────────┘
            ▼                      ▼
      ┌─────────────────────────────────┐
      │           optimizer             │
      │  L_data + L_pde + L_reg → Adam  │
      └──────────────┬──────────────────┘
                     ▼
               ┌──────────┐
               │  main.py │  (Phase 1 → Phase 2 orchestrator)
               └──────────┘
```
