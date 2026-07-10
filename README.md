# PDE-SLAM

> **Aquatic SLAM constrained by an advection-diffusion PDE**  
> A modular JAX/diffrax framework for solving and visualising 2-D
> advection-diffusion PDEs over water-feature scalar fields.

---

## Repository Layout

```
PDE-SLAM/
├── pde_slam/
│   ├── __init__.py          # package version & re-exports
│   ├── interpolator.py      # scattered → grid field initialisation (RBF / spline)
│   └── solver.py            # differentiable 2D advection-diffusion FD solver
├── tests/
│   ├── conftest.py          # shared test fixtures
│   ├── test_interpolator.py
│   └── test_solver.py
├── test_scripts/
│   ├── demo_interpolator.py # visual demo of field interpolation
│   └── demo_solver.py       # visual demo of PDE time-stepping
├── scripts/
│   └── generate_synthetic_survey.py
├── pyproject.toml
└── README.md
```

---

## Core Modules

### `interpolator.py`

Converts a cloud of scattered `(x, y, value)` sensor observations into a
dense `(ny, nx)` grid, ready to be used as an initial condition for the PDE
solver.

Two backends:

| Backend   | Method                       | Notes                     |
|-----------|------------------------------|---------------------------|
| `"rbf"`   | Radial Basis Function (TPS)  | Smooth, accurate, O(N³)   |
| `"spline"`| Cubic Delaunay + nearest NN  | Faster, requires ≥ 4 pts  |

### `solver.py`

Differentiable 2-D finite-difference solver for the scalar
advection-diffusion equation:

$$\frac{\partial \phi}{\partial t} + \mathbf{u} \cdot \nabla \phi = D \, \nabla^2 \phi$$

- **Diffusion**: 2nd-order central differences.
- **Advection**: 1st-order upwind scheme.
- **Time integration**: `diffrax` (default: Heun, 2nd-order explicit).
- Fully differentiable via JAX autodiff.

---

## Quick Start

### 1. Environment Setup

```bash
uv sync --all-extras
source .venv/bin/activate
```

### 2. Run Demos

```bash
python test_scripts/demo_interpolator.py
python test_scripts/demo_solver.py
```

### 3. Run Tests

```bash
pytest
```

---

## Key Dependencies

| Package   | Role                                    |
|-----------|-----------------------------------------|
| `jax`     | Autodiff, JIT compilation, vectorisation|
| `diffrax` | Differentiable ODE/PDE time integration |
| `scipy`   | RBF / spline interpolation              |
| `numpy`   | Array utilities                         |
| `matplotlib` | Visualisation                        |
