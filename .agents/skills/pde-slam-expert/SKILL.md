---
name: pde-slam-expert
description: >
  Expert in the PDE-SLAM project: JAX-based differentiable 2D advection-diffusion
  PDE solver for aquatic SLAM. Triggered for tasks involving solver numerics,
  JAX/diffrax patterns, field interpolation, numerical stability, autodiff, or
  extending the pde_slam package.
---

# PDE-SLAM Expert Skill

You are an expert in the **PDE-SLAM** codebase — a JAX/diffrax framework for
solving and differentiating 2-D advection-diffusion PDEs over aquatic scalar
fields (salinity, temperature, etc.) for marine SLAM.

## Project Layout

```
pde_slam/
├── __init__.py          # public API re-exports
├── interpolator.py      # scattered → grid field initialisation (RBF / spline)
└── solver.py            # differentiable 2-D FD advection-diffusion solver
tests/                   # pytest suite (conftest.py + test_*.py)
test_scripts/            # visual demos (demo_solver.py, demo_interpolator.py)
scripts/                 # data generation utilities
pyproject.toml           # uv/hatch project, Python ≥ 3.12
```

## Core Abstractions

### `SpatialGrid` (interpolator.py)
- Defines a uniform Cartesian domain `[x_min, x_max] × [y_min, y_max]` with
  `(nx, ny)` cells.
- Exposes `dx`, `dy` grid spacings used by the solver stencils.

### `FieldInterpolator` (interpolator.py)
- Converts scattered `(x, y, value)` sensor points → dense `(ny, nx)` JAX array.
- Two backends: `"rbf"` (thin-plate spline, smooth, O(N³)) and
  `"spline"` (cubic Delaunay + nearest-neighbour, faster).

### `PDEParams` (solver.py)
- `NamedTuple` holding `u_field: Array` (shape `(ny, nx, 2)`) and `D: Array`
  (scalar diffusivity). Treat as a JAX pytree leaf for `jax.grad` / `jax.jit`.

### `AdvectionDiffusionSolver` (solver.py)
- Wraps diffrax time-integration of `dφ/dt = D∇²φ − u·∇φ`.
- **Diffusion**: 2nd-order central differences (`_laplacian_cd2`).
- **Advection**: 1st-order upwind / donor-cell (`_advection_upwind`).
- **Integrator**: `diffrax.Heun` (explicit 2nd-order) by default.
- **Adjoint**: `diffrax.RecursiveCheckpointAdjoint` for memory-efficient gradients.
- `solve(phi0, pde_params, t0, t_end, saveat=None)` returns `Array (ny, nx)` or
  `Array (T, ny, nx)` when `saveat` timestamps are given.

## Coding Rules

1. **JAX-first**: use `jnp.*` everywhere inside `pde_slam/`. Never use Python
   loops over array elements; use `jnp.where`, `lax.scan`, `vmap` instead.
2. **JIT compatibility**: all functions called inside `jax.jit` must be
   statically-shaped and free of Python-side conditionals on traced values.
3. **Type annotations**: always include `from __future__ import annotations` and
   annotate with `jax.Array` (alias `Array` from `from jax import Array`).
4. **Boundary conditions**: default is zero-Neumann (edge-padding). Document any
   deviation explicitly.
5. **Stability checks**: advective Courant <= 1, diffusion number <= 0.5. Use
   `solver.courant_number()` / `solver.diffusion_number()` to verify before
   long runs.
6. **Linting**: ruff with `line-length = 100`, selectors `E F I N UP B SIM`.
   Run `ruff check pde_slam/` and `ruff format pde_slam/` before committing.
7. **Tests**: add pytest cases in `tests/` using fixtures from `conftest.py`.
   Run `pytest -v` to verify.
8. **Package manager**: use `uv` -- `uv sync --all-extras`, never `pip install`
   directly.

## Numerical Guidance

- **Choosing `dt_max`**: for stability, use
  `dt_max <= min(dx, dy) / (2 * max|u|)` (advection) and
  `dt_max <= 0.5 / (D * (1/dx^2 + 1/dy^2))` (diffusion). Take the minimum.
- **Higher-order advection**: to upgrade from 1st-order upwind, implement a
  MUSCL or WENO stencil in `_advection_upwind` while preserving the same
  function signature.
- **Implicit diffusion**: swap `diffrax.Heun` for `diffrax.Kvaerno3` or
  `diffrax.ImplicitEuler` for stiff diffusion-dominated regimes.
- **Gradient flow**: `jax.grad(loss)(pde_params)` works out-of-the-box because
  `PDEParams` is a NamedTuple (pytree). Ensure `u_field` and `D` are JAX arrays,
  not Python floats, when differentiating.

## Common Patterns

```python
# Minimal solve
from pde_slam import SpatialGrid, AdvectionDiffusionSolver, PDEParams
import jax.numpy as jnp

grid = SpatialGrid(0, 500, 0, 500, nx=64, ny=64)
solver = AdvectionDiffusionSolver(grid, dt_max=1.0)

u = jnp.zeros((64, 64, 2))          # no flow
D = jnp.array(0.1)                  # diffusivity m^2 s^-1
phi0 = jnp.zeros((64, 64)).at[32, 32].set(1.0)

params = PDEParams(u_field=u, D=D)
phi_end = solver.solve(phi0, params, t0=0.0, t_end=60.0)
```

```python
# Gradient w.r.t. diffusivity
import jax

def loss(params):
    phi = solver.solve(phi0, params, t0=0.0, t_end=60.0)
    return jnp.mean((phi - phi_target) ** 2)

grad = jax.grad(loss)(params)   # grad.D, grad.u_field
```

## References

See `references/` for extended notes on:
- `numerical_schemes.md` -- scheme derivations and stability analysis
- `diffrax_patterns.md`  -- diffrax API patterns used in this project
