# PDE-SLAM — Project Rules

## Code Style
- Use `jnp.*` (JAX numpy) everywhere inside `pde_slam/`. Avoid `np.*` in library code.
- Never write Python loops over array elements. Use `jnp.where`, `jax.lax.scan`, or `jax.vmap`.
- All public functions must have NumPy-style docstrings with Parameters and Returns sections.
- Always include `from __future__ import annotations` at the top of every module.
- Keep line length at 100 characters (ruff enforced).

## Numerical Safety
- Always verify Courant number <= 1 and diffusion number <= 0.5 before running long solves.
- Document the boundary condition assumption (default: zero-Neumann) in any new stencil.

## Testing
- New solver features must have a corresponding test in `tests/`.
- Run `pytest -v` before marking work complete.
- Use fixtures from `tests/conftest.py` for grid and field setup.

## Tooling
- Use `uv` for all dependency management (`uv sync --all-extras`).
- Run `ruff check pde_slam/ tests/` and `ruff format pde_slam/ tests/` before committing.
- Type-check with `mypy pde_slam/` for any new public API.
