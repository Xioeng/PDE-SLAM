# PDE-SLAM — Project Rules

## Architecture & Code Style
- **Object-Oriented with JAX**: Maintain an Object-Oriented (OO) interface across the library (e.g. `RbpfSlam`, `DiffDriveKinematics`, `PinnFieldMap`, `SpatialGrid`, `SpatiotemporalInterpolator`), storing state cleanly on instances while executing array operations with JAX (`jnp.*`).
- **Selective JIT Compilation**: Only `@jax.jit` compile pure, statically-shaped mathematical kernels and performance-critical inner steps (e.g., motion propagation, Kalman updates, PINN forward passes, PDE loss gradient steps). Avoid blanket `@jax.jit` on large stateful methods or non-critical orchestration code.
- **Vectorization**: Never write Python loops over array or particle elements. Use `jnp.where`, `jax.lax.scan`, or `jax.vmap`.
- **Concise & Modular Functions**: Keep functions short and focused on a single responsibility (avoid long monolithic functions unless a single contiguous JIT kernel is strictly required).
- **Public API & Docstrings**: All public classes and functions must have NumPy-style docstrings with Parameters and Returns sections.
- **Formatting**: Keep line length at 88 characters (enforced by `ruff`).

## Numerical Safety & Physics
- Evaluate PDE physical constraints via JAX autodiff residuals ($\mathcal{R}_{\text{PDE}} = u_t + \mathbf{v} \cdot \nabla u - D \nabla^2 u$).
- Ground truth solutions are ingested from pre-computed external hydrodynamic simulation datasets (e.g. `.npz` files) via continuous spatio-temporal lookup tables rather than running forward time-stepping solvers.

## Testing & Tooling
- Use `uv` for all dependency management (`uv sync --all-extras`).
- New features and refactored modules must have corresponding unit tests in `tests/`.
- Run `pytest -v` before marking work complete.
- Run `ruff check pde_slam/ tests/` and `ruff format pde_slam/ tests/` before committing.
- Type-check with `mypy pde_slam/` for public APIs.
