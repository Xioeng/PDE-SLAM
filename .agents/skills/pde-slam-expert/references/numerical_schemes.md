# Numerical Schemes — PDE-SLAM

## Advection-Diffusion Equation

```
dφ/dt + u·∇φ = D∇²φ
```

## Diffusion: 2nd-Order Central Differences

```
(∇²φ)_i,j ≈ (φ_{i,j+1} - 2φ_{i,j} + φ_{i,j-1}) / dx²
           + (φ_{i+1,j} - 2φ_{i,j} + φ_{i-1,j}) / dy²
```

Boundary condition: zero-Neumann (∂φ/∂n = 0), implemented via `jnp.pad(..., mode="edge")`.

## Advection: 1st-Order Upwind (Donor-Cell)

For x-direction, with ux > 0 (flow left → right):
```
(∂φ/∂x)_i,j ≈ (φ_{i,j} - φ_{i,j-1}) / dx   (backward difference)
```
For ux < 0 (flow right → left):
```
(∂φ/∂x)_i,j ≈ (φ_{i,j+1} - φ_{i,j}) / dx   (forward difference)
```

Selection via `jnp.where(ux >= 0, bwd, fwd)` — JIT compatible.

## Stability Conditions

### Advective CFL (Courant–Friedrichs–Lewy)
```
C = max|u| * dt / dx  <=  1
```

### Diffusion Number (von Neumann)
```
r = D * dt * (1/dx² + 1/dy²)  <=  0.5
```

Failing either bound causes instability. Use `AdvectionDiffusionSolver.courant_number()`
and `.diffusion_number()` to check at runtime.

## Upgrading Advection Scheme

To use MUSCL (2nd-order) instead of 1st-order upwind, replace `_advection_upwind`:

```python
def _advection_muscl(phi, u_field, dx, dy):
    """2nd-order MUSCL with minmod limiter."""
    def minmod(a, b):
        return jnp.where(a * b > 0, jnp.where(jnp.abs(a) < jnp.abs(b), a, b), 0.0)

    phi_p = jnp.pad(phi, 2, mode="edge")
    # slope limiting per direction ...
    # (full implementation left as extension)
```

## Time Integration Options (diffrax)

| Regime            | Recommended solver         | Notes                        |
|-------------------|----------------------------|------------------------------|
| Explicit, stable  | `diffrax.Heun`             | Default, 2nd-order           |
| Explicit, faster  | `diffrax.Euler`            | 1st-order, less accurate     |
| Stiff diffusion   | `diffrax.Kvaerno3`         | Implicit, A-stable           |
| Very stiff        | `diffrax.ImplicitEuler`    | Unconditionally stable       |
