# diffrax Patterns — PDE-SLAM

## Core diffrax call structure used in this project

```python
sol = diffrax.diffeqsolve(
    terms=diffrax.ODETerm(vector_field_fn),
    solver=diffrax.Heun(),
    t0=t0,
    t1=t_end,
    dt0=dt_initial,
    y0=phi0,
    stepsize_controller=diffrax.ConstantStepSize(),
    adjoint=diffrax.RecursiveCheckpointAdjoint(),
    max_steps=max_steps,
    saveat=diffrax.SaveAt(t1=True),  # or SaveAt(ts=[...])
)
```

## ODETerm vector field signature

diffrax requires: `vector_field(t, y, args) -> dy/dt`

In PDE-SLAM, `args` is unused (PDE params captured via closure):
```python
term = diffrax.ODETerm(
    lambda t, y, _args: _pde_rhs(t, y, pde_params, dx, dy)
)
```

## Adjoint methods

| Method                               | Memory     | Use case                        |
|--------------------------------------|------------|---------------------------------|
| `RecursiveCheckpointAdjoint()`       | O(sqrt(T)) | Default — balanced              |
| `BacksolveAdjoint()`                 | O(1)       | Very long trajectories          |
| `DirectAdjoint()`                    | O(T)       | Short trajectories, fast        |
| `NoAdjoint()`                        | O(T)       | Forward-only (no grad needed)   |

## SaveAt patterns

```python
# Only final state
saveat = diffrax.SaveAt(t1=True)
result = sol.ys[-1]   # shape (ny, nx)

# Multiple timestamps
ts = jnp.linspace(t0, t_end, 10)
saveat = diffrax.SaveAt(ts=ts)
result = sol.ys       # shape (10, ny, nx)
```

## Stepsize controllers

```python
# Fixed step (current default — predictable max_steps)
diffrax.ConstantStepSize()

# Adaptive step (better for variable-stiffness problems)
diffrax.PIDController(rtol=1e-4, atol=1e-6)
# Note: with adaptive steps, set max_steps conservatively high
```

## Gradient through solve

```python
import jax

@jax.jit
def loss(params: PDEParams) -> Array:
    phi = solver.solve(phi0, params, t0=0.0, t_end=T)
    return jnp.mean((phi - target) ** 2)

# Gradient
grad_fn = jax.grad(loss)
grads = grad_fn(params)  # PDEParams(u_field=..., D=...)

# Value + gradient
val, grads = jax.value_and_grad(loss)(params)
```

## Common pitfalls

1. **`max_steps` too small**: raises `diffrax.RESULTS.max_steps_reached`.
   Formula: `int((t_end - t0) / dt_max) + 2`.

2. **Python scalars vs JAX arrays**: `D = 0.1` (float) works in forward pass
   but breaks `jax.grad`. Always use `D = jnp.array(0.1)`.

3. **Dynamic shapes in JIT**: `saveat` timestamps must be a static-shaped array;
   do not pass a Python list of variable length inside jit.
