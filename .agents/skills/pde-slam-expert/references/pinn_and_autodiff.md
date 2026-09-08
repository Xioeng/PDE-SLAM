# PINN Architecture & Autodiff Residuals — PDE-SLAM

## Governing Physical Equation (2D Advection-Diffusion)

$$\frac{\partial u}{\partial t} + \mathbf{v}_{\text{flow}} \cdot \nabla u = D \nabla^2 u$$

In non-conservative 2D form:
$$\mathcal{R}(t, x, y) = \frac{\partial u}{\partial t} + v_x \frac{\partial u}{\partial x} + v_y \frac{\partial u}{\partial y} - D \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right) = 0$$

## Autodiff Residual Evaluation

Instead of forward grid time-stepping, exact continuous derivatives are evaluated with JAX automatic differentiation:

```python
def pinn_pde_residual(params, config, t=0.0, x=0.0, y=0.0):
    def scalar_fn(_t, _x, _y):
        p_val = jnp.stack([_t, _x, _y], axis=-1)
        return pinn_forward(params, p_val, config)

    dt = jax.grad(scalar_fn, argnums=0)(t, x, y)
    dx = jax.grad(scalar_fn, argnums=1)(t, x, y)
    dy = jax.grad(scalar_fn, argnums=2)(t, x, y)
    dx2 = jax.grad(lambda _t, _x, _y: jax.grad(scalar_fn, argnums=1)(_t, _x, _y), argnums=1)(t, x, y)
    dy2 = jax.grad(lambda _t, _x, _y: jax.grad(scalar_fn, argnums=2)(_t, _x, _y), argnums=2)(t, x, y)

    return dt + params.v_flow[0] * dx + params.v_flow[1] * dy - jnp.exp(params.log_D) * (dx2 + dy2)
```

## Wang & Perdikaris Modified MLP Architecture
To overcome gradient pathologies in continuous spatio-temporal PINNs:
- Computes encoder features: $U = \phi(X W_u + b_u)$, $V = \phi(X W_v + b_v)$.
- Hidden layer updates with residual gating:
  $$H^{(l+1)} = (1 - \phi(H^{(l)} W^{(l)} + b^{(l)})) \odot U + \phi(H^{(l)} W^{(l)} + b^{(l)}) \odot V$$
