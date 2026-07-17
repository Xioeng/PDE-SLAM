"""
water_features.py
=================
Utilities for generating spatial initial conditions (Gaussian plumes, multiple/random plumes)
and running the PDE solver to construct virtual sensors as spatiotemporal interpolators.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from pde_slam.interpolators.grid import SpatialGrid
from pde_slam.interpolators.spatiotemporal import SpatiotemporalInterpolator
from pde_slam.solver import AdvectionDiffusionSolver, PDEParams


def create_gaussian_plume(
    grid: SpatialGrid,
    center: tuple[float, float] | Array = (0.0, 0.0),
    width: float | Array = 10.0,
    amplitude: float | Array = 1.0,
) -> Array:
    """Evaluate a single 2-D Gaussian plume on the spatial grid.

    Parameters
    ----------
    grid : SpatialGrid
        The spatial grid on which to evaluate the plume.
    center : tuple of float or Array
        The (east, north) center coordinates of the plume [m].
    width : float or Array
        The standard deviation/width of the plume [m].
    amplitude : float or Array
        The peak amplitude/value of the plume.

    Returns
    -------
    field : Array
        2-D array of shape ``(ny, nx)`` containing the Gaussian plume.
    """
    x0, y0 = center
    XX = jnp.asarray(grid.XX)  # noqa: N806
    YY = jnp.asarray(grid.YY)  # noqa: N806

    return amplitude * jnp.exp(-((XX - x0) ** 2 + (YY - y0) ** 2) / (2.0 * width**2))


def create_random_plumes(
    grid: SpatialGrid,
    num_plumes: int = 3,
    seed: int | None = None,
) -> Array:
    """Generate a 2-D field containing multiple randomized Gaussian plumes.

    Parameters
    ----------
    grid : SpatialGrid
        The spatial grid on which to evaluate the plumes.
    num_plumes : int, optional
        The number of randomized plumes to generate (default is 3).
    seed : int, optional
        Random seed for reproducible plume parameters.

    Returns
    -------
    field : Array
        2-D array of shape ``(ny, nx)`` containing the sum of randomized plumes.
    """
    key = jax.random.PRNGKey(seed if seed is not None else 42)
    key_x, key_y, key_w, key_a = jax.random.split(key, 4)

    # Place centers within the inner 70% of the grid to prevent boundary clipping
    x_min, x_max = grid.x_min, grid.x_max
    y_min, y_max = grid.y_min, grid.y_max

    centers_x = jax.random.uniform(key_x, (num_plumes,), minval=x_min * 0.7, maxval=x_max * 0.7)
    centers_y = jax.random.uniform(key_y, (num_plumes,), minval=y_min * 0.7, maxval=y_max * 0.7)

    # Randomized width between 5% and 15% of the average grid span
    grid_span = 0.5 * ((x_max - x_min) + (y_max - y_min))
    widths = jax.random.uniform(
        key_w, (num_plumes,), minval=grid_span * 0.05, maxval=grid_span * 0.15
    )

    # Randomized amplitude between 0.5 and 1.5
    amplitudes = jax.random.uniform(key_a, (num_plumes,), minval=0.5, maxval=1.5)

    XX = jnp.asarray(grid.XX)  # noqa: N806
    YY = jnp.asarray(grid.YY)  # noqa: N806

    def single_plume(cx: Array, cy: Array, w: Array, amp: Array) -> Array:
        return amp * jnp.exp(-((XX - cx) ** 2 + (YY - cy) ** 2) / (2.0 * w**2))

    plumes_field = jax.vmap(single_plume)(centers_x, centers_y, widths, amplitudes)
    return jnp.sum(plumes_field, axis=0)


def simulate_virtual_sensor(
    grid: SpatialGrid,
    solver: AdvectionDiffusionSolver,
    phi0: Array,
    pde_params: PDEParams,
    ts: Array,
) -> SpatiotemporalInterpolator | list[SpatiotemporalInterpolator]:
    """Simulate a Passive Scalar Field using the PDE solver and wrap it as a virtual sensor.

    Parameters
    ----------
    grid : SpatialGrid
        The spatial grid of the simulation.
    solver : AdvectionDiffusionSolver
        The PDE solver instance.
    phi0 : Array
        Initial condition field of shape ``(ny, nx)`` or multiple fields of shape ``(K, ny, nx)``.
    pde_params : PDEParams
        Physical parameters (velocity field, diffusivity) for the PDE.
    ts : Array
        Sorted 1-D array of timestamps [s] at which the field is simulated.

    Returns
    -------
    sensor : SpatiotemporalInterpolator or list of SpatiotemporalInterpolator
        A spatiotemporal interpolator (or list of them) representing the simulated field(s).
    """
    ts_jax = jnp.asarray(ts, dtype=jnp.float32)
    t0_val = float(ts_jax[0])
    t_end_val = float(ts_jax[-1])

    if phi0.ndim == 3:
        # Batch solve over the first dimension (number of independent fields)
        num_pdes = phi0.shape[0]
        if pde_params.u_field.ndim == 3:
            u_fields = jnp.broadcast_to(pde_params.u_field, (num_pdes, *pde_params.u_field.shape))
        else:
            u_fields = pde_params.u_field

        D_val = pde_params.D  # noqa: N806
        if D_val.ndim == 0:
            D_val = jnp.broadcast_to(D_val, (num_pdes,))  # noqa: N806

        batched_params = PDEParams(u_field=u_fields, D=D_val)

        solve_vmap = jax.vmap(
            lambda p0, p_p: solver.solve(p0, p_p, t0=t0_val, t_end=t_end_val, saveat=ts_jax)
        )
        snapshots = solve_vmap(phi0, batched_params)  # shape (K, nt, ny, nx)
        return [SpatiotemporalInterpolator(grid, ts_jax, snap) for snap in snapshots]
    else:
        snapshots = solver.solve(phi0, pde_params, t0=t0_val, t_end=t_end_val, saveat=ts_jax)
        return SpatiotemporalInterpolator(grid, ts_jax, snapshots)
