"""
test_dead_reckoning.py
======================
Unit and mathematical verification tests for DeadReckoningEstimator (Online).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pde_slam.kinematics import (
    DeadReckoningEstimator,
    compute_position_eigenvalues,
)
from pde_slam.kinematics.dead_reckoning import (
    _diff_drive_jacobians,
    _diff_drive_step,
)


class TestDeadReckoningJacobians:
    """Test analytical Jacobians against automatic differentiation."""

    def test_analytical_vs_autodiff_jacobians(self) -> None:
        x = jnp.array([10.5, -4.2, 0.75], dtype=jnp.float64)
        u = jnp.array([1.8, -0.3], dtype=jnp.float64)
        dt = 0.1

        # Analytical Jacobians
        f_analytical, g_analytical = _diff_drive_jacobians(x, u, dt)

        # Autodiff Jacobians using jax.jacfwd
        f_autodiff = jax.jacfwd(_diff_drive_step, argnums=0)(x, u, dt)
        g_autodiff = jax.jacfwd(_diff_drive_step, argnums=1)(x, u, dt)

        np.testing.assert_allclose(f_analytical, f_autodiff, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(g_analytical, g_autodiff, rtol=1e-12, atol=1e-12)


class TestDeadReckoningStateAndEigenvalues:
    """Test state accessors, eigenvalue calculations, and properties."""

    def test_eigenvalue_computation_matches_numpy(self) -> None:
        sigma_2d = jnp.array([[4.0, 1.5], [1.5, 2.0]], dtype=jnp.float64)
        l_min, l_max = compute_position_eigenvalues(sigma_2d)

        eigvals = np.linalg.eigvalsh(np.array(sigma_2d))
        np.testing.assert_allclose(float(l_min), eigvals[0], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(float(l_max), eigvals[1], rtol=1e-9, atol=1e-9)

    def test_estimator_online_stepping_metrics(self) -> None:
        q_u = jnp.diag(jnp.array([0.1**2, 0.05**2]))
        estimator = DeadReckoningEstimator(x0=1.0, y0=2.0, heading0=0.0, q_u=q_u)

        assert estimator.x_m == pytest.approx(1.0)
        assert estimator.y_m == pytest.approx(2.0)
        assert estimator.heading_rad == pytest.approx(0.0)
        assert estimator.position_variance == pytest.approx(0.0)
        assert estimator.max_eigenvalue == pytest.approx(0.0)
        assert estimator.max_std == pytest.approx(0.0)

        # Step forward with single incoming command
        dt = 0.5
        estimator.step(v=2.0, omega=0.1, dt=dt)

        assert estimator.x_m > 1.0
        assert estimator.position_variance > 0.0
        assert estimator.max_eigenvalue > 0.0
        assert estimator.max_std > 0.0
        # Largest eigenvalue must be <= trace
        assert estimator.max_eigenvalue <= estimator.position_variance
        assert estimator.max_std <= estimator.position_std

    def test_reset_and_q_u_setter(self) -> None:
        estimator = DeadReckoningEstimator(x0=0.0, y0=0.0, heading0=0.0)
        estimator.step(v=1.0, omega=0.0, dt=1.0)
        assert estimator.x_m > 0.0

        estimator.reset(x0=5.0, y0=-3.0, heading0=1.2)
        assert estimator.x_m == pytest.approx(5.0)
        assert estimator.y_m == pytest.approx(-3.0)
        assert estimator.heading_rad == pytest.approx(1.2)
        assert estimator.max_eigenvalue == pytest.approx(0.0)

        new_q = jnp.diag(jnp.array([0.04, 0.01]))
        estimator.q_u = new_q
        np.testing.assert_allclose(estimator.q_u, new_q)


class TestMonteCarloCovarianceValidationOnline:
    """Monte Carlo statistical validation of online step-by-step covariance."""

    def test_online_stepping_against_monte_carlo(self) -> None:
        x0 = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float64)
        v_sigma = 0.15
        w_sigma = 0.05
        q_u = jnp.diag(jnp.array([v_sigma**2, w_sigma**2]))
        dt = 0.1
        n_steps = 15

        v_nom = 1.5
        w_nom = 0.2

        # Step-by-step online estimator
        estimator = DeadReckoningEstimator(
            x0=float(x0[0]),
            y0=float(x0[1]),
            heading0=float(x0[2]),
            q_u=q_u,
        )

        for _ in range(n_steps):
            estimator.step(v=v_nom, omega=w_nom, dt=dt)

        p_analytical = estimator.covariance

        # Monte Carlo rollout with 20,000 particles
        n_particles = 20000
        key = jax.random.PRNGKey(42)
        k_v, k_w = jax.random.split(key, 2)

        v_samples = v_nom + v_sigma * jax.random.normal(k_v, (n_particles, n_steps))
        w_samples = w_nom + w_sigma * jax.random.normal(k_w, (n_particles, n_steps))

        def rollout(v_seq: jax.Array, w_seq: jax.Array) -> jax.Array:
            curr = x0
            for i in range(n_steps):
                curr = _diff_drive_step(curr, jnp.array([v_seq[i], w_seq[i]]), dt)
            return curr

        mc_final_states = jax.vmap(rollout)(v_samples, w_samples)
        mc_cov = jnp.cov(mc_final_states.T)

        # Propagated covariance must closely match empirical MC covariance
        np.testing.assert_allclose(
            np.array(p_analytical),
            np.array(mc_cov),
            rtol=0.08,
            atol=0.01,
        )
