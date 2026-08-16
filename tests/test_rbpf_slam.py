"""Unit tests for Rao-Blackwellized Particle Filter (RBPF) SLAM module."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from pde_slam.optimization.rbpf import RbpfSlam, RbpfState
from pde_slam.pinn import PinnFieldMap


class TestRbpfSlamInitialization:

    def test_init_state_shape(self) -> None:
        key = jax.random.PRNGKey(0)
        x0 = jnp.array([10.0, -20.0])
        heading0 = 0.5
        num_particles = 50

        state = RbpfSlam.init_state(x0, heading0, num_particles, key)

        assert isinstance(state, RbpfState)
        assert state.poses.shape == (50, 2)
        assert state.headings.shape == (50,)
        assert state.log_weights.shape == (50,)
        assert state.trajectories.shape == (50, 1, 2)

    def test_init_log_weights_normalized(self) -> None:
        key = jax.random.PRNGKey(42)
        num_particles = 100
        state = RbpfSlam.init_state(jnp.zeros(2), 0.0, num_particles, key)

        total_weight = jnp.sum(jnp.exp(state.log_weights))
        pytest.approx(float(total_weight), abs=1e-5) == 1.0


class TestRbpfSlamPrediction:

    def test_predict_propagates_particles(self) -> None:
        key = jax.random.PRNGKey(1)
        k_init, k_pred = jax.random.split(key)
        state = RbpfSlam.init_state(jnp.zeros(2), 0.0, num_particles=20, key=k_init)

        new_state = RbpfSlam.predict(
            state=state,
            thrust=100.0,
            heading_cmd=0.0,
            dt=1.0,
            k_thrust=6.0,
            thrust_noise_std=0.1,
            heading_noise_std=0.01,
            key=k_pred,
        )

        assert new_state.poses.shape == (20, 2)
        assert new_state.trajectories.shape == (20, 2, 2)
        # Check forward movement along y (North) direction for 0 rad heading
        assert float(jnp.mean(new_state.poses[:, 1])) > 5.0


class TestRbpfSlamMeasurementUpdate:

    def test_update_measurement_normalizes_weights(self) -> None:
        key = jax.random.PRNGKey(2)
        state = RbpfSlam.init_state(jnp.zeros(2), 0.0, num_particles=10, key=key)

        obs_val = 25.0
        predicted_vals = jnp.linspace(20.0, 30.0, 10)

        updated_state = RbpfSlam.update_measurement(
            state=state,
            obs_val=obs_val,
            predicted_vals=predicted_vals,
            obs_std=1.0,
        )

        total_weight = jnp.sum(jnp.exp(updated_state.log_weights))
        pytest.approx(float(total_weight), abs=1e-5) == 1.0


class TestRbpfSlamResampling:

    def test_effective_particle_number(self) -> None:
        num_particles = 10
        # Equal weights -> N_eff = num_particles
        log_w = jnp.full((num_particles,), -jnp.log(num_particles))
        n_eff = RbpfSlam.effective_particle_number(log_w)
        pytest.approx(float(n_eff), abs=1e-4) == 10.0

    def test_resample_triggers_when_needed(self) -> None:
        key = jax.random.PRNGKey(3)
        num_particles = 10
        state = RbpfSlam.init_state(jnp.zeros(2), 0.0, num_particles, key=key)

        # Force weight concentration on first particle
        skewed_log_w = jnp.array([0.0] + [-100.0] * 9)
        skewed_log_w = skewed_log_w - jax.scipy.special.logsumexp(skewed_log_w)
        skewed_state = RbpfState(
            poses=state.poses,
            headings=state.headings,
            log_weights=skewed_log_w,
            trajectories=state.trajectories,
        )

        resampled_state = RbpfSlam.resample_if_needed(skewed_state, key, threshold_ratio=0.5)

        # After resampling, log weights should be reset to uniform
        expected_uniform_log_w = -jnp.log(num_particles)
        assert jnp.allclose(resampled_state.log_weights, expected_uniform_log_w, atol=1e-5)


class TestRbpfSlamBestEstimate:

    def test_get_best_estimate_returns_valid_shapes(self) -> None:
        key = jax.random.PRNGKey(4)
        k_init, k_pred = jax.random.split(key)
        state = RbpfSlam.init_state(jnp.zeros(2), 0.0, num_particles=15, key=k_init)

        state = RbpfSlam.predict(
            state=state,
            thrust=5.0,
            heading_cmd=0.0,
            dt=1.0,
            k_thrust=1.0,
            thrust_noise_std=0.0,
            heading_noise_std=0.0,
            key=k_pred,
        )

        mean_traj, mean_pose = RbpfSlam.get_best_estimate(state)
        assert mean_traj.shape == (2, 2)
        assert mean_pose.shape == (2,)


class TestPinnFieldMap:

    def test_normalize_inputs_bounds(self) -> None:
        from pde_slam.pinn import PinnDomainConfig
        config = PinnDomainConfig(x_bounds=(-150.0, 150.0), y_bounds=(-150.0, 150.0), t_max=100.0)
        p = jnp.array([[0.0, -150.0, -150.0], [100.0, 150.0, 150.0]])
        p_norm = PinnFieldMap.normalize_inputs(p, config)

        assert jnp.allclose(p_norm[0], jnp.array([0.0, -1.0, -1.0]))
        assert jnp.allclose(p_norm[1], jnp.array([1.0, 1.0, 1.0]))

    def test_forward_output_shape(self) -> None:
        from pde_slam.pinn import PinnDomainConfig
        config = PinnDomainConfig(x_bounds=(-150.0, 150.0), y_bounds=(-150.0, 150.0), t_max=100.0)
        key = jax.random.PRNGKey(42)
        params = PinnFieldMap.init_params(key, hidden_dim=32, num_layers=3)
        p = jnp.array([[10.0, 0.0, 0.0], [20.0, 10.0, -5.0]])

        out = PinnFieldMap.forward(params, p, config)
        assert out.shape == (2,)

    def test_pde_residual_computation(self) -> None:
        from pde_slam.pinn import PinnDomainConfig
        config = PinnDomainConfig(x_bounds=(-150.0, 150.0), y_bounds=(-150.0, 150.0), t_max=100.0)
        key = jax.random.PRNGKey(10)
        params = PinnFieldMap.init_params(key, v_flow_init=jnp.array([0.5, -0.2]), D_init=0.1, hidden_dim=32, num_layers=3)
        res = PinnFieldMap.pde_residual(params, config, t=10.0, x=5.0, y=-2.0)

        assert res.shape == ()
        assert jnp.isfinite(res)

    def test_sample_trajectory_collocation_points_bounds(self) -> None:
        from pde_slam.pinn import sample_trajectory_collocation_points, PinnDomainConfig
        key = jax.random.PRNGKey(7)
        config = PinnDomainConfig(x_bounds=(-150.0, 150.0), y_bounds=(-150.0, 150.0), t_max=100.0)
        traj = jnp.array([[0.0, 10.0, 20.0], [5.0, 30.0, 40.0]])

        colloc_pts = sample_trajectory_collocation_points(
            traj, t_curr=10.0, num_colloc=50, key=key, margin=5.0, config=config
        )

        assert colloc_pts.shape == (50, 3)
        assert jnp.all(colloc_pts[:, 0] >= 0.0) and jnp.all(colloc_pts[:, 0] <= 10.0)
        assert jnp.all(colloc_pts[:, 1] >= 5.0) and jnp.all(colloc_pts[:, 1] <= 35.0)
        assert jnp.all(colloc_pts[:, 2] >= 15.0) and jnp.all(colloc_pts[:, 2] <= 45.0)
