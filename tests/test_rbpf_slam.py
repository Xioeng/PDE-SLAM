"""Unit tests for Rao-Blackwellized Particle Filter (RBPF) SLAM module."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from pde_slam.pinn import (
    PinnConfig,
    PinnFieldMap,
    pinn_forward,
    pinn_forward_mlp,
    pinn_forward_modified_mlp,
)
from pde_slam.slam import RbpfSlam, RbpfState


class TestRbpfSlamInitialization:
    def test_init_state_shape(self) -> None:
        initial_state = jnp.array([10.0, -20.0, 0.5])
        std_dev = jnp.array([0.2, 0.2, 0.02])
        n_particles = 50

        filter_obj = RbpfSlam(n_particles=n_particles, seed=0)
        filter_obj.initialize(initial_state, std_dev, n_fields=1)

        state = filter_obj.state
        assert isinstance(state, RbpfState)
        assert filter_obj.poses.shape == (50, 2)
        assert filter_obj.headings.shape == (50,)
        assert filter_obj.log_weights.shape == (50,)
        assert filter_obj.trajectories.shape == (50, 1, 2)

    def test_init_log_weights_normalized(self) -> None:
        n_particles = 100
        filter_obj = RbpfSlam(n_particles=n_particles, seed=42)
        filter_obj.initialize(jnp.zeros(3), jnp.zeros(3))

        total_weight = jnp.sum(jnp.exp(filter_obj.log_weights))
        assert float(total_weight) == pytest.approx(1.0, abs=1e-5)


class TestRbpfSlamPrediction:
    def test_predict_propagates_particles(self) -> None:
        filter_obj = RbpfSlam(
            n_particles=20,
            process_noise=jnp.diag(jnp.array([0.1, 0.01])) ** 2,
            seed=1,
        )
        filter_obj.initialize(jnp.zeros(3), jnp.zeros(3))

        filter_obj.predict(control=jnp.array([10.0, 0.0]), dt=1.0)

        assert filter_obj.poses.shape == (20, 2)
        assert filter_obj.trajectories.shape == (20, 2, 2)
        # Check forward movement along x (East) direction for 0 rad heading
        assert float(jnp.mean(filter_obj.poses[:, 0])) > 5.0


class TestRbpfSlamMeasurementUpdate:
    def test_update_measurement_normalizes_weights(self) -> None:
        filter_obj = RbpfSlam(n_particles=10, measurement_noise=1.0, seed=2)
        filter_obj.initialize(jnp.zeros(3), jnp.zeros(3))

        obs_val = jnp.array([25.0])
        predicted_vals = jnp.linspace(20.0, 30.0, 10)

        filter_obj.update(
            measurement=obs_val,
            predicted_vals=predicted_vals,
        )

        total_weight = jnp.sum(jnp.exp(filter_obj.log_weights))
        assert float(total_weight) == pytest.approx(1.0, abs=1e-5)


class TestRbpfSlamResampling:
    def test_effective_particle_number(self) -> None:
        n_particles = 10
        filter_obj = RbpfSlam(n_particles=n_particles)
        filter_obj.initialize(jnp.zeros(3), jnp.zeros(3))
        # Equal weights -> N_eff = num_particles
        filter_obj.log_weights = jnp.full((n_particles,), -jnp.log(n_particles))
        n_eff = filter_obj.effective_particle_number()
        assert float(n_eff) == pytest.approx(10.0, abs=1e-4)

    def test_resample_triggers_when_needed(self) -> None:
        n_particles = 10
        filter_obj = RbpfSlam(n_particles=n_particles, threshold_ratio=0.5, seed=3)
        filter_obj.initialize(jnp.zeros(3), jnp.zeros(3))

        # Force weight concentration on first particle
        skewed_log_w = jnp.array([0.0] + [-100.0] * 9)
        skewed_log_w = skewed_log_w - jax.scipy.special.logsumexp(skewed_log_w)
        filter_obj.log_weights = skewed_log_w

        filter_obj.resample()

        # After resampling, log weights should be reset to uniform
        expected_uniform_log_w = -jnp.log(n_particles)
        assert jnp.allclose(filter_obj.log_weights, expected_uniform_log_w, atol=1e-5)


class TestRbpfSlamBestEstimate:
    def test_get_best_estimate_returns_valid_shapes(self) -> None:
        filter_obj = RbpfSlam(
            n_particles=15,
            process_noise=jnp.zeros((2, 2)),
            seed=4,
        )
        filter_obj.initialize(jnp.zeros(3), jnp.zeros(3))

        filter_obj.predict(control=jnp.array([5.0, 0.0]), dt=1.0)

        mean_traj, mean_pose = filter_obj.get_best_estimate()
        assert mean_traj.shape == (2, 2)
        assert mean_pose.shape == (2,)


class TestRbpfSlamInstanceInterface:
    def test_rbpf_slam_instance_interface(self) -> None:
        filter_obj = RbpfSlam(
            n_particles=15,
            process_noise=jnp.diag(jnp.array([0.2, 0.02])) ** 2,
            measurement_noise=0.2**2,
            threshold_ratio=0.5,
            seed=123,
        )

        filter_obj.initialize(
            initial_state=jnp.array([5.0, 10.0, 0.1]), std_dev=jnp.zeros(3)
        )
        assert filter_obj.poses.shape == (15, 2)

        filter_obj.predict(control=jnp.array([10.0, 0.0]), dt=1.0)
        assert filter_obj.trajectories.shape == (15, 2, 2)

        preds = jnp.full((15,), 10.0)
        filter_obj.update(measurement=jnp.array([10.0]), predicted_vals=preds)
        assert float(jnp.sum(jnp.exp(filter_obj.log_weights))) == pytest.approx(
            1.0, abs=1e-5
        )

        filter_obj.resample()
        assert filter_obj.poses.shape == (15, 2)

    def test_oo_measurement_mode_switching(self) -> None:
        filter_obj = RbpfSlam(n_particles=10, seed=42)
        filter_obj.initialize(initial_state=jnp.zeros(3), std_dev=jnp.zeros(3))

        def dummy_oracle(t: float | jax.Array, poses: jax.Array) -> jax.Array:
            return jnp.full((poses.shape[0],), 15.0)

        def dummy_interpolator(poses: jax.Array) -> jax.Array:
            return jnp.full((poses.shape[0],), 15.0)

        # Initialize with oracle mode
        filter_obj.oracle_fn = dummy_oracle
        filter_obj.measurement_mode = "oracle"
        filter_obj.update(measurement=jnp.array([15.0]), t_now=1.0)
        assert float(jnp.sum(jnp.exp(filter_obj.log_weights))) == pytest.approx(
            1.0, abs=1e-5
        )

        # Easily switch mode to interpolator at runtime
        filter_obj.interpolator = dummy_interpolator
        filter_obj.measurement_mode = "interpolator"
        filter_obj.update(measurement=jnp.array([15.0]))
        assert float(jnp.sum(jnp.exp(filter_obj.log_weights))) == pytest.approx(
            1.0, abs=1e-5
        )

        # Switch to custom callable mode
        filter_obj.map_fn = lambda t, poses: jnp.full((poses.shape[0],), 15.0)
        filter_obj.measurement_mode = "custom"
        filter_obj.update(measurement=jnp.array([15.0]), t_now=2.0)
        assert float(jnp.sum(jnp.exp(filter_obj.log_weights))) == pytest.approx(
            1.0, abs=1e-5
        )

        # Switch to GP mode
        from pde_slam.interpolators.gp import GaussianProcessField

        gp = GaussianProcessField()
        gp.fit(jnp.zeros((5, 2)), jnp.full((5,), 15.0))
        filter_obj.gp_map = gp
        filter_obj.measurement_mode = "gp"
        filter_obj.update(measurement=jnp.array([15.0]))
        assert float(jnp.sum(jnp.exp(filter_obj.log_weights))) == pytest.approx(
            1.0, abs=1e-5
        )

    def test_variance_aware_update_degradation(self) -> None:
        """When variance is very high, weights should remain nearly uniform."""
        filter_obj = RbpfSlam(n_particles=10, measurement_noise=0.1**2, seed=99)
        filter_obj.initialize(initial_state=jnp.zeros(3), std_dev=jnp.zeros(3))

        # High discrepancy between particles and observation, but massive variance
        preds = jnp.linspace(0.0, 50.0, 10)
        high_vars = jnp.full((10,), 1e6)

        filter_obj.update(
            measurement=jnp.array([0.0]),
            predicted_vals=preds,
            predicted_variances=high_vars,
        )
        weights = jnp.exp(filter_obj.log_weights)
        # Weights should be uniform ~ 0.1
        assert jnp.allclose(weights, 0.1, atol=1e-3)


class TestPinnFieldMap:
    def test_normalize_inputs_bounds(self) -> None:
        config = PinnConfig(
            x_bounds=(-150.0, 150.0), y_bounds=(-150.0, 150.0), t_max=100.0
        )
        p = jnp.array([[0.0, -150.0, -150.0], [100.0, 150.0, 150.0]])
        pinn_map = PinnFieldMap(config=config)
        p_norm = pinn_map.normalize_inputs(p)

        assert jnp.allclose(p_norm[0], jnp.array([0.0, -1.0, -1.0]))
        assert jnp.allclose(p_norm[1], jnp.array([1.0, 1.0, 1.0]))

    def test_forward_output_shape(self) -> None:
        config = PinnConfig(
            x_bounds=(-150.0, 150.0),
            y_bounds=(-150.0, 150.0),
            t_max=100.0,
            hidden_dim=32,
            num_layers=3,
        )
        key = jax.random.PRNGKey(42)
        pinn_map = PinnFieldMap(config=config, key=key)
        assert pinn_map.params is not None

        p = jnp.array([[10.0, 0.0, 0.0], [20.0, 10.0, -5.0]])

        out = pinn_map.forward(p)
        assert out.shape == (2,)

    def test_pde_residual_computation(self) -> None:
        config = PinnConfig(
            x_bounds=(-150.0, 150.0),
            y_bounds=(-150.0, 150.0),
            t_max=100.0,
            v_flow_init=jnp.array([0.5, -0.2]),
            log_D_init=float(jnp.log(jnp.array(0.1))),
            hidden_dim=32,
            num_layers=3,
        )
        key = jax.random.PRNGKey(10)
        pinn_map = PinnFieldMap(config=config, key=key)
        assert pinn_map.params is not None
        res = pinn_map.pde_residual(t=10.0, x=5.0, y=-2.0)

        assert res.shape == (1,)
        assert jnp.isfinite(res)

    def test_sample_trajectory_collocation_points_bounds(self) -> None:
        key = jax.random.PRNGKey(7)
        config = PinnConfig(
            x_bounds=(-150.0, 150.0),
            y_bounds=(-150.0, 150.0),
            t_max=100.0,
            num_colloc=50,
            margin=5.0,
        )
        pinn_map = PinnFieldMap(config=config)
        traj = jnp.array([[0.0, 10.0, 20.0], [5.0, 30.0, 40.0]])

        colloc_pts = pinn_map.sample_collocation_points(traj, t_curr=10.0, key=key)

        assert colloc_pts.shape == (50, 3)
        assert jnp.all(colloc_pts[:, 0] >= 0.0) and jnp.all(colloc_pts[:, 0] <= 10.0)
        assert jnp.all(colloc_pts[:, 1] >= 5.0) and jnp.all(colloc_pts[:, 1] <= 35.0)
        assert jnp.all(colloc_pts[:, 2] >= 15.0) and jnp.all(colloc_pts[:, 2] <= 45.0)

    def test_fit_with_shuffle(self) -> None:
        key = jax.random.PRNGKey(42)
        config = PinnConfig(
            x_bounds=(-10.0, 10.0),
            y_bounds=(-10.0, 10.0),
            t_max=10.0,
            num_steps=2,
        )
        pinn_map = PinnFieldMap(config=config, key=key)
        buf_pts = jnp.array([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
        buf_vals = jnp.array([10.0, 20.0, 30.0])

        params, opt_state, loss = pinn_map.fit(buf_pts, buf_vals, key, shuffle=True)
        assert params is not None
        assert opt_state is not None
        assert isinstance(loss, float)

    def test_multi_field_pinn_map(self) -> None:
        key = jax.random.PRNGKey(42)
        config = PinnConfig(
            x_bounds=(-10.0, 10.0),
            y_bounds=(-10.0, 10.0),
            t_max=10.0,
            n_fields=3,
            log_D_init=(float(jnp.log(0.5)), float(jnp.log(1.0)), float(jnp.log(0.2))),
            num_steps=2,
        )
        pinn_map = PinnFieldMap(config=config, key=key)
        assert pinn_map.params is not None
        assert pinn_map.D is not None
        assert pinn_map.D.shape == (3,)

        p = jnp.array([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]])
        preds = pinn_map.forward(p)
        assert preds.shape == (2, 3)

        res = pinn_map.pde_residual(t=1.0, x=2.0, y=3.0)
        assert res.shape == (3,)

        buf_pts = jnp.array([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]])
        buf_vals = jnp.array([[10.0, 20.0, 30.0], [15.0, 25.0, 35.0]])
        params, opt_state, loss = pinn_map.fit(buf_pts, buf_vals, key)
        assert params.log_D.shape == (3,)
        assert isinstance(loss, float)

    def test_modified_mlp_forward_and_residual(self) -> None:
        key = jax.random.PRNGKey(42)
        config = PinnConfig(
            x_bounds=(-100.0, 100.0),
            y_bounds=(-100.0, 100.0),
            t_max=50.0,
            arch="modified_mlp",
            hidden_dim=32,
            num_layers=3,
            n_fields=2,
            log_D_init=(float(jnp.log(0.1)), float(jnp.log(0.2))),
        )
        pinn_map = PinnFieldMap(config=config, key=key)
        assert pinn_map.params is not None
        assert pinn_map.params.W_u is not None
        assert pinn_map.params.W_v is not None
        assert pinn_map.params.W_u.shape == (3, 32)
        assert pinn_map.params.W_v.shape == (3, 32)

        p = jnp.array([[0.0, 10.0, -10.0], [25.0, 0.0, 5.0]])
        preds = pinn_map.forward(p)
        assert preds.shape == (2, 2)

        res = pinn_map.pde_residual(t=10.0, x=5.0, y=-5.0)
        assert res.shape == (2,)
        assert jnp.all(jnp.isfinite(res))

    def test_modified_mlp_fit_online(self) -> None:
        key = jax.random.PRNGKey(42)
        config = PinnConfig(
            x_bounds=(-10.0, 10.0),
            y_bounds=(-10.0, 10.0),
            t_max=10.0,
            arch="modified_mlp",
            num_steps=3,
        )
        pinn_map = PinnFieldMap(config=config, key=key)
        buf_pts = jnp.array([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
        buf_vals = jnp.array([5.0, 10.0, 15.0])

        params, opt_state, loss = pinn_map.fit(buf_pts, buf_vals, key)
        assert params.W_u is not None
        assert params.W_v is not None
        assert isinstance(loss, float)

    def test_explicit_forward_functions_consistency(self) -> None:
        key = jax.random.PRNGKey(123)
        # 1. Standard MLP
        cfg_mlp = PinnConfig(arch="mlp", hidden_dim=16, num_layers=3)
        map_mlp = PinnFieldMap(config=cfg_mlp, key=key)
        assert map_mlp.params is not None

        p = jnp.array([[5.0, 1.0, -2.0]])
        out_generic = pinn_forward(map_mlp.params, p, cfg_mlp)
        out_explicit = pinn_forward_mlp(map_mlp.params, p, cfg_mlp)
        assert jnp.allclose(out_generic, out_explicit)

        # 2. Modified MLP
        cfg_mod = PinnConfig(arch="modified_mlp", hidden_dim=16, num_layers=3)
        map_mod = PinnFieldMap(config=cfg_mod, key=key)
        assert map_mod.params is not None

        out_mod_generic = pinn_forward(map_mod.params, p, cfg_mod)
        out_mod_explicit = pinn_forward_modified_mlp(map_mod.params, p, cfg_mod)
        assert jnp.allclose(out_mod_generic, out_mod_explicit)

    def test_invalid_arch_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown PINN architecture"):
            PinnFieldMap(config=PinnConfig(arch="non_existent_arch"))


class TestRbpfSlamKalmanUpdates:
    def test_linear_state_kalman_filter_update(self) -> None:
        filter_obj = RbpfSlam(
            n_particles=10,
            measurement_noise=0.1,
            p0_lin=0.1,
            lin_process_noise=1e-3,
            seed=77,
        )
        filter_obj.initialize(jnp.zeros(3), jnp.zeros(3), n_fields=1)

        assert filter_obj.xl is not None
        assert filter_obj.P is not None
        assert filter_obj.xl.shape == (10, 1)
        assert filter_obj.P.shape == (10, 1, 1)

        # Constant offset observation: obs = 10.0, map = 8.0 -> bias error = 2.0
        preds = jnp.full((10,), 8.0)
        filter_obj.update(
            measurement=jnp.array([10.0]),
            predicted_vals=preds,
            predicted_variances=jnp.zeros(10),
        )

        assert filter_obj.xl is not None
        assert filter_obj.P is not None
        assert float(jnp.mean(filter_obj.xl)) > 0.5
        # P decreases after observation: P < 0.1
        assert float(jnp.mean(filter_obj.P)) < 0.1

    def test_mixed_variance_neutral_likelihood(self) -> None:
        filter_obj = RbpfSlam(
            n_particles=10,
            measurement_noise=0.1**2,
            untrusted_var_thresh=5.0,
            enable_neutral_correction=True,
            seed=88,
        )
        filter_obj.initialize(jnp.zeros(3), jnp.zeros(3))

        # 5 particles in trusted area (var = 0.0),
        # 5 particles in untrusted area (var = 10.0)
        preds = jnp.full((10,), 0.0)
        vars_arr = jnp.array([0.0] * 5 + [10.0] * 5)

        filter_obj.update(
            measurement=jnp.array([0.0]),
            predicted_vals=preds,
            predicted_variances=vars_arr,
        )
        weights = jnp.exp(filter_obj.log_weights)
        assert float(jnp.sum(weights)) == pytest.approx(1.0, abs=1e-5)
        # Untrusted particles should not be zeroed out
        assert bool(jnp.all(weights > 0.01))

    def test_estimate_mean_and_covariance(self) -> None:
        filter_obj = RbpfSlam(n_particles=20, seed=99)
        filter_obj.initialize(jnp.array([10.0, 20.0, 0.5]), jnp.zeros(3))

        mean_state, cov_mat = filter_obj.estimate()
        assert mean_state.shape == (3,)
        assert cov_mat.shape == (2, 2)
        assert float(mean_state[0]) == pytest.approx(10.0, abs=0.5)
        assert float(mean_state[1]) == pytest.approx(20.0, abs=0.5)
