# RBPF-SLAM & Object-Oriented JAX Patterns — PDE-SLAM

## Object-Oriented State Management with JAX

The library uses stateful OO classes (`RbpfSlam`, `DiffDriveKinematics`, `PinnFieldMap`) with clean instance state, wrapping pure JAX mathematical functions.

### Design Pattern:

1. **Stateful Container Class**:
   Stores arrays (`self.poses`, `self.headings`, `self.xl`, `self.P`, `self.log_weights`) and PRNG keys (`self.key`).
2. **Selective `@jax.jit` Kernels**:
   Only pure, statically-shaped subroutines (like single particle transitions, batch Kalman updates, log-likelihood evaluations) are compiled with JIT.
3. **Resampling**:
   Systematic resampling is triggered conditionally when $N_{\text{eff}} < \text{threshold} \times N$.

```python
class RbpfSlam:
    def __init__(self, n_particles=100, ...):
        self.N = n_particles
        self.key = jax.random.PRNGKey(seed)
        ...

    def predict(self, control, dt):
        self.key, subkey = jax.random.split(self.key)
        # Vectorized differential drive propagation
        self.poses, self.headings, self.speeds = _predict_particles_jit(
            self.poses, self.headings, self.speeds, control, dt, subkey, self.Q_nl
        )

    def update(self, measurement, t_now=None, ...):
        # Multi-field Kalman update with map variance inflation
        ...
```
