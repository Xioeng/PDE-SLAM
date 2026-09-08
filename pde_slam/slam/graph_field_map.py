"""
pde_slam/slam/graph_field_map.py
=================================
Multi-field PINN mapping keyed to a GraphSlam pose graph: one independent
PinnFieldMap per scalar field, instead of a single shared multi-output
network.

Observations are stored as (graph node index, field values) rather than
baked-in (x, y) coordinates. Positions are looked up live from
``graph_slam.nodes`` every time :meth:`GraphFieldMapper.fit` runs, so a loop
closure that corrects past node positions (via ``GraphSlam.optimize()``) is
automatically reflected the next time each field's map is refit -- no
separate relabeling step is needed, since nothing here ever caches a
position, only the index used to look one up.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from pde_slam.pinn import PinnConfig, PinnFieldMap, PinnParams, pinn_forward
from pde_slam.slam.graph_slam import GraphSlam


@partial(jax.jit, static_argnames=("config",))
def _value_and_spatial_grad(
    params: PinnParams, config: PinnConfig, t: Array, xy: Array
) -> tuple[Array, Array]:
    """Single-point PINN value and its gradient w.r.t. (x, y) at fixed ``t``.

    Always called with fixed-shape inputs (a scalar ``t``, a ``(2,)`` point),
    so -- unlike the growing observation buffers elsewhere in this package --
    this compiles once and is reused every call regardless of how large the
    graph or the observation buffer has grown.
    """

    def f(xy_: Array) -> Array:
        p = jnp.array([t, xy_[0], xy_[1]])
        return pinn_forward(params, p, config)

    return jax.value_and_grad(f)(xy)


def _field_config(base_config: PinnConfig, field_idx: int) -> PinnConfig:
    """Derives a single-field ``PinnConfig`` from a shared multi-field one.

    ``n_fields`` is forced to 1. If ``log_D_init`` is a per-field tuple, the
    entry at ``field_idx`` is used as this field's scalar initial value.
    """
    log_d_init = base_config.log_D_init
    if isinstance(log_d_init, (tuple, list)):
        log_d_init = (
            float(log_d_init[field_idx]) if field_idx < len(log_d_init) else float(log_d_init[0])
        )
    return base_config._replace(n_fields=1, log_D_init=log_d_init)


class GraphFieldMapper:
    """Manages one independent :class:`~pde_slam.pinn.PinnFieldMap` per scalar
    field, tied to a :class:`~pde_slam.slam.graph_slam.GraphSlam` pose graph.

    Where the RBPF survey example uses a single multi-output PINN (one
    network, ``n_fields`` outputs) fit against the RBPF's consensus pose,
    this class gives each field its own dedicated single-output network, and
    ties every buffered observation to a graph *node index* instead of a
    fixed coordinate -- so ``fit()`` always trains against each observation's
    current (possibly loop-closure-corrected) position.

    Parameters
    ----------
    field_names : list of str
        Names of the scalar fields to map; one ``PinnFieldMap`` is created
        per name.
    pinn_config : PinnConfig
        Shared architecture/training config used as the template for each
        field's network (``n_fields`` is forced to 1 per network; if
        ``log_D_init`` is a per-field tuple, each network gets its own entry).
    key : Array
        JAX PRNG key, split once per field for independent initialization.
    """

    def __init__(self, field_names: list[str], pinn_config: PinnConfig, key: Array) -> None:
        self.field_names = list(field_names)
        keys = jax.random.split(key, len(self.field_names))
        self.field_maps: dict[str, PinnFieldMap] = {
            name: PinnFieldMap(config=_field_config(pinn_config, idx), key=k)
            for idx, (name, k) in enumerate(zip(self.field_names, keys, strict=True))
        }

        self._obs_node_idx: list[int] = []
        self._obs_time: list[float] = []
        self._obs_values: dict[str, list[float]] = {name: [] for name in self.field_names}
        self._node_to_obs_idx: dict[int, int] = {}
        # Running min-max scaler per field, and the range captured at each node.
        # Training happens in normalized units, but the *error* estimate used to
        # weight a correction must not come from the training loss: that loss is
        # in-sample (the map has memorized the very points it is scored on), so
        # it collapses toward zero and makes the factor wildly overconfident --
        # enough to overpower odometry and drag the graph to wherever a
        # drift-fitted map points. A fraction of the field's observed dynamic
        # range is a stable, data-driven stand-in that never collapses.
        self._field_min: dict[str, float] = {}
        self._field_max: dict[str, float] = {}
        self._node_scale: dict[int, dict[str, float]] = {}

        # Fixed-position anchors (e.g. known start-of-survey ground-truth
        # points): unlike node-tied observations, these never move under
        # graph optimization. Without at least a few of these, jointly fitting
        # both the map and the (drift-corrected) trajectory is underdetermined
        # -- a smooth field can be reparametrized to stay self-consistent with
        # a *wrong* trajectory (e.g. a linear field absorbing a linear drift),
        # driving training loss down without the positions actually being
        # correct. A handful of fixed anchors breaks that symmetry, the same
        # role IC anchors play in the RBPF example.
        self._fixed_pts: list[tuple[float, float, float]] = []
        self._fixed_values: dict[str, list[float]] = {name: [] for name in self.field_names}

    def add_observation(self, node_idx: int, t: float, values: dict[str, float]) -> None:
        """Buffers one multi-field observation tied to a graph node index.

        Parameters
        ----------
        node_idx : int
            Index into ``graph_slam.nodes`` for the pose this observation was
            taken at. Looked up live at fit time, never copied, so a later
            loop-closure correction to that node is automatically picked up.
        t : float
            Observation timestamp [s].
        values : dict of str to float
            Observed value for each name in ``self.field_names``.
        """
        self._node_to_obs_idx[int(node_idx)] = len(self._obs_node_idx)
        self._obs_node_idx.append(int(node_idx))
        self._obs_time.append(float(t))
        for name in self.field_names:
            self._obs_values[name].append(float(values[name]))
        self._update_scaler(values)
        self._node_scale[int(node_idx)] = self.field_ranges()

    def _update_scaler(self, values: dict[str, float]) -> None:
        """Folds one observation into the running per-field min-max scaler."""
        for name in self.field_names:
            v = float(values[name])
            self._field_min[name] = min(self._field_min.get(name, v), v)
            self._field_max[name] = max(self._field_max.get(name, v), v)

    def field_ranges(self) -> dict[str, float]:
        """Current observed dynamic range (max - min) per field."""
        return {
            name: float(self._field_max.get(name, 0.0) - self._field_min.get(name, 0.0))
            for name in self.field_names
        }

    def add_fixed_observation(
        self, position_xy: np.ndarray, t: float, values: dict[str, float]
    ) -> None:
        """Buffers one multi-field observation at a fixed, known position.

        Use this for points whose position is independently known (e.g. a
        ground-truth start-of-survey fix), not looked up from a graph node --
        see the class docstring for why a few of these matter.

        Parameters
        ----------
        position_xy : np.ndarray
            Known ``(x, y)`` position, shape ``(2,)``.
        t : float
            Observation timestamp [s].
        values : dict of str to float
            Observed value for each name in ``self.field_names``.
        """
        xy = np.asarray(position_xy, dtype=np.float64).reshape(2)
        self._fixed_pts.append((float(t), float(xy[0]), float(xy[1])))
        for name in self.field_names:
            self._fixed_values[name].append(float(values[name]))
        self._update_scaler(values)

    @property
    def n_observations(self) -> int:
        """Number of buffered observations (node-tied and fixed combined)."""
        return len(self._obs_node_idx) + len(self._fixed_pts)

    def fit(self, graph_slam: GraphSlam, key: Array) -> dict[str, float]:
        """Refits every field's PinnFieldMap using each observation's current
        position, looked up live from ``graph_slam.nodes``.

        Call this after :meth:`GraphSlam.optimize` corrects the graph (as
        well as on its own periodic schedule) so a loop closure's pose
        correction propagates into the field maps: since positions are
        re-read from the graph every call rather than cached, there is
        nothing extra to "relabel" -- the next fit is automatically
        consistent with whatever the graph currently believes.

        Parameters
        ----------
        graph_slam : GraphSlam
            Pose graph backing every buffered observation's position.
        key : Array
            JAX PRNG key, split once per field.

        Returns
        -------
        dict of str to float
            Final training loss per field name (``nan`` if there is no data
            yet).
        """
        if not self._obs_node_idx and not self._fixed_pts:
            return {name: float("nan") for name in self.field_names}

        if self._obs_node_idx:
            node_idx_arr = np.asarray(self._obs_node_idx)
            positions = graph_slam.nodes[node_idx_arr, :2]
            times = np.asarray(self._obs_time)
            dyn_pts = np.column_stack([times, positions])
        else:
            dyn_pts = np.zeros((0, 3))

        fixed_pts = np.asarray(self._fixed_pts) if self._fixed_pts else np.zeros((0, 3))
        buf_pts = np.concatenate([fixed_pts, dyn_pts], axis=0)

        keys = jax.random.split(key, len(self.field_names))
        losses: dict[str, float] = {}
        for name, k in zip(self.field_names, keys, strict=True):
            buf_vals = np.concatenate(
                [np.asarray(self._fixed_values[name]), np.asarray(self._obs_values[name])]
            )
            _, _, loss = self.field_maps[name].fit(buf_pts, buf_vals, key=k)
            losses[name] = loss
        return losses

    def compute_position_factor(
        self,
        node_idx: int,
        graph_slam: GraphSlam,
        measurement_noise_std: float | dict[str, float] = 0.05,
        field_losses: dict[str, float] | None = None,
        damping: float = 1e-3,
        max_correction: float | None = 2.0,
        map_error_frac: float = 0.05,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Computes a unary position factor for one node from every field's
        current prediction-vs-observation mismatch.

        For each field, autodiff gives the PINN's predicted value and its
        gradient w.r.t. the node's ``(x, y)`` at its current estimate; this is
        exactly a one-step Gauss-Newton linearization of "where would this
        reading actually have come from". All fields' contributions are
        summed (their Fisher information adds, same as combining independent
        measurements).

        Trusting a field at a fixed ``measurement_noise_std`` regardless of
        how well-trained it is causes a confident, badly wrong correction
        while the network is still undertrained -- its residual is then
        dominated by model error, not measurement noise, but a fixed sigma
        can't tell the difference. Passing ``field_losses`` (the dict
        returned by :meth:`fit`) fixes this: each field's *effective*
        variance is ``max(loss, measurement_noise_std**2)``, so a
        high-loss (undertrained) field contributes a soft, low-information
        correction and only sharpens as its loss drops -- which is what
        makes the correction actually improve as more field data is
        acquired, rather than overcorrecting from the very first step.

        Parameters
        ----------
        node_idx : int
            Node to compute a correction for.
        graph_slam : GraphSlam
            Pose graph whose current node position is linearized around.
        measurement_noise_std : float or dict of str to float, default=0.05
            Per-field observation noise std (floor on the effective sigma);
            a single float applies to every field.
        field_losses : dict of str to float or None, default=None
            Per-field training loss from the most recent :meth:`fit` call.
            When given, each field's effective variance is
            ``max(loss, measurement_noise_std**2)`` instead of a fixed sigma.
        damping : float, default=1e-3
            Small ridge term added to the combined information matrix before
            solving, for numerical stability when a field's gradient is
            near-zero (e.g. a spatially flat region of the map).
        max_correction : float or None, default=2.0
            Trust-region cap [m] on ``||delta_xy||`` for this single
            Gauss-Newton linearization. A single step can otherwise badly
            overshoot when a field's gradient is poorly conditioned (weak
            sensitivity in some direction) or the linearization point isn't
            yet accurate -- the same reason Levenberg-Marquardt uses a trust
            region rather than a raw Newton step. Set to ``None`` to disable.

        Returns
        -------
        tuple of (np.ndarray, np.ndarray) or None
            ``(target_xy, information)`` -- ``target_xy`` shape ``(2,)``,
            ``information`` shape ``(2,2)`` -- or ``None`` if this node has no
            buffered observation yet.
        """
        obs_idx = self._node_to_obs_idx.get(int(node_idx))
        if obs_idx is None:
            return None

        t = self._obs_time[obs_idx]
        pos_xy_np = np.asarray(graph_slam.nodes[node_idx, :2], dtype=np.float64)
        pos_xy = jnp.asarray(pos_xy_np)

        info_total = np.zeros((2, 2))
        linear_total = np.zeros(2)

        for name, fm in self.field_maps.items():
            sigma = (
                measurement_noise_std[name]
                if isinstance(measurement_noise_std, dict)
                else measurement_noise_std
            )
            min_var = sigma**2
            # Error scale from the node's stored min-max range (stable), not
            # from the in-sample training loss (which collapses to ~0 and makes
            # this factor overconfident enough to overpower odometry).
            scale = self._node_scale.get(int(node_idx), {}).get(name, 0.0)
            range_var = (map_error_frac * scale) ** 2 if scale > 0.0 else 0.0
            variance = max(range_var, min_var)
            if field_losses is not None and name in field_losses:
                # Training loss can only *widen* the estimate (a map that can't
                # even fit its own data is at least that wrong), never narrow it.
                variance = max(variance, float(field_losses[name]))

            obs_val = self._obs_values[name][obs_idx]
            value, grad_xy = _value_and_spatial_grad(fm.params, fm.config, t, pos_xy)
            residual = float(obs_val) - float(value)
            grad_np = np.asarray(grad_xy, dtype=np.float64)

            weight = 1.0 / variance
            info_total += weight * np.outer(grad_np, grad_np)
            linear_total += weight * grad_np * residual

        # Damping regularizes only the SOLVE for delta (the information matrix
        # is singular in directions no field gradient observes). It must NOT be
        # folded into the returned information: doing so would advertise
        # confidence the measurement doesn't have, acting as a spurious
        # isotropic anchor toward target_xy in exactly those unobservable
        # directions.
        delta_xy = np.linalg.solve(info_total + damping * np.eye(2), linear_total)

        if max_correction is not None:
            delta_norm = float(np.linalg.norm(delta_xy))
            if delta_norm > max_correction:
                delta_xy = delta_xy * (max_correction / delta_norm)

        target_xy = pos_xy_np + delta_xy
        return target_xy, info_total

    def apply_field_correction(
        self,
        node_idx: int,
        graph_slam: GraphSlam,
        measurement_noise_std: float | dict[str, float] = 0.05,
        field_losses: dict[str, float] | None = None,
        damping: float = 1e-3,
        max_correction: float | None = 2.0,
        map_error_frac: float = 0.05,
    ) -> bool:
        """Computes and applies (via ``graph_slam.add_unary_position_factor``)
        the current field-measurement position factor for one node.

        Parameters
        ----------
        node_idx : int
            Node to correct.
        graph_slam : GraphSlam
            Pose graph to add the resulting factor to.
        measurement_noise_std : float or dict of str to float, default=0.05
            Forwarded to :meth:`compute_position_factor`.
        field_losses : dict of str to float or None, default=None
            Forwarded to :meth:`compute_position_factor`; pass the dict
            returned by the most recent :meth:`fit` call so undertrained
            fields contribute a correspondingly soft correction.
        damping : float, default=1e-3
            Forwarded to :meth:`compute_position_factor`.
        max_correction : float or None, default=2.0
            Forwarded to :meth:`compute_position_factor`.

        Returns
        -------
        bool
            True if a factor was computed and applied; False (no-op) if this
            node has no buffered observation yet.
        """
        result = self.compute_position_factor(
            node_idx,
            graph_slam,
            measurement_noise_std,
            field_losses,
            damping,
            max_correction,
            map_error_frac,
        )
        if result is None:
            return False
        target_xy, information = result
        graph_slam.add_unary_position_factor(node_idx, target_xy, information)
        return True

    def predict(self, t: float, positions: Array) -> dict[str, Array]:
        """Predicts every field's value at the given time and positions.

        Parameters
        ----------
        t : float
            Query timestamp [s].
        positions : Array
            (N, 2) or (2,) query positions.

        Returns
        -------
        dict of str to Array
            Predicted value(s) per field name.
        """
        return {name: fm.predict(t, positions) for name, fm in self.field_maps.items()}
