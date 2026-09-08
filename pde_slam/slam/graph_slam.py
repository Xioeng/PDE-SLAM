"""
pde_slam/slam/graph_slam.py
===========================
Pose-graph SLAM: keeps a history of robot pose nodes connected by odometry
edges, detects revisits to add loop-closure edges, and periodically
re-optimizes the whole graph (Gauss-Newton on SE(2)) to correct accumulated
dead-reckoning drift.

Implemented in plain NumPy rather than JAX. The graph here is small
(hundreds-to-thousands of 3-DoF nodes) and the per-step/per-optimization work
is sequential linear algebra, not large batched array math -- exactly the
regime where JAX's eager per-op dispatch and XLA compilation overhead exceed
the actual compute cost (see the RBPF/PINN survey loop in this same package
for a worked example). NumPy has no such dispatch tax.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def wrap_angle(theta: np.ndarray | float) -> np.ndarray | float:
    """Wraps angle(s) to ``(-pi, pi]``.

    Parameters
    ----------
    theta : np.ndarray or float
        Angle(s) in radians.

    Returns
    -------
    np.ndarray or float
        Wrapped angle(s), same type/shape as input.
    """
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


@dataclass
class PositionLoopClosure:
    """A world-frame position-equality constraint between two nodes.

    Asserts nodes ``i`` and ``j`` sit at the same physical location without
    constraining their headings -- which is exactly what a scalar-field
    measurement match supports: matching readings say "same place", they say
    nothing about which way the vehicle was pointing. (A full SE(2)
    :class:`PoseEdge` would additionally pin the relative heading, which is
    unknown for a generic revisit and wrong to assert.)

    Parameters
    ----------
    i, j : int
        Node indices asserted to coincide in position.
    information : np.ndarray
        (2,2) information (inverse-covariance) matrix for the constraint.
    """

    i: int
    j: int
    information: np.ndarray


@dataclass
class PositionFactor:
    """A unary position constraint on one node: pulls it toward ``target_xy``.

    Unlike :class:`PoseEdge`, this affects only a single node's ``(x, y)`` --
    never its heading, since a scalar field reading alone doesn't observe
    orientation. Used to fold a PINN field-measurement correction (see
    :class:`~pde_slam.slam.graph_field_map.GraphFieldMapper`) into the same
    Gauss-Newton optimization as the odometry/loop-closure edges.

    Parameters
    ----------
    node_idx : int
        Node this factor applies to.
    target_xy : np.ndarray
        Target position implied by the measurement, shape ``(2,)``.
    information : np.ndarray
        (2,2) information (inverse-covariance) matrix for this factor.
    """

    node_idx: int
    target_xy: np.ndarray
    information: np.ndarray


@dataclass
class PoseEdge:
    """A relative-pose constraint between two graph nodes.

    Parameters
    ----------
    i : int
        Index of the "from" node.
    j : int
        Index of the "to" node.
    measurement : np.ndarray
        Measured relative pose ``[dx, dy, dtheta]`` of ``j`` in the frame
        of ``i``, shape ``(3,)``.
    information : np.ndarray
        Information (inverse-covariance) matrix for this edge, shape ``(3,3)``.
    kind : str, default="odometry"
        Either ``"odometry"`` (consecutive nodes) or ``"loop_closure"``.
    """

    i: int
    j: int
    measurement: np.ndarray
    information: np.ndarray
    kind: str = "odometry"


class GraphSlam:
    """Pose-graph SLAM with odometry tracking, loop-closure detection, and
    Gauss-Newton graph optimization.

    Designed as a drop-in companion to :class:`~pde_slam.slam.rbpf.RbpfSlam`
    for the RBPF survey examples: call :meth:`predict` every step with the
    same ``control``/``dt`` used elsewhere, call
    :meth:`detect_loop_closures` periodically (e.g. every K steps, or when a
    checkpoint is reached) to look for revisits, and call :meth:`optimize`
    afterwards to fold any newly found loop closures into a globally
    consistent trajectory estimate.

    Loop closures here are revisit-only: there is no scan-matching or
    feature-based sensor in this aquatic-survey setting, so a "closure" is
    declared when the current node's estimated position comes within
    ``loop_closure_radius`` of an old-enough past node, and the constraint
    asserts the two nodes coincide (identity relative pose, tight
    information) rather than supplying an independently measured relative
    pose. If a real relative-pose measurement becomes available (e.g. from
    matching two sets of field observations), pass it directly to
    :meth:`add_loop_closure_edge` instead of relying on detection.

    Parameters
    ----------
    process_noise : np.ndarray or None, default=None
        (2,2) diagonal covariance for (v, omega) used to size odometry edge
        information matrices. Defaults to ``diag([0.05, 0.02]) ** 2``.
    loop_closure_radius : float, default=2.0
        Max Euclidean distance [m] between a candidate past node and the
        current node for a loop closure to be proposed.
    min_loop_gap : int, default=20
        Minimum node-index separation before a loop closure can be proposed;
        prevents trivially closing a loop against very recent nodes.
    loop_closure_information : np.ndarray or None, default=None
        (3,3) information matrix asserted for a detected loop-closure edge.
        Defaults to a tight/confident ``diag([50, 50, 50])``.
    """

    def __init__(
        self,
        process_noise: np.ndarray | None = None,
        loop_closure_radius: float = 2.0,
        min_loop_gap: int = 20,
        loop_closure_information: np.ndarray | None = None,
    ) -> None:
        self.nodes: np.ndarray = np.zeros((0, 3))
        self.edges: list[PoseEdge] = []
        self.process_noise = (
            np.asarray(process_noise, dtype=np.float64)
            if process_noise is not None
            else np.diag([0.05, 0.02]) ** 2
        )
        self.loop_closure_radius = float(loop_closure_radius)
        self.min_loop_gap = int(min_loop_gap)
        self.loop_closure_information = (
            np.asarray(loop_closure_information, dtype=np.float64)
            if loop_closure_information is not None
            else np.diag([50.0, 50.0, 50.0])
        )
        self.loop_closures: list[tuple[int, int]] = []
        self.position_factors: dict[int, PositionFactor] = {}
        # Scalar field readings taken at each node, kept on the graph itself so
        # nodes can be associated by what was measured there rather than by
        # where the (drifting) estimate currently thinks they are.
        self.node_measurements: dict[int, np.ndarray] = {}
        self.position_loop_closures: list[PositionLoopClosure] = []

    @property
    def n_nodes(self) -> int:
        """Current number of pose nodes in the graph."""
        return self.nodes.shape[0]

    def initialize(self, initial_pose: np.ndarray) -> None:
        """Seeds the graph with a single root node and clears all edges.

        Parameters
        ----------
        initial_pose : np.ndarray
            Initial ``[x, y, theta]`` pose, shape ``(3,)``.
        """
        pose = np.asarray(initial_pose, dtype=np.float64).reshape(3)
        pose[2] = wrap_angle(pose[2])
        self.nodes = pose.reshape(1, 3)
        self.edges = []
        self.loop_closures = []
        self.position_factors = {}
        self.node_measurements = {}
        self.position_loop_closures = []

    def add_unary_position_factor(
        self, node_idx: int, target_xy: np.ndarray, information: np.ndarray
    ) -> None:
        """Sets (replacing any prior one) a unary position factor on one node.

        Each node holds at most one active position factor: recomputing it
        (e.g. as a field map's training improves) replaces the previous one
        rather than stacking multiple stale pulls on the same node.

        Parameters
        ----------
        node_idx : int
            Node this factor applies to.
        target_xy : np.ndarray
            Target position implied by the measurement, shape ``(2,)``.
        information : np.ndarray
            (2,2) information matrix for this factor.
        """
        self.position_factors[int(node_idx)] = PositionFactor(
            node_idx=int(node_idx),
            target_xy=np.asarray(target_xy, dtype=np.float64).reshape(2),
            information=np.asarray(information, dtype=np.float64).reshape(2, 2),
        )

    def predict(self, control: np.ndarray, dt: float) -> int:
        """Integrates one differential-drive odometry step, appends a new
        node, and connects it to the previous node with an odometry edge.

        Parameters
        ----------
        control : np.ndarray
            ``[v, omega]`` commanded/measured velocities.
        dt : float
            Time step [s].

        Returns
        -------
        int
            Index of the newly added node.
        """
        if self.n_nodes == 0:
            raise ValueError("Call initialize() before predict().")

        v, omega = float(control[0]), float(control[1])
        prev_pose = self.nodes[-1]
        theta_i = prev_pose[2]

        dx = v * np.cos(theta_i) * dt
        dy = v * np.sin(theta_i) * dt
        dtheta = omega * dt

        new_pose = np.array(
            [prev_pose[0] + dx, prev_pose[1] + dy, wrap_angle(theta_i + dtheta)]
        )
        self.nodes = np.vstack([self.nodes, new_pose])

        # Odometry measurement in the frame of node i: R_i^T @ [dx, dy], dtheta.
        c, s = np.cos(theta_i), np.sin(theta_i)
        local_dx = c * dx + s * dy
        local_dy = -s * dx + c * dy
        measurement = np.array([local_dx, local_dy, wrap_angle(dtheta)])

        i, j = self.n_nodes - 2, self.n_nodes - 1
        self.edges.append(
            PoseEdge(
                i=i,
                j=j,
                measurement=measurement,
                information=self._odometry_information(v, dt),
                kind="odometry",
            )
        )
        return j

    def _odometry_information(self, v: float, dt: float) -> np.ndarray:
        """Builds an odometry edge's information matrix from process noise,
        scaling positional uncertainty with the step's displacement so faster
        motion accumulates proportionally more absolute drift.
        """
        speed_scale = max(abs(v) * dt, 1e-3)
        var_xy = self.process_noise[0, 0] * speed_scale**2 + 1e-6
        var_theta = self.process_noise[1, 1] * dt**2 + 1e-6
        cov = np.diag([var_xy, var_xy, var_theta])
        return np.linalg.inv(cov)

    def add_loop_closure_edge(
        self,
        i: int,
        j: int,
        measurement: np.ndarray | None = None,
        information: np.ndarray | None = None,
    ) -> None:
        """Adds an explicit loop-closure edge between two existing nodes.

        Parameters
        ----------
        i, j : int
            Node indices being linked (order matters: measurement is the
            pose of ``j`` relative to ``i``).
        measurement : np.ndarray or None, default=None
            Relative pose ``[dx, dy, dtheta]``; defaults to identity (assumes
            the two nodes are the same physical location).
        information : np.ndarray or None, default=None
            (3,3) information matrix; defaults to ``self.loop_closure_information``.
        """
        if not (0 <= i < self.n_nodes and 0 <= j < self.n_nodes):
            raise ValueError(f"Node indices out of range: i={i}, j={j}, n={self.n_nodes}")

        meas = (
            np.asarray(measurement, dtype=np.float64)
            if measurement is not None
            else np.zeros(3)
        )
        info = (
            np.asarray(information, dtype=np.float64)
            if information is not None
            else self.loop_closure_information
        )
        self.edges.append(
            PoseEdge(i=int(i), j=int(j), measurement=meas, information=info, kind="loop_closure")
        )
        self.loop_closures.append((int(i), int(j)))

    def set_node_measurement(self, node_idx: int, values: np.ndarray) -> None:
        """Attaches the scalar field readings taken at a node to the graph.

        Parameters
        ----------
        node_idx : int
            Node the readings were taken at.
        values : np.ndarray
            Scalar readings, shape ``(n_fields,)``. Compare like-for-like
            across nodes, so pass them in consistent (ideally normalized)
            units -- :meth:`detect_measurement_loop_closures` thresholds on
            the Euclidean distance between these vectors.
        """
        self.node_measurements[int(node_idx)] = np.asarray(
            values, dtype=np.float64
        ).ravel()

    def add_position_loop_closure(
        self, i: int, j: int, information: np.ndarray | float = 1.0
    ) -> None:
        """Asserts nodes ``i`` and ``j`` occupy the same position (headings
        left free). See :class:`PositionLoopClosure`.

        Parameters
        ----------
        i, j : int
            Node indices to tie together.
        information : np.ndarray or float, default=1.0
            (2,2) information matrix, or a scalar weight applied isotropically.
        """
        info = np.asarray(information, dtype=np.float64)
        if info.ndim == 0:
            info = float(info) * np.eye(2)
        self.position_loop_closures.append(
            PositionLoopClosure(i=int(i), j=int(j), information=info.reshape(2, 2))
        )
        self.loop_closures.append((int(i), int(j)))

    def detect_measurement_loop_closures(
        self,
        current_idx: int | None = None,
        tol: float = 0.25,
        min_gap: int | None = None,
        max_radius: float | None = None,
        max_matches: int = 1,
        information: np.ndarray | float = 1.0,
    ) -> list[int]:
        """Associates nodes by what they *measured*, not by where the estimate
        currently places them, and ties the best matches together with
        position-equality constraints.

        Proximity-based detection (:meth:`detect_loop_closures`) searches the
        drifted estimate, so accumulated drift can make two unrelated nodes
        appear co-located -- a false match that then corrupts the graph.
        Matching stored scalar signatures instead is independent of the pose
        estimate. Scalar fields do alias (many places share a salinity value),
        so use several fields together and optionally gate on ``max_radius``.

        Parameters
        ----------
        current_idx : int or None, default=None
            Node to match against history; defaults to the latest node.
        tol : float, default=0.25
            Maximum Euclidean distance between measurement vectors to accept a
            match (in whatever units were passed to
            :meth:`set_node_measurement`).
        min_gap : int or None, default=None
            Minimum node-index separation; defaults to ``self.min_loop_gap``.
        max_radius : float or None, default=None
            If set, additionally require the candidate's current position
            estimate to lie within this distance -- a sanity gate against
            matching a far-away look-alike.
        max_matches : int, default=1
            Keep at most this many best (closest-signature) matches.
        information : np.ndarray or float, default=1.0
            Information for the resulting constraints.

        Returns
        -------
        list of int
            Indices of past nodes newly tied to ``current_idx``.
        """
        if current_idx is None:
            current_idx = self.n_nodes - 1
        gap = self.min_loop_gap if min_gap is None else int(min_gap)

        current_meas = self.node_measurements.get(int(current_idx))
        if current_meas is None or current_idx < gap:
            return []

        candidates: list[tuple[float, int]] = []
        current_pos = self.nodes[current_idx, :2]
        for j, meas in self.node_measurements.items():
            if j > current_idx - gap or meas.shape != current_meas.shape:
                continue
            if (int(j), int(current_idx)) in self.loop_closures:
                continue
            dist = float(np.linalg.norm(meas - current_meas))
            if dist > tol:
                continue
            if max_radius is not None:
                if float(np.linalg.norm(self.nodes[j, :2] - current_pos)) > max_radius:
                    continue
            candidates.append((dist, int(j)))

        candidates.sort()
        matched = [j for _, j in candidates[: max(0, int(max_matches))]]
        for j in matched:
            self.add_position_loop_closure(j, current_idx, information=information)
        return matched

    def detect_loop_closures(self, current_idx: int | None = None) -> list[int]:
        """Searches past nodes for revisits of the given (default: latest) node
        and adds a loop-closure edge (identity relative pose, tight
        information) for each new match found.

        Parameters
        ----------
        current_idx : int or None, default=None
            Node index to check for revisits; defaults to the most recent node.

        Returns
        -------
        list of int
            Indices of past nodes newly matched as loop closures.
        """
        if current_idx is None:
            current_idx = self.n_nodes - 1
        if current_idx < self.min_loop_gap:
            return []

        current_pos = self.nodes[current_idx, :2]
        candidate_idx = np.arange(0, current_idx - self.min_loop_gap + 1)
        if candidate_idx.size == 0:
            return []

        dists = np.linalg.norm(self.nodes[candidate_idx, :2] - current_pos, axis=1)
        matched = candidate_idx[dists <= self.loop_closure_radius]

        newly_closed: list[int] = []
        for j in matched:
            j = int(j)
            if (j, current_idx) in self.loop_closures:
                continue
            self.add_loop_closure_edge(j, current_idx)
            newly_closed.append(j)
        return newly_closed

    @staticmethod
    def _relative_pose(pose_i: np.ndarray, pose_j: np.ndarray) -> np.ndarray:
        """SE(2) relative pose of ``j`` as seen from ``i``: ``inv(pose_i) @ pose_j``."""
        theta_i = pose_i[2]
        c, s = np.cos(theta_i), np.sin(theta_i)
        dx, dy = pose_j[0] - pose_i[0], pose_j[1] - pose_i[1]
        local_dx = c * dx + s * dy
        local_dy = -s * dx + c * dy
        dtheta = wrap_angle(pose_j[2] - theta_i)
        return np.array([local_dx, local_dy, dtheta])

    def _edge_error_and_jacobians(
        self, edge: PoseEdge
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Computes the edge residual ``e = z - h(x_i, x_j)`` and its
        Jacobians ``A = de/dx_i``, ``B = de/dx_j``.
        """
        pose_i, pose_j = self.nodes[edge.i], self.nodes[edge.j]
        predicted = self._relative_pose(pose_i, pose_j)
        error = edge.measurement - predicted
        error[2] = wrap_angle(error[2])

        theta_i = pose_i[2]
        c, s = np.cos(theta_i), np.sin(theta_i)
        dx, dy = pose_j[0] - pose_i[0], pose_j[1] - pose_i[1]

        # Jacobians of h (predicted) w.r.t. pose_i / pose_j; de/dx = -dh/dx.
        dpred_dxi = np.array(
            [
                [-c, -s, -s * dx + c * dy],
                [s, -c, -c * dx - s * dy],
                [0.0, 0.0, -1.0],
            ]
        )
        dpred_dxj = np.array(
            [
                [c, s, 0.0],
                [-s, c, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        return error, -dpred_dxi, -dpred_dxj

    def optimize(
        self, max_iters: int = 10, tol: float = 1e-6, damping: float = 1e-6
    ) -> float:
        """Runs Gauss-Newton pose-graph optimization over all nodes/edges.

        The first node is anchored (given a very large information matrix)
        to fix SE(2)'s gauge freedom (the whole graph is only defined up to a
        rigid transform without it); ``self.nodes`` is updated in place.

        Parameters
        ----------
        max_iters : int, default=10
            Maximum Gauss-Newton iterations.
        tol : float, default=1e-6
            Stop early once the largest per-coordinate update falls below this.
        damping : float, default=1e-6
            Small Levenberg-style diagonal damping added to the normal
            equations for numerical stability.

        Returns
        -------
        float
            Final total weighted squared residual (0.0 if there are no edges).
        """
        n = self.n_nodes
        if n == 0 or (
            not self.edges and not self.position_factors and not self.position_loop_closures
        ):
            return 0.0

        anchor_info = np.diag([1e9, 1e9, 1e9])
        total_cost = 0.0

        for _ in range(max_iters):
            H = np.zeros((3 * n, 3 * n))
            b = np.zeros(3 * n)
            total_cost = 0.0

            H[0:3, 0:3] += anchor_info

            for edge in self.edges:
                e, A, B = self._edge_error_and_jacobians(edge)
                omega = edge.information
                i3, j3 = 3 * edge.i, 3 * edge.j

                H[i3 : i3 + 3, i3 : i3 + 3] += A.T @ omega @ A
                H[i3 : i3 + 3, j3 : j3 + 3] += A.T @ omega @ B
                H[j3 : j3 + 3, i3 : i3 + 3] += B.T @ omega @ A
                H[j3 : j3 + 3, j3 : j3 + 3] += B.T @ omega @ B

                b[i3 : i3 + 3] += A.T @ omega @ e
                b[j3 : j3 + 3] += B.T @ omega @ e

                total_cost += float(e @ omega @ e)

            # Unary position factors (e.g. PINN field-measurement corrections):
            # affect only a node's (x, y) sub-block, never its heading, since
            # de/dpose_xy = -I2 here (a direct position anchor, not a relative
            # pose constraint like the edges above).
            for pf in self.position_factors.values():
                i2 = 3 * pf.node_idx
                pose_xy = self.nodes[pf.node_idx, :2]
                e_xy = pf.target_xy - pose_xy
                omega_xy = pf.information

                H[i2 : i2 + 2, i2 : i2 + 2] += omega_xy
                b[i2 : i2 + 2] += -omega_xy @ e_xy

                total_cost += float(e_xy @ omega_xy @ e_xy)

            # Position-equality loop closures (world frame, headings free):
            # residual e = p_i - p_j should be zero, so de/dp_i = +I,
            # de/dp_j = -I over the (x, y) sub-blocks only.
            for plc in self.position_loop_closures:
                i2, j2 = 3 * plc.i, 3 * plc.j
                e_pos = self.nodes[plc.i, :2] - self.nodes[plc.j, :2]
                omega_p = plc.information

                H[i2 : i2 + 2, i2 : i2 + 2] += omega_p
                H[i2 : i2 + 2, j2 : j2 + 2] -= omega_p
                H[j2 : j2 + 2, i2 : i2 + 2] -= omega_p
                H[j2 : j2 + 2, j2 : j2 + 2] += omega_p

                b[i2 : i2 + 2] += omega_p @ e_pos
                b[j2 : j2 + 2] += -omega_p @ e_pos

                total_cost += float(e_pos @ omega_p @ e_pos)

            H += damping * np.eye(3 * n)
            dx = np.linalg.solve(H, -b)

            self.nodes = self.nodes + dx.reshape(n, 3)
            self.nodes[:, 2] = wrap_angle(self.nodes[:, 2])

            if np.max(np.abs(dx)) < tol:
                break

        return total_cost

    def get_trajectory(self) -> np.ndarray:
        """Returns the current (optimized) trajectory positions.

        Returns
        -------
        np.ndarray
            ``(N, 2)`` array of ``[x, y]`` node positions.
        """
        return self.nodes[:, :2].copy()
