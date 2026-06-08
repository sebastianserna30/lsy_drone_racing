"""Trajectory planners for drone racing: cubic spline and minimum-snap."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R


class SplinePlanner:
    """Cubic spline reference trajectory through racing gates.

    Improvements over the original:
    - Waypoints are gate centres only — no entry/exit detour, no sideways takeoff.
    - Zero-velocity boundary conditions at start and end — drone begins from rest.
    - Curvature-weighted time allocation — sharp turns get more time, straights
      less, so ``t_total`` can be pushed lower without the drone cutting corners.

    Args:
        start_pos: Drone start position, shape (3,).
        obs: Environment observation dict with key ``gates_pos``.
        t_total: Total lap time budget in seconds.
        curvature_weight: How much extra time to give sharp turns relative to
            straights. 0 = pure arc-length (old behaviour). 2 (default) means a
            90-degree turn gets roughly 2× as much time per metre as a straight.
    """

    def __init__(
        self,
        start_pos: np.ndarray,
        obs: dict,
        t_total: float = 6.2,
        curvature_weight: float = 2.0,
        obstacles_pos: np.ndarray | None = None,
        clearance: float = 0.35,
    ):
        """Build waypoints and fit a cubic spline through the race track."""
        self.t_total = t_total
        self.waypoints = self._create_waypoints(start_pos, obs)
        if obstacles_pos is not None and len(obstacles_pos) > 0:
            self.waypoints = self._insert_obstacle_detours(self.waypoints, obstacles_pos, clearance)
        t = self._allocate_times(self.waypoints, t_total, curvature_weight)
        self._pos_spline = CubicSpline(
            t, self.waypoints, bc_type=((1, np.zeros(3)), (1, np.zeros(3)))
        )
        self._vel_spline = self._pos_spline.derivative()
        self._acc_spline = self._vel_spline.derivative()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_coordinates(
        self, times: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return pos, vel, acc, yaw arrays evaluated at the given times."""
        pos = self._pos_spline(times)
        vel = self._vel_spline(times)
        acc = self._acc_spline(times)
        yaw = np.zeros((len(times), 1))
        return pos, vel, acc, yaw

    def evaluate_pos(self, t: float) -> np.ndarray:
        """Return the position at scalar time t."""
        return self._pos_spline(t)

    def get_trajectory(self, n: int = 100) -> np.ndarray:
        """Return n evenly-spaced positions along the full trajectory."""
        return self._pos_spline(np.linspace(0, self.t_total, n))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _turn_angle(d: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> float:
        """Angle at detour point d when routing p0→d→p1 (0 = straight, π = U-turn)."""
        v1, v2 = d - p0[:2], p1[:2] - d
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-9 or n2 < 1e-9:
            return float(np.pi)
        return float(np.arccos(np.clip(np.dot(v1 / n1, v2 / n2), -1.0, 1.0)))

    @staticmethod
    def _insert_obstacle_detours(
        waypoints: np.ndarray,
        obstacles_pos: np.ndarray,
        clearance: float,
        trigger_dist: float = 0.25,
    ) -> np.ndarray:
        """Insert a bypass waypoint wherever a segment passes within trigger_dist of an obstacle.

        Obstacle avoidance is purely in the XY plane (obstacles are vertical poles).
        trigger_dist: how close the segment must come to an obstacle to trigger a detour.
        clearance: distance from obstacle center to the detour waypoint.
        Direction: both sides of the segment are tried. The side with the smaller turn angle
        is preferred, UNLESS that side is nearly collinear (<5°) with the segment — meaning
        the detour barely deviates and isn't a real avoidance. In that case the other side is used.
        One detour per obstacle maximum.
        """
        detoured = set()
        result = [waypoints[0]]
        for i in range(len(waypoints) - 1):
            p0, p1 = waypoints[i], waypoints[i + 1]
            seg_xy = p1[:2] - p0[:2]
            seg_len = np.linalg.norm(seg_xy)
            if seg_len < 0.15:
                result.append(p1)
                continue
            seg_dir = seg_xy / seg_len
            perp = np.array([-seg_dir[1], seg_dir[0]])  # CCW (left) perpendicular

            best_obs, best_dist, best_t, best_j = None, np.inf, 0.5, -1
            for j, obs in enumerate(obstacles_pos):
                if j in detoured:
                    continue
                t = np.dot(obs[:2] - p0[:2], seg_dir) / seg_len
                if not (0.05 < t < 0.95):
                    continue
                closest_xy = p0[:2] + t * seg_len * seg_dir
                dist = np.linalg.norm(obs[:2] - closest_xy)
                if dist < trigger_dist and dist < best_dist:
                    best_dist, best_obs, best_t, best_j = dist, obs, t, j

            if best_obs is not None:
                d_left = best_obs[:2] + perp * clearance
                d_right = best_obs[:2] - perp * clearance
                al = SplinePlanner._turn_angle(d_left, p0, p1)
                ar = SplinePlanner._turn_angle(d_right, p0, p1)
                # If the smoother side is nearly collinear it doesn't actually avoid the
                # obstacle — switch to the other side which routes around it.
                if min(al, ar) < np.radians(5.0):
                    detour_xy = d_right if al < ar else d_left
                else:
                    detour_xy = d_left if al <= ar else d_right
                detour_z = p0[2] + best_t * (p1[2] - p0[2])
                result.append(np.array([detour_xy[0], detour_xy[1], detour_z]))
                detoured.add(best_j)

            result.append(p1)
        return np.array(result)

    @staticmethod
    def _create_waypoints(start_pos: np.ndarray, obs: dict) -> np.ndarray:
        """Entry + centre per gate; exit added only for dip gates.

        Normal gates (straight-ish approach): entry + centre (2 waypoints).
        Dip gates (next gate is in near-opposite direction, theta >= 120°):
          entry + centre + exit (3 waypoints), where the exit is placed back on
          the entry side of the gate so the drone arcs out the way it came in
          before heading to the next gate — preserving the original dip behaviour.
        """
        entry_offset = 0.23
        entry_offset_prev = 0.10
        exit_offset = 0.23
        exit_offset_next = 0.10
        dip_degree = 120
        dip_centre_shift = 0.2

        waypoints = [start_pos]
        gates_pos = obs["gates_pos"]
        gates_quat = obs["gates_quat"]

        for i, (pos, quat) in enumerate(zip(gates_pos, gates_quat)):
            normal = R.from_quat(quat).apply([1.0, 0.0, 0.0])
            vec_prev = pos - waypoints[-1]
            vec_prev_norm = vec_prev / np.linalg.norm(vec_prev)

            if i + 1 < len(gates_pos):
                vec_to_next = gates_pos[i + 1] - pos
                vec_to_next_norm = vec_to_next / np.linalg.norm(vec_to_next)
            else:
                vec_to_next_norm = vec_prev_norm

            theta = np.degrees(np.arccos(np.clip(np.dot(normal, vec_to_next_norm), -1.0, 1.0)))
            is_dip = theta >= dip_degree

            # Entry — 0.23 m in front of gate along its normal, approach side
            gate_in_dir = entry_offset_prev * vec_prev_norm
            add_normal_in = np.dot(gate_in_dir, normal) - entry_offset
            entry = pos - gate_in_dir + add_normal_in * normal
            waypoints.append(entry)

            # Centre — shifted along normal for dip gates
            centre = np.array(pos, copy=True)
            if is_dip:
                centre += normal * dip_centre_shift
            waypoints.append(centre)

            # Exit — dip gates (back through entry side) OR last gate (continue through)
            is_last = i == len(gates_pos) - 1
            if is_dip or is_last:
                gate_out_dir = exit_offset_next * vec_to_next_norm
                length_normal_out = np.dot(gate_out_dir, normal)
                if is_dip:
                    add_normal_out = exit_offset + length_normal_out  # reversed: exit entry side
                    exit_ = centre + gate_out_dir - add_normal_out * normal
                else:
                    add_normal_out = exit_offset - length_normal_out  # normal: exit far side
                    exit_ = centre + gate_out_dir + add_normal_out * normal
                waypoints.append(exit_)

        return np.array(waypoints)

    @staticmethod
    def _allocate_times(waypoints: np.ndarray, t_total: float, k: float) -> np.ndarray:
        """Arc-length + curvature-weighted segment time allocation.

        Each segment's time share is proportional to its arc length multiplied by
        a curvature weight: ``weight = 1 + k * (1 - cos(turn_angle))``.
        ``1 - cos(θ)`` is 0 for straight segments and 2 for a U-turn, so ``k``
        directly controls how aggressively turns are slowed down.
        """
        segs = np.diff(waypoints, axis=0)
        dists = np.linalg.norm(segs, axis=1)
        dirs = segs / dists[:, None]

        # Turning sharpness at each waypoint (0 at endpoints)
        n = len(waypoints)
        sharpness = np.zeros(n)
        for i in range(1, n - 1):
            cos_a = np.clip(np.dot(dirs[i - 1], dirs[i]), -1.0, 1.0)
            sharpness[i] = 1.0 - cos_a

        # Segment weight = average sharpness of its two endpoint waypoints
        seg_weights = 1.0 + k * (sharpness[:-1] + sharpness[1:]) / 2.0
        raw = dists * seg_weights
        seg_times = t_total * raw / raw.sum()
        return np.concatenate([[0.0], np.cumsum(seg_times)])


class MinSnapPlanner:
    """Minimum-snap polynomial trajectory through racing gates.

    Uses degree-7 polynomials (8 coefficients per segment) and minimises the
    integral of squared snap (4th derivative of position) — the physically correct
    cost for quadrotors because snap is proportional to the required angular
    acceleration of the rotors.

    Compared to the cubic ``SplinePlanner``:
    - Waypoints are gate centres only (no entry/exit detour → shorter path).
    - Takeoff goes straight up, not sideways.
    - Zero velocity + acceleration + jerk at start (drone begins at rest).
    - Zero velocity + acceleration at end (smooth stop / hand-off).
    - C4 continuous through interior waypoints (position through snap).
    - Time per segment is arc-length proportional (same as cubic baseline).

    Same public interface as ``SplinePlanner``: ``get_coordinates``,
    ``evaluate_pos``, ``get_trajectory``, ``t_total``, ``waypoints``.
    """

    _N = 8  # polynomial coefficients per segment (degree 7)
    _D = 4  # derivative order to minimise (snap)

    def __init__(self, start_pos: np.ndarray, obs: dict, t_total: float = 6.2):
        """Build waypoints and solve for minimum-snap polynomial coefficients."""
        self.t_total = t_total
        self.waypoints = self._build_waypoints(start_pos, obs)
        self._seg_times = self._allocate_times(self.waypoints, t_total)
        # Cumulative start time of each segment
        self._t_starts = np.concatenate([[0.0], np.cumsum(self._seg_times[:-1])])
        # coeffs shape: (n_seg, N, 3)
        self._coeffs = self._solve(self.waypoints, self._seg_times)

    # ------------------------------------------------------------------
    # Public API (identical interface to SplinePlanner)
    # ------------------------------------------------------------------

    def get_coordinates(
        self, times: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return pos, vel, acc, yaw arrays evaluated at the given times."""
        pos = self._eval(times, 0)
        vel = self._eval(times, 1)
        acc = self._eval(times, 2)
        yaw = np.zeros((len(times), 1))
        return pos, vel, acc, yaw

    def evaluate_pos(self, t: float) -> np.ndarray:
        """Return the position at scalar time t."""
        return self._eval(np.array([t]), 0)[0]

    def get_trajectory(self, n: int = 100) -> np.ndarray:
        """Return n evenly-spaced positions along the full trajectory."""
        return self._eval(np.linspace(0, self.t_total, n), 0)

    # ------------------------------------------------------------------
    # Waypoints and time allocation
    # ------------------------------------------------------------------

    def _build_waypoints(self, start_pos: np.ndarray, obs: dict) -> np.ndarray:
        """Start position + gate centres only — no entry/exit detour, no separate takeoff.

        The zero-velocity BC at start lets the polynomial lift off naturally toward
        gate 1 without a dedicated takeoff waypoint that would get too little time
        and cause polynomial overshoot.
        """
        return np.vstack([start_pos, obs["gates_pos"]])

    @staticmethod
    def _allocate_times(waypoints: np.ndarray, t_total: float) -> np.ndarray:
        dists = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
        return t_total * dists / dists.sum()

    # ------------------------------------------------------------------
    # Polynomial evaluation
    # ------------------------------------------------------------------

    def _eval(self, times: np.ndarray, deriv: int) -> np.ndarray:
        times = np.clip(np.atleast_1d(np.asarray(times, dtype=float)), 0.0, self.t_total)
        out = np.zeros((len(times), 3))
        segs = np.clip(
            np.searchsorted(self._t_starts, times, side="right") - 1, 0, len(self._seg_times) - 1
        )
        for i, (t, s) in enumerate(zip(times, segs)):
            t_loc = np.clip(t - self._t_starts[s], 0.0, self._seg_times[s])
            out[i] = self._poly_eval(self._coeffs[s], t_loc, deriv)
        return out

    @staticmethod
    def _poly_eval(coeffs: np.ndarray, t: float, deriv: int) -> np.ndarray:
        """Evaluate a degree-7 polynomial or its derivative at scalar ``t``.

        ``coeffs`` has shape ``(8, 3)``.  The polynomial is
        ``p(t) = sum_k coeffs[k] * t^k``.
        """
        n = coeffs.shape[0]
        k = np.arange(n, dtype=float)
        # Falling-factorial prefactor:  product_{j=0}^{deriv-1} (k - j)
        factors = np.ones(n)
        for j in range(deriv):
            factors *= np.maximum(k - j, 0.0)
        mask = k >= deriv
        t_pow = np.where(mask, t ** np.maximum(k - deriv, 0.0), 0.0)
        return (factors[:, None] * t_pow[:, None] * coeffs).sum(axis=0)

    # ------------------------------------------------------------------
    # QP via KKT
    # ------------------------------------------------------------------

    def _solve(self, wps: np.ndarray, seg_times: np.ndarray) -> np.ndarray:
        """Return polynomial coefficients shape (n_seg, N, 3)."""
        n_seg = len(seg_times)
        n_var = n_seg * self._N
        Q = self._cost_matrix(seg_times)
        A, b = self._constraints(wps, seg_times)
        n_con = A.shape[0]
        # KKT system: [Q A^T; A 0][c; λ] = [0; b]
        KKT = np.block([[Q, A.T], [A, np.zeros((n_con, n_con))]])
        coeffs = np.zeros((n_seg, self._N, 3))
        for d in range(3):
            rhs = np.concatenate([np.zeros(n_var), b[:, d]])
            sol = np.linalg.lstsq(KKT, rhs, rcond=None)[0]
            coeffs[:, :, d] = sol[:n_var].reshape(n_seg, self._N)
        return coeffs

    def _cost_matrix(self, seg_times: np.ndarray) -> np.ndarray:
        """Block-diagonal Q that integrates squared snap over each segment."""
        n_seg = len(seg_times)
        n_var = n_seg * self._N
        Q = np.zeros((n_var, n_var))
        for i, T in enumerate(seg_times):
            off = i * self._N
            for j in range(self._D, self._N):
                for k in range(self._D, self._N):
                    fj = float(np.prod([j - m for m in range(self._D)]))
                    fk = float(np.prod([k - m for m in range(self._D)]))
                    exp = j + k - 2 * self._D + 1
                    Q[off + j, off + k] = fj * fk * T**exp / exp
        # Small regularisation for numerical stability
        Q += 1e-8 * np.eye(n_var)
        return Q

    def _constraints(self, wps: np.ndarray, seg_times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Build equality constraint matrix A (n_con × n_var) and RHS b (n_con × 3)."""
        n_seg = len(seg_times)
        N = self._N
        D = self._D
        n_var = n_seg * N
        rows: list[np.ndarray] = []
        b_rows: list[np.ndarray] = []

        def add(row: np.ndarray, rhs: np.ndarray) -> None:
            rows.append(row)
            b_rows.append(rhs)

        # 1. Position at start of every segment  p_i(0) = wps[i]
        for i in range(n_seg):
            row = np.zeros(n_var)
            row[i * N] = 1.0  # only constant term survives at t=0
            add(row, wps[i])

        # 2. Position at end of every segment  p_i(T_i) = wps[i+1]
        for i in range(n_seg):
            row = np.zeros(n_var)
            T = seg_times[i]
            for k in range(N):
                row[i * N + k] = T**k
            add(row, wps[i + 1])

        # 3. Derivative continuity (orders 1..D) at interior junctions
        for i in range(n_seg - 1):
            T = seg_times[i]
            for deriv in range(1, D + 1):
                row = np.zeros(n_var)
                # d^deriv p_i / dt^deriv at t=T_i
                for k in range(deriv, N):
                    f = float(np.prod([k - m for m in range(deriv)]))
                    row[i * N + k] += f * T ** (k - deriv)
                # minus d^deriv p_{i+1} / dt^deriv at t=0
                f0 = float(np.prod([deriv - m for m in range(deriv)]))  # = deriv!
                row[(i + 1) * N + deriv] -= f0
                add(row, np.zeros(3))

        # 4. Zero velocity, acceleration, jerk at start
        for deriv in range(1, D):  # 1, 2, 3
            row = np.zeros(n_var)
            f = float(np.prod([deriv - m for m in range(deriv)]))  # = deriv!
            row[deriv] = f
            add(row, np.zeros(3))

        # 5. Zero velocity, acceleration at end
        T_last = seg_times[-1]
        for deriv in range(1, 3):  # 1, 2
            row = np.zeros(n_var)
            for k in range(deriv, N):
                f = float(np.prod([k - m for m in range(deriv)]))
                row[(n_seg - 1) * N + k] = f * T_last ** (k - deriv)
            add(row, np.zeros(3))

        return np.array(rows), np.array(b_rows)
