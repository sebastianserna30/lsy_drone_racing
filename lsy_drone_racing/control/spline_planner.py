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
            self.waypoints = self._insert_obstacle_detours(
                self.waypoints, obstacles_pos, clearance, t_total, curvature_weight
            )
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
    def _local_min_clearance(
        p0: np.ndarray,
        d_xy: np.ndarray,
        p1: np.ndarray,
        obs_xy: np.ndarray,
        n: int = 80,
    ) -> float:
        """Min XY distance from a 3-point arc (p0→d_xy→p1) to obs_xy."""
        pts = np.stack([p0[:2], d_xy, p1[:2]])
        seg1 = np.linalg.norm(d_xy - p0[:2])
        seg2 = np.linalg.norm(p1[:2] - d_xy)
        ts = np.array([0.0, seg1, seg1 + seg2])
        if ts[-1] < 1e-9:
            return float(np.linalg.norm(d_xy - obs_xy))
        cs = CubicSpline(ts, pts, bc_type="not-a-knot")
        curve = cs(np.linspace(ts[0], ts[-1], n))
        return float(np.linalg.norm(curve - obs_xy, axis=1).min())

    @staticmethod
    def _assemble_waypoints(
        waypoints: np.ndarray,
        inserts: dict[int, np.ndarray],
        extra_i: int | None = None,
        extra_d: np.ndarray | None = None,
    ) -> np.ndarray:
        """Rebuild the full waypoint array from base waypoints + per-segment detour inserts.

        ``inserts`` maps a segment index ``i`` (between ``waypoints[i]`` and
        ``waypoints[i+1]``) to the detour point placed there. ``extra_i``/``extra_d``
        let the caller splice in one *trial* detour without mutating ``inserts`` — used
        to score candidate angles.
        """
        result = [waypoints[0]]
        for i in range(len(waypoints) - 1):
            if i in inserts:
                result.append(inserts[i])
            elif extra_i is not None and i == extra_i:
                result.append(extra_d)
            result.append(waypoints[i + 1])
        return np.array(result)

    @staticmethod
    def _find_optimal_detour(
        waypoints: np.ndarray,
        seg_i: int,
        p0: np.ndarray,
        p1: np.ndarray,
        obs_xy: np.ndarray,
        clearance: float,
        obstacles_pos: np.ndarray,
        frac_t: float,
        inserts: dict[int, np.ndarray],
        t_total: float,
        k: float,
        n_angles: int = 72,
    ) -> np.ndarray:
        """Search a continuous ring of detour positions and return the best one.

        Candidate positions lie on the clearance circle around the obstacle:
        ``d(theta) = obs_xy + clearance * [cos theta, sin theta]``. Each candidate is
        spliced into the *full* waypoint list (including detours already decided for
        earlier obstacles) and the real cubic spline is rebuilt with the same time
        allocation the final trajectory uses. The window between ``p0`` and ``p1`` is
        then sampled and scored by:

        - **loop gate (hard):** progress along the chord ``p0->p1`` must stay monotonic;
          a spline that bulges out and reverses (the obs1 "loop") is rejected. This is a
          *global* property of the curve, invisible to a local 3-point arc — which is why
          earlier local heuristics failed.
        - **clearance gate (hard):** the sampled curve must stay >= ``clearance`` from the
          avoided obstacle and clear of every other obstacle.
        - **smoothness objective (soft):** among feasible candidates, minimise the
          integral of squared curvature plus a small path-length penalty.

        No per-obstacle rules, no turn-angle threshold: obs1 picks the south side because
        the north side loops; obs3 picks the north side because the south side violates
        clearance. Both fall out of the same gates. If nothing is feasible, fall back to
        the candidate with the largest min-clearance (never crashes — MPPI's collision
        cost is the final backstop).
        """
        seg_xy = p1[:2] - p0[:2]
        seg_len = np.linalg.norm(seg_xy)
        seg_dir = seg_xy / seg_len
        detour_z = float(p0[2] + frac_t * (p1[2] - p0[2]))

        # Index of p0 in the assembled list — the trial detour sits at idx_p0+1, p1 at +2.
        idx_p0 = seg_i + sum(1 for key in inserts if key < seg_i)

        other_obs = [
            o[:2] for o in obstacles_pos if np.linalg.norm(o[:2] - obs_xy) > 1e-6
        ]

        best: tuple[float, np.ndarray] | None = None      # (objective, detour)
        fallback: tuple[float, np.ndarray] | None = None  # (min_clear, detour)

        # Sweep angle (full ring) and a few radii. The radius freedom lets a side stay
        # smooth where the tight radius=clearance position would loop — this is what
        # removes the left/right flip across the clearance slider.
        radii = clearance * np.array([1.0, 1.25, 1.5])
        for a in range(n_angles):
            theta = 2.0 * np.pi * a / n_angles
            direction = np.array([np.cos(theta), np.sin(theta)])
            for r in radii:
                d_xy = obs_xy + r * direction
                # Track bounds — infeasible if the detour leaves the arena.
                if not (-2.5 <= d_xy[0] <= 2.5 and -1.5 <= d_xy[1] <= 1.5):
                    continue
                detour = np.array([d_xy[0], d_xy[1], detour_z])

                wps = SplinePlanner._assemble_waypoints(waypoints, inserts, seg_i, detour)
                t = SplinePlanner._allocate_times(wps, t_total, k)
                spline = CubicSpline(t, wps, bc_type=((1, np.zeros(3)), (1, np.zeros(3))))

                ts = np.linspace(t[idx_p0], t[idx_p0 + 2], 120)
                pos = spline(ts)[:, :2]

                # clearance against the avoided obstacle and any others
                d_avoid = float(np.linalg.norm(pos - obs_xy, axis=1).min())
                d_other = min(
                    (float(np.linalg.norm(pos - o, axis=1).min()) for o in other_obs),
                    default=np.inf,
                )

                # loop gate: chord progress must not reverse
                s = (pos - p0[:2]) @ seg_dir
                backstep = float(-np.diff(s).min())  # largest backward step (>0 = reversal)
                looped = backstep > 0.02 * seg_len

                feasible = (
                    d_avoid >= clearance * 0.92
                    and d_other >= max(0.18, clearance * 0.7)
                    and not looped
                )

                if feasible:
                    vel = spline.derivative(1)(ts)[:, :2]
                    acc = spline.derivative(2)(ts)[:, :2]
                    speed = np.linalg.norm(vel, axis=1)
                    cross = np.abs(vel[:, 0] * acc[:, 1] - vel[:, 1] * acc[:, 0])
                    curv = cross / np.clip(speed**3, 1e-6, None)
                    arclen = float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1)))
                    obj = float(np.trapezoid(curv**2, ts)) + 0.3 * arclen
                    if best is None or obj < best[0]:
                        best = (obj, detour)
                else:
                    mc = min(d_avoid, d_other)
                    if fallback is None or mc > fallback[0]:
                        fallback = (mc, detour)

        if best is not None:
            return best[1]
        return fallback[1]  # type: ignore[union-attr]  # always set if loop ran

    @staticmethod
    def _insert_obstacle_detours(
        waypoints: np.ndarray,
        obstacles_pos: np.ndarray,
        clearance: float,
        t_total: float,
        curvature_weight: float,
        trigger_dist: float = 0.25,
    ) -> np.ndarray:
        """Insert a bypass waypoint wherever a segment passes within trigger_dist of an obstacle.

        Obstacle avoidance is purely in the XY plane (obstacles are vertical poles).
        trigger_dist: how close the segment must come to an obstacle to trigger a detour.
        clearance: minimum distance from obstacle center to the detoured curve.
        Direction and exact placement come from :meth:`_find_optimal_detour`, which
        searches a full ring of candidate positions and scores each by rebuilding the
        real spline — no fixed angle/side rules, stable across the clearance range.
        One detour per obstacle maximum. Obstacles are processed in path order so each
        detour's score sees the detours already committed earlier on the path.
        """
        detoured: set[int] = set()
        inserts: dict[int, np.ndarray] = {}
        for i in range(len(waypoints) - 1):
            p0, p1 = waypoints[i], waypoints[i + 1]
            seg_xy = p1[:2] - p0[:2]
            seg_len = np.linalg.norm(seg_xy)
            if seg_len < 0.15:
                continue
            seg_dir = seg_xy / seg_len

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
                inserts[i] = SplinePlanner._find_optimal_detour(
                    waypoints, i, p0, p1, best_obs[:2], clearance,
                    obstacles_pos, best_t, inserts, t_total, curvature_weight,
                )
                detoured.add(best_j)

        return SplinePlanner._assemble_waypoints(waypoints, inserts)

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
    def _allocate_times(
        waypoints: np.ndarray,
        t_total: float,
        k: float,
        v_max: float = 4.5,
        a_lat: float = 10.0,
        a_long: float = 12.0,
        a_brake: float = 12.0,
        n_samples: int = 300,
    ) -> np.ndarray:
        """Curvature-limited (friction-circle) segment time allocation.

        Replaces the old arc-length x (1 + k*(1-cos turn)) heuristic, which
        produced a spiky speed profile. This builds a provisional chord-length
        spline purely to *read* the path curvature, then computes the fastest
        speed profile that respects three physical limits:

        - ``a_lat``  — lateral (cornering) accel: slow speed in tight turns,
          ``v_lim = sqrt(a_lat / kappa)``, capped at ``v_max`` on straights.
        - ``a_long`` — forward accel: a *forward* pass from a standing start
          (``v=0``) limits how fast speed can build (kills the start overshoot).
        - ``a_brake``— braking decel: a *backward* pass forces braking to happen
          *before* a turn, not in it.

        Speed is then converted to time (``dt = ds / v``) and rescaled to
        ``t_total`` so it stays the hard lap-time budget. The limits are physical
        constants (tunable), NOT derived from any reference trajectory.

        ``k`` (curvature_weight) is superseded by this method and now unused; it
        is kept in the signature so existing callers don't break.
        """
        n_wp = len(waypoints)
        if n_wp < 3:
            return np.linspace(0.0, t_total, n_wp)

        # 1. Provisional chord-length parameterisation to read geometry.
        #    Default (not-a-knot) BC — NOT the final zero-velocity BC — so the
        #    parameter derivative never vanishes at the ends and curvature stays
        #    well-defined there.
        chord = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
        u_knot = np.concatenate([[0.0], np.cumsum(chord)])
        path = CubicSpline(u_knot, waypoints)

        # 2. Dense resample + per-step arc length.
        u = np.linspace(0.0, u_knot[-1], n_samples)
        pts = path(u)
        ds = np.linalg.norm(np.diff(pts, axis=0), axis=1)  # (n_samples-1,)

        # 3. Curvature  kappa = |d1 x d2| / |d1|^3.
        d1 = path(u, 1)
        d2 = path(u, 2)
        cross = np.linalg.norm(np.cross(d1, d2), axis=1)
        speed1 = np.linalg.norm(d1, axis=1)
        kappa = np.clip(cross / np.clip(speed1**3, 1e-9, None), 1e-6, None)

        # 4. Curvature-limited speed cap (friction circle).
        v_lim = np.minimum(v_max, np.sqrt(a_lat / kappa))

        # 5. Forward pass from rest:  v^2 = v0^2 + 2 a ds.
        v = v_lim.copy()
        v[0] = 0.0
        for i in range(1, n_samples):
            v[i] = min(v[i], np.sqrt(v[i - 1] ** 2 + 2.0 * a_long * ds[i - 1]))

        # 6. Backward pass for braking. No forced terminal stop: the last
        #    waypoint sits past the final gate (which the drone crosses at
        #    speed), so braking is only for interior curvature-limited turns.
        for i in range(n_samples - 2, -1, -1):
            v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2.0 * a_brake * ds[i]))

        # 7. Arc length -> time using the average speed over each step.
        v_seg = 0.5 * (v[:-1] + v[1:])
        dt = ds / np.clip(v_seg, 1e-6, None)
        t_sample = np.concatenate([[0.0], np.cumsum(dt)])

        # 8. Knot time = interpolated time at each waypoint's chord position.
        t_knot = np.interp(u_knot, u, t_sample)

        # 9. Rescale to the lap-time budget.
        if t_knot[-1] > 1e-9:
            t_knot = t_knot * (t_total / t_knot[-1])
        return t_knot


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
