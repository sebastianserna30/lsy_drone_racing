"""Nominal-Path with Sensor-Triggered Replan Controller.

A simplified, deterministic controller that computes waypoints once (or upon sensor resolution).
Avoids complex spline-refinement loops that cause oscillations.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
from dataclasses import dataclass, field

import numpy as np
from crazyflow.sim.visualize import draw_line, draw_points
from drone_models.core import load_params
from scipy.interpolate import make_interp_spline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.run_logger import RunLogger

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from crazyflow import Sim

# Tunable constants
APPROACH_DIST = 0.70         # m: distance along gate normal for entry/exit
DIP_EXIT_DIST = 0.50         # m: dip exit distance (z-clamped to clear gate frame)
DIP_NEXT_APPROACH_DIST = 1.0 # m: longer approach for gate immediately after a dip
R_SAFE = 0.30          # m: xy clearance from obstacles
DETOUR_MARGIN = 0.05   # m: extra nudge for detours
T_TOTAL = 12.0         # s: total trajectory duration
ANGLE_ALPHA = 1.0      # turning-angle time weight: segment weight += alpha * turn_angle(rad)
SENSOR_RANGE = 0.7     # m: matches environment sensor_range


@dataclass
class Waypoint:
    pos: NDArray[np.floating]
    kind: str                              # "free", "gate", "obstacle"
    source_idx: int | None = None          # index into gates/obstacles arrays
    quat: NDArray[np.floating] | None = None  # gate orientation (None for free/obstacle)
    resolved: bool = False                 # True once sensor revealed true position
    nominal_pos: NDArray[np.floating] = field(init=False)
    nominal_quat: NDArray[np.floating] | None = field(init=False)

    def __post_init__(self):
        self.nominal_pos = self.pos.copy()
        self.nominal_quat = self.quat.copy() if self.quat is not None else None


class AttitudeController_2(Controller):
    """Trajectory following controller with sensor-triggered replanning."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)
        self._freq = config.env.freq
        self._control_mode = config.env.control_mode

        drone_params = load_params(config.sim.physics, config.sim.drone_model)
        self.drone_mass = drone_params["mass"]
        self.g = 9.81
        self.kp = np.array([0.8, 0.8, 2.5])
        self.ki = np.array([0.05, 0.05, 0.05])
        self.kd = np.array([0.4, 0.4, 0.8])
        self.ki_range = np.array([2.0, 2.0, 0.4])
        self.i_error = np.zeros(3)

        # Nominal positions — read-only after init, used for distance checks and _plan
        self._nominal_gates_pos = obs["gates_pos"].copy()
        self._nominal_gates_quat = obs["gates_quat"].copy()
        self._nominal_obstacles_pos = obs["obstacles_pos"].copy()

        self._start_pos = obs["pos"].copy()
        self._tick = 0
        self._finished = False
        self._logger = RunLogger(config, rng_key=info.get("rng_key"))

        self._plan()

    def _plan(self):
        """Build full-path spline from nominal positions. Called at init and episode reset."""
        self._wps = self.compute_waypoints(
            self._start_pos,
            self._nominal_gates_pos,
            self._nominal_gates_quat,
            self._nominal_obstacles_pos,
        )
        self._fit_spline(np.array([wp.pos for wp in self._wps]), t_current=0.0)

    def _fit_spline(self, positions: NDArray[np.floating], t_current: float):
        """Fit angle-weighted spline over positions, starting at t_current."""
        dists = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        dirs = np.diff(positions, axis=0)
        dirs_unit = dirs / np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-8)
        dots = np.clip(np.sum(dirs_unit[:-1] * dirs_unit[1:], axis=1), -1.0, 1.0)
        turn_angles = np.arccos(dots)
        seg_weights = np.ones(len(dists))
        seg_weights[:-1] += ANGLE_ALPHA * turn_angles
        cumulative = np.concatenate([[0.0], np.cumsum(dists * seg_weights)])
        t = t_current + cumulative / cumulative[-1] * (T_TOTAL - t_current)
        self._spline_t0 = t[0]
        self._des_pos_spline = make_interp_spline(t, positions, k=3)
        self._des_vel_spline = self._des_pos_spline.derivative(1)

    def _push_clear(self, wp: NDArray[np.floating], obs_xy: NDArray[np.floating]) -> NDArray[np.floating]:
        """Shift wp straight away from any obstacle within R_SAFE + DETOUR_MARGIN (iterative)."""
        p = wp.copy()
        r = R_SAFE + DETOUR_MARGIN
        for _ in range(8):
            clear = True
            for o_xy in obs_xy:
                d = float(np.linalg.norm(p[:2] - o_xy))
                if d < r:
                    away = p[:2] - o_xy
                    n = float(np.linalg.norm(away))
                    away = away / n if n > 1e-6 else np.array([0.0, 1.0])
                    p[:2] += (r - d) * away
                    clear = False
            if clear:
                break
        return p

    def compute_waypoints(
        self,
        start_pos: NDArray[np.floating],
        gates_pos: NDArray[np.floating],
        gates_quat: NDArray[np.floating],
        obstacles_pos: NDArray[np.floating],
    ) -> list[Waypoint]:
        """Deterministic waypoint generation from gate/obstacle positions."""
        obs_xy = obstacles_pos[:, :2]

        #0. Start position
        base_wps: list[Waypoint] = [Waypoint(start_pos, "free")]

        #1. Initial climb: start → takeoff
        takeoff = start_pos.copy() 
        takeoff += [0, 0, 0.3]
        base_wps.append(Waypoint(takeoff, "free"))

        #2. takoff to gate0 — climb diagonally to the midpoint between takeoff and gate 0, at gate
        #    height. The z-overshoot from the climb then has distance to decay before gate 0.
        wp_to_gate0 = start_pos.copy()
        wp_to_gate0[:2] = (start_pos[:2] + gates_pos[0][:2]) / 2.0
        wp_to_gate0[2] = float(gates_pos[0][2])
        base_wps.append(Waypoint(wp_to_gate0, "free"))

        # 2. Gate passage: entry → centre → exit, with entry/exit pushed clear of obstacles.
        # If the previous gate used a dip exit (pointing toward this gate), skip this gate's
        # entry — the dip exit already sets up the approach and adding an entry creates a kink.
        prev_was_dip = False
        for i, (pos, quat) in enumerate(zip(gates_pos, gates_quat)):
            normal = R.from_quat(quat).apply([1.0, 0.0, 0.0])
            if np.dot(normal, pos - base_wps[-1].pos) < 0:
                normal = -normal

            if i + 1 < len(gates_pos):
                dir_to_next = gates_pos[i + 1] - pos
                dir_to_next /= np.linalg.norm(dir_to_next)
                is_dip = np.dot(normal, dir_to_next) < 0
            else:
                dir_to_next = None
                is_dip = False

            if is_dip:
                exit_raw = pos + DIP_EXIT_DIST * dir_to_next
                exit_raw[2] = pos[2]
                exit_ = self._push_clear(exit_raw, obs_xy)
            else:
                exit_ = self._push_clear(pos + APPROACH_DIST * normal, obs_xy)

            if prev_was_dip:
                entry = self._push_clear(pos - DIP_NEXT_APPROACH_DIST * normal, obs_xy)
                base_wps.extend([
                    Waypoint(entry, "gate", i, quat),
                    Waypoint(pos,   "gate", i, quat),
                    Waypoint(exit_, "gate", i, quat),
                ])
            else:
                entry = self._push_clear(pos - APPROACH_DIST * normal, obs_xy)
                base_wps.extend([
                    Waypoint(entry, "gate", i, quat),
                    Waypoint(pos,   "gate", i, quat),
                    Waypoint(exit_, "gate", i, quat),
                ])

            prev_was_dip = is_dip

        # 3. Segment-based obstacle detours (minimum perpendicular offset to clear)
        final_wps: list[Waypoint] = [base_wps[0]]

        for wp_from, wp_to in zip(base_wps[:-1], base_wps[1:]):
            from_xy = wp_from.pos[:2]
            to_xy   = wp_to.pos[:2]
            seg = to_xy - from_xy
            seg_len_sq = float(np.dot(seg, seg))

            detour_wp = None
            if seg_len_sq > 1e-6:
                worst_dist, worst_t, worst_obs_xy, worst_obs_idx = np.inf, None, None, None
                for obs_idx, obs_xy_i in enumerate(obs_xy):
                    t_proj = float(np.dot(obs_xy_i - from_xy, seg) / seg_len_sq)
                    if 0.0 < t_proj < 1.0:
                        closest_pt = from_xy + t_proj * seg
                        dist = float(np.linalg.norm(obs_xy_i - closest_pt))
                        if dist < R_SAFE and dist < worst_dist:
                            worst_dist, worst_t, worst_obs_xy, worst_obs_idx = dist, t_proj, obs_xy_i, obs_idx

                if worst_obs_xy is not None:
                    closest_pt = from_xy + worst_t * seg
                    perp = np.array([-seg[1], seg[0]]) / np.sqrt(seg_len_sq)
                    if np.dot(perp, closest_pt - worst_obs_xy) < 0:
                        perp = -perp
                    nudge_dist = (R_SAFE - worst_dist) + DETOUR_MARGIN
                    detour_xy = closest_pt + nudge_dist * perp
                    detour_z = wp_from.pos[2] + worst_t * (wp_to.pos[2] - wp_from.pos[2])
                    detour_wp = Waypoint(
                        np.array([detour_xy[0], detour_xy[1], detour_z]),
                        "obstacle",
                        worst_obs_idx,
                    )

            if detour_wp is not None:
                final_wps.append(detour_wp)
            final_wps.append(wp_to)

        # Prune: if a detour is immediately followed by a non-detour waypoint within
        # APPROACH_DIST, the gate entry/exit is redundant — drop it to avoid a tight kink.
        pruned_wps: list[Waypoint] = [final_wps[0]]
        for wp in final_wps[1:]:
            prev = pruned_wps[-1]
            if prev.kind == "obstacle" and wp.kind != "obstacle" and np.linalg.norm(wp.pos - prev.pos) < APPROACH_DIST:
                continue
            pruned_wps.append(wp)

        return pruned_wps

    def _replan_if_needed(self, obs: dict[str, NDArray[np.floating]]):
        """Nudge waypoints when sensor range reveals true gate/obstacle positions (Level 2)."""
        changed = False

        unresolved_gate_indices = {wp.source_idx for wp in self._wps if wp.kind == "gate" and not wp.resolved}
        for i in unresolved_gate_indices:
            if np.linalg.norm(obs["pos"] - self._nominal_gates_pos[i]) < SENSOR_RANGE:
                true_pos = obs["gates_pos"][i]
                if not np.allclose(true_pos, self._nominal_gates_pos[i], atol=1e-3):
                    delta = true_pos - self._nominal_gates_pos[i]
                    true_quat = obs["gates_quat"][i]
                    for wp in self._wps:
                        if wp.kind == "gate" and wp.source_idx == i:
                            wp.pos = wp.nominal_pos + delta
                            wp.quat = true_quat
                            wp.resolved = True
                            print(f"Updated gate {i}")
                    changed = True

        unresolved_obs_indices = {wp.source_idx for wp in self._wps if wp.kind == "obstacle" and not wp.resolved}
        for i in unresolved_obs_indices:
            if np.linalg.norm(obs["pos"] - self._nominal_obstacles_pos[i]) < SENSOR_RANGE:
                true_pos = obs["obstacles_pos"][i]
                if not np.allclose(true_pos, self._nominal_obstacles_pos[i], atol=1e-3):
                    delta = true_pos - self._nominal_obstacles_pos[i]
                    for wp in self._wps:
                        if wp.kind == "obstacle" and wp.source_idx == i:
                            wp.pos = wp.nominal_pos + delta
                            wp.resolved = True
                            print(f"Updated obstacle {i}")
                    changed = True

        if changed:
            self._fit_spline(np.array([wp.pos for wp in self._wps]), t_current=self._spline_t0)

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        t = min(self._tick / self._freq, T_TOTAL)
        if t >= T_TOTAL:
            self._finished = True

        self._replan_if_needed(obs)

        des_pos = self._des_pos_spline(t)
        des_vel = self._des_vel_spline(t)

        if self._control_mode == "state":
            action = np.zeros(13, dtype=np.float32)
            action[0:3] = des_pos
            action[3:6] = des_vel
            action[9] = 0.0
        else:
            pos_err = des_pos - obs["pos"]
            vel_err = des_vel - obs["vel"]
            self.i_error = np.clip(self.i_error + pos_err / self._freq, -self.ki_range, self.ki_range)
            thrust_vec = self.kp * pos_err + self.ki * self.i_error + self.kd * vel_err
            thrust_vec[2] += self.drone_mass * self.g
            z_axis = R.from_quat(obs["quat"]).as_matrix()[:, 2]
            thrust = thrust_vec.dot(z_axis)
            z_des = thrust_vec / np.linalg.norm(thrust_vec)
            x_c = np.array([1.0, 0.0, 0.0])
            y_des = np.cross(z_des, x_c)
            y_des /= np.linalg.norm(y_des)
            x_des = np.cross(y_des, z_des)
            rot_des = np.vstack([x_des, y_des, z_des]).T
            euler = R.from_matrix(rot_des).as_euler("xyz")
            action = np.concatenate([euler, [thrust]], dtype=np.float32)

        self._logger.log_tick(t, obs, des_pos, des_vel, action, np.array([wp.pos for wp in self._wps]))
        return action

    def step_callback(self, action, obs, reward, terminated, truncated, info):
        self._tick += 1
        self._logger.maybe_save(obs, terminated, truncated, self._finished)
        return self._finished

    def episode_callback(self):
        self.i_error[:] = 0
        self._tick = 0
        self._plan()
        self._logger.reset()

    def render_callback(self, sim: Sim):
        t = min(self._tick / self._freq, T_TOTAL)
        setpoint = self._des_pos_spline(t).reshape(1, -1)
        draw_points(sim, setpoint, rgba=(1.0, 0.0, 0.0, 1.0), size=0.02)
        traj = self._des_pos_spline(np.linspace(self._spline_t0, T_TOTAL, 100))
        draw_line(sim, traj, rgba=(0.0, 1.0, 0.0, 1.0))
        gate_wps   = np.array([wp.pos for wp in self._wps if wp.kind != "obstacle"])
        detour_wps = np.array([wp.pos for wp in self._wps if wp.kind == "obstacle"])
        if len(gate_wps):
            draw_points(sim, gate_wps,   rgba=(0.0, 0.0, 1.0, 1.0), size=0.03)
        if len(detour_wps):
            draw_points(sim, detour_wps, rgba=(1.0, 0.5, 0.0, 1.0), size=0.03)
