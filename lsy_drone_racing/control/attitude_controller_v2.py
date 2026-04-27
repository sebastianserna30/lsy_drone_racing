"""Nominal-Path with Sensor-Triggered Replan Controller.

A simplified, deterministic controller that computes waypoints once (or upon sensor resolution).
Avoids complex spline-refinement loops that cause oscillations.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

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
APPROACH_DIST = 0.40   # m: distance along gate normal for entry/exit
DIP_EXIT_DIST = 0.50   # m: dip exit distance (z-clamped to clear gate frame)
R_SAFE = 0.30         # m: xy clearance from obstacles
DETOUR_MARGIN = 0.05   # m: extra nudge for detours
T_TOTAL = 11.0        # s: total trajectory duration
ANGLE_ALPHA = 1.0     # turning-angle time weight: segment weight += alpha * turn_angle(rad)


class AttitudeController_2(Controller):
    """Trajectory following controller with sensor-triggered replanning."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)
        self._freq = config.env.freq
        self._control_mode = config.env.control_mode

        # Physics params for attitude control (if mode is attitude)
        drone_params = load_params(config.sim.physics, config.sim.drone_model)
        self.drone_mass = drone_params["mass"]
        self.g = 9.81
        self.kp = np.array([0.8, 0.8, 2.5])
        self.ki = np.array([0.05, 0.05, 0.05])
        self.kd = np.array([0.4, 0.4, 0.8])
        self.ki_range = np.array([2.0, 2.0, 0.4])
        self.i_error = np.zeros(3)

        # Store nominal positions (Level 2/3 info)
        self._nominal_gates_pos = obs["gates_pos"].copy()
        self._nominal_gates_quat = obs["gates_quat"].copy()
        self._nominal_obstacles_pos = obs["obstacles_pos"].copy()

        # Known positions (to be updated when sensor range is triggered)
        self._known_gates_pos = self._nominal_gates_pos.copy()
        self._known_gates_quat = self._nominal_gates_quat.copy()
        self._known_obstacles_pos = self._nominal_obstacles_pos.copy()
        self._gates_resolved = np.zeros(len(self._known_gates_pos), dtype=bool)
        self._obstacles_resolved = np.zeros(len(self._known_obstacles_pos), dtype=bool)

        self._start_pos = obs["pos"].copy()
        self._tick = 0
        self._finished = False
        self._logger = RunLogger(config, rng_key=info.get("rng_key"))

        # Initial plan
        self._replan()

    def _replan(self):
        """Re-generate waypoints and spline."""
        self._waypoints, self._is_detour = self.compute_waypoints(
            self._start_pos,
            self._known_gates_pos,
            self._known_gates_quat,
            self._known_obstacles_pos,
        )
        # Curvature-weighted time: segments leading into sharp turns get extra time budget.
        # Weight for segment i = 1 + ANGLE_ALPHA * turning_angle_at_waypoint_(i+1).
        # This slows the setpoint before each turn, reducing tracking overshoot.
        dists = np.linalg.norm(np.diff(self._waypoints, axis=0), axis=1)
        dirs = np.diff(self._waypoints, axis=0)
        dirs_unit = dirs / np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-8)
        dots = np.clip(np.sum(dirs_unit[:-1] * dirs_unit[1:], axis=1), -1.0, 1.0)
        turn_angles = np.arccos(dots)  # (N-2,): angle at each interior waypoint
        seg_weights = np.ones(len(dists))
        seg_weights[:-1] += ANGLE_ALPHA * turn_angles  # weight segment before each turn
        cumulative = np.concatenate([[0.0], np.cumsum(dists * seg_weights)])
        t = cumulative / cumulative[-1] * T_TOTAL
        self._des_pos_spline = make_interp_spline(t, self._waypoints, k=3)
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
        obstacles_pos: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], NDArray[np.bool_]]:
        """Deterministic waypoint generation from gate/obstacle positions."""
        obs_xy = obstacles_pos[:, :2]

        # 1. Takeoff — climb diagonally to the midpoint between start and gate 0, at gate
        #    height. The z-overshoot from the climb then has distance to decay before gate 0.
        takeoff = start_pos.copy()
        takeoff[:2] = (start_pos[:2] + gates_pos[0][:2]) / 2.0
        takeoff[2] = float(gates_pos[0][2])
        base_wps = [start_pos, takeoff]
        base_is_detour = [False, False]

        # 2. Gate passage: entry → centre → exit, with entry/exit pushed clear of obstacles
        for i, (pos, quat) in enumerate(zip(gates_pos, gates_quat)):
            normal = R.from_quat(quat).apply([1.0, 0.0, 0.0])
            # Orient normal in the direction of travel (from last waypoint toward this gate)
            if np.dot(normal, pos - base_wps[-1]) < 0:
                normal = -normal

            entry = self._push_clear(pos - APPROACH_DIST * normal, obs_xy)

            # Dip detection: if the next gate is behind the exit direction, the drone would
            # have to make a sharp U-turn. Place exit toward next gate instead.
            if i + 1 < len(gates_pos):
                dir_to_next = gates_pos[i + 1] - pos
                dir_to_next /= np.linalg.norm(dir_to_next)
                if np.dot(normal, dir_to_next) < 0:
                    # Dip: exit on same side as entry. Clamp z to gate_z+0.10 so the
                    # spline doesn't climb through the frame on the U-turn overshoot,
                    # while still giving a head start on the climb to the next gate.
                    exit_raw = pos + DIP_EXIT_DIST * dir_to_next
                    exit_raw[2] = pos[2]  # stay at gate height; climb happens after clearing the frame
                    exit_ = self._push_clear(exit_raw, obs_xy)
                else:
                    exit_ = self._push_clear(pos + APPROACH_DIST * normal, obs_xy)
            else:
                exit_ = self._push_clear(pos + APPROACH_DIST * normal, obs_xy)

            base_wps.extend([entry, pos, exit_])
            base_is_detour.extend([False, False, False])

        # 3. Segment-based obstacle detours (minimum perpendicular offset to clear)
        final_wps = [base_wps[0]]
        final_is_detour = [False]

        for a, b, b_det in zip(base_wps[:-1], base_wps[1:], base_is_detour[1:]):
            a_xy, b_xy = a[:2], b[:2]
            seg = b_xy - a_xy
            seg_len_sq = float(np.dot(seg, seg))

            detour = None
            if seg_len_sq > 1e-6:
                worst_d, worst_t, worst_o = np.inf, None, None
                for o_xy in obs_xy:
                    t = float(np.dot(o_xy - a_xy, seg) / seg_len_sq)
                    if 0.0 < t < 1.0:
                        closest = a_xy + t * seg
                        d = float(np.linalg.norm(o_xy - closest))
                        if d < R_SAFE and d < worst_d:
                            worst_d, worst_t, worst_o = d, t, o_xy

                if worst_o is not None:
                    closest_xy = a_xy + worst_t * seg
                    perp = np.array([-seg[1], seg[0]]) / np.sqrt(seg_len_sq)
                    if np.dot(perp, closest_xy - worst_o) < 0:
                        perp = -perp
                    nudge = (R_SAFE - worst_d) + DETOUR_MARGIN
                    detour_xy = closest_xy + nudge * perp
                    detour_z = a[2] + worst_t * (b[2] - a[2])
                    detour = np.array([detour_xy[0], detour_xy[1], detour_z])

            if detour is not None:
                final_wps.append(detour)
                final_is_detour.append(True)
            final_wps.append(b)
            final_is_detour.append(b_det)

        # Prune: if a detour is immediately followed by a non-detour waypoint within
        # APPROACH_DIST, the gate entry/exit is redundant — drop it to avoid a tight kink.
        pruned_wps = [final_wps[0]]
        pruned_det = [final_is_detour[0]]
        for wp, is_det in zip(final_wps[1:], final_is_detour[1:]):
            if pruned_det[-1] and not is_det and np.linalg.norm(wp - pruned_wps[-1]) < APPROACH_DIST:
                continue
            pruned_wps.append(wp)
            pruned_det.append(is_det)

        return np.array(pruned_wps), np.array(pruned_det, dtype=bool)

    def _replan_if_needed(self, obs: dict[str, NDArray[np.floating]]):
        """Replan when sensor range reveals a gate or obstacle's true position (Level 2)."""
        changed = False
        for i in range(len(self._known_gates_pos)):
            if obs["gates_visited"][i] and not self._gates_resolved[i]:
                self._known_gates_pos[i] = obs["gates_pos"][i]
                self._known_gates_quat[i] = obs["gates_quat"][i]
                self._gates_resolved[i] = True
                changed = True
        for i in range(len(self._known_obstacles_pos)):
            if obs["obstacles_visited"][i] and not self._obstacles_resolved[i]:
                self._known_obstacles_pos[i] = obs["obstacles_pos"][i]
                self._obstacles_resolved[i] = True
                changed = True
        if changed:
            self._replan()

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
            # 13-dim state: [x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate]
            action = np.zeros(13, dtype=np.float32)
            action[0:3] = des_pos
            action[3:6] = des_vel
            action[9] = 0.0 # yaw
        else:
            # PID Attitude mode
            pos_err = des_pos - obs["pos"]
            vel_err = des_vel - obs["vel"]
            self.i_error = np.clip(self.i_error + pos_err / self._freq, -self.ki_range, self.ki_range)
            
            thrust_vec = self.kp * pos_err + self.ki * self.i_error + self.kd * vel_err
            thrust_vec[2] += self.drone_mass * self.g
            
            z_axis = R.from_quat(obs["quat"]).as_matrix()[:, 2]
            thrust = thrust_vec.dot(z_axis)
            
            z_des = thrust_vec / np.linalg.norm(thrust_vec)
            x_c = np.array([1.0, 0.0, 0.0]) # yaw 0
            y_des = np.cross(z_des, x_c)
            y_des /= np.linalg.norm(y_des)
            x_des = np.cross(y_des, z_des)
            
            rot_des = np.vstack([x_des, y_des, z_des]).T
            euler = R.from_matrix(rot_des).as_euler("xyz")
            action = np.concatenate([euler, [thrust]], dtype=np.float32)

        self._logger.log_tick(t, obs, des_pos, des_vel, action, self._waypoints)
        return action

    def step_callback(self, action, obs, reward, terminated, truncated, info):
        self._tick += 1
        self._logger.maybe_save(obs, terminated, truncated, self._finished)
        return self._finished

    def episode_callback(self):
        self.i_error[:] = 0
        self._tick = 0
        self._known_gates_pos = self._nominal_gates_pos.copy()
        self._known_obstacles_pos = self._nominal_obstacles_pos.copy()
        self._gates_resolved[:] = False
        self._obstacles_resolved[:] = False
        self._replan()
        self._logger.reset()

    def render_callback(self, sim: Sim):
        t = min(self._tick / self._freq, T_TOTAL)
        setpoint = self._des_pos_spline(t).reshape(1, -1)
        draw_points(sim, setpoint, rgba=(1.0, 0.0, 0.0, 1.0), size=0.02)
        traj = self._des_pos_spline(np.linspace(0, T_TOTAL, 100))
        draw_line(sim, traj, rgba=(0.0, 1.0, 0.0, 1.0))
        gate_wps   = self._waypoints[~self._is_detour]
        detour_wps = self._waypoints[self._is_detour]
        if len(gate_wps):
            draw_points(sim, gate_wps,   rgba=(0.0, 0.0, 1.0, 1.0), size=0.03)
        if len(detour_wps):
            draw_points(sim, detour_wps, rgba=(1.0, 0.5, 0.0, 1.0), size=0.03)
