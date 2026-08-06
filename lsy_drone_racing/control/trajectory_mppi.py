"""TODO."""

from __future__ import annotations  # Python 3.10 type hints

import os
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from crazyflow.control import Control
from crazyflow.sim import Physics, Sim
from crazyflow.sim.visualize import draw_line, draw_points

# from crazyflow_experiments.sim2real.control.trajectory_generator import (
#    TrajectoryGenerator3DPeriodicMotion,
# )
from drone_models.core import load_params
from jax import random, vmap
from jax.lax import scan
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.spline_planner import SplinePlanner

HOVER_THRUST = 0.43  # collective thrust (N) that approximately balances gravity for cf21B_500

if TYPE_CHECKING:
    from crazyflow.sim.data import SimData
    from numpy.typing import NDArray


class AttitudeMPPIController(Controller):
    """Multi-modal MPPI attitude controller for drone racing."""

    def get_gate_frame_pos(
        self, gates_pos: NDArray[np.floating], gates_quat: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Returns frame-centre possitions for each gate.

        Output shape (n_gates*2, 3)
        """
        n_gates = gates_pos.shape[0]
        gate_frame_pos = np.zeros((n_gates * 2, 3))

        for i in range(n_gates):
            rotation = R.from_quat(gates_quat[i])

            side_axis = rotation.apply([0.0, 1.0, 0.0])

            gate_frame_pos[2 * i] = gates_pos[i] - 0.28 * side_axis
            gate_frame_pos[2 * i + 1] = gates_pos[i] + 0.28 * side_axis

        return gate_frame_pos

    def __init__(
        self, initial_obs: dict[str, NDArray[np.floating]], info: dict, initial_info: dict
    ):
        """Initialize the MPPI controller.

        Args:
            initial_obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: Additional environment information from the reset.
            initial_info: seems to be config: The configuration of the environment.
        """
        super().__init__(initial_obs, info, initial_info)

        self.initial_obs = initial_obs
        self.initial_info = initial_info

        ctrl_cfg = initial_info["controller"]
        mppi_cfg = ctrl_cfg["mppi"]

        self.N = mppi_cfg["N"]
        self.T = mppi_cfg["T"]
        self.dt = self.T / self.N
        self.dt_array = jnp.arange(0, self.T, self.dt)
        self.f = int(self.N / self.T)
        assert np.isclose(self.f, self.N / self.T), (
            "N must be divisible by T for consistent time steps"
        )
        self.ctrl_dt = 1 / ctrl_cfg["ctrl_freq"]
        self.n_samples = mppi_cfg["n_samples"]
        self.K = mppi_cfg["K"]
        self.M = self.n_samples // self.K  # samples per mode
        # changedPractical: sampled control dim is now 5 = [roll, pitch, yaw, thrust, v_theta].
        # The sim only ever receives the first 4; v_theta is the MPCC progress speed integrated
        # in the rollout. Use self.num_inputs wherever the control dim was previously hardcoded as 4
        self.num_inputs = 5

        # changedPractical
        self._t = 0.0

        if jax.default_backend() == "cpu":
            available_device = "cpu"
        else:
            available_device = "cuda"

        self.sim = Sim(
            n_worlds=self.n_samples,
            n_drones=1,
            attitude_freq=self.f,
            freq=self.f,
            physics=Physics.so_rpy_rotor_drag,
            control=Control.attitude,
            drone_model="cf21B_500",
            device=available_device,  # TODO get from info
        )
        self.sim.reset()

        self.step_fn = self.sim.build_step_fn()

        self.noise_sigmas = jnp.full(
            (self.K, self.N, self.num_inputs), fill_value=mppi_cfg["noise_sigma"], 
            device=self.sim.device
        )
        self.temperature = mppi_cfg["temperature"]
        self.elite_percentage = mppi_cfg["elite_percentage"]
        self.beta = mppi_cfg["beta"]
        self.alpha = mppi_cfg["alpha"]
        self.min_variance = mppi_cfg["min_variance"]

        # changedPractical: cost-function weights and spline params, read from config so they can
        # be tuned without editing code. Defaults match the previously hardcoded values, so configs
        # lacking these keys behave identically. Scalars are baked in as trace-time constants when
        # compute_cost is JIT-traced (same pattern as obstacle_radius/drone_radius below).
        cost_cfg = mppi_cfg.get("cost", {})
        self.w_pos = float(cost_cfg.get("pos", 40.0))
        self.w_z = float(cost_cfg.get("z", 80.0))  # extra penalty for altitude error
        self.w_vel = float(cost_cfg.get("vel", 1.0))
        self.w_ang_vel = float(cost_cfg.get("ang_vel", 0.0))
        self.w_ang_acc = float(cost_cfg.get("ang_acc", 0.0))
        self.w_tilt = float(
            cost_cfg.get("tilt", 1.0)
        )  # was 5.0; loosened to allow aggressive roll/pitch
        self.w_thrust = float(
            cost_cfg.get("thrust", 1.0)
        )  # was 0.0; regularise toward hover thrust
        self.w_yaw = float(cost_cfg.get("yaw", 2.0))  # was 0.0; added to stabilise yaw oscillation
        self.w_obstacle = float(cost_cfg.get("obstacle", 1000.0))
        self.w_obstacle_exp = float(cost_cfg.get("obstacle_exp", 20.0))
        self.w_opp_drone = float(cost_cfg.get("opp_drone", 2000.0))
        self.w_opp_drone_exp = float(cost_cfg.get("opp_drone_exp", 2000.0))
        self.w_floor = float(cost_cfg.get("floor", 500.0))
        self.floor_z = float(cost_cfg.get("floor_z", 0.1))
        # changedPractical: MPCC contour/lag/progress weights. The reference is
        # reparameterized from time t to arc-length progress theta (see _build_theta_lut).
        # Each rollout carries its own theta, advanced by a sampled v_theta (5th control
        # channel). pos error is split into lag (along path tangent) and contour
        # (perpendicular); progress reward (-w_progress * v_theta) pushes the drone forward.
        self.w_contour = float(cost_cfg.get("contour", 40.0))  # off-path penalty
        self.w_lag = float(cost_cfg.get("lag", 800.0))  # ahead/behind-of-theta penalty (heavy)
        self.w_progress = float(cost_cfg.get("progress", 2.0))  # mu: reward for advancing theta
        spline_cfg = mppi_cfg.get("spline", {})
        self._spline_t_total = float(spline_cfg.get("t_total", 4.1))
        self._spline_curvature_weight = float(spline_cfg.get("curvature_weight", 2.0))
        self._spline_clearance = float(spline_cfg.get("clearance", 0.22))
        self._lut_samples = int(spline_cfg.get("lut_samples", 1500))
        # changedPractical: v_theta (progress speed, m/s) sampling params. Separate from the
        # attitude/thrust noise_sigma because it has different units. Bounds: never reverse
        # (>=0), cap above the planner's v_max (4.5) for headroom.
        self.v_theta_init = float(mppi_cfg.get("v_theta_init", 2.0))  # ~s_total/t_total
        self.v_theta_max = float(mppi_cfg.get("v_theta_max", 6.0))
        self.v_theta_sigma = float(mppi_cfg.get("v_theta_sigma", 0.5))
        # v_theta channel gets its own (larger) noise scale — different units from rad/thrust
        self.noise_sigmas = self.noise_sigmas.at[:, :, 4].set(self.v_theta_sigma)

        # changedPractical: initialise thrust channel to hover so MPPI starts from a stable
        # baseline, and v_theta channel to a nominal forward progress speed
        _init = jnp.zeros((self.K, self.N, self.num_inputs), device=self.sim.device)
        _init = _init.at[:, :, 3].set(HOVER_THRUST)
        self.mean_controls = _init.at[:, :, 4].set(self.v_theta_init)

        # Shape: (Num_Obstacles, 3)
        # changedPractical
        self.obstacles = jnp.array(initial_obs["obstacles_pos"], device=self.sim.device)
        gate_frame_pos = self.get_gate_frame_pos(
            initial_obs["gates_pos"], initial_obs["gates_quat"]
        )
        self.gate_frame_obstacles = jnp.array(gate_frame_pos, device=self.sim.device)

        # changedPractical
        # self.low_level_ctrl_freq = initial_info["low_level_ctrl_freq"]
        # self.drone_params = load_params("first_principles", initial_info["drone_model"])
        self.drone_params = load_params("first_principles", initial_info["drone"]["model"])
        self.drone_mass = self.drone_params["mass"]
        # changedPractical: 5-dim bounds — channel 4 is v_theta (progress speed, m/s),
        # bounded [0, v_theta_max] so the drone never runs the reference backward
        self.act_low = -jnp.ones(self.num_inputs, device=self.sim.device) * jnp.pi / 2
        self.act_low = self.act_low.at[3].set(self.drone_params["thrust_min"] * 4)
        self.act_low = self.act_low.at[4].set(0.0)
        self.act_high = jnp.ones(self.num_inputs, device=self.sim.device) * jnp.pi / 2
        self.act_high = self.act_high.at[3].set(self.drone_params["thrust_max"] * 4)
        self.act_high = self.act_high.at[4].set(self.v_theta_max)
        self.thrust = np.zeros(4)
        # changedPractical: EMA filter on executed action to damp mode-switching oscillations
        self._prev_action = np.array([0.0, 0.0, 0.0, HOVER_THRUST])  # [roll, pitch, yaw, thrust]
        self._action_ema = float(
            mppi_cfg.get("action_ema", 0.4)
        )  # blend: ema * new + (1-ema) * prev

        self._start_pos = initial_obs["pos"].copy()
        self._last_gates_pos = None
        self._last_gates_quat = None
        self._planner = SplinePlanner(
            self._start_pos,
            initial_obs,
            t_total=self._spline_t_total,  # changedPractical: was 5.0;
            curvature_weight=self._spline_curvature_weight,  
            #changedPractical: k=0.5 best per-segment match vs PPO (k>0.5 over-penalises the G0→G1)
            obstacles_pos=initial_obs[
                "obstacles_pos"
            ],  # changedPractical: obs1 at (1.0,0.25) is 0.03m from no-detour spline
            clearance=self._spline_clearance,  
            # changedPractical: 0.16-0.21 flips obs3 detour to SW (wrong side, path 8.0m); 
            # 0.22 stays NE (path 7.74m, min_obs 0.21m)
        )

        # changedPractical: build the arc-length LUT for the MPCC reparameterization. The spline
        # is fixed at reset (planner is never replanned), so the LUT can be baked into the JIT'd
        # cost as trace-time constants via self — no need to thread it through the call chain like
        # `obstacles` (which DO change at level 2).
        self._build_theta_lut(self._lut_samples)
        # committed progress along the path (arc length, m). Rollouts start each horizon here.
        self._theta = 0.0

        self._finished = False
        # changedPractical
        # self._t_start = initial_obs["t"]
        self._t_start = self._t
        # self._t_end = initial_info["planner_cycles"] * initial_info["planner_cycle_time"]
        self._t_end = self._planner.t_total

        ### Generate trajectory
        # changedPractical
        """
        Replaced since not pressent (see above)

        self._planner = TrajectoryGenerator3DPeriodicMotion(
            traj_type=initial_info["traj_type"],
            num_cycles=initial_info["planner_cycles"],
            cycle_time=initial_info["planner_cycle_time"],
            scaling_xyz=initial_info["scaling"],
            center_pos=initial_info["pos_hover"],
            axis_order=initial_info["axis_order"],
            yaw_mode=initial_info["yaw_mode"],
            yaw_setting=initial_info["yaw_setting"],
        )
        """

        # changedPractical
        # key = jax.random.PRNGKey(0)
        self._rng_key = jax.random.PRNGKey(0)
        self._rng_key, subkey = random.split(self._rng_key)
        #info_short = {"rng_key": subkey, "obstacles": jnp.array(initial_obs["obstacles_pos"])}

        self._log_buf: dict | None = None  # must be set before warmup calls compute_control
        for i in range(10):
            a = self.compute_control(initial_obs, info)  # Warm up the controller
            jax.block_until_ready(a)
        # changedPractical: warmup advanced self._t by 10*ctrl_dt (~0.2s). Reset BOTH the master
        # clock and _t_start to 0 so (a) the first real step queries the spline at t≈0 and
        # (b) the logged timestamp (self._t) carries no warmup offset — otherwise every logged
        # sample plots ~0.2s late
        # TODO: Check this before real-hardware test;
        self._t = 0.0
        self._t_start = 0.0
        self._theta = 0.0  
        # changedPractical: drop warmup progress so the first real step starts at theta=0

        if os.getenv("LOG_DRONE_DATA"):
            # Buffer is populated lazily via setdefault in _log_step_costs; see that method
            # for the logged keys (per-mode, per-term full-horizon rollout costs).
            self._log_buf = {}

    def _build_theta_lut(self, n_samples: int) -> None:
        """Reparameterize the (time-based) spline to arc-length progress theta; cache a LUT.

        Densely sample the planner's position spline in time, accumulate segment lengths to
        get arc length (= theta), and store position + unit-tangent tables keyed on theta.
        Inside the rollout, ref(theta) is recovered with jnp.interp (see _ref_at_theta). This is
        the JAX/MPPI analog of basic_mpcc.py's cs.interpolant LUTs, but keyed on arc length
        (so v_theta has units m/s) instead of time.
        """
        t_dense = np.linspace(0.0, self._planner.t_total, n_samples)
        pos = np.asarray(self._planner._pos_spline(t_dense))  # (n, 3)
        seg = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])  # arc-length grid = theta, (n,)
        vel = np.asarray(self._planner._vel_spline(t_dense))
        tangent = vel / np.clip(np.linalg.norm(vel, axis=1, keepdims=True), 1e-9, None)
        self._s_total = float(s[-1])
        # device copies baked into the JIT'd cost
        self._theta_grid = jnp.asarray(s, device=self.sim.device)
        self._ref_pos_lut = jnp.asarray(pos, device=self.sim.device)
        self._ref_tan_lut = jnp.asarray(tangent, device=self.sim.device)
        # host copies for logging / rendering
        self._theta_grid_np = s
        self._ref_pos_np = pos

    def _ref_at_theta(self, theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """LUT lookup: reference position and unit path tangent at progress theta (arc length).

        theta is a scalar; vmap over rollouts. Clipped to [0, s_total] so the rollout never
        extrapolates past the path ends (mirrors the query_times clamp in compute_control).
        """
        tg = self._theta_grid
        theta = jnp.clip(theta, tg[0], tg[-1])
        pos = jnp.stack(
            [
                jnp.interp(theta, tg, self._ref_pos_lut[:, 0]),
                jnp.interp(theta, tg, self._ref_pos_lut[:, 1]),
                jnp.interp(theta, tg, self._ref_pos_lut[:, 2]),
            ]
        )
        tan = jnp.stack(
            [
                jnp.interp(theta, tg, self._ref_tan_lut[:, 0]),
                jnp.interp(theta, tg, self._ref_tan_lut[:, 1]),
                jnp.interp(theta, tg, self._ref_tan_lut[:, 2]),
            ]
        )
        tan = tan / jnp.clip(jnp.linalg.norm(tan), 1e-9, None)
        return pos, tan

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict
    ) -> NDArray[np.floating]:
        """Compute the next desired collective thrust and roll/pitch/yaw of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The collective thrust and orientation [r_des, p_des, y_des, t_des] as a numpy array.
        """
        self._t += self.ctrl_dt

        obs["rotor_vel"] = self.thrust
        obs_device = {k: jax.device_put(v, self.sim.device) for k, v in obs.items()}
        # changedPractical
        t = self._t - self._t_start

        # changedPractical: clamp query times so rollout never extrapolates past spline end
        query_times = np.clip(t + self.dt_array, 0.0, self._planner.t_total)
        des_pos, des_vel, des_acc, des_yaw = self._planner.get_coordinates(query_times)
        # changedPractical
        
        if "opponent_traj" in info:
            opp_traj = info["opponent_traj"]
        else:
            opp_traj = np.zeros_like(des_pos)
        

        self.opp_traj = np.array(opp_traj)

        opp_vel = np.diff(opp_traj, axis=0) / self.dt
        opp_vel = np.vstack((opp_vel, opp_vel[-1]))

        refs = {
            "pos": jnp.array(des_pos, device=self.sim.device),
            "vel": jnp.array(des_vel, device=self.sim.device),
            "acc": jnp.array(des_acc, device=self.sim.device),
            "yaw": jnp.array(des_yaw, device=self.sim.device),
            "opp_pos": jnp.array(opp_traj, device=self.sim.device),
            "opp_vel": jnp.array(opp_vel, device=self.sim.device),
        }

        # 1. Update Step
        # Now returns a batch of means and sigmas

        # changedPractical
        self._rng_key, subkey = jax.random.split(self._rng_key)

        # changedPractical: MPCC — every rollout starts the horizon at the committed progress
        # self._theta (arc length, m), then integrates its own sampled v_theta forward.
        theta0 = jnp.full((self.n_samples,), self._theta, device=self.sim.device)

        (
            new_means,
            new_sigmas,
            best_mode_idx,
            costs_grouped,
            positions_grouped,
            mode_term_costs,
        ) = (
            self._mppi_core_update(
                subkey,
                obs_device,
                refs,
                theta0,
                self.mean_controls,  # Shape: (K, Horizon, U)
                self.noise_sigmas,  # Shape: (K, Horizon, U)
            )
        )

        # 2. Extract Action from the "Winner" Mode
        # We only execute the action from the best cluster
        best_action = new_means[best_mode_idx, 0]  # 5-dim: [roll, pitch, yaw, thrust, v_theta]

        # 3. Receding Horizon Update (Shift & Interpolate)
        # We need to shift ALL K means, not just the active one.
        # We vmap your existing helper over the first dimension (K).
        # in_axes=(0, None, None) means:
        #   - arg 0 (mean/sigma): split along axis 0
        #   - arg 1 (dt): broadcast
        #   - arg 2 (ctrl_dt): broadcast
        vmap_shift = jax.vmap(self.shift_and_interpolate, in_axes=(0, None, None))
        self.mean_controls = vmap_shift(new_means, self.dt, self.ctrl_dt)

        # We also shift the sigmas!
        # This ensures that if the drone is "confident" about the path 2 seconds away,
        # that confidence rolls forward correctly.
        self.noise_sigmas = vmap_shift(new_sigmas, self.dt, self.ctrl_dt)

        # Optional: Reset logic (Pseudo-code)
        # If you wanted to prevent mode collapse, this is where you'd check
        # if means are too close and reset one.
        # self.mean_controls = self._check_and_reset_modes(self.mean_controls)

        self.best_mode_idx = best_mode_idx  # Store for visualization outside of JIT
        self.means = new_means
        self.costs = costs_grouped
        self.positions = positions_grouped
        self.mode_term_costs = mode_term_costs  # per-mode best-sample cost breakdown

        traj_best_group = positions_grouped[best_mode_idx]
        best_traj_idx = np.argmin(costs_grouped[best_mode_idx])
        self.best_traj = np.array(traj_best_group[best_traj_idx])


        # changedPractical: split the 5-dim winner into the 4 real attitude/thrust commands and
        # the progress speed v_theta. Only the 4 real channels are EMA-smoothed and returned to
        # the env; v_theta is internal and used to advance the committed progress self._theta.
        full_action = np.asarray(best_action)  # back to CPU, 5-dim
        v_theta_cmd = float(full_action[4])
        action = full_action[:4]
        # changedPractical: EMA smoothing to damp inter-step oscillations from mode switching
        action = self._action_ema * action + (1.0 - self._action_ema) * self._prev_action
        self._prev_action = action
        # commit progress: advance theta by the executed v_theta over one control cycle
        self._theta = float(min(self._theta + v_theta_cmd * self.ctrl_dt, self._s_total))
        self.v_theta_cmd = v_theta_cmd  # store for logging / debugging
        if self._theta >= self._s_total - 1e-3:
            self._finished = True
        self.thrust += (
            self.drone_params["thrust_dyn_coef"] * (action[3] - self.thrust) * self.ctrl_dt
        )

        if self._log_buf is not None:
            self._log_step_costs(obs, action)

        return action

    def _log_step_costs(
        self,
        obs: dict[str, NDArray[np.floating]],
        action: NDArray[np.floating],
    ) -> None:
        """Log the full-horizon rollout cost breakdown of the K parallel modes.

        For each control step we record, per cost term, a length-K vector: the
        horizon-summed cost of each mode's best (lowest-total) sample — i.e. the cost of
        the trajectory that mode proposes. This is what the MPPI optimizer actually scores,
        so it's the right signal for cost engineering / weight tuning. ``setdefault`` lets
        subclass-specific terms (e.g. cost_opp_drone) appear without pre-declaring them.
        """
        terms = {k: np.asarray(v) for k, v in self.mode_term_costs.items()}  # each (K,)
        costs_km = np.asarray(self.costs)  # (K, M) total per sample

        fields = {
            "t": self._t,
            "pos": np.asarray(obs["pos"]).copy(),
            "action": np.asarray(action).copy(),
            "target_gate": int(obs.get("target_gate", -1)),
            "best_mode_idx": int(self.best_mode_idx),
            # per-mode totals: best sample per mode (K,) and the global best
            "mode_cost_total": costs_km.min(axis=1),
            "min_cost": float(costs_km.min()),
        }
        # per-mode, per-term horizon cost: each logged as a (K,) vector
        for name, vec in terms.items():
            fields[f"cost_{name}"] = vec
        # total per mode summed from the logged terms (matches mode_cost_total)
        fields["cost_total"] = np.sum([v for v in terms.values()], axis=0)

        for key, val in fields.items():
            self._log_buf.setdefault(key, []).append(val)

    def step_callback(
        self,
        action: NDArray[np.floating] | None = None,
        obs: dict[str, NDArray[np.floating]] | None = None,
        reward: float | None = None,
        terminated: bool | None = None,
        truncated: bool | None = None,
        info: dict | None = None,
    ) -> bool:
        """Increment the tick counter."""
        self.obstacles = jnp.array(obs["obstacles_pos"], device=self.sim.device)

        gates_changed = (
            self._last_gates_pos is None
            or not jnp.allclose(obs["gates_pos"], self._last_gates_pos)
            or not jnp.allclose(obs["gates_quat"], self._last_gates_quat)
        )

        if gates_changed:
            gate_frame_pos = self.get_gate_frame_pos(obs["gates_pos"], obs["gates_quat"])
            self.gate_frame_obstacles = jnp.array(gate_frame_pos, device=self.sim.device)

            update_planner = False

            if update_planner:
                self._planner = SplinePlanner(
                    self._start_pos,
                    obs,
                    t_total=self._spline_t_total,  
                    curvature_weight=2.0,  
                    #changedPractical: k=0.5 best per-segment match vs PPO (k>0.5 over-penalises the
                    # G0→G1 curve)
                    obstacles_pos=obs[
                        "obstacles_pos"
                    ],  # changedPractical: obs1 at (1.0,0.25) is 0.03m from no-detour spline
                    clearance=0.22,  # changedPractical: 0.16-0.21 flips obs3 detour to SW
                    #(wrong side, path 8.0m); 0.22 stays NE (path 7.74m, min_obs 0.21m)
                )
            self._last_gates_pos = obs["gates_pos"].copy()
            self._last_gates_quat = obs["gates_quat"].copy()

        # changedPractical: finish when all gates are passed, not only when spline time expires
        if obs.get("target_gate", 0) == -1:
            self._finished = True
        return self._finished

    def episode_callback(self):
        """Save logged data to disk if LOG_DRONE_DATA is set."""
        if self._log_buf is None or not self._log_buf.get("t"):
            return
        log_dir = os.getenv("LOG_DRONE_DATA", ".")
        out_path = (
            Path(log_dir) / "mppi_data.npz" if not log_dir.endswith(".npz") else Path(log_dir)
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_path,
            **{k: np.array(v) for k, v in self._log_buf.items()},
            spline=self._planner.get_trajectory(300),
            waypoints=self._planner.waypoints,
            gates_pos=self.initial_obs["gates_pos"],
            gates_quat=self.initial_obs["gates_quat"],
            obstacles_pos=self.initial_obs["obstacles_pos"],
        )
        print(f"[MPPI] Saved {len(self._log_buf['t'])} steps → {out_path}")

    @partial(jax.jit, static_argnames=["self"])
    def _mppi_core_update(
        self,
        key: jax.Array,
        obs: dict[str, jnp.ndarray],
        refs: dict[str, jnp.ndarray],
        theta0: jnp.ndarray,
        current_means: jnp.ndarray,
        noise_sigmas: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Internal MPPI update: sample, roll out, evaluate costs, update means.

        current_means: (K, Horizon, nu) - K different control strategies
        noise_sigmas: (K, Horizon, nu) - Adaptive variance for each strategy
        theta0: (n_samples,) - starting progress (arc length) for each rollout
        """
        # --- 1. PREPARE BOUNDS ---
        # Shape: (K, Horizon, 4) -> Broadcast for sampling later
        # We need to look at bounds relative to each specific mean
        lower_bound = self.act_low - current_means
        upper_bound = self.act_high - current_means

        # --- 2. SAMPLING (Stratified by Mean) ---
        # We generate M samples for EACH of the K means.
        # Total samples = K * M = self.n_samples

        # We need separate keys for each cluster to ensure independence
        keys = jax.random.split(key, self.K)

        def sample_per_mean(
            k_key: jax.Array,
            k_mean: jnp.ndarray,
            k_sigma: jnp.ndarray,
            k_lb: jnp.ndarray,
            k_ub: jnp.ndarray,
        ) -> jnp.ndarray:
            # Generate noise for ONE mean
            # Shape: (M, Horizon, 4)
            k_noise = self.get_truncated_normal_jax(
                k_key,
                mean=0,
                sd=k_sigma,
                x_min=k_lb[None, ...],
                x_max=k_ub[None, ...],
                shape=(self.M, self.N, self.num_inputs),
            )
            return k_noise

        # Vectorize sampling over K means
        # noise shape: (K, M, Horizon, 4)
        noise = vmap(sample_per_mean)(keys, current_means, noise_sigmas, lower_bound, upper_bound)

        # --- 3. NOISE SMOOTHING ---
        # Apply beta smoothing. We can flatten K and M dimensions for this operation
        # or vmap twice. Flattening is easier.
        noise_flat = noise.reshape(-1, self.N, self.num_inputs)  # (N, H, num_inputs)

        def smooth_scan(carry: jnp.ndarray, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            new_val = self.beta * carry + (1 - self.beta) * x
            return new_val, new_val

        # Initialize smooth scan with zeros
        _, noise_flat = jax.lax.scan(
            lambda c, x: vmap(smooth_scan)(c, x),
            jnp.zeros((noise_flat.shape[0], self.num_inputs)),
            # jnp.ones((noise_flat.shape[0], self.num_inputs)) * noise_flat[:, 0, :],
            noise_flat.transpose(1, 0, 2),
        )
        noise_flat = noise_flat.transpose(1, 0, 2)

        # Reshape back to grouped format: (K, M, H, nu)
        noise = noise_flat.reshape(self.K, self.M, self.N, self.num_inputs)

        # --- 4. ROLLOUTS ---
        # Calculate candidate controls: Mean[k] + Noise[k, m]
        # Broadcasting: (K, 1, H, 4) + (K, M, H, 4)
        candidate_controls = current_means[:, None, ...] + noise

        # Flatten for the physics engine (Physics doesn't care about K clusters)
        # 1. Transpose from (K, M, N, nu) -> (N, K, M, nu)
        # 2. Reshape to (N, K*M, nu)
        controls_flat = candidate_controls.transpose(2, 0, 1, 3).reshape(self.N,-1, self.num_inputs)

        costs_flat, positions_flat, terms_flat = self.rollout_sim(
            obs, theta0, (controls_flat, refs, self.obstacles, self.gate_frame_obstacles)
        )

        # Reshape costs back to groups: (K, M)
        costs_grouped = costs_flat.reshape(self.K, self.M)
        positions_grouped = positions_flat.transpose(1, 0, 2, 3).reshape(self.K, self.M, self.N, 3)
        # Per-mode cost breakdown: take each mode's best (lowest-total) sample, the
        # trajectory that mode "proposes". mode_term_costs[name] has shape (K,).
        best_sample = jnp.argmin(costs_grouped, axis=1)  # (K,)
        k_idx = jnp.arange(self.K)
        mode_term_costs = {
            name: v.reshape(self.K, self.M)[k_idx, best_sample] for name, v in terms_flat.items()
        }

        # --- 5. PER-MODE UPDATE (The Core Logic) ---
        # We define a function that updates ONE mean, then vmap it over K

        def update_single_mode(
            k_mean: jnp.ndarray, k_noise: jnp.ndarray, k_costs: jnp.ndarray, k_sigma: jnp.ndarray
        ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            # k_mean: (H, 4)
            # k_noise: (M, H, 4)
            # k_costs: (M,)

            # 1. Sort Indices (Local Elites)
            sorted_idx = jnp.argsort(k_costs)
            n_elites = int(self.M * self.elite_percentage)

            # Safety check: ensure at least 1 elite
            n_elites = max(n_elites, 1)
            elite_idx = sorted_idx[:n_elites]

            # 2. Gather Elite Data
            elite_noise = k_noise[elite_idx]
            elite_costs = k_costs[elite_idx]

            # 3. Weights
            min_cost = jnp.min(elite_costs)
            weights = jnp.exp(-(elite_costs - min_cost) / self.temperature)

            # Normalize weights (add epsilon for numerical stability)
            weights = weights / (jnp.sum(weights) + 1e-10)

            # 4. Weighted Update (Delta)
            delta = jnp.sum(weights[:, None, None] * elite_noise, axis=0)

            # 5. Covariance Update
            diff = elite_noise - delta
            weighted_var = jnp.sum(weights[:, None, None] * (diff**2), axis=0)

            # Smoothing the sigma
            alpha = self.alpha
            new_sigma_sq = alpha * weighted_var + (1.0 - alpha) * (k_sigma**2)

            # Clipping
            min_variance = self.min_variance
            new_sigma_sq = jnp.maximum(new_sigma_sq, min_variance)
            new_k_sigma = jnp.sqrt(new_sigma_sq)

            # Return updated mean, updated sigma, and the best cost of this mode
            return k_mean + delta, new_k_sigma, min_cost

        # Vectorize the update logic over the K dimension!
        # This updates all means in parallel.
        updated_means, updated_sigmas, cluster_best_costs = vmap(update_single_mode)(
            current_means, noise, costs_grouped, noise_sigmas
        )

        # --- 6. WINNER SELECTION ---
        # We need to pick one action to actually execute.
        # We pick the mean that had the lowest cost elite.
        best_mode_idx = jnp.argmin(cluster_best_costs)

        return (
            updated_means,
            updated_sigmas,
            best_mode_idx,
            costs_grouped,
            positions_grouped,
            mode_term_costs,
        )

    def get_truncated_normal_jax(
        self,
        key: jax.Array,
        mean: jnp.ndarray,
        sd: jnp.ndarray,
        x_min: jnp.ndarray,
        x_max: jnp.ndarray,
        shape: tuple = (1,),
    ) -> jnp.ndarray:
        """Samples from a truncated normal distribution using JAX.

        Args:
            key: JAX PRNGKey
            mean: Desired mean (mu) of the underlying gaussian
            sd: Desired standard deviation (sigma) of the underlying gaussian
            x_min: Lower bound
            x_max: Upper bound
            shape: Output shape tuple

        Returns:
            jnp.ndarray: Sampled values within [x_min, x_max]
        """
        # 1. Convert physical bounds to standard normal 'Z-scores'
        alpha = (x_min - mean) / sd
        beta = (x_max - mean) / sd

        # 2. Sample from standard truncated normal (mean=0, std=1)
        # jax.random.truncated_normal expects bounds for the standard normal
        standard_samples = random.truncated_normal(key, lower=alpha, upper=beta, shape=shape)

        # 3. Scale and shift back to original distribution
        return (standard_samples * sd) + mean

    @partial(jax.jit, static_argnames=["self"])
    def shift_and_interpolate(
        self, controls: jnp.ndarray, dt_plan: float, dt_ctrl: float
    ) -> jnp.ndarray:
        """Shift the control trajectory forward by dt_ctrl using linear interpolation.

        Args:
            controls: Array of shape (Horizon, Control_Dim).
            dt_plan: Time duration of one step in the planning horizon.
            dt_ctrl: Time duration of one control cycle (latency/execution time).

        Returns:
            Shifted controls of shape (Horizon, Control_Dim).
        """
        horizon_len = controls.shape[0]

        # Create the time grids
        # old_times: [0, dt_plan, 2*dt_plan, ...]
        old_times = jnp.arange(horizon_len) * dt_plan

        # target_times: [dt_ctrl, dt_ctrl + dt_plan, ...]
        target_times = old_times + dt_ctrl

        # Define the 1D interpolation logic
        def interp_fn(y: jnp.ndarray) -> jnp.ndarray:
            # right=y[-1] implements Zero-Order Hold for the end of the horizon
            return jnp.interp(target_times, old_times, y, left=y[0], right=y[-1])

        # Vectorize across the Control_Dim (axis 1)
        # in_axes=1 means we iterate over columns (controls), keeping time (rows) intact per op
        return jax.vmap(interp_fn, in_axes=1, out_axes=1)(controls)

    @partial(jax.jit, static_argnames=["self"])
    def compute_cost(
        self,
        data: SimData,
        theta: jnp.ndarray,
        v_theta: jnp.ndarray,
        reference: dict[str, jnp.ndarray],
        obstacles: jnp.ndarray,
        gate_frame_obstacles: jnp.ndarray,
    ) -> dict[str, jnp.ndarray]:
        """Compute the per-term cost for a given state (MPCC contour/lag/progress formulation).

        Returns a dict mapping term name -> cost of shape (n_rollouts,). The optimizer
        sums these; keeping them separate lets us accumulate each term over the horizon
        and log the per-mode cost breakdown for cost engineering.

        theta: (n_rollouts,) current progress (arc length) of each rollout.
        v_theta: (n_rollouts,) progress speed of each rollout (for the progress reward).
        """
        pos = data.states.pos[:, 0, :]  # Shape: (n_rollouts, 3)
        ang_vel = data.states.ang_vel[:, 0, :]  # Shape: (n_rollouts, 3)
        ang_acc = data.states_deriv.ang_vel[:, 0, :]  # Shape: (n_rollouts, 3)
        cmd = data.controls.attitude.staged_cmd[:, 0, :]  # Shape: (n_rollouts, 4)

        ## 1. MPCC tracking cost — contour + lag + progress (replaces pos/vel tracking)
        # Reference position and unit path tangent at each rollout's own theta (LUT lookup).
        ref_pos, tangent = vmap(self._ref_at_theta)(theta)  # (n_rollouts, 3) each
        err = pos - ref_pos  # (n_rollouts, 3)
        # Lag error: signed component of err ALONG the path tangent (ahead +, behind -).
        e_lag = jnp.sum(tangent * err, axis=-1)  # (n_rollouts,)
        # Contour error: component PERPENDICULAR to the path (off-track distance).
        e_con_vec = err - e_lag[:, None] * tangent  # (n_rollouts, 3)
        lag_cost = e_lag**2 * self.w_lag
        contour_cost = jnp.sum(e_con_vec**2, axis=-1) * self.w_contour
        # Extra altitude penalty toward the reference height (kept from the old cost).
        z_cost = jnp.abs(pos[..., 2] - ref_pos[..., 2]) * self.w_z
        # Progress reward: advancing theta lowers cost (this is what removes the t_total ceiling).
        progress_cost = -self.w_progress * v_theta
        ang_vel_error = jnp.linalg.norm(ang_vel, axis=-1)
        ang_vel_cost = ang_vel_error**2 * self.w_ang_vel
        ang_acc_error = jnp.linalg.norm(ang_acc, axis=-1)
        ang_acc_cost = ang_acc_error**2 * self.w_ang_acc

        ## 2. Control Cost (Efficiency + Stability)
        # Penalize high tilt (roll/pitch)
        tilt_cost = (
            jnp.linalg.norm(cmd[:, :2], axis=-1) ** 2 * self.w_tilt
        )  # changedPractical: was 5.0; loosened to allow aggressive roll/pitch
        # Penalize thrust deviations from gravity
        thrust_cost = (
            cmd[:, 3] - HOVER_THRUST
        ) ** 2 * self.w_thrust  # changedPractical: was 0.0; regularise toward hover thrust
        # Penalize yaw deviations
        yaw_cost = (
            cmd[:, 2] - 0 #des_yaw should be 0
        ) ** 2 * self.w_yaw  # changedPractical: was 0.0; added to stabilise yaw oscillation

        binary_obstacle_cost = True
        ## 3. Obstacle Cost (Safety)
        save_dist_obst = (self.initial_info["experiment"]["env"]["obstacle_radius"]
            + self.initial_info["experiment"]["env"]["drone_radius"])
        obs_diff = jnp.linalg.norm(pos[..., None, :2] - obstacles[None, :, :2], axis=-1)

        if binary_obstacle_cost:
            obstacle_hits = jnp.where(obs_diff < save_dist_obst,1,0,)
            obstacle_cost = self.w_obstacle * jnp.sum(obstacle_hits, axis=-1)
        else:
            obstacle_cost = self.w_obstacle_exp * jnp.sum(
                jnp.exp(-((obs_diff / save_dist_obst) ** 2)), axis=1) 

        ## 4. Gate Obstacle Cost (Safety)
        save_dist_gate_obst = (self.initial_info["experiment"]["env"]["gate_frame_radius"]
            + self.initial_info["experiment"]["env"]["drone_radius"])
        gate_obs_diff = jnp.linalg.norm(
            pos[..., None, :2] - gate_frame_obstacles[None, :, :2], axis=-1
        )

        if binary_obstacle_cost:
            gate_obstacle_hits = jnp.where(gate_obs_diff< save_dist_gate_obst,1,0,)
            gate_obstacle_cost = self.w_obstacle * jnp.sum(gate_obstacle_hits, axis=-1)
        else:
            gate_obstacle_cost = self.w_obstacle_exp * jnp.sum(
                jnp.exp(-((gate_obs_diff / save_dist_gate_obst) ** 2)), axis=1) 

        ## 5. Floor penalty — prevents rollouts from sinking into the ground
        # changedPractical: penalise rollout states below z=0.1m, deters early liftoff instability
        floor_cost = jnp.where(
            pos[..., 2] < self.floor_z, (self.floor_z - pos[..., 2]) ** 2 * self.w_floor, 0.0
        )

        # Per-term costs, each shape (n_rollouts,). The optimizer/logger sum these.
        # changedPractical: MPCC terms (lag/contour/progress) replace the old pos/vel tracking.
        return {
            "lag": lag_cost,
            "contour": contour_cost,
            "z": z_cost,
            "progress": progress_cost,
            "ang_vel": ang_vel_cost,
            "ang_acc": ang_acc_cost,
            "tilt": tilt_cost,
            "thrust": thrust_cost,
            "yaw": yaw_cost,
            "obstacle": obstacle_cost,
            "gate_obstacle": gate_obstacle_cost,
            "floor": floor_cost,
        }

    @partial(jax.jit, static_argnames=["self"])
    def apply_input(
        self,
        carry: tuple[SimData, jnp.ndarray],
        info: tuple[jnp.ndarray, dict[str, jnp.ndarray], jnp.ndarray, jnp.ndarray],
    ) -> tuple[tuple[SimData, jnp.ndarray], jnp.ndarray]:
        """Roll out the sim for one step and advance the MPCC progress theta.

        Args:
            carry: Tuple of (sim data, theta) where theta is (n_rollouts,) progress (arc length).
            info: Tuple of (cmd5, ref, obstacles, gate_frame_obstacles). cmd5 is (n_rollouts, 5):
                the first 4 channels are the attitude/thrust command sent to the sim; the 5th is
                v_theta, the progress speed used to integrate theta.
        """
        data, theta = carry
        cmd, ref, obstacles, gate_frame_obstacles = info
        v_theta = cmd[:, 4]  # (n_rollouts,)
        cmd4 = cmd[:, :4]  # only the real attitude/thrust channels go to the sim
        # Integrate progress forward by one planning step (clamped to the path ends).
        theta_next = jnp.clip(theta + v_theta * self.dt, 0.0, self._s_total)
        # Step sim with the 4 real channels
        data = data.replace(
            controls=data.controls.replace(
                attitude=data.controls.attitude.replace(staged_cmd=cmd4[:, None, :])
            )
        )
        next_data = self.step_fn(data, self.sim.freq // self.sim.control_freq)
        cost_terms = self.compute_cost(
            next_data, theta_next, v_theta, ref, obstacles, gate_frame_obstacles
        )
        return (next_data, theta_next), (cost_terms, next_data.states.pos)

    @partial(jax.jit, static_argnames=["self"])
    def rollout_sim(
        self,
        obs: dict,
        theta0: jnp.ndarray,
        infos: tuple[jnp.ndarray, dict[str, jnp.ndarray], jnp.ndarray, jnp.ndarray],
    ) -> jnp.ndarray:
        """Rolls out the sim for scan.

        theta0: (n_rollouts,) starting progress (arc length) shared by all rollouts.
        """
        data = self.sim.data
        pos = data.states.pos.at[...].set(obs["pos"])
        quat = data.states.quat.at[...].set(obs["quat"])
        vel = data.states.vel.at[...].set(obs["vel"])
        ang_vel = data.states.ang_vel.at[...].set(obs["ang_vel"])
        rotor_vel = data.states.rotor_vel.at[...].set(obs["rotor_vel"])
        data = data.replace(
            states=data.states.replace(
                pos=pos, quat=quat, vel=vel, ang_vel=ang_vel, rotor_vel=rotor_vel
            )
        )
        controls_flat, refs, obstacles, gate_frame_obstacles = infos
        # scan iterates over axis 0 (N time steps); tile obstacles so each step gets a slice
        obstacles_tiled = jnp.broadcast_to(obstacles[None], (self.N,) + obstacles.shape)
        gate_frame_obstacles_tiled = jnp.broadcast_to(
            gate_frame_obstacles[None], (self.N,) + gate_frame_obstacles.shape
        )

        # changedPractical: carry (sim data, theta) so each rollout integrates its own progress
        _, (cost_terms, positions) = scan(
            self.apply_input,
            (data, theta0),
            (controls_flat, refs, obstacles_tiled, gate_frame_obstacles_tiled),
        )
        # cost_terms: dict of (N, n_rollouts). Sum each term over the horizon -> (n_rollouts,).
        terms_summed = {k: jnp.sum(v, axis=0) for k, v in cost_terms.items()}
        total = sum(terms_summed.values())  # (n_rollouts,)
        return total, positions, terms_summed

    # changedPractical

    def _draw_reference(self, sim: Sim):
        """Draw the reference spline, current setpoint, and waypoints."""
        # changedPractical: setpoint is now the reference at the committed progress theta
        # (arc length), not at wall-clock time.
        idx = min(
            int(np.searchsorted(self._theta_grid_np, self._theta)),
            len(self._ref_pos_np) - 1,
        )
        setpoint = self._ref_pos_np[idx].reshape(1, -1)
        draw_points(sim, setpoint, rgba=(1.0, 0.0, 0.0, 1.0), size=0.02)
        trajectory = self._planner.get_trajectory(100)
        draw_line(sim, trajectory, rgba=(0.0, 1.0, 0.0, 1.0))
        draw_points(sim, self._planner.waypoints, rgba=(0.0, 0.0, 1.0, 1.0), size=0.03)

    def _draw_mppi_rollouts(self, sim: Sim):
        """Draw the top MPPI rollout trajectories per cluster, best in white."""
        if not hasattr(self, "positions") or self.positions is None:
            return

        _cluster_colors = [
            (1.0, 0.5, 0.0),  # orange
            (0.5, 0.0, 1.0),  # purple
            (0.0, 1.0, 1.0),  # cyan
            (1.0, 1.0, 0.0),  # yellow
            (1.0, 0.0, 0.5),  # pink
            (0.0, 0.5, 1.0),  # sky blue
        ]

        positions = np.asarray(self.positions)  # (K, M, N, 3)
        costs = np.asarray(self.costs)  # (K, M)
        best_k = int(self.best_mode_idx)
        n_viz = 3

        for k in range(self.K):
            rgb = _cluster_colors[k % len(_cluster_colors)]
            sorted_idx = np.argsort(costs[k])[:n_viz]
            for rank, m in enumerate(sorted_idx):
                alpha = 1.0 - (rank / n_viz) * 0.75
                draw_line(sim, positions[k, m], rgba=(*rgb, alpha))

        best_m = int(np.argmin(costs[best_k]))
        draw_line(sim, positions[best_k, best_m], rgba=(1.0, 1.0, 1.0, 1.0))

    def _draw_obstacles(self, sim: Sim):
        draw_points(sim, self.obstacles, rgba=(1.0, 0.0, 0.0, 1.0), size=0.02)
        draw_points(sim, self.gate_frame_obstacles, rgba=(1.0, 0.0, 0.0, 1.0), size=0.02)

    # own renderings
    def render_callback(self, sim: Sim):
        """Visualize the desired trajectory and the current setpoint."""
        self._draw_reference(sim)
        self._draw_mppi_rollouts(sim)
        self._draw_obstacles(sim)
