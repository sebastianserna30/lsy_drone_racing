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


def single_drone_terms(
    pos: jnp.ndarray,
    ang_vel: jnp.ndarray,
    ang_acc: jnp.ndarray,
    cmd: jnp.ndarray,
    ref_pos: jnp.ndarray,
    tangent: jnp.ndarray,
    v_theta: jnp.ndarray,
    obstacles: jnp.ndarray,
    gate_frame_obstacles: jnp.ndarray,
    save_dist_obst: float,
    save_dist_gate_obst: float,
    w: dict[str, float],
) -> dict[str, jnp.ndarray]:
    """Pure per-drone MPCC cost terms (no `self`, no opponent collision).

    changedPractical: extracted verbatim from the old AttitudeMPPIController.compute_cost body so
    the single-agent path and the n_agents-generic path share one implementation. Each input is
    (n_rollouts, ...) for ONE drone; the caller supplies ref_pos/tangent (from the LUT) and the
    static weight dict `w`. Returns a dict term_name -> (n_rollouts,). Byte-identical to the old
    inline math (binary obstacle cost, no anisotropic collision — that lives in the caller).
    """
    ## 1. MPCC tracking cost — contour + lag + progress
    err = pos - ref_pos  # (n_rollouts, 3)
    # Lag error: signed component of err ALONG the path tangent (ahead +, behind -).
    e_lag = jnp.sum(tangent * err, axis=-1)  # (n_rollouts,)
    # Contour error: component PERPENDICULAR to the path (off-track distance).
    e_con_vec = err - e_lag[:, None] * tangent  # (n_rollouts, 3)
    lag_cost = e_lag**2 * w["lag"]
    contour_cost = jnp.sum(e_con_vec**2, axis=-1) * w["contour"]
    # Extra altitude penalty toward the reference height (kept from the old cost).
    z_cost = jnp.abs(pos[..., 2] - ref_pos[..., 2]) * w["z"]
    # Progress reward: advancing theta lowers cost (this is what removes the t_total ceiling).
    progress_cost = -w["progress"] * v_theta
    ang_vel_error = jnp.linalg.norm(ang_vel, axis=-1)
    ang_vel_cost = ang_vel_error**2 * w["ang_vel"]
    ang_acc_error = jnp.linalg.norm(ang_acc, axis=-1)
    ang_acc_cost = ang_acc_error**2 * w["ang_acc"]

    ## 2. Control Cost (Efficiency + Stability)
    tilt_cost = jnp.linalg.norm(cmd[:, :2], axis=-1) ** 2 * w["tilt"]
    thrust_cost = (cmd[:, 3] - HOVER_THRUST) ** 2 * w["thrust"]
    yaw_cost = (cmd[:, 2] - 0) ** 2 * w["yaw"]  # des_yaw should be 0

    ## 3. Obstacle Cost (Safety) — binary hit count
    obs_diff = jnp.linalg.norm(pos[..., None, :2] - obstacles[None, :, :2], axis=-1)
    obstacle_hits = jnp.where(obs_diff < save_dist_obst, 1, 0)
    obstacle_cost = w["obstacle"] * jnp.sum(obstacle_hits, axis=-1)

    ## 4. Gate Obstacle Cost (Safety) — binary hit count
    gate_obs_diff = jnp.linalg.norm(pos[..., None, :2] - gate_frame_obstacles[None, :, :2], axis=-1)
    gate_obstacle_hits = jnp.where(gate_obs_diff < save_dist_gate_obst, 1, 0)
    gate_obstacle_cost = w["obstacle"] * jnp.sum(gate_obstacle_hits, axis=-1)

    ## 5. Floor penalty — prevents rollouts from sinking into the ground
    floor_cost = jnp.where(
        pos[..., 2] < w["floor_z"], (w["floor_z"] - pos[..., 2]) ** 2 * w["floor"], 0.0
    )

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
        # changedPractical: n_agents-generic. agent 0 = ego (the drone we execute); agents 1.. are
        # opponents modeled in the SAME shared rollout batch (one lax.scan, n_drones=n_agents).
        # Single-agent configs give n_agents=1 and reduce to the original controller behavior.
        _pos0 = np.asarray(initial_obs["pos"])
        self.n_agents = _pos0.shape[0] if _pos0.ndim == 2 else 1
        # Per-agent mode counts (ragged K allowed). ego K from mppi.K; per-agent overrides from
        # the optional mppi.agents list. Each K must divide n_samples (worlds are shared, paired 1:1).
        _agents_cfg = mppi_cfg.get("agents", [])

        def _agent_cfg(a: int, key: str, default):
            if a < len(_agents_cfg) and key in _agents_cfg[a]:
                return _agents_cfg[a][key]
            return default

        self.K = [int(_agent_cfg(a, "K", mppi_cfg["K"])) for a in range(self.n_agents)]
        self.M = [self.n_samples // k for k in self.K]  # samples per mode, per agent
        for a, k in enumerate(self.K):
            assert self.n_samples % k == 0, (
                f"n_samples ({self.n_samples}) must be divisible by K[{a}]={k}"
            )
        self._agent_cfg = _agent_cfg  # stashed for per-agent spline params below
        # changedPractical: sampled control dim is now 5 = [roll, pitch, yaw, thrust, v_theta].
        # The sim only ever receives the first 4; v_theta is the MPCC progress speed integrated
        # in the rollout. Use self.num_inputs wherever the control dim was previously hardcoded as 4.
        self.num_inputs = 5

        # changedPractical
        self._t = 0.0

        if jax.default_backend() == "cpu":
            available_device = "cpu"
        else:
            available_device = "cuda"

        self.sim = Sim(
            n_worlds=self.n_samples,
            n_drones=self.n_agents,  # changedPractical: both ego + opponents share one rollout batch
            attitude_freq=self.f,
            freq=self.f,
            physics=Physics.so_rpy_rotor_drag,
            control=Control.attitude,
            drone_model="cf21B_500",
            device=available_device,  # TODO get from info
        )
        self.sim.reset()

        self.step_fn = self.sim.build_step_fn()

        self._noise_sigma0 = float(mppi_cfg["noise_sigma"])
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
        # changedPractical: anisotropic opponent keep-out (folded in from mppi++_overtake_improvements).
        # Ellipse elongated ALONG the OTHER drone's real velocity heading: axial (fore/aft) > lateral,
        # so sitting behind is costly, drawing alongside is cheap -> the low-cost escape is a
        # sideways overtake, not a brake. Falls back to isotropic when the other drone is ~stationary.
        self.use_anisotropic_opp = bool(cost_cfg.get("use_anisotropic_opp", False))
        self.opp_axial = float(cost_cfg.get("opp_axial", 0.25))  # fore/aft keep-out half-length (m)
        self.opp_lateral = float(cost_cfg.get("opp_lateral", 0.10))  # sideways keep-out half-width (m)
        # behind-aware contour relaxation: when another drone is ahead-on-line and close, scale down
        # our off-path (contour) cost so we can leave the line to pass. Opt-in (off on the branch).
        self.use_behind_contour = bool(cost_cfg.get("use_behind_contour", False))
        self.behind_radius = float(cost_cfg.get("behind_radius", 0.6))
        self.contour_behind_scale = float(cost_cfg.get("contour_behind_scale", 0.15))
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
        # changedPractical: static weight dict passed to the pure single_drone_terms(). Values are
        # baked as trace-time constants when the jitted cost is compiled (same as before).
        self._weights = {
            "lag": self.w_lag,
            "contour": self.w_contour,
            "z": self.w_z,
            "progress": self.w_progress,
            "ang_vel": self.w_ang_vel,
            "ang_acc": self.w_ang_acc,
            "tilt": self.w_tilt,
            "thrust": self.w_thrust,
            "yaw": self.w_yaw,
            "obstacle": self.w_obstacle,
            "floor": self.w_floor,
            "floor_z": self.floor_z,
        }
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
        # changedPractical: per-agent (ragged K) means & sigmas, kept as length-n_agents lists.
        # noise_sigmas[a]/mean_controls[a] have shape (K_a, N, 5). Thrust channel starts at hover,
        # v_theta channel at the nominal forward progress speed; the v_theta noise channel gets its
        # own (larger) scale (different units from rad/thrust).
        self.noise_sigmas = []
        self.mean_controls = []
        for k in self.K:
            sig = jnp.full((k, self.N, self.num_inputs), self._noise_sigma0, device=self.sim.device)
            sig = sig.at[:, :, 4].set(self.v_theta_sigma)
            self.noise_sigmas.append(sig)
            m = jnp.zeros((k, self.N, self.num_inputs), device=self.sim.device)
            m = m.at[:, :, 3].set(HOVER_THRUST)
            m = m.at[:, :, 4].set(self.v_theta_init)
            self.mean_controls.append(m)

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

        # changedPractical: per-agent start positions (A,3). Agent 0 = ego; agents 1.. = opponents.
        # Preserve the observation dtype (no float64 cast) so the ego spline is bit-identical to the
        # pre-refactor single-agent controller (the chaotic MPPI amplifies any tiny LUT perturbation).
        _start = np.asarray(initial_obs["pos"])
        if _start.ndim == 1:
            _start = _start[None, :]
        self._start_pos = _start.copy()  # (A, 3)
        self._last_gates_pos = None
        self._last_gates_quat = None
        # changedPractical: one SplinePlanner per agent (same gates, per-agent start + optional
        # per-agent t_total/clearance so an opponent flies a different racing line than the ego).
        self._planner = [
            SplinePlanner(
                self._start_pos[a],
                initial_obs,
                t_total=float(self._agent_cfg(a, "t_total", self._spline_t_total)),
                curvature_weight=float(
                    self._agent_cfg(a, "curvature_weight", self._spline_curvature_weight)
                ),
                obstacles_pos=initial_obs["obstacles_pos"],
                clearance=float(self._agent_cfg(a, "clearance", self._spline_clearance)),
            )
            for a in range(self.n_agents)
        ]

        # changedPractical: build per-agent arc-length LUTs for the MPCC reparameterization, stacked
        # on a leading agent axis for vmap in the cost. Splines are fixed at reset.
        self._build_theta_luts(self._lut_samples)
        # committed progress (arc length, m) per agent. Rollouts start each horizon here.
        self._theta = [0.0] * self.n_agents

        self._finished = False
        # changedPractical
        # self._t_start = initial_obs["t"]
        self._t_start = self._t
        # self._t_end = initial_info["planner_cycles"] * initial_info["planner_cycle_time"]
        self._t_end = self._planner[0].t_total  # ego

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
        self._theta = [0.0] * self.n_agents  # changedPractical: drop warmup progress; start at theta=0

        if os.getenv("LOG_DRONE_DATA"):
            # Buffer is populated lazily via setdefault in _log_step_costs; see that method
            # for the logged keys (per-mode, per-term full-horizon rollout costs).
            self._log_buf = {}

    def _build_theta_luts(self, n_lut: int) -> None:
        """Reparameterize each agent's (time-based) spline to arc-length progress theta; cache LUTs.

        For every agent, densely sample the planner's position spline in time, accumulate segment
        lengths to get arc length (= theta), and store position + unit-tangent tables keyed on
        theta. Tables are stacked on a leading agent axis (A, n_lut, ...) so the cost can vmap the
        LUT lookup over agents. Inside the rollout, ref(theta) is recovered with jnp.interp
        (see _ref_at_theta). Keyed on arc length (so v_theta has units m/s) instead of time.
        """
        grids, pos_luts, tan_luts, s_totals = [], [], [], []
        for planner in self._planner:
            t_dense = np.linspace(0.0, planner.t_total, n_lut)
            pos = np.asarray(planner._pos_spline(t_dense))  # (n_lut, 3)
            seg = np.linalg.norm(np.diff(pos, axis=0), axis=1)
            s = np.concatenate([[0.0], np.cumsum(seg)])  # arc-length grid = theta, (n_lut,)
            vel = np.asarray(planner._vel_spline(t_dense))
            tangent = vel / np.clip(np.linalg.norm(vel, axis=1, keepdims=True), 1e-9, None)
            grids.append(s)
            pos_luts.append(pos)
            tan_luts.append(tangent)
            s_totals.append(float(s[-1]))
        dev = self.sim.device
        # stacked device copies (A, n_lut, ...) baked into the JIT'd cost
        self._theta_grid = jnp.asarray(np.stack(grids), device=dev)  # (A, n_lut)
        self._ref_pos_lut = jnp.asarray(np.stack(pos_luts), device=dev)  # (A, n_lut, 3)
        self._ref_tan_lut = jnp.asarray(np.stack(tan_luts), device=dev)  # (A, n_lut, 3)
        self._s_total = jnp.asarray(np.array(s_totals), device=dev)  # (A,)
        # host copies (per agent) for logging / rendering / theta re-anchoring
        self._theta_grid_np = [np.asarray(g) for g in grids]
        self._ref_pos_np = [np.asarray(p) for p in pos_luts]
        self._s_total_np = np.array(s_totals)

    def _ref_at_theta(
        self,
        theta: jnp.ndarray,
        tg: jnp.ndarray,
        pos_lut: jnp.ndarray,
        tan_lut: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """LUT lookup: reference position and unit path tangent at progress theta (arc length).

        theta is a scalar; vmap over rollouts. LUT arrays (tg, pos_lut, tan_lut) are passed
        explicitly so the same helper serves any agent's spline (no self.-bound single LUT).
        Clipped to [0, s_total] so the rollout never extrapolates past the path ends.
        """
        theta = jnp.clip(theta, tg[0], tg[-1])
        pos = jnp.stack(
            [
                jnp.interp(theta, tg, pos_lut[:, 0]),
                jnp.interp(theta, tg, pos_lut[:, 1]),
                jnp.interp(theta, tg, pos_lut[:, 2]),
            ]
        )
        tan = jnp.stack(
            [
                jnp.interp(theta, tg, tan_lut[:, 0]),
                jnp.interp(theta, tg, tan_lut[:, 1]),
                jnp.interp(theta, tg, tan_lut[:, 2]),
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
        A = self.n_agents

        # changedPractical: normalize the observed state to per-agent arrays (A, ...). Single-agent
        # obs is (…,); multi obs is (A, …). Agent 0 is always the ego (executed) drone.
        def _per_agent(v: NDArray, dim: int) -> NDArray:
            v = np.asarray(v)
            return v[None, :] if v.ndim == 1 else v  # (A, dim)

        pos_obs = _per_agent(obs["pos"], 3)  # (A, 3)
        quat_obs = _per_agent(obs["quat"], 4)
        vel_obs = _per_agent(obs["vel"], 3)
        angv_obs = _per_agent(obs["ang_vel"], 3)
        # rotor state per agent: reuse the ego thrust filter for all agents (opponent rotor state
        # is unobserved; a hover-ish proxy is adequate for a prediction rollout).
        rotor_obs = np.broadcast_to(np.asarray(self.thrust), (A, 4))
        obs_state = {
            "pos": jnp.asarray(pos_obs, device=self.sim.device),
            "quat": jnp.asarray(quat_obs, device=self.sim.device),
            "vel": jnp.asarray(vel_obs, device=self.sim.device),
            "ang_vel": jnp.asarray(angv_obs, device=self.sim.device),
            "rotor_vel": jnp.asarray(rotor_obs, device=self.sim.device),
        }

        # changedPractical: per-agent starting progress theta0. Ego integrates its committed
        # progress; opponents re-anchor by projecting their OBSERVED position onto their own spline
        # LUT each step (self-correcting, robust to prediction drift).
        theta_starts = []
        for a in range(A):
            if a == 0:
                theta_starts.append(self._theta[0])
            else:
                d = np.linalg.norm(self._ref_pos_np[a] - pos_obs[a][None, :], axis=1)
                th = float(self._theta_grid_np[a][int(d.argmin())])
                self._theta[a] = th
                theta_starts.append(th)
        theta0 = jnp.asarray(np.array(theta_starts), device=self.sim.device)  # (A,)
        theta0 = jnp.broadcast_to(theta0[None, :], (self.n_samples, A))  # (W, A)

        self._rng_key, subkey = jax.random.split(self._rng_key)

        # ONE shared rollout: returns per-agent lists.
        (
            new_means,
            new_sigmas,
            best_idx,
            costs_grouped,
            positions_grouped,
            mode_term_costs,
        ) = self._mppi_core_update(
            subkey,
            obs_state,
            theta0,
            self.mean_controls,
            self.noise_sigmas,
            self.obstacles,
            self.gate_frame_obstacles,
        )

        # Execute ONLY the ego (agent 0) winner; opponents are internal predictions.
        ego_best = best_idx[0]
        best_action = new_means[0][ego_best, 0]  # 5-dim: [roll, pitch, yaw, thrust, v_theta]

        # Receding-horizon shift of every agent's means & sigmas.
        vmap_shift = jax.vmap(self.shift_and_interpolate, in_axes=(0, None, None))
        self.mean_controls = [vmap_shift(m, self.dt, self.ctrl_dt) for m in new_means]
        self.noise_sigmas = [vmap_shift(s, self.dt, self.ctrl_dt) for s in new_sigmas]

        # Store for visualization / logging (ego-focused, plus per-agent for opponent rendering).
        self.best_mode_idx = int(ego_best)
        self.means = new_means
        self.costs = costs_grouped[0]
        self.positions = positions_grouped[0]
        self.all_positions = positions_grouped  # per agent (K_a, M_a, N, 3)
        self.all_best_idx = [int(b) for b in best_idx]
        self.all_costs = costs_grouped
        self.mode_term_costs = mode_term_costs[0]  # ego per-mode best-sample cost breakdown

        # changedPractical: split the 5-dim winner into the 4 real attitude/thrust commands and
        # the progress speed v_theta. Only the 4 real channels are EMA-smoothed and returned to
        # the env; v_theta is internal and used to advance the committed progress self._theta[0].
        full_action = np.asarray(best_action)  # back to CPU, 5-dim
        v_theta_cmd = float(full_action[4])
        action = full_action[:4]
        # changedPractical: EMA smoothing to damp inter-step oscillations from mode switching
        action = self._action_ema * action + (1.0 - self._action_ema) * self._prev_action
        self._prev_action = action
        # commit ego progress: advance theta by the executed v_theta over one control cycle
        s_total_ego = float(self._s_total_np[0])
        self._theta[0] = float(min(self._theta[0] + v_theta_cmd * self.ctrl_dt, s_total_ego))
        self.v_theta_cmd = v_theta_cmd  # store for logging / debugging
        if self._theta[0] >= s_total_ego - 1e-3:
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
                    t_total=self._spline_t_total,  # changedPractical: was 5.0; target matches PPO lap time (3.20s), 3.7 works limit(level0)
                    curvature_weight=2.0,  #  changedPractical: k=0.5 best per-segment match vs PPO (k>0.5 over-penalises the G0→G1 curve)
                    obstacles_pos=obs[
                        "obstacles_pos"
                    ],  # changedPractical: obs1 at (1.0,0.25) is 0.03m from no-detour spline
                    clearance=0.22,  # changedPractical: 0.16-0.21 flips obs3 detour to SW (wrong side, path 8.0m); 0.22 stays NE (path 7.74m, min_obs 0.21m)
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
            spline=self._planner[0].get_trajectory(300),
            waypoints=self._planner[0].waypoints,
            gates_pos=self.initial_obs["gates_pos"],
            gates_quat=self.initial_obs["gates_quat"],
            obstacles_pos=self.initial_obs["obstacles_pos"],
        )
        print(f"[MPPI] Saved {len(self._log_buf['t'])} steps → {out_path}")

    def _sample_agent(
        self, key: jax.Array, mean: jnp.ndarray, sigma: jnp.ndarray, K: int, M: int
    ) -> jnp.ndarray:
        """Sample + beta-smooth truncated-normal noise for ONE agent's K modes.

        Returns (K, M, N, nu). Bounds are relative to each mode's mean so samples stay in the
        physical action box. Called inside the jitted core update; K/M are static python ints.
        """
        lower_bound = self.act_low - mean  # (K, N, nu)
        upper_bound = self.act_high - mean
        keys = jax.random.split(key, K)

        def sample_per_mean(k_key, k_sigma, k_lb, k_ub):
            return self.get_truncated_normal_jax(
                k_key, mean=0, sd=k_sigma,
                x_min=k_lb[None, ...], x_max=k_ub[None, ...],
                shape=(M, self.N, self.num_inputs),
            )

        noise = vmap(sample_per_mean)(keys, sigma, lower_bound, upper_bound)  # (K, M, N, nu)

        # beta smoothing along the horizon (flatten K,M for the scan)
        noise_flat = noise.reshape(-1, self.N, self.num_inputs)

        def smooth_scan(carry, x):
            new_val = self.beta * carry + (1 - self.beta) * x
            return new_val, new_val

        _, noise_flat = jax.lax.scan(
            lambda c, x: vmap(smooth_scan)(c, x),
            jnp.zeros((noise_flat.shape[0], self.num_inputs)),
            noise_flat.transpose(1, 0, 2),
        )
        noise_flat = noise_flat.transpose(1, 0, 2)
        return noise_flat.reshape(K, M, self.N, self.num_inputs)

    def _update_modes(
        self,
        mean: jnp.ndarray,
        sigma: jnp.ndarray,
        noise: jnp.ndarray,
        costs_grouped: jnp.ndarray,
        K: int,
        M: int,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Elite/softmax per-mode MPPI update for ONE agent (vmapped over its K modes).

        Returns (new_means (K,N,nu), new_sigmas (K,N,nu), best_mode_idx). Byte-identical to the
        original update_single_mode logic.
        """

        def update_single_mode(k_mean, k_noise, k_costs, k_sigma):
            sorted_idx = jnp.argsort(k_costs)
            n_elites = max(int(M * self.elite_percentage), 1)
            elite_idx = sorted_idx[:n_elites]
            elite_noise = k_noise[elite_idx]
            elite_costs = k_costs[elite_idx]
            min_cost = jnp.min(elite_costs)
            weights = jnp.exp(-(elite_costs - min_cost) / self.temperature)
            weights = weights / (jnp.sum(weights) + 1e-10)
            delta = jnp.sum(weights[:, None, None] * elite_noise, axis=0)
            diff = elite_noise - delta
            weighted_var = jnp.sum(weights[:, None, None] * (diff**2), axis=0)
            new_sigma_sq = self.alpha * weighted_var + (1.0 - self.alpha) * (k_sigma**2)
            new_sigma_sq = jnp.maximum(new_sigma_sq, self.min_variance)
            return k_mean + delta, jnp.sqrt(new_sigma_sq), min_cost

        updated_means, updated_sigmas, cluster_best_costs = vmap(update_single_mode)(
            mean, noise, costs_grouped, sigma
        )
        best_mode_idx = jnp.argmin(cluster_best_costs)
        return updated_means, updated_sigmas, best_mode_idx

    @partial(jax.jit, static_argnames=["self"])
    def _mppi_core_update(
        self,
        key: jax.Array,
        obs: dict[str, jnp.ndarray],
        theta0: jnp.ndarray,
        means: list[jnp.ndarray],
        sigmas: list[jnp.ndarray],
        obstacles: jnp.ndarray,
        gate_frame_obstacles: jnp.ndarray,
    ) -> tuple[list, list, list, list, list, list]:
        """Joint MPPI update over ALL agents in ONE shared rollout batch.

        means/sigmas: length-n_agents lists; means[a]/sigmas[a] are (K_a, N, nu). theta0: (W, A).
        The expensive rollout is a single lax.scan; the per-agent sampling and elite update are a
        short (length-n_agents) Python loop, needed only because K may be ragged across agents.

        Returns per-agent lists: new_means, new_sigmas, best_mode_idx, costs_grouped (K_a, M_a),
        positions_grouped (K_a, M_a, N, 3), mode_term_costs (dict term -> (K_a,)).
        """
        # Per-agent RNG keys. For a single agent, reuse `key` directly so the sampled noise is
        # byte-identical to the pre-refactor single-agent controller.
        if self.n_agents == 1:
            agent_keys = [key]
        else:
            agent_keys = list(jax.random.split(key, self.n_agents))

        # --- 1. SAMPLE + assemble the shared control tensor (N, W, A, nu) ---
        noises, cands = [], []
        for a in range(self.n_agents):
            noise_a = self._sample_agent(agent_keys[a], means[a], sigmas[a], self.K[a], self.M[a])
            noises.append(noise_a)
            # candidate controls Mean[k] + Noise[k,m] -> (N, K_a*M_a=W, nu). world = k*M_a + m.
            cand_a = (means[a][:, None, ...] + noise_a).transpose(2, 0, 1, 3).reshape(
                self.N, self.n_samples, self.num_inputs
            )
            cands.append(cand_a)
        controls = jnp.stack(cands, axis=2)  # (N, W, A, nu)

        # --- 2. ONE rollout of all agents together ---
        totals, positions, terms = self.rollout_sim(
            obs, theta0, (controls, obstacles, gate_frame_obstacles)
        )
        # totals: (W, A); positions: (N, W, A, 3); terms: dict of (W, A)

        # --- 3. Per-agent grouped elite update (ragged K) ---
        new_means, new_sigmas, best_idx = [], [], []
        costs_grouped, positions_grouped, mode_term_costs = [], [], []
        for a in range(self.n_agents):
            Ka, Ma = self.K[a], self.M[a]
            costs_a = totals[:, a].reshape(Ka, Ma)  # (Ka, Ma)
            pos_a = positions[:, :, a, :].transpose(1, 0, 2).reshape(Ka, Ma, self.N, 3)
            terms_a = {name: v[:, a].reshape(Ka, Ma) for name, v in terms.items()}
            # per-mode breakdown: each mode's best (lowest-total) sample -> (Ka,)
            best_sample = jnp.argmin(costs_a, axis=1)
            k_idx = jnp.arange(Ka)
            mterm_a = {name: v[k_idx, best_sample] for name, v in terms_a.items()}
            m, s, bi = self._update_modes(means[a], sigmas[a], noises[a], costs_a, Ka, Ma)
            new_means.append(m)
            new_sigmas.append(s)
            best_idx.append(bi)
            costs_grouped.append(costs_a)
            positions_grouped.append(pos_a)
            mode_term_costs.append(mterm_a)

        return (
            new_means,
            new_sigmas,
            best_idx,
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
        obstacles: jnp.ndarray,
        gate_frame_obstacles: jnp.ndarray,
    ) -> dict[str, jnp.ndarray]:
        """Compute the per-term cost for ALL agents (MPCC contour/lag/progress formulation).

        Returns a dict term_name -> cost of shape (n_worlds, n_agents). The optimizer sums these;
        keeping them separate lets us accumulate each term over the horizon and log the per-mode
        breakdown. The pairwise drone-drone collision is symmetric and self-masked, so it is 0
        when n_agents==1 (the cost then reduces exactly to the single-agent formulation).

        theta / v_theta: (n_worlds, n_agents) progress (arc length) and progress speed per agent.
        """
        pos = data.states.pos  # (W, A, 3)
        vel = data.states.vel  # (W, A, 3) — real heading for the anisotropic keep-out
        ang_vel = data.states.ang_vel  # (W, A, 3)
        ang_acc = data.states_deriv.ang_vel  # (W, A, 3)
        cmd = data.controls.attitude.staged_cmd  # (W, A, 4)

        save_dist_obst = (
            self.initial_info["experiment"]["env"]["obstacle_radius"]
            + self.initial_info["experiment"]["env"]["drone_radius"]
        )
        save_dist_gate_obst = (
            self.initial_info["experiment"]["env"]["gate_frame_radius"]
            + self.initial_info["experiment"]["env"]["drone_radius"]
        )
        # Per-agent MPCC + obstacle terms via the pure single_drone_terms(). The loop unrolls over
        # the small agent axis at trace time; each agent uses its own spline LUT.
        per_agent, tangents = [], []
        for a in range(self.n_agents):
            ref_pos, tangent = vmap(self._ref_at_theta, in_axes=(0, None, None, None))(
                theta[:, a], self._theta_grid[a], self._ref_pos_lut[a], self._ref_tan_lut[a]
            )
            tangents.append(tangent)
            per_agent.append(
                single_drone_terms(
                    pos[:, a], ang_vel[:, a], ang_acc[:, a], cmd[:, a], ref_pos, tangent,
                    v_theta[:, a], obstacles, gate_frame_obstacles,
                    save_dist_obst, save_dist_gate_obst, self._weights,
                )
            )
        terms = {k: jnp.stack([pa[k] for pa in per_agent], axis=1) for k in per_agent[0]}  # (W, A)
        tangent = jnp.stack(tangents, axis=1)  # (W, A, 3) — per-agent path tangent

        # --- Symmetric pairwise drone-drone collision. Self-masked -> 0 for n_agents==1. ---
        # delta[w, a, b] = pos_a - pos_b. Orient the keep-out by the OTHER drone b's real velocity
        # heading (anisotropic overtake bubble); fall back to isotropic when b is ~stationary.
        safe = self.initial_info["experiment"]["env"]["drone_radius"] * 2.5
        delta = pos[:, :, None, :] - pos[:, None, :, :]  # (W, A, A, 3)
        d_iso = jnp.linalg.norm(delta, axis=-1) / safe  # (W, A, A)
        eye = jnp.eye(self.n_agents)
        if self.use_anisotropic_opp:
            spd_b = jnp.linalg.norm(vel, axis=-1)  # (W, A) speed of each drone
            hb = vel / (spd_b[..., None] + 1e-6)  # (W, A, 3) unit heading of b
            hb = hb[:, None, :, :]  # (W, 1, A, 3) broadcast over acting-drone axis a
            d_par = jnp.sum(delta * hb, axis=-1)  # (W, A, A) along b's heading
            d_perp = jnp.linalg.norm(delta - d_par[..., None] * hb, axis=-1)  # (W, A, A)
            d_aniso = jnp.sqrt((d_par / self.opp_axial) ** 2 + (d_perp / self.opp_lateral) ** 2)
            moving_b = (spd_b > 0.2)[:, None, :]  # (W, 1, A)
            d = jnp.where(moving_b, d_aniso, d_iso)
        else:
            d = d_iso
        coll = jnp.exp(-(d**2)) * (1.0 - eye)  # (W, A, A), self-pairs masked
        terms["opp_drone"] = self.w_opp_drone_exp * coll.sum(-1)  # (W, A)

        # --- behind-aware contour relaxation (opt-in) ---
        # If another drone b is ahead of a along a's path tangent AND within behind_radius, scale
        # down a's contour cost so it may leave the racing line to overtake.
        if self.use_behind_contour:
            ahead = jnp.sum(tangent[:, :, None, :] * (-delta), axis=-1)  # (W, A, A): b ahead of a
            dist = jnp.linalg.norm(delta, axis=-1)  # (W, A, A)
            behind = ((ahead > 0.0) & (dist < self.behind_radius) & (eye == 0.0)).any(axis=-1)
            terms["contour"] = terms["contour"] * jnp.where(
                behind, self.contour_behind_scale, 1.0
            )
        return terms

    @partial(jax.jit, static_argnames=["self"])
    def apply_input(
        self,
        carry: tuple[SimData, jnp.ndarray],
        info: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> tuple[tuple[SimData, jnp.ndarray], jnp.ndarray]:
        """Roll out the sim for one step (all agents) and advance each agent's MPCC progress theta.

        Args:
            carry: (sim data, theta) where theta is (n_worlds, n_agents) progress (arc length).
            info: (cmd5, obstacles, gate_frame_obstacles). cmd5 is (n_worlds, n_agents, 5): the
                first 4 channels are the attitude/thrust command sent to the sim; the 5th is
                v_theta, the progress speed used to integrate theta.
        """
        data, theta = carry
        cmd, obstacles, gate_frame_obstacles = info
        v_theta = cmd[..., 4]  # (W, A)
        cmd4 = cmd[..., :4]  # (W, A, 4) real attitude/thrust channels
        # Integrate progress forward one planning step (clamped per agent to its own path end).
        theta_next = jnp.clip(theta + v_theta * self.dt, 0.0, self._s_total[None, :])
        data = data.replace(
            controls=data.controls.replace(
                attitude=data.controls.attitude.replace(staged_cmd=cmd4)
            )
        )
        next_data = self.step_fn(data, self.sim.freq // self.sim.control_freq)
        cost_terms = self.compute_cost(
            next_data, theta_next, v_theta, obstacles, gate_frame_obstacles
        )
        return (next_data, theta_next), (cost_terms, next_data.states.pos)

    @partial(jax.jit, static_argnames=["self"])
    def rollout_sim(
        self,
        obs: dict,
        theta0: jnp.ndarray,
        infos: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> jnp.ndarray:
        """Rolls out the sim (all agents) for scan.

        theta0: (n_worlds, n_agents) starting progress (arc length); each agent's worlds all start
        at that agent's committed progress. obs values are (n_agents, ...) broadcast over worlds.
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
        controls_flat, obstacles, gate_frame_obstacles = infos
        # scan iterates over axis 0 (N time steps); tile obstacles so each step gets a slice
        obstacles_tiled = jnp.broadcast_to(obstacles[None], (self.N,) + obstacles.shape)
        gate_frame_obstacles_tiled = jnp.broadcast_to(
            gate_frame_obstacles[None], (self.N,) + gate_frame_obstacles.shape
        )

        # changedPractical: carry (sim data, theta) so each rollout integrates its own progress
        _, (cost_terms, positions) = scan(
            self.apply_input,
            (data, theta0),
            (controls_flat, obstacles_tiled, gate_frame_obstacles_tiled),
        )
        # cost_terms: dict of (N, W, A). Sum each term over the horizon -> (W, A).
        terms_summed = {k: jnp.sum(v, axis=0) for k, v in cost_terms.items()}
        total = sum(terms_summed.values())  # (W, A)
        return total, positions, terms_summed

    # changedPractical

    def _draw_reference(self, sim: Sim):
        """Draw the reference spline, current setpoint, and waypoints."""
        # changedPractical: setpoint is now the reference at the ego's committed progress theta
        # (arc length), not at wall-clock time. Per-agent LUTs/theta are lists; index [0] = ego.
        idx = min(
            int(np.searchsorted(self._theta_grid_np[0], self._theta[0])),
            len(self._ref_pos_np[0]) - 1,
        )
        setpoint = self._ref_pos_np[0][idx].reshape(1, -1)
        draw_points(sim, setpoint, rgba=(1.0, 0.0, 0.0, 1.0), size=0.02)
        trajectory = self._planner[0].get_trajectory(100)
        draw_line(sim, trajectory, rgba=(0.0, 1.0, 0.0, 1.0))
        draw_points(sim, self._planner[0].waypoints, rgba=(0.0, 0.0, 1.0, 1.0), size=0.03)

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

        positions = np.asarray(self.positions)  # (K, M, N, 3) — ego (agent 0)
        costs = np.asarray(self.costs)  # (K, M)
        best_k = int(self.best_mode_idx)
        n_viz = 3

        for k in range(self.K[0]):
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
