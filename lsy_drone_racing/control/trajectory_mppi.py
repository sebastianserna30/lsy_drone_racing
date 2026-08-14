"""Multi-modal MPPI attitude controller.

A fixed cubic-spline reference is built once at reset and reparameterized by arc length; the
controller then optimizes attitude commands to track it, sampling K modes in parallel and
rolling every sample through a batched GPU sim. The building blocks live in
``lsy_drone_racing.control.mppi``.
"""

from __future__ import annotations  # Python 3.10 type hints

import os
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from crazyflow.control import Control
from crazyflow.sim import Physics, Sim
from drone_models.core import load_params
from jax import random
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.mppi import (
    ConfigError,
    MPPIConfig,
    diagnostics,
    ibr,
    opponents,
    optimizer,
    reference,
    rollout,
    sampling,
)
from lsy_drone_racing.control.mppi import cost as cost_mod
from lsy_drone_racing.control.mppi.cost import HOVER_THRUST  # noqa: F401  (re-exported)
from lsy_drone_racing.control.spline_planner import SplinePlanner

if TYPE_CHECKING:
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
        self,
        initial_obs: dict[str, NDArray[np.floating]],
        info: dict,
        initial_info: dict,
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

        self.cfg = cfg = MPPIConfig.from_config(initial_info)

        self.N = cfg.N
        self.T = cfg.T
        self.dt = cfg.dt
        self.f = int(self.N / self.T)
        self.ctrl_dt = cfg.ctrl_dt
        self.n_samples = cfg.n_samples
        # Agent 0 is the ego (the drone we execute); agents 1.. are opponents.
        _pos0 = np.asarray(initial_obs["pos"])
        self.n_agents = _pos0.shape[0] if _pos0.ndim == 2 else 1

        def _agent_cfg(a: int, key: str, default: object) -> object:
            """Per-agent override from [[controller.mppi.agents]], or the shared value."""
            if (
                a < len(cfg.agents)
                and (value := getattr(cfg.agents[a], key)) is not None
            ):
                return value
            return default

        self.K = [int(_agent_cfg(a, "K", cfg.K)) for a in range(self.n_agents)]
        self.M = [self.n_samples // k for k in self.K]  # samples per mode, per agent
        for a, k in enumerate(self.K):
            assert self.n_samples % k == 0, (
                f"n_samples ({self.n_samples}) must be divisible by K[{a}]={k}"
            )
        self._agent_cfg = _agent_cfg  # stashed for per-agent spline params below
        # "mppi" optimizes the opponents in the same rollout batch, which assumes they run our
        # controller. The predictor models instead simulate only the ego and score it against a
        # kinematic opponent trajectory — half the sim cost, and no assumption about a real
        # opponent's controller.
        self.opponent_model = cfg.opponent.model
        self.n_sim_agents = self.n_agents if self.opponent_model == "mppi" else 1
        self._match_pace = cfg.opponent.match_pace
        # [roll, pitch, yaw, thrust, v_theta]. The sim only ever receives the first 4; v_theta is
        # the MPCC progress speed, integrated in the rollout.
        self.num_inputs = 5

        self._t = 0.0

        if jax.default_backend() == "cpu":
            available_device = "cpu"
        else:
            available_device = "cuda"

        self.sim = Sim(
            n_worlds=self.n_samples,
            n_drones=self.n_sim_agents,
            attitude_freq=self.f,
            freq=self.f,
            physics=Physics.so_rpy_rotor_drag,
            control=Control.attitude,
            drone_model="cf21B_500",
            device=available_device,  # TODO get from info
        )
        self.sim.reset()

        self.step_fn = self.sim.build_step_fn()

        # Plain Python floats, so they fold into the JAX trace as compile-time constants.
        cost, opp = cfg.cost, cfg.opponent
        self.w_opp_drone_exp = opp.drone_exp

        # Insertion order is load-bearing: the rollout sums these in iteration order and float
        # addition is not associative, so reordering perturbs the trajectory.
        self._weights = {
            "lag": cost.lag,  # ahead/behind-of-theta penalty (heavy)
            "contour": cost.contour,  # off-path penalty
            "z": cost.z,  # extra penalty for altitude error
            "progress": cost.progress,  # mu: reward for advancing theta
            "ang_vel": cost.ang_vel,
            "ang_acc": cost.ang_acc,
            "tilt": cost.tilt,  # was 5.0; loosened to allow aggressive roll/pitch
            "thrust": cost.thrust,  # was 0.0; regularise toward hover thrust
            "yaw": cost.yaw,  # was 0.0; added to stabilise yaw oscillation
            "obstacle": cost.obstacle,
            "floor": cost.floor,
            "floor_z": cost.floor_z,
        }
        self._spline_t_total = cfg.spline.t_total
        self.v_theta_max = cfg.v_theta_max
        # Per-agent (K_a, N, 5) means and sigmas. v_theta gets its own noise scale because it is
        # in m/s while the other four channels are rad and N. Predictor-mode opponents are
        # kinematic and never sampled, so only sim agents get an entry.
        self.noise_sigmas = []
        self.mean_controls = []
        for k in self.K[: self.n_sim_agents]:
            sig = jnp.full(
                (k, self.N, self.num_inputs), cfg.noise_sigma, device=self.sim.device
            )
            sig = sig.at[:, :, 4].set(cfg.v_theta_sigma)
            self.noise_sigmas.append(sig)
            m = jnp.zeros((k, self.N, self.num_inputs), device=self.sim.device)
            m = m.at[:, :, 3].set(HOVER_THRUST)
            m = m.at[:, :, 4].set(cfg.v_theta_init)
            self.mean_controls.append(m)

        # Opponent representative worlds for "mppi" mode. Sample m=0 of every opponent mode is
        # forced to zero noise in _mppi_core_update, so world k*M_a carries that mode's mean.
        # Scoring the ego against these instead of the world-paired opponent sample is what stops
        # its collision cost being a lottery over random opponent draws.
        rep_w, rep_a = [], []
        if self.opponent_model == "mppi":
            for a in range(1, self.n_sim_agents):
                for k in range(self.K[a]):
                    rep_w.append(k * self.M[a])
                    rep_a.append(a)
        if rep_w:
            self._rep_w = jnp.asarray(
                np.array(rep_w, dtype=np.int32), device=self.sim.device
            )
            self._rep_a = jnp.asarray(
                np.array(rep_a, dtype=np.int32), device=self.sim.device
            )
        else:
            self._rep_w = self._rep_a = (
                None  # never dereferenced (single-agent / predictor mode)
            )

        self.obstacles = jnp.array(initial_obs["obstacles_pos"], device=self.sim.device)
        gate_frame_pos = self.get_gate_frame_pos(
            initial_obs["gates_pos"], initial_obs["gates_quat"]
        )
        self.gate_frame_obstacles = jnp.array(gate_frame_pos, device=self.sim.device)

        self.drone_params = load_params("first_principles", cfg.drone_model)
        self.drone_mass = self.drone_params["mass"]
        # Channel 4 (v_theta) is bounded at 0 below, so the drone never runs the reference
        # backwards.
        self.act_low = -jnp.ones(self.num_inputs, device=self.sim.device) * jnp.pi / 2
        self.act_low = self.act_low.at[3].set(self.drone_params["thrust_min"] * 4)
        self.act_low = self.act_low.at[4].set(0.0)
        self.act_high = jnp.ones(self.num_inputs, device=self.sim.device) * jnp.pi / 2
        self.act_high = self.act_high.at[3].set(self.drone_params["thrust_max"] * 4)
        self.act_high = self.act_high.at[4].set(self.v_theta_max)
        # Held on self, which is a static JIT argument, so these stay trace constants.
        self._sampler = sampling.SamplerParams(
            N=self.N,
            num_inputs=self.num_inputs,
            beta=cfg.beta,
            elite_percentage=cfg.elite_percentage,
            temperature=cfg.temperature,
            alpha=cfg.alpha,
            min_variance=cfg.min_variance,
            act_low=self.act_low,
            act_high=self.act_high,
        )
        # The wake cylinder extends only DOWNWARD from the opponent, so the ego's cheapest exit
        # from it is upward, through the opponent itself. That path is blocked only by the
        # collision term, so a wake penalty at or above `drone_exp` makes flying into the other
        # drone the rational move. Checked here rather than in config.py because the opponent
        # block is dead code at n_agents == 1, where the ordering does not matter.
        if self.n_agents > 1 and opp.downwash >= opp.drone_exp:
            raise ConfigError(
                f"controller.mppi.opponent: downwash ({opp.downwash}) must be < drone_exp "
                f"({opp.drone_exp}). The wake keep-out has no top, so its only exit is through "
                "the opponent; if that exit is cheaper than the collision, the ego climbs into "
                "the other drone instead of avoiding it."
            )
        self._opp_cost = cost_mod.OpponentCostParams(
            drone_radius=cfg.geometry.drone_radius,
            drone_exp=opp.drone_exp,
            use_anisotropic=opp.use_anisotropic,
            axial=opp.axial,
            lateral=opp.lateral,
            core_radius=opp.core_radius,
            blend_v0=opp.blend_v0,
            blend_width=opp.blend_width,
            downwash=opp.downwash,
            downwash_radius=opp.downwash_radius,
            downwash_dz=opp.downwash_dz,
        )
        self.thrust = np.zeros(4)
        self._prev_action = np.array(
            [0.0, 0.0, 0.0, HOVER_THRUST]
        )  # [roll, pitch, yaw, thrust]
        self._action_ema = float(cfg.action_ema)  # damps mode-switching oscillation

        # Keep the observation dtype: casting to float64 here shifts the spline in the last bits,
        # and the chaotic MPPI amplifies that into a visibly different trajectory.
        _start = np.asarray(initial_obs["pos"])
        if _start.ndim == 1:
            _start = _start[None, :]
        self._start_pos = _start.copy()  # (A, 3)
        self._tracker = opponents.OpponentTracker(
            self.n_agents,
            initial_obs,
            opponents.TrackerParams(
                ctrl_dt=self.ctrl_dt,
                vel_ema=opp.vel_ema,
                stale_inflate_rate=opp.stale_inflate_rate,
                stale_max=opp.stale_max,
            ),
        )
        self._predictor = opponents.PredictorParams(
            model=self.opponent_model,
            horizon=self.N,
            dt=self.dt,
            v_theta_max=self.v_theta_max,
            offset_tau=opp.pred_offset_tau,
        )
        # False forces the next anchor_theta call to bootstrap with a global search.
        self._opp_anchored = [False] * self.n_agents
        self._opp_pred_np = (
            None  # kinematic opponent predictions (N, A-1, 3) for rendering
        )
        self._opp_inflate_np = np.ones(max(self.n_agents - 1, 1), dtype=np.float32)
        self._last_gates_pos = None
        self._last_gates_quat = None
        # One planner per agent: same gates, but a per-agent start and optional per-agent timing,
        # so an opponent can fly a different racing line than the ego.
        self._planner = [
            SplinePlanner(
                self._start_pos[a],
                initial_obs,
                t_total=float(self._agent_cfg(a, "t_total", self._spline_t_total)),
                curvature_weight=float(
                    self._agent_cfg(a, "curvature_weight", cfg.spline.curvature_weight)
                ),
                obstacles_pos=initial_obs["obstacles_pos"],
                clearance=float(self._agent_cfg(a, "clearance", cfg.spline.clearance)),
                dip_allowed=bool(
                    self._agent_cfg(a, "dip_allowed", cfg.spline.dip_allowed)
                ),
            )
            for a in range(self.n_agents)
        ]

        # Splines are fixed at reset; the controller never replans waypoints mid-run.
        self._paths = reference.build_paths(
            self._planner, cfg.spline.lut_samples, self.sim.device
        )
        self._stage_cost_fn = cost_mod.build_stage_cost_fn(
            weights=self._weights,
            paths=self._paths,
            geometry=cfg.geometry,
            n_sim_agents=self.n_sim_agents,
        )
        self._opp_state_fn = cost_mod.build_opp_state_fn(
            opponent_model=self.opponent_model, rep_w=self._rep_w, rep_a=self._rep_a
        )
        self._coupled_cost_fn = cost_mod.build_coupled_cost_fn(
            opp_params=self._opp_cost, n_sim_agents=self.n_sim_agents
        )
        # Best response needs every agent to carry a bank of candidate rollouts, which only the
        # joint "mppi" model provides; the kinematic predictors have a single trajectory and
        # nothing to iterate against, so they keep the one-shot coupled cost.
        # ibr_iters < 0 disables best response entirely and falls back to the older scheme, where
        # the ego is scored against the opponents' mode means. Kept as the A/B control: without
        # it there is no way to measure what the Nash coupling is actually buying.
        self._use_ibr = (
            self.opponent_model == "mppi" and self.n_agents > 1 and opp.ibr_iters >= 0
        )
        self._ibr_fn = (
            ibr.build_ibr_fn(
                opp_params=self._opp_cost,
                n_sim_agents=self.n_sim_agents,
                dt=self.dt,
                n_iters=opp.ibr_iters,
                mode=opp.ibr_mode,
            )
            if self._use_ibr
            else None
        )
        self._rollout_fn = rollout.build_rollout_fn(
            sim=self.sim,
            step_fn=self.step_fn,
            stage_cost_fn=self._stage_cost_fn,
            opp_state_fn=self._opp_state_fn,
            horizon=self.N,
            dt=self.dt,
            s_total=self._paths.s_total[None, : self.n_sim_agents],
        )
        self._update_fn = optimizer.build_update_fn(
            rollout_fn=self._rollout_fn,
            coupled_cost_fn=self._coupled_cost_fn,
            ibr_fn=self._ibr_fn,
            sampler=self._sampler,
            K=self.K,
            M=self.M,
            horizon=self.N,
            n_samples=self.n_samples,
            num_inputs=self.num_inputs,
            n_sim_agents=self.n_sim_agents,
            n_agents=self.n_agents,
        )
        self._anchor_params = reference.AnchorParams(
            v_theta_max=self.v_theta_max,
            ctrl_dt=self.ctrl_dt,
            fwd=opp.anchor_fwd,
            reacquire_dist=opp.anchor_reacquire_dist,
        )
        # committed progress (arc length, m) per agent. Rollouts start each horizon here.
        self._theta = [0.0] * self.n_agents

        self._finished = False
        self._rng_key = jax.random.PRNGKey(0)
        self._rng_key, subkey = random.split(self._rng_key)

        self._logger = (
            diagnostics.CostLogger()
        )  # must exist before warmup calls compute_control
        for i in range(10):
            a = self.compute_control(initial_obs, info)  # Warm up the controller
            jax.block_until_ready(a)
        # Warmup ran 10 steps on a frozen initial_obs. Undo everything it advanced, or the first
        # real step starts 0.2 s into the reference and every logged sample plots late.
        self._t = 0.0
        self._theta = [0.0] * self.n_agents
        self._opp_anchored = [False] * self.n_agents

        if os.getenv("LOG_DRONE_DATA"):
            self._logger.enable()

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

        # Single-agent obs is (...,); multi is (A, ...). Normalize to (A, ...), ego first.
        def _per_agent(v: NDArray, dim: int) -> NDArray:
            v = np.asarray(v)
            return v[None, :] if v.ndim == 1 else v  # (A, dim)

        pos_obs = _per_agent(obs["pos"], 3)  # (A, 3)
        quat_obs = _per_agent(obs["quat"], 4)
        vel_obs = _per_agent(obs["vel"], 3)
        angv_obs = _per_agent(obs["ang_vel"], 3)
        # Filter before anything consumes the obs: a NaN reaching the rollout poisons every
        # sample.
        if A > 1:
            pos_obs, quat_obs, vel_obs, angv_obs, opp_inflate = self._tracker.update(
                pos_obs, quat_obs, vel_obs, angv_obs
            )
            self._opp_inflate_np = opp_inflate  # (A-1,), kept on the host for rendering
        # Opponent rotor state is unobservable, so reuse the ego thrust filter as a proxy.
        rotor_obs = np.broadcast_to(np.asarray(self.thrust), (self.n_sim_agents, 4))
        obs_state = {
            "pos": jnp.asarray(pos_obs[: self.n_sim_agents], device=self.sim.device),
            "quat": jnp.asarray(quat_obs[: self.n_sim_agents], device=self.sim.device),
            "vel": jnp.asarray(vel_obs[: self.n_sim_agents], device=self.sim.device),
            "ang_vel": jnp.asarray(
                angv_obs[: self.n_sim_agents], device=self.sim.device
            ),
            "rotor_vel": jnp.asarray(rotor_obs, device=self.sim.device),
        }

        # The ego integrates its committed progress; opponents re-anchor onto their own spline
        # from the observed position each step, which is self-correcting under prediction drift.
        theta_starts = []
        for a in range(A):
            if a == 0:
                theta_starts.append(self._theta[0])
            else:
                th, self._opp_anchored[a] = reference.anchor_theta(
                    self._paths,
                    a,
                    pos_obs[a],
                    self._tracker.vel_filt[a],
                    float(self._theta[a]),
                    self._opp_anchored[a],
                    self._anchor_params,
                )
                self._theta[a] = th
                theta_starts.append(th)
        # Drive the modelled opponent's reference forward at the speed it is actually flying,
        # instead of at whatever pace the ego's own progress reward pulls it to.
        if A > 1 and self.opponent_model == "mppi" and self._match_pace:
            for a in range(1, self.n_sim_agents):
                v_prog = opponents.forward_speed(
                    self._paths,
                    a,
                    theta_starts[a],
                    self._tracker.vel_filt[a],
                    self.v_theta_max,
                )
                # Re-pinned every step, so the progress reward cannot ratchet the modelled
                # opponent's pace up over time. Sigma stays > 0: truncated_normal divides by it.
                self.mean_controls[a] = self.mean_controls[a].at[:, :, 4].set(v_prog)
                self.noise_sigmas[a] = self.noise_sigmas[a].at[:, :, 4].set(0.05)

        # Predictor-mode opponents keep theta on the host for the prediction, but never enter
        # the rollout.
        theta0 = jnp.asarray(
            np.array(theta_starts[: self.n_sim_agents]), device=self.sim.device
        )  # (A_sim,)
        theta0 = jnp.broadcast_to(theta0[None, :], (self.n_samples, self.n_sim_agents))

        # "mppi" mode gathers its representatives in-batch and needs only the staleness
        # inflation; predictor modes build the opponent trajectories here on the host.
        self._opp_pred_np = None
        opp_pred_pos = np.zeros((self.N, 1, 3), dtype=np.float32)  # dummy when unused
        opp_pred_vel = np.zeros((self.N, 1, 3), dtype=np.float32)
        inflate_arr = np.ones(1, dtype=np.float32)
        if A > 1:
            if self._use_ibr:
                # (A, A): entry (a, b) is the inflation applied when agent a is scored against
                # agent b. Only the ego's row carries real staleness — an opponent's view of us
                # is never stale, because our own state comes from the flight controller.
                inflate_arr = np.ones(
                    (self.n_sim_agents, self.n_sim_agents), dtype=np.float32
                )
                inflate_arr[0, 1:] = opp_inflate[: self.n_sim_agents - 1]
            elif self.opponent_model == "mppi":
                # rep order matches self._rep_w/_rep_a construction: (agent a, mode k)
                inflate_arr = np.concatenate(
                    [
                        np.full(self.K[a], opp_inflate[a - 1], dtype=np.float32)
                        for a in range(1, self.n_sim_agents)
                    ]
                )
            else:
                opp_pred_pos, opp_pred_vel = opponents.predict(
                    self._predictor,
                    self._paths,
                    self._tracker.held["pos"],
                    self._tracker.vel_filt,
                    self._theta,
                    self.n_agents,
                )
                self._opp_pred_np = opp_pred_pos
                inflate_arr = opp_inflate
        opp_pred_pos_j = jnp.asarray(opp_pred_pos, device=self.sim.device)
        opp_pred_vel_j = jnp.asarray(opp_pred_vel, device=self.sim.device)
        opp_inflate_j = jnp.asarray(inflate_arr, device=self.sim.device)

        self._rng_key, subkey = jax.random.split(self._rng_key)

        # ONE shared rollout: returns per-agent lists.
        (
            new_means,
            new_sigmas,
            best_idx,
            costs_grouped,
            positions_grouped,
            mode_term_costs,
            ibr_best_samples,
        ) = self._update_fn(
            subkey,
            obs_state,
            theta0,
            self.mean_controls,
            self.noise_sigmas,
            self.obstacles,
            self.gate_frame_obstacles,
            opp_pred_pos_j,
            opp_pred_vel_j,
            opp_inflate_j,
        )

        # Execute ONLY the ego (agent 0) winner; opponents are internal predictions.
        ego_best = best_idx[0]
        best_action = new_means[0][
            ego_best, 0
        ]  # 5-dim: [roll, pitch, yaw, thrust, v_theta]

        # Receding-horizon shift of every agent's means & sigmas.
        vmap_shift = jax.vmap(sampling.shift_and_interpolate, in_axes=(0, None, None))
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
        self.mode_term_costs = mode_term_costs[
            0
        ]  # ego per-mode best-sample cost breakdown
        # Which rollout each agent settled on at the best-response fixed point. Worth watching:
        # if these keep flipping between control steps the iteration is oscillating rather than
        # converging, which is the failure mode of Jacobi updates on a tightly coupled game.
        self.ibr_best_samples = (
            None if ibr_best_samples is None else np.asarray(ibr_best_samples)
        )

        # v_theta is internal: it advances the committed progress rather than reaching the env.
        full_action = np.asarray(best_action)  # back to CPU, 5-dim
        v_theta_cmd = float(full_action[4])
        action = full_action[:4]
        action = (
            self._action_ema * action + (1.0 - self._action_ema) * self._prev_action
        )
        self._prev_action = action
        # commit ego progress: advance theta by the executed v_theta over one control cycle
        s_total_ego = float(self._paths.s_total_np[0])
        self._theta[0] = float(
            min(self._theta[0] + v_theta_cmd * self.ctrl_dt, s_total_ego)
        )
        self.v_theta_cmd = v_theta_cmd  # store for logging / debugging
        if self._theta[0] >= s_total_ego - 1e-3:
            self._finished = True
        self.thrust += (
            self.drone_params["thrust_dyn_coef"]
            * (action[3] - self.thrust)
            * self.ctrl_dt
        )

        if self._logger.active:
            # The opponent trajectories the ego was actually scored against this step, (N, P, 3):
            # the zero-noise mode representatives in "mppi" mode, the kinematic prediction
            # otherwise. Logged so prediction error can be measured against the true positions.
            opp_pred_log = self._opp_pred_np
            if A > 1 and self.opponent_model == "mppi":
                reps = [
                    np.asarray(self.all_positions[a][:, 0])
                    for a in range(1, self.n_sim_agents)
                ]
                opp_pred_log = np.concatenate(reps, axis=0).transpose(
                    1, 0, 2
                )  # (N, P, 3)
            self._logger.log_step(
                self._t,
                obs,
                action,
                self.mode_term_costs,
                self.costs,
                self.best_mode_idx,
                opp_pred_log,
                self.ibr_best_samples,
            )

        return action

    def step_callback(
        self,
        action: NDArray[np.floating] | None = None,
        obs: dict[str, NDArray[np.floating]] | None = None,
        reward: float | None = None,
        terminated: bool | None = None,
        truncated: bool | None = None,
        info: dict | None = None,
    ) -> bool:
        """Refresh obstacle/gate keep-outs from the sensed positions."""
        self.obstacles = jnp.array(obs["obstacles_pos"], device=self.sim.device)

        gates_changed = (
            self._last_gates_pos is None
            or not jnp.allclose(obs["gates_pos"], self._last_gates_pos)
            or not jnp.allclose(obs["gates_quat"], self._last_gates_quat)
        )

        if gates_changed:
            gate_frame_pos = self.get_gate_frame_pos(
                obs["gates_pos"], obs["gates_quat"]
            )
            self.gate_frame_obstacles = jnp.array(
                gate_frame_pos, device=self.sim.device
            )

            self._last_gates_pos = obs["gates_pos"].copy()
            self._last_gates_quat = obs["gates_quat"].copy()

        if obs.get("target_gate", 0) == -1:
            self._finished = True
        return self._finished

    def episode_callback(self):
        """Save logged data to disk if LOG_DRONE_DATA is set."""
        self._logger.save(
            spline=self._planner[0].get_trajectory(300),
            waypoints=self._planner[0].waypoints,
            gates_pos=self.initial_obs["gates_pos"],
            gates_quat=self.initial_obs["gates_quat"],
            obstacles_pos=self.initial_obs["obstacles_pos"],
        )

    def render_callback(self, sim: Sim):
        """Visualize the desired trajectory and the current setpoint."""
        diagnostics.draw_reference(
            sim,
            self._planner[0],
            self._paths.pos_np[0],
            self._paths.theta_grid_np[0],
            self._theta[0],
        )
        if getattr(self, "positions", None) is not None:
            diagnostics.draw_rollouts(
                sim, self.positions, self.costs, self.best_mode_idx
            )
            # Predictor modes have no opponent rollouts; draw the kinematic prediction instead.
            if self._opp_pred_np is not None:
                diagnostics.draw_opponent_predictions(sim, self._opp_pred_np)
            for a in range(1, self.n_sim_agents):
                diagnostics.draw_opponent_rollouts(
                    sim, self.all_positions[a], self.all_costs[a]
                )
            # The keep-out the ego is actually penalised by, drawn on the tracked opponent.
            if self.n_agents > 1:
                diagnostics.draw_opponent_keepout(
                    sim,
                    self._tracker.held["pos"][1:],
                    self._tracker.vel_filt[1:],
                    self._opp_inflate_np,
                    self._opp_cost,
                )
                diagnostics.draw_opponent_downwash(
                    sim, self._tracker.held["pos"][1:], self._opp_inflate_np, self._opp_cost
                )
        diagnostics.draw_obstacles(sim, self.obstacles, self.gate_frame_obstacles)
