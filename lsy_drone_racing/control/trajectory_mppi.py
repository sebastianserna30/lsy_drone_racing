"""TODO."""

from __future__ import annotations  # Python 3.10 type hints

import os
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from crazyflow.sim.visualize import draw_line, draw_points
from crazyflow.control import Control
from crazyflow.sim import Physics, Sim
from crazyflow.sim.data import SimData

# from crazyflow_experiments.sim2real.control.trajectory_generator import (
#    TrajectoryGenerator3DPeriodicMotion,
# )
from drone_models.core import load_params
from drone_models.so_rpy_rotor_drag import dynamics
from jax import random, vmap
from jax.lax import scan

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.spline_planner import SplinePlanner

if TYPE_CHECKING:
    from numpy.typing import NDArray


class AttitudeMPPIController(Controller):
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

        self.N = initial_info["controller"]["mppi"]["N"]
        self.T = initial_info["controller"]["mppi"]["T"]
        self.dt = self.T / self.N
        self.dt_array = jnp.arange(0, self.T, self.dt)
        self.f = int(self.N / self.T)
        assert np.isclose(self.f, self.N / self.T), (
            "N must be divisible by T for consistent time steps"
        )
        self.ctrl_dt = 1 / initial_info["controller"]["ctrl_freq"]
        self.n_samples = initial_info["controller"]["mppi"]["n_samples"]
        self.K = initial_info["controller"]["mppi"]["K"]
        self.M = self.n_samples // self.K  # samples per mode

        # changedPractical
        self._t = 0.0

        self.sim = Sim(
            n_worlds=self.n_samples,
            n_drones=1,
            attitude_freq=self.f,
            freq=self.f,
            physics=Physics.so_rpy_rotor_drag,
            control=Control.attitude,
            drone_model="cf21B_500",
            device="cuda",  # TODO get from info
        )
        self.sim.reset()

        self.step_fn = self.sim.build_step_fn()

        self.noise_sigmas = jnp.full(
            (self.K, self.N, 4),
            fill_value=initial_info["controller"]["mppi"]["noise_sigma"],
            device=self.sim.device,
        )
        self.temperature = initial_info["controller"]["mppi"]["temperature"]
        self.elite_percentage = initial_info["controller"]["mppi"]["elite_percentage"]
        self.beta = initial_info["controller"]["mppi"]["beta"]
        self.alpha = initial_info["controller"]["mppi"]["alpha"]
        self.min_variance = initial_info["controller"]["mppi"]["min_variance"]

        _init = jnp.zeros((self.K, self.N, 4), device=self.sim.device)
        self.mean_controls = _init.at[:, :, 3].set(0.43)

        # Shape: (Num_Obstacles, 3)
        # changedPractical
        self.obstacles = jnp.array(initial_obs["obstacles_pos"], device=self.sim.device)

        # changedPractical
        # self.low_level_ctrl_freq = initial_info["low_level_ctrl_freq"]
        # self.drone_params = load_params("first_principles", initial_info["drone_model"])
        self.drone_params = load_params("first_principles", initial_info["drone"]["model"])
        self.drone_mass = self.drone_params["mass"]
        self.act_low = -jnp.ones(4, device=self.sim.device) * jnp.pi / 2
        self.act_low = self.act_low.at[3].set(self.drone_params["thrust_min"] * 4)
        self.act_high = jnp.ones(4, device=self.sim.device) * jnp.pi / 2
        self.act_high = self.act_high.at[3].set(self.drone_params["thrust_max"] * 4)
        self.thrust = np.zeros(4)

        # changedPractical
        self._start_pos = initial_obs["pos"].copy()
        self._planner = SplinePlanner(
            self._start_pos, initial_obs, 6.0,
            obstacles_pos=initial_obs["obstacles_pos"],
            clearance=0.20,
        )

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
        info_short = {"rng_key": subkey, "obstacles": jnp.array(initial_obs["obstacles_pos"])}

        self._log_buf: dict | None = None  # must be set before warmup calls compute_control
        for i in range(10):
            a = self.compute_control(initial_obs, info_short)  # Warm up the controller
            jax.block_until_ready(a)

        if os.getenv("LOG_DRONE_DATA"):
            self._log_buf = {
                "t": [], "pos": [], "vel": [], "action": [],
                "des_pos": [], "des_vel": [], "min_cost": [], "target_gate": [],
                "cost_pos": [], "cost_z": [], "cost_vel": [], "min_obs_dist": [],
            }

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
        if t >= self._t_end:
            self._finished = True

        des_pos, des_vel, des_acc, des_yaw = self._planner.get_coordinates(t + self.dt_array)
        refs = {
            "pos": jnp.array(des_pos, device=self.sim.device),
            "vel": jnp.array(des_vel, device=self.sim.device),
            "acc": jnp.array(des_acc, device=self.sim.device),
            "yaw": jnp.array(des_yaw, device=self.sim.device),
        }

        # 1. Update Step
        # Now returns a batch of means and sigmas

        # changedPractical
        self._rng_key, subkey = jax.random.split(self._rng_key)

        new_means, new_sigmas, best_mode_idx, costs_grouped, positions_grouped = (
            self._mppi_core_update(
                subkey,
                obs_device,
                refs,
                self.mean_controls,  # Shape: (K, Horizon, U)
                self.noise_sigmas,  # Shape: (K, Horizon, U)
            )
        )

        # 2. Extract Action from the "Winner" Mode
        # We only execute the action from the best cluster
        best_action = new_means[best_mode_idx, 0]

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
        action = np.asarray(best_action)  # back to CPU
        self.thrust += (
            self.drone_params["thrust_dyn_coef"] * (action[3] - self.thrust) * self.ctrl_dt
        )

        if self._log_buf is not None:
            _pos = obs["pos"]
            _des_p = np.asarray(des_pos[0])
            _des_v = np.asarray(des_vel[0])
            _pos_err = np.linalg.norm(_pos - _des_p)
            _z_err = abs(_pos[2] - _des_p[2])
            _vel_err = np.linalg.norm(obs["vel"] - _des_v)
            _obs_arr = np.asarray(self.obstacles)
            _min_obs = float(np.min(np.linalg.norm(_pos[:2] - _obs_arr[:, :2], axis=-1))) if len(_obs_arr) else np.inf
            self._log_buf["t"].append(self._t)
            self._log_buf["pos"].append(_pos.copy())
            self._log_buf["vel"].append(obs["vel"].copy())
            self._log_buf["action"].append(action.copy())
            self._log_buf["des_pos"].append(_des_p)
            self._log_buf["des_vel"].append(_des_v)
            self._log_buf["min_cost"].append(float(np.min(np.asarray(self.costs))))
            self._log_buf["target_gate"].append(int(obs.get("target_gate", -1)))
            self._log_buf["cost_pos"].append(_pos_err**2 * 40.0)
            self._log_buf["cost_z"].append(_z_err * 80.0)
            self._log_buf["cost_vel"].append(_vel_err**2 * 1.0)
            self._log_buf["min_obs_dist"].append(_min_obs)

        return action

    def step_callback(
        self,
        action: NDArray[np.floating] | None = None,
        obs: dict[str, NDArray[np.floating]] | None = None,
        reward: float | None = None,
        terminated: bool | None = None,
        truncated: bool | None = None,
        info: dict | None = None,
    ):
        """Increment the tick counter."""
        # changedPractical
        self.obstacles = jnp.array(obs["obstacles_pos"], device=self.sim.device)
        return self._finished

    def episode_callback(self):
        """Save logged data to disk if LOG_DRONE_DATA is set."""
        if self._log_buf is None or not self._log_buf["t"]:
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
        key,
        obs: dict[str, jnp.ndarray],
        refs: dict[str, jnp.ndarray],
        current_means,
        noise_sigmas,
    ):
        """Internal MPPI update function that performs sampling, rollouts, cost evaluation, and mean updates.

        current_means: (K, Horizon, 4) - K different control strategies
        noise_sigmas: (K, Horizon, 4) - Adaptive variance for each strategy
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

        def sample_per_mean(k_key, k_mean, k_sigma, k_lb, k_ub):
            # Generate noise for ONE mean
            # Shape: (M, Horizon, 4)
            k_noise = self.get_truncated_normal_jax(
                k_key,
                mean=0,
                sd=k_sigma,
                x_min=k_lb[None, ...],
                x_max=k_ub[None, ...],
                shape=(self.M, self.N, 4),
            )
            return k_noise

        # Vectorize sampling over K means
        # noise shape: (K, M, Horizon, 4)
        noise = vmap(sample_per_mean)(keys, current_means, noise_sigmas, lower_bound, upper_bound)

        # --- 3. NOISE SMOOTHING ---
        # Apply beta smoothing. We can flatten K and M dimensions for this operation
        # or vmap twice. Flattening is easier.
        noise_flat = noise.reshape(-1, self.N, 4)  # (N, H, 4)

        def smooth_scan(carry, x):
            new_val = self.beta * carry + (1 - self.beta) * x
            return new_val, new_val

        # Initialize smooth scan with zeros
        _, noise_flat = jax.lax.scan(
            lambda c, x: vmap(smooth_scan)(c, x),
            jnp.zeros((noise_flat.shape[0], 4)),
            # jnp.ones((noise_flat.shape[0], 4)) * noise_flat[:, 0, :],
            noise_flat.transpose(1, 0, 2),
        )
        noise_flat = noise_flat.transpose(1, 0, 2)

        # Reshape back to grouped format: (K, M, H, 4)
        noise = noise_flat.reshape(self.K, self.M, self.N, 4)

        # --- 4. ROLLOUTS ---
        # Calculate candidate controls: Mean[k] + Noise[k, m]
        # Broadcasting: (K, 1, H, 4) + (K, M, H, 4)
        candidate_controls = current_means[:, None, ...] + noise

        # Flatten for the physics engine (Physics doesn't care about K clusters)
        # 1. Transpose from (K, M, N, 4) -> (N, K, M, 4)
        # 2. Reshape to (N, K*M, 4)
        controls_flat = candidate_controls.transpose(2, 0, 1, 3).reshape(self.N, -1, 4)

        costs_flat, positions_flat = self.rollout_sim(obs, (controls_flat, refs, self.obstacles))

        # Reshape costs back to groups: (K, M)
        costs_grouped = costs_flat.reshape(self.K, self.M)
        positions_grouped = positions_flat.transpose(1, 0, 2, 3).reshape(self.K, self.M, self.N, 3)

        # --- 5. PER-MODE UPDATE (The Core Logic) ---
        # We define a function that updates ONE mean, then vmap it over K

        def update_single_mode(k_mean, k_noise, k_costs, k_sigma):
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

        return updated_means, updated_sigmas, best_mode_idx, costs_grouped, positions_grouped

    def get_truncated_normal_jax(self, key, mean, sd, x_min, x_max, shape=(1,)):
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
    def shift_and_interpolate(self, controls, dt_plan, dt_ctrl):
        """Shifts the control trajectory forward by dt_ctrl using linear interpolation.
        x
                Args:
                    controls: Array of shape (Horizon, Control_Dim)
                    dt_plan: Time duration of one step in the planning horizon
                    dt_ctrl: Time duration of one control cycle (latency/execution time)

                Returns:
                    Shifted controls of shape (Horizon, Control_Dim)
        """
        horizon_len = controls.shape[0]

        # Create the time grids
        # old_times: [0, dt_plan, 2*dt_plan, ...]
        old_times = jnp.arange(horizon_len) * dt_plan

        # target_times: [dt_ctrl, dt_ctrl + dt_plan, ...]
        target_times = old_times + dt_ctrl

        # Define the 1D interpolation logic
        def interp_fn(y):
            # right=y[-1] implements Zero-Order Hold for the end of the horizon
            return jnp.interp(target_times, old_times, y, left=y[0], right=y[-1])

        # Vectorize across the Control_Dim (axis 1)
        # in_axes=1 means we iterate over columns (controls), keeping time (rows) intact per op
        return jax.vmap(interp_fn, in_axes=1, out_axes=1)(controls)

    @partial(jax.jit, static_argnames=["self"])
    def compute_cost(
        self, data: SimData, reference: dict[str, jnp.ndarray], obstacles: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute the cost for a given state."""
        pos = data.states.pos[:, 0, :]  # Shape: (n_rollouts, 3)
        vel = data.states.vel[:, 0, :]  # Shape: (n_rollouts, 3)
        ang_vel = data.states.ang_vel[:, 0, :]  # Shape: (n_rollouts, 3)
        ang_acc = data.states_deriv.ang_vel[:, 0, :]  # Shape: (n_rollouts, 3)
        des_pos = reference["pos"][..., None, :]  # Shape: (1, 3)
        des_vel = reference["vel"][..., None, :]  # Shape: (1, 3)
        des_yaw = reference["yaw"][..., None]  # Shape: (1,)
        cmd = data.controls.attitude.staged_cmd[:, 0, :]  # Shape: (n_rollouts, 4)

        ## 1. State Cost (Tracking)
        pos_error = jnp.linalg.norm(pos - des_pos, axis=-1)
        # pos_cost = pos_error**2 * 10.0
        # changedPractical
        pos_cost = pos_error**2 * 40.0
        z_cost = jnp.abs(pos[..., 2] - des_pos[..., 2]) * 80.0  # Extra penalty for altitude error
        vel_error = jnp.linalg.norm(vel - des_vel, axis=-1)
        vel_cost = vel_error**2 * 1.0
        ang_vel_error = jnp.linalg.norm(ang_vel, axis=-1)
        ang_vel_cost = ang_vel_error**2 * 0.0
        ang_acc_error = jnp.linalg.norm(ang_acc, axis=-1)
        ang_acc_cost = ang_acc_error**2 * 0.0
        state_cost = pos_cost + vel_cost + ang_vel_cost + ang_acc_cost + z_cost

        ## 2. Control Cost (Efficiency + Stability)
        # Penalize high tilt (roll/pitch)
        tilt_cost = jnp.linalg.norm(cmd[:, :2], axis=-1) ** 2 * 5.0
        # Penalize thrust deviations from gravity
        thrust_cost = (cmd[:, 3] - 0.43) ** 2 * 0.0
        # Penalize yaw deviations
        yaw_cost = (cmd[:, 2] - des_yaw) ** 2 * 0.0
        input_cost = tilt_cost + thrust_cost + yaw_cost

        ## 3. Obstacle Cost (Safety)
        obs_diff = jnp.linalg.norm(pos[..., None, :2] - obstacles[None, :, :2], axis=-1)
        obstacle_hits = jnp.where(
            obs_diff
            < self.initial_info["experiment"]["env"]["obstacle_radius"]
            + self.initial_info["experiment"]["env"]["drone_radius"],
            1,
            0,
        )
        obstacle_cost = 1000.0 * jnp.sum(obstacle_hits, axis=-1)

        return state_cost + input_cost + obstacle_cost

    @partial(jax.jit, static_argnames=["self"])
    def apply_input(
        self, data: SimData, info: tuple[jnp.ndarray, dict[str, jnp.ndarray]]
    ) -> tuple[SimData, jnp.ndarray]:
        """Rolls out the sim.

        Args:
            state: Initial state of the sim in the form a dictionary with the last observation.
            input: Input to apply. Shape (N, 4), where N is the number of drones, and 4 is the control dimension (roll, pitch, yaw, thrust).
        """
        cmd, ref, obstacles = info
        # Step sim with input
        data = data.replace(
            controls=data.controls.replace(
                attitude=data.controls.attitude.replace(staged_cmd=cmd[:, None, :])
            )
        )
        next_data = self.step_fn(data, self.sim.freq // self.sim.control_freq)
        cost = self.compute_cost(next_data, ref, obstacles)
        return next_data, (cost, next_data.states.pos)

    @partial(jax.jit, static_argnames=["self"])
    def rollout_sim(
        self, obs: dict, infos: tuple[jnp.ndarray, dict[str, jnp.ndarray], jnp.ndarray]
    ) -> jnp.ndarray:
        """Rolls out the sim for scan."""
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
        controls_flat, refs, obstacles = infos
        # scan iterates over axis 0 (N time steps); tile obstacles so each step gets a slice
        obstacles_tiled = jnp.broadcast_to(obstacles[None], (self.N,) + obstacles.shape)
        _, (costs, positions) = scan(self.apply_input, data, (controls_flat, refs, obstacles_tiled))
        return jnp.sum(costs, axis=0), positions

    # changedPractical

    def _draw_reference(self, sim: Sim):
        """Draw the reference spline, current setpoint, and waypoints."""
        setpoint = self._planner.evaluate_pos(self._t).reshape(1, -1)
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
        n_viz = 5

        for k in range(self.K):
            rgb = _cluster_colors[k % len(_cluster_colors)]
            sorted_idx = np.argsort(costs[k])[:n_viz]
            for rank, m in enumerate(sorted_idx):
                alpha = 1.0 - (rank / n_viz) * 0.75
                draw_line(sim, positions[k, m], rgba=(*rgb, alpha))

        best_m = int(np.argmin(costs[best_k]))
        draw_line(sim, positions[best_k, best_m], rgba=(1.0, 1.0, 1.0, 1.0))

    # own renderings
    def render_callback(self, sim: Sim):
        """Visualize the desired trajectory and the current setpoint."""
        self._draw_reference(sim)
        self._draw_mppi_rollouts(sim)
