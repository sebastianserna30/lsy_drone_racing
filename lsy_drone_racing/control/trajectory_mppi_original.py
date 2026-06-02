"""TODO."""

from __future__ import annotations  # Python 3.10 type hints

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from crazyflow.control import Control
from crazyflow.sim import Physics, Sim
from crazyflow.sim.data import SimData
from crazyflow_experiments.sim2real.control.trajectory_generator import (
    TrajectoryGenerator3DPeriodicMotion,
)
from drone_models.core import load_params
from drone_models.so_rpy_rotor_drag import dynamics
from jax import random, vmap
from jax.lax import scan

if TYPE_CHECKING:
    from numpy.typing import NDArray


class AttitudeMPPIController:
    def __init__(self, initial_obs: dict[str, NDArray[np.floating]], initial_info: dict):
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

        self.sim = Sim(
            n_worlds=self.n_samples,
            n_drones=1,
            attitude_freq=self.f,
            freq=self.f,
            physics=Physics.so_rpy_rotor_drag,
            control=Control.attitude,
            drone_model="cf21B_500",
            device="gpu",  # TODO get from info
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

        self.mean_controls = jnp.zeros((self.K, self.N, 4), device=self.sim.device)

        # Shape: (Num_Obstacles, 3)
        self.obstacles = jnp.array(initial_info["obstacles"], device=self.sim.device)

        self.low_level_ctrl_freq = initial_info["low_level_ctrl_freq"]
        self.drone_params = load_params("first_principles", initial_info["drone_model"])
        self.drone_mass = self.drone_params["mass"]
        self.act_low = -jnp.ones(4, device=self.sim.device) * jnp.pi / 2
        self.act_low = self.act_low.at[3].set(self.drone_params["thrust_min"] * 4)
        self.act_high = jnp.ones(4, device=self.sim.device) * jnp.pi / 2
        self.act_high = self.act_high.at[3].set(self.drone_params["thrust_max"] * 4)
        self.thrust = np.zeros(4)

        self._finished = False
        self._t_start = initial_obs["t"]
        self._t_end = initial_info["planner_cycles"] * initial_info["planner_cycle_time"]

        ### Generate trajectory
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

        key = jax.random.PRNGKey(0)
        key, subkey = random.split(key)
        info_short = {"rng_key": subkey, "obstacles": jnp.array(initial_info["obstacles"])}
        for i in range(10):
            a = self.compute_control(initial_obs, info_short)  # Warm up the controller
            jax.block_until_ready(a)

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
        obs["rotor_vel"] = self.thrust
        obs_device = {k: jax.device_put(v, self.sim.device) for k, v in obs.items()}
        t = obs["t"] - self._t_start
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
        new_means, new_sigmas, best_mode_idx, costs_grouped, positions_grouped = (
            self._mppi_core_update(
                info["rng_key"],
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
        return self._finished

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

        costs_flat, positions_flat = self.rollout_sim(obs, (controls_flat, refs))

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
    def compute_cost(self, data: SimData, reference: dict[str, jnp.ndarray]) -> jnp.ndarray:
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
        pos_cost = pos_error**2 * 10.0
        z_cost = jnp.abs(pos[..., 2] - des_pos[..., 2]) * 20.0  # Extra penalty for altitude error
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
        thrust_cost = (cmd[:, 3] - 0.43) ** 2 * 5.0
        # Penalize yaw deviations
        yaw_cost = (cmd[:, 2] - des_yaw) ** 2 * 100.0
        input_cost = tilt_cost + thrust_cost + yaw_cost

        ## 3. Obstacle Cost (Safety)
        obs_diff = jnp.linalg.norm(pos[..., None, :2] - self.obstacles[None, :, :2], axis=-1)
        obstacle_hits = jnp.where(
            obs_diff < self.initial_info["obstacle_radius"] + self.initial_info["drone_radius"],
            1,
            0,
        )
        obstacle_cost = 1000.0 * jnp.sum(obstacle_hits, axis=-1)  # Sum over all obstacles

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
        cmd, ref = info
        # Step sim with input
        data = data.replace(
            controls=data.controls.replace(
                attitude=data.controls.attitude.replace(staged_cmd=cmd[:, None, :])
            )
        )
        next_data = self.step_fn(data, self.sim.freq // self.sim.control_freq)
        cost = self.compute_cost(next_data, ref)
        return next_data, (cost, next_data.states.pos)

    @partial(jax.jit, static_argnames=["self"])
    def rollout_sim(
        self, obs: dict, infos: tuple[jnp.ndarray, dict[str, jnp.ndarray]]
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
        _, (costs, positions) = scan(self.apply_input, data, infos)
        return jnp.sum(costs, axis=0), positions


# The following part is just for visualization and should be removed in the future!


def create_jax_model(p: dict):
    """Creates an acados model from a symbolic drone_model."""
    dyn_fixed = partial(
        dynamics,  # The original dynamics function definition
        mass=p["mass"],
        gravity_vec=p["gravity_vec"],
        J=p["J"],
        J_inv=p["J_inv"],
        thrust_time_coef=p["thrust_time_coef"],
        acc_coef=p["acc_coef"],
        cmd_f_coef=p["cmd_f_coef"],
        rpy_coef=p["rpy_coef"],
        rpy_rates_coef=p["rpy_rates_coef"],
        cmd_rpy_coef=p["cmd_rpy_coef"],
        drag_matrix=p["drag_matrix"],
    )

    def dynamics_adapter(x, u, dt):
        """Adapter to convert state vector x and control u into individual arguments for the dynamics function.

        Assumed x layout (dim 13):
        0:3   -> pos
        3:7   -> quat
        7:10  -> vel
        10:13 -> ang_vel
        """
        # 1. Unpack x into constituent components
        pos = x[0:3]
        quat = x[3:7]
        vel = x[7:10]
        ang_vel = x[10:13]

        # 2. Call the dynamics function (which will be partial'd below)
        # We return the derivatives as a tuple, or you can repack them into x_dot here

        pos_dot, quat_dot, vel_dot, ang_vel_dot, rotor_vel_dot = dyn_fixed(
            pos, quat, vel, ang_vel, u
        )
        # x_dot = jnp.concatenate((pos_dot, quat_dot, vel_dot, ang_vel_dot))
        # x_next = x + x_dot * dt
        ## jax.debug.print("xdot: {}", x_dot)
        # quat_next = x_next[3:7]
        # quat_next = quat_next / jnp.linalg.norm(quat_next)
        # x_next = x_next.at[3:7].set(quat_next)
        # 1. Update Standard States (Position, Velocity, Omega)
        # 1. Integrate Position, Velocity, Angular Velocity (Standard Euler)
        pos_next = pos + pos_dot * dt
        vel_next = vel + vel_dot * dt
        ang_vel_next = ang_vel + ang_vel_dot * dt

        # 2. Integrate Quaternion (Exponential Map for Scalar Last)
        omega_norm = jnp.linalg.norm(ang_vel)

        # Calculate half-angle
        half_theta = 0.5 * omega_norm * dt

        # Create Delta Quaternion: dq = [sin(t/2)*u, cos(t/2)]
        # We handle the limit where omega -> 0 to avoid division by zero
        scale = jnp.where(
            omega_norm < 1e-6,
            0.5 * dt,  # First-order approximation for small angles
            jnp.sin(half_theta) / omega_norm,
        )

        # Construct dq in [x, y, z, w] order
        dq_x = ang_vel[0] * scale
        dq_y = ang_vel[1] * scale
        dq_z = ang_vel[2] * scale
        dq_w = jnp.cos(half_theta)

        # 3. Multiply: q_next = q_current * dq
        # Extract current components [x, y, z, w]
        qx, qy, qz, qw = quat

        # Quaternion multiplication formula (Scalar Last)
        # vector_part = s1*v2 + s2*v1 + cross(v1, v2)
        # scalar_part = s1*s2 - dot(v1, v2)
        q_next_x = qw * dq_x + qx * dq_w + qy * dq_z - qz * dq_y
        q_next_y = qw * dq_y - qx * dq_z + qy * dq_w + qz * dq_x
        q_next_z = qw * dq_z + qx * dq_y - qy * dq_x + qz * dq_w
        q_next_w = qw * dq_w - qx * dq_x - qy * dq_y - qz * dq_z

        q_next = jnp.array([q_next_x, q_next_y, q_next_z, q_next_w])

        # 4. Normalize (Crucial)
        q_next = q_next / jnp.linalg.norm(q_next)

        # 5. Reassemble the state
        # Assuming the order in x_next is [pos(3), quat(4), vel(3), ang_vel(3)]
        x_next = jnp.concatenate((pos_next, q_next, vel_next, ang_vel_next))
        # ang_vel = x_next[10:13]
        # ang_vel = ang_vel.clip(-1, 1)
        # x_next.at[10:13].set(ang_vel)
        # jax.debug.print("clipped ang_vel: {}", ang_vel)

        return x_next

    # Initialize the nonlinear model for NMPC formulation

    return dynamics_adapter


drone_model = "cf21B_500"
drone_params = load_params("so_rpy_rotor_drag", drone_model)
drone_radius = 0.086
obstacle_radius = 0.055
dynamics_step = create_jax_model(drone_params)


@jax.jit
def rollout_fn(state, ctrls, dt):
    def body(carry, u):
        n_st = dynamics_step(carry, u, dt)
        return n_st, n_st

    _, traj = jax.lax.scan(body, state, ctrls)
    return traj
