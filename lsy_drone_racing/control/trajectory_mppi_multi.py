"""This module wraps the AttitudeController to handle batched multi-agent environments.

In multi-agent simulations, observations are batched across all drones.
The rank index is used to select the state of the current drone.
"""

from __future__ import annotations  # Python 3.10 type hints

import copy
from functools import partial
from types import SimpleNamespace
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from crazyflow.sim import Sim
from crazyflow.sim.visualize import draw_line, draw_points
from ml_collections import ConfigDict

from lsy_drone_racing.control.trajectory_mppi import (
    AttitudeMPPIController as SingleAttitudeMPPIController,
)

if TYPE_CHECKING:
    from crazyflow.sim.data import SimData
    from numpy.typing import NDArray


class AttitudeMPPIController(SingleAttitudeMPPIController):
    """Example of a controller using the collective thrust and attitude interface."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the attitude controller.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: Additional environment information from the reset.
            config: The configuration of the environment.
        """
        self.rank = info["rank"]
        if self.rank == 0:
            self.opponent = 1
        else:
            self.opponent = 0

        controller_cfg = config["controller"][self.rank]
        # Convert to a plain dict first
        config_dict = config.to_dict()

        # Replace controller
        config_dict["controller"] = controller_cfg

        # Create a new ConfigDict
        config = ConfigDict(config_dict)
        
        #for save deployment in the lab change to False
        self.opp_mppi = config["controller"]["mppi"]["opponent_mppi"]
        
        if self.opp_mppi:
            config_opp = copy.deepcopy(config)
            K_orig = config_opp.controller.mppi.K
            new_n_samples = int(config_opp.controller.mppi.n_samples/K_orig)

            #only use one mean and less samples for opponent
            config_opp.controller.mppi.n_samples = new_n_samples
            config_opp.controller.mppi.K = 1

            self._opponent = SingleAttitudeMPPIController(
                {k: v[self.opponent] for k, v in obs.items()}, info, config_opp
            )
        else:
            self._opponent = SimpleNamespace()


        # changedPractical: opponent-interaction tuning (anisotropic keep-out + behind-aware
        # contour relaxation). Set BEFORE super().__init__ — the base warms up the controller
        # (calls compute_control -> compute_cost) inside __init__, so these must exist first.
        cost_cfg = config["controller"]["mppi"]["cost"]
        # Anisotropic (elliptical) opponent keep-out semi-axes (m): axial > lateral makes sitting
        # fore/aft of the opponent expensive and being beside it cheap -> pass instead of brake.
        self.use_anisotropic_opp = bool(cost_cfg["use_anisotropic_opp"])
        self.opp_axial = float(cost_cfg["opp_axial"])       # default value 0.25
        self.opp_lateral = float(cost_cfg["opp_lateral"])    # default value 0.10 

        self.binary_cost_opp = bool(cost_cfg["binary_cost_opp"])
        

        self.render_traj = config["controller"]["mppi"].get("render_traj",True)

        super().__init__({k: v[self.rank] for k, v in obs.items()}, info, config)

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired collective thrust and roll/pitch/yaw of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The orientation as roll, pitch, yaw angles, and the collective thrust
            [r_des, p_des, y_des, t_des] as a numpy array.
        """
        first_value = next(iter(obs.values()))

        if first_value.ndim == 2:
            # This is the normal mode for multilevel where other drone is pressent in observations
            if self.opp_mppi:
                self._opponent.compute_control({k: v[self.opponent] for k, v in obs.items()}, info)
            else:
                opp_pos = obs["pos"][self.opponent]
                opp_vel = obs["vel"][self.opponent]
                self._opponent.best_traj = np.array(opp_pos[None, :] + (opp_vel[None, :] * self.dt_array[:, None]))

            info["opponent_traj"] = self._opponent.best_traj
            return super().compute_control({k: v[self.rank] for k, v in obs.items()}, info)

        # This backup is needed for the warmup of the controller
        if self.opp_mppi:
            self._opponent.compute_control(obs, info)
        else:
            self._opponent.best_traj = np.zeros((self.N, 3), dtype=np.float32)

        info["opponent_traj"] = self._opponent.best_traj
        return super().compute_control(obs, info)

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
        """Add the opponent-drone collision term to the base per-term cost dict.

        changedPractical: signature follows the MPCC base controller (theta, v_theta added);
        this override only adds the opponent-collision term and forwards the rest.
        """
        ## 6. Collision Cost (Safety)
        safe_dist = (
            self.initial_info["experiment"]["env"]["drone_radius"] * 2.5
        )  # 2 would be theory and added buffer

        pos = data.states.pos[:, 0, :]  # Shape: (n_rollouts, 3)
        opp_pos = reference["opp_pos"][..., None, :]  # Shape: (1, 3)
        opp_vel = reference["opp_vel"]  # (3,) opponent heading this horizon step
        delta = pos - opp_pos  # (n_rollouts, 3)
        dist_drones = jnp.linalg.norm(pos - opp_pos, axis=-1)  # (n_rollouts,)

        if self.binary_cost_opp:
            # per-rollout (n_rollouts,): single opponent, so no sum over an axis.
            # (the old jnp.sum(..., axis=-1) collapsed this to a scalar that added the same
            #  constant to every rollout — a no-op for elite/softmax selection)
            opponent_drone_hits = jnp.where(dist_drones < safe_dist, 1, 0)
            coll_cost = self.w_opp_drone * opponent_drone_hits
        elif self.use_anisotropic_opp:
            # anisotropic (elliptical) keep-out elongated ALONG the
            # opponent's heading. Split delta into along-heading and perpendicular parts, scale
            # by semi-axes (axial > lateral). Fore/aft of the opponent -> small scaled dist ->
            # high cost; beside it -> large scaled dist -> low cost. So the low-cost escape is a
            # sideways overtake, not a brake. Falls back to isotropic when opponent ~stationary.
            opp_speed = jnp.linalg.norm(opp_vel)
            heading = opp_vel / (opp_speed + 1e-6)
            d_along = delta @ heading  # signed
            d_perp = jnp.linalg.norm(delta - d_along[:, None] * heading, axis=-1)
            d_aniso = jnp.sqrt(
                (d_along / self.opp_axial) ** 2 + (d_perp / self.opp_lateral) ** 2
            )
            d_scaled = jnp.where(opp_speed > 0.2, d_aniso, dist_drones / safe_dist)
            coll_cost = self.w_opp_drone_exp * jnp.exp(-(d_scaled**2))
        else:
            # smooth collsion cost using exponent (isotropic circle)
            coll_cost = self.w_opp_drone_exp * jnp.exp(-((dist_drones / safe_dist) ** 2))

        use_collision_cost = True
        if use_collision_cost:
            collosion_cost = coll_cost
        else:
            collosion_cost = jnp.zeros(pos.shape[0])

        terms = super().compute_cost(
            data, theta, v_theta, reference, obstacles, gate_frame_obstacles
        )
        terms["opp_drone"] = collosion_cost
        return terms

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
        return super().step_callback(action, {k: v[self.rank] for k, v in obs.items()})

    def render_callback(self, sim: Sim):
        """Visualize the desired trajectory and the current setpoint."""
        if self.render_traj:
            super().render_callback(sim)
            draw_line(sim, self._opponent.best_traj, rgba=(0.0, 0.0, 1.0, 1.0))
