"""This module wraps the AttitudeController to handle batched multi-agent environments.

In multi-agent simulations, observations are batched across all drones.
The rank index is used to select the state of the current drone.
"""

from __future__ import annotations  # Python 3.10 type hints

from functools import partial
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
            info["opponent_pos"] = obs["pos"][self.opponent]
            info["opponent_vel"] = obs["vel"][self.opponent]
            return super().compute_control({k: v[self.rank] for k, v in obs.items()}, info)

        # This backup is needed for the warmup of the controller
        return super().compute_control(obs, info)

    @partial(jax.jit, static_argnames=["self"])
    def compute_cost(
        self,
        data: SimData,
        reference: dict[str, jnp.ndarray],
        obstacles: jnp.ndarray,
        gate_frame_obstacles: jnp.ndarray,
    ) -> dict[str, jnp.ndarray]:
        """Add the opponent-drone collision term to the base per-term cost dict."""
        ## 6. Collision Cost (Safety)
        safe_dist = (
            self.initial_info["experiment"]["env"]["drone_radius"] * 2.5
        )  # 2 would be theory and added buffer

        pos = data.states.pos[:, 0, :]  # Shape: (n_rollouts, 3)
        opp_pos = reference["opp_pos"][..., None, :]  # Shape: (1, 3)
        dist_drones = jnp.linalg.norm(pos - opp_pos, axis=-1)  # (n_rollouts,)

        binary_cost = False

        if binary_cost:
            # per-rollout (n_rollouts,): single opponent, so no sum over an axis.
            # (the old jnp.sum(..., axis=-1) collapsed this to a scalar that added the same
            #  constant to every rollout — a no-op for elite/softmax selection)
            opponent_drone_hits = jnp.where(dist_drones < safe_dist, 1, 0)
            coll_cost = self.w_opp_drone * opponent_drone_hits
        else:
            # smooth collsion cost using exponent
            coll_cost = self.w_opp_drone_exp * jnp.exp(-((dist_drones / safe_dist) ** 2))

        use_collision_cost = True
        if use_collision_cost:
            collosion_cost = coll_cost
        else:
            collosion_cost = jnp.zeros(pos.shape[0])

        terms = super().compute_cost(data, reference, obstacles, gate_frame_obstacles)
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
        super().render_callback(sim)
        draw_line(sim, self.opp_traj, rgba=(0.0, 0.0, 1.0, 1.0))
