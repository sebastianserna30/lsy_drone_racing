"""This module wraps the AttitudeController to handle batched multi-agent environments.

In multi-agent simulations, observations are batched across all drones.
The rank index is used to select the state of the current drone.
"""

from __future__ import annotations  # Python 3.10 type hints

from typing import TYPE_CHECKING

import numpy as np
from ml_collections import ConfigDict

from lsy_drone_racing.control.trajectory_mppi import (
    AttitudeMPPIController as SingleAttitudeMPPIController,
)

if TYPE_CHECKING:
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
            return super().compute_control({k: v[self.rank] for k, v in obs.items()}, info)
        
        return super().compute_control(obs, info)
    
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
        return super().step_callback(action,{k: v[self.rank] for k, v in obs.items()})

