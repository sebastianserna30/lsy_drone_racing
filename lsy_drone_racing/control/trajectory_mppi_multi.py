"""Multi-agent wrapper around the n_agents-generic AttitudeMPPIController.

In multi-agent simulations, observations are batched across all drones. This thin wrapper reorders
the batched observation so the ego drone (this controller's rank) is agent 0 and the opponent is
agent 1, then delegates everything to the base controller, which models BOTH drones in one shared
MPPI rollout batch (n_agents=2). The opponent is no longer a separate nested MPPI: it is drone 1 in
the base's rollout, with its own spline, and the symmetric collision term lives in the base cost.

The base also supports kinematic opponent models, where the opponent is not simulated at all and
the ego is scored against a predicted trajectory instead. This wrapper is unaffected either way:
it only reorders the batched obs.
"""

from __future__ import annotations  # Python 3.10 type hints

from typing import TYPE_CHECKING

import numpy as np
from ml_collections import ConfigDict

from lsy_drone_racing.control.trajectory_mppi import (
    AttitudeMPPIController as SingleAttitudeMPPIController,
)

if TYPE_CHECKING:
    from crazyflow.sim import Sim
    from numpy.typing import NDArray

# Per-drone state keys carry an agent axis into the base; every other key (gates, obstacles,
# target_gate, ...) is an environment view we take from the ego's slice.
_STATE_KEYS = ("pos", "quat", "vel", "ang_vel")


class AttitudeMPPIController(SingleAttitudeMPPIController):
    """Multi-agent MPPI: reorder the batched obs (ego first) and hand it to the n_agents base."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the multi-agent MPPI controller.

        Args:
            obs: The initial batched observation of the environment's state (drone axis first).
            info: Additional environment information from the reset (contains ``rank``).
            config: The configuration of the environment.
        """
        self.rank = info["rank"]
        self.opponent = 1 - self.rank  # two-drone race

        # Pick this rank's controller sub-config, exactly as before.
        controller_cfg = config["controller"][self.rank]
        config_dict = config.to_dict()
        config_dict["controller"] = controller_cfg
        config = ConfigDict(config_dict)

        # Reorder the batched obs so the ego is agent 0, then let the base build a 2-agent sim.
        # During the base's warmup (which calls self.compute_control), the obs is already reordered,
        # so guard against a second reorder with a flag.
        self._warming_up = True
        super().__init__(self._reorder(obs), info, config)
        self._warming_up = False

    def _reorder(self, obs: dict[str, NDArray[np.floating]]) -> dict[str, NDArray[np.floating]]:
        """Reorder a batched observation so the ego (self.rank) is agent 0, opponent agent 1.

        Per-drone state keys keep an agent axis (ego first); all other keys become the ego's view.
        """
        order = [self.rank, self.opponent]
        out = {}
        for k, v in obs.items():
            v = np.asarray(v)
            out[k] = v[order] if k in _STATE_KEYS else v[self.rank]
        return out

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Reorder the batched obs (ego first) and delegate to the base controller.

        Returns the ego (agent 0) collective thrust and roll/pitch/yaw [r, p, y, t].
        """
        obs = obs if self._warming_up else self._reorder(obs)
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
        """Advance the base bookkeeping using the ego's environment view."""
        return super().step_callback(action, {k: v[self.rank] for k, v in obs.items()})

    def render_callback(self, sim: Sim):
        """Visualize the ego reference/rollouts (base handles ego + per-agent draws)."""
        super().render_callback(sim)
