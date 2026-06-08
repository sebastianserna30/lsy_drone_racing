"""Level 0 controller using dynamic trajectory generation through gate waypoints.

Builds a cubic spline at init time from the known gate positions and orientations.
For each gate, entry and exit waypoints are computed along the gate normal so the
drone approaches and exits cleanly perpendicular to the gate plane.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray


class Level0Controller(Controller):
    """Trajectory-following controller for level 0 (fixed track, perfect knowledge)."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Build the waypoint list and fit a spline through the gates.

        Args:
            obs: Initial observation. Keys used: `pos`, `gates_pos`, `gates_quat`.
            info: Initial environment info (unused at level 0).
            config: Race configuration. Used to read the environment frequency.
        """
        super().__init__(obs, info, config)
        self._freq = config.env.freq
        self._tick = 0
        self._t_total = 8.0  # seconds — tune this later
        self._finished = False

        gate_in_offset = 0.3  # metres along gate normal for entry/exit points — TODO: tune
        gate_out_offset = 0.5  # metres along gate normal for entry/exit points — TODO: tune

        # --- Build waypoints ---
        waypoints = [obs["pos"].copy()]  # start at current drone position

        takeoff = obs["pos"].copy()
        takeoff[2] = 0.5  # lift to 0.5m before going anywhere
        waypoints.append(takeoff)

        gates_pos = obs["gates_pos"]  # shape (n_gates, 3)
        gates_quat = obs["gates_quat"]  # shape (n_gates, 4), format [x, y, z, w]

        for i, (pos, quat) in enumerate(zip(gates_pos, gates_quat)):
            normal = Rotation.from_quat(quat).apply([1.0, 0.0, 0.0])
            entry = pos - gate_in_offset * normal
            exit_ = pos + gate_out_offset * normal
            waypoints.append(entry)
            waypoints.append(pos)
            waypoints.append(exit_)

        waypoints = np.array(waypoints)  # shape (1 + n_gates*3, 3)
        self._waypoints = waypoints

        print("Waypoints:")
        for i, wp in enumerate(waypoints):
            print(f"  {i}: {wp}")

        # Assign timestamps proportional to cumulative distance between waypoints
        dists = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
        t = np.concatenate([[0.0], np.cumsum(dists / dists.sum() * self._t_total)])

        self._spline = CubicSpline(t, waypoints)
        self._t_total_actual = t[-1]

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Return the next desired state along the pre-planned spline.

        Args:
            obs: Current observation (unused — open loop at level 0).
            info: Unused.

        Returns:
            State command [x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate].
        """
        t = min(self._tick / self._freq, self._t_total)
        if t >= self._t_total:
            self._finished = True

        des_pos = self._spline(t)
        return np.concatenate([des_pos, np.zeros(10)], dtype=np.float32)

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Increment tick counter.

        Returns:
            True if the controller has finished, False otherwise.
        """
        self._tick += 1

        return obs["target_gate"] == -1

    def episode_callback(self):
        """Reset internal state between episodes."""
        self._tick = 0
        self._finished = False

    def render_callback(self, sim: Sim):
        """Draw the planned trajectory and current setpoint."""
        from crazyflow.sim.visualize import draw_line, draw_points

        setpoint = self._spline(self._tick / self._freq).reshape(1, -1)
        draw_points(sim, setpoint, rgba=(1.0, 0.0, 0.0, 1.0), size=0.02)
        trajectory = self._spline(np.linspace(0, self._t_total, 100))
        draw_line(sim, trajectory, rgba=(0.0, 1.0, 0.0, 1.0))
        draw_points(sim, self._waypoints, rgba=(0.0, 0.0, 1.0, 1.0), size=0.05)
