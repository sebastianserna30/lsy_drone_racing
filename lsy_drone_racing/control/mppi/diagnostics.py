"""Cost logging and MuJoCo visualization for the MPPI controller.

Neither concern affects flight: the logger only runs when ``LOG_DRONE_DATA`` is set, and the
draw calls only when the sim is rendering. Keeping them here keeps the controller readable.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from crazyflow.sim.visualize import draw_line, draw_points

if TYPE_CHECKING:
    from crazyflow.sim import Sim
    from numpy.typing import NDArray

# One colour per MPPI mode, cycled if there are more modes than colours.
CLUSTER_COLORS = [
    (1.0, 0.5, 0.0),  # orange
    (0.5, 0.0, 1.0),  # purple
    (0.0, 1.0, 1.0),  # cyan
    (1.0, 1.0, 0.0),  # yellow
    (1.0, 0.0, 0.5),  # pink
    (0.0, 0.5, 1.0),  # sky blue
]


class CostLogger:
    """Accumulates the per-mode cost breakdown of every control step, and writes it out.

    Enabled by the ``LOG_DRONE_DATA`` environment variable, which may name either a directory
    or a ``.npz`` file.
    """

    def __init__(self):
        """Create an inactive logger; call :meth:`enable` to start buffering."""
        self.buf: dict[str, list] | None = None

    @staticmethod
    def from_env() -> CostLogger:
        """Return a logger, enabled when LOG_DRONE_DATA is set."""
        logger = CostLogger()
        if os.getenv("LOG_DRONE_DATA"):
            logger.enable()
        return logger

    def enable(self) -> None:
        """Start buffering. Keys are created lazily on first append."""
        self.buf = {}

    @property
    def active(self) -> bool:
        """Whether anything is being recorded."""
        return self.buf is not None

    def log_step(
        self,
        t: float,
        obs: dict[str, NDArray[np.floating]],
        action: NDArray[np.floating],
        mode_term_costs: dict[str, NDArray[np.floating]],
        costs: NDArray[np.floating],
        best_mode_idx: int,
    ) -> None:
        """Record the full-horizon rollout cost breakdown of the K parallel modes.

        For each control step we record, per cost term, a length-K vector: the horizon-summed
        cost of each mode's best (lowest-total) sample — i.e. the cost of the trajectory that
        mode proposes. That is what the MPPI optimizer actually scores, so it is the right
        signal for cost engineering / weight tuning.

        Args:
            t: controller time (s).
            obs: this step's observation (the ego's view).
            action: the executed [roll, pitch, yaw, thrust].
            mode_term_costs: term name -> (K,) cost of each mode's best sample.
            costs: (K, M) total cost per sample.
            best_mode_idx: the winning mode.
        """
        if self.buf is None:
            return
        terms = {k: np.asarray(v) for k, v in mode_term_costs.items()}  # each (K,)
        costs_km = np.asarray(costs)  # (K, M) total per sample

        fields = {
            "t": t,
            "pos": np.asarray(obs["pos"]).copy(),
            "action": np.asarray(action).copy(),
            "target_gate": int(obs.get("target_gate", -1)),
            "best_mode_idx": int(best_mode_idx),
            # per-mode totals: best sample per mode (K,) and the global best
            "mode_cost_total": costs_km.min(axis=1),
            "min_cost": float(costs_km.min()),
        }
        # per-mode, per-term horizon cost: each logged as a (K,) vector
        for name, vec in terms.items():
            fields[f"cost_{name}"] = vec
        # total per mode summed from the logged terms (matches mode_cost_total)
        fields["cost_total"] = np.sum(list(terms.values()), axis=0)

        for key, val in fields.items():
            self.buf.setdefault(key, []).append(val)

    def save(self, **extra: NDArray[np.floating]) -> None:
        """Write the buffer to the path in LOG_DRONE_DATA, plus any static `extra` arrays."""
        if self.buf is None or not self.buf.get("t"):
            return
        log_dir = os.getenv("LOG_DRONE_DATA", ".")
        out_path = (
            Path(log_dir) / "mppi_data.npz" if not log_dir.endswith(".npz") else Path(log_dir)
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_path, **{k: np.array(v) for k, v in self.buf.items()}, **extra)
        print(f"[MPPI] Saved {len(self.buf['t'])} steps → {out_path}")


def draw_reference(
    sim: Sim, planner: object, ref_pos_np: NDArray, theta_grid_np: NDArray, theta: float
) -> None:
    """Draw the reference spline, its waypoints, and the current setpoint.

    The setpoint is the reference at the ego's committed progress theta (arc length), not at
    wall-clock time.
    """
    idx = min(int(np.searchsorted(theta_grid_np, theta)), len(ref_pos_np) - 1)
    draw_points(sim, ref_pos_np[idx].reshape(1, -1), rgba=(1.0, 0.0, 0.0, 1.0), size=0.02)
    draw_line(sim, planner.get_trajectory(100), rgba=(0.0, 1.0, 0.0, 1.0))
    draw_points(sim, planner.waypoints, rgba=(0.0, 0.0, 1.0, 1.0), size=0.03)


def draw_rollouts(
    sim: Sim,
    positions: NDArray,
    costs: NDArray,
    best_mode_idx: int,
    n_viz: int = 3,
) -> None:
    """Draw the cheapest `n_viz` ego rollouts per mode, with the overall best in white.

    Args:
        sim: the render target.
        positions: (K, M, N, 3) ego rollout positions.
        costs: (K, M) total cost per sample.
        best_mode_idx: the winning mode.
        n_viz: rollouts drawn per mode, faded by rank.
    """
    positions, costs = np.asarray(positions), np.asarray(costs)
    for k in range(positions.shape[0]):
        rgb = CLUSTER_COLORS[k % len(CLUSTER_COLORS)]
        for rank, m in enumerate(np.argsort(costs[k])[:n_viz]):
            draw_line(sim, positions[k, m], rgba=(*rgb, 1.0 - (rank / n_viz) * 0.75))

    best_m = int(np.argmin(costs[best_mode_idx]))
    draw_line(sim, positions[best_mode_idx, best_m], rgba=(1.0, 1.0, 1.0, 1.0))


def draw_opponent_predictions(sim: Sim, opp_pred: NDArray) -> None:
    """Draw the kinematic opponent trajectories the ego collision is scored against (red).

    Predictor modes have no opponent rollouts to draw.
    """
    for p in range(opp_pred.shape[1]):
        draw_line(sim, np.asarray(opp_pred[:, p]), rgba=(1.0, 0.0, 0.0, 1.0))


def draw_opponent_rollouts(sim: Sim, positions: NDArray, costs: NDArray) -> None:
    """Draw one line per opponent mode: its cheapest sample, in that mode's colour.

    Drawing every opponent rollout is far too many geoms, so only the best of each mode.
    """
    positions, costs = np.asarray(positions), np.asarray(costs)
    for k in range(positions.shape[0]):
        rgb = CLUSTER_COLORS[k % len(CLUSTER_COLORS)]
        draw_line(sim, positions[k, int(np.argmin(costs[k]))], rgba=(*rgb, 1.0))


def draw_obstacles(sim: Sim, obstacles: NDArray, gate_frame_obstacles: NDArray) -> None:
    """Mark the pillar and gate-frame keep-out centres."""
    draw_points(sim, obstacles, rgba=(1.0, 0.0, 0.0, 1.0), size=0.02)
    draw_points(sim, gate_frame_obstacles, rgba=(1.0, 0.0, 0.0, 1.0), size=0.02)
