"""Per-tick run logger for controller post-mortem debugging.

Records observations, desired setpoints, actions, and waypoints every tick. On episode end,
classifies the outcome (success / collision_gate / collision_obstacle / out_of_bounds / timeout)
and writes a ``.npz`` (full per-tick arrays) plus a ``.json`` summary into ``tries/``.

Usage from a controller::

    from lsy_drone_racing.control.run_logger import RunLogger

    # in __init__:
    self._logger = RunLogger(config)

    # in compute_control, after computing des_pos/des_vel/action:
    self._logger.log_tick(t, obs, des_pos, des_vel, action, self._waypoints)

    # in step_callback:
    self._logger.maybe_save(obs, terminated, truncated, self._finished)

    # in episode_callback:
    self._logger.reset()
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class RunLogger:
    """Lightweight per-tick logger that dumps a run to ``tries/`` on episode end."""

    def __init__(self, config: dict, save_dir: Path | None = None, rng_key: list | None = None):
        self._log: list[dict] = []
        self._saved = False
        self._run_start = datetime.now()
        self._save_dir = save_dir or Path(__file__).parents[2] / "tries"
        self._save_dir.mkdir(exist_ok=True)
        self._config_seed = int(getattr(config.env, "seed", -1))
        self._rng_key = rng_key  # pre-reset JAX key; set as config seed to reproduce this episode
        self._gates_nominal = np.array(
            [list(g["pos"]) for g in config.env.track.gates], dtype=np.float32
        )
        self._obstacles_nominal = np.array(
            [list(o["pos"]) for o in config.env.track.obstacles], dtype=np.float32
        )
        self._pos_limit_low = np.array([-2.5, -1.5, -0.05], dtype=np.float32)
        self._pos_limit_high = np.array([2.5, 1.5, 2.0], dtype=np.float32)

    def log_tick(
        self,
        t: float,
        obs: dict[str, NDArray[np.floating]],
        des_pos: NDArray[np.floating],
        des_vel: NDArray[np.floating],
        action: NDArray[np.floating],
        waypoints: NDArray[np.floating],
    ) -> None:
        """Record one tick. Copies small arrays so later mutation is safe."""
        self._log.append({
            "t": float(t),
            "pos": obs["pos"].copy(),
            "vel": obs["vel"].copy(),
            "quat": obs["quat"].copy(),
            "target_gate": int(obs["target_gate"]),
            "gates_pos": obs["gates_pos"].copy(),
            "gates_quat": obs["gates_quat"].copy(),
            "obstacles_pos": obs["obstacles_pos"].copy(),
            "des_pos": np.asarray(des_pos, dtype=np.float32).copy(),
            "des_vel": np.asarray(des_vel, dtype=np.float32).copy(),
            "action": np.asarray(action).copy(),
            "waypoints": np.asarray(waypoints).copy(),
        })

    def maybe_save(
        self,
        obs: dict[str, NDArray[np.floating]],
        terminated: bool,
        truncated: bool,
        finished: bool,
    ) -> None:
        """Save the run if the episode just ended (and we haven't saved yet)."""
        if (terminated or truncated or finished) and not self._saved:
            self._save_run(obs, terminated, truncated)
            self._saved = True

    def reset(self) -> None:
        """Clear state for a new episode."""
        self._log = []
        self._saved = False
        self._run_start = datetime.now()
        # Note: _config_seed is fixed per RunLogger instance (one per episode in sim.py)

    def _classify_outcome(
        self, obs: dict[str, NDArray[np.floating]], terminated: bool, truncated: bool
    ) -> tuple[str, str | None, int]:
        """Decide why the run ended. Returns (outcome, hit_kind, hit_index)."""
        if int(obs["target_gate"]) == -1:
            return "success", None, -1
        # Once disabled, the env warps the drone to a sentinel pos. Use the last logged real pos.
        real_pos = self._log[-1]["pos"] if self._log else obs["pos"]
        if np.any(real_pos < self._pos_limit_low) or np.any(real_pos > self._pos_limit_high):
            return "out_of_bounds", None, -1
        if truncated and not terminated:
            return "timeout", None, -1
        # Otherwise we were disabled by a contact. Closest object is our best guess.
        drone_xy = real_pos[:2]
        gates_xy = self._log[-1]["gates_pos"][:, :2] if self._log else obs["gates_pos"][:, :2]
        obs_xy = (
            self._log[-1]["obstacles_pos"][:, :2] if self._log else obs["obstacles_pos"][:, :2]
        )
        d_gates = np.linalg.norm(gates_xy - drone_xy, axis=1)
        d_obs = np.linalg.norm(obs_xy - drone_xy, axis=1)
        if d_gates.size and (not d_obs.size or d_gates.min() < d_obs.min()):
            return "collision_gate", "gate", int(d_gates.argmin())
        return "collision_obstacle", "obstacle", int(d_obs.argmin())

    def _save_run(
        self, obs: dict[str, NDArray[np.floating]], terminated: bool, truncated: bool
    ) -> None:
        """Dump per-tick log + JSON summary to ``tries/``."""
        if not self._log:
            return
        outcome, hit_kind, hit_index = self._classify_outcome(obs, terminated, truncated)
        real_pos = self._log[-1]["pos"]
        ts = self._run_start.strftime("%Y%m%d_%H%M%S")
        base = self._save_dir / f"{ts}_{outcome}"

        # Waypoint count can change tick-to-tick (detours appear/disappear); handle both cases.
        stack_keys = [
            "t", "pos", "vel", "quat", "target_gate",
            "gates_pos", "gates_quat", "obstacles_pos",
            "des_pos", "des_vel", "action",
        ]
        arrays = {k: np.array([r[k] for r in self._log]) for k in stack_keys}
        wp_lens = [len(r["waypoints"]) for r in self._log]
        if all(L == wp_lens[0] for L in wp_lens):
            arrays["waypoints"] = np.stack([r["waypoints"] for r in self._log])
        else:
            arrays["waypoints_concat"] = np.concatenate([r["waypoints"] for r in self._log])
            arrays["waypoints_lengths"] = np.array(wp_lens, dtype=np.int32)
        arrays["gates_nominal"] = self._gates_nominal
        arrays["obstacles_nominal"] = self._obstacles_nominal
        np.savez(base.with_suffix(".npz"), **arrays)

        first = self._log[0]
        summary = {
            "outcome": outcome,
            "hit_kind": hit_kind,
            "hit_index": hit_index,
            "final_pos": real_pos.tolist(),
            "warped_pos_at_termination": obs["pos"].tolist(),
            "final_target_gate": int(obs["target_gate"]),
            "n_ticks": len(self._log),
            "duration_s": float(self._log[-1]["t"]),
            "config_seed": self._config_seed,
            "rng_key": [int(x) for x in self._rng_key] if self._rng_key is not None else None,
            "initial_drone_pos": first["pos"].tolist(),
            "initial_gates_pos": first["gates_pos"].tolist(),
            "initial_obstacles_pos": first["obstacles_pos"].tolist(),
            "gates_nominal": self._gates_nominal.tolist(),
            "obstacles_nominal": self._obstacles_nominal.tolist(),
            "gates_final": obs["gates_pos"].tolist(),
            "obstacles_final": obs["obstacles_pos"].tolist(),
        }
        with open(base.with_suffix(".json"), "w") as f:
            json.dump(summary, f, indent=2)
