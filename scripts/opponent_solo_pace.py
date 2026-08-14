"""Measure the opponent's uncontested lap time at each of its progress weights.

The opponent-pace study sweeps ``controller[0][mppi][cost].progress``, which is a cost weight and
means nothing to a reader. This labels that axis in seconds by flying the opponent's own
configuration ALONE, in a single-drone race -- "opponent speed in single agent".

It has to be measured rather than read off the contested runs: ``--stop_on_ego_finish`` cuts the
opponent off in every race the ego wins, so its recorded times there are censored and biased fast,
and even without early stopping a contested lap is slowed by the ego being in the way.

Run as:

    $ python scripts/opponent_solo_pace.py --n_seeds 5
    $ python scripts/opponent_solo_pace.py --config study_pass.toml --values 0.1,0.4,1.5

Writes ``opponent_solo_pace.csv`` mapping progress weight -> mean solo lap time.
"""

from __future__ import annotations

import copy
import csv
import logging
import tempfile
from pathlib import Path

import fire
import gymnasium
import numpy as np
import toml
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

from lsy_drone_racing.utils import load_config, load_controller  # isort: skip

logger = logging.getLogger(__name__)
ROOT = Path(__file__).parents[1]
DEFAULT_VALUES = (0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5)


def _solo_lap(raw: dict, seed: int, tmp: Path) -> float:
    """Fly one drone alone and return its lap time, or nan if it never finished."""
    raw = copy.deepcopy(raw)
    raw["env"]["seed"] = seed
    raw["sim"]["render"] = False
    path = tmp / f"solo_{seed}.toml"
    path.write_text(toml.dumps(raw))
    cfg = load_config(path)
    ctrl_cls = load_controller(ROOT / "lsy_drone_racing" / "control" / cfg.controller["file"])
    freq = int(cfg.env.kwargs[0]["freq"])
    env = JaxToNumpy(gymnasium.make(
        "DroneRacing-v0",
        freq=freq,
        sim_config=cfg.sim,
        track=cfg.env.track,
        sensor_range=cfg.env.kwargs[0]["sensor_range"],
        control_mode=cfg.env.kwargs[0]["control_mode"],
        disturbances=cfg.env.get("disturbances"),
        randomizations=cfg.env.get("randomizations"),
        seed=seed,
    ))
    obs, info = env.reset()
    ctrl_cfg = copy.deepcopy(cfg)
    ctrl_cfg.env.freq = np.int64(freq)
    ctrl = ctrl_cls(obs, info | {"rank": 0}, ctrl_cfg)
    finish, i = float("nan"), 0
    while True:
        action = ctrl.compute_control(obs, info | {"rank": 0})
        obs, reward, terminated, truncated, info = env.step(action)
        ctrl.step_callback(action, obs, reward, terminated, truncated, info | {"rank": 0})
        if int(np.asarray(obs["target_gate"]).ravel()[0]) == -1 and np.isnan(finish):
            finish = i / freq
            break
        i += 1
        if terminated or truncated:
            break
    ctrl.episode_callback()
    ctrl.episode_reset()
    env.close()
    return finish


def measure(
    config: str = "study_pass.toml", n_seeds: int = 5, values: str | None = None, seed0: int = 1000
) -> int:
    """Fly the opponent's config solo at each progress weight and record its lap time.

    Args:
        config: multi-agent config in ``config/``; ``controller[0]`` is taken as the opponent.
        n_seeds: seeds per value. Lap time is continuous, so this needs far fewer runs than a
            rate does -- a handful is plenty to label an axis.
        values: comma-separated progress weights; defaults to the study grid.
        seed0: first seed, matching the study so the tracks are the same ones.

    Returns:
        Process exit code.
    """
    if values is None:
        grid = list(DEFAULT_VALUES)
    elif isinstance(values, (list, tuple)):
        grid = [float(v) for v in values]
    else:
        grid = [float(v) for v in str(values).split(",") if str(v).strip()]

    base = toml.load(ROOT / "config" / config)
    opponent = copy.deepcopy(base["controller"][0])
    # Single-drone env: one controller, one drone on the track, and the single-agent controller
    # file (the _multi wrapper only reorders a batched observation that will not exist here).
    opponent["file"] = "trajectory_mppi.py"
    # A single-drone config carries [controller] as one table, not an array of them.
    base["controller"] = opponent
    base["env"]["id"] = "DroneRacing-v0"
    base["env"]["kwargs"] = [base["env"]["kwargs"][0]]
    base["env"]["track"]["drones"] = [base["env"]["track"]["drones"][0]]

    rows = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for value in grid:
            raw = copy.deepcopy(base)
            raw["controller"]["mppi"]["cost"]["progress"] = value
            laps = np.array([_solo_lap(raw, seed0 + i, tmp) for i in range(n_seeds)])
            ok = np.isfinite(laps)
            rows.append({
                "progress": value,
                "solo_lap_time": float(laps[ok].mean()) if ok.any() else float("nan"),
                "solo_lap_std": float(laps[ok].std()) if ok.sum() > 1 else 0.0,
                "finished": int(ok.sum()),
                "n_seeds": n_seeds,
            })
            print(
                f"progress={value:<5} solo_lap={rows[-1]['solo_lap_time']:.2f}s "
                f"+-{rows[-1]['solo_lap_std']:.2f}  finished {ok.sum()}/{n_seeds}",
                flush=True,
            )

    out = ROOT / "opponent_solo_pace.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    raise SystemExit(fire.Fire(measure))
