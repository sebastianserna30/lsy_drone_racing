"""Wall-clock cost of one MPPI control step, and the control rate it implies.

The question this answers: can the IBR (Nash) arm actually run at the 50 Hz the env steps at, or
is it writing cheques the hardware cannot cash? A controller whose median step is 12 ms but whose
p99 is 40 ms does not run at 50 Hz -- it silently misses deadlines on exactly the steps where the
drones are close, which is when the coupling matters.

So the reported number is the tail, not the mean. ``hz_p99`` is the rate the controller sustains
if every step must fit; ``over_budget_pct`` is how often it blows the 20 ms budget at 50 Hz.

One MPPI instance is timed, in a real closed loop against the open-loop opponent -- the deployment
shape, where the other drone is real hardware and only our own solver runs on this machine. Timing
both MPPIs of the study config would measure GPU contention that deployment never sees.

Run as:

    $ pixi run -e gpu python scripts/bench_ctrl_freq.py --preset ibr_iters
    $ pixi run -e gpu python scripts/bench_ctrl_freq.py --preset samples --steps 150
    $ pixi run -e gpu python scripts/bench_ctrl_freq.py --preset all --out bench.csv

Writes a CSV of one row per arm.
"""

from __future__ import annotations

import copy
import csv
import logging
import platform
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

import fire
import gymnasium
import numpy as np
import toml
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

from lsy_drone_racing.utils import load_config, load_controller  # isort: skip

if TYPE_CHECKING:
    from ml_collections import ConfigDict

logger = logging.getLogger(__name__)
ROOT = Path(__file__).parents[1]

EGO_RANK = 1  # drone 0 is the open-loop opponent, drone 1 is the MPPI under test
CTRL_BUDGET_MS = 20.0  # 50 Hz env step

# An arm is (label, opponent-section overrides, mppi-section overrides). ibr_iters = -1 disables
# best response and falls back to the one-shot coupled cost, which is the A/B control: without it
# there is no way to say what the iteration costs.
PRESETS: dict[str, list[tuple[str, dict, dict]]] = {
    "ibr_iters": [
        ("const_vel (no joint rollout)", {"model": "const_vel"}, {}),
        ("parallel (ibr off)", {"model": "mppi", "ibr_iters": -1}, {}),
        ("ibr 0 iters", {"model": "mppi", "ibr_iters": 0}, {}),
        ("ibr 1 iter", {"model": "mppi", "ibr_iters": 1}, {}),
        ("ibr 3 iters", {"model": "mppi", "ibr_iters": 3}, {}),
        ("ibr 5 iters", {"model": "mppi", "ibr_iters": 5}, {}),
        ("ibr 10 iters", {"model": "mppi", "ibr_iters": 10}, {}),
    ],
    "ibr_mode": [
        ("scan (Gauss-Seidel) 3", {"model": "mppi", "ibr_iters": 3, "ibr_mode": "scan"}, {}),
        ("vmap (Jacobi) 3", {"model": "mppi", "ibr_iters": 3, "ibr_mode": "vmap"}, {}),
        ("scan (Gauss-Seidel) 10", {"model": "mppi", "ibr_iters": 10, "ibr_mode": "scan"}, {}),
        ("vmap (Jacobi) 10", {"model": "mppi", "ibr_iters": 10, "ibr_mode": "vmap"}, {}),
    ],
    "samples": [
        (f"ibr 3, {n} samples", {"model": "mppi", "ibr_iters": 3}, {"n_samples": n})
        for n in (1000, 5000, 10000, 25000, 50000, 100000)
    ],
    "horizon": [
        (f"ibr 3, N={n}", {"model": "mppi", "ibr_iters": 3}, {"N": n, "T": 0.02 * n})
        for n in (10, 20, 30, 40)
    ],
}
PRESETS["all"] = PRESETS["ibr_iters"] + PRESETS["ibr_mode"][1::2] + PRESETS["samples"]


def _build_env(cfg: ConfigDict, seed: int) -> JaxToNumpy:
    """Same construction path as the study harness, so timings match how runs are actually made."""
    freqs = np.array([kw["freq"] for kw in cfg.env.kwargs], dtype=np.int64)
    return JaxToNumpy(
        gymnasium.make(
            "MultiDroneRacing-v0",
            freq=int(freqs.max()),
            sim_config=cfg.sim,
            track=cfg.env.track,
            sensor_range=cfg.env.kwargs[0]["sensor_range"],
            control_mode=cfg.env.kwargs[0]["control_mode"],
            disturbances=cfg.env.get("disturbances"),
            randomizations=cfg.env.get("randomizations"),
            seed=seed,
        )
    )


def _time_episode(cfg: ConfigDict, classes: list, env: JaxToNumpy, steps: int, warmup: int) -> dict:
    """Fly one episode, timing every ego ``compute_control`` call.

    ``compute_control`` returns a numpy array, so the conversion off the device already blocks on
    the solve -- a bare ``perf_counter`` around the call is an honest end-to-end measurement, not
    an async dispatch time.
    """
    obs, info = env.reset()
    n_drones = env.unwrapped.sim.n_drones
    freqs = np.array([kw["freq"] for kw in cfg.env.kwargs], dtype=np.int64)
    base_freq = int(freqs.max())
    periods = base_freq // freqs

    controllers, build_s = [], []
    for rank, cls in enumerate(classes):
        ctrl_cfg = copy.deepcopy(cfg)
        ctrl_cfg.env.freq = np.int64(ctrl_cfg.env.kwargs[rank]["freq"])
        t0 = time.perf_counter()
        controllers.append(cls(obs, info | {"rank": rank}, ctrl_cfg))
        build_s.append(time.perf_counter() - t0)

    samples: list[float] = []
    actions = np.zeros((n_drones, env.action_space.shape[1]), dtype=np.float32)
    finish = np.full(n_drones, np.nan, dtype=np.float64)
    i = 0
    while len(samples) < steps + warmup:
        disabled = np.asarray(env.unwrapped.data.disabled_drones[0])
        mask = (i % periods) == 0
        for rank, ctrl in enumerate(controllers):
            if disabled[rank] or not mask[rank]:
                continue
            if rank == EGO_RANK:
                t0 = time.perf_counter()
                actions[rank] = ctrl.compute_control(obs, info | {"rank": rank})
                samples.append((time.perf_counter() - t0) * 1e3)
            else:
                actions[rank] = ctrl.compute_control(obs, info | {"rank": rank})
        obs, reward, terminated, truncated, info = env.step(actions)
        for rank, ctrl in enumerate(controllers):
            if not disabled[rank] and mask[rank]:
                ctrl.step_callback(
                    actions[rank], obs, reward, terminated, truncated, info | {"rank": rank}
                )
        gate = np.asarray(obs["target_gate"]).ravel()
        finish[(gate == -1) & np.isnan(finish)] = i / base_freq
        i += 1
        if terminated or truncated or not np.isnan(finish[EGO_RANK]):
            break
    for ctrl in controllers:
        ctrl.episode_callback()
        ctrl.episode_reset()

    # The reset() warmup already compiled the update, but the first few live steps still hit
    # fresh branches (sensor range, tracker init); drop them rather than let them set the tail.
    ms = np.asarray(samples[warmup:], dtype=np.float64)
    return {
        "build_s": build_s[EGO_RANK],
        "ms": ms,
        "ego_finished": bool(np.isfinite(finish[EGO_RANK])),
        "steps_flown": i,
    }


def _time_isolated(
    cfg: ConfigDict, classes: list, env: JaxToNumpy, steps: int, warmup: int
) -> dict:
    """Time ``compute_control`` alone, on a frozen observation, with the env never stepped.

    This is the number deployment cares about. In flight the other drone is real hardware and
    nothing else is on the GPU, whereas in sim the env's own 500 Hz crazyflow physics runs on the
    same device between control steps and serialises against the solver. The in-loop measurement
    charges the controller for that contention; this one does not.

    The obs is frozen, so the controller re-solves the same problem every step. That understates
    nothing in the solve itself (the shapes and the iteration count are fixed, and MPPI does no
    early exit), but it does hold the drones at their start separation -- see the in-loop arm for
    the cost when the keep-out is actually active.
    """
    obs, info = env.reset()
    controllers, build_s = [], []
    for rank, cls in enumerate(classes):
        ctrl_cfg = copy.deepcopy(cfg)
        ctrl_cfg.env.freq = np.int64(ctrl_cfg.env.kwargs[rank]["freq"])
        t0 = time.perf_counter()
        controllers.append(cls(obs, info | {"rank": rank}, ctrl_cfg))
        build_s.append(time.perf_counter() - t0)

    ego = controllers[EGO_RANK]
    ego_info = info | {"rank": EGO_RANK}
    samples = []
    for _ in range(steps + warmup):
        t0 = time.perf_counter()
        ego.compute_control(obs, ego_info)
        samples.append((time.perf_counter() - t0) * 1e3)
    for ctrl in controllers:
        ctrl.episode_callback()
        ctrl.episode_reset()
    return {
        "build_s": build_s[EGO_RANK],
        "ms": np.asarray(samples[warmup:], dtype=np.float64),
        "ego_finished": False,
        "steps_flown": 0,
    }


def _row(label: str, opp: dict, mppi: dict, res: dict) -> dict:
    """Summarise one arm. The percentiles are the point -- the mean hides the deadline misses."""
    ms = res["ms"]
    p99 = float(np.percentile(ms, 99))
    return {
        "arm": label,
        "mode": res["mode"],
        "model": opp.get("model", "mppi"),
        "ibr_iters": opp.get("ibr_iters", ""),
        "ibr_mode": opp.get("ibr_mode", "scan"),
        "n_samples": mppi.get("n_samples", ""),
        "N": mppi.get("N", ""),
        "n_steps": int(ms.size),
        "mean_ms": round(float(ms.mean()), 3),
        "p50_ms": round(float(np.percentile(ms, 50)), 3),
        "p90_ms": round(float(np.percentile(ms, 90)), 3),
        "p99_ms": round(p99, 3),
        "max_ms": round(float(ms.max()), 3),
        # Sustained rate if every step must fit inside its period.
        "hz_p99": round(1e3 / p99, 1),
        "hz_p50": round(1e3 / float(np.percentile(ms, 50)), 1),
        "over_budget_pct": round(100.0 * float((ms > CTRL_BUDGET_MS).mean()), 2),
        "build_s": round(res["build_s"], 2),
        "ego_finished": res["ego_finished"],
    }


def bench(
    preset: str = "ibr_iters",
    config: str = "multi_level0.toml",
    steps: int = 120,
    warmup: int = 10,
    seed: int = 7,
    mode: str = "both",
    out: str | None = None,
) -> None:
    """Time one MPPI control step across a preset of arms and print the implied control rate.

    Args:
        preset: which arm list to run -- ``ibr_iters``, ``ibr_mode``, ``samples``, ``horizon`` or
            ``all``.
        config: base TOML in ``config/``. The opponent/mppi sections are overridden per arm.
        steps: timed control steps per arm (after warmup). An episode is ~300 steps at 50 Hz, so
            an in-loop arm may end early if the ego finishes or crashes; ``n_steps`` records what
            was actually measured.
        warmup: leading steps discarded before the statistics.
        seed: track seed, held fixed so every arm flies the same geometry.
        mode: ``isolated`` (solver alone, deployment-like), ``inloop`` (flying, with the env's
            physics contending for the same GPU) or ``both``.
        out: CSV path; defaults to ``bench_ctrl_freq_<preset>.csv``.
    """
    if preset not in PRESETS:
        raise ValueError(f"preset must be one of {sorted(PRESETS)}, got {preset!r}")
    modes = ("isolated", "inloop") if mode == "both" else (mode,)
    timers = {"isolated": _time_isolated, "inloop": _time_episode}
    if any(m not in timers for m in modes):
        raise ValueError(f"mode must be isolated | inloop | both, got {mode!r}")
    arms = PRESETS[preset]
    raw = toml.load(ROOT / "config" / config)
    raw["sim"]["render"] = False
    raw["env"]["seed"] = seed

    print(f"host: {platform.node()}  |  budget {CTRL_BUDGET_MS:.0f} ms (50 Hz)")
    rows = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for label, opp, mppi in arms:
            cfg_raw = copy.deepcopy(raw)
            # The MPPI is controller[EGO_RANK]; controller[0] is the open-loop opponent.
            sec = cfg_raw["controller"][EGO_RANK]["mppi"]
            sec["opponent"].update(opp)
            sec.update(mppi)
            path = tmp / "cfg.toml"
            path.write_text(toml.dumps(cfg_raw))
            cfg = load_config(path)
            classes = [
                load_controller(ROOT / "lsy_drone_racing" / "control" / c["file"])
                for c in cfg.controller
            ]
            for m in modes:
                env = _build_env(cfg, seed)
                try:
                    res = timers[m](cfg, classes, env, steps, warmup)
                finally:
                    env.close()
                if res["ms"].size == 0:
                    logger.warning("arm %s (%s) produced no timed steps", label, m)
                    continue
                res["mode"] = m
                row = _row(label, opp, mppi, res)
                rows.append(row)
                print(
                    f"{row['arm']:<30} {m:<9} p50 {row['p50_ms']:6.2f} ms  "
                    f"p99 {row['p99_ms']:6.2f} ms  max {row['max_ms']:6.2f} ms  -> "
                    f"{row['hz_p99']:6.1f} Hz (p99)  over-budget {row['over_budget_pct']:5.1f}%  "
                    f"n={row['n_steps']}"
                )

    out_path = Path(out) if out else ROOT / f"bench_ctrl_freq_{preset}.csv"
    if rows:
        with out_path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {out_path}")


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO)
    fire.Fire(bench)


if __name__ == "__main__":
    main()
