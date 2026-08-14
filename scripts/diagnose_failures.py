"""Run episodes and report WHY each drone stopped, not just whether it finished.

``multi_evaluate.py`` records finish times, so a failed run is an indistinguishable NaN: a
start-line collision, a clipped gate frame and a drone that flew out of the arena all look the
same. This script watches for the moment the environment disables each drone and classifies the
cause from the last state before the disable, so a low completion rate can be attributed.

The environment disables a drone when it leaves the position limits, touches anything with a
contact mask, or finishes the track (``race_core._disabled_drones``). Contacts are decoded
geometrically rather than by reading MuJoCo's contact buffer, because by the time control
returns the offending drone has already been warped to z = -1 and the buffer no longer describes
the collision.

Run as:

    $ python scripts/diagnose_failures.py --n_runs 20
    $ python scripts/diagnose_failures.py --n_runs 20 --config deploy_multi.toml --n_samples 10000
"""

from __future__ import annotations

import copy
import logging
import tempfile
from collections import Counter
from pathlib import Path

import fire
import gymnasium
import numpy as np
import toml
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

# lsy_drone_racing pulls in crazyflow, which refuses to load if scipy was imported first without
# SCIPY_ARRAY_API set. Keep this import above the scipy one.
from lsy_drone_racing.utils import load_config, load_controller  # isort: skip
from scipy.spatial.transform import Rotation as R  # noqa: E402  (see above)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parents[1]
# Half-width of a gate frame side, matching AttitudeMPPIController.get_gate_frame_pos.
GATE_FRAME_HALF_WIDTH = 0.28
# Slack added to each contact radius. The disable fires on the true mesh contact, whereas we
# test sphere distances one control step earlier, so the drone has moved since.
MARGIN = 0.08


def _gate_frame_points(gates_pos: np.ndarray, gates_quat: np.ndarray) -> np.ndarray:
    """Both upright frame members of every gate, as (n_gates * 2, 3) keep-out centres."""
    points = []
    for pos, quat in zip(gates_pos, gates_quat):
        side = R.from_quat(quat).apply([0.0, 1.0, 0.0])
        points.append(pos - GATE_FRAME_HALF_WIDTH * side)
        points.append(pos + GATE_FRAME_HALF_WIDTH * side)
    return np.asarray(points)


def _classify(
    rank: int,
    pos: np.ndarray,
    obs: dict,
    limits_low: np.ndarray,
    limits_high: np.ndarray,
    radii: dict[str, float],
) -> tuple[str, float]:
    """Name the most likely cause of a disable from the last state before it.

    Returns (cause, margin) where margin is how far inside the offending radius the drone was.
    Candidates are scored by penetration depth rather than raw distance, so that a drone which
    is near both a gate and its opponent is attributed to whichever it was actually inside.
    """
    me = pos[rank]
    if np.any(me < limits_low) or np.any(me > limits_high):
        return "out_of_bounds", 0.0

    candidates: list[tuple[str, float]] = []

    # Opponent drone: full 3D separation between two airframes.
    for other in range(pos.shape[0]):
        if other == rank:
            continue
        reach = 2 * radii["drone"] + MARGIN
        candidates.append(("hit_opponent", reach - float(np.linalg.norm(me - pos[other]))))

    # Track observations are per-drone (each drone senses its own view), so select this one's.
    def per_drone(key: str) -> np.ndarray:
        arr = np.asarray(obs[key])
        return arr[rank] if arr.ndim == 3 else arr

    # Pillars are vertical, so only the horizontal offset matters.
    obstacles = np.atleast_2d(per_drone("obstacles_pos"))
    reach = radii["obstacle"] + radii["drone"] + MARGIN
    d_obs = np.linalg.norm(obstacles[:, :2] - me[:2], axis=-1)
    candidates.append(("hit_obstacle", reach - float(d_obs.min())))

    frames = _gate_frame_points(per_drone("gates_pos"), per_drone("gates_quat"))
    reach = radii["gate_frame"] + radii["drone"] + MARGIN
    d_gate = np.linalg.norm(frames - me, axis=-1)
    candidates.append(("hit_gate_frame", reach - float(d_gate.min())))

    if me[2] < radii["drone"]:
        candidates.append(("hit_ground", radii["drone"] - float(me[2])))

    cause, depth = max(candidates, key=lambda c: c[1])
    return (cause, depth) if depth > 0 else ("unexplained", depth)


def diagnose(
    n_runs: int = 20,
    config: str = "deploy_multi.toml",
    n_samples: int | None = None,
    seed: int | None = None,
    seeds: str | None = None,
    drone_exp: float | None = None,
    downwash: float | None = None,
) -> int:
    """Run episodes and print a breakdown of how each drone's episode ended.

    Args:
        n_runs: number of episodes. Ignored when `seeds` is given.
        config: config file in ``config/``.
        n_samples: override the MPPI sample count, to trade fidelity for speed.
        seed: override the environment seed for the whole session.
        drone_exp: override the opponent-collision weight, for A/B sweeps of it.
        downwash: override the wake penalty weight; 0 disables it.
        seeds: comma-separated seeds, one episode each, reset explicitly so every run is
            REPRODUCIBLE and labelled with the seed that produced it. Without this the config's
            ``env.seed = -1`` means successive resets draw from a shared stream and no
            individual run can be replayed.

    Returns:
        Process exit code.
    """
    # fire hands us a tuple for "1,2,3", an int for a single value, and a str if quoted.
    if seeds is None:
        seed_list = None
    elif isinstance(seeds, (list, tuple)):
        seed_list = [int(s) for s in seeds]
    elif isinstance(seeds, int):
        seed_list = [seeds]
    else:
        seed_list = [int(s) for s in str(seeds).split(",") if s.strip()]
    if seed_list:
        n_runs = len(seed_list)
    cfg_path = ROOT / "config" / config
    if any(v is not None for v in (n_samples, seed, drone_exp, downwash)):
        raw = toml.load(cfg_path)
        if n_samples is not None:
            raw["controller"][1]["mppi"]["n_samples"] = n_samples
        if seed is not None:
            raw["env"]["seed"] = seed
        if drone_exp is not None:
            raw["controller"][1]["mppi"]["opponent"]["drone_exp"] = float(drone_exp)
        if downwash is not None:
            raw["controller"][1]["mppi"]["opponent"]["downwash"] = float(downwash)
        # Written to a temp dir, not config/, so overrides never litter the tracked configs.
        tmp_dir = Path(tempfile.mkdtemp(prefix="diagnose_"))
        cfg_path = tmp_dir / config
        cfg_path.write_text(toml.dumps(raw))
    cfg = load_config(cfg_path)
    cfg.sim.render = False

    geometry = cfg.controller[1]["mppi"]["geometry"]
    radii = {
        "drone": float(geometry["drone_radius"]),
        "obstacle": float(geometry["obstacle_radius"]),
        "gate_frame": float(geometry["gate_frame_radius"]),
    }
    limits_low = np.asarray(cfg.env.track.safety_limits.pos_limit_low)
    limits_high = np.asarray(cfg.env.track.safety_limits.pos_limit_high)

    names = [c["file"] for c in cfg.controller]
    classes = [load_controller(ROOT / "lsy_drone_racing" / "control" / n) for n in names]
    freqs = np.array([kw["freq"] for kw in cfg.env.kwargs], dtype=np.int64)
    base_freq = int(freqs.max())
    periods = base_freq // freqs

    def make_env(env_seed: int) -> JaxToNumpy:
        """Build the env exactly as multi_sim does, seeding through the CONSTRUCTOR.

        The constructor and ``reset(seed=...)`` are different seeding paths: the constructor maps
        the seed through numpy to a JAX key, whereas reset hands the raw seed to ``seed_sim``.
        Only the constructor path matches what ``env.seed`` in the TOML does, so a seed reported
        here replays by editing the config and nothing else.
        """
        return JaxToNumpy(gymnasium.make(
            "MultiDroneRacing-v0",
            freq=base_freq,
            sim_config=cfg.sim,
            track=cfg.env.track,
            sensor_range=cfg.env.kwargs[0]["sensor_range"],
            control_mode=cfg.env.kwargs[0]["control_mode"],
            disturbances=cfg.env.get("disturbances"),
            randomizations=cfg.env.get("randomizations"),
            seed=env_seed,
        ))

    env = make_env(cfg.env.seed)
    n_drones = env.unwrapped.sim.n_drones
    action_dim = env.action_space.shape[1]

    outcomes: list[dict] = []
    for run in range(n_runs):
        run_seed = seed_list[run] if seed_list else None
        if run_seed is not None:
            env.close()
            env = make_env(run_seed)
        obs, info = env.reset()
        controllers = []
        for rank, cls in enumerate(classes):
            ctrl_cfg = copy.deepcopy(cfg)
            ctrl_cfg.env.freq = np.int64(ctrl_cfg.env.kwargs[rank]["freq"])
            controllers.append(cls(obs, info | {"rank": rank}, ctrl_cfg))

        actions = np.zeros((n_drones, action_dim), dtype=np.float32)
        recorded = np.zeros(n_drones, dtype=bool)
        # The disabling step warps the drone underground, so keep the previous state to judge.
        prev_pos = np.asarray(obs["pos"]).copy()
        prev_obs = {k: np.asarray(v).copy() for k, v in obs.items()}
        i = 0

        while True:
            was_disabled = np.asarray(env.unwrapped.data.disabled_drones[0]).copy()
            mask = (i % periods) == 0
            for rank, ctrl in enumerate(controllers):
                if was_disabled[rank]:
                    continue
                if mask[rank]:
                    actions[rank] = ctrl.compute_control(obs, info | {"rank": rank})

            obs, reward, terminated, truncated, info = env.step(actions)
            now_disabled = np.asarray(env.unwrapped.data.disabled_drones[0])
            # Open-loop controllers advance their internal clock in step_callback, so skipping
            # it leaves them frozen at t=0 and they never leave the start pad.
            for rank, ctrl in enumerate(controllers):
                if not was_disabled[rank] and mask[rank]:
                    ctrl.step_callback(
                        actions[rank], obs, reward, terminated, truncated, info | {"rank": rank}
                    )

            for rank in range(n_drones):
                if recorded[rank] or not now_disabled[rank]:
                    continue
                recorded[rank] = True
                # target_gate == -1 means the drone completed the track; that is a finish, not
                # a crash, and the env disables it for the same reason it disables a wreck.
                if int(np.asarray(prev_obs["target_gate"]).ravel()[rank]) == -1:
                    cause, depth = "finished", 0.0
                else:
                    cause, depth = _classify(
                        rank, prev_pos, prev_obs, limits_low, limits_high, radii
                    )
                outcomes.append({
                    "run": run, "seed": run_seed, "drone": rank,
                    "controller": names[rank], "cause": cause,
                    "t": i / base_freq, "depth": depth,
                    "gate": int(np.asarray(prev_obs["target_gate"]).ravel()[rank]),
                })

            prev_pos = np.asarray(obs["pos"]).copy()
            prev_obs = {k: np.asarray(v).copy() for k, v in obs.items()}
            i += 1
            if terminated or truncated or recorded.all():
                break

        for rank in range(n_drones):
            if not recorded[rank]:
                gate = int(np.asarray(obs["target_gate"]).ravel()[rank])
                outcomes.append({
                    "run": run, "seed": run_seed, "drone": rank, "controller": names[rank],
                    "cause": "finished" if gate == -1 else "timeout",
                    "t": i / base_freq, "depth": 0.0, "gate": gate,
                })
        for ctrl in controllers:
            ctrl.episode_callback()
            ctrl.episode_reset()
        done = [o for o in outcomes if o["run"] == run]
        tag = f"seed {run_seed}" if run_seed is not None else f"run {run:>3}"
        print(f"{tag:>12}: " + "  ".join(
            f"d{o['drone']}={o['cause']}@gate{o['gate']}@{o['t']:.2f}s"
            for o in sorted(done, key=lambda o: o["drone"])
        ), flush=True)

    env.close()
    _report(outcomes, names, n_runs)
    return 0


def _report(outcomes: list[dict], names: list[str], n_runs: int) -> None:
    """Print the per-drone cause breakdown and where on the track failures cluster."""
    print(f"\n{'=' * 62}\nFAILURE BREAKDOWN over {n_runs} runs\n{'=' * 62}")
    for rank, name in enumerate(names):
        mine = [o for o in outcomes if o["drone"] == rank]
        counts = Counter(o["cause"] for o in mine)
        print(f"\n{name}  (drone {rank})")
        for cause, n in counts.most_common():
            share = 100.0 * n / max(len(mine), 1)
            gates = sorted({o["gate"] for o in mine if o["cause"] == cause})
            where = "" if cause == "finished" else f"   at target_gate {gates}"
            print(f"  {cause:<16} {n:>3}  ({share:4.1f}%){where}")

    both = sum(
        1 for r in range(n_runs)
        if all(o["cause"] == "finished" for o in outcomes if o["run"] == r)
    )
    print(f"\nboth finished: {both}/{n_runs} ({100.0 * both / max(n_runs, 1):.1f}%)")

    # A race is won by whoever crosses the last gate first; finishing while the other crashes
    # counts as a win too, since the opponent never completes the track.
    ours = next((i for i, n in enumerate(names) if "mppi" in n), len(names) - 1)
    theirs = 1 - ours if len(names) == 2 else None
    if theirs is None:
        return
    won = lost = walkover = 0
    print(f"\nHEAD-TO-HEAD (drone {ours} = {names[ours]})")
    for r in range(n_runs):
        row = {o["drone"]: o for o in outcomes if o["run"] == r}
        me, opp = row.get(ours), row.get(theirs)
        if me is None or opp is None or me["cause"] != "finished":
            continue
        tag = f"seed {me['seed']}" if me.get("seed") is not None else f"run {r}"
        if opp["cause"] != "finished":
            walkover += 1
            print(f"  {tag:>12}: WIN  (ours {me['t']:.2f}s, opponent {opp['cause']})")
        elif me["t"] < opp["t"]:
            won += 1
            print(f"  {tag:>12}: WIN  (ours {me['t']:.2f}s vs {opp['t']:.2f}s)")
        else:
            lost += 1
            print(f"  {tag:>12}: LOSS (ours {me['t']:.2f}s vs {opp['t']:.2f}s)")
    finishes = won + lost + walkover
    print(f"\n  of {finishes} runs where we finished: {won} won on time, "
          f"{walkover} won by opponent DNF, {lost} lost on time")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    raise SystemExit(fire.Fire(diagnose))
