"""Capture the MPPI reference runs used to prove a refactor changed nothing.

Runs a fixed set of configurations with a pinned seed and rendering off, and writes one
``.npz`` run log per configuration (see ``scripts/diff_runs.py`` for the comparison side).
The set covers the single-agent path plus all three opponent models, because they take
different branches through sampling, rollout and cost.

Run as:

    $ python scripts/capture_baselines.py --out runs/baseline
    $ python scripts/capture_baselines.py --out runs/step1 --only single

Then:

    $ python scripts/diff_runs.py runs/baseline runs/step1

Each configuration is run in its own subprocess: four GPU sims in one process would not fit,
and a fresh process also rules out JAX state leaking between runs.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import fire
import toml

ROOT = Path(__file__).parents[1]

# Pinned so every capture reproduces the same track randomization. race_core now uses this
# integer directly as the JAX key, so the same value always yields the same episode.
SEED = 20260728

# name -> (sim script, config file, extra overrides). Overrides are dotted paths; an integer
# segment indexes into a TOML array-of-tables, so `controller.1` is the second [[controller]].
BASELINES: dict[str, tuple[str, str, dict[str, Any]]] = {
    # level2.toml would be the natural single-agent choice, but it omits
    # experiment.env.gate_frame_radius and so crashes the MPPI controller on the first control
    # step. level2_MPPI.toml is the same track with a complete controller block.
    "single": ("sim.py", "level2_MPPI.toml", {}),
    "multi_mppi": (
        "multi_sim.py",
        "multi_level0.toml",
        {"controller.1.mppi.opponent.model": "mppi"},
    ),
    "multi_spline_progress": (
        "multi_sim.py",
        "multi_level0.toml",
        {"controller.1.mppi.opponent.model": "spline_progress"},
    ),
    "multi_const_vel": (
        "multi_sim.py",
        "multi_level0.toml",
        {"controller.1.mppi.opponent.model": "const_vel"},
    ),
}


def _set_path(cfg: dict, dotted: str, value: Any) -> None:
    """Set a dotted key in a nested TOML dict; integer segments index arrays-of-tables."""
    node = cfg
    keys = dotted.split(".")
    for key in keys[:-1]:
        node = node[int(key)] if key.isdigit() else node[key]
    last = keys[-1]
    if last.isdigit():
        node[int(last)] = value
    else:
        node[last] = value


def _write_config(config: str, overrides: dict[str, Any], out_dir: Path, seed: int) -> Path:
    """Write a copy of `config` with the seed pinned, rendering off, and `overrides` applied."""
    cfg = toml.load(ROOT / "config" / config)
    _set_path(cfg, "env.seed", seed)
    _set_path(cfg, "sim.render", False)
    for dotted, value in overrides.items():
        _set_path(cfg, dotted, value)
    path = out_dir / f"{Path(config).stem}.toml"
    path.write_text(toml.dumps(cfg))
    return path


def capture(out: str = "runs/baseline", only: str | None = None, seed: int = SEED) -> int:
    """Run every baseline configuration and write its log to `out`.

    Args:
        out: directory for the ``.npz`` run logs. Created if missing.
        only: run just this baseline (one of the keys in ``BASELINES``) instead of all.
        seed: environment seed. Vary it to check a change across several tracks.

    Returns:
        Process exit code: 0 when every run produced a log.
    """
    selected = BASELINES if only is None else {only: BASELINES[only]}
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    failed = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, (script, config, overrides) in selected.items():
            log_path = out_dir / f"{name}.npz"
            cfg_path = _write_config(config, overrides, Path(tmp), seed)

            print(f"\n=== {name}: {script} {config} -> {log_path} ===", flush=True)
            env = os.environ | {"LOG_DRONE_DATA": str(log_path)}
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / script), "--config", str(cfg_path)],
                env=env,
                cwd=ROOT,
            )
            if result.returncode != 0:
                print(f"  run failed with exit code {result.returncode}")
                failed.append(name)
            elif not log_path.exists():
                print("  run produced no log -- is LOG_DRONE_DATA honoured by the controller?")
                failed.append(name)

    if failed:
        print(f"\nfailed: {sorted(failed)}")
        return 1
    print(f"\ncaptured {len(selected)} baselines into {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(fire.Fire(capture))
