"""Diff two MPPI run logs to prove a refactor changed nothing.

The MPPI controller writes a ``.npz`` per episode when ``LOG_DRONE_DATA`` is set (see
``AttitudeMPPIController.episode_callback``). This script compares two such logs — or two
directories of them — and requires every array to be *exactly* equal. Not ``allclose``: the
controller is chaotic, so a change that is invisible at 1e-9 on step 5 is a different
trajectory by step 200. Bit-identity is the only useful signal.

Run as:

    $ python scripts/diff_runs.py runs/baseline runs/step1
    $ python scripts/diff_runs.py runs/baseline/single.npz runs/step1/single.npz

Exits 0 when identical, 1 otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import fire
import numpy as np


def _first_difference(a: np.ndarray, b: np.ndarray) -> str:
    """Describe where two same-shaped arrays first diverge, along the leading (step) axis."""
    unequal = a != b
    if a.dtype.kind == "f":  # NaN != NaN, but two logs with NaN in the same slot agree
        unequal &= ~(np.isnan(a) & np.isnan(b))
    if not unequal.any():
        return "arrays differ only in NaN placement"
    steps = np.flatnonzero(unequal.reshape(unequal.shape[0], -1).any(axis=1))
    step = int(steps[0])
    return (
        f"first differs at step {step} of {a.shape[0]} "
        f"({len(steps)} steps differ); {a[step].ravel()[:4]} != {b[step].ravel()[:4]}"
    )


def _diff_file(old: Path, new: Path) -> list[str]:
    """Compare one pair of npz logs. Returns a list of human-readable differences."""
    a, b = np.load(old, allow_pickle=True), np.load(new, allow_pickle=True)
    problems = []

    if missing := sorted(set(a.files) - set(b.files)):
        problems.append(f"keys missing from new log: {missing}")
    if added := sorted(set(b.files) - set(a.files)):
        problems.append(f"keys added in new log: {added}")

    for key in sorted(set(a.files) & set(b.files)):
        x, y = a[key], b[key]
        if x.shape != y.shape:
            problems.append(f"{key}: shape {x.shape} -> {y.shape}")
            continue
        if x.dtype != y.dtype:
            problems.append(f"{key}: dtype {x.dtype} -> {y.dtype}")
            continue
        if x.dtype.kind in "fc":
            same = np.array_equal(x, y, equal_nan=True)
        else:
            same = np.array_equal(x, y)
        if not same:
            problems.append(f"{key}: {_first_difference(x, y)}")
    return problems


def diff(old: str, new: str) -> int:
    """Compare two run logs, or two directories of run logs.

    Args:
        old: baseline ``.npz`` file, or a directory of them.
        new: ``.npz`` file or directory to compare against the baseline.

    Returns:
        Process exit code: 0 if every array is bit-identical, 1 otherwise.
    """
    old_path, new_path = Path(old), Path(new)

    if old_path.is_dir():
        pairs = [(f, new_path / f.name) for f in sorted(old_path.glob("*.npz"))]
        if not pairs:
            print(f"no .npz logs found in {old_path}")
            return 1
    else:
        pairs = [(old_path, new_path)]

    failed = False
    for old_file, new_file in pairs:
        name = old_file.name
        if not new_file.exists():
            print(f"MISSING  {name}: no counterpart at {new_file}")
            failed = True
            continue
        problems = _diff_file(old_file, new_file)
        if problems:
            failed = True
            print(f"DIFFERS  {name}")
            for problem in problems:
                print(f"           {problem}")
        else:
            print(f"identical {name}")

    print("\nnot bit-identical" if failed else "\nall runs bit-identical")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(fire.Fire(diff))
