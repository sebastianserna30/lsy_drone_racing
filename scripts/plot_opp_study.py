"""Plot the opponent-pace study while it is still running.

``sweep_study.py --study opp_progress`` runs seed-major and rewrites its CSV after every seed, so
this can be called at any time and always sees a complete curve at equal n across the grid.

Only our own drone is plotted. The opponent's finish rate and lap time are not shown: under
``--stop_on_ego_finish`` the opponent is cut off in every run we win, so those columns are
censored and would read as though the opponent were faster and more reliable than it is.

Run as:

    $ python scripts/plot_opp_study.py
    $ python scripts/plot_opp_study.py --csv sweep_opp_progress_2026-08-05_15-35-48.csv
"""

from __future__ import annotations

import csv as csv_mod
import logging
from pathlib import Path

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logger = logging.getLogger(__name__)
ROOT = Path(__file__).parents[1]

FINISH = "#4C78A8"
OVERTAKE = "#F58518"


def _latest(root: Path) -> Path | None:
    """Newest opp_progress summary CSV, ignoring the per-run companion files."""
    found = [
        p
        for p in root.glob("sweep_opp_progress_*.csv")
        if not p.name.endswith("_runs.csv")
    ]
    return max(found, key=lambda p: p.name) if found else None


def _read(path: Path) -> list[dict]:
    with path.open() as fh:
        return [
            {
                k: float(v) if v not in ("", "nan") else float("nan")
                for k, v in row.items()
            }
            for row in csv_mod.DictReader(fh)
        ]


def _conditional(runs_path: Path) -> dict[float, dict]:
    """Overtakes as a share of the runs we FINISHED, per parameter value.

    ``overtake_pct`` divides by all runs, so a value where we crash a lot reads as poor at
    overtaking when it may simply be poor at surviving. Conditioning on our own finishes answers
    "when we got to the end, how often were we in front" -- which is the overtaking question with
    the crash rate divided out. Every overtake is a finish, so this is well defined.
    """
    by_value: dict[float, list[tuple[bool, bool]]] = {}
    with runs_path.open() as fh:
        for row in csv_mod.DictReader(fh):
            finished = row["our_time"] not in ("", "nan")
            by_value.setdefault(float(row["value"]), []).append(
                (finished, bool(int(row["ego_first"])))
            )
    out = {}
    for value, runs in by_value.items():
        n_fin = sum(1 for f, _ in runs if f)
        n_ot = sum(1 for _, o in runs if o)
        p = n_ot / n_fin if n_fin else float("nan")
        out[value] = {
            "pct": 100.0 * p if n_fin else float("nan"),
            # Binomial SE on the finished subset, so the bar widens where we crashed more.
            "se": 100.0 * (p * (1 - p) / n_fin) ** 0.5 if n_fin else float("nan"),
            "n": n_fin,
        }
    return out


def _solo_pace(root: Path) -> dict[float, float]:
    """Progress weight -> opponent's uncontested lap time, from opponent_solo_pace.py."""
    path = root / "opponent_solo_pace.csv"
    if not path.exists():
        return {}
    with path.open() as fh:
        return {
            float(r["progress"]): float(r["solo_lap_time"])
            for r in csv_mod.DictReader(fh)
        }


def plot(csv: str | None = None, out: str | None = None) -> int:
    """Draw our finish rate and overtake rate against opponent pace, plus our lap time.

    Args:
        csv: summary CSV to read; defaults to the newest one in the repo root.
        out: output image path, relative to the repo root. Defaults to a name carrying the run's
            timestamp, so each sweep keeps its own plot instead of overwriting the last one --
            the default CSV is whichever is newest, so a shared filename would silently swap the
            data under an unchanged-looking image.

    Returns:
        Process exit code.
    """
    path = ROOT / csv if csv else _latest(ROOT)
    if path is None or not path.exists():
        print(
            "no sweep_opp_progress_*.csv found -- has the sweep written its first seed yet?"
        )
        return 1
    if out is None:
        out = f"opp_study_{path.stem.removeprefix('sweep_opp_progress_')}.png"
    rows = _read(path)
    if not rows:
        print(f"{path} is empty")
        return 1

    n = int(rows[0]["n_seeds"])
    cond = _conditional(path.with_name(path.name.replace(".csv", "_runs.csv")))
    pace = _solo_pace(ROOT)
    # Label the axis in seconds when the solo measurement exists -- a progress weight means
    # nothing to a reader, an opponent lap time does. Falls back to the weight otherwise.
    if pace and all(r["value"] in pace for r in rows):
        x = [pace[r["value"]] for r in rows]
        xlabel = "opponent lap time, single agent (s)"
    else:
        x = [r["value"] for r in rows]
        xlabel = "opponent progress weight (pace)"
        logger.warning("no opponent_solo_pace.csv -- run scripts/opponent_solo_pace.py")

    fig, (ax_pct, ax_t) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax_pct.errorbar(
        x,
        [r["our_success_pct"] for r in rows],
        # yerr=[r["our_success_se_pp"] for r in rows],
        fmt="o-",
        capsize=3,
        color=FINISH,
        label="our finish rate (of all runs)",
    )
    ax_pct.errorbar(
        x,
        [cond[r["value"]]["pct"] for r in rows],
        # yerr=[cond[r["value"]]["se"] for r in rows],
        fmt="o-",
        capsize=3,
        color=OVERTAKE,
        label="overtake (of runs we finished)",
    )
    ax_pct.set_ylabel("percentage")
    ax_pct.set_ylim(-5, 105)
    ax_pct.set_title(f"Our finish rate vs overtaking (n = {n})")
    ax_pct.legend(fontsize=8)

    ax_t.plot(x, [r["our_avg_time"] for r in rows], "o-", color=FINISH)
    ax_t.set_ylabel("our lap time (s)")
    ax_t.set_title("Our speed")

    for ax in (ax_pct, ax_t):
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"Opponent pace sweep - ego pinned at progress 1.5, n = {n} seeds", y=1.0
    )
    fig.tight_layout()
    fig.savefig(ROOT / out, dpi=150, bbox_inches="tight")
    print(f"read  {path.name}  (n = {n} seeds per point)\nplot -> {ROOT / out}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    raise SystemExit(fire.Fire(plot))
