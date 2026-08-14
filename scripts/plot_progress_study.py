"""Plot the three-arm opponent-model study: progress weight vs lap time and finish rate.

Each arm is one way the EGO models the opponent while the opponent itself is held fixed (the
open-loop spline follower at ``t_total = 7.8``), so the three curves differ in one thing only:

* ``const_vel`` -- straight-line predictor
* ``parallel``  -- joint MPPI rollout, single unilateral response (``ibr_iters = -1``)
* ``ibr``       -- iterative best response (``ibr_iters = 3``)

All arms share one seed set, so the comparison is paired at every progress value. The finish-rate
panel carries binomial error bars; at n = 40 the standard error near p = 0.5 is about 8 pp, so two
curves that overlap by less than that are not distinguishable.

Run as:

    $ python scripts/plot_progress_study.py
    $ python scripts/plot_progress_study.py --out study.png
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logger = logging.getLogger(__name__)
ROOT = Path(__file__).parents[1]

# Arm label -> the config stem it was produced from, in the order they should be drawn.
ARMS = (("const velocity", "constvel"), ("parallel rollouts", "parallel"), ("IBR", "ibr"))
COLOURS = {"const velocity": "#4C78A8", "parallel rollouts": "#F58518", "IBR": "#54A24B"}


def _latest_summary(arm: str, runs_dir: Path) -> Path | None:
    """The ``sweep_progress_*.csv`` belonging to one arm.

    The sweep script stamps its output with a timestamp but not with the arm, so the arms would
    be indistinguishable in the repo root. Each arm's own log names its summary file, which
    removes the guesswork.
    """
    marker = runs_dir / "study_logs" / f"{arm}.log"
    if not marker.exists():
        return None
    for line in marker.read_text().splitlines():
        if line.startswith("summary ->"):
            return Path(line.split("->", 1)[1].strip())
    return None


def _read(path: Path) -> list[dict]:
    with path.open() as fh:
        return [{k: float(v) if v not in ("", "nan") else float("nan") for k, v in row.items()}
                for row in csv.DictReader(fh)]


def plot(out: str = "progress_study.png", runs_dir: str | None = None) -> int:
    """Draw the two-panel comparison and write a merged CSV for the report.

    Args:
        out: output image path.
        runs_dir: directory holding the sweep CSVs and ``study_logs/``; defaults to the repo root.

    Returns:
        Process exit code.
    """
    root = Path(runs_dir) if runs_dir else ROOT
    fig, (ax_t, ax_s) = plt.subplots(1, 2, figsize=(11, 4.2))
    merged: list[dict] = []
    found = 0

    for label, arm in ARMS:
        path = _latest_summary(arm, root)
        if path is None or not path.exists():
            logger.warning("no summary CSV for arm %s -- skipping", arm)
            continue
        rows = _read(path)
        found += 1
        x = [r["value"] for r in rows]
        ax_t.plot(x, [r["our_avg_time"] for r in rows], "o-", color=COLOURS[label], label=label)
        ax_s.errorbar(
            x,
            [r["our_success_pct"] for r in rows],
            yerr=[r["our_success_se_pp"] for r in rows],
            fmt="o-",
            capsize=3,
            color=COLOURS[label],
            label=label,
        )
        for r in rows:
            merged.append({"arm": label, **r})

    if not found:
        print("no arm CSVs found yet -- the sweep is probably still running")
        return 1

    # The opponent never reacts, so its lap time is the same in every run and makes a fair
    # reference line for "did we actually pass anyone".
    opp = [r["opponent_time"] for r in merged if r["opponent_time"] == r["opponent_time"]]
    if opp:
        ax_t.axhline(sum(opp) / len(opp), ls="--", lw=1, color="grey", label="opponent")

    ax_t.set_xlabel("progress cost weight")
    ax_t.set_ylabel("our average lap time (s)")
    ax_t.set_title("Speed")
    ax_t.grid(alpha=0.3)
    ax_t.legend(fontsize=8)

    ax_s.set_xlabel("progress cost weight")
    ax_s.set_ylabel("our finish rate (%)")
    ax_s.set_title("Safety (binomial SE, n = 40)")
    ax_s.set_ylim(-5, 105)
    ax_s.grid(alpha=0.3)
    ax_s.legend(fontsize=8)

    fig.suptitle("Opponent modelling: speed/safety trade vs progress weight", y=1.0)
    fig.tight_layout()
    fig.savefig(root / out, dpi=150, bbox_inches="tight")

    # Pareto view. At a fixed progress weight the arms do not fly at the same pace, so a better
    # finish rate there may simply have been bought with lap time. Plotting the two against each
    # other separates "this model is better" (curve sits up and to the left) from "this model is
    # the same trade, relabelled" (both arms fall on one curve).
    fig2, ax_p = plt.subplots(figsize=(6, 4.6))
    for label, _ in ARMS:
        pts = [r for r in merged if r["arm"] == label]
        if not pts:
            continue
        ax_p.plot(
            [r["our_avg_time"] for r in pts],
            [r["our_success_pct"] for r in pts],
            "o-",
            color=COLOURS[label],
            label=label,
        )
        for r in pts:
            ax_p.annotate(
                f"{r['value']:g}",
                (r["our_avg_time"], r["our_success_pct"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
                color=COLOURS[label],
            )
    if opp:
        ax_p.axvline(sum(opp) / len(opp), ls="--", lw=1, color="grey", label="opponent time")
    ax_p.set_xlabel("our average lap time (s)")
    ax_p.set_ylabel("our finish rate (%)")
    ax_p.set_title("Pareto front (labels = progress weight)")
    ax_p.grid(alpha=0.3)
    ax_p.legend(fontsize=8)
    fig2.tight_layout()
    pareto_path = root / out.replace(".png", "_pareto.png")
    fig2.savefig(pareto_path, dpi=150, bbox_inches="tight")

    csv_out = root / "progress_study_merged.csv"
    with csv_out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(merged[0]))
        w.writeheader()
        w.writerows(merged)
    print(
        f"plot -> {root / out}\npareto -> {pareto_path}\n"
        f"merged CSV -> {csv_out}  ({found}/{len(ARMS)} arms)"
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    raise SystemExit(fire.Fire(plot))
