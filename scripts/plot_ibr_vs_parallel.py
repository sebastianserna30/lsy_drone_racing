"""Compare IBR against parallel rollouts on the opponent-pace sweep.

Two panels, one question each: does best-response iteration change how often we FINISH, and does
it change how often we OVERTAKE. Both arms ran the same grid on the same paired seeds with only
``ibr_iters`` differing (3 vs -1), so the two curves are directly comparable point by point.

Colour identifies the arm and is consistent across both panels; marker and line style repeat that
identity, so the arms stay distinguishable in greyscale or with colour-vision deficiency.

Run as:

    $ python scripts/plot_ibr_vs_parallel.py
    $ python scripts/plot_ibr_vs_parallel.py --conditional False
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

IBR_CSV = "sweep_opp_progress_2026-08-05_15-35-47.csv"
PAR_CSV = "sweep_opp_progress_2026-08-05_20-52-17.csv"
# Validated as a categorical pair: CVD separation dE 25.2, both above the chroma floor and 3:1
# contrast on white. Marker and dash carry the same identity so colour is never load-bearing.
ARMS = (
    ("IBR", IBR_CSV, "#1A73C8", "o-"),
    ("Parallel rollouts", PAR_CSV, "#B3560B", "o-"),
)


def _read(path: Path) -> list[dict]:
    with path.open() as fh:
        return [
            {
                k: float(v) if v not in ("", "nan") else float("nan")
                for k, v in row.items()
            }
            for row in csv_mod.DictReader(fh)
        ]


def _conditional(runs_path: Path) -> dict[float, float]:
    """Overtakes as a percentage of the runs we FINISHED, per parameter value."""
    by_value: dict[float, list[tuple[bool, bool]]] = {}
    with runs_path.open() as fh:
        for row in csv_mod.DictReader(fh):
            by_value.setdefault(float(row["value"]), []).append(
                (row["our_time"] not in ("", "nan"), bool(int(row["ego_first"])))
            )
    out = {}
    for value, runs in by_value.items():
        n_fin = sum(1 for f, _ in runs if f)
        n_ot = sum(1 for _, o in runs if o)
        out[value] = 100.0 * n_ot / n_fin if n_fin else float("nan")
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


def plot(
    out: str = "ibr_vs_parallel.png",
    conditional: bool = True,
    ibr_csv: str = IBR_CSV,
    par_csv: str = PAR_CSV,
) -> int:
    """Draw finish rate and overtake rate for both arms, side by side.

    Args:
        out: output image path, relative to the repo root.
        conditional: denominator for the overtake panel. True divides by the runs that arm
            FINISHED; False divides by ALL its runs, so crashes count against it. The arms have
            very different crash rates, so this choice changes which arm looks better -- see the
            table printed alongside.
        ibr_csv: summary CSV for the IBR arm.
        par_csv: summary CSV for the parallel-rollout arm.

    Returns:
        Process exit code.
    """
    pace = _solo_pace(ROOT)
    series = []
    for label, name, colour, style in (
        (ARMS[0][0], ibr_csv, ARMS[0][2], ARMS[0][3]),
        (ARMS[1][0], par_csv, ARMS[1][2], ARMS[1][3]),
    ):
        path = ROOT / name
        if not path.exists():
            print(f"missing {path.name}")
            return 1
        rows = _read(path)
        cond = _conditional(path.with_name(path.name.replace(".csv", "_runs.csv")))
        series.append(
            {
                "label": label,
                "colour": colour,
                "style": style,
                "n": int(rows[0]["n_seeds"]),
                "x": [pace.get(r["value"], r["value"]) for r in rows],
                "finish": [r["our_success_pct"] for r in rows],
                "overtake": [
                    cond[r["value"]] if conditional else r["overtake_pct"] for r in rows
                ],
                "values": [r["value"] for r in rows],
            }
        )

    xlabel = (
        "opponent lap time, single agent (s)"
        if pace
        else "opponent progress weight (pace)"
    )
    if not pace:
        logger.warning("no opponent_solo_pace.csv -- run scripts/opponent_solo_pace.py")
    denom = "of runs we finished" if conditional else "of all runs"

    fig, (ax_f, ax_o) = plt.subplots(1, 2, figsize=(12, 4.4))
    for s in series:
        ax_f.plot(
            s["x"],
            s["finish"],
            s["style"],
            color=s["colour"],
            label=s["label"],
            lw=2,
            ms=7,
        )
        ax_o.plot(
            s["x"],
            s["overtake"],
            s["style"],
            color=s["colour"],
            label=s["label"],
            lw=2,
            ms=7,
        )

    ax_f.set_ylabel("finish rate (%)")
    ax_f.set_title("Finishing")
    ax_o.set_ylabel(f"overtake rate (%, {denom})")
    ax_o.set_title("Overtaking")
    for ax in (ax_f, ax_o):
        ax.set_xlabel(xlabel)
        ax.set_ylim(
            -5, 105
        )  # same scale in both panels, so the two are comparable by eye
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    n = series[0]["n"]
    fig.suptitle(
        f"Iterative best response vs parallel rollouts (n = {n} paired seeds)", y=1.0
    )
    fig.tight_layout()
    fig.savefig(ROOT / out, dpi=150, bbox_inches="tight")

    # The numbers behind the picture: a chart alone is not an accessible presentation of a result.
    print(f"plot -> {ROOT / out}   (overtake {denom})\n")
    hdr = f"{'opp lap':>8} | " + " | ".join(f"{s['label']:^22}" for s in series)
    print(hdr)
    print(
        f"{'(s)':>8} | " + " | ".join(f"{'finish':>10}{'overtake':>12}" for _ in series)
    )
    order = sorted(
        range(len(series[0]["x"])), key=lambda i: series[0]["x"][i], reverse=True
    )
    for i in order:
        line = f"{series[0]['x'][i]:>8.2f} |"
        for s in series:
            line += f" {s['finish'][i]:>9.0f}% {s['overtake'][i]:>10.0f}% |"
        print(line)
    for s in series:
        print(
            f"  {s['label']:20s} mean finish {sum(s['finish']) / len(s['finish']):5.1f}%   "
            f"mean overtake {sum(s['overtake']) / len(s['overtake']):5.1f}%"
        )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    raise SystemExit(fire.Fire(plot))
