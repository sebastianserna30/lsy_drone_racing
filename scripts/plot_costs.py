"""Plot the MPPI per-mode cost breakdown collected via LOG_DRONE_DATA.

Usage:
    python scripts/plot_costs.py [--log-dir /tmp/logs] [--mode winner|all]

Expects:
    <log-dir>/mppi_data.npz   (from trajectory_mppi.py with LOG_DRONE_DATA set)

Each cost term is logged per control step as a length-K vector: the full-horizon cost of
each of the K parallel modes' best sample (the trajectory that mode proposes). This is the
signal the MPPI optimizer scores, so it's what you want for cost engineering / weight tuning.

    --mode winner  (default): break down the executed (winning) mode's cost over time.
    --mode means:             side-by-side comparison of the K means — cost spread (noise
                              diagnostic) and which mean is executed each step (smoothing).
    --mode all:               per-term grid, one line per mode.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Cost terms in display order, with a colour each. Missing terms are skipped, so the
# single-agent log (no opp_drone) and multi-agent log share this list.
COST_TERMS = [
    # MPCC tracking terms (mppi++): contour/lag/progress replace the old pos/vel tracking.
    ("cost_lag", "lag", "tab:red"),
    ("cost_contour", "contour", "tab:pink"),
    ("cost_progress", "progress", "tab:green"),
    # legacy pos/vel tracking terms (kept so older logs still plot; skipped if absent)
    ("cost_pos", "pos", "tab:red"),
    ("cost_z", "z", "tab:green"),
    ("cost_vel", "vel", "tab:orange"),
    ("cost_ang_vel", "ang_vel", "tab:olive"),
    ("cost_ang_acc", "ang_acc", "tab:brown"),
    ("cost_tilt", "tilt", "tab:blue"),
    ("cost_thrust", "thrust", "tab:cyan"),
    ("cost_yaw", "yaw", "tab:purple"),
    ("cost_obstacle", "obstacle", "magenta"),
    ("cost_gate_obstacle", "gate_obstacle", "deeppink"),
    ("cost_opp_drone", "opp_drone", "black"),
    ("cost_floor", "floor", "gray"),
]


def load(path: Path) -> dict | None:
    """Load an NPZ file and return its contents as a dict, or None if missing."""
    if not path.exists():
        print(f"[warn] {path} not found — skipping")
        return None
    return dict(np.load(path, allow_pickle=False))


def winner_series(arr_tk: np.ndarray, best_idx: np.ndarray) -> np.ndarray:
    """Given a (T, K) array and the per-step winning mode index (T,), return (T,)."""
    return arr_tk[np.arange(arr_tk.shape[0]), best_idx]


def mark_gates(ax: plt.Axes, t: np.ndarray, target_gate: np.ndarray) -> None:
    """Draw vertical lines at gate crossings."""
    for step in np.where(np.diff(target_gate) != 0)[0]:
        ax.axvline(float(t[step]), color="blue", lw=0.8, ls="--", alpha=0.4)
        ax.text(
            float(t[step]) + 0.02,
            ax.get_ylim()[1] * 0.9,
            f"→G{int(target_gate[step + 1])}",
            fontsize=7,
            color="blue",
        )


def plot_winner(mppi: dict, present: list, t: np.ndarray, best: np.ndarray, tg):
    """Break down the executed (winning) mode's cost over time."""
    # winner term series, each (T,)
    win = {k: winner_series(np.asarray(mppi[k], dtype=float), best) for k, _, _ in present}

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("MPPI cost breakdown — executed (winning) mode", fontsize=14)

    # ── 1. Each term, linear ──────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.set_title("Per-term cost over time")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("full-horizon cost")
    for k, lbl, c in present:
        ax.plot(t, win[k], color=c, lw=1.0, label=lbl)
    if tg is not None:
        mark_gates(ax, t, tg)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # ── 2. Each term, log scale ───────────────────────────────────────────────
    ax = axes[0, 1]
    ax.set_title("Per-term cost (log scale)")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("full-horizon cost (log)")
    for k, lbl, c in present:
        ax.plot(t, np.clip(win[k], 1e-6, None), color=c, lw=1.0, label=lbl)
    ax.set_yscale("log")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3, which="both")

    # ── 3. Stacked contribution ───────────────────────────────────────────────
    ax = axes[1, 0]
    ax.set_title("Stacked cost contribution")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("full-horizon cost")
    ax.stackplot(
        t,
        *[win[k] for k, _, _ in present],
        labels=[lbl for _, lbl, _ in present],
        colors=[c for _, _, c in present],
        alpha=0.8,
    )
    if tg is not None:
        mark_gates(ax, t, tg)
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.grid(True, alpha=0.3)

    # ── 4. Total cost: winner vs all modes ────────────────────────────────────
    ax = axes[1, 1]
    ax.set_title("Total cost — winner vs all modes")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("full-horizon cost")
    totals = np.asarray(mppi["cost_total"], dtype=float)  # (T, K)
    for k in range(totals.shape[1]):
        ax.plot(t, totals[:, k], color="gray", lw=0.6, alpha=0.4)
    ax.plot(t, winner_series(totals, best), "k-", lw=1.6, label="executed mode")
    ax.plot(
        t,
        np.sum([win[k] for k, _, _ in present], axis=0),
        "r--",
        lw=1.0,
        alpha=0.7,
        label="winner Σterms (check)",
    )
    if tg is not None:
        mark_gates(ax, t, tg)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_all_modes(mppi: dict, present: list, t: np.ndarray, best: np.ndarray, tg):
    """Compare the K modes: total cost and each term, one line per mode."""
    totals = np.asarray(mppi["cost_total"], dtype=float)  # (T, K)
    n_modes = totals.shape[1]
    cmap = plt.get_cmap("tab10")

    n = len(present) + 1
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 3.2 * nrow), squeeze=False)
    fig.suptitle(f"MPPI per-mode cost comparison (K={n_modes} modes)", fontsize=14)
    axflat = axes.flatten()

    # total cost first
    ax = axflat[0]
    ax.set_title("total")
    for k in range(n_modes):
        ax.plot(t, totals[:, k], color=cmap(k % 10), lw=0.9, label=f"mode {k}")
    if tg is not None:
        mark_gates(ax, t, tg)
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylabel("full-horizon cost")

    # one panel per term
    for ax, (key, lbl, _) in zip(axflat[1:], present):
        data = np.asarray(mppi[key], dtype=float)  # (T, K)
        ax.set_title(lbl)
        for k in range(n_modes):
            ax.plot(t, data[:, k], color=cmap(k % 10), lw=0.9)
        ax.grid(True, alpha=0.3)
    for ax in axflat[n:]:
        ax.axis("off")

    plt.tight_layout()
    return fig


def plot_means(mppi: dict, present: list, t: np.ndarray, best: np.ndarray, tg):
    """Side-by-side comparison of the K means (cost spread + selection over time).

    Two diagnostics:
    - cost spread across means  -> if the means barely differ, sampling noise is too low
      (modes collapsed); a healthy spread means the modes explore distinct trajectories.
    - which mean is executed each step -> rapid switching suggests more smoothing is needed.
    """
    totals = np.asarray(mppi["cost_total"], dtype=float)  # (T, K)
    n_modes = totals.shape[1]
    cmap = plt.get_cmap("tab10")
    mcol = [cmap(k % 10) for k in range(n_modes)]

    counts = np.bincount(best, minlength=n_modes)
    frac = counts / max(len(best), 1)
    n_switch = int(np.sum(np.diff(best) != 0))
    switch_rate = n_switch / max(len(best) - 1, 1)

    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(4.2 * n_modes, 11))
    gs = gridspec.GridSpec(3, n_modes, height_ratios=[2.2, 1.0, 2.6], hspace=0.32, wspace=0.25)
    fig.suptitle(f"MPPI mean comparison (K={n_modes})", fontsize=14)

    # ── Row 0: total cost per mean over time (cost spread → noise diagnostic) ──
    ax_tot = fig.add_subplot(gs[0, :])
    ax_tot.set_title("Total full-horizon cost per mean  (spread → sampling-noise diagnostic)")
    ax_tot.set_ylabel("cost")
    for k in range(n_modes):
        ax_tot.plot(t, totals[:, k], color=mcol[k], lw=1.1, label=f"mean {k} ({frac[k] * 100:.0f}%)")
    ax_tot.set_yscale("log")
    if tg is not None:
        mark_gates(ax_tot, t, tg)
    ax_tot.legend(fontsize=8, ncol=n_modes, loc="upper right")
    ax_tot.grid(True, alpha=0.3, which="both")

    # ── Row 1: which mean was executed each step (switching → smoothing diagnostic) ──
    ax_sel = fig.add_subplot(gs[1, :], sharex=ax_tot)
    ax_sel.set_title(
        f"Executed mean over time  —  {n_switch} switches "
        f"({switch_rate * 100:.0f}% of steps)  → smoothing diagnostic"
    )
    ax_sel.set_ylabel("mean idx")
    ax_sel.set_xlabel("t (s)")
    ax_sel.step(t, best, where="post", color="0.6", lw=0.8, zorder=1)
    ax_sel.scatter(t, best, c=[mcol[k] for k in best], s=14, zorder=2)
    ax_sel.set_yticks(range(n_modes))
    ax_sel.set_ylim(-0.5, n_modes - 0.5)
    ax_sel.grid(True, alpha=0.3, axis="x")

    # ── Row 2: per-mean stacked term breakdown, side by side (shared y) ───────
    stacks = [np.array([np.asarray(mppi[key], dtype=float)[:, k] for key, _, _ in present])
              for k in range(n_modes)]  # list of (n_terms, T)
    ymax = max((s.sum(axis=0).max() for s in stacks), default=1.0) * 1.05
    for k in range(n_modes):
        ax = fig.add_subplot(gs[2, k], sharex=ax_tot)
        ax.set_title(f"mean {k}", color=mcol[k])
        ax.stackplot(
            t, *stacks[k],
            labels=[lbl for _, lbl, _ in present],
            colors=[c for _, _, c in present],
            alpha=0.85,
        )
        ax.set_ylim(0, ymax)
        ax.grid(True, alpha=0.3)
        if k == 0:
            ax.set_ylabel("stacked cost")
            ax.legend(fontsize=6, ncol=2, loc="upper right")
        else:
            ax.tick_params(labelleft=False)
        ax.set_xlabel("t (s)")

    return fig


def main(log_dir: str = "/tmp/logs", mode: str = "winner") -> None:
    """Plot the MPPI per-mode cost breakdown from mppi_data.npz in log_dir."""
    d = Path(log_dir)
    mppi = load(d / "mppi_data.npz")
    if mppi is None:
        print("No mppi_data.npz found. Run a sim with LOG_DRONE_DATA set first.")
        return

    if "best_mode_idx" not in mppi or "cost_total" not in mppi:
        print("Log lacks per-mode cost fields. Re-run the sim with the updated controller.")
        return

    t = np.asarray(mppi["t"])
    best = np.asarray(mppi["best_mode_idx"]).astype(int)
    tg = mppi.get("target_gate")
    present = [(k, lbl, c) for k, lbl, c in COST_TERMS if k in mppi]

    if mode == "means":
        plot_means(mppi, present, t, best, tg)
        out = d / "costs_means.png"
    elif mode == "all":
        plot_all_modes(mppi, present, t, best, tg)
        out = d / "costs_modes.png"
    else:
        plot_winner(mppi, present, t, best, tg)
        out = d / "costs.png"

    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="/tmp/logs", help="Directory with mppi_data.npz")
    parser.add_argument(
        "--mode",
        default="winner",
        choices=["winner", "means", "all"],
        help="winner: breakdown of executed mode; means: side-by-side mean comparison "
        "(cost spread + selection over time); all: per-term grid across modes",
    )
    args = parser.parse_args()
    main(args.log_dir, args.mode)
