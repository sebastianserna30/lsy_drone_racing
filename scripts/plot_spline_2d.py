"""Top-down 2D view of the spline vs PPO path for quick visual tuning.

Loads track geometry from an existing data file (no full sim needed) and
re-instantiates SplinePlanner with the current code so you can iterate
on waypoint/detour logic and immediately see the result.

Usage:
    pixi run python scripts/plot_spline_2d.py
    pixi run python scripts/plot_spline_2d.py --data /tmp/logs/mppi_data.npz
                                              --ppo /tmp/logs/ppo_data.npz
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("SCIPY_ARRAY_API", "1")  # must be set before scipy is imported

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from scipy.spatial.transform import Rotation


def gate_endpoints_2d(
    pos: np.ndarray, quat: np.ndarray, half_width: float = 0.20
) -> tuple[np.ndarray, np.ndarray]:
    """Return the two endpoints of the gate opening projected onto the XY plane."""
    rot = Rotation.from_quat(quat).as_matrix()
    gate_y_world = rot[:, 1]  # gate Y-axis = width direction
    gate_y_xy = gate_y_world[:2]
    norm = np.linalg.norm(gate_y_xy)
    if norm > 1e-6:
        gate_y_xy /= norm
    return pos[:2] - half_width * gate_y_xy, pos[:2] + half_width * gate_y_xy


def main(
    data: str = "/tmp/logs/mppi_data.npz",
    ppo: str = "/tmp/logs/ppo_data.npz",
    t_total: float = 6.0,
    clearance: float = 0.20,
    start: tuple[float, float, float] = (-1.5, 0.75, 0.01),
    out: str | None = None,
) -> None:
    """Plot a top-down 2D view of the spline trajectory versus the PPO reference path."""
    data_path = Path(data)
    ppo_path = Path(ppo)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_path}. Run a sim with LOG_DRONE_DATA first."
        )

    d = np.load(data_path, allow_pickle=False)
    gates_pos = d["gates_pos"]
    gates_quat = d["gates_quat"]
    obstacles_pos = d["obstacles_pos"] if "obstacles_pos" in d else np.zeros((0, 3))

    start_pos = np.array(start)
    obs_dict = {"gates_pos": gates_pos, "gates_quat": gates_quat}

    # ── Build spline ──────────────────────────────────────────────────────────
    from lsy_drone_racing.control.spline_planner import SplinePlanner

    planner_clean = SplinePlanner(start_pos, obs_dict, t_total=t_total)
    planner_oa = SplinePlanner(
        start_pos, obs_dict, t_total=t_total,
        obstacles_pos=obstacles_pos if len(obstacles_pos) else None,
        clearance=clearance,
    )

    spline_clean = planner_clean.get_trajectory(400)
    spline_oa = planner_oa.get_trajectory(400)
    wps_oa = planner_oa.waypoints

    # ── Load PPO path ─────────────────────────────────────────────────────────
    ppo_pos = None
    if ppo_path.exists():
        ppo_pos = np.load(ppo_path)["pos"]

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_aspect("equal")
    ax.set_title("Top-down view — spline vs PPO", fontsize=13)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, alpha=0.25)

    # PPO reference
    if ppo_pos is not None:
        ax.plot(
            ppo_pos[:, 0], ppo_pos[:, 1], color="magenta", lw=1.8,
            label="PPO (optimal)", zorder=3,
        )

    # Spline without obstacle avoidance (dashed, for comparison)
    ax.plot(
        spline_clean[:, 0], spline_clean[:, 1], "g--", lw=1.2, alpha=0.5,
        label="spline (no OA)", zorder=2,
    )

    # Spline with obstacle avoidance
    ax.plot(spline_oa[:, 0], spline_oa[:, 1], "g-", lw=2.0, label="spline (with OA)", zorder=3)

    # Waypoints (with OA)
    ax.scatter(wps_oa[:, 0], wps_oa[:, 1], c="green", s=30, zorder=5)
    # Mark detour waypoints differently
    for wp in wps_oa:
        if not any(np.allclose(wp, w, atol=0.01) for w in planner_clean.waypoints):
            ax.scatter(wp[0], wp[1], c="lime", s=80, marker="*", zorder=6, label="detour wp")

    # Start
    ax.scatter(*start_pos[:2], c="black", s=80, zorder=7, label="start")
    ax.annotate("start", start_pos[:2], textcoords="offset points", xytext=(5, 5), fontsize=8)

    # Gates
    for i, (gpos, gquat) in enumerate(zip(gates_pos, gates_quat)):
        p1, p2 = gate_endpoints_2d(gpos, gquat)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "b-", lw=4, solid_capstyle="round", zorder=4)
        ax.annotate(
            f"G{i}\nz={gpos[2]:.2f}m", gpos[:2],
            textcoords="offset points", xytext=(6, 4), fontsize=8, color="blue",
        )

    # Obstacles (vertical poles — show XY footprint as circle)
    obs_drawn = False
    for j, obs in enumerate(obstacles_pos):
        circ = Circle(obs[:2], radius=0.10, color="red", alpha=0.35, zorder=4)
        ax.add_patch(circ)
        ax.annotate(
            f"O{j}", obs[:2], textcoords="offset points", xytext=(4, 4), fontsize=8, color="red"
        )
        if not obs_drawn:
            ax.scatter(*obs[:2], c="red", s=0, label="obstacle (r=0.10m)")  # for legend
            obs_drawn = True

    # Track bounds
    ax.set_xlim(-2.7, 2.0)
    ax.set_ylim(-1.7, 1.7)

    # Deduplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, lbl in zip(handles, labels):
        if lbl not in seen:
            seen[lbl] = h
    ax.legend(seen.values(), seen.keys(), fontsize=8, loc="upper left")

    plt.tight_layout()
    out_path = Path(out) if out else data_path.parent / "spline_2d.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", default="/tmp/logs/mppi_data.npz",
        help="NPZ with gates/obstacles (from a prior sim run)",
    )
    parser.add_argument("--ppo", default="/tmp/logs/ppo_data.npz", help="NPZ with PPO positions")
    parser.add_argument("--t-total", type=float, default=6.0, help="Spline total time budget")
    parser.add_argument(
        "--clearance", type=float, default=0.20, help="Obstacle detour clearance (m)"
    )
    parser.add_argument(
        "--start", type=float, nargs=3, default=[-1.5, 0.75, 0.01], metavar=("X", "Y", "Z")
    )
    parser.add_argument("--out", default=None, help="Output PNG path")
    args = parser.parse_args()
    main(
        data=args.data, ppo=args.ppo, t_total=args.t_total,
        clearance=args.clearance, start=tuple(args.start), out=args.out,
    )
