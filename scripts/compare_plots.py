"""Compare MPPI vs PPO trajectory data collected via LOG_DRONE_DATA.

Usage:
    python scripts/compare_plots.py [--log-dir /tmp/logs]

Expects:
    <log-dir>/mppi_data.npz   (from trajectory_mppi.py with LOG_DRONE_DATA set)
    <log-dir>/ppo_data.npz    (from sbx_song_logger.py with LOG_DRONE_DATA set)

Either file is optional — plots will show whatever is available.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def load(path: Path) -> dict | None:
    """Load an NPZ file and return its contents as a dict, or None if missing."""
    if not path.exists():
        print(f"[warn] {path} not found — skipping")
        return None
    d = np.load(path, allow_pickle=False)
    return dict(d)


def gate_normal(quat_xyzw: np.ndarray) -> np.ndarray:
    """Return the gate's X-axis (normal through the aperture) in world frame."""
    x, y, z, w = quat_xyzw
    return np.array([
        1 - 2 * (y * y + z * z),
        2 * (x * y + w * z),
        2 * (x * z - w * y),
    ])


def draw_gate(ax: plt.Axes, pos: np.ndarray, quat: np.ndarray, half_size: float = 0.20):
    """Draw a gate as a square ring in 3D."""
    from scipy.spatial.transform import Rotation as R
    rot = R.from_quat(quat).as_matrix()
    y_ax = rot[:, 1]
    z_ax = rot[:, 2]
    corners = [
        pos + half_size * y_ax + half_size * z_ax,
        pos - half_size * y_ax + half_size * z_ax,
        pos - half_size * y_ax - half_size * z_ax,
        pos + half_size * y_ax - half_size * z_ax,
        pos + half_size * y_ax + half_size * z_ax,
    ]
    corners = np.array(corners)
    ax.plot(corners[:, 0], corners[:, 1], corners[:, 2], "b-", lw=2)


def main(log_dir: str = "/tmp/logs") -> None:
    """Plot comparison of MPPI and PPO trajectory data from log_dir."""
    d = Path(log_dir)
    mppi = load(d / "mppi_data.npz")
    ppo = load(d / "ppo_data.npz")

    if mppi is None and ppo is None:
        print("No data files found. Run sims with LOG_DRONE_DATA set first.")
        return

    gates_pos = (mppi or ppo)["gates_pos"]
    gates_quat = (mppi or ppo)["gates_quat"]
    obstacles_pos = mppi["obstacles_pos"] if mppi is not None and "obstacles_pos" in mppi else None

    fig = plt.figure(figsize=(18, 11))
    fig.suptitle("MPPI vs PPO — trajectory comparison", fontsize=14)

    # ── 1. 3D trajectory ──────────────────────────────────────────────────────
    ax3d = fig.add_subplot(2, 3, 1, projection="3d")
    ax3d.set_title("3D trajectory")
    ax3d.set_xlabel("x (m)")
    ax3d.set_ylabel("y (m)")
    ax3d.set_zlabel("z (m)")

    if mppi is not None:
        p = mppi["pos"]
        ax3d.plot(p[:, 0], p[:, 1], p[:, 2], "r-", lw=1.2, label="MPPI actual")
        s = mppi["spline"]
        ax3d.plot(s[:, 0], s[:, 1], s[:, 2], "g--", lw=1, alpha=0.7, label="MPPI spline")
        wp = mppi["waypoints"]
        ax3d.scatter(wp[:, 0], wp[:, 1], wp[:, 2], c="g", s=20, zorder=5, label="waypoints")

    if ppo is not None:
        p = ppo["pos"]
        ax3d.plot(p[:, 0], p[:, 1], p[:, 2], "m-", lw=1.2, label="PPO actual")

    # Gates
    for i, (gpos, gquat) in enumerate(zip(gates_pos, gates_quat)):
        draw_gate(ax3d, gpos, gquat)
        ax3d.text(gpos[0], gpos[1], gpos[2] + 0.08, f"G{i}", fontsize=7, color="blue")

    # Obstacles (vertical poles — show as red X at XY position, across altitude range)
    if obstacles_pos is not None:
        for j, obs in enumerate(obstacles_pos):
            ax3d.plot([obs[0], obs[0]], [obs[1], obs[1]], [0.0, 2.0], "r-", lw=2, alpha=0.5)
            ax3d.text(obs[0], obs[1], 2.05, f"O{j}", fontsize=6, color="red")

    # Start marker
    start = mppi["pos"][0] if mppi is not None else ppo["pos"][0]
    ax3d.scatter(*start, c="k", s=60, zorder=10, label="start")
    ax3d.legend(fontsize=7)

    # ── 2. Speed |v| over time ───────────────────────────────────────────────
    ax_spd = fig.add_subplot(2, 3, 2)
    ax_spd.set_title("Speed over time")
    ax_spd.set_xlabel("t (s)")
    ax_spd.set_ylabel("|v| (m/s)")

    def _gate_crossing_times(data: dict) -> list[tuple[int, float, float]]:
        """Return (gate_idx, t, speed) for each gate passage."""
        tg, t, vel = data["target_gate"], data["t"], data["vel"]
        spd = np.linalg.norm(vel, axis=1)
        crossings = []
        for step in np.where(np.diff(tg) != 0)[0]:
            gate_idx = int(tg[step])  # gate just passed
            crossings.append((gate_idx, float(t[step]), float(spd[step])))
        return crossings

    if mppi is not None:
        spd = np.linalg.norm(mppi["vel"], axis=1)
        ax_spd.plot(mppi["t"], spd, "r-", label="MPPI actual")
        des_spd = np.linalg.norm(mppi["des_vel"], axis=1)
        ax_spd.plot(mppi["t"], des_spd, "g--", alpha=0.7, label="MPPI reference")
        for gate_idx, t_cross, spd_cross in _gate_crossing_times(mppi):
            ax_spd.scatter(t_cross, spd_cross, marker="*", s=180, color="red", zorder=5)
            ax_spd.text(t_cross + 0.04, spd_cross + 0.05, f"G{gate_idx}", fontsize=7, color="red")

    if ppo is not None:
        spd = np.linalg.norm(ppo["vel"], axis=1)
        ax_spd.plot(ppo["t"], spd, "m-", label="PPO actual")
        for gate_idx, t_cross, spd_cross in _gate_crossing_times(ppo):
            ax_spd.scatter(t_cross, spd_cross, marker="*", s=180, color="purple", zorder=5)
            ax_spd.text(
                t_cross + 0.04, spd_cross + 0.05, f"G{gate_idx}", fontsize=7, color="purple"
            )

    ax_spd.legend(fontsize=8)
    ax_spd.grid(True, alpha=0.3)

    # ── 3. Altitude over time ─────────────────────────────────────────────────
    ax_z = fig.add_subplot(2, 3, 3)
    ax_z.set_title("Altitude over time")
    ax_z.set_xlabel("t (s)")
    ax_z.set_ylabel("z (m)")

    if mppi is not None:
        ax_z.plot(mppi["t"], mppi["pos"][:, 2], "r-", label="MPPI actual")
        ax_z.plot(mppi["t"], mppi["des_pos"][:, 2], "g--", alpha=0.7, label="MPPI reference")

    if ppo is not None:
        ax_z.plot(ppo["t"], ppo["pos"][:, 2], "m-", label="PPO actual")

    # Gate altitudes
    for i, gpos in enumerate(gates_pos):
        ax_z.axhline(gpos[2], color="blue", lw=0.7, ls=":", alpha=0.6)
        ax_z.text(
            0.01, gpos[2] + 0.02, f"G{i}", fontsize=7, color="blue",
            transform=ax_z.get_yaxis_transform(),
        )

    ax_z.legend(fontsize=8)
    ax_z.grid(True, alpha=0.3)

    # ── 4. Roll and pitch commands ─────────────────────────────────────────────
    ax_rp = fig.add_subplot(2, 3, 4)
    ax_rp.set_title("Roll / pitch commands")
    ax_rp.set_xlabel("t (s)")
    ax_rp.set_ylabel("angle (rad)")

    if mppi is not None:
        ax_rp.plot(mppi["t"], mppi["action"][:, 0], "r-", alpha=0.8, label="MPPI roll")
        ax_rp.plot(mppi["t"], mppi["action"][:, 1], "r--", alpha=0.8, label="MPPI pitch")

    if ppo is not None:
        ax_rp.plot(ppo["t"], ppo["action"][:, 0], "m-", alpha=0.8, label="PPO roll")
        ax_rp.plot(ppo["t"], ppo["action"][:, 1], "m--", alpha=0.8, label="PPO pitch")

    ax_rp.axhline(np.pi / 2, color="k", lw=0.7, ls=":", label="±π/2 limit")
    ax_rp.axhline(-np.pi / 2, color="k", lw=0.7, ls=":")
    ax_rp.legend(fontsize=7)
    ax_rp.grid(True, alpha=0.3)

    # ── 5. MPPI position tracking error ──────────────────────────────────────
    ax_err = fig.add_subplot(2, 3, 5)
    ax_err.set_title("MPPI position tracking error")
    ax_err.set_xlabel("t (s)")
    ax_err.set_ylabel("error (m)")

    if mppi is not None:
        err = np.linalg.norm(mppi["pos"] - mppi["des_pos"], axis=1)
        err_x = np.abs(mppi["pos"][:, 0] - mppi["des_pos"][:, 0])
        err_y = np.abs(mppi["pos"][:, 1] - mppi["des_pos"][:, 1])
        err_z = np.abs(mppi["pos"][:, 2] - mppi["des_pos"][:, 2])
        ax_err.plot(mppi["t"], err, "r-", lw=1.5, label="|total|")
        ax_err.plot(mppi["t"], err_x, alpha=0.5, label="x")
        ax_err.plot(mppi["t"], err_y, alpha=0.5, label="y")
        ax_err.plot(mppi["t"], err_z, alpha=0.5, label="z")
        ax_err.legend(fontsize=8)
    else:
        ax_err.text(0.5, 0.5, "No MPPI data", ha="center", va="center", transform=ax_err.transAxes)

    ax_err.grid(True, alpha=0.3)

    # ── 6. MPPI cost breakdown + obstacle proximity ───────────────────────────
    ax_cost = fig.add_subplot(2, 3, 6)
    ax_cost.set_title("MPPI cost breakdown + obstacle proximity")
    ax_cost.set_xlabel("t (s)")
    ax_cost.set_ylabel("cost (actual state)")

    if mppi is not None:
        t = mppi["t"]
        if "cost_pos" in mppi:
            ax_cost.plot(t, mppi["cost_pos"], color="red", lw=1, label="pos cost (×40)")
            ax_cost.plot(t, mppi["cost_z"], color="green", lw=1, label="z cost (×80)")
            ax_cost.plot(t, mppi["cost_vel"], color="orange", lw=1, label="vel cost (×1)")

        # Gate crossing markers
        tg = mppi["target_gate"]
        for step in np.where(np.diff(tg) != 0)[0]:
            ax_cost.axvline(t[step], color="blue", lw=1, ls="--", alpha=0.5)
            ax_cost.text(
                t[step] + 0.02, ax_cost.get_ylim()[1] * 0.85, f"→G{tg[step+1]}",
                fontsize=7, color="blue",
            )

        # Obstacle proximity on secondary axis
        if "min_obs_dist" in mppi:
            ax2 = ax_cost.twinx()
            ax2.plot(
                t, mppi["min_obs_dist"], color="purple", lw=1.2, ls="--",
                label="nearest obstacle (m)",
            )
            # Collision threshold line
            ax2.axhline(0.15, color="purple", lw=0.7, ls=":", alpha=0.6)
            ax2.set_ylabel("dist to nearest obstacle (m)", color="purple")
            ax2.tick_params(axis="y", colors="purple")
            ax2.legend(fontsize=7, loc="upper right")

        ax_cost.legend(fontsize=7, loc="upper left")
    else:
        ax_cost.text(
            0.5, 0.5, "No MPPI data", ha="center", va="center", transform=ax_cost.transAxes
        )

    ax_cost.grid(True, alpha=0.3)

    plt.tight_layout()
    out = d / "comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-dir", default="/tmp/logs",
        help="Directory with mppi_data.npz and ppo_data.npz",
    )
    args = parser.parse_args()
    main(args.log_dir)
