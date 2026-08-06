"""Utilities for saving simple 2D trajectory plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _extract_xy(coords: np.ndarray | list[list[float]]) -> np.ndarray:
    """Return a 2D array of xy coordinates from a 2D/3D input."""
    arr = np.asarray(coords, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim > 2:
        arr = arr.reshape(-1, arr.shape[-1])
    if arr.shape[1] >= 2:
        return arr[:, :2]
    raise ValueError("coordinates must have at least 2 columns")


def _gate_frame_positions(
    gates_pos: np.ndarray | list[list[float]],
    gates_quat: np.ndarray | list[list[float]],
) -> np.ndarray:
    """Return the two gate-frame side positions used by the MPPI controller."""
    from scipy.spatial.transform import Rotation as R

    gates_pos_arr = np.asarray(gates_pos, dtype=float)
    gates_quat_arr = np.asarray(gates_quat, dtype=float)
    if gates_pos_arr.ndim == 1:
        gates_pos_arr = gates_pos_arr.reshape(1, -1)
    if gates_quat_arr.ndim == 1:
        gates_quat_arr = gates_quat_arr.reshape(1, -1)
    if gates_pos_arr.ndim > 2:
        gates_pos_arr = gates_pos_arr.reshape(-1, gates_pos_arr.shape[-1])
    if gates_quat_arr.ndim > 2:
        gates_quat_arr = gates_quat_arr.reshape(-1, gates_quat_arr.shape[-1])

    frame_positions = []
    for gate_pos, gate_quat in zip(gates_pos_arr, gates_quat_arr):
        rotation = R.from_quat(gate_quat)
        side_axis = rotation.apply([0.0, 1.0, 0.0])
        frame_positions.append(gate_pos - 0.28 * side_axis)
        frame_positions.append(gate_pos + 0.28 * side_axis)
    return np.asarray(frame_positions, dtype=float)


def _is_multi_trajectory_input(positions: object) -> bool:
    """Return True when the input is a collection of trajectories rather than one trajectory."""
    if not isinstance(positions, (list, tuple)) or len(positions) == 0:
        return False
    first = positions[0]
    if not isinstance(first, (list, tuple, np.ndarray)) or len(first) == 0:
        return False
    first_item = first[0]
    return isinstance(first_item, (list, tuple, np.ndarray))


def _prepare_trajectory(
    trajectory: np.ndarray | list[list[float]],
    time_values: np.ndarray | list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize a single trajectory and optional per-point time values."""
    traj_arr = np.asarray(trajectory, dtype=float)
    if traj_arr.ndim == 1:
        traj_arr = traj_arr.reshape(1, -1)

    if traj_arr.shape[1] == 2:
        traj_xy = traj_arr
    elif traj_arr.shape[1] >= 3:
        traj_xy = traj_arr[:, :2]
    else:
        raise ValueError("positions must have shape (N, 2) or (N, 3)")

    if time_values is None:
        traj_times = np.arange(len(traj_xy), dtype=float)
    else:
        traj_times = np.asarray(time_values, dtype=float)
        if traj_times.ndim == 0:
            traj_times = traj_times.reshape(1)
        if len(traj_times) != len(traj_arr):
            raise ValueError("time_values must have the same length as the trajectory")

    if len(traj_xy) > 1:
        keep_mask = np.ones(len(traj_xy), dtype=bool)
        diff = np.linalg.norm(np.diff(traj_xy, axis=0), axis=1)
        keep_mask[1:] = diff > 1e-9
        traj_xy = traj_xy[keep_mask]
        traj_times = traj_times[keep_mask]

    return traj_xy, traj_times


def save_top_view_trajectory(
    positions: np.ndarray | list[list[float]] | list[np.ndarray],
    output_path: str | Path,
    gate_positions: np.ndarray | list[list[float]] | None = None,
    gate_quaternions: np.ndarray | list[list[float]] | None = None,
    obstacle_positions: np.ndarray | list[list[float]] | None = None,
    time_values: np.ndarray | list[np.ndarray] | list[float] | None = None,
    color_by_time: bool = False,
    show: bool = False,
) -> Path | None:
    """Save a 2D top-view trajectory plot as a PNG file.

    Args:
        positions: Array of drone positions with shape (N, 3) or (N, 2).
        output_path: Destination path for the PNG file.
        gate_positions: Optional gate positions to overlay on the plot.
        gate_quaternions: Optional gate orientations used to draw gate-frame points.
        obstacle_positions: Optional obstacle positions to overlay on the plot.
        time_values: Optional per-point timestamps used with ``color_by_time``.
        color_by_time: Whether to color the trajectory by elapsed time instead of a single color.
        show: Whether to display the plot interactively.

    Returns:
        The output path if the plot was written successfully, otherwise None.
    """
    if _is_multi_trajectory_input(positions):
        trajectories_xy = []
        trajectory_times = []
        if time_values is None:
            time_values_per_traj = [None] * len(positions)
        elif isinstance(time_values, (list, tuple)):
            time_values_per_traj = list(time_values)
        else:
            raise ValueError("time_values must be None, a 1D array, or a list of arrays")
        for idx, traj in enumerate(positions):
            traj_xy, traj_times = _prepare_trajectory(traj, time_values_per_traj[idx])
            trajectories_xy.append(traj_xy)
            trajectory_times.append(traj_times)
    else:
        trajectory_times = []
        positions_array = np.asarray(positions, dtype=float)
        if positions_array.ndim == 1:
            positions_array = positions_array.reshape(1, -1)
        if positions_array.shape[1] == 2:
            positions_xy = positions_array
        elif positions_array.shape[1] >= 3:
            positions_xy = positions_array[:, :2]
        else:
            raise ValueError("positions must have shape (N, 2) or (N, 3)")
        if len(positions_xy) > 1:
            keep_mask = np.ones(len(positions_xy), dtype=bool)
            diff = np.linalg.norm(np.diff(positions_xy, axis=0), axis=1)
            keep_mask[1:] = diff > 1e-9
            positions_xy = positions_xy[keep_mask]
            if len(positions_xy) == 0:
                positions_xy = positions_array[:, :2][:1]
        if time_values is None:
            traj_times = np.arange(len(positions_xy), dtype=float)
        else:
            traj_times = np.asarray(time_values, dtype=float)
            if traj_times.ndim == 0:
                traj_times = traj_times.reshape(1)
            if len(traj_times) != len(positions_array):
                raise ValueError("time_values must have the same length as the trajectory")
            if len(positions_xy) > 1:
                keep_mask = np.ones(len(traj_times), dtype=bool)
                diff = np.linalg.norm(np.diff(positions_array[:, :2], axis=0), axis=1)
                keep_mask[1:] = diff > 1e-9
                traj_times = traj_times[keep_mask]
            if len(traj_times) == 0:
                traj_times = np.array([0.0], dtype=float)
        trajectories_xy = [positions_xy]
        trajectory_times = [traj_times]

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    labels = ["trajectory opponent", "own trajectory"]

    fig, ax = plt.subplots(figsize=(6, 4))
    scalar_map = None
    if color_by_time:
        all_times = [times for times in trajectory_times if len(times) > 0]
        if len(all_times) > 0:
            all_times = np.concatenate(all_times)
            cmap = plt.cm.viridis
            norm = plt.Normalize(
                vmin=float(np.min(all_times)),
                vmax=float(np.max(all_times)),
            )
            scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            scalar_map.set_array([])
        else:
            scalar_map = None

    for idx, traj_xy in enumerate(trajectories_xy):
        if traj_xy.shape[0] > 1:
            color = ["tab:blue", "tab:green", "tab:red", "tab:purple"][idx % 4]
            label = f"trajectory {idx + 1}" if len(trajectories_xy) > 1 else "trajectory"
            label = labels[idx] if len(trajectories_xy) > 1 else "trajectory"
            if color_by_time:
                traj_times = trajectory_times[idx]
                if len(traj_times) != len(traj_xy):
                    traj_times = np.linspace(0.0, 1.0, len(traj_xy))
                if scalar_map is not None:
                    cmap = scalar_map.cmap
                    norm = scalar_map.norm
                else:
                    cmap = plt.cm.viridis
                    norm = plt.Normalize(vmin=0.0, vmax=1.0)
                for seg_idx in range(len(traj_xy) - 2):
                    segment_times = traj_times[seg_idx : seg_idx + 2]
                    segment_color = cmap(norm(float(np.mean(segment_times))))
                    ax.plot(
                        traj_xy[seg_idx : seg_idx + 2, 0],
                        traj_xy[seg_idx : seg_idx + 2, 1],
                        color=segment_color,
                        linewidth=1.8,
                    )
                ax.scatter(
                    traj_xy[0, 0],
                    traj_xy[0, 1],
                    color=cmap(norm(float(traj_times[0]))),
                    s=40,
                    zorder=3,
                )
                ax.scatter(
                    traj_xy[-2, 0],
                    traj_xy[-2, 1],
                    color=cmap(norm(float(traj_times[-1]))),
                    s=35,
                    zorder=3,
                    marker="x",
                )
            else:
                ax.plot(traj_xy[:-1, 0], traj_xy[:-1, 1], color=color, linewidth=1.8, label=label)
                ax.scatter(traj_xy[0, 0], traj_xy[0, 1], color=color, s=40, zorder=3)
                ax.scatter(traj_xy[-2, 0], traj_xy[-2, 1], color=color, s=35, zorder=3, marker="x")
        else:
            ax.scatter(traj_xy[0, 0], traj_xy[0, 1], color="tab:blue", s=40, zorder=3)

    if color_by_time and scalar_map is not None:
        cbar = fig.colorbar(scalar_map, ax=ax, pad=0.04)
        cbar.set_label("time")


    if gate_positions is not None and gate_quaternions is not None:
        frame_positions = _gate_frame_positions(gate_positions, gate_quaternions)
        frame_xy = _extract_xy(frame_positions)
        ax.scatter(
            frame_xy[:, 0],
            frame_xy[:, 1],
            color="tab:orange",
            s=18,
            marker="o",
            label="gate frame points",
        )

    if obstacle_positions is not None:
        obstacle_xy = _extract_xy(obstacle_positions)
        ax.scatter(
            obstacle_xy[:, 0],
            obstacle_xy[:, 1],
            color="none",
            edgecolors="dimgray",
            linewidths=1.2,
            s=60,
            marker="o",
            label="obstacles",
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Drone trajectory (top view)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    if show:
        plt.show()

    return out_path
