"""Compare the const_vel and mppi opponent models against what the opponent actually did.

The claim under test: ``const_vel`` extrapolates the filtered velocity in a straight line, so it
is right exactly as long as the opponent flies straight. The moment the opponent turns, the
prediction leaves on the tangent and the error grows with the square of the horizon, right when
a racing controller most needs to know where the other drone will be. The ``mppi`` model instead
rolls the opponent through the same dynamics and its own racing line, so a curve is just another
part of the path.

Both predictions come from ONE run and the SAME observed state: ``const_vel`` is a pure function
of the tracked position and filtered velocity (:func:`opponents.predict`), so it can be evaluated
on the host each step alongside the joint rollout, and neither model gets a different opponent to
predict. Ground truth is the opponent's own later observations -- the planning step and the
control step are both 20 ms here, so horizon step n of the prediction made at control step k is
scored against the observation at step k + n.

Each model gets its own top-down panel, sharing a box and a moment, so the comparison is between
two pictures of the same instant rather than two lines fighting over one. The mppi panel also
carries the bank its prediction is chosen from -- one line per MPPI mode, coloured as the MuJoCo
view colours them. const_vel has no bank to draw, which is itself the difference.

The caveat this cannot show: the ``mppi`` model assumes the opponent runs OUR controller on its
own spline. The simulated opponent roughly does, which is why it looks this good. A real opposing
team does not, and the advantage shrinks accordingly.

Run as:

    $ pixi run -e gpu python scripts/compare_opp_models.py
    $ pixi run -e gpu python scripts/compare_opp_models.py --map_only --stride 2
    $ pixi run -e gpu python scripts/compare_opp_models.py --from_cache opp_models_record.pkl
"""

from __future__ import annotations

import copy
import logging
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import fire
import gymnasium
import matplotlib
import numpy as np
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

# lsy_drone_racing must be imported before scipy: crazyflow sets SCIPY_ARRAY_API on import and
# refuses to load if scipy came first. Same reason as the isort skip in sweep_study.py.
from lsy_drone_racing.control.mppi import opponents  # isort: skip
from lsy_drone_racing.control.mppi.diagnostics import CLUSTER_COLORS  # isort: skip
from lsy_drone_racing.utils import load_config, load_controller  # isort: skip

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FFMpegWriter, PillowWriter  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from lsy_drone_racing.control.trajectory_mppi import AttitudeMPPIController

logger = logging.getLogger(__name__)
ROOT = Path(__file__).parents[1]

# One colour per model, plus the truth they are both scored against. Validated as a categorical
# pair (CVD separation dE 25.2) in scripts/plot_ibr_vs_parallel.py and reused across the deck.
MPPI_C = "#1A73C8"
CONST_C = "#B3560B"
TRUE_C = "#222222"
MODELS = (("mppi", MPPI_C), ("const_vel", CONST_C))


# ----------------------------------------------------------------------------------------------
# Record: fly once, keep both predictions and the truth
# ----------------------------------------------------------------------------------------------
def _opp_traj(ctrl: AttitudeMPPIController, a: int = 1) -> NDArray:
    """The opponent rollout the ego was actually scored against this step, (N, 3).

    Under best response that is the agent's choice at the fixed point; without it, the cheapest
    sample the opponent has. Either way it is the trajectory the collision cost saw, not a
    separate prediction made for this plot.
    """
    if ctrl.ibr_best_samples is not None:
        w = int(np.asarray(ctrl.ibr_best_samples)[a])
    else:
        w = int(np.argmin(np.asarray(ctrl.all_costs[a])))
    return np.asarray(ctrl.all_positions[a][w // ctrl.M[a], w % ctrl.M[a]])


def _opp_modes(ctrl: AttitudeMPPIController, a: int = 1) -> NDArray:
    """One line per opponent MPPI mode -- its cheapest sample -- as (K, N, 3).

    Exactly what :func:`diagnostics.draw_opponent_rollouts` puts in the MuJoCo view: the whole
    bank is far too many lines to draw, so each mode is represented by the sample it would
    actually propose.
    """
    costs = np.asarray(ctrl.all_costs[a])  # (K, M)
    return np.stack(
        [
            np.asarray(ctrl.all_positions[a][k, int(np.argmin(costs[k]))])
            for k in range(ctrl.K[a])
        ]
    )


def fly_and_record(
    config: str, seed: int | None, max_steps: int, n_samples: int | None = None
) -> dict:
    """Fly one episode, recording both opponent predictions and the opponent's true track.

    Args:
        config: config file in ``config/``. The MPPI controller must run
            ``opponent.model = "mppi"``, which is what supplies the joint-rollout prediction.
        seed: environment seed, or None to use the config's.
        max_steps: give up after this many environment steps.
        n_samples: override the rollout width, to fit a GPU that is already busy. Lower widths
            weaken the mppi arm slightly, so report numbers from the deployed width.

    Returns:
        A record dict of stacked per-step arrays plus the static track.
    """
    cfg = load_config(ROOT / "config" / config)
    cfg.sim.render = False
    if seed is not None:
        cfg.env.seed = int(seed)

    # The joint rollout has to be running for its prediction to exist at all, so select it here
    # rather than depending on whatever the config currently has. This does not tilt the
    # comparison: the const_vel arm is evaluated on the host from the same tracked state either
    # way, and the opponent being predicted flies its own open-loop line regardless of what the
    # ego believes about it.
    ego_idx = next((r for r, c in enumerate(cfg.controller) if "mppi" in c), None)
    if ego_idx is None:
        raise SystemExit(f"No MPPI controller in {config}: nothing to take a prediction from.")
    was = cfg.controller[ego_idx]["mppi"]["opponent"]["model"]
    if was != "mppi":
        logger.info("Setting controller[%d] opponent.model %r -> 'mppi' for this run", ego_idx, was)
        cfg.controller[ego_idx]["mppi"]["opponent"]["model"] = "mppi"
    if n_samples is not None:
        logger.warning(
            "Rollout width %d -> %d: a narrower batch is a handicap on the mppi arm, so treat "
            "this as a pipeline check rather than the reported number.",
            cfg.controller[ego_idx]["mppi"]["n_samples"], n_samples,
        )
        cfg.controller[ego_idx]["mppi"]["n_samples"] = int(n_samples)

    control_path = ROOT / "lsy_drone_racing/control"
    names = [c["file"] for c in cfg.controller]
    classes = [load_controller(control_path / n) for n in names]
    freqs = np.array([kw["freq"] for kw in cfg.env.kwargs], dtype=np.int64)
    base_freq = int(np.max(freqs))
    periods = base_freq // freqs

    env = JaxToNumpy(
        gymnasium.make(
            "MultiDroneRacing-v0",
            freq=base_freq,
            sim_config=cfg.sim,
            track=cfg.env.track,
            sensor_range=cfg.env.kwargs[0]["sensor_range"],
            control_mode=cfg.env.kwargs[0]["control_mode"],
            disturbances=cfg.env.get("disturbances"),
            randomizations=cfg.env.get("randomizations"),
            seed=cfg.env.seed,
        )
    )
    n_drones = env.unwrapped.sim.n_drones
    obs, info = env.reset()

    controllers = []
    for rank, cls in enumerate(classes):
        c_cfg = copy.deepcopy(cfg)
        c_cfg.env.freq = np.int64(c_cfg.env.kwargs[rank]["freq"])
        controllers.append(cls(obs, info | {"rank": rank}, c_cfg))

    ego_rank = ego_idx
    ego = controllers[ego_rank]
    if getattr(ego, "opponent_model", None) != "mppi":
        raise SystemExit(
            f"controller[{ego_rank}] of {config} did not come up on the joint rollout "
            f"(opponent_model = {getattr(ego, 'opponent_model', None)!r})."
        )
    opp_rank = 1 - ego_rank
    # Ground truth is sampled at control steps, so the planning step has to land on them.
    stride = ego.dt / ego.ctrl_dt
    if abs(stride - round(stride)) > 1e-6:
        raise SystemExit(
            f"Planning step {ego.dt:.4f} s is not a multiple of the control step "
            f"{ego.ctrl_dt:.4f} s, so predictions cannot be aligned to observations."
        )
    # The const_vel predictor, built with the same horizon and step as the real one.
    const_params = opponents.PredictorParams(
        model="const_vel",
        horizon=ego.N,
        dt=ego.dt,
        v_theta_max=ego.v_theta_max,
        offset_tau=ego.cfg.opponent.pred_offset_tau,
    )
    logger.info(
        "Recording rank %d (%s) predicting rank %d; horizon %d x %.3f s",
        ego_rank, names[ego_rank], opp_rank, ego.N, ego.dt,
    )

    rec: dict[str, list] = {
        k: [] for k in ("t", "opp_pos", "ego_pos", "pred_mppi", "pred_const", "modes_mppi")
    }
    actions = np.zeros((n_drones, env.action_space.shape[1]), dtype=np.float32)
    finished = np.full(n_drones, False, dtype=bool)
    i = 0
    while i < max_steps:
        infos = [info | {"rank": r} for r in range(n_drones)]
        disabled = env.unwrapped.data.disabled_drones[0]
        active = (i % periods) == 0
        for r, (ctrl, ci) in enumerate(zip(controllers, infos)):
            if disabled[r]:
                finished[r] = True
                continue
            if active[r]:
                actions[r] = ctrl.compute_control(obs, ci)

        if active[ego_rank] and not disabled[ego_rank] and not disabled[opp_rank]:
            # Same inputs the deployed const_vel predictor would get: the tracked position and
            # the EMA-filtered velocity, agent 1 being the opponent in the ego's ordering.
            pred_const, _ = opponents.predict(
                const_params,
                ego._paths,
                ego._tracker.held["pos"],
                ego._tracker.vel_filt,
                ego._theta,
                ego.n_agents,
            )
            rec["t"].append(i / base_freq)
            rec["opp_pos"].append(np.asarray(obs["pos"])[opp_rank].copy())
            rec["ego_pos"].append(np.asarray(obs["pos"])[ego_rank].copy())
            rec["pred_mppi"].append(_opp_traj(ego))
            rec["modes_mppi"].append(_opp_modes(ego))
            rec["pred_const"].append(pred_const[:, 0])  # (N, 3), single opponent

        obs, reward, terminated, truncated, info = env.step(actions)
        for r, (ctrl, ci) in enumerate(zip(controllers, infos)):
            if not disabled[r] and active[r]:
                finished[r] = ctrl.step_callback(
                    actions[r], obs, reward, terminated, truncated, ci
                )
        i += 1
        if terminated | truncated | finished.all():
            break
    env.close()

    if len(rec["t"]) < ego.N + 2:
        raise SystemExit(f"Only {len(rec['t'])} usable steps recorded; nothing to compare.")
    record = {k: np.asarray(v) for k, v in rec.items()}
    record |= {
        "dt": ego.dt,
        "ctrl_dt": ego.ctrl_dt,
        "stride": int(round(stride)),
        "horizon": ego.N,
        "gates_pos": np.asarray(obs["gates_pos"][ego_rank]).copy(),
        "gates_quat": np.asarray(obs["gates_quat"][ego_rank]).copy(),
        "obstacles_pos": np.asarray(obs["obstacles_pos"][ego_rank]).copy(),
    }
    logger.info("Recorded %d control steps (%.1f s)", len(record["t"]), record["t"][-1])
    return record


# ----------------------------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------------------------
def truth_window(rec: dict, k: int) -> NDArray:
    """What the opponent actually did over the horizon of the prediction made at step k.

    Returns (N, 3) sampled at the same times as the prediction, or fewer rows near the end of
    the recording, where the future runs out.
    """
    idx = k + np.arange(1, rec["horizon"] + 1) * rec["stride"]
    idx = idx[idx < len(rec["opp_pos"])]
    return rec["opp_pos"][idx]


def errors(rec: dict) -> dict[str, NDArray]:
    """Per-step prediction error of both models, in metres.

    ``end`` is the error at the far end of the horizon -- "where we think it will be in
    ``N*dt`` seconds" -- and ``mean`` averages over the whole horizon. Steps whose horizon runs
    past the end of the recording are NaN rather than truncated, so the two models are always
    compared over the same window.
    """
    n_steps, horizon = len(rec["t"]), rec["horizon"]
    out = {}
    for model in ("mppi", "const_vel"):
        pred = rec["pred_mppi"] if model == "mppi" else rec["pred_const"]
        end = np.full(n_steps, np.nan)
        mean = np.full(n_steps, np.nan)
        for k in range(n_steps):
            true = truth_window(rec, k)
            if len(true) < horizon:
                break
            dist = np.linalg.norm(pred[k] - true, axis=-1)  # (N,)
            end[k], mean[k] = dist[-1], dist.mean()
        out[f"{model}_end"], out[f"{model}_mean"] = end, mean
    return out


def curvature(rec: dict, smooth: int = 9) -> NDArray:
    """Path curvature of the opponent's true track (1/m), smoothed.

    This is the quantity const_vel is blind to: a straight-line extrapolation of the velocity
    is exact at zero curvature, and its horizon error grows like kappa * (v * T)^2 / 2.
    """
    pos = rec["opp_pos"][:, :2]
    dt = rec["ctrl_dt"]
    vel = np.gradient(pos, dt, axis=0)
    speed = np.linalg.norm(vel, axis=-1)
    heading = np.unwrap(np.arctan2(vel[:, 1], vel[:, 0]))
    turn = np.abs(np.gradient(heading, dt))
    kappa = turn / np.clip(speed, 0.25, None)  # ignore curvature while nearly stationary
    kernel = np.ones(smooth) / smooth
    return np.convolve(kappa, kernel, mode="same")


def turn_mask(rec: dict, kappa_thresh: float) -> NDArray:
    """Boolean per step: is the opponent turning hard enough to call it a curve?"""
    return curvature(rec) > kappa_thresh


def _smooth(y: NDArray, window: int) -> NDArray:
    """Moving average that keeps NaNs NaN, so the tail with no future stays blank."""
    out = np.full_like(y, np.nan)
    good = np.isfinite(y)
    if not good.any():
        return out
    kernel = np.ones(window) / window
    padded = np.pad(y[good], window // 2, mode="edge")
    out[good] = np.convolve(padded, kernel, mode="valid")[: int(good.sum())]
    return out


def _spans(mask: NDArray) -> list[tuple[int, int]]:
    """Contiguous True runs of a boolean mask, as (start, stop) index pairs."""
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[::2], edges[1::2]))


# ----------------------------------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------------------------------
def _draw_track(ax: plt.Axes, rec: dict) -> None:
    """Gate openings, gate posts and obstacle pillars, top-down."""
    for p, q in zip(rec["gates_pos"], rec["gates_quat"]):
        side = R.from_quat(q).apply([0.0, 1.0, 0.0])[:2]
        seg = np.stack([p[:2] - 0.2 * side, p[:2] + 0.2 * side])
        ax.plot(seg[:, 0], seg[:, 1], color="0.75", lw=2, zorder=1)
        for s in (-1.0, 1.0):
            ax.add_patch(Circle(p[:2] + s * 0.28 * side, 0.035, color="0.45", zorder=1))
    for o in rec["obstacles_pos"]:
        ax.add_patch(Circle(o[:2], 0.055, color="0.45", zorder=1))


def draw_map(
    ax: plt.Axes, rec: dict, err: dict, k: int, lims: tuple, model: str, show_modes: bool
) -> None:
    """One model's top-down panel: what the opponent will do, and what this model predicts.

    The two models get their own panel so the comparison is between two pictures of the same
    moment rather than two lines fighting over one.
    """
    color = dict(MODELS)[model]
    ax.set_aspect("equal")
    ax.set_xlim(lims[0], lims[1])
    ax.set_ylim(lims[2], lims[3])
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    _draw_track(ax, rec)

    # The whole lap, faint, so the current moment has a track to sit on.
    ax.plot(rec["opp_pos"][:, 0], rec["opp_pos"][:, 1], color="0.85", lw=1.0, zorder=1)

    # The bank the mppi prediction is chosen from, drawn as the MuJoCo view draws it: one line
    # per mode, in that mode's colour. const_vel has no bank -- that is half the point.
    if show_modes and model == "mppi" and "modes_mppi" in rec:
        for i, line in enumerate(rec["modes_mppi"][k]):
            ax.plot(
                line[:, 0], line[:, 1], color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
                lw=1.0, alpha=0.8, zorder=2,
            )

    true = truth_window(rec, k)
    if len(true):
        ax.plot(
            true[:, 0], true[:, 1], color=TRUE_C, lw=3.5, alpha=0.9, zorder=3,
            solid_capstyle="round",
        )
    pred = rec["pred_mppi"][k] if model == "mppi" else rec["pred_const"][k]
    ax.plot(pred[:, 0], pred[:, 1], color=color, lw=2.8, zorder=4)
    ax.scatter(*pred[-1, :2], s=55, color=color, zorder=5, edgecolor="white", lw=0.8)

    ax.scatter(*rec["opp_pos"][k, :2], s=110, color=TRUE_C, zorder=6, edgecolor="white", lw=1.2)
    ax.scatter(
        *rec["ego_pos"][k, :2], s=70, color="0.55", zorder=5, edgecolor="white", lw=1.0,
        marker="s",
    )
    ax.set_title(
        f"{model}   ·   {err[f'{model}_end'][k]:.2f} m off", fontsize=12, loc="left", color=color
    )


def draw_error(
    ax: plt.Axes, rec: dict, err: dict, k: int, kappa_thresh: float,
    t_window: tuple[float, float] | None = None,
) -> None:
    """Error over the lap, with the curved sections shaded and a cursor at now.

    ``t_window`` clips both axes to the animated span, so a short clip fills the panel instead
    of sitting as a sliver inside the whole run.
    """
    t = rec["t"]
    for start, stop in _spans(turn_mask(rec, kappa_thresh)):
        ax.axvspan(t[start], t[min(stop, len(t) - 1)], color="0.88", zorder=0)
    for model, color in MODELS:
        raw = err[f"{model}_end"]
        # The mppi trace is noisy step to step: it re-picks a different sample out of the bank
        # every step, and neighbouring samples differ by centimetres. Draw the raw trace faint
        # and a short moving average on top, so neither the noise nor the trend is hidden.
        ax.plot(t, raw, color=color, lw=0.8, alpha=0.3, zorder=2)
        ax.plot(t, _smooth(raw, 9), color=color, lw=2.0, label=model, zorder=3)
    ax.axvline(t[k], color=TRUE_C, lw=1.0, alpha=0.6, zorder=4)
    ax.set_xlabel("time [s]")
    ax.set_ylabel(f"error {rec['horizon'] * rec['dt']:.1f} s ahead [m]")
    lo, hi = (t[0], t[-1]) if t_window is None else t_window
    shown = (t >= lo) & (t <= hi)
    finite = np.concatenate([err[f"{m}_end"][shown] for m, _ in MODELS])
    ax.set_ylim(0.0, float(np.nanmax(finite)) * 1.08)
    ax.set_xlim(lo, hi)
    ax.grid(alpha=0.25)
    handles = [Line2D([], [], color=c, lw=2, label=m) for m, c in MODELS]
    handles.append(Line2D([], [], color="0.88", lw=8, label="opponent turning"))
    ax.legend(handles=handles, loc="upper left", fontsize=8, ncol=3, framealpha=0.9)


def _limits(rec: dict, margin: float = 0.35) -> tuple:
    """A square box around the whole lap, so the map never jumps between frames."""
    pts = rec["opp_pos"][:, :2]
    lo, hi = pts.min(axis=0) - margin, pts.max(axis=0) + margin
    span = max(hi - lo)
    mid = (lo + hi) / 2
    return (mid[0] - span / 2, mid[0] + span / 2, mid[1] - span / 2, mid[1] + span / 2)


def _follow(rec: dict, margin: float = 0.25, smooth: int = 7) -> tuple[NDArray, float]:
    """A panning window that keeps the drone and both predictions in frame at a fixed zoom.

    Over a whole lap the interesting detail -- how far the tangent leaves the true path -- is a
    few tens of centimetres inside a 3.4 m box, i.e. invisible. The zoom is held constant across
    the animation (the widest moment any frame needs) so the eye can compare frames, and only
    the centre moves; smoothing the centre keeps the pan from jittering with the predictions.

    Returns per-step centres (n_steps, 2) and the shared span (m).
    """
    n_steps = len(rec["t"])
    centres, spans = np.zeros((n_steps, 2)), np.zeros(n_steps)
    for k in range(n_steps):
        pts = [rec["opp_pos"][k, :2], rec["pred_mppi"][k][:, :2], rec["pred_const"][k][:, :2]]
        true = truth_window(rec, k)
        if len(true):
            pts.append(true[:, :2])
        pts = np.concatenate([np.atleast_2d(p) for p in pts])
        lo, hi = pts.min(axis=0) - margin, pts.max(axis=0) + margin
        centres[k] = (lo + hi) / 2
        spans[k] = max(hi - lo)
    # One zoom for the whole animation: the 95th percentile, so a single wild const_vel
    # excursion does not shrink every other frame to fit it.
    span = float(np.clip(np.percentile(spans, 95), 1.2, 3.2))
    kernel = np.ones(smooth) / smooth
    for axis in (0, 1):
        padded = np.pad(centres[:, axis], smooth // 2, mode="edge")
        centres[:, axis] = np.convolve(padded, kernel, mode="valid")[:n_steps]
    return centres, span


def _figure(
    rec: dict, err: dict, lims: tuple | None, kappa_thresh: float, map_only: bool,
    show_modes: bool = True, t_window: tuple[float, float] | None = None,
) -> tuple:
    """The figure and a redraw closure over it. ``lims=None`` pans with the drone.

    One top-down panel per model, sharing a box so the two are directly comparable, plus the
    error trace underneath unless ``map_only``.
    """
    if map_only:
        fig = plt.figure(figsize=(12.6, 6.6), layout="constrained")
        grid = fig.add_gridspec(1, 2)
        axes = {m: fig.add_subplot(grid[0, i]) for i, (m, _) in enumerate(MODELS)}
        ax_err = None
    else:
        fig = plt.figure(figsize=(12.6, 9.6), layout="constrained")
        grid = fig.add_gridspec(2, 2, height_ratios=[1.55, 1.0])
        axes = {m: fig.add_subplot(grid[0, i]) for i, (m, _) in enumerate(MODELS)}
        ax_err = fig.add_subplot(grid[1, :])
    centres, span = _follow(rec) if lims is None else (None, 0.0)
    horizon_s = rec["horizon"] * rec["dt"]

    def redraw(k: int) -> None:
        box = lims
        if box is None:
            cx, cy = centres[k]
            box = (cx - span / 2, cx + span / 2, cy - span / 2, cy + span / 2)
        for model, _ in MODELS:
            ax = axes[model]
            ax.clear()
            draw_map(ax, rec, err, k, box, model, show_modes)
        handles = [
            Line2D([], [], color=TRUE_C, lw=3, label="what the opponent actually does"),
            Line2D([], [], color=MPPI_C, lw=2.6, label="prediction used"),
        ]
        if show_modes and "modes_mppi" in rec:
            handles.append(Line2D([], [], color=CLUSTER_COLORS[0], lw=1, label="sampled modes"))
        axes["mppi"].legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)
        axes["const_vel"].legend(
            handles=[
                Line2D([], [], color=TRUE_C, lw=3, label="what the opponent actually does"),
                Line2D([], [], color=CONST_C, lw=2.6, label="prediction used"),
            ],
            loc="upper right", fontsize=8, framealpha=0.9,
        )
        if not map_only:
            ax_err.clear()
            draw_error(ax_err, rec, err, k, kappa_thresh, t_window)
        fig.suptitle(
            f"Opponent prediction at t = {rec['t'][k]:.2f} s   ·   looking {horizon_s:.1f} s "
            "ahead",
            fontsize=14,
        )

    return fig, redraw


def render_animation(
    rec: dict, err: dict, out: Path, kappa_thresh: float, stride: int, fps: int,
    fmt: str, map_only: bool, follow: bool, show_modes: bool,
    t_start: float | None = None, t_end: float | None = None,
) -> Path:
    """Animate the lap, one frame per `stride` control steps.

    ``t_start``/``t_end`` clip which part of the lap is animated. The error panel still shows
    the whole lap, so a clipped animation keeps the context of where its window sits.
    """
    valid = np.flatnonzero(np.isfinite(err["mppi_end"]))
    frames = list(range(int(valid[0]), int(valid[-1]) + 1, max(stride, 1)))
    t = rec["t"]
    frames = [
        k
        for k in frames
        if (t_start is None or t[k] >= t_start) and (t_end is None or t[k] <= t_end)
    ]
    if not frames:
        raise SystemExit(
            f"No steps between t_start={t_start} and t_end={t_end}; the recording spans "
            f"{t[int(valid[0])]:.2f}-{t[int(valid[-1])]:.2f} s."
        )
    lims = None if follow else _limits(rec)
    # Clip the error panel to what is actually being animated, so a short clip fills it.
    window = None
    if t_start is not None or t_end is not None:
        window = (float(t[frames[0]]), float(t[frames[-1]]))
    fig, redraw = _figure(rec, err, lims, kappa_thresh, map_only, show_modes, window)
    path = out.with_suffix(".mp4" if fmt == "mp4" else ".gif")
    writer = FFMpegWriter(fps=fps, bitrate=3000) if fmt == "mp4" else PillowWriter(fps=fps)
    with writer.saving(fig, str(path), dpi=100):
        for k in frames:
            redraw(k)
            writer.grab_frame()
    plt.close(fig)
    logger.info("Wrote %s (%d frames)", path.name, len(frames))
    return path


def render_summary(rec: dict, err: dict, out: Path, kappa_thresh: float) -> Path:
    """The static claim: error over the lap, and the straight/curve split behind it."""
    fig, (ax_t, ax_bar) = plt.subplots(
        1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1.7, 1.0]}, layout="constrained"
    )
    draw_error(ax_t, rec, err, 0, kappa_thresh)
    ax_t.lines[-1].set_visible(False)  # no "now" cursor on a static figure
    ax_t.set_title("Prediction error over the lap", fontsize=11, loc="left")

    curve = turn_mask(rec, kappa_thresh)
    stats, labels = [], []
    for name, mask in (("straight", ~curve), ("curve", curve)):
        for model, _ in MODELS:
            e = err[f"{model}_end"][mask]
            stats.append(np.nanmean(e))
            labels.append((name, model))
    x = np.array([0.0, 0.28, 0.85, 1.13])
    colors = [c for _, c in MODELS] * 2
    ax_bar.bar(x, stats, width=0.26, color=colors)
    for xi, value in zip(x, stats):
        ax_bar.text(xi, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    ax_bar.set_xticks([0.14, 0.99])
    ax_bar.set_xticklabels(
        [f"straight\n({int(np.sum(~curve))} steps)", f"curve\n({int(np.sum(curve))} steps)"]
    )
    ax_bar.set_ylabel(f"mean error {rec['horizon'] * rec['dt']:.1f} s ahead [m]")
    ax_bar.set_ylim(0.0, max(stats) * 1.2)
    ax_bar.grid(axis="y", alpha=0.25)
    ax_bar.set_title("Where the gap comes from", fontsize=11, loc="left")
    ax_bar.legend(
        handles=[Line2D([], [], color=c, lw=8, label=m) for m, c in MODELS],
        loc="upper left", fontsize=8, frameon=False,
    )

    fig.suptitle(
        "Opponent prediction: straight-line extrapolation vs a joint rollout "
        f"(horizon {rec['horizon'] * rec['dt']:.1f} s)",
        fontsize=13,
    )
    path = out.parent / f"{out.stem}_summary.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(
    config: str = "multi_level0.toml",
    out: str = "opp_models",
    seed: int | None = 1,
    max_steps: int = 1500,
    n_samples: int | None = None,
    kappa_thresh: float = 0.9,
    stride: int = 2,
    fmt: str = "gif",
    fps: int = 10,
    map_only: bool = False,
    follow: bool = True,
    show_modes: bool = True,
    t_start: float | None = None,
    t_end: float | None = None,
    from_cache: str | None = None,
) -> None:
    """Compare the const_vel and mppi opponent predictions against the opponent's real track.

    Args:
        config: config file in ``config/``; one controller must run opponent.model = "mppi".
        out: output stem, relative to the repo root.
        seed: environment seed; None uses the config's.
        max_steps: environment steps before giving up.
        n_samples: override the rollout width, to fit alongside another job on the GPU.
        kappa_thresh: curvature (1/m) above which a step counts as "in a curve". 0.9 is a
            ~1.1 m turn radius; raise it to shade only the tightest corners.
        stride: control steps per animation frame.
        fmt: "gif" or "mp4".
        fps: animation frame rate.
        map_only: animate the map alone, dropping the error panel.
        follow: pan with the drone at a fixed zoom. False shows the whole lap, where the
            prediction error is too small to see.
        show_modes: draw the opponent's MPPI mode candidates on the mppi panel, as the MuJoCo
            view does. Needs a record made after modes were added; older caches just omit them.
        t_start: animate only from this time (s). The error panel still shows the whole lap.
        t_end: animate only up to this time (s).
        from_cache: redraw the run cached in this ``*_record.pkl`` instead of flying again.
            Every run without it writes ``<out>_record.pkl``.
    """
    out_path = ROOT / out
    cache = Path(from_cache) if from_cache else out_path.parent / f"{out_path.stem}_record.pkl"
    if from_cache:
        rec = pickle.loads(cache.read_bytes())
        logger.info("Reusing the run recorded in %s (%d steps)", cache.name, len(rec["t"]))
    else:
        rec = fly_and_record(config, seed, max_steps, n_samples)
        cache.write_bytes(pickle.dumps(rec))

    err = errors(rec)
    curve = turn_mask(rec, kappa_thresh)
    for name, mask in (("straight", ~curve), ("curve", curve)):
        logger.info(
            "%-8s (%3d steps): mean error %.1f s ahead — mppi %.3f m, const_vel %.3f m",
            name, int(mask.sum()), rec["horizon"] * rec["dt"],
            float(np.nanmean(err["mppi_end"][mask])),
            float(np.nanmean(err["const_vel_end"][mask])),
        )
    if show_modes and "modes_mppi" not in rec:
        logger.warning("%s predates the mode candidates; drawing without them.", cache.name)
    anim = render_animation(
        rec, err, out_path, kappa_thresh, stride, fps, fmt, map_only, follow, show_modes,
        t_start, t_end,
    )
    summary = render_summary(rec, err, out_path, kappa_thresh)
    logger.info("Wrote %s and %s", anim.name, summary.name)


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("lsy_drone_racing").setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    fire.Fire(main, serialize=lambda _: None)
