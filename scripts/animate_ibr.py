"""Animate the iterative best response converging at ONE frozen control step.

The point this makes visible: within a single 20 ms control step, the two drones' plans are not
computed once. The rollouts are computed once, and then each agent re-picks its best trajectory
against the other's *current* pick, in turn, until the picks stop moving -- a fixed point of the
joint game. This script freezes the race at one such step and steps through those picks.

How it gets the intermediate picks: :func:`ibr.build_ibr_fn` is built with ``trace=True``, which
makes the solve report the choice after every move instead of only the fixed point. Nothing else
about the control step changes -- same rollouts, same cost, same sampler -- so what is drawn is
the real solve, not a re-enactment.

Choosing the frozen step matters more than it sounds: where the drones are far apart the coupling
is nil, the seed is already the fixed point, and the animation is a still image. So the script
flies the whole episode and keeps the step whose solve moved the plans the most.

Outputs, next to each other: an animation (GIF or MP4) and a storyboard PNG with one panel per
move, for slides that cannot play video.

Run as:

    $ pixi run -e gpu python scripts/animate_ibr.py
    $ pixi run -e gpu python scripts/animate_ibr.py --config multi_level0.toml --seed 7
    $ pixi run -e gpu python scripts/animate_ibr.py --freeze_step 120 --fmt mp4
    $ pixi run -e gpu python scripts/animate_ibr.py --map_only   # top-down panel alone
    # restyle the step already captured, without flying the episode again:
    $ pixi run -e gpu python scripts/animate_ibr.py --from_cache ibr_convergence_capture.pkl
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
from lsy_drone_racing.control.mppi import diagnostics, ibr as ibr_mod, optimizer  # isort: skip
from lsy_drone_racing.utils import load_config, load_controller  # isort: skip

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FFMpegWriter, PillowWriter  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Ellipse  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from lsy_drone_racing.control.trajectory_mppi import AttitudeMPPIController

logger = logging.getLogger(__name__)
ROOT = Path(__file__).parents[1]

# Agent colours. Validated as a categorical pair in scripts/plot_ibr_vs_parallel.py, and reused
# here so the deck's figures and animations identify the two drones the same way.
EGO_C = "#1A73C8"
OPP_C = "#B3560B"


# ----------------------------------------------------------------------------------------------
# Capture: fly the race, freeze one step, keep everything needed to redraw it
# ----------------------------------------------------------------------------------------------
def _trace_ibr(
    ctrl: AttitudeMPPIController, ibr_mode: str | None, ibr_iters: int | None
) -> tuple[str, int]:
    """Rebuild this controller's best-response solve so it reports every intermediate choice.

    The update function closes over the solve at build time, so both have to be rebuilt. Every
    other argument comes off the controller, i.e. the rebuilt update is the same one it was
    using -- only the trace (and any mode/iteration override) differs.

    Returns the mode and pass count actually used; the config dataclasses are frozen, so an
    override lives here rather than on ``ctrl.cfg``.
    """
    mode = ibr_mode or ctrl.cfg.opponent.ibr_mode
    iters = ctrl.cfg.opponent.ibr_iters if ibr_iters is None else int(ibr_iters)
    ctrl._ibr_fn = ibr_mod.build_ibr_fn(
        opp_params=ctrl._opp_cost,
        n_sim_agents=ctrl.n_sim_agents,
        dt=ctrl.dt,
        n_iters=iters,
        mode=mode,
        trace=True,
    )
    ctrl._update_fn = optimizer.build_update_fn(
        rollout_fn=ctrl._rollout_fn,
        coupled_cost_fn=ctrl._coupled_cost_fn,
        ibr_fn=ctrl._ibr_fn,
        sampler=ctrl._sampler,
        K=ctrl.K,
        M=ctrl.M,
        horizon=ctrl.N,
        n_samples=ctrl.n_samples,
        num_inputs=ctrl.num_inputs,
        n_sim_agents=ctrl.n_sim_agents,
        n_agents=ctrl.n_agents,
    )
    return mode, iters


def _world_traj(ctrl: AttitudeMPPIController, a: int, w: int) -> NDArray:
    """The (N, 3) rollout of agent `a` in rollout world `w`.

    The cached positions reach the controller already grouped per mode, (K_a, M_a, N, 3), so the
    flat world index the solve returns has to be split the same way the grouping did it.
    """
    return np.asarray(ctrl.all_positions[a][w // ctrl.M[a], w % ctrl.M[a]])


def _capture(
    ctrl: AttitudeMPPIController, obs: dict, rank: int, t: float, step: int, solve: tuple[str, int]
) -> dict:
    """Everything needed to redraw this control step, pulled off the controller after it ran."""
    hist = np.asarray(ctrl.ibr_best_samples)  # (F, A) world index per agent, seed first
    n_sim = ctrl.n_sim_agents
    traj = np.stack(
        [[_world_traj(ctrl, a, int(hist[f, a])) for a in range(n_sim)] for f in range(len(hist))]
    )  # (F, A, N, 3)
    # The sample bank behind those choices: each mode's cheapest rollout, drawn as context.
    modes = [
        np.stack(
            [
                np.asarray(ctrl.all_positions[a][k, int(np.argmin(ctrl.all_costs[a][k]))])
                for k in range(ctrl.K[a])
            ]
        )
        for a in range(n_sim)
    ]
    # Agent order as the controller sees it: its own rank first, then the others.
    order = [rank] + [r for r in range(len(np.asarray(obs["pos"]))) if r != rank]
    # The ego's state comes from the observation, not the tracker: OpponentTracker.update loops
    # from a = 1, so slot 0 keeps its reset value forever. The opponents' filtered velocity is
    # taken from the tracker, because that is the heading the keep-out was actually built from.
    vel = np.asarray(ctrl._tracker.vel_filt).copy()
    vel[0] = np.asarray(obs["vel"])[rank]
    return {
        "t": t,
        "step": step,
        "rank": rank,
        "history": hist,
        "traj": traj,
        "modes": modes,
        "pos": np.asarray(obs["pos"])[order],  # (A, 3) ego first
        "vel": vel,  # (A, 3) ego first
        "inflate": np.asarray(ctrl._opp_inflate_np).copy(),
        "refs": [p.get_trajectory(400) for p in ctrl._planner],
        "gates_pos": np.asarray(obs["gates_pos"][rank]).copy(),
        "gates_quat": np.asarray(obs["gates_quat"][rank]).copy(),
        "obstacles_pos": np.asarray(obs["obstacles_pos"][rank]).copy(),
        "opp_params": ctrl._opp_cost,
        "dt": ctrl.dt,
        "ibr_mode": solve[0],
        "ibr_iters": solve[1],
    }


def _outcome(cap: dict) -> str:
    """What the solve did at this step: settled, cycled, or was still moving when it ran out.

    A fixed point is a repeated *choice*, so this reads the world indices, not the geometry: two
    different rollouts can be visually identical without the iteration having settled.
    """
    hist = [tuple(row) for row in cap["history"]]
    if len(hist) < 2:
        return "moving"
    if hist[-1] == hist[-2]:
        return "converged"
    if hist[-1] in hist[:-2]:  # revisits an earlier state -> limit cycle
        return "cycling"
    return "moving"


def _settled_after(cap: dict) -> int:
    """How many moves the fixed point took: the last move that changed anyone's choice."""
    hist = [tuple(row) for row in cap["history"]]
    changed = [f for f in range(1, len(hist)) if hist[f] != hist[f - 1]]
    return changed[-1] if changed else 0


def _plan_shift(cap: dict) -> NDArray:
    """Per move, how far the chosen plans travelled (m).

    This is the quantity that goes to zero on convergence, and the score used to pick which
    control step is worth animating.
    """
    traj = cap["traj"]  # (F, A, N, 3)
    if len(traj) < 2:
        return np.zeros(1)
    return np.linalg.norm(traj[1:] - traj[:-1], axis=-1).max(axis=(1, 2))  # (F-1,)


def _separation(cap: dict) -> NDArray:
    """Per frame, the closest the two chosen plans come to each other over the horizon (m).

    Time-indexed, matching the cost: the ego at horizon step n is scored against the opponent at
    step n, not against the whole opponent line.
    """
    traj = cap["traj"]
    return np.linalg.norm(traj[:, 0] - traj[:, 1], axis=-1).min(axis=-1)  # (F,)


def fly_and_freeze(
    config: str,
    seed: int | None,
    freeze_step: int | None,
    max_steps: int,
    min_step: int,
    pick: str,
    ibr_mode: str | None,
    ibr_iters: int | None,
) -> dict:
    """Fly one episode and return the captured control step worth animating.

    Args:
        config: config file in ``config/``. Needs two drones and an MPPI controller with
            ``opponent.model = "mppi"`` and ``ibr_iters >= 0``.
        seed: environment seed, or None to use the config's.
        freeze_step: freeze this control step instead of searching.
        max_steps: give up after this many steps.
        min_step: ignore steps before this one. The drones sit still on the start line, where
            the keep-out is a sphere and there is no racing line to give up, so the largest
            plan movement of the whole episode is usually there and is the least interesting
            interaction in it.
        pick: which step to keep -- "converged" (moved the most and then settled),
            "cycling" (moved the most and never settled), or "moved" (moved the most,
            settled or not).
        ibr_mode: override the config's ``ibr_mode``.
        ibr_iters: override the config's ``ibr_iters``.

    Returns:
        The capture dict of the chosen step.
    """
    cfg = load_config(ROOT / "config" / config)
    cfg.sim.render = False
    if seed is not None:
        cfg.env.seed = int(seed)

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

    ego_rank = next(
        (r for r, c in enumerate(controllers) if getattr(c, "_ibr_fn", None) is not None), None
    )
    if ego_rank is None:
        raise SystemExit(
            f"No best-response controller in {config}: needs an MPPI controller with "
            "opponent.model = 'mppi' and opponent.ibr_iters >= 0."
        )
    ego = controllers[ego_rank]
    solve = _trace_ibr(ego, ibr_mode, ibr_iters)
    logger.info(
        "Tracing IBR on controller rank %d (%s): %s, %d passes",
        ego_rank, names[ego_rank], solve[0], solve[1],
    )

    best_cap, best_score = None, -1.0
    tally: dict[str, int] = {}
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

        if active[ego_rank] and not disabled[ego_rank]:
            step = i // periods[ego_rank]
            if step == freeze_step or (freeze_step is None and step >= min_step):
                cap = _capture(ego, obs, ego_rank, i / base_freq, step, solve)
                outcome = _outcome(cap)
                tally[outcome] = tally.get(outcome, 0) + 1
                # Rank by how far the solve moved the plans -- a step where the seed is already
                # the fixed point animates as a still image -- but only among steps that ended
                # the way the caller asked to see.
                score = float(_plan_shift(cap).sum())
                if pick in ("moved", outcome) and score > best_score:
                    best_cap, best_score = cap, score
                if freeze_step is not None:
                    best_cap = cap
                    break

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

    total = sum(tally.values())
    logger.info(
        "Solve outcome over %d control steps: %s",
        total,
        ", ".join(f"{k} {v} ({100 * v / max(total, 1):.0f}%)" for k, v in sorted(tally.items())),
    )
    if best_cap is None:
        raise SystemExit(
            f"No control step ended '{pick}' in {total} steps. Try --pick moved, more "
            "--ibr_iters, or a different --seed."
        )
    logger.info(
        "Frozen at step %d (t = %.2f s, %s): plans moved %.3f m over the solve, "
        "closest approach of the two plans %.2f m -> %.2f m",
        best_cap["step"], best_cap["t"], _outcome(best_cap), float(_plan_shift(best_cap).sum()),
        _separation(best_cap)[0], _separation(best_cap)[-1],
    )
    return best_cap


# ----------------------------------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------------------------------
def _move_labels(cap: dict) -> list[str]:
    """Caption per frame: who moved to produce it."""
    n_sim = cap["traj"].shape[1]
    names = ["ego"] + [f"opponent {a}" if n_sim > 2 else "opponent" for a in range(1, n_sim)]
    labels = ["seed: each drone's own racing line (no coupling)"]
    for it in range(cap["ibr_iters"]):
        if cap["ibr_mode"] == "vmap":
            labels.append(f"pass {it + 1}: both drones re-pick simultaneously")
        else:
            for a in range(n_sim):
                labels.append(f"pass {it + 1}: {names[a]} best-responds")
    return labels[: len(cap["traj"])]


def _draw_track(ax: plt.Axes, cap: dict) -> None:
    """Gate openings, gate posts and obstacle pillars, top-down."""
    for p, q in zip(cap["gates_pos"], cap["gates_quat"]):
        side = R.from_quat(q).apply([0.0, 1.0, 0.0])[:2]
        seg = np.stack([p[:2] - 0.2 * side, p[:2] + 0.2 * side])
        ax.plot(seg[:, 0], seg[:, 1], color="0.75", lw=2, zorder=1)
        for s in (-1.0, 1.0):
            ax.add_patch(Circle(p[:2] + s * 0.28 * side, 0.035, color="0.45", zorder=1))
    for o in cap["obstacles_pos"]:
        ax.add_patch(Circle(o[:2], 0.055, color="0.45", zorder=1))


def _keepout_tube(ax: plt.Axes, cap: dict, f: int, every: int = 5) -> None:
    """The opponent's keep-out, drawn along the trajectory the ego is scored against.

    Geometry comes from :func:`diagnostics._keepout_frame`, the same helper the MuJoCo view uses,
    so the drawn bubble is the r = 1 level set of the cost rather than a stand-in for it.
    """
    opp = cap["traj"][f, 1]  # (N, 3)
    vel = np.diff(opp, axis=0, prepend=opp[:1]) / cap["dt"]
    vel[0] = vel[1] if len(vel) > 1 else vel[0]
    inflate = float(cap["inflate"][0]) if len(cap["inflate"]) else 1.0
    for n in range(0, len(opp), every):
        semi, mat, _ = diagnostics._keepout_frame(cap["opp_params"], vel[n], inflate)
        angle = np.degrees(np.arctan2(mat[1, 0], mat[0, 0]))  # heading = local x axis
        ax.add_patch(
            Ellipse(
                opp[n, :2], 2 * semi[0], 2 * semi[1], angle=angle,
                facecolor=OPP_C, edgecolor=OPP_C, lw=0.5, alpha=0.055, zorder=2,
            )
        )


def draw_state(ax: plt.Axes, cap: dict, f: int, lims: tuple, show_bank: bool = True) -> None:
    """Draw the frozen scene with the plans chosen at move `f`."""
    ax.set_aspect("equal")
    ax.set_xlim(lims[0], lims[1])
    ax.set_ylim(lims[2], lims[3])
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    _draw_track(ax, cap)

    for a, color in ((0, EGO_C), (1, OPP_C)):
        ref = cap["refs"][a]
        ax.plot(ref[:, 0], ref[:, 1], color=color, lw=0.8, ls=":", alpha=0.5, zorder=1)

    if show_bank:  # the sample bank the choice is made from
        for a in range(cap["traj"].shape[1]):
            for line in cap["modes"][a]:
                ax.plot(line[:, 0], line[:, 1], color="0.6", lw=0.6, alpha=0.45, zorder=2)

    _keepout_tube(ax, cap, f)

    if f > 0:  # where each plan sat before this move
        for a, color in ((0, EGO_C), (1, OPP_C)):
            prev = cap["traj"][f - 1, a]
            ax.plot(prev[:, 0], prev[:, 1], color=color, lw=1.6, ls="--", alpha=0.35, zorder=3)

    for a, color, name in ((0, EGO_C, "ego"), (1, OPP_C, "opponent")):
        traj = cap["traj"][f, a]
        moved = f > 0 and not np.array_equal(cap["history"][f, a], cap["history"][f - 1, a])
        ax.plot(
            traj[:, 0], traj[:, 1], color=color, lw=3.0 if moved else 2.2,
            alpha=1.0, zorder=5, label=f"{name} plan",
        )
        ax.scatter(*cap["pos"][a, :2], s=90, color=color, edgecolor="white", lw=1.2, zorder=6)
        v = cap["vel"][a, :2]
        ax.arrow(
            *cap["pos"][a, :2], *(0.25 * v), width=0.012, color=color, alpha=0.8, zorder=6,
            length_includes_head=True,
        )


def draw_altitude(ax: plt.Axes, cap: dict, f: int) -> None:
    """Altitude along the horizon, since a top-down map hides a purely vertical dodge.

    The wake keep-out only extends downward, so "get out of the way" here often means "climb",
    which the map cannot show at all.
    """
    traj = cap["traj"]
    steps = np.arange(traj.shape[2]) * cap["dt"]
    for a, color, name in ((0, EGO_C, "ego"), (1, OPP_C, "opponent")):
        if f > 0:
            ax.plot(steps, traj[f - 1, a, :, 2], color=color, lw=1.2, ls="--", alpha=0.35)
        ax.plot(steps, traj[f, a, :, 2], color=color, lw=2.2, label=name)
    ax.set_xlabel("horizon time [s]")
    ax.set_ylabel("altitude [m]")
    zs = traj[..., 2]
    ax.set_ylim(float(zs.min()) - 0.08, float(zs.max()) + 0.08)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8, frameon=False)


def draw_convergence(ax: plt.Axes, cap: dict, f: int) -> None:
    """Right panel: how far the plans moved, and how close they still come to each other."""
    sep, shift = _separation(cap), _plan_shift(cap)
    x = np.arange(len(sep))
    ax.plot(x, sep, color="0.8", lw=1.2, zorder=1)
    ax.plot(x[: f + 1], sep[: f + 1], "o-", color="#333333", lw=1.8, ms=5, zorder=3)
    ax.axhline(
        2 * cap["opp_params"].lateral, color=OPP_C, ls="--", lw=1.0,
        label=f"keep-out width ({2 * cap['opp_params'].lateral:.2f} m)",
    )
    ax.set_xlabel("best-response move")
    ax.set_ylabel("closest approach of the two plans [m]")
    ax.set_xlim(-0.3, len(sep) - 0.7)
    ax.set_ylim(0.0, max(float(sep.max()) * 1.15, 2.2 * cap["opp_params"].lateral))
    ax.set_xticks(x)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    if f > 0:
        ax.set_title(f"plans moved {shift[f - 1]:.3f} m on this move", fontsize=9, color="0.3")
    else:
        ax.set_title("before any coupling", fontsize=9, color="0.3")


def _limits(cap: dict, margin: float, full_track: bool) -> tuple:
    """Axis box: tight around the interaction, or the whole track."""
    if full_track:
        return (-2.0, 2.0, -1.5, 1.5)
    pts = np.concatenate(
        [cap["traj"].reshape(-1, 3), cap["pos"]] + [m.reshape(-1, 3) for m in cap["modes"]]
    )[:, :2]
    lo, hi = pts.min(axis=0) - margin, pts.max(axis=0) + margin
    span = max(hi - lo)  # square box, so the aspect never crops a panel
    mid = (lo + hi) / 2
    return (mid[0] - span / 2, mid[0] + span / 2, mid[1] - span / 2, mid[1] + span / 2)


def _figure(cap: dict, lims: tuple, map_only: bool = False) -> tuple:
    """The figure and a redraw closure over it.

    Three panels by default (map, altitude, convergence); ``map_only`` drops the side panels for
    a square top-down animation that carries on a slide without a legend of its own.
    """
    if map_only:
        fig = plt.figure(figsize=(7.0, 7.0))
        ax_map = fig.add_subplot(1, 1, 1)
        ax_alt = ax_conv = None
    else:
        fig = plt.figure(figsize=(13.5, 6.2))
        grid = fig.add_gridspec(2, 2, width_ratios=[1.45, 1.0], hspace=0.45, wspace=0.22)
        ax_map = fig.add_subplot(grid[:, 0])
        ax_alt = fig.add_subplot(grid[0, 1])
        ax_conv = fig.add_subplot(grid[1, 1])
    labels = _move_labels(cap)

    def redraw(f: int) -> None:
        ax_map.clear()
        draw_state(ax_map, cap, f, lims)
        if not map_only:
            ax_alt.clear()
            ax_conv.clear()
            draw_altitude(ax_alt, cap, f)
            draw_convergence(ax_conv, cap, f)
        move = f"move {f}/{len(labels) - 1} — {labels[f]}"
        if map_only:  # no room for a suptitle on a square figure; fold it into the panel title
            ax_map.set_title(
                f"{move}\nt = {cap['t']:.2f} s · {cap['ibr_mode']} · {cap['ibr_iters']} passes",
                fontsize=10, loc="left",
            )
        else:
            ax_map.set_title(move, fontsize=11, loc="left")
        handles = [
            Line2D([], [], color=EGO_C, lw=3, label="ego plan"),
            Line2D([], [], color=OPP_C, lw=3, label="opponent plan"),
            Line2D([], [], color="0.5", lw=2, ls="--", label="previous move"),
            Line2D([], [], color="0.6", lw=1, label="MPPI mode candidates"),
        ]
        ax_map.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)
        if map_only:  # no side panels to balance against, so square up the margins
            fig.tight_layout()
        else:
            fig.suptitle(
                f"MPPI + iterative best response — one control step frozen at "
                f"t = {cap['t']:.2f} s ({cap['ibr_mode']}, {cap['ibr_iters']} passes)",
                fontsize=13,
            )

    return fig, redraw


def render_animation(
    cap: dict, out: Path, lims: tuple, hold: int, fps: int, fmt: str, map_only: bool = False
) -> Path:
    """Write the animation, holding each move for `hold` frames and the fixed point for longer."""
    n = len(cap["traj"])
    order = [f for f in range(n) for _ in range(hold)] + [n - 1] * (2 * hold)
    fig, redraw = _figure(cap, lims, map_only)
    path = out.with_suffix(".mp4" if fmt == "mp4" else ".gif")
    writer = (
        FFMpegWriter(fps=fps, bitrate=3000) if fmt == "mp4" else PillowWriter(fps=fps)
    )
    with writer.saving(fig, str(path), dpi=110):
        for f in order:
            redraw(f)
            writer.grab_frame()
    plt.close(fig)
    return path


def render_storyboard(cap: dict, out: Path, lims: tuple) -> Path:
    """One panel per move, for slides that cannot play an animation."""
    n = len(cap["traj"])
    cols = min(n, 4)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(
        rows, cols, figsize=(4.0 * cols, 4.4 * rows), squeeze=False,
        layout="constrained",  # titles above every panel, without colliding with the x labels
    )
    labels = _move_labels(cap)
    for f in range(rows * cols):
        ax = axes[f // cols][f % cols]
        if f >= n:
            ax.axis("off")
            continue
        draw_state(ax, cap, f, lims, show_bank=False)
        ax.set_title(f"{f}. {labels[f]}", fontsize=9, loc="left")
    verdict = (
        f"the plans stop moving after {_settled_after(cap)} of {n - 1} moves"
        if _outcome(cap) == "converged"
        else f"the choices cycle instead of settling ({cap['ibr_mode']}, {n - 1} moves)"
    )
    fig.suptitle(
        f"Best response at one control step (t = {cap['t']:.2f} s): {verdict}", fontsize=13
    )
    path = out.parent / f"{out.stem}_storyboard.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(
    config: str = "multi_level0.toml",
    out: str = "ibr_convergence",
    seed: int | None = 1,
    freeze_step: int | None = None,
    max_steps: int = 1500,
    min_step: int = 25,
    pick: str = "converged",
    ibr_mode: str | None = None,
    ibr_iters: int | None = None,
    fmt: str = "gif",
    fps: int = 4,
    hold: int = 4,
    margin: float = 0.35,
    full_track: bool = False,
    map_only: bool = False,
    from_cache: str | None = None,
) -> None:
    """Freeze one control step of a two-drone race and animate the best-response solve.

    Args:
        config: config file in ``config/`` (needs an MPPI controller with IBR enabled).
        out: output stem, relative to the repo root.
        seed: environment seed; None uses the config's.
        freeze_step: freeze this controller step instead of searching for one.
        max_steps: environment steps before giving up.
        min_step: ignore steps before this one, to skip the standstill on the start line.
        pick: which step to keep: "converged" (settles, the convergence story),
            "cycling" (never settles -- the Jacobi failure mode), or "moved" (either).
        ibr_mode: override the config's solve order: "scan" (Gauss-Seidel) or "vmap" (Jacobi).
            Pair with ``--pick cycling`` to show why Jacobi is the wrong choice here.
        ibr_iters: override the number of best-response passes. More passes = more frames.
        fmt: "gif" or "mp4".
        fps: animation frame rate.
        hold: rendered frames per best-response move.
        margin: padding (m) around the interaction when framing the map.
        full_track: frame the whole track instead of zooming into the interaction.
        map_only: animate the top-down map alone, dropping the altitude and convergence panels.
        from_cache: redraw the step cached in this ``*_capture.pkl`` instead of flying again.
            Every run without it writes ``<out>_capture.pkl``, so restyling is seconds, not
            minutes. All the flight options are ignored when this is set.
    """
    if pick not in ("converged", "cycling", "moved"):
        raise SystemExit(f"--pick must be converged | cycling | moved, got {pick!r}")
    out_path = ROOT / out
    # Flying the episode to find the step costs minutes; restyling the figure costs seconds.
    # Cache the frozen step so the second is not held hostage to the first.
    cache = Path(from_cache) if from_cache else out_path.parent / f"{out_path.stem}_capture.pkl"
    if from_cache:
        cap = pickle.loads(cache.read_bytes())
        logger.info("Reusing the step captured in %s (t = %.2f s)", cache.name, cap["t"])
    else:
        cap = fly_and_freeze(
            config, seed, freeze_step, max_steps, min_step, pick, ibr_mode, ibr_iters
        )
        cache.write_bytes(pickle.dumps(cap))
    lims = _limits(cap, margin, full_track)
    anim = render_animation(cap, out_path, lims, hold, fps, fmt, map_only)
    board = render_storyboard(cap, out_path, lims)
    shift = _plan_shift(cap)
    logger.info("Wrote %s and %s", anim.name, board.name)
    logger.info(
        "Per-move plan shift [m]: %s", np.array2string(shift, precision=3, suppress_small=True)
    )


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("lsy_drone_racing").setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    fire.Fire(main, serialize=lambda _: None)
