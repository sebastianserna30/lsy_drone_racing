"""Parameter sweeps for the multi-agent study plots, written out as CSV.

Several studies, one harness:

* ``opponent_speed`` sweeps the OPEN-LOOP opponent's ``[controller.attitude].t_total`` and reports
  how often both drones finish and how often we overtake. That opponent never reacts to us, so its
  finish time in a contested run is also its uncontested time -- no separate solo runs are needed
  to label the x axis.
* ``opp_progress`` is the same idea for an opponent that is itself an MPPI: its pace knob is
  ``controller[0][mppi][cost].progress``, not the open-loop spline's ``t_total``. Sweeping the
  OPPONENT rather than the ego is what creates the slow/fast regimes in which a pass is available
  at all -- with the opponent pinned at one pace, ``overtake_pct`` is 0 by construction.
* ``progress`` sweeps the EGO's ``[controller.mppi.cost].progress`` and reports our lap time
  against our finish rate, i.e. the speed/safety trade. Note the lap-time panel of this study is
  close to tautological (more progress weight, faster lap); the finish-rate panel is the
  informative one.

Every parameter value is run on the SAME seed list (paired design). Pairing removes track
randomisation from the comparison, which is where nearly all the statistical power comes from --
an unpaired sweep needs several times as many runs to say anything.

Run as:

    $ python scripts/sweep_study.py --study opponent_speed --n_seeds 30
    $ python scripts/sweep_study.py --study progress --n_seeds 30 --n_samples 10000
    $ python scripts/sweep_study.py --study opp_progress --config study_gt_opp.toml --n_seeds 40

Writes ``sweep_<study>_<timestamp>.csv`` with one row per parameter value, plus a per-run CSV so
the raw data can be re-analysed without re-running.
"""

from __future__ import annotations

import copy
import csv
import logging
import tempfile
from datetime import datetime
from pathlib import Path

import fire
import gymnasium
import numpy as np
import toml
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

from lsy_drone_racing.utils import load_config, load_controller  # isort: skip

logger = logging.getLogger(__name__)
ROOT = Path(__file__).parents[1]

# Defaults chosen to bracket the current settings (t_total 7.8, progress 1.5).
OPPONENT_T_TOTALS = (4.8, 5.3, 5.8, 6.3, 7.1, 7.8, 8.4, 10.8)
PROGRESS_VALUES = (0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.5)
# Opponent pace, from well below our pace to above it, so the grid spans "we should get past it"
# through "it should get away from us".
OPP_PROGRESS_VALUES = (0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5)
# Log-spaced between the old inert value (5, which let the ego fly through the opponent) and the
# current 2000 (which stops collisions but also stops overtaking). The usable setting is the
# smallest one that keeps the finish rate up while still allowing a pass.
DRONE_EXP_VALUES = (5.0, 25.0, 100.0, 250.0, 500.0, 1000.0, 2000.0)


def _episode(
    cfg, classes: list, names: list[str], env: JaxToNumpy, stop_on_ego_finish: bool = False
) -> tuple[np.ndarray, bool]:
    """Run one episode; return each drone's finish time (nan if it never finished) and ego_first.

    With ``stop_on_ego_finish`` the episode ends the moment the ego (drone 1) crosses the last
    gate, instead of waiting for the opponent to finish or for the 30 s cap. Against a slow
    opponent that is most of the episode, so it is a large saving.

    The catch is that the opponent then has no finish time, and the old "overtake" definition
    (both finished, ours lower) would score every win as a loss. So overtaking is keyed on
    ``ego_first`` instead: the ego crossed the last gate first. The ego starts behind, so that is
    a pass. ``opp_alive`` records whether the opponent was still flying at that moment, which is
    the stricter reading (it did not simply crash out) -- kept so either definition can be
    recovered from the per-run CSV without re-running.
    """
    obs, info = env.reset()
    n_drones = env.unwrapped.sim.n_drones
    freqs = np.array([kw["freq"] for kw in cfg.env.kwargs], dtype=np.int64)
    base_freq = int(freqs.max())
    periods = base_freq // freqs

    controllers = []
    for rank, cls in enumerate(classes):
        ctrl_cfg = copy.deepcopy(cfg)
        ctrl_cfg.env.freq = np.int64(ctrl_cfg.env.kwargs[rank]["freq"])
        controllers.append(cls(obs, info | {"rank": rank}, ctrl_cfg))

    finish = np.full(n_drones, np.nan, dtype=np.float64)
    actions = np.zeros((n_drones, env.action_space.shape[1]), dtype=np.float32)
    opp_alive = False
    i = 0
    while True:
        disabled = np.asarray(env.unwrapped.data.disabled_drones[0])
        mask = (i % periods) == 0
        for rank, ctrl in enumerate(controllers):
            if not disabled[rank] and mask[rank]:
                actions[rank] = ctrl.compute_control(obs, info | {"rank": rank})
        obs, reward, terminated, truncated, info = env.step(actions)
        # Open-loop controllers advance their clock here; skipping it freezes them on the pad.
        for rank, ctrl in enumerate(controllers):
            if not disabled[rank] and mask[rank]:
                ctrl.step_callback(
                    actions[rank], obs, reward, terminated, truncated, info | {"rank": rank}
                )
        gate = np.asarray(obs["target_gate"]).ravel()
        newly = (gate == -1) & np.isnan(finish)
        finish[newly] = i / base_freq
        i += 1
        # Ego is drone 1; drone 0 is the opponent. Read liveness AFTER the step so it describes
        # the moment the ego crossed, not the step before it.
        if stop_on_ego_finish and not np.isnan(finish[1]):
            opp_alive = bool(
                np.isnan(finish[0]) and not np.asarray(env.unwrapped.data.disabled_drones[0])[0]
            )
            break
        if terminated or truncated or not np.isnan(finish).any():
            break
    for ctrl in controllers:
        ctrl.episode_callback()
        ctrl.episode_reset()
    # "We crossed the line first" -- true whether or not the opponent ever finished. Defined the
    # same way in both stopping modes, so runs from either are directly comparable.
    ego_first = bool(
        not np.isnan(finish[1]) and (np.isnan(finish[0]) or finish[1] < finish[0])
    )
    return finish, ego_first, opp_alive


def _one_run(
    raw: dict, seed: int, tmp: Path, stop_on_ego_finish: bool = False
) -> tuple[np.ndarray, bool, bool]:
    """Run one seed at one parameter setting in a freshly built env."""
    raw = copy.deepcopy(raw)
    raw["env"]["seed"] = seed
    raw["sim"]["render"] = False
    path = tmp / f"cfg_{seed}.toml"
    path.write_text(toml.dumps(raw))
    cfg = load_config(path)
    names = [c["file"] for c in cfg.controller]
    classes = [load_controller(ROOT / "lsy_drone_racing" / "control" / n) for n in names]
    freqs = np.array([kw["freq"] for kw in cfg.env.kwargs], dtype=np.int64)
    # Seed through the CONSTRUCTOR: reset(seed=...) takes a different path and would not
    # match env.seed in the TOML, making the runs unreplayable.
    env = JaxToNumpy(gymnasium.make(
        "MultiDroneRacing-v0",
        freq=int(freqs.max()),
        sim_config=cfg.sim,
        track=cfg.env.track,
        sensor_range=cfg.env.kwargs[0]["sensor_range"],
        control_mode=cfg.env.kwargs[0]["control_mode"],
        disturbances=cfg.env.get("disturbances"),
        randomizations=cfg.env.get("randomizations"),
        seed=seed,
    ))
    result = _episode(cfg, classes, names, env, stop_on_ego_finish)
    env.close()
    return result


def _summarise(raw_rows: list[dict], grid: list[float]) -> list[dict]:
    """Aggregate the per-run records into one summary row per parameter value."""
    rows = []
    for value in grid:
        sel = [r for r in raw_rows if r["value"] == value]
        if not sel:
            continue
        opp = np.array([r["opp_time"] for r in sel], dtype=np.float64)
        ours = np.array([r["our_time"] for r in sel], dtype=np.float64)
        n = len(sel)
        both = np.isfinite(opp) & np.isfinite(ours)
        # "Overtake" = we cross the last gate first. The ego starts behind, so that is a pass.
        # Keyed on ego_first rather than (both finished & ours < opp) because with early stopping
        # the opponent has no finish time in exactly the runs we won.
        overtake = np.array([bool(r["ego_first"]) for r in sel])
        # Stricter reading: we got past while it was still racing, rather than it crashing out.
        overtake_live = np.array([bool(r["ego_first"] and r["opp_alive"]) for r in sel])
        p = float(np.isfinite(ours).mean())
        rows.append({
            "value": value,
            "opponent_time": float(np.nanmean(opp)) if np.isfinite(opp).any() else float("nan"),
            # With stop_on_ego_finish this collapses -- the opponent is cut off in every run we
            # win -- so it is no longer a safety measure. Read our_success_pct instead.
            "both_finished_pct": 100.0 * both.sum() / n,
            "overtake_pct": 100.0 * overtake.sum() / n,
            "overtake_live_pct": 100.0 * overtake_live.sum() / n,
            # A reactive opponent can crash on its own; when it does, our finish rate flatters us
            # because a dead opponent cannot block. Always read this column alongside the others.
            "opp_success_pct": 100.0 * np.isfinite(opp).sum() / n,
            "our_success_pct": 100.0 * np.isfinite(ours).sum() / n,
            "our_avg_time": float(np.nanmean(ours)) if np.isfinite(ours).any() else float("nan"),
            "n_seeds": n,
            # Binomial standard error on the finish rate, in percentage points, so the plot can
            # carry error bars instead of implying more precision than there is.
            "our_success_se_pp": 100.0 * float(np.sqrt(max(p * (1 - p), 0) / n)),
        })
    return rows


def _write(rows: list[dict], raw_rows: list[dict], out: Path, raw_out: Path) -> None:
    """Rewrite both CSVs from scratch, so they are always complete and consistent on disk."""
    if rows:
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    if raw_rows:
        with raw_out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(raw_rows[0]))
            w.writeheader()
            w.writerows(raw_rows)


def sweep(
    study: str = "progress",
    config: str = "MPPI.toml",
    n_seeds: int = 30,
    n_samples: int | None = None,
    values: str | None = None,
    seed0: int = 1000,
    stop_on_ego_finish: bool = False,
) -> int:
    """Sweep one parameter over a paired seed set and write the plot data as CSV.

    Args:
        study: "opponent_speed" (the open-loop opponent's t_total), "opp_progress" (an MPPI
            opponent's progress weight, i.e. its pace), "progress" (the EGO's progress weight), or
            "drone_exp" (the opponent keep-out weight, which trades collisions against the ability
            to overtake).
        config: config file in ``config/``.
        n_seeds: seeds per parameter value. See the module docstring on why this should not be
            small; 10 gives a 95% CI of roughly +/- 30 percentage points on a rate near 50%.
        n_samples: override the MPPI sample count. Note reduced sampling changes outcomes, so
            headline numbers should use the deployed value.
        values: comma-separated parameter values, overriding the defaults.
        seed0: first seed; the set is seed0, seed0+1, ... Fixed across parameter values (paired).
        stop_on_ego_finish: end each episode as soon as the ego crosses the last gate, rather
            than waiting for the opponent or the 30 s cap. Much faster against a slow opponent.
            Costs the opponent's lap time (and so both_finished_pct) in every run we win;
            our_success_pct, our_avg_time and overtake_pct are unaffected.

    Returns:
        Process exit code.
    """
    studies = ("opponent_speed", "opp_progress", "progress", "drone_exp")
    if study not in studies:
        raise ValueError(f"study must be one of {studies}")
    default_grid = {
        "opponent_speed": OPPONENT_T_TOTALS,
        "opp_progress": OPP_PROGRESS_VALUES,
        "progress": PROGRESS_VALUES,
        "drone_exp": DRONE_EXP_VALUES,
    }
    if values is None:
        grid = list(default_grid[study])
    elif isinstance(values, (list, tuple)):  # fire parses "0.4,1.5" into a tuple
        grid = [float(v) for v in values]
    else:
        grid = [float(v) for v in str(values).split(",") if str(v).strip()]
    seeds = [seed0 + i for i in range(n_seeds)]
    base = toml.load(ROOT / "config" / config)
    # Fail here rather than mid-sweep: an open-loop opponent has no progress weight to sweep, and
    # the reverse mistake (an MPPI opponent under "opponent_speed") has no t_total.
    if study == "opp_progress" and "mppi" not in base["controller"][0]:
        raise ValueError(
            f"study 'opp_progress' needs an MPPI opponent, but controller[0] in {config} has no "
            "[controller.mppi] section (it is likely the open-loop attitude controller)"
        )
    if study == "opponent_speed" and "attitude" not in base["controller"][0]:
        raise ValueError(
            f"study 'opponent_speed' needs an open-loop opponent, but controller[0] in {config} "
            "has no [controller.attitude] section"
        )
    if n_samples is not None:
        base["controller"][1]["mppi"]["n_samples"] = n_samples

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = ROOT / f"sweep_{study}_{stamp}.csv"
    raw_out = ROOT / f"sweep_{study}_{stamp}_runs.csv"
    rows, raw_rows = [], []

    # SEED-MAJOR: each seed runs the whole grid before the next seed starts, and both CSVs are
    # rewritten after every seed. So at any moment the files hold a complete curve at equal n
    # across all x values -- the run can be read, plotted or stopped at any point, instead of
    # having to reach the end of one parameter value before the next one has any data at all.
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for i, seed in enumerate(seeds, start=1):
            for value in grid:
                cfg_raw = copy.deepcopy(base)
                if study == "opponent_speed":
                    cfg_raw["controller"][0]["attitude"]["t_total"] = value
                elif study == "opp_progress":
                    cfg_raw["controller"][0]["mppi"]["cost"]["progress"] = value
                elif study == "drone_exp":
                    cfg_raw["controller"][1]["mppi"]["opponent"]["drone_exp"] = value
                else:
                    cfg_raw["controller"][1]["mppi"]["cost"]["progress"] = value

                finish, ego_first, opp_alive = _one_run(cfg_raw, seed, tmp, stop_on_ego_finish)
                raw_rows.append({
                    "value": value,
                    "seed": seed,
                    "opp_time": finish[0],
                    "our_time": finish[1],
                    "ego_first": int(ego_first),
                    "opp_alive": int(opp_alive),
                })

            rows = _summarise(raw_rows, grid)
            _write(rows, raw_rows, out, raw_out)
            print(f"--- seed {seed} done ({i}/{len(seeds)}) ---", flush=True)
            for row in rows:
                print(
                    f"{study}={row['value']:<6} opp_t={row['opponent_time']:.2f}  "
                    f"oppOK={row['opp_success_pct']:.0f}%  "
                    f"overtake={row['overtake_pct']:.0f}%"
                    f"(live {row['overtake_live_pct']:.0f}%)  "
                    f"ours={row['our_success_pct']:.0f}%+-{row['our_success_se_pp']:.0f}pp  "
                    f"t={row['our_avg_time']:.2f}s  n={row['n_seeds']}",
                    flush=True,
                )

    print(f"\nsummary -> {out}\nper-run  -> {raw_out}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    raise SystemExit(fire.Fire(sweep))
