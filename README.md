# Autonomous Drone Racing — MPPI / MPCC Controller

<div align="center">
  <a href="https://github.com/learnsyslab/crazyflow">
    <img width="800" height="450" src="docs/img/race.gif">
  </a>
  <br/>
  <sub><sup style="font-size: 0.8em;"><a href="https://github.com/learnsyslab/crazyflow">Powered by Crazyflow</a></sup></sub>
  </br>
  </br>
</div>

This is our fork of [`learnsyslab/lsy_drone_racing`](https://github.com/learnsyslab/lsy_drone_racing) for the
TUM **Autonomous Drone Racing** practical course (SS26), multi-agent track: two Crazyflie
nano-quadcopters racing head-to-head on a shared course.

Everything upstream is unchanged except the controllers. What we added is a sampling-based
**MPPI controller with an MPCC progress formulation**, an **opponent model**, and a
**game-theoretic (iterative best response) solve** for the two-drone case.

---

## What we built

### Two layers

1. **Reference trajectory** (`control/spline_planner.py`) — a cubic spline through gate
   entry/exit waypoints, built once at reset from `obs["gates_pos"]` and never replanned.
   Waypoints come from each gate's normal and approach direction, with a ring search to place
   detours around obstacles.
2. **MPPI tracking controller** (`control/trajectory_mppi.py` + `control/mppi/`) — at every
   50 Hz step it samples thousands of attitude-command sequences, rolls them out through the
   real `crazyflow.Sim` physics on the GPU, and takes a cost-weighted average of the best ones.

The planner decides *where* to go; MPPI decides *how* to fly there. MPPI never moves the spline.

### MPCC: progress instead of time

The reference is reparameterized by **arc length** rather than time
(`control/mppi/reference.py`). Each rollout carries a progress variable `theta` and advances it
by a sampled progress speed `v_theta` — a fifth control channel next to thrust/roll/pitch/yaw.
Cost becomes contour error (perpendicular to the path), lag error (along it), and a reward for
advancing `theta`.

The practical consequence: the spline's `t_total` stops being a speed ceiling. Under the old
time-keyed reference the drone could never beat the spline's own schedule; under arc-length
parameterization `progress` is the speed knob and `v_theta_max` the cap.

### Multi-modal sampling

`K` clusters × `M` samples each, every cluster carrying its own mean and adaptively-updated
noise sigma (`control/mppi/sampling.py`). A single Gaussian around one mean collapses onto one
side of an obstacle; several modes let the optimizer keep two genuinely different plans alive
(e.g. left vs. right of a pillar) until one clearly wins.

### Opponent handling

Three interchangeable opponent models, selected by `[controller.mppi.opponent].model`:

| `model` | What the opponent is assumed to do | Cost |
| --- | --- | --- |
| `const_vel` | straight-line extrapolation of its filtered velocity | cheapest |
| `spline_progress` | advances along its own inferred spline (used for real flight) | cheap |
| `mppi` | simulated as a full second MPPI agent inside the same rollout batch | expensive |

On top of the prediction sits an **anisotropic keep-out** (`control/mppi/cost.py`): an ellipse
elongated along the opponent's heading rather than a sphere, so passing side-by-side is cheap
while closing from behind is not. A **downwash** term penalizes flying underneath the opponent,
where the sim provides no wake but reality does.

> One hard-won constraint, documented in the config: `drone_exp` **must stay above** `downwash`.
> The wake cylinder only extends downward, so its only exit is upward *through* the other drone.
> Make escaping the wake cheaper than the collision it passes through and the ego climbs
> straight into the opponent.

### Iterative best response (the Nash part)

With `model = "mppi"`, both drones have a bank of scored rollouts. Because the opponent-coupled
cost is a pure function of the resulting trajectories, each agent can be re-scored against the
other's *current* pick without touching the sim again. Repeat until the picks stop moving and
you have a fixed point of the joint game (`control/mppi/ibr.py`).

Two schedules:

- `ibr_mode = "scan"` — Gauss-Seidel, each agent moves in turn and the next one sees it. **Converges.**
- `ibr_mode = "vmap"` — Jacobi, everyone scores against the same snapshot and moves at once.
  Can oscillate forever between two joint choices, with the answer depending on the parity of
  `ibr_iters`. See `test_jacobi_can_oscillate_while_gauss_seidel_settles`.

Use `scan`. Jacobi is kept only because it is the faster shape and the failure is worth showing.

### Package layout

`control/mppi/` — each module owns one concern, all testable without a GPU:

| Module | Concern |
| --- | --- |
| `config.py` | strict config parsing — every key required, unknown keys rejected, no silent defaults |
| `reference.py` | arc-length reparameterization, theta anchoring |
| `sampling.py` | noise sampling, per-mode distribution update |
| `rollout.py` | the horizon `lax.scan` over the batched sim (the expensive part) |
| `cost.py` | MPCC terms, obstacles, floor, opponent keep-out, downwash |
| `optimizer.py` | one MPPI update: sample → roll out → re-fit |
| `opponents.py` | tracking hygiene (dropout, NaN, velocity low-pass) + kinematic prediction |
| `ibr.py` | the iterative best response solve |
| `diagnostics.py` | cost logging and MuJoCo overlays; never affects flight |

---

## Running it

All Python goes through pixi — there is no bare `python` on PATH.

```bash
pixi install -e gpu
```

### Single agent

```bash
pixi run -e gpu python scripts/sim.py --config level0_MPPI.toml --render
pixi run -e gpu python scripts/sim.py --config level2_MPPI.toml --n_runs 10
```

### Two agents

```bash
pixi run -e gpu python scripts/multi_sim.py --config multi_level0.toml --render
pixi run -e gpu python scripts/multi_sim.py --config multi_level0.toml --n_runs 20
```

### Evaluation

```bash
pixi run -e gpu python scripts/evaluate.py        # 20 runs, writes evaluation.csv
pixi run -e gpu python scripts/multi_evaluate.py  # two-drone equivalent
```

### Hardware

```bash
pixi run -e deploy python scripts/deploy.py --config deploy_single.toml
pixi run -e deploy python scripts/deploy.py --config deploy_multi.toml
```

### Tests

```bash
pixi run -e gpu-tests python -m pytest tests/unit -q
```

The MPPI unit tests never start a GPU sim, so they run in seconds. `test_mppi_config.py`
dynamically discovers every config in `config/` that loads an MPPI controller and asserts it
satisfies the strict schema — a new config is covered the moment it is added, which is what
keeps the shipped TOMLs from drifting apart as keys change.

### Configs

| Config | Use |
| --- | --- |
| `level0_MPPI.toml`, `level2_MPPI.toml` | single agent, perfect knowledge / randomized |
| `single.toml`, `deploy_single.toml` | tuned single-agent sim / hardware |
| `multi_level0.toml` | two agents, the main development config |
| `deploy_multi.toml` | two agents on hardware (`spline_progress` opponent) |
| `study_*.toml` | one arm each of the studies below |

---

## Key knobs

Under `[controller.mppi]`:

| Key | Meaning |
| --- | --- |
| `N`, `T` | horizon steps and duration (`dt = T / N`) |
| `n_samples`, `K` | total rollouts, number of modes |
| `temperature`, `elite_percentage` | how sharply the cost-weighted average favours good samples |
| `v_theta_init/max/sigma` | nominal, capped and explored progress speed |

Under `[controller.mppi.cost]` — `progress` is the speed/safety dial. Raise it for a faster lap
and a lower finish rate; the studies below quantify the trade.

Under `[controller.mppi.opponent]` — `model`, `ibr_iters`, `ibr_mode`, the keep-out geometry
(`axial`, `lateral`, `use_anisotropic`), and `downwash`.

---

## Studies

Every sweep uses a **paired design**: each parameter value is run on the same seed list, so
track randomization cancels between arms. That is where nearly all the statistical power comes
from — an unpaired sweep needs several times as many runs to say anything.

```bash
pixi run -e gpu python scripts/sweep_study.py --study progress --n_seeds 40
pixi run -e gpu python scripts/sweep_study.py --study opp_progress --config study_gt_opp.toml --n_seeds 40
pixi run -e gpu python scripts/opponent_solo_pace.py --n_seeds 5
pixi run -e gpu python scripts/bench_ctrl_freq.py --preset all
pixi run -e gpu python scripts/animate_ibr.py --config multi_level0.toml
```

Raw logs are in `study_logs/`. Plots regenerate from the CSVs via `scripts/plot_*.py` — they
are deliberately not committed.

### Does modelling the opponent help?

Three arms × 7 `progress` values × 40 seeds = **840 runs** (`study_logs/progress_study.log`),
ego finish rate:

| `progress` | `const_vel` | `parallel` (joint rollout, IBR off) | IBR |
| ---: | ---: | ---: | ---: |
| 0.1 | 82% | 82% | 90% |
| 1.0 | 68% | 82% | 92% |
| 1.5 | **42%** | **80%** | **90%** |

The gap opens exactly where it should — at high `progress`, when the ego is flying fast enough
for the encounter to matter. Rolling both drones out jointly is worth ~38 pp there; the IBR
passes on top add ~10 pp. At low `progress` everything works, because nothing is contested.

### Can we actually overtake?

Sweeping the *opponent's* pace over 40 seeds × 7 values = **280 runs**
(`study_logs/pass_sweep.log`). Overtake rate falls from **50%** against the slowest opponent
(8.48 s lap) to **22–30%** once the opponent is lapping in 4.7–5.0 s — passing is available when
there is a pace difference to exploit, and not otherwise.

### What does it cost?

`study_logs/bench_ctrl_freq/RESULTS.md`, RTX 4070 Laptop, against the **20 ms** budget of the
50 Hz env step:

| Arm | p50 (ms) | p99 (ms) | Hz (p99) |
| --- | ---: | ---: | ---: |
| `const_vel` (no joint rollout) | 30.6 | 34.1 | 29.3 |
| `parallel` (IBR off) | 62.8 | 66.7 | 15.0 |
| **IBR 3 iters (shipped)** | **90.0** | **94.5** | **10.6** |
| IBR 10 iters | 149.1 | 155.2 | 6.4 |

Decomposed: the joint two-agent rollout is the expensive half (+32 ms), and IBR itself is
~9–10 ms per iteration, dead linear.

**This is the honest headline: the shipped configuration runs at ~10 Hz against a 50 Hz budget,
over deadline on 100% of steps.** It is a simulation result. Closing that gap — fewer samples,
a shorter horizon, or fewer IBR passes — is the open work.

---

## Branches

`main` carries the work described above. Other branches worth knowing:

- **`mppi++`** — the earlier two-agent controller, before the `n_agents`-generic refactor. Still
  useful as a reference point, and the source of the opponent-as-MPPI idea.
- **`mppi++-n-agents`** — where the current `main` was developed.
- **`mppi`**, **`mpc`** — earlier single-agent attempts.

### Opponent model on `mppi++`

On `mppi++` the opponent model is **not** a TOML key. The toggle is a hardcoded attribute in
`lsy_drone_racing/control/trajectory_mppi_multi.py`:

```python
self.opp_mppi = True  # opponent simulated as a second MPPI
self.opp_mppi = False  # opponent as constant-velocity extrapolation
```

With `opp_mppi = True` the opponent is a second `SingleAttitudeMPPIController` sharing the
config but reduced to `K = 1` mode and `n_samples / K` samples. With it `False`, the opponent's
predicted trajectory is straight-line extrapolation, `opp_pos + opp_vel * dt`.

Edit the attribute in the source to switch — there is no config surface for it on that branch.
That surface is exactly what became `[controller.mppi.opponent].model` on `main`, where the
same two behaviours are `"mppi"` and `"const_vel"` and no source edit is needed.

---

## Upstream

### Documentation

The course framework's [official documentation](https://lsy-drone-racing.readthedocs.io/en/latest/getting_started/general.html)
covers installation, the environment interface, and deployment.

### Difficulty levels

Each task setup is defined by a TOML file. The competition environment always uses **level 2**.

|      Evaluation Scenario      | Rand. Inertial Properties | Randomized Obstacles, Gates | Random Tracks |             Notes              |
| :---------------------------: | :-----------------------: | :-------------------------: | :-----------: | :----------------------------: |
| [Level 0](config/level0.toml) |           *No*            |            *No*             |     *No*      |       Perfect knowledge        |
| [Level 1](config/level1.toml) |          **Yes**          |            *No*             |     *No*      |        Adaptive control        |
| [Level 2](config/level2.toml) |          **Yes**          |           **Yes**           |     *No*      |          Re-planning           |
| [Level 3](config/level3.toml) |          **Yes**          |           **Yes**           |    **Yes**    |        Online planning         |
|         **sim2real**          |     **Real hardware**     |           **Yes**           |    **Yes**    | Simulation-to-reality transfer |

### Dependencies

Built on open-source packages from the [Learning Systems Lab (LSY)](https://www.ce.cit.tum.de/lsy/home/) at TUM:

- [**crazyflow**](https://github.com/learnsyslab/crazyflow) – high-speed, high-fidelity drone simulator with strong sim-to-real performance.
- [**drone-models**](https://github.com/learnsyslab/drone-models) – accurate drone models for simulation and model-based control.
- [**drone-controllers**](https://github.com/learnsyslab/drone-controllers) – controllers for the Crazyflie quadrotor.
