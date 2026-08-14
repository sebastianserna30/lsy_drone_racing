# MPPI control-step latency

Measured 2026-08-06 on helios (RTX 4070 Laptop, 8 GB) with `scripts/bench_ctrl_freq.py`.
60 timed `compute_control` calls per arm, seed 7, base config `multi_level0.toml`.

Budget: **20 ms** (the 50 Hz env step). `hz_p99 = 1000 / p99_ms` is the rate sustained when every
step must fit its deadline.

- **p50** — median step. Half the steps were faster.
- **p99** — 99th percentile. The tail; only the slowest 1% exceeded it. This is the number that
  matters, because a control loop has a deadline, not an average: a step that overruns is not
  compensated by a fast one after it.
- **isolated** — solver alone, env never stepped (deployment-shaped: in flight nothing else is on
  the GPU). **inloop** — flying, with the env's own physics contending for the same device.

## IBR iterations (100 000 samples, N=30, `ibr_mode="scan"`)

| Arm | Mode | p50 (ms) | p90 (ms) | p99 (ms) | max (ms) | Hz (p99) | Over budget |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `const_vel` (no joint rollout) | isolated | 30.6 | 33.2 | 34.1 | 34.3 | 29.3 | 100% |
| `const_vel` (no joint rollout) | inloop | 30.7 | 34.1 | 36.9 | 38.2 | 27.1 | 100% |
| `parallel` (IBR off, `ibr_iters=-1`) | isolated | 62.8 | 64.5 | 66.7 | 66.8 | 15.0 | 100% |
| `parallel` (IBR off, `ibr_iters=-1`) | inloop | 61.8 | 64.1 | 66.1 | 66.2 | 15.1 | 100% |
| IBR 0 iters | isolated | 60.6 | 64.1 | 66.7 | 68.8 | 15.0 | 100% |
| IBR 0 iters | inloop | 58.1 | 62.3 | 65.5 | 67.5 | 15.3 | 100% |
| IBR 1 iter | isolated | 71.4 | 73.9 | 75.1 | 75.3 | 13.3 | 100% |
| IBR 1 iter | inloop | 68.9 | 72.0 | 73.6 | 74.3 | 13.6 | 100% |
| **IBR 3 iters (shipped)** | **isolated** | **90.0** | **92.2** | **94.5** | **94.5** | **10.6** | **100%** |
| **IBR 3 iters (shipped)** | **inloop** | **89.2** | **96.2** | **104.6** | **107.8** | **9.6** | **100%** |
| IBR 5 iters | isolated | 106.3 | 108.6 | 111.8 | 114.3 | 8.9 | 100% |
| IBR 5 iters | inloop | 102.4 | 105.4 | 109.0 | 109.2 | 9.2 | 100% |
| IBR 10 iters | isolated | 149.1 | 151.7 | 155.2 | 157.0 | 6.4 | 100% |
| IBR 10 iters | inloop | 144.2 | 149.1 | 151.8 | 151.8 | 6.6 | 100% |

Decomposition:

- The **joint 2-agent rollout is the expensive half**: +32 ms going `const_vel` → 2-agent MPPI.
- **IBR itself is ~9–10 ms per iteration**, dead linear (0→1: +10.8; 3→5: +8.2/iter; 5→10: +8.6/iter).
- `ibr_iters=0` costs the same as `parallel`, confirming 0 iters is just seed + final re-score.
- isolated ≈ inloop on every arm, so the env's physics is not distorting the measurement.

## Sample count (`ibr_iters=3`, N=30, isolated)

| Samples | p50 (ms) | p90 (ms) | p99 (ms) | max (ms) | Hz (p99) | Over budget |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 000 | 43.6 | 46.2 | 48.4 | 49.8 | 20.7 | 98.3% |
| 5 000 | 44.4 | 46.4 | 47.6 | 47.8 | 21.0 | 98.3% |
| 10 000 | 44.8 | 46.1 | 47.9 | 48.3 | 20.9 | 100% |
| 25 000 | 44.2 | 45.1 | 48.0 | 48.1 | 20.8 | 100% |
| 50 000 | 45.7 | 47.0 | 47.5 | 47.5 | 21.0 | 100% |
| 100 000 | 88.4 | 90.7 | 91.8 | 91.9 | 10.9 | 100% |

**Samples are free below 50k.** Flat ~44 ms from 1k to 50k, then it doubles at 100k. Below 50k the
30-step sequential `lax.scan` over the rollout physics is launch-bound, not compute-bound — the
last 50 000 samples cost 2× and buy nothing below that. Dropping 100k → 50k takes `ibr_iters=3`
from 11 Hz to 21 Hz for free.

## Horizon (`ibr_iters=3`, 100 000 samples, isolated)

`T` is scaled with `N` so `dt` stays 0.02 s; only the number of scan steps changes.

| N | Horizon (s) | p50 (ms) | p90 (ms) | p99 (ms) | max (ms) | Hz (p99) | Over budget |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 0.2 | 29.9 | 31.2 | 31.4 | 31.5 | 31.9 | 100% |
| 20 | 0.4 | 57.8 | 61.8 | 63.2 | 64.1 | 15.8 | 100% |
| 30 | 0.6 | 89.4 | 91.8 | 93.8 | 94.9 | 10.7 | 100% |
| 40 | 0.8 | 94.0 | 97.8 | 99.4 | 99.5 | 10.1 | 100% |

~3 ms per horizon step at 100k samples, so lookahead trades directly against rate.

## Bottom line

At the shipped settings the IBR arm runs at **~11 Hz against a 50 Hz budget** — 4.7× too slow, and
over budget on 100% of steps. This is invisible in sim, where the env waits for the controller, but
it is a hard wall on hardware.

Best combination of the knobs above (50k samples, `ibr_iters=3`, N=30) reaches ~21 Hz. Getting to
50 Hz means attacking the sequential rollout scan itself, not the knobs.

## Reproducing

```
pixi run -e gpu python scripts/bench_ctrl_freq.py --preset ibr_iters --steps 60
pixi run -e gpu python scripts/bench_ctrl_freq.py --preset samples --steps 60 --mode isolated
pixi run -e gpu python scripts/bench_ctrl_freq.py --preset horizon --steps 60 --mode isolated
```
