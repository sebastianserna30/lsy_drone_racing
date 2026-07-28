"""Opponent state estimation and kinematic prediction.

All host-side NumPy: this runs once per control step on a handful of drones, not per rollout.

Two responsibilities, deliberately separated:

* :class:`OpponentTracker` — measurement hygiene. Real tracking (mocap) drops out, and a NaN
  reaching the rollout poisons every sample. The tracker holds the last valid state, clocks how
  stale it is, and low-passes the velocity so the anisotropic keep-out heading does not chatter.
* :func:`predict` — where the opponent will be over the horizon, under a kinematic model. Used
  by the ``spline_progress`` and ``const_vel`` opponent models, where the opponent is not
  simulated at all and the ego is scored against these trajectories instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from lsy_drone_racing.control.mppi.reference import ReferencePaths

STATE_KEYS = ("pos", "quat", "vel", "ang_vel")


@dataclass(frozen=True)
class TrackerParams:
    """Filtering and staleness parameters for :class:`OpponentTracker`."""

    ctrl_dt: float
    vel_ema: float  # EMA low-pass coefficient on the observed opponent velocity
    stale_inflate_rate: float  # keep-out inflation per second of staleness
    stale_max: float  # staleness cap (s), so inflation tops out at 1 + rate * max


class OpponentTracker:
    """Holds the last valid opponent state and reports how much to inflate the keep-out.

    Agent 0 is the ego and is never filtered: its own state comes from the flight controller,
    not from tracking another vehicle.
    """

    def __init__(
        self,
        n_agents: int,
        initial_obs: dict[str, NDArray[np.floating]],
        params: TrackerParams,
    ):
        """Seed the held state from the reset observation.

        Args:
            n_agents: total agents including the ego.
            initial_obs: reset observation; per-drone keys may be (3,) or (A, 3).
            params: filtering and staleness parameters.
        """
        self.n_agents = n_agents
        self.params = params
        self.held = {k: _agents_view(initial_obs[k]) for k in STATE_KEYS}
        self.vel_filt = self.held["vel"].copy()  # (A, 3)
        self.stale_t = np.zeros(n_agents)  # seconds since the last valid sample

    def update(
        self,
        pos: NDArray[np.floating],
        quat: NDArray[np.floating],
        vel: NDArray[np.floating],
        ang_vel: NDArray[np.floating],
    ) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
        """Validate and filter this step's opponent measurements.

        A non-finite sample means the measurement is missing, so the last valid state is
        substituted and the staleness clock advances. The keep-out is inflated with staleness
        (capped), because a stale opponent could be anywhere within its reachable set.

        Args: each (A, ...) with agent 0 the ego.

        Returns:
            (pos, quat, vel, ang_vel) with dropouts substituted, plus per-opponent keep-out
            inflation of shape (n_agents - 1,).
        """
        pos, quat = pos.copy(), quat.copy()
        vel, ang_vel = vel.copy(), ang_vel.copy()
        inflate = np.ones(self.n_agents - 1, dtype=np.float32)

        for a in range(1, self.n_agents):
            sample = (pos[a], quat[a], vel[a], ang_vel[a])
            if all(np.all(np.isfinite(x)) for x in sample):
                self.stale_t[a] = 0.0
                for key, value in zip(STATE_KEYS, sample):
                    self.held[key][a] = value
                self.vel_filt[a] = (
                    self.params.vel_ema * vel[a] + (1.0 - self.params.vel_ema) * self.vel_filt[a]
                )
            else:
                self.stale_t[a] += self.params.ctrl_dt
                pos[a] = self.held["pos"][a]
                quat[a] = self.held["quat"][a]
                vel[a] = self.held["vel"][a]
                ang_vel[a] = self.held["ang_vel"][a]
            inflate[a - 1] = (
                1.0
                + min(self.stale_t[a], self.params.stale_max) * self.params.stale_inflate_rate
            )
        return pos, quat, vel, ang_vel, inflate


def _agents_view(v: NDArray) -> NDArray:
    """Normalize a per-drone observation to a writable (A, ...) float64 array."""
    v = np.asarray(v, dtype=np.float64)
    return (v[None, :] if v.ndim == 1 else v).copy()


@dataclass(frozen=True)
class PredictorParams:
    """Parameters of the kinematic opponent predictors."""

    model: str  # "spline_progress" or "const_vel"
    horizon: int  # N rollout steps
    dt: float  # planning step (s)
    v_theta_max: float  # cap on predicted progress speed
    offset_tau: float  # decay time (s) of the opponent's off-spline offset


def predict(
    params: PredictorParams,
    paths: ReferencePaths,
    held_pos: NDArray[np.floating],
    vel_filt: NDArray[np.floating],
    theta: list[float],
    n_agents: int,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Kinematic opponent predictions for the rollout cost (opp-aware fix #2).

    Returns (pos, vel), each (N, n_agents-1, 3), sampled at the rollout cost times
    dt, 2*dt, ..., N*dt (the cost is evaluated AFTER each sim step).

    spline_progress: the opponent's re-anchored theta is advanced at its observed forward
    speed (filtered velocity projected on the path tangent); the current off-spline offset
    decays with offset_tau, so a real opponent flying a different line than our guessed spline
    is still predicted near where it actually is.
    const_vel: straight-line extrapolation of the filtered velocity.

    Args:
        params: predictor selection and timing.
        paths: reference LUTs, used by spline_progress.
        held_pos: (A, 3) last valid positions.
        vel_filt: (A, 3) EMA-filtered velocities.
        theta: per-agent progress, already re-anchored this step.
        n_agents: total agents including the ego.
    """
    t_steps = np.arange(1, params.horizon + 1) * params.dt  # (N,)
    preds_p, preds_v = [], []
    for a in range(1, n_agents):
        p0 = held_pos[a]
        v0 = vel_filt[a]
        if params.model == "const_vel":
            pos_a = p0[None, :] + v0[None, :] * t_steps[:, None]
            vel_a = np.broadcast_to(v0, (params.horizon, 3)).copy()
        else:  # spline_progress
            g = paths.theta_grid_np[a]
            lut_p = paths.pos_np[a]
            lut_t = paths.tan_np[a]
            th0 = float(theta[a])  # re-anchored from the observed pos by the caller
            tan0 = np.array([np.interp(th0, g, lut_t[:, i]) for i in range(3)])
            tan0 /= max(np.linalg.norm(tan0), 1e-9)
            v_prog = float(np.clip(np.dot(v0, tan0), 0.0, params.v_theta_max))
            th = np.clip(th0 + v_prog * t_steps, 0.0, paths.s_total_np[a])
            pos_a = np.stack([np.interp(th, g, lut_p[:, i]) for i in range(3)], axis=-1)
            tan = np.stack([np.interp(th, g, lut_t[:, i]) for i in range(3)], axis=-1)
            tan /= np.clip(np.linalg.norm(tan, axis=-1, keepdims=True), 1e-9, None)
            vel_a = v_prog * tan
            # start the prediction at the REAL opponent: decay its off-spline offset
            ref0 = np.array([np.interp(th0, g, lut_p[:, i]) for i in range(3)])
            pos_a = pos_a + (p0 - ref0)[None, :] * np.exp(-t_steps[:, None] / params.offset_tau)
        preds_p.append(pos_a)
        preds_v.append(vel_a)
    return (
        np.stack(preds_p, axis=1).astype(np.float32),
        np.stack(preds_v, axis=1).astype(np.float32),
    )
