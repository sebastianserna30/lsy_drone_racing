"""Horizon rollout of the batched sim.

One ``lax.scan`` over N planning steps, stepping every world and every simulated agent together
and accumulating the per-term cost. This is the expensive part of a control step.

Built via a factory (the same shape as ``ta_mppi/rollouts.py``): :func:`build_rollout_fn` closes
over the sim, the step function, the horizon geometry and the cost callable, and returns a
jitted rollout. Closing over them rather than passing them keeps every scalar a compile-time
constant in the trace.

Shapes: W = rollout worlds, A = simulated agents, N = horizon steps, nu = 5 control channels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp
from jax import Array
from jax.lax import scan

if TYPE_CHECKING:
    from crazyflow.sim import Sim
    from crazyflow.sim.data import SimData


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RolloutCache:
    """Per-step rollout state that the POST-SCAN coupled cost is allowed to read.

    Only quantities a coupled term actually needs belong here. Decoupled terms are summed
    inside the scan and never cached, which is what keeps this narrow: every scalar channel
    added costs ``N * W * A * 4`` bytes -- 24 MB at the deployed geometry (N=30, W=100000,
    A=2) -- and is re-read once per agent per best-response iteration, so an over-wide cache
    costs bandwidth on every pass rather than just once.

    ``pos`` is the array the rollout already had to materialise for rendering, so it is free.
    ``opp_pos``/``opp_vel`` are only (N, P, 3): the handful of opponent trajectories the ego is
    scored against, gathered inside the scan rather than cached in full.
    """

    pos: Array  # (N, W, A, 3) every simulated agent's trajectory
    opp_pos: Array  # (N, P, 3) opponent poses the coupled cost scores against
    opp_vel: Array  # (N, P, 3) matching velocities, for the anisotropic keep-out heading


# (obs, theta0, infos) -> (decoupled total (W,A), cache, decoupled terms dict of (W,A))
RolloutFn = Callable[[dict, Array, tuple], tuple[Array, RolloutCache, dict[str, Array]]]

# (data, theta, v_theta, obstacles, gate_obstacles) -> decoupled terms
StageCostFn = Callable[..., dict[str, Array]]

# (data, opp_pred_pos_t, opp_pred_vel_t) -> (opp pos (P,3), opp vel (P,3))
OppStateFn = Callable[..., tuple[Array, Array]]


def build_rollout_fn(
    sim: Sim,
    step_fn: Callable,
    stage_cost_fn: StageCostFn,
    opp_state_fn: OppStateFn,
    horizon: int,
    dt: float,
    s_total: Array,
) -> RolloutFn:
    """Bake the sim and the decoupled cost into a jitted horizon rollout.

    The rollout accumulates only the decoupled cost; the opponent-coupled terms are computed
    afterwards from the returned :class:`RolloutCache`, so best-response iteration can re-score
    them without re-running the sim.

    Args:
        sim: the batched rollout sim (n_worlds = n_samples, n_drones = n_sim_agents).
        step_fn: the sim's built step function.
        stage_cost_fn: per-step decoupled cost, returning term_name -> (W, A).
        opp_state_fn: per-step opponent pose the coupled cost will score against.
        horizon: N planning steps.
        dt: planning step duration (s), used to integrate progress theta.
        s_total: (A_sim,) arc length of each simulated agent's path, to clamp theta.

    Returns:
        A jitted ``rollout(obs, theta0, infos)``.
    """
    steps_per_ctrl = sim.freq // sim.control_freq

    def apply_input(
        carry: tuple[SimData, Array], info: tuple[Array, ...]
    ) -> tuple[tuple[SimData, Array], tuple[dict[str, Array], Array]]:
        """Advance one planning step (all agents) and score the resulting state.

        Args:
            carry: (sim data, theta) where theta is (W, A) progress in arc length.
            info: (cmd5, obstacles, gate_frame_obstacles, opp_pred_pos, opp_pred_vel). cmd5 is
                (W, A, 5): the first 4 channels are the attitude/thrust command sent to the
                sim; the 5th is v_theta, the progress speed used to integrate theta.
                opp_pred_pos/opp_pred_vel are this step's kinematic opponent predictions
                (P, 3), used only by the predictor opponent models.
        """
        data, theta = carry
        cmd, obstacles, gate_frame_obstacles, opp_pred_pos, opp_pred_vel = info
        v_theta = cmd[..., 4]  # (W, A)
        cmd4 = cmd[..., :4]  # (W, A, 4) real attitude/thrust channels
        # Integrate progress forward one planning step (clamped per agent to its own path end).
        theta_next = jnp.clip(theta + v_theta * dt, 0.0, s_total)
        data = data.replace(
            controls=data.controls.replace(
                attitude=data.controls.attitude.replace(staged_cmd=cmd4)
            )
        )
        next_data = step_fn(data, steps_per_ctrl)
        cost_terms = stage_cost_fn(
            next_data, theta_next, v_theta, obstacles, gate_frame_obstacles
        )
        opp_pos, opp_vel = opp_state_fn(next_data, opp_pred_pos, opp_pred_vel)
        return (next_data, theta_next), (cost_terms, next_data.states.pos, opp_pos, opp_vel)

    def rollout(
        obs: dict[str, Array], theta0: Array, infos: tuple[Array, ...]
    ) -> tuple[Array, RolloutCache, dict[str, Array]]:
        """Roll the sim forward over the horizon from the current state.

        Args:
            obs: current state, values (A_sim, ...) broadcast over worlds.
            theta0: (W, A_sim) starting progress; each agent's worlds all start at that agent's
                committed progress.
            infos: (controls (N,W,A,nu), obstacles, gate_frame_obstacles, opp_pred_pos (N,P,3),
                opp_pred_vel (N,P,3)).

        Returns:
            (decoupled total cost (W, A), the :class:`RolloutCache`, per-term horizon sums
            (W, A)). The coupled terms are not included in the total -- the caller adds them
            after best-response selection.
        """
        data = sim.data
        pos = data.states.pos.at[...].set(obs["pos"])
        quat = data.states.quat.at[...].set(obs["quat"])
        vel = data.states.vel.at[...].set(obs["vel"])
        ang_vel = data.states.ang_vel.at[...].set(obs["ang_vel"])
        rotor_vel = data.states.rotor_vel.at[...].set(obs["rotor_vel"])
        data = data.replace(
            states=data.states.replace(
                pos=pos, quat=quat, vel=vel, ang_vel=ang_vel, rotor_vel=rotor_vel
            )
        )
        controls_flat, obstacles, gate_frame_obstacles, opp_pred_pos, opp_pred_vel = infos
        # scan iterates over axis 0 (N time steps); tile obstacles so each step gets a slice
        obstacles_tiled = jnp.broadcast_to(obstacles[None], (horizon,) + obstacles.shape)
        gate_frame_obstacles_tiled = jnp.broadcast_to(
            gate_frame_obstacles[None], (horizon,) + gate_frame_obstacles.shape
        )

        # Carry (sim data, theta) so each rollout integrates its own progress.
        _, (cost_terms, positions, opp_pos, opp_vel) = scan(
            apply_input,
            (data, theta0),
            (
                controls_flat,
                obstacles_tiled,
                gate_frame_obstacles_tiled,
                opp_pred_pos,
                opp_pred_vel,
            ),
        )
        # cost_terms: dict of (N, W, A). Sum each term over the horizon -> (W, A).
        terms_summed = {k: jnp.sum(v, axis=0) for k, v in cost_terms.items()}
        total = sum(terms_summed.values())  # (W, A)
        cache = RolloutCache(pos=positions, opp_pos=opp_pos, opp_vel=opp_vel)
        return total, cache, terms_summed

    return jax.jit(rollout)
