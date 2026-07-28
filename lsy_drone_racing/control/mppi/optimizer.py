"""One MPPI update step: sample, roll out, re-fit the sampling distribution.

This is the algorithm proper — it calls the sampler, the rollout and the elite update rather
than implementing any of them. Every simulated agent shares a single rollout batch, so the
expensive part is one ``lax.scan``; the surrounding per-agent work is a short Python loop,
needed only because the mode count K may differ between agents.

Shapes: W = rollout worlds (= n_samples), A = simulated agents, N = horizon, nu = 5 channels.
Per agent a: K_a modes, M_a = n_samples // K_a samples per mode, world index = k * M_a + m.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp

from lsy_drone_racing.control.mppi import sampling

if TYPE_CHECKING:
    from jax import Array

    from lsy_drone_racing.control.mppi.rollout import RolloutFn
    from lsy_drone_racing.control.mppi.sampling import SamplerParams


def build_update_fn(
    rollout_fn: RolloutFn,
    sampler: SamplerParams,
    K: list[int],
    M: list[int],
    horizon: int,
    n_samples: int,
    num_inputs: int,
    n_sim_agents: int,
) -> Callable[..., tuple[list, list, list, list, list, list]]:
    """Bake the rollout and sampling geometry into a jitted MPPI update.

    Args:
        rollout_fn: the jitted horizon rollout.
        sampler: sampling hyperparameters and the action box.
        K: per-agent mode counts (may be ragged).
        M: per-agent samples per mode.
        horizon: N planning steps.
        n_samples: rollout worlds.
        num_inputs: control channels (5).
        n_sim_agents: agents whose dynamics are simulated.

    Returns:
        A jitted ``update(key, obs, theta0, means, sigmas, obstacles, gate_frame_obstacles,
        opp_pred_pos, opp_pred_vel, opp_inflate)``.
    """

    def update(
        key: Array,
        obs: dict[str, Array],
        theta0: Array,
        means: list[Array],
        sigmas: list[Array],
        obstacles: Array,
        gate_frame_obstacles: Array,
        opp_pred_pos: Array,
        opp_pred_vel: Array,
        opp_inflate: Array,
    ) -> tuple[list, list, list, list, list, list]:
        """Run one joint MPPI update over all simulated agents.

        Args:
            key: PRNG key for this step.
            obs: current state, values (A_sim, ...).
            theta0: (W, A_sim) starting progress.
            means: per-agent (K_a, N, nu) current means.
            sigmas: per-agent (K_a, N, nu) current standard deviations.
            obstacles: pillar positions.
            gate_frame_obstacles: gate-frame keep-out centres.
            opp_pred_pos: (N, P, 3) opponent predictions; dummy in "mppi" mode, where the
                representatives are gathered in-batch instead.
            opp_pred_vel: (N, P, 3) opponent predicted velocities.
            opp_inflate: (P,) staleness keep-out inflation.

        Returns:
            Per-agent lists: new_means, new_sigmas, best_mode_idx, costs (K_a, M_a),
            positions (K_a, M_a, N, 3), mode_term_costs (term -> (K_a,)).
        """
        agent_keys = list(jax.random.split(key, n_sim_agents))

        noises, cands = [], []
        for a in range(n_sim_agents):
            noise_a = sampling.sample_modes(sampler, agent_keys[a], means[a], sigmas[a], K[a], M[a])
            # Force sample m=0 of every opponent mode to zero noise, so world k*M_a rolls out
            # that mode's mean — the representative the ego collision is scored against.
            if a > 0:
                noise_a = noise_a.at[:, 0].set(0.0)
            noises.append(noise_a)
            cand_a = (
                (means[a][:, None, ...] + noise_a)
                .transpose(2, 0, 1, 3)
                .reshape(horizon, n_samples, num_inputs)
            )
            cands.append(cand_a)
        controls = jnp.stack(cands, axis=2)  # (N, W, A, nu)

        totals, positions, terms = rollout_fn(
            obs,
            theta0,
            (controls, obstacles, gate_frame_obstacles, opp_pred_pos, opp_pred_vel, opp_inflate),
        )

        new_means, new_sigmas, best_idx = [], [], []
        costs_grouped, positions_grouped, mode_term_costs = [], [], []
        for a in range(n_sim_agents):
            Ka, Ma = K[a], M[a]
            costs_a = totals[:, a].reshape(Ka, Ma)
            pos_a = positions[:, :, a, :].transpose(1, 0, 2).reshape(Ka, Ma, horizon, 3)
            terms_a = {name: v[:, a].reshape(Ka, Ma) for name, v in terms.items()}
            # per-mode breakdown: each mode's best (lowest-total) sample -> (Ka,)
            best_sample = jnp.argmin(costs_a, axis=1)
            k_idx = jnp.arange(Ka)
            mterm_a = {name: v[k_idx, best_sample] for name, v in terms_a.items()}
            m, s, bi = sampling.update_modes(
                sampler, means[a], sigmas[a], noises[a], costs_a, Ka, Ma
            )
            new_means.append(m)
            new_sigmas.append(s)
            best_idx.append(bi)
            costs_grouped.append(costs_a)
            positions_grouped.append(pos_a)
            mode_term_costs.append(mterm_a)

        return new_means, new_sigmas, best_idx, costs_grouped, positions_grouped, mode_term_costs

    return jax.jit(update)
