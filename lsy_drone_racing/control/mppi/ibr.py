"""Iterative best response over the cached rollouts (the Nash coupling).

Ported from the course reference ``ta_mppi/cost.py`` (``compute_cost_vmap`` / ``compute_cost_scan``).

The idea it rests on: the rollouts are computed once, and the opponent-coupled cost is a pure
function of the resulting trajectories. So each agent can be re-scored against the others'
*current* choice as many times as we like without touching the sim again. Repeat until the
choices stop moving and you have a fixed point of the joint game -- each agent's plan is a best
response to the others', which is what "Nash" means here.

Every agent carries the coupled cost, not just the ego. That is the substantive difference from
the previous scheme, where opponents were scored on their racing line alone and so never reacted
to us: with a one-sided cost the opponent's choice does not depend on ours, the iteration reaches
its fixed point after a single pass, and the whole loop is dead weight. Mutual avoidance is what
makes iterating worth anything.

The bet this makes is explicit: the opponent is assumed to avoid us as hard as we avoid it. A
real opponent that does not yield will not behave like this model, and the ego will have planned
into a gap that never opens.

Shapes: W = rollout worlds, A = simulated agents, N = horizon, P = A - 1 opponents per agent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp
from jax.lax import scan

from lsy_drone_racing.control.mppi.cost import opponent_terms

if TYPE_CHECKING:
    from jax import Array

    from lsy_drone_racing.control.mppi.cost import OpponentCostParams

IBR_MODES = ("vmap", "scan")


def headings_from_positions(pos: Array, dt: float) -> Array:
    """Per-step velocity of every rollout, differenced from the cached positions.

    The anisotropic keep-out needs a heading for each opponent trajectory. Caching the sim's own
    velocity would cost as much memory again as the position cache, and buys nothing: the
    heading is normalised before use, and a one-step displacement is a smoother estimate than
    the instantaneous velocity anyway -- which is the same noise the iso<->aniso speed blend
    exists to suppress.

    Args: pos (N, W, A, 3); dt the planning step (s).
    Returns (N, W, A, 3). Step 0 is backfilled from step 1, which has no earlier state to
    difference against.
    """
    delta = (pos[1:] - pos[:-1]) / dt  # (N-1, W, A, 3)
    return jnp.concatenate([delta[:1], delta], axis=0)


def build_ibr_fn(
    opp_params: OpponentCostParams,
    n_sim_agents: int,
    dt: float,
    n_iters: int,
    mode: str,
) -> Callable[..., tuple[Array, dict[str, Array], Array]]:
    """Bake the keep-out geometry and iteration schedule into a jitted best-response solve.

    Args:
        opp_params: opponent keep-out geometry, shared by every agent.
        n_sim_agents: agents whose dynamics are in the rollout batch. IBR is only meaningful
            when every agent is simulated, i.e. under ``opponent.model = "mppi"``.
        dt: planning step (s), for differencing headings out of the position cache.
        n_iters: best-response passes. 0 gives a single unilateral best response: each agent
            answers the others' decoupled optimum and nobody gets to react to the reaction.
        mode: "scan" for Gauss-Seidel (agents move in turn, each seeing the previous updates
            within the same pass) or "vmap" for Jacobi (all agents move simultaneously,
            matching the reference's ``compute_cost_vmap``).

            Prefer "scan". With two tightly coupled drones Jacobi does not converge: both
            agents dodge the other's *previous* choice at the same instant, so they swap
            positions and swap back, oscillating with period two. The returned answer then
            depends on the parity of ``n_iters``, which is not a property any tuning can fix.
            Gauss-Seidel converges here because the second agent already sees the first one
            move. See ``test_jacobi_can_oscillate_while_gauss_seidel_settles``.

    Returns:
        A jitted ``ibr(decoupled_terms, pos_cache, inflate)`` returning
        (total cost (W, A), coupled terms each (W, A), chosen sample per agent (A,)).
    """
    if mode not in IBR_MODES:
        raise ValueError(f"ibr_mode must be one of {IBR_MODES}, got {mode!r}")
    # Opponent index lists are static: A is tiny, so the agent axis unrolls at trace time.
    # With only two drones this is a single opponent each and the reference's
    # `find_close_drones` top-k is a no-op; it would be needed again beyond ~4 agents.
    others = [[b for b in range(n_sim_agents) if b != a] for a in range(n_sim_agents)]

    def coupled_for(
        agent: int, pos: Array, vel: Array, best: Array, inflate: Array
    ) -> tuple[Array, Array]:
        """Collision + downwash for one agent against the others' currently chosen rollouts."""
        opp = others[agent]
        # best[b] is traced, so these are dynamic gathers -- (N, 3) each, i.e. cheap.
        opp_pos = jnp.stack([pos[:, best[b], b, :] for b in opp], axis=1)  # (N, P, 3)
        opp_vel = jnp.stack([vel[:, best[b], b, :] for b in opp], axis=1)  # (N, P, 3)
        inflate_a = inflate[agent, jnp.asarray(opp)]  # (P,)

        def step(ego_p: Array, opp_p: Array, opp_v: Array) -> tuple[Array, Array]:
            return opponent_terms(opp_params, ego_p, opp_p, opp_v, inflate_a)

        coll, wake = jax.vmap(step)(pos[:, :, agent, :], opp_pos, opp_vel)  # (N, W) each
        return jnp.sum(coll, axis=0), jnp.sum(wake, axis=0)  # (W,) each

    def coupled_all(pos: Array, vel: Array, best: Array, inflate: Array) -> dict[str, Array]:
        """Every agent's coupled terms against a fixed set of choices -> term -> (W, A)."""
        per_agent = [coupled_for(a, pos, vel, best, inflate) for a in range(n_sim_agents)]
        return {
            "opp_drone": jnp.stack([coll for coll, _ in per_agent], axis=1),
            "downwash": jnp.stack([wake for _, wake in per_agent], axis=1),
        }

    def totals_from(decoupled: dict[str, Array], coupled: dict[str, Array]) -> Array:
        """Sum the merged term dict in insertion order, as the non-IBR path does.

        Float addition is not associative, so summing the merged dict left to right rather than
        adding a coupled subtotal keeps this consistent with the single-shot path.
        """
        return sum((decoupled | coupled).values())  # (W, A)

    def ibr(
        decoupled: dict[str, Array], pos: Array, inflate: Array
    ) -> tuple[Array, dict[str, Array], Array]:
        """Solve for a joint best response and return the coupled costs at that fixed point.

        Args:
            decoupled: per-term horizon sums from the rollout, each (W, A).
            pos: (N, W, A, 3) cached rollout positions.
            inflate: (A, A) keep-out inflation, ``inflate[a, b]`` applying when agent a is
                scored against agent b. Only the ego's row reflects real tracking staleness;
                an opponent's view of us is not stale, because we know our own state.

        Returns:
            (total (W, A), coupled terms each (W, A), chosen sample index per agent (A,)).
        """
        vel = headings_from_positions(pos, dt)
        # Seed the iteration with each agent's decoupled optimum: the line it would fly if the
        # others were not there. Coupling then pulls the choices apart.
        best = jnp.argmin(sum(decoupled.values()), axis=0)  # (A,)

        if mode == "vmap":

            def iteration(best: Array, _: None) -> tuple[Array, None]:
                """Jacobi: score every agent against the same snapshot, then all move at once."""
                totals = totals_from(decoupled, coupled_all(pos, vel, best, inflate))
                return jnp.argmin(totals, axis=0), None

        else:

            def iteration(best: Array, _: None) -> tuple[Array, None]:
                """Gauss-Seidel: each agent moves in turn and the next one sees it."""
                for a in range(n_sim_agents):
                    coupled = coupled_all(pos, vel, best, inflate)
                    totals = totals_from(decoupled, coupled)
                    best = best.at[a].set(jnp.argmin(totals[:, a]))
                return best, None

        best, _ = scan(iteration, best, None, length=n_iters)
        # Final re-score, so the returned costs are the ones at the fixed point rather than the
        # ones that produced it. This is also what the MPPI distribution update is fitted to.
        coupled = coupled_all(pos, vel, best, inflate)
        return totals_from(decoupled, coupled), coupled, best

    return jax.jit(ibr)
