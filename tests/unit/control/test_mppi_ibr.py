"""Tests for the iterative best-response solve.

The properties worth pinning down are the ones that make the iteration meaningful rather than
decorative: that it actually reaches a fixed point, that the coupling is mutual (an agent's
choice moves when the other's does), and that the two update schedules agree once settled.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from lsy_drone_racing.control.mppi import cost, ibr

W, N, A = 16, 5, 2
DT = 0.02


@pytest.fixture
def params() -> cost.OpponentCostParams:
    """Keep-out params with the anisotropic bubble on, as the deployed config runs it."""
    return cost.OpponentCostParams(
        drone_radius=0.12,
        drone_exp=1000.0,
        use_anisotropic=True,
        axial=0.25,
        lateral=0.24,
        core_radius=0.18,
        blend_v0=0.2,
        blend_width=0.05,
        downwash=1000.0,
        downwash_radius=0.3,
        downwash_dz=0.6,
    )


def _flat_inflate() -> jnp.ndarray:
    """No staleness anywhere, so the keep-out is at its nominal size."""
    return jnp.ones((A, A), dtype=jnp.float32)


@pytest.mark.unit
def test_headings_backfill_the_first_step():
    """Step 0 has no earlier state to difference, so it borrows step 1's velocity."""
    pos = jnp.asarray(np.arange(N * W * A * 3, dtype=np.float32).reshape(N, W, A, 3))
    vel = ibr.headings_from_positions(pos, DT)
    assert vel.shape == pos.shape
    np.testing.assert_allclose(np.asarray(vel[0]), np.asarray(vel[1]), rtol=0, atol=0)
    np.testing.assert_allclose(
        np.asarray(vel[2]), np.asarray((pos[2] - pos[1]) / DT), rtol=1e-6
    )


@pytest.mark.unit
def test_far_apart_agents_just_take_their_decoupled_optimum(params: cost.OpponentCostParams):
    """With nobody in reach, the coupled term is ~0 and best response must not move the pick."""
    rng = np.random.default_rng(0)
    # Agent 0 near the origin, agent 1 parked 50 m away: far outside any keep-out.
    pos = np.zeros((N, W, A, 3), dtype=np.float32)
    pos[:, :, 0, :] = rng.normal(scale=0.1, size=(N, W, 3))
    pos[:, :, 1, :] = 50.0 + rng.normal(scale=0.1, size=(N, W, 3))
    decoupled = {"track": jnp.asarray(rng.uniform(size=(W, A)), dtype=jnp.float32)}

    solve = ibr.build_ibr_fn(params, A, DT, n_iters=3, mode="scan")
    total, coupled, best = solve(decoupled, jnp.asarray(pos), _flat_inflate())

    np.testing.assert_array_equal(
        np.asarray(best), np.asarray(jnp.argmin(decoupled["track"], axis=0))
    )
    np.testing.assert_allclose(np.asarray(coupled["opp_drone"]), 0.0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(total), np.asarray(decoupled["track"]), atol=1e-6)


@pytest.mark.unit
def test_gauss_seidel_reaches_a_fixed_point(params: cost.OpponentCostParams):
    """Past convergence, extra passes must be a no-op -- that is what makes it a fixed point."""
    rng = np.random.default_rng(1)
    # Overlapping clouds, so the keep-out actually bites and the agents have to negotiate.
    pos = jnp.asarray(rng.normal(scale=0.3, size=(N, W, A, 3)), dtype=jnp.float32)
    decoupled = {"track": jnp.asarray(rng.uniform(size=(W, A)), dtype=jnp.float32)}

    settled = None
    for n_iters in (6, 7, 8):
        _, _, best = ibr.build_ibr_fn(params, A, DT, n_iters, "scan")(
            decoupled, pos, _flat_inflate()
        )
        if settled is None:
            settled = np.asarray(best)
        else:
            np.testing.assert_array_equal(np.asarray(best), settled)


@pytest.mark.unit
def test_jacobi_can_oscillate_while_gauss_seidel_settles(params: cost.OpponentCostParams):
    """Why the deployed default is "scan": simultaneous updates need not converge at all.

    Two drones dodging each other's *previous* choice at the same instant swap places and swap
    back, period two, so the Jacobi answer flips with the parity of n_iters. This pins the
    behaviour down rather than leaving it to be rediscovered as flaky race results.
    """
    rng = np.random.default_rng(1)
    pos = jnp.asarray(rng.normal(scale=0.3, size=(N, W, A, 3)), dtype=jnp.float32)
    decoupled = {"track": jnp.asarray(rng.uniform(size=(W, A)), dtype=jnp.float32)}

    def best_after(n_iters: int, mode: str) -> np.ndarray:
        return np.asarray(
            ibr.build_ibr_fn(params, A, DT, n_iters, mode)(decoupled, pos, _flat_inflate())[2]
        )

    # Jacobi: even and odd iteration counts land on different answers, and keep alternating.
    assert not np.array_equal(best_after(6, "vmap"), best_after(7, "vmap"))
    np.testing.assert_array_equal(best_after(6, "vmap"), best_after(8, "vmap"))
    np.testing.assert_array_equal(best_after(7, "vmap"), best_after(9, "vmap"))
    # Gauss-Seidel on the same problem does not care.
    np.testing.assert_array_equal(best_after(6, "scan"), best_after(7, "scan"))


@pytest.mark.unit
def test_the_coupling_actually_changes_the_choice(params: cost.OpponentCostParams):
    """If best response never moved anyone off their decoupled pick, the whole module is inert.

    Constructed so the decoupled optimum puts both agents on top of each other: the collision
    term has to push at least one of them onto a different rollout.
    """
    rng = np.random.default_rng(2)
    pos = np.asarray(rng.normal(scale=2.0, size=(N, W, A, 3)), dtype=np.float32)
    decoupled = np.asarray(rng.uniform(size=(W, A)), dtype=np.float32)
    # Make sample 0 the cheapest for both agents, and put their sample-0 rollouts in contact.
    decoupled[0, :] = -1.0
    pos[:, 0, 0, :] = 0.0
    pos[:, 0, 1, :] = 0.01
    terms = {"track": jnp.asarray(decoupled)}

    _, _, best = ibr.build_ibr_fn(params, A, DT, 4, "scan")(
        terms, jnp.asarray(pos), _flat_inflate()
    )
    assert not np.array_equal(np.asarray(best), np.zeros(A))


@pytest.mark.unit
def test_both_schedules_agree_once_settled(params: cost.OpponentCostParams):
    """Jacobi and Gauss-Seidel take different paths; on a converging problem they should meet."""
    rng = np.random.default_rng(3)
    pos = jnp.asarray(rng.normal(scale=0.3, size=(N, W, A, 3)), dtype=jnp.float32)
    decoupled = {"track": jnp.asarray(rng.uniform(size=(W, A)), dtype=jnp.float32)}

    _, _, jacobi = ibr.build_ibr_fn(params, A, DT, 8, "vmap")(decoupled, pos, _flat_inflate())
    _, _, seidel = ibr.build_ibr_fn(params, A, DT, 8, "scan")(decoupled, pos, _flat_inflate())
    np.testing.assert_array_equal(np.asarray(jacobi), np.asarray(seidel))


@pytest.mark.unit
def test_unknown_mode_is_rejected(params: cost.OpponentCostParams):
    """A typo in ibr_mode must fail at build time, not silently pick a schedule."""
    with pytest.raises(ValueError, match="ibr_mode"):
        ibr.build_ibr_fn(params, A, DT, 3, "gauss-seidel")
