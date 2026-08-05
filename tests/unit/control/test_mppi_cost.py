"""Tests for the MPPI cost terms.

The opponent terms encode most of the multi-agent behaviour and were previously unreachable
without a GPU sim and a full controller. These pin down the properties the racing logic relies
on: that being behind a moving opponent costs more than being alongside it, that the hard core
cannot be squeezed through, and that the wake penalty is asymmetric in z.
"""

from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from lsy_drone_racing.control.mppi import cost

W = 4


@pytest.fixture
def params() -> cost.OpponentCostParams:
    """Keep-out params with the anisotropic bubble switched on."""
    return cost.OpponentCostParams(
        drone_radius=0.12,
        drone_exp=2000.0,
        use_anisotropic=True,
        axial=0.25,
        lateral=0.10,
        core_radius=0.18,
        blend_v0=0.2,
        blend_width=0.05,
        downwash=1000.0,
        downwash_radius=0.3,
        downwash_dz=0.6,
    )


@pytest.fixture
def weights() -> dict[str, float]:
    """Cost weights with one term active at a time where possible."""
    return {
        "lag": 1.0, "contour": 1.0, "z": 0.0, "progress": 1.0, "ang_vel": 0.0, "ang_acc": 0.0,
        "tilt": 0.0, "thrust": 0.0, "yaw": 0.0, "obstacle": 1000.0, "floor": 500.0,
        "floor_z": 0.1,
    }


def terms_at(positions: np.ndarray, weights: dict, **over: float) -> dict:
    """Evaluate single_drone_terms for drones at `positions` tracking the origin along +x."""
    n = len(positions)
    pos = jnp.asarray(positions, dtype=jnp.float32)
    zeros3 = jnp.zeros((n, 3))
    return cost.single_drone_terms(
        pos=pos,
        ang_vel=zeros3,
        ang_acc=zeros3,
        cmd=jnp.zeros((n, 4)),
        ref_pos=jnp.zeros((n, 3)),
        tangent=jnp.tile(jnp.array([1.0, 0.0, 0.0]), (n, 1)),
        v_theta=jnp.ones((n,)),
        obstacles=jnp.array([[10.0, 10.0, 0.0]]),
        gate_frame_obstacles=jnp.array([[10.0, 10.0, 0.0]]),
        save_dist_obst=over.get("save_dist_obst", 0.175),
        save_dist_gate_obst=0.22,
        w=weights,
    )


@pytest.mark.unit
def test_lag_and_contour_split_the_position_error(weights: dict):
    """Error along the tangent is lag; error perpendicular to it is contour."""
    # tangent is +x: an x-offset is pure lag, a y-offset is pure contour
    t = terms_at(np.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]]), weights)
    lag, contour = np.asarray(t["lag"]), np.asarray(t["contour"])
    assert lag[0] == pytest.approx(0.25) and contour[0] == pytest.approx(0.0, abs=1e-6)
    assert lag[1] == pytest.approx(0.0, abs=1e-6) and contour[1] == pytest.approx(0.25)


@pytest.mark.unit
def test_progress_is_a_reward_not_a_penalty(weights: dict):
    """Advancing theta must lower the cost, which is what removes the t_total speed ceiling."""
    t = terms_at(np.zeros((1, 3)), weights)
    assert float(np.asarray(t["progress"])[0]) < 0.0


@pytest.mark.unit
def test_floor_penalty_only_below_floor_z(weights: dict):
    """The floor term must be exactly zero above floor_z and grow quadratically below it."""
    t = terms_at(np.array([[0.0, 0.0, 0.5], [0.0, 0.0, 0.05], [0.0, 0.0, 0.0]]), weights)
    floor = np.asarray(t["floor"])
    assert floor[0] == 0.0
    assert floor[1] > 0.0
    assert floor[2] > floor[1]


@pytest.mark.unit
def test_obstacle_cost_is_a_hit_count(weights: dict):
    """Inside the safety distance the penalty applies; outside it is zero."""
    near = cost.single_drone_terms(
        pos=jnp.array([[0.0, 0.0, 1.0], [5.0, 0.0, 1.0]]),
        ang_vel=jnp.zeros((2, 3)), ang_acc=jnp.zeros((2, 3)), cmd=jnp.zeros((2, 4)),
        ref_pos=jnp.zeros((2, 3)), tangent=jnp.tile(jnp.array([1.0, 0.0, 0.0]), (2, 1)),
        v_theta=jnp.ones((2,)),
        obstacles=jnp.array([[0.0, 0.0, 1.0]]),
        gate_frame_obstacles=jnp.array([[99.0, 99.0, 0.0]]),
        save_dist_obst=0.175, save_dist_gate_obst=0.22, w=weights,
    )
    hits = np.asarray(near["obstacle"])
    assert hits[0] == pytest.approx(weights["obstacle"])
    assert hits[1] == 0.0


@pytest.mark.unit
def test_anisotropic_keepout_penalises_being_behind_more_than_alongside(
    params: cost.OpponentCostParams,
):
    """The whole point of the ellipse: the cheap escape is sideways, not braking.

    The opponent moves along +x, so a drone trailing it at distance d sits on the long axial
    axis (cheap normalized distance -> high cost), while one abeam at the same d sits on the
    short lateral axis.
    """
    d = 0.35
    ego = jnp.array([[-d, 0.0, 1.0], [0.0, d, 1.0]])  # behind, then alongside
    opp_pos = jnp.array([[0.0, 0.0, 1.0]])
    opp_vel = jnp.array([[2.0, 0.0, 0.0]])  # well above blend_v0
    coll, _ = cost.opponent_terms(params, ego, opp_pos, opp_vel, jnp.ones(1))
    behind, alongside = np.asarray(coll)
    assert behind > alongside


@pytest.mark.unit
def test_isotropic_when_opponent_is_stationary(params: cost.OpponentCostParams):
    """Below blend_v0 the heading is estimation noise, so the keep-out must be a sphere."""
    d = 0.3
    ego = jnp.array([[-d, 0.0, 1.0], [0.0, d, 1.0]])
    coll, _ = cost.opponent_terms(
        params, ego, jnp.array([[0.0, 0.0, 1.0]]), jnp.zeros((1, 3)), jnp.ones(1)
    )
    assert float(coll[0]) == pytest.approx(float(coll[1]), rel=1e-5)


@pytest.mark.unit
def test_hard_core_cannot_be_squeezed_through_the_thin_axis(params: cost.OpponentCostParams):
    """opp_lateral (0.10) alone is thinner than two airframes; the core must dominate inside it."""
    ego = jnp.array([[0.0, 0.05, 1.0]])  # 5 cm abeam: inside core_radius=0.18
    opp_vel = jnp.array([[2.0, 0.0, 0.0]])
    opp_pos = jnp.array([[0.0, 0.0, 1.0]])
    with_core, _ = cost.opponent_terms(params, ego, opp_pos, opp_vel, jnp.ones(1))
    no_core, _ = cost.opponent_terms(
        replace(params, core_radius=1e-6), ego, opp_pos, opp_vel, jnp.ones(1)
    )
    # A vanishing core leaves only the thin lateral axis, so the penalty drops. The real core
    # must therefore cost strictly more at this offset.
    assert float(with_core[0]) > float(no_core[0])


@pytest.mark.unit
def test_staleness_inflation_widens_the_keepout(params: cost.OpponentCostParams):
    """A stale opponent measurement must make the ego give it more room, not less."""
    ego = jnp.array([[0.0, 0.45, 1.0]])
    opp_pos, opp_vel = jnp.array([[0.0, 0.0, 1.0]]), jnp.array([[2.0, 0.0, 0.0]])
    fresh, _ = cost.opponent_terms(params, ego, opp_pos, opp_vel, jnp.ones(1))
    stale, _ = cost.opponent_terms(params, ego, opp_pos, opp_vel, jnp.full(1, 1.5))
    assert float(stale[0]) > float(fresh[0])


@pytest.mark.unit
def test_downwash_is_asymmetric_in_z(params: cost.OpponentCostParams):
    """Only the drone BELOW the opponent is in the wake; above it is free."""
    opp_pos, opp_vel = jnp.array([[0.0, 0.0, 1.0]]), jnp.array([[1.0, 0.0, 0.0]])
    below = jnp.array([[0.0, 0.0, 0.7]])
    above = jnp.array([[0.0, 0.0, 1.3]])
    _, wake_below = cost.opponent_terms(params, below, opp_pos, opp_vel, jnp.ones(1))
    _, wake_above = cost.opponent_terms(params, above, opp_pos, opp_vel, jnp.ones(1))
    assert float(wake_below[0]) == pytest.approx(params.downwash)
    assert float(wake_above[0]) == 0.0


@pytest.mark.unit
def test_downwash_ends_below_downwash_dz(params: cost.OpponentCostParams):
    """Far enough below, the wake no longer reaches."""
    opp_pos, opp_vel = jnp.array([[0.0, 0.0, 2.0]]), jnp.array([[1.0, 0.0, 0.0]])
    far = jnp.array([[0.0, 0.0, 2.0 - params.downwash_dz - 0.1]])
    _, wake = cost.opponent_terms(params, far, opp_pos, opp_vel, jnp.ones(1))
    assert float(wake[0]) == 0.0


# Tests for symmetric_collision and behind_contour_factor were removed with those functions: the
# Nash IBR gives every agent the real anisotropic keep-out, and best-responding to the opponent's
# actual plan subsumes the hand-rolled contour relaxation.


@pytest.mark.unit
def test_coupled_cost_matches_a_per_step_loop(params: cost.OpponentCostParams):
    """The post-scan coupled cost must reproduce the in-scan version to float32 rounding.

    Moving the opponent terms out of the rollout scan is a refactor, not a model change, so the
    horizon sum of per-step ``opponent_terms`` has to agree whether the steps are walked one at
    a time or by ``vmap``. It does NOT agree bit-for-bit: vmapping changes XLA's fusion, which
    reassociates the arithmetic. The residual is ~1e-17 absolute, far below anything physical,
    but it does mean this refactor cannot be verified by exact run diffing the way the earlier
    package split was -- behaviour has to be judged over seeds instead.
    """
    rng = np.random.default_rng(0)
    horizon, n_opp, n_sim_agents = 7, 2, 2
    ego_pos = jnp.asarray(rng.normal(size=(horizon, W, 3)), dtype=jnp.float32)
    opp_pos = jnp.asarray(rng.normal(size=(horizon, n_opp, 3)), dtype=jnp.float32)
    opp_vel = jnp.asarray(rng.normal(size=(horizon, n_opp, 3)), dtype=jnp.float32)
    inflate = jnp.asarray(rng.uniform(1.0, 1.5, size=(n_opp,)), dtype=jnp.float32)

    # The old shape: score each step separately, then sum over the horizon.
    per_step = [
        cost.opponent_terms(params, ego_pos[t], opp_pos[t], opp_vel[t], inflate)
        for t in range(horizon)
    ]
    want_coll = jnp.sum(jnp.stack([c for c, _ in per_step]), axis=0)
    want_wake = jnp.sum(jnp.stack([d for _, d in per_step]), axis=0)

    got = cost.build_coupled_cost_fn(params, n_sim_agents)(ego_pos, opp_pos, opp_vel, inflate)

    # Only the ego (agent 0) carries a coupled cost; the rest of the column stays zero.
    np.testing.assert_allclose(
        np.asarray(got["opp_drone"][:, 0]), np.asarray(want_coll), rtol=1e-5, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(got["downwash"][:, 0]), np.asarray(want_wake), rtol=1e-5, atol=1e-12
    )
    assert got["opp_drone"].shape == (W, n_sim_agents)
    np.testing.assert_array_equal(np.asarray(got["opp_drone"][:, 1:]), 0.0)
