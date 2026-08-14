"""Tests for the MPPI sampler and per-mode distribution update.

These exercise the math directly, with no GPU sim and no controller instance — which was the
point of pulling them out of the controller class.
"""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lsy_drone_racing.control.mppi import sampling

N, NU, K, M = 6, 5, 3, 8


@pytest.fixture
def params() -> sampling.SamplerParams:
    """A small sampler with a deliberately tight action box."""
    return sampling.SamplerParams(
        N=N,
        num_inputs=NU,
        beta=0.6,
        elite_percentage=0.25,
        temperature=0.25,
        alpha=0.1,
        min_variance=0.02,
        act_low=jnp.array([-1.0, -1.0, -1.0, 0.0, 0.0]),
        act_high=jnp.array([1.0, 1.0, 1.0, 2.0, 6.0]),
    )


@pytest.fixture
def mean() -> jnp.ndarray:
    """Per-mode means sitting inside the action box."""
    return jnp.zeros((K, N, NU)).at[:, :, 3].set(0.43)


@pytest.fixture
def sigma() -> jnp.ndarray:
    """Per-mode standard deviations."""
    return jnp.full((K, N, NU), 0.2)


@pytest.mark.unit
def test_sample_modes_shape_and_determinism(
    params: sampling.SamplerParams, mean: jnp.ndarray, sigma: jnp.ndarray
):
    """Same key must give the same noise; the shape is (K, M, N, nu)."""
    key = jax.random.PRNGKey(0)
    a = sampling.sample_modes(params, key, mean, sigma, K, M)
    b = sampling.sample_modes(params, key, mean, sigma, K, M)
    assert a.shape == (K, M, N, NU)
    assert np.array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.unit
def test_sample_modes_respects_the_action_box(
    params: sampling.SamplerParams, mean: jnp.ndarray, sigma: jnp.ndarray
):
    """Noise is sampled relative to each mode's mean, so mean + noise must stay in bounds.

    Beta smoothing is a convex combination along the horizon, so it cannot push a sample out
    of a box that every unsmoothed draw was already inside.
    """
    noise = sampling.sample_modes(params, jax.random.PRNGKey(1), mean, sigma, K, M)
    candidates = np.asarray(mean[:, None] + noise)
    assert (candidates >= np.asarray(params.act_low) - 1e-5).all()
    assert (candidates <= np.asarray(params.act_high) + 1e-5).all()


@pytest.mark.unit
def test_sample_modes_smoothing_reduces_step_to_step_variation(
    params: sampling.SamplerParams, mean: jnp.ndarray, sigma: jnp.ndarray
):
    """Beta smoothing is the whole reason the sampler is not white noise."""
    key = jax.random.PRNGKey(2)
    smooth = sampling.sample_modes(params, key, mean, sigma, K, M)
    rough = sampling.sample_modes(replace(params, beta=0.0), key, mean, sigma, K, M)

    def jitter(x: jnp.ndarray) -> float:
        """Mean absolute step-to-step change along the horizon."""
        return float(np.abs(np.diff(np.asarray(x), axis=-2)).mean())

    assert jitter(smooth) < jitter(rough)


@pytest.mark.unit
def test_update_modes_moves_mean_toward_the_cheapest_sample(
    params: sampling.SamplerParams, mean: jnp.ndarray, sigma: jnp.ndarray
):
    """The elite/softmax update must shift each mean in the direction of its best sample."""
    noise = jnp.zeros((K, M, N, NU)).at[:, 0].set(0.5)  # sample 0 is offset in every channel
    costs = jnp.ones((K, M)).at[:, 0].set(0.0)  # ...and it is the cheapest
    new_mean, _, best = sampling.update_modes(params, mean, sigma, noise, costs, K, M)
    assert np.all(np.asarray(new_mean) > np.asarray(mean))
    assert 0 <= int(best) < K


@pytest.mark.unit
def test_update_modes_clamps_variance(params: sampling.SamplerParams, mean: jnp.ndarray):
    """Sigma must never collapse below min_variance, or the modes stop exploring."""
    tiny = jnp.full((K, N, NU), 1e-6)
    noise = jnp.zeros((K, M, N, NU))
    costs = jnp.zeros((K, M))
    _, new_sigma, _ = sampling.update_modes(params, mean, tiny, noise, costs, K, M)
    assert np.all(np.asarray(new_sigma) >= np.sqrt(params.min_variance) - 1e-6)


@pytest.mark.unit
def test_update_modes_picks_the_globally_best_mode(
    params: sampling.SamplerParams, mean: jnp.ndarray, sigma: jnp.ndarray
):
    """best_mode_idx is the mode whose elite cost is lowest."""
    noise = jnp.zeros((K, M, N, NU))
    costs = jnp.ones((K, M)).at[2].set(-5.0)
    _, _, best = sampling.update_modes(params, mean, sigma, noise, costs, K, M)
    assert int(best) == 2


@pytest.mark.unit
def test_shift_and_interpolate_shifts_a_ramp():
    """Shifting a linear ramp by dt_ctrl must add exactly dt_ctrl worth of slope."""
    dt_plan, dt_ctrl = 0.02, 0.02
    ramp = jnp.arange(N, dtype=jnp.float32)[:, None] * jnp.ones((1, NU))
    shifted = np.asarray(sampling.shift_and_interpolate(ramp, dt_plan, dt_ctrl))
    # one full plan step of shift: entry i becomes what entry i+1 was, last is held
    assert shifted[0] == pytest.approx(1.0)
    assert shifted[-1] == pytest.approx(float(N - 1))  # zero-order hold at the end


@pytest.mark.unit
def test_truncated_normal_respects_bounds():
    """Samples must land inside [x_min, x_max] regardless of the requested sigma."""
    lo, hi = -0.3, 0.4
    out = np.asarray(
        sampling.truncated_normal(
            jax.random.PRNGKey(3),
            mean=0.0,
            sd=jnp.array(5.0),  # far wider than the bounds
            x_min=jnp.array(lo),
            x_max=jnp.array(hi),
            shape=(2000,),
        )
    )
    assert out.min() >= lo - 1e-5
    assert out.max() <= hi + 1e-5
