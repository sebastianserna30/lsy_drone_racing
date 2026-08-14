"""Noise sampling and the per-mode MPPI distribution update.

Pure functions over explicit arrays. Everything here is called from inside the controller's
jitted core update, so the functions are not jitted individually; the hyperparameters ride
along in a frozen ``SamplerParams`` that the controller holds as a static attribute, which
keeps them compile-time constants exactly as ``self.beta`` and friends were before.

Shapes throughout, for ONE agent:
    mean, sigma   (K, N, nu)     K modes over an N-step horizon of nu-dim controls
    noise         (K, M, N, nu)  M samples per mode
    costs         (K, M)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import random, vmap

if TYPE_CHECKING:
    from jax import Array


@dataclass(frozen=True)
class SamplerParams:
    """MPPI sampling and update hyperparameters, plus the physical action box.

    ``act_low``/``act_high`` are device arrays of shape (nu,); the rest are Python scalars, so
    they fold into the JAX trace as constants.
    """

    N: int  # horizon length
    num_inputs: int  # nu: [roll, pitch, yaw, thrust, v_theta]
    beta: float  # noise smoothing along the horizon, in [0,1]
    elite_percentage: float
    temperature: float
    alpha: float  # weight of the new variance in the sigma update
    min_variance: float
    act_low: Array
    act_high: Array


def truncated_normal(
    key: Array, mean: Array | float, sd: Array, x_min: Array, x_max: Array, shape: tuple = (1,)
) -> Array:
    """Sample from a truncated normal distribution.

    Args:
        key: JAX PRNGKey.
        mean: desired mean (mu) of the underlying gaussian.
        sd: desired standard deviation (sigma) of the underlying gaussian.
        x_min: lower bound.
        x_max: upper bound.
        shape: output shape.

    Returns:
        Samples within [x_min, x_max].
    """
    # 1. Convert physical bounds to standard normal 'Z-scores'
    alpha = (x_min - mean) / sd
    beta = (x_max - mean) / sd

    # 2. Sample from standard truncated normal (mean=0, std=1)
    standard_samples = random.truncated_normal(key, lower=alpha, upper=beta, shape=shape)

    # 3. Scale and shift back to original distribution
    return (standard_samples * sd) + mean


def sample_modes(
    params: SamplerParams, key: Array, mean: Array, sigma: Array, K: int, M: int
) -> Array:
    """Sample + beta-smooth truncated-normal noise for ONE agent's K modes.

    Returns (K, M, N, nu). Bounds are relative to each mode's mean so samples stay in the
    physical action box. K/M are static Python ints.

    The RNG tree is load-bearing: one ``split(key, K)`` then one ``truncated_normal`` call per
    mode. Changing it changes every sampled number and so every trajectory.
    """
    lower_bound = params.act_low - mean  # (K, N, nu)
    upper_bound = params.act_high - mean
    keys = jax.random.split(key, K)

    def sample_per_mean(k_key: Array, k_sigma: Array, k_lb: Array, k_ub: Array) -> Array:
        return truncated_normal(
            k_key,
            mean=0,
            sd=k_sigma,
            x_min=k_lb[None, ...],
            x_max=k_ub[None, ...],
            shape=(M, params.N, params.num_inputs),
        )

    noise = vmap(sample_per_mean)(keys, sigma, lower_bound, upper_bound)  # (K, M, N, nu)

    # beta smoothing along the horizon (flatten K,M for the scan)
    noise_flat = noise.reshape(-1, params.N, params.num_inputs)

    def smooth_scan(carry: Array, x: Array) -> tuple[Array, Array]:
        new_val = params.beta * carry + (1 - params.beta) * x
        return new_val, new_val

    _, noise_flat = jax.lax.scan(
        lambda c, x: vmap(smooth_scan)(c, x),
        jnp.zeros((noise_flat.shape[0], params.num_inputs)),
        noise_flat.transpose(1, 0, 2),
    )
    noise_flat = noise_flat.transpose(1, 0, 2)
    return noise_flat.reshape(K, M, params.N, params.num_inputs)


def update_modes(
    params: SamplerParams, mean: Array, sigma: Array, noise: Array, costs: Array, K: int, M: int
) -> tuple[Array, Array, Array]:
    """Elite/softmax per-mode MPPI update for ONE agent (vmapped over its K modes).

    Each mode keeps its own mean and adaptive sigma: the lowest-cost ``elite_percentage`` of its
    samples are softmax-weighted into a mean shift, and their spread blends into the sigma.

    Args:
        params: sampling hyperparameters.
        mean: (K, N, nu) current per-mode means.
        sigma: (K, N, nu) current per-mode standard deviations.
        noise: (K, M, N, nu) the sampled perturbations that produced `costs`.
        costs: (K, M) total rollout cost per sample.
        K: number of modes (static).
        M: samples per mode (static).

    Returns:
        (new_means (K,N,nu), new_sigmas (K,N,nu), best_mode_idx).
    """

    def update_single_mode(
        k_mean: Array, k_noise: Array, k_costs: Array, k_sigma: Array
    ) -> tuple[Array, Array, Array]:
        sorted_idx = jnp.argsort(k_costs)
        n_elites = max(int(M * params.elite_percentage), 1)
        elite_idx = sorted_idx[:n_elites]
        elite_noise = k_noise[elite_idx]
        elite_costs = k_costs[elite_idx]
        min_cost = jnp.min(elite_costs)
        weights = jnp.exp(-(elite_costs - min_cost) / params.temperature)
        weights = weights / (jnp.sum(weights) + 1e-10)
        delta = jnp.sum(weights[:, None, None] * elite_noise, axis=0)
        diff = elite_noise - delta
        weighted_var = jnp.sum(weights[:, None, None] * (diff**2), axis=0)
        new_sigma_sq = params.alpha * weighted_var + (1.0 - params.alpha) * (k_sigma**2)
        new_sigma_sq = jnp.maximum(new_sigma_sq, params.min_variance)
        return k_mean + delta, jnp.sqrt(new_sigma_sq), min_cost

    updated_means, updated_sigmas, cluster_best_costs = vmap(update_single_mode)(
        mean, noise, costs, sigma
    )
    best_mode_idx = jnp.argmin(cluster_best_costs)
    return updated_means, updated_sigmas, best_mode_idx


@jax.jit
def shift_and_interpolate(controls: Array, dt_plan: float, dt_ctrl: float) -> Array:
    """Shift a control trajectory forward by dt_ctrl using linear interpolation.

    The receding-horizon step: what was planned for t + dt_ctrl becomes the plan for t.

    Args:
        controls: (Horizon, Control_Dim).
        dt_plan: duration of one step in the planning horizon.
        dt_ctrl: duration of one control cycle (latency/execution time).

    Returns:
        Shifted controls of shape (Horizon, Control_Dim).
    """
    horizon_len = controls.shape[0]

    # old_times: [0, dt_plan, 2*dt_plan, ...]; target_times the same grid shifted by dt_ctrl
    old_times = jnp.arange(horizon_len) * dt_plan
    target_times = old_times + dt_ctrl

    def interp_fn(y: Array) -> Array:
        # right=y[-1] implements Zero-Order Hold for the end of the horizon
        return jnp.interp(target_times, old_times, y, left=y[0], right=y[-1])

    # Vectorize across Control_Dim (axis 1), keeping time (rows) intact per op
    return jax.vmap(interp_fn, in_axes=1, out_axes=1)(controls)
