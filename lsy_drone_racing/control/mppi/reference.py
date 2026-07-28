"""Arc-length reparameterization of the reference splines, and progress (theta) anchoring.

MPCC needs the reference keyed on arc length rather than time, so that the progress speed
v_theta has units of m/s and the horizon is not capped by the spline's ``t_total``. Each agent's
time-based ``SplinePlanner`` is densely sampled once at reset and cached as a lookup table.

Two consumers with different needs:

* the jitted rollout evaluates ``ref_at_theta`` on device, and wants the LUTs stacked on a
  leading agent axis so it can index one agent's table at trace time;
* the host-side opponent predictor and renderer want per-agent NumPy copies.

``ReferencePaths`` holds both views of the same numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from jax import Array
    from numpy.typing import NDArray

    from lsy_drone_racing.control.spline_planner import SplinePlanner


@dataclass(frozen=True)
class ReferencePaths:
    """Every agent's spline as an arc-length LUT, in both device and host form.

    Device arrays carry a leading agent axis (A, n_lut, ...) and are baked into the JIT'd cost
    as constants. Host lists are per agent.
    """

    theta_grid: Array  # (A, n_lut) arc length
    pos_lut: Array  # (A, n_lut, 3)
    tan_lut: Array  # (A, n_lut, 3)
    s_total: Array  # (A,)

    theta_grid_np: list[NDArray[np.floating]]
    pos_np: list[NDArray[np.floating]]
    tan_np: list[NDArray[np.floating]]
    s_total_np: NDArray[np.floating]

    @property
    def n_agents(self) -> int:
        """Number of agents with a reference path."""
        return len(self.theta_grid_np)


def build_paths(planners: list[SplinePlanner], n_lut: int, device: object) -> ReferencePaths:
    """Reparameterize each planner's time-based spline to arc-length progress theta.

    For every agent, densely sample the position spline in time, accumulate segment lengths to
    get arc length (= theta), and store position + unit-tangent tables keyed on theta. Inside
    the rollout, ref(theta) is recovered with ``jnp.interp`` (see :func:`ref_at_theta`).

    Args:
        planners: one ``SplinePlanner`` per agent.
        n_lut: number of samples in the table. Higher is smoother but costs trace memory.
        device: JAX device for the stacked copies.

    Returns:
        The LUTs in both device and host form.
    """
    grids, pos_luts, tan_luts, s_totals = [], [], [], []
    for planner in planners:
        t_dense = np.linspace(0.0, planner.t_total, n_lut)
        pos = np.asarray(planner._pos_spline(t_dense))  # (n_lut, 3)
        seg = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])  # arc-length grid = theta, (n_lut,)
        vel = np.asarray(planner._vel_spline(t_dense))
        tangent = vel / np.clip(np.linalg.norm(vel, axis=1, keepdims=True), 1e-9, None)
        grids.append(s)
        pos_luts.append(pos)
        tan_luts.append(tangent)
        s_totals.append(float(s[-1]))
    return ReferencePaths(
        # stacked device copies (A, n_lut, ...) baked into the JIT'd cost
        theta_grid=jnp.asarray(np.stack(grids), device=device),
        pos_lut=jnp.asarray(np.stack(pos_luts), device=device),
        tan_lut=jnp.asarray(np.stack(tan_luts), device=device),
        s_total=jnp.asarray(np.array(s_totals), device=device),
        # host copies (per agent) for logging / rendering / theta re-anchoring
        theta_grid_np=[np.asarray(g) for g in grids],
        pos_np=[np.asarray(p) for p in pos_luts],
        # host tangent LUTs for the kinematic opponent predictor
        tan_np=[np.asarray(t) for t in tan_luts],
        s_total_np=np.array(s_totals),
    )


def ref_at_theta(theta: Array, tg: Array, pos_lut: Array, tan_lut: Array) -> tuple[Array, Array]:
    """LUT lookup: reference position and unit path tangent at progress theta (arc length).

    theta is a scalar; vmap over rollouts. LUT arrays are passed explicitly so the same helper
    serves any agent's spline. Clipped to [0, s_total] so the rollout never extrapolates past
    the path ends.
    """
    theta = jnp.clip(theta, tg[0], tg[-1])
    pos = jnp.stack(
        [
            jnp.interp(theta, tg, pos_lut[:, 0]),
            jnp.interp(theta, tg, pos_lut[:, 1]),
            jnp.interp(theta, tg, pos_lut[:, 2]),
        ]
    )
    tan = jnp.stack(
        [
            jnp.interp(theta, tg, tan_lut[:, 0]),
            jnp.interp(theta, tg, tan_lut[:, 1]),
            jnp.interp(theta, tg, tan_lut[:, 2]),
        ]
    )
    tan = tan / jnp.clip(jnp.linalg.norm(tan), 1e-9, None)
    return pos, tan


@dataclass(frozen=True)
class AnchorParams:
    """Search-window parameters for :func:`anchor_theta`."""

    v_theta_max: float
    ctrl_dt: float
    fwd: float  # forward slack (m of arc length) on top of what one control step can cover
    reacquire_dist: float  # residual above which the anchor is considered lost


def anchor_theta(
    paths: ReferencePaths,
    agent: int,
    pos: NDArray[np.floating],
    vel: NDArray[np.floating],
    theta_prev: float,
    anchored: bool,
    params: AnchorParams,
) -> tuple[float, bool]:
    """Anchor an observed position onto an agent's spline as progress theta (arc length).

    A plain global argmin over the LUT is ambiguous wherever the path passes near itself. The
    worst case is a dip gate (see SplinePlanner._create_waypoints): the reference reverses back
    through the gate, so the outbound and return legs sit ~0.4 m apart in space but ~1.5 m apart
    in arc length. A global
    argmin then snaps to the wrong leg, the path tangent flips, and the
    clip(dot(v, tan), 0, ...) in the opponent predictor collapses the predicted progress speed
    to zero — the opponent gets predicted standing still exactly where we most need it. On the
    nominal track, an opponent flying 0.15 m off the reference mis-anchors over ~9% of the lap.

    Three guards remove the ambiguity:
      * Heading gate — only LUT points whose tangent agrees with the observed direction of
        travel are candidates. This is what separates the two legs of the U-turn, which are
        near-antiparallel (measured min dot = -0.995 on the nominal track).
      * Monotone window — a tracked anchor searches only [theta_prev, theta_prev + reachable],
        reachable = v_theta_max * ctrl_dt + fwd. Progress along a racing line is non-decreasing,
        so the anchor may lag (and catch up next step) but does not slide backwards; without
        that clamp it walks back down the outbound leg while the opponent flies the return leg,
        and never recovers. Note the window is not an absolute ratchet: if the best match inside
        it is still worse than reacquire_dist, the re-acquire branch below takes over and *may*
        move theta backwards. That is deliberate — a lost anchor must be able to recover.
      * Re-acquire — with no previous anchor, or when the windowed match is worse than
        reacquire_dist (opponent lost, long mocap dropout, or a line far from our guessed
        spline), fall back to a heading-gated GLOBAL search.

    Args:
        paths: the reference LUTs.
        agent: which agent's spline to anchor onto.
        pos: (3,) observed position.
        vel: (3,) filtered observed velocity, used for the heading gate.
        theta_prev: the previous anchor, used as the window start.
        anchored: whether `theta_prev` is a valid previous anchor.
        params: search-window parameters.

    Returns:
        (theta, anchored) — the new anchor and the updated tracking flag.
    """
    grid = paths.theta_grid_np[agent]  # (n_lut,) arc length
    d = np.linalg.norm(paths.pos_np[agent] - pos[None, :], axis=1)

    speed = float(np.linalg.norm(vel))
    if speed > 0.1:  # below that the heading is estimation noise; accept every point
        gate = paths.tan_np[agent] @ (vel / speed) > 0.0
    else:
        gate = np.ones(len(grid), dtype=bool)

    if anchored:
        hi = theta_prev + params.v_theta_max * params.ctrl_dt + params.fwd
        i0 = int(min(np.searchsorted(grid, theta_prev), len(grid) - 1))
        i1 = int(max(np.searchsorted(grid, hi), i0 + 1))
        idx = i0 + np.flatnonzero(gate[i0:i1])
        if idx.size:
            j = int(idx[int(d[idx].argmin())])
            if d[j] <= params.reacquire_dist:
                return float(grid[j]), True

    idx = np.flatnonzero(gate)
    if idx.size == 0:  # every tangent disagrees (flying backwards): drop the gate
        idx = np.arange(len(grid))
    j = int(idx[int(d[idx].argmin())])
    return float(grid[j]), True
