"""Tests for the arc-length reference LUTs and theta anchoring.

The anchoring guards exist because of a specific failure: on a dip gate the reference doubles
back on itself, so a naive nearest-point search snaps to the wrong leg, the tangent flips, and
the opponent gets predicted standing still. These tests build exactly that geometry.
"""

from __future__ import annotations

import numpy as np
import pytest

from lsy_drone_racing.control.mppi import reference


class FakeSpline:
    """Minimal stand-in for SplinePlanner exposing what build_paths touches."""

    def __init__(self, points: np.ndarray, t_total: float = 1.0):
        """Store the polyline the fake spline interpolates."""
        self.t_total = t_total
        self._points = points

    def _pos_spline(self, t: np.ndarray) -> np.ndarray:
        """Linearly interpolate the polyline at normalized times `t`."""
        u = np.clip(t / self.t_total, 0.0, 1.0) * (len(self._points) - 1)
        lo = np.floor(u).astype(int)
        hi = np.minimum(lo + 1, len(self._points) - 1)
        frac = (u - lo)[:, None]
        return self._points[lo] * (1 - frac) + self._points[hi] * frac

    def _vel_spline(self, t: np.ndarray) -> np.ndarray:
        """Finite-difference velocity of the polyline."""
        eps = self.t_total * 1e-4
        return (self._pos_spline(t + eps) - self._pos_spline(t - eps)) / (2 * eps)


def straight_line(n: int = 40) -> FakeSpline:
    """A path running along +x from the origin to (4, 0, 1)."""
    pts = np.stack([np.linspace(0, 4, n), np.zeros(n), np.ones(n)], axis=-1)
    return FakeSpline(pts)


def hairpin() -> FakeSpline:
    """Out along +x at y=0, then back along -x at y=0.4 — the dip-gate geometry.

    The two legs sit 0.4 m apart in space but ~4 m apart in arc length, so a global
    nearest-point search is ambiguous exactly where it matters.
    """
    n = 60
    out = np.stack([np.linspace(0, 4, n), np.zeros(n), np.ones(n)], axis=-1)
    back = np.stack([np.linspace(4, 0, n), np.full(n, 0.4), np.ones(n)], axis=-1)
    return FakeSpline(np.concatenate([out, back]))


@pytest.fixture
def params() -> reference.AnchorParams:
    """Anchor window matching the shipped defaults."""
    return reference.AnchorParams(v_theta_max=6.0, ctrl_dt=0.02, fwd=0.10, reacquire_dist=0.75)


@pytest.mark.unit
def test_arc_length_grid_is_monotone_and_starts_at_zero():
    """Progress theta is cumulative arc length: non-decreasing, starting at 0."""
    paths = reference.build_paths([straight_line()], n_lut=200, device=None)
    grid = paths.theta_grid_np[0]
    assert grid[0] == 0.0
    assert np.all(np.diff(grid) >= 0.0)
    assert paths.s_total_np[0] == pytest.approx(4.0, rel=1e-3)


@pytest.mark.unit
def test_tangents_are_unit_length():
    """The contour/lag split projects onto the tangent, so it must be normalized."""
    paths = reference.build_paths([straight_line()], n_lut=100, device=None)
    norms = np.linalg.norm(paths.tan_np[0], axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-5)


@pytest.mark.unit
def test_ref_at_theta_interpolates_and_clips():
    """Lookup must land on the path inside the range and clamp outside it."""
    paths = reference.build_paths([straight_line()], n_lut=200, device=None)
    tg, pos_lut, tan_lut = paths.theta_grid[0], paths.pos_lut[0], paths.tan_lut[0]

    mid, tan = reference.ref_at_theta(2.0, tg, pos_lut, tan_lut)
    assert np.asarray(mid)[0] == pytest.approx(2.0, abs=1e-2)
    assert np.asarray(tan)[0] == pytest.approx(1.0, abs=1e-3)  # pointing along +x

    past_end, _ = reference.ref_at_theta(99.0, tg, pos_lut, tan_lut)
    assert np.asarray(past_end)[0] == pytest.approx(4.0, abs=1e-2)  # clipped, not extrapolated


@pytest.mark.unit
def test_anchor_bootstraps_without_a_previous_anchor(params: reference.AnchorParams):
    """With anchored=False the search is global, and must find the nearest point."""
    paths = reference.build_paths([straight_line()], n_lut=200, device=None)
    theta, anchored = reference.anchor_theta(
        paths, 0, np.array([2.0, 0.05, 1.0]), np.array([1.0, 0.0, 0.0]), 0.0, False, params
    )
    assert anchored
    assert theta == pytest.approx(2.0, abs=0.05)


@pytest.mark.unit
def test_tracked_anchor_does_not_slide_backwards(params: reference.AnchorParams):
    """A tracked anchor may lag, but must not rewind on a small backward perturbation.

    Otherwise it walks back down the outbound leg of a hairpin and never recovers. 0.3 m is
    inside reacquire_dist, so the windowed branch handles it and the anchor stays put.
    """
    paths = reference.build_paths([straight_line()], n_lut=200, device=None)
    theta, _ = reference.anchor_theta(
        paths, 0, np.array([1.7, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]), 2.0, True, params
    )
    assert theta >= 2.0


@pytest.mark.unit
def test_large_backward_jump_reacquires_and_may_rewind(params: reference.AnchorParams):
    """The monotone window is not an absolute ratchet, by design.

    A match worse than reacquire_dist means the anchor is lost (long dropout, or a line far
    from our guessed spline). Recovery is worth more than monotonicity, so the global search
    takes over and theta may move backwards.
    """
    paths = reference.build_paths([straight_line()], n_lut=200, device=None)
    theta, anchored = reference.anchor_theta(
        paths, 0, np.array([1.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]), 2.0, True, params
    )
    assert anchored
    assert theta == pytest.approx(1.0, abs=0.05)  # rewound to where the drone actually is


@pytest.mark.unit
def test_anchor_advances_within_the_reachable_window(params: reference.AnchorParams):
    """A small forward step must be tracked, and capped by what one control step can cover."""
    paths = reference.build_paths([straight_line()], n_lut=400, device=None)
    reachable = params.v_theta_max * params.ctrl_dt + params.fwd  # 0.22 m
    theta, _ = reference.anchor_theta(
        paths, 0, np.array([2.1, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]), 2.0, True, params
    )
    assert 2.0 <= theta <= 2.0 + reachable + 1e-6


@pytest.mark.unit
def test_heading_gate_separates_the_two_legs_of_a_hairpin(params: reference.AnchorParams):
    """The dip-gate bug: without the heading gate this snaps to the wrong leg.

    A drone on the RETURN leg (travelling -x) sits 0.4 m from the outbound leg, which a global
    argmin may well prefer. The tangent there points +x, opposite to travel, so the gate
    excludes it and the anchor lands on the return leg — past the halfway arc length.
    """
    paths = reference.build_paths([hairpin()], n_lut=800, device=None)
    half = paths.s_total_np[0] / 2.0

    theta, _ = reference.anchor_theta(
        paths,
        0,
        np.array([2.0, 0.4, 1.0]),  # on the return leg
        np.array([-1.0, 0.0, 0.0]),  # travelling -x
        0.0,
        False,
        params,
    )
    assert theta > half, "anchored onto the outbound leg despite travelling the other way"


@pytest.mark.unit
def test_reacquire_when_the_windowed_match_is_poor(params: reference.AnchorParams):
    """A drone that jumps far from its anchor (mocap dropout) must re-acquire globally."""
    paths = reference.build_paths([straight_line()], n_lut=400, device=None)
    # previous anchor near the start, but the drone is now 3 m down the path
    theta, anchored = reference.anchor_theta(
        paths, 0, np.array([3.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]), 0.2, True, params
    )
    assert anchored
    assert theta == pytest.approx(3.0, abs=0.05)


@pytest.mark.unit
def test_low_speed_drops_the_heading_gate(params: reference.AnchorParams):
    """Below 0.1 m/s the heading is estimation noise, so every point stays a candidate."""
    paths = reference.build_paths([straight_line()], n_lut=200, device=None)
    theta, _ = reference.anchor_theta(
        paths, 0, np.array([1.5, 0.0, 1.0]), np.zeros(3), 0.0, False, params
    )
    assert theta == pytest.approx(1.5, abs=0.05)


@pytest.mark.unit
def test_paths_stack_multiple_agents():
    """Device LUTs carry a leading agent axis so the cost can index one agent at trace time."""
    paths = reference.build_paths([straight_line(), hairpin()], n_lut=128, device=None)
    assert paths.theta_grid.shape == (2, 128)
    assert paths.pos_lut.shape == (2, 128, 3)
    assert paths.n_agents == 2
    assert paths.s_total_np[1] > paths.s_total_np[0]  # the hairpin is the longer path
