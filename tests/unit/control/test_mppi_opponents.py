"""Tests for opponent tracking and kinematic prediction.

Mocap dropout handling is safety-critical and effectively impossible to exercise in sim, where
the observation is always finite. These tests feed the tracker the NaNs that real flight
produces.
"""

from __future__ import annotations

import numpy as np
import pytest

from lsy_drone_racing.control.mppi import opponents, reference
from tests.unit.control.test_mppi_reference import straight_line

CTRL_DT = 0.02


def obs(pos: np.ndarray, vel: np.ndarray) -> dict[str, np.ndarray]:
    """Two-agent observation with the given positions and velocities."""
    return {
        "pos": pos,
        "quat": np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (len(pos), 1)),
        "vel": vel,
        "ang_vel": np.zeros_like(pos),
    }


@pytest.fixture
def tracker_params() -> opponents.TrackerParams:
    """Tracker parameters matching the shipped defaults."""
    return opponents.TrackerParams(
        ctrl_dt=CTRL_DT, vel_ema=0.3, stale_inflate_rate=1.0, stale_max=0.5
    )


@pytest.fixture
def tracker(tracker_params: opponents.TrackerParams) -> opponents.OpponentTracker:
    """A 2-agent tracker seeded at the origin and (1, 0, 1)."""
    start = obs(
        np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]),
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    return opponents.OpponentTracker(2, start, tracker_params)


@pytest.mark.unit
def test_valid_sample_passes_through_and_clears_staleness(tracker: opponents.OpponentTracker):
    """A finite measurement must be used as-is and reset the staleness clock."""
    pos = np.array([[0.0, 0.0, 1.0], [1.5, 0.0, 1.0]])
    vel = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    out_pos, _, _, _, inflate = tracker.update(pos, np.zeros((2, 4)), vel, np.zeros((2, 3)))
    assert np.allclose(out_pos[1], [1.5, 0.0, 1.0])
    assert tracker.stale_t[1] == 0.0
    assert inflate[0] == pytest.approx(1.0)


@pytest.mark.unit
def test_nan_dropout_holds_the_last_valid_state(tracker: opponents.OpponentTracker):
    """A NaN must never reach the rollout — it would poison every sample."""
    good = np.array([[0.0, 0.0, 1.0], [1.5, 0.0, 1.0]])
    tracker.update(good, np.zeros((2, 4)), np.zeros((2, 3)), np.zeros((2, 3)))

    dropped = np.array([[0.0, 0.0, 1.0], [np.nan, np.nan, np.nan]])
    out_pos, _, _, _, _ = tracker.update(
        dropped, np.zeros((2, 4)), np.zeros((2, 3)), np.zeros((2, 3))
    )
    assert np.all(np.isfinite(out_pos))
    assert np.allclose(out_pos[1], [1.5, 0.0, 1.0])  # held, not propagated


@pytest.mark.unit
def test_staleness_inflates_the_keepout_and_saturates(tracker: opponents.OpponentTracker):
    """Inflation grows while the measurement is missing, then tops out at 1 + rate * max."""
    dropped = np.array([[0.0, 0.0, 1.0], [np.nan, 0.0, 1.0]])
    inflations = [
        tracker.update(dropped, np.zeros((2, 4)), np.zeros((2, 3)), np.zeros((2, 3)))[4][0]
        for _ in range(60)  # 1.2 s, well past stale_max = 0.5
    ]
    assert inflations[0] < inflations[5] < inflations[-1]
    assert inflations[-1] == pytest.approx(1.5)  # 1 + 1.0 * 0.5


@pytest.mark.unit
def test_recovery_after_dropout_resets_inflation(tracker: opponents.OpponentTracker):
    """Once the measurement returns, the ego must stop giving away extra room."""
    dropped = np.array([[0.0, 0.0, 1.0], [np.nan, 0.0, 1.0]])
    for _ in range(10):
        tracker.update(dropped, np.zeros((2, 4)), np.zeros((2, 3)), np.zeros((2, 3)))
    good = np.array([[0.0, 0.0, 1.0], [1.5, 0.0, 1.0]])
    _, _, _, _, inflate = tracker.update(
        good, np.zeros((2, 4)), np.zeros((2, 3)), np.zeros((2, 3))
    )
    assert inflate[0] == pytest.approx(1.0)


@pytest.mark.unit
def test_velocity_ema_converges_without_overshooting(tracker: opponents.OpponentTracker):
    """The low-pass must approach the true velocity monotonically, never exceed it."""
    pos = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    vel = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    seen = []
    for _ in range(40):
        tracker.update(pos, np.zeros((2, 4)), vel, np.zeros((2, 3)))
        seen.append(tracker.vel_filt[1, 0])
    assert all(a <= b + 1e-12 for a, b in zip(seen, seen[1:]))  # monotone
    assert max(seen) <= 3.0 + 1e-9
    assert seen[-1] == pytest.approx(3.0, abs=1e-3)  # converged


@pytest.mark.unit
def test_ego_is_never_filtered(tracker: opponents.OpponentTracker):
    """Agent 0 is our own drone; its state comes from the flight controller, not tracking."""
    pos = np.array([[np.nan, np.nan, np.nan], [1.5, 0.0, 1.0]])
    out_pos, _, _, _, _ = tracker.update(pos, np.zeros((2, 4)), np.zeros((2, 3)), np.zeros((2, 3)))
    assert np.all(np.isnan(out_pos[0])), "the ego slot must be passed through untouched"


@pytest.fixture
def paths() -> reference.ReferencePaths:
    """Two identical straight-line references (ego + one opponent)."""
    return reference.build_paths([straight_line(), straight_line()], n_lut=400, device=None)


@pytest.mark.unit
def test_const_vel_extrapolates_a_straight_line(paths: reference.ReferencePaths):
    """const_vel is pure extrapolation: position must advance by v * t each step."""
    params = opponents.PredictorParams(
        model="const_vel", horizon=5, dt=0.1, v_theta_max=6.0, offset_tau=0.5
    )
    held = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    vel = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    pos, out_vel = opponents.predict(params, paths, held, vel, [0.0, 0.0], 2)

    assert pos.shape == (5, 1, 3)
    assert pos[0, 0, 0] == pytest.approx(1.0 + 2.0 * 0.1)
    assert pos[4, 0, 0] == pytest.approx(1.0 + 2.0 * 0.5)
    assert np.allclose(out_vel[:, 0], [2.0, 0.0, 0.0])


@pytest.mark.unit
def test_spline_progress_follows_the_path(paths: reference.ReferencePaths):
    """The opponent is predicted along its own spline at its observed forward speed."""
    params = opponents.PredictorParams(
        model="spline_progress", horizon=5, dt=0.1, v_theta_max=6.0, offset_tau=0.5
    )
    held = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    vel = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    pos, out_vel = opponents.predict(params, paths, held, vel, [0.0, 1.0], 2)

    assert np.all(np.diff(pos[:, 0, 0]) > 0)  # advancing along +x
    assert pos[4, 0, 0] == pytest.approx(1.0 + 2.0 * 0.5, abs=0.05)
    assert out_vel[0, 0, 0] == pytest.approx(2.0, abs=0.05)


@pytest.mark.unit
def test_spline_progress_starts_at_the_real_opponent(paths: reference.ReferencePaths):
    """An opponent flying off our guessed line must still be predicted where it actually is.

    The off-spline offset decays with offset_tau rather than being ignored, so the first
    prediction sits near the measurement and only later converges onto the reference.
    """
    params = opponents.PredictorParams(
        model="spline_progress", horizon=20, dt=0.1, v_theta_max=6.0, offset_tau=0.5
    )
    held = np.array([[0.0, 0.0, 1.0], [1.0, 0.6, 1.0]])  # 0.6 m off the line in +y
    vel = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    pos, _ = opponents.predict(params, paths, held, vel, [0.0, 1.0], 2)

    assert pos[0, 0, 1] > 0.4, "prediction jumped onto the reference, ignoring the real offset"
    assert abs(pos[-1, 0, 1]) < 0.05, "offset should have decayed away by the end of the horizon"


@pytest.mark.unit
def test_spline_progress_never_predicts_backwards(paths: reference.ReferencePaths):
    """Progress speed is clipped at 0: an opponent is never predicted running the path backwards."""
    params = opponents.PredictorParams(
        model="spline_progress", horizon=5, dt=0.1, v_theta_max=6.0, offset_tau=0.5
    )
    held = np.array([[0.0, 0.0, 1.0], [2.0, 0.0, 1.0]])
    vel = np.array([[0.0, 0.0, 0.0], [-3.0, 0.0, 0.0]])  # travelling against the path
    pos, out_vel = opponents.predict(params, paths, held, vel, [0.0, 2.0], 2)
    assert np.all(np.diff(pos[:, 0, 0]) >= -1e-6)
    assert np.allclose(out_vel[:, 0], 0.0)


@pytest.mark.unit
def test_spline_progress_clamps_at_the_end_of_the_path(paths: reference.ReferencePaths):
    """Predictions must not run off the end of the spline."""
    params = opponents.PredictorParams(
        model="spline_progress", horizon=30, dt=0.2, v_theta_max=6.0, offset_tau=0.5
    )
    held = np.array([[0.0, 0.0, 1.0], [3.8, 0.0, 1.0]])
    vel = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    pos, _ = opponents.predict(params, paths, held, vel, [0.0, 3.8], 2)
    assert pos[:, 0, 0].max() <= paths.s_total_np[1] + 0.05
