"""Standalone downwash avoidance scenario — no gates, no env.

A single drone starts already flying and must cross the arena while
avoiding the downwash zones of 2 obstacle drones (one static, one
moving) rendered as full drone models. A translucent cone is drawn
below each obstacle drone to visualise its downwash region.

Run as:
    $ python scripts/downwash_sim.py              # headless, prints summary
    $ python scripts/downwash_sim.py --render true # with GUI
"""

from __future__ import annotations

import logging
from pathlib import Path

import fire
import jax
import jax.numpy as jnp
import numpy as np

# lsy_drone_racing must come before scipy so crazyflow sets SCIPY_ARRAY_API first
from lsy_drone_racing.control.trajectory_mppi import AttitudeMPPIController
from lsy_drone_racing.utils import load_config

from crazyflow.control import Control
from crazyflow.sim import Physics, Sim
from crazyflow.sim.visualize import draw_capsule, draw_line, draw_points

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CTRL_FREQ = 50  # Hz — outer control loop
SIM_FREQ = 500  # Hz — physics
N_PHYS_STEPS = SIM_FREQ // CTRL_FREQ  # physics steps per control cycle = 10

DRONE_START = np.array([-2.0, 0.0, 1.0], dtype=np.float32)
DRONE_GOAL = np.array([2.0, 0.0, 1.0], dtype=np.float32)
GOAL_RADIUS = 0.25
MAX_STEPS = 900  # 18 s at 50 Hz

CONE_HEIGHT = 1.2  # downwash cone depth (m)
CONE_RADIUS = 0.5  # base radius of downwash cone (m)
CONE_RGBA = (0.2, 0.6, 1.0, 0.35)  # translucent blue

DOWNWASH_DETECT_R = 0.35  # horizontal radius for summary logging

# Obstacle drone scenario — drones hover at z=1.5 (above our z=1.0 reference).
# Their downwash cones extend down to z≈0.3, so our drone is well inside each
# cone and must route around them horizontally.
# One static drone (drone 1) and one moving drone (drone 2).
# The static drone is offset off the y=0 centerline: a dead-center head-on
# obstacle makes MPPI's left/right dodge costs symmetric, so it cancels to zero
# lateral motion and the drone stalls. Offsetting it gives a clear escape side.
_STATIC_OPP = np.array([[-0.5, 0.25, 1.5]], dtype=np.float32)

_OPP_AMP = 0.4  # sinusoidal amplitude (m)
_OPP_FREQ = 0.3  # sinusoidal frequency (Hz)


# ---------------------------------------------------------------------------
# Obstacle drone positions
# ---------------------------------------------------------------------------


def obstacle_positions(t: float) -> np.ndarray:
    """Return (2, 3) float32 array of obstacle drone positions at time t.

    Row 0 is the static drone; row 1 oscillates sinusoidally along y.
    """
    y = _OPP_AMP * np.sin(2 * np.pi * _OPP_FREQ * t)
    dynamic = np.array([[0.5, y, 1.5]], dtype=np.float32)
    return np.vstack([_STATIC_OPP, dynamic])


# ---------------------------------------------------------------------------
# Downwash cone visualisation
# ---------------------------------------------------------------------------


def draw_downwash_cone(sim: Sim, apex: np.ndarray) -> None:
    """Draw a downwash cone below apex.

    Geom budget: 2 rings × 12 segs + 6 capsule spokes = 30 per cone × 2 = 60 total.
    Spokes use draw_capsule (1 geom each, visually thick) for better visibility.
    """
    n_rings = 2
    n_ring_pts = 12
    n_spokes = 6

    for ring_i in range(1, n_rings + 1):
        frac = ring_i / n_rings
        z = apex[2] - frac * CONE_HEIGHT
        r = CONE_RADIUS * frac
        angles = np.linspace(0, 2 * np.pi, n_ring_pts + 1)
        ring = np.column_stack(
            [apex[0] + r * np.cos(angles), apex[1] + r * np.sin(angles), np.full(n_ring_pts + 1, z)]
        ).astype(np.float32)
        draw_line(sim, ring, rgba=CONE_RGBA)

    for angle in np.linspace(0, 2 * np.pi, n_spokes, endpoint=False):
        p2 = np.array(
            [
                apex[0] + CONE_RADIUS * np.cos(angle),
                apex[1] + CONE_RADIUS * np.sin(angle),
                apex[2] - CONE_HEIGHT,
            ],
            dtype=np.float32,
        )
        draw_capsule(sim, apex, p2, radius=0.012, rgba=CONE_RGBA)


# ---------------------------------------------------------------------------
# Subclass: straight-line reference, 5 opponents
# ---------------------------------------------------------------------------


class DownwashTestController(AttitudeMPPIController):
    """MPPI with straight-line reference and 2-slot opponent_pos.

    The base controller builds its reference spline from ``initial_obs["gates_pos"]``.
    This scenario has no gates, so we seed a single dummy gate at the goal: the
    SplinePlanner then fits a start->goal trajectory, i.e. the straight crossing we
    want. No separate planner class needed for this PoC.
    """

    N_OPPONENTS = 2

    def __init__(self, initial_obs: dict, info: dict, initial_info):
        self.n_opponents = self.N_OPPONENTS  # must be set BEFORE super().__init__
        # Seed one dummy gate at the goal so the base SplinePlanner has >=2
        # waypoints (start + goal) and produces a straight start->goal reference.
        initial_obs = dict(initial_obs)
        initial_obs["gates_pos"] = DRONE_GOAL[None].astype(np.float32).copy()
        initial_obs["gates_quat"] = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        # Widen the MPPI keep-out so the drone routes clear of the downwash, not
        # just clear of bare contact. The cost threshold is obstacle_radius +
        # drone_radius (0.12). A full CONE_RADIUS (0.5 m) keep-out walls off the
        # path and stalls the drone; ~0.4 m bends it around while staying mostly
        # out of the downwash cone.
        initial_info["controller"]["mppi"]["geometry"]["obstacle_radius"] = 0.40 - 0.12
        super().__init__(initial_obs, info, initial_info)

    def render_callback(self, sim: Sim) -> None:
        """Draw reference spline + only the single best MPPI rollout to save geoms.

        Geom budget breakdown (maxgeom=1000):
          reference spline  99 segs
          waypoints          3 pts
          setpoint           1 pt
          best rollout      24 segs
          2 × cones          60 geoms (drawn in simulate())
          Total             ~187
        """
        self._draw_reference(sim)
        if not hasattr(self, "positions") or self.positions is None:
            return
        best_k = int(self.best_mode_idx)
        best_m = int(np.argmin(np.asarray(self.costs[best_k])))
        traj = np.asarray(self.positions[best_k, best_m])  # (N, 3)
        draw_line(sim, traj, rgba=(1.0, 1.0, 1.0, 1.0))


# ---------------------------------------------------------------------------
# Helpers: read / write sim state
# ---------------------------------------------------------------------------

HOVER_QUAT = jnp.array([0.0, 0.0, 0.0, 1.0])
HOVER_CMD = jnp.array([0.0, 0.0, 0.0, 0.43])  # roll/pitch/yaw=0, hover thrust


def set_drone_state(
    sim: Sim,
    pos: np.ndarray,  # (n_drones, 3)
    quat: np.ndarray,  # (n_drones, 4)
    vel: np.ndarray,  # (n_drones, 3)
    ang_vel: np.ndarray,  # (n_drones, 3)
) -> None:
    """Overwrite all drone states in world 0 of sim (in-place on sim.data)."""
    p = sim.data.states.pos.at[0].set(jnp.array(pos))
    q = sim.data.states.quat.at[0].set(jnp.array(quat))
    v = sim.data.states.vel.at[0].set(jnp.array(vel))
    av = sim.data.states.ang_vel.at[0].set(jnp.array(ang_vel))
    sim.data = sim.data.replace(states=sim.data.states.replace(pos=p, quat=q, vel=v, ang_vel=av))


def get_our_obs(sim: Sim, rotor_vel: np.ndarray) -> dict:
    """Build obs dict for drone 0 from sim state (single-agent shape)."""
    s = sim.data.states
    return {
        "pos": np.array(s.pos[0, 0]),
        "quat": np.array(s.quat[0, 0]),
        "vel": np.array(s.vel[0, 0]),
        "ang_vel": np.array(s.ang_vel[0, 0]),
        "rotor_vel": rotor_vel,
        "obstacles_pos": np.zeros((0, 3), dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------


def simulate(render: bool = False) -> None:
    """Run the downwash avoidance scenario.

    Args:
        render: Open the Crazyflow GUI.
    """
    config = load_config(Path(__file__).parents[1] / "config" / "level0_MPPI.toml")

    n_drones_total = 1 + DownwashTestController.N_OPPONENTS  # 3
    sim = Sim(
        n_worlds=1,
        n_drones=n_drones_total,
        attitude_freq=SIM_FREQ,
        freq=SIM_FREQ,
        physics=Physics.so_rpy_rotor_drag,
        control=Control.attitude,
        drone_model="cf21B_500",
        device="cuda",
    )
    sim.reset()
    step_fn = sim.build_step_fn()

    # Initial state: our drone at start, obstacle drones at t=0 positions
    opp0 = obstacle_positions(0.0)
    init_pos = np.vstack([DRONE_START[None], opp0])  # (3, 3)
    init_quat = np.tile([0.0, 0.0, 0.0, 1.0], (n_drones_total, 1))  # (3, 4)
    init_vel = np.zeros((n_drones_total, 3), dtype=np.float32)
    init_ang = np.zeros((n_drones_total, 3), dtype=np.float32)
    set_drone_state(sim, init_pos, init_quat, init_vel, init_ang)

    # Obstacle drone hover commands (drones 1-2 just hover)
    hover_cmds = jnp.tile(HOVER_CMD, (1, n_drones_total, 1))  # (1, 3, 4)
    sim.data = sim.data.replace(
        controls=sim.data.controls.replace(
            attitude=sim.data.controls.attitude.replace(staged_cmd=hover_cmds)
        )
    )

    # Build fake initial_obs for the controller (no gates in this scenario)
    initial_obs = {
        "pos": DRONE_START.copy(),
        "quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "vel": np.zeros(3, dtype=np.float32),
        "ang_vel": np.zeros(3, dtype=np.float32),
        "obstacles_pos": np.zeros((0, 3), dtype=np.float32),
        "gates_pos": np.zeros((0, 3), dtype=np.float32),
        "gates_quat": np.zeros((0, 4), dtype=np.float32),
    }
    controller = DownwashTestController(initial_obs, info={}, initial_info=config)

    cam_cfg = {"distance": 6.0, "azimuth": 150.0, "elevation": -20.0, "lookat": [0.0, 0.0, 0.8]}
    fps = 60
    t = 0.0
    i = 0
    min_clearance = np.inf
    downwash_steps = 0
    rotor_vel = np.zeros(4, dtype=np.float32)

    while i < MAX_STEPS:
        opp_pos = obstacle_positions(t)  # (2, 3)

        # Inject opponent positions as MPPI obstacles before computing action.
        # The collision cost (XY-only) reads self.obstacles, so the drone routes
        # around the hovering opponents horizontally — i.e. out of their downwash.
        controller.obstacles = jax.device_put(jnp.array(opp_pos), controller.sim.device)

        obs = get_our_obs(sim, rotor_vel)
        action = controller.compute_control(obs, {})
        rotor_vel = controller.thrust.copy()

        # Apply our drone's action
        cmd = sim.data.controls.attitude.staged_cmd.at[0, 0].set(jnp.array(action))
        sim.data = sim.data.replace(
            controls=sim.data.controls.replace(
                attitude=sim.data.controls.attitude.replace(staged_cmd=cmd)
            )
        )

        # Step physics
        sim.data = step_fn(sim.data, N_PHYS_STEPS)

        # Override ONLY obstacle drone states (indices 1-2); never touch drone-0.
        # Without this, obstacle drones fall under gravity each step.
        n_opp = DownwashTestController.N_OPPONENTS
        opp_jnp = jnp.array(opp_pos)
        new_pos = sim.data.states.pos.at[0, 1:].set(opp_jnp)
        new_quat = sim.data.states.quat.at[0, 1:].set(jnp.tile(HOVER_QUAT, (n_opp, 1)))
        new_vel = sim.data.states.vel.at[0, 1:].set(jnp.zeros((n_opp, 3)))
        new_ang_vel = sim.data.states.ang_vel.at[0, 1:].set(jnp.zeros((n_opp, 3)))
        sim.data = sim.data.replace(
            states=sim.data.states.replace(
                pos=new_pos, quat=new_quat, vel=new_vel, ang_vel=new_ang_vel
            )
        )

        # Tracking
        our_pos = np.array(sim.data.states.pos[0, 0])
        diff_xy = np.linalg.norm(opp_pos[:, :2] - our_pos[:2], axis=-1)
        below = our_pos[2] < opp_pos[:, 2]
        if (diff_xy[below] < DOWNWASH_DETECT_R).any():
            downwash_steps += 1
        min_clearance = min(min_clearance, float(diff_xy.min()))

        # Render
        if render and (i * fps) % CTRL_FREQ < fps:
            for apex in opp_pos:
                draw_downwash_cone(sim, apex)
            controller.render_callback(sim)
            sim.render(mode="human", camera=-1, cam_config=cam_cfg)

        reached = np.linalg.norm(our_pos - DRONE_GOAL) < GOAL_RADIUS
        if reached:
            break

        t += 1.0 / CTRL_FREQ
        i += 1

    sim.close()

    our_pos = np.array(sim.data.states.pos[0, 0])
    print("\n--- Downwash Avoidance Summary ---")
    print(f"Reached goal:                     {np.linalg.norm(our_pos - DRONE_GOAL) < GOAL_RADIUS}")
    print(
        f"Steps in any downwash zone:       {downwash_steps}  ({downwash_steps / CTRL_FREQ:.1f} s)"
    )
    print(f"Minimum XY clearance (any drone): {min_clearance:.3f} m")
    print(f"Total steps:                      {i}  ({i / CTRL_FREQ:.1f} s)")


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("lsy_drone_racing").setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    fire.Fire(simulate, serialize=lambda _: None)
