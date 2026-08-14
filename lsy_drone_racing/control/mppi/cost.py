"""MPPI cost terms.

Pure functions over explicit arrays, called from inside the controller's jitted rollout scan.
Two families:

* :func:`single_drone_terms` — the per-drone MPCC cost (contour/lag/progress, control effort,
  obstacles, floor). Identical for every agent; the caller supplies that agent's reference.
* the opponent family — ego-only terms scored against P opponent predictions. These are what
  makes the multi-agent case multi-agent.

Weights ride along in dicts / a frozen ``OpponentCostParams`` that the controller holds as a
static attribute, so they stay compile-time constants in the trace.

Shape conventions: W = rollout worlds, A = simulated agents, P = opponent predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp
from jax import vmap

from lsy_drone_racing.control.mppi import reference

if TYPE_CHECKING:
    from crazyflow.sim.data import SimData
    from jax import Array

    from lsy_drone_racing.control.mppi.config import Geometry
    from lsy_drone_racing.control.mppi.reference import ReferencePaths

HOVER_THRUST = 0.43  # collective thrust (N) that approximately balances gravity for cf21B_500


def single_drone_terms(
    pos: Array,
    ang_vel: Array,
    ang_acc: Array,
    cmd: Array,
    ref_pos: Array,
    tangent: Array,
    v_theta: Array,
    obstacles: Array,
    gate_frame_obstacles: Array,
    save_dist_obst: float,
    save_dist_gate_obst: float,
    w: dict[str, float],
) -> dict[str, Array]:
    """Per-drone MPCC cost terms (no opponent coupling).

    Each input is (n_rollouts, ...) for ONE drone; the caller supplies ref_pos/tangent (from that
    agent's arc-length LUT) and the static weight dict `w`. Returns term_name -> (n_rollouts,).

    The returned dict's key order is load-bearing: the caller sums the terms in iteration order
    and float addition is not associative, so reordering perturbs the trajectory.
    """
    ## 1. MPCC tracking cost — contour + lag + progress
    err = pos - ref_pos  # (n_rollouts, 3)
    # Lag error: signed component of err ALONG the path tangent (ahead +, behind -).
    e_lag = jnp.sum(tangent * err, axis=-1)  # (n_rollouts,)
    # Contour error: component PERPENDICULAR to the path (off-track distance).
    e_con_vec = err - e_lag[:, None] * tangent  # (n_rollouts, 3)
    lag_cost = e_lag**2 * w["lag"]
    contour_cost = jnp.sum(e_con_vec**2, axis=-1) * w["contour"]
    # Extra altitude penalty toward the reference height (kept from the old cost).
    z_cost = jnp.abs(pos[..., 2] - ref_pos[..., 2]) * w["z"]
    # Progress reward: advancing theta lowers cost (this is what removes the t_total ceiling).
    progress_cost = -w["progress"] * v_theta
    ang_vel_error = jnp.linalg.norm(ang_vel, axis=-1)
    ang_vel_cost = ang_vel_error**2 * w["ang_vel"]
    ang_acc_error = jnp.linalg.norm(ang_acc, axis=-1)
    ang_acc_cost = ang_acc_error**2 * w["ang_acc"]

    ## 2. Control Cost (Efficiency + Stability)
    tilt_cost = jnp.linalg.norm(cmd[:, :2], axis=-1) ** 2 * w["tilt"]
    thrust_cost = (cmd[:, 3] - HOVER_THRUST) ** 2 * w["thrust"]
    yaw_cost = (cmd[:, 2] - 0) ** 2 * w["yaw"]  # des_yaw should be 0

    ## 3. Obstacle Cost (Safety) — binary hit count
    obs_diff = jnp.linalg.norm(pos[..., None, :2] - obstacles[None, :, :2], axis=-1)
    obstacle_hits = jnp.where(obs_diff < save_dist_obst, 1, 0)
    obstacle_cost = w["obstacle"] * jnp.sum(obstacle_hits, axis=-1)

    ## 4. Gate Obstacle Cost (Safety) — binary hit count
    gate_obs_diff = jnp.linalg.norm(pos[..., None, :2] - gate_frame_obstacles[None, :, :2], axis=-1)
    gate_obstacle_hits = jnp.where(gate_obs_diff < save_dist_gate_obst, 1, 0)
    gate_obstacle_cost = w["obstacle"] * jnp.sum(gate_obstacle_hits, axis=-1)

    ## 5. Floor penalty — prevents rollouts from sinking into the ground
    floor_cost = jnp.where(
        pos[..., 2] < w["floor_z"], (w["floor_z"] - pos[..., 2]) ** 2 * w["floor"], 0.0
    )

    return {
        "lag": lag_cost,
        "contour": contour_cost,
        "z": z_cost,
        "progress": progress_cost,
        "ang_vel": ang_vel_cost,
        "ang_acc": ang_acc_cost,
        "tilt": tilt_cost,
        "thrust": thrust_cost,
        "yaw": yaw_cost,
        "obstacle": obstacle_cost,
        "gate_obstacle": gate_obstacle_cost,
        "floor": floor_cost,
    }


@dataclass(frozen=True)
class OpponentCostParams:
    """Keep-out geometry and weights for the ego's opponent-coupled cost terms."""

    drone_radius: float
    drone_exp: float  # weight of the smooth exponential collision cost
    use_anisotropic: bool
    axial: float  # fore/aft keep-out half-length (m)
    lateral: float  # sideways keep-out half-width (m)
    core_radius: float  # hard isotropic core the ellipse may never admit closer than
    blend_v0: float  # iso<->aniso blend centre (m/s)
    blend_width: float  # blend width (m/s)
    downwash: float  # below-opponent wake penalty
    downwash_radius: float  # wake cylinder radius (m)
    downwash_dz: float  # wake depth below the opponent (m)


def opponent_terms(
    params: OpponentCostParams, ego_pos: Array, opp_pos: Array, opp_vel: Array, inflate: Array
) -> tuple[Array, Array]:
    """Ego collision + downwash cost against P opponent predictions (worst-case over P).

    An anisotropic ellipse with a hard isotropic core: opp_lateral alone is thinner than two
    airframes plus state-estimate error, so the ellipse may never admit closer than
    core_radius. The speed blend replaces a hard moving/static switch that chattered on real
    velocity estimates. The downwash term is the only protection in real flight, since the sim
    does not model downwash at all.

    Args: ego_pos (W, 3); opp_pos / opp_vel (P, 3); inflate (P,).
    Returns (collision cost (W,), downwash cost (W,)).
    """
    safe = params.drone_radius * 2.5
    delta = ego_pos[:, None, :] - opp_pos[None, :, :]  # (W, P, 3)
    dist = jnp.linalg.norm(delta, axis=-1)  # (W, P)
    r_iso = dist / (safe * inflate[None, :])
    if params.use_anisotropic:
        # Ellipse elongated ALONG the opponent's velocity heading: axial (fore/aft) > lateral,
        # so sitting behind is costly, drawing alongside is cheap -> sideways overtake.
        spd = jnp.linalg.norm(opp_vel, axis=-1)  # (P,)
        heading = opp_vel / (spd[:, None] + 1e-6)  # (P, 3) unit heading
        d_par = jnp.sum(delta * heading[None, :, :], axis=-1)  # (W, P)
        d_perp = jnp.linalg.norm(delta - d_par[..., None] * heading[None, :, :], axis=-1)
        r_aniso = jnp.sqrt(
            (d_par / (params.axial * inflate[None, :])) ** 2
            + (d_perp / (params.lateral * inflate[None, :])) ** 2
        )
        # fix #4: hard isotropic core — min picks the smaller normalized distance, i.e. the
        # LARGER cost, so the full penalty always applies inside opp_core_radius.
        r_core = dist / (params.core_radius * inflate[None, :])
        r_aniso = jnp.minimum(r_aniso, r_core)
        # fix #5: smooth iso<->aniso blend on opponent speed (was a hard spd>0.2 switch that
        # chatters on a noisy real velocity estimate; the estimate is EMA-filtered upstream).
        blend = jax.nn.sigmoid((spd - params.blend_v0) / params.blend_width)  # (P,)
        r = blend[None, :] * r_aniso + (1.0 - blend[None, :]) * r_iso
    else:
        r = r_iso
    coll = params.drone_exp * jnp.max(jnp.exp(-(r**2)), axis=-1)  # worst case over P
    # fix #3: downwash keep-out — binary penalty for sitting BELOW an opponent inside a
    # cylinder of radius downwash_radius extending downwash_dz downward from it.
    dz = opp_pos[None, :, 2] - ego_pos[:, 2, None]  # (W, P), >0 when ego is below the opponent
    lat = jnp.linalg.norm(delta[..., :2], axis=-1)  # (W, P) lateral offset
    in_wake = (
        (dz > 0.0) & (dz < params.downwash_dz) & (lat < params.downwash_radius * inflate[None, :])
    )
    downwash = params.downwash * jnp.max(in_wake.astype(jnp.float32), axis=-1)
    return coll, downwash


def build_stage_cost_fn(
    weights: dict[str, float],
    paths: ReferencePaths,
    geometry: Geometry,
    n_sim_agents: int,
) -> Callable[..., dict[str, Array]]:
    """Assemble the per-step DECOUPLED cost over all simulated agents.

    Every term here is a pure function of one agent's own state, so it can be accumulated
    inside the rollout scan and never needs re-evaluating. The opponent-coupled terms live in
    :func:`build_coupled_cost_fn` and run after the scan, on the cached trajectories, so the
    best-response iteration can re-score them without re-rolling the sim.

    Baked into a closure so the weights, LUTs and agent counts stay compile-time constants;
    `rollout.build_rollout_fn` takes the result as its `stage_cost_fn`.

    Args:
        weights: per-drone cost weights (see :func:`single_drone_terms`).
        paths: arc-length reference LUTs, one per agent.
        geometry: physical radii.
        n_sim_agents: agents whose dynamics are in the rollout batch.

    Returns:
        A jitted ``stage_cost_fn(data, theta, v_theta, obstacles, gate_frame_obstacles)``
        returning term_name -> (W, A).
    """
    save_dist_obst = geometry.obstacle_radius + geometry.drone_radius
    save_dist_gate_obst = geometry.gate_frame_radius + geometry.drone_radius

    def stage_cost_fn(
        data: SimData,
        theta: Array,
        v_theta: Array,
        obstacles: Array,
        gate_frame_obstacles: Array,
    ) -> dict[str, Array]:
        """Per-term decoupled cost for all simulated agents (MPCC contour/lag/progress).

        Terms are kept separate rather than summed so the rollout can accumulate each over the
        horizon and the logger can report the per-mode breakdown.
        """
        pos = data.states.pos  # (W, A, 3)
        ang_vel = data.states.ang_vel  # (W, A, 3)
        ang_acc = data.states_deriv.ang_vel  # (W, A, 3)
        cmd = data.controls.attitude.staged_cmd  # (W, A, 4)

        # The loop unrolls over the small agent axis at trace time; each agent uses its own LUT.
        per_agent = []
        for a in range(n_sim_agents):
            ref_pos, tangent = vmap(reference.ref_at_theta, in_axes=(0, None, None, None))(
                theta[:, a], paths.theta_grid[a], paths.pos_lut[a], paths.tan_lut[a]
            )
            per_agent.append(
                single_drone_terms(
                    pos[:, a], ang_vel[:, a], ang_acc[:, a], cmd[:, a], ref_pos, tangent,
                    v_theta[:, a], obstacles, gate_frame_obstacles,
                    save_dist_obst, save_dist_gate_obst, weights,
                )
            )
        return {k: jnp.stack([pa[k] for pa in per_agent], axis=1) for k in per_agent[0]}

    return jax.jit(stage_cost_fn)


def build_opp_state_fn(
    opponent_model: str, rep_w: Array | None, rep_a: Array | None
) -> Callable[..., tuple[Array, Array]]:
    """Per-step opponent pose the ego is scored against, emitted from inside the rollout scan.

    Two sources, one shape. In "mppi" mode the opponents are simulated in the same batch, so
    their representative worlds are gathered from the live sim state; in the predictor modes
    they are not simulated at all and the host-side kinematic trajectory is passed straight
    through. Either way the coupled cost consumes (P, 3) per step.

    Args:
        opponent_model: "mppi", "spline_progress" or "const_vel".
        rep_w: (P,) world indices of the opponent representatives, "mppi" mode only.
        rep_a: (P,) agent indices of the opponent representatives, "mppi" mode only.

    Returns:
        ``opp_state_fn(data, opp_pred_pos_t, opp_pred_vel_t) -> (pos (P, 3), vel (P, 3))``.
    """

    def opp_state_fn(
        data: SimData, opp_pred_pos_t: Array, opp_pred_vel_t: Array
    ) -> tuple[Array, Array]:
        if opponent_model == "mppi":
            return data.states.pos[rep_w, rep_a], data.states.vel[rep_w, rep_a]
        return opp_pred_pos_t, opp_pred_vel_t

    return opp_state_fn


def build_coupled_cost_fn(
    opp_params: OpponentCostParams, n_sim_agents: int
) -> Callable[..., dict[str, Array]]:
    """Assemble the opponent-coupled cost, evaluated AFTER the rollout on cached trajectories.

    Split out from the stage cost precisely so it is re-evaluable: iterative best response
    re-scores these terms against a changing opponent selection many times per control step,
    and re-running the sim rollout for each pass would be unaffordable.

    Args:
        opp_params: opponent keep-out geometry.
        n_sim_agents: agents whose dynamics are in the rollout batch.

    Returns:
        A jitted ``coupled_cost_fn(ego_pos, opp_pos, opp_vel, opp_inflate)`` where ego_pos is
        (N, W, 3), opp_pos/opp_vel are (N, P, 3) and opp_inflate is (P,), returning
        term_name -> (W, A) already summed over the horizon.
    """

    def coupled_cost_fn(
        ego_pos: Array, opp_pos: Array, opp_vel: Array, opp_inflate: Array
    ) -> dict[str, Array]:
        """Horizon-summed collision + downwash cost for the acting agent.

        vmapped over the leading horizon axis; the per-step arithmetic is exactly what the
        in-scan version used to do, so the horizon sum is unchanged.
        """
        def step(ego_p: Array, opp_p: Array, opp_v: Array) -> tuple[Array, Array]:
            return opponent_terms(opp_params, ego_p, opp_p, opp_v, opp_inflate)

        coll, downwash = vmap(step)(ego_pos, opp_pos, opp_vel)  # (N, W) each
        coll, downwash = jnp.sum(coll, axis=0), jnp.sum(downwash, axis=0)  # (W,) each
        n_worlds = ego_pos.shape[1]
        return {
            "opp_drone": jnp.zeros((n_worlds, n_sim_agents)).at[:, 0].set(coll),
            "downwash": jnp.zeros((n_worlds, n_sim_agents)).at[:, 0].set(downwash),
        }

    return jax.jit(coupled_cost_fn)
