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
    behind_radius: float
    contour_behind_scale: float


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


def symmetric_collision(params: OpponentCostParams, pos: Array, n_agents: int) -> Array:
    """Old symmetric (world-paired, isotropic) drone-drone collision, for every agent.

    Only used when ``opponent.yields`` is on: it is what makes modeled opponents avoid the ego
    in their own MPPI update. Off by default, because a real opponent will not yield and the ego
    must not plan as if it would.

    Args: pos (W, A, 3). Returns (W, A), self-masked.
    """
    safe = params.drone_radius * 2.5
    delta = pos[:, :, None, :] - pos[:, None, :, :]  # (W, A, A, 3)
    d_pair = jnp.linalg.norm(delta, axis=-1) / safe  # (W, A, A)
    eye = jnp.eye(n_agents)
    return (jnp.exp(-(d_pair**2)) * (1.0 - eye)).sum(-1)  # (W, A), self-masked


def behind_contour_factor(
    params: OpponentCostParams, ego_pos: Array, ego_tangent: Array, opp_pos: Array
) -> Array:
    """Contour-cost scale factor letting the ego leave the racing line to overtake.

    When an opponent prediction is ahead of the ego along its path tangent AND within
    behind_radius, the ego's off-path cost is scaled down so leaving the line becomes cheap.

    Args: ego_pos (W, 3); ego_tangent (W, 3); opp_pos (P, 3).
    Returns (W,) multiplier, either contour_behind_scale or 1.0.
    """
    rel = opp_pos[None, :, :] - ego_pos[:, None, :]  # (W, P, 3) opp relative to ego
    ahead = jnp.sum(ego_tangent[:, None, :] * rel, axis=-1)  # (W, P): opp ahead of ego
    dist = jnp.linalg.norm(rel, axis=-1)  # (W, P)
    behind = ((ahead > 0.0) & (dist < params.behind_radius)).any(axis=-1)  # (W,)
    return jnp.where(behind, params.contour_behind_scale, 1.0)


def build_cost_fn(
    weights: dict[str, float],
    opp_params: OpponentCostParams,
    paths: ReferencePaths,
    geometry: Geometry,
    n_agents: int,
    n_sim_agents: int,
    opponent_model: str,
    opp_yields: bool,
    use_behind_contour: bool,
    rep_w: Array | None,
    rep_a: Array | None,
) -> Callable[..., dict[str, Array]]:
    """Assemble the per-step cost over all simulated agents.

    Baked into a closure so the weights, LUTs and agent counts stay compile-time constants;
    `rollout.build_rollout_fn` takes the result as its `cost_fn`.

    Args:
        weights: per-drone cost weights (see :func:`single_drone_terms`).
        opp_params: opponent keep-out geometry.
        paths: arc-length reference LUTs, one per agent.
        geometry: physical radii.
        n_agents: total agents, including predictor-mode opponents that are never simulated.
        n_sim_agents: agents whose dynamics are in the rollout batch.
        opponent_model: "mppi", "spline_progress" or "const_vel".
        opp_yields: give modelled opponents a collision term in their own update.
        use_behind_contour: relax the ego contour cost to allow overtaking.
        rep_w: (P,) world indices of the opponent representatives, "mppi" mode only.
        rep_a: (P,) agent indices of the opponent representatives, "mppi" mode only.

    Returns:
        A jitted ``cost_fn(data, theta, v_theta, obstacles, gate_frame_obstacles, opp_pred_pos,
        opp_pred_vel, opp_inflate)`` returning term_name -> (W, A).
    """
    save_dist_obst = geometry.obstacle_radius + geometry.drone_radius
    save_dist_gate_obst = geometry.gate_frame_radius + geometry.drone_radius

    def cost_fn(
        data: SimData,
        theta: Array,
        v_theta: Array,
        obstacles: Array,
        gate_frame_obstacles: Array,
        opp_pred_pos: Array,
        opp_pred_vel: Array,
        opp_inflate: Array,
    ) -> dict[str, Array]:
        """Per-term cost for all simulated agents (MPCC contour/lag/progress formulation).

        Terms are kept separate rather than summed so the rollout can accumulate each over the
        horizon and the logger can report the per-mode breakdown.

        The ego is scored against a fixed set of P opponent predictions, worst case over P: in
        "mppi" mode the zero-noise mode-mean representative worlds, in predictor modes the
        kinematic trajectories passed in. For n_agents == 1 the opponent block is skipped
        entirely and this reduces to the single-agent cost.
        """
        pos = data.states.pos  # (W, A, 3)
        vel = data.states.vel  # (W, A, 3) — real heading for the anisotropic keep-out
        ang_vel = data.states.ang_vel  # (W, A, 3)
        ang_acc = data.states_deriv.ang_vel  # (W, A, 3)
        cmd = data.controls.attitude.staged_cmd  # (W, A, 4)

        # The loop unrolls over the small agent axis at trace time; each agent uses its own LUT.
        per_agent, tangents = [], []
        for a in range(n_sim_agents):
            ref_pos, tangent = vmap(reference.ref_at_theta, in_axes=(0, None, None, None))(
                theta[:, a], paths.theta_grid[a], paths.pos_lut[a], paths.tan_lut[a]
            )
            tangents.append(tangent)
            per_agent.append(
                single_drone_terms(
                    pos[:, a], ang_vel[:, a], ang_acc[:, a], cmd[:, a], ref_pos, tangent,
                    v_theta[:, a], obstacles, gate_frame_obstacles,
                    save_dist_obst, save_dist_gate_obst, weights,
                )
            )
        terms = {k: jnp.stack([pa[k] for pa in per_agent], axis=1) for k in per_agent[0]}
        tangent = jnp.stack(tangents, axis=1)  # (W, A, 3)

        if n_agents > 1:
            if opponent_model == "mppi":
                rep_pos = pos[rep_w, rep_a]  # (P, 3)
                rep_vel = vel[rep_w, rep_a]  # (P, 3)
            else:
                rep_pos, rep_vel = opp_pred_pos, opp_pred_vel  # (P, 3)
            coll, downwash = opponent_terms(
                opp_params, pos[:, 0], rep_pos, rep_vel, opp_inflate
            )
            n_worlds = pos.shape[0]
            terms["opp_drone"] = jnp.zeros((n_worlds, n_sim_agents)).at[:, 0].set(coll)
            terms["downwash"] = jnp.zeros((n_worlds, n_sim_agents)).at[:, 0].set(downwash)
            if opponent_model == "mppi" and opp_yields:
                # Off by default: a real opponent will not yield, so the ego must not plan as if
                # it would.
                coll_pair = symmetric_collision(opp_params, pos, n_sim_agents)
                terms["opp_drone"] = terms["opp_drone"].at[:, 1:].set(
                    opp_params.drone_exp * coll_pair[:, 1:]
                )

            if use_behind_contour:
                terms["contour"] = terms["contour"].at[:, 0].multiply(
                    behind_contour_factor(opp_params, pos[:, 0], tangent[:, 0], rep_pos)
                )
        return terms

    return jax.jit(cost_fn)
