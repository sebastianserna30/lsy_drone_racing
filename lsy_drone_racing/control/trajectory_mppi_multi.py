"""This module wraps the AttitudeController to handle batched multi-agent environments.

In multi-agent simulations, observations are batched across all drones.
The rank index is used to select the state of the current drone.
"""

from __future__ import annotations  # Python 3.10 type hints

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from crazyflow.sim import Sim
from crazyflow.sim.visualize import draw_line, draw_points
from ml_collections import ConfigDict

from lsy_drone_racing.control.trajectory_mppi import (
    AttitudeMPPIController as SingleAttitudeMPPIController,
)

if TYPE_CHECKING:
    from crazyflow.sim.data import SimData
    from numpy.typing import NDArray


class AttitudeMPPIController(SingleAttitudeMPPIController):
    """Example of a controller using the collective thrust and attitude interface."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the attitude controller.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: Additional environment information from the reset.
            config: The configuration of the environment.
        """
        self.rank = info["rank"]
        if self.rank == 0:
            self.opponent = 1
        else:
            self.opponent = 0

        controller_cfg = config["controller"][self.rank]
        # Convert to a plain dict first
        config_dict = config.to_dict()

        # Replace controller
        config_dict["controller"] = controller_cfg

        # Create a new ConfigDict
        config = ConfigDict(config_dict)

        # changedPractical: opponent-interaction tuning (anisotropic keep-out + behind-aware
        # contour relaxation). Set BEFORE super().__init__ — the base warms up the controller
        # (calls compute_control -> compute_cost) inside __init__, so these must exist first.
        cost_cfg = config["controller"]["mppi"].get("cost", {})
        # Anisotropic (elliptical) opponent keep-out semi-axes (m): axial > lateral makes sitting
        # fore/aft of the opponent expensive and being beside it cheap -> pass instead of brake.
        self.opp_axial = float(cost_cfg.get("opp_axial", 0.25))
        self.opp_lateral = float(cost_cfg.get("opp_lateral", 0.10))
        self.use_anisotropic_opp = bool(cost_cfg.get("use_anisotropic_opp", True))
        # Behind-aware contour relaxation: when the opponent is ahead-on-our-line and within
        # behind_radius, scale the contour (off-path) cost down so the drone can swerve to pass.
        self.contour_behind_scale = float(cost_cfg.get("contour_behind_scale", 0.15))
        self.behind_radius = float(cost_cfg.get("behind_radius", 0.6))
        self.use_behind_contour = bool(cost_cfg.get("use_behind_contour", True))
        # changedPractical: opponent-bubble visualization style.
        #   "gaussian" -> nested translucent ellipsoid shells (alpha ~ exp(-r^2) = the cost bump)
        #   "solid"    -> one translucent ellipsoid at the d_aniso=1 level set
        #   "line"     -> the old horizontal ellipse outline
        self.bubble_style = str(cost_cfg.get("bubble_style", "gaussian"))
        self.bubble_shells = int(cost_cfg.get("bubble_shells", 6))

        super().__init__({k: v[self.rank] for k, v in obs.items()}, info, config)

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired collective thrust and roll/pitch/yaw of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The orientation as roll, pitch, yaw angles, and the collective thrust
            [r_des, p_des, y_des, t_des] as a numpy array.
        """
        first_value = next(iter(obs.values()))

        if first_value.ndim == 2:
            # This is the normal mode for multilevel where other drone is pressent in observations
            info["opponent_pos"] = obs["pos"][self.opponent]
            info["opponent_vel"] = obs["vel"][self.opponent]
            # changedPractical: host copies for render_callback (anisotropic bubble viz).
            self._opp_pos_host = np.asarray(info["opponent_pos"], dtype=float)
            self._opp_vel_host = np.asarray(info["opponent_vel"], dtype=float)
            return super().compute_control(
                {k: v[self.rank] for k, v in obs.items()}, info
            )

        # This backup is needed for the warmup of the controller
        return super().compute_control(obs, info)

    @partial(jax.jit, static_argnames=["self"])
    def compute_cost(
        self,
        data: SimData,
        theta: jnp.ndarray,
        v_theta: jnp.ndarray,
        reference: dict[str, jnp.ndarray],
        obstacles: jnp.ndarray,
        gate_frame_obstacles: jnp.ndarray,
    ) -> dict[str, jnp.ndarray]:
        """Add the opponent-drone collision term to the base per-term cost dict.

        changedPractical: signature follows the MPCC base controller (theta, v_theta added);
        this override only adds the opponent-collision term and forwards the rest.
        """
        ## 6. Collision Cost (Safety)
        safe_dist = (
            self.initial_info["experiment"]["env"]["drone_radius"] * 2.5
        )  # 2 would be theory and added buffer

        pos = data.states.pos[:, 0, :]  # Shape: (n_rollouts, 3)
        opp_pos = reference["opp_pos"]  # (3,) this horizon step
        opp_vel = reference["opp_vel"]  # (3,) opponent heading this horizon step
        delta = pos - opp_pos[None, :]  # (n_rollouts, 3)
        dist_drones = jnp.linalg.norm(delta, axis=-1)  # (n_rollouts,)

        binary_cost = False

        if binary_cost:
            # per-rollout (n_rollouts,): single opponent, so no sum over an axis.
            # (the old jnp.sum(..., axis=-1) collapsed this to a scalar that added the same
            #  constant to every rollout — a no-op for elite/softmax selection)
            opponent_drone_hits = jnp.where(dist_drones < safe_dist, 1, 0)
            coll_cost = self.w_opp_drone * opponent_drone_hits
        elif self.use_anisotropic_opp:
            # anisotropic (elliptical) keep-out elongated ALONG the
            # opponent's heading. Split delta into along-heading and perpendicular parts, scale
            # by semi-axes (axial > lateral). Fore/aft of the opponent -> small scaled dist ->
            # high cost; beside it -> large scaled dist -> low cost. So the low-cost escape is a
            # sideways overtake, not a brake. Falls back to isotropic when opponent ~stationary.
            opp_speed = jnp.linalg.norm(opp_vel)
            heading = opp_vel / (opp_speed + 1e-6)
            d_along = delta @ heading  # signed
            d_perp = jnp.linalg.norm(delta - d_along[:, None] * heading, axis=-1)
            d_aniso = jnp.sqrt(
                (d_along / self.opp_axial) ** 2 + (d_perp / self.opp_lateral) ** 2
            )
            d_scaled = jnp.where(opp_speed > 0.2, d_aniso, dist_drones / safe_dist)
            coll_cost = self.w_opp_drone_exp * jnp.exp(-(d_scaled**2))
        else:
            # smooth collsion cost using exponent (isotropic circle)
            coll_cost = self.w_opp_drone_exp * jnp.exp(
                -((dist_drones / safe_dist) ** 2)
            )

        use_collision_cost = True
        if use_collision_cost:
            collosion_cost = coll_cost
        else:
            collosion_cost = jnp.zeros(pos.shape[0])

        terms = super().compute_cost(
            data, theta, v_theta, reference, obstacles, gate_frame_obstacles
        )

        # changedPractical (#3): behind-aware contour relaxation. Project the opponent onto our
        # own path tangent; if it is ahead of us (opp_ahead > 0) AND close, we are stuck behind
        # on the racing line -> scale the off-path (contour) penalty down so the drone is allowed
        # to leave the line, pass, and rejoin. lag + gate-frame cost still hold the rest.
        if False:  # self.use_behind_contour:
            _, tangent = jax.vmap(self._ref_at_theta)(
                theta
            )  # (n_rollouts, 3) unit tangent
            opp_ahead = jnp.sum(
                tangent * (opp_pos[None, :] - pos), axis=-1
            )  # (n_rollouts,)
            behind = (opp_ahead > 0.0) & (dist_drones < self.behind_radius)
            terms["contour"] = terms["contour"] * jnp.where(
                behind, self.contour_behind_scale, 1.0
            )

        terms["opp_drone"] = collosion_cost
        return terms

    def step_callback(
        self,
        action: NDArray[np.floating] | None = None,
        obs: dict[str, NDArray[np.floating]] | None = None,
        reward: float | None = None,
        terminated: bool | None = None,
        truncated: bool | None = None,
        info: dict | None = None,
    ) -> bool:
        """Increment the tick counter."""
        return super().step_callback(action, {k: v[self.rank] for k, v in obs.items()})

    def render_callback(self, sim: Sim):
        """Visualize the desired trajectory and the current setpoint."""
        super().render_callback(sim)
        draw_line(sim, self.opp_traj, rgba=(0.0, 0.0, 1.0, 1.0))
        self._draw_opp_bubble(sim)

    def _opp_bubble_frame(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """Return (center, semi_axes, rot_mat_flat, anisotropic?) for the opponent bubble.

        The anisotropic keep-out is a prolate spheroid: semi-axis opp_axial ALONG the
        opponent heading and opp_lateral in BOTH perpendicular directions (the cost's
        d_perp is the full 3D off-heading distance). We build an orthonormal frame whose
        local x-axis is the heading. When the opponent is ~stationary the cost falls back
        to an isotropic sphere of radius safe_dist, so we return that instead.
        """
        o = self._opp_pos_host
        u = self._opp_vel_host
        speed = float(np.linalg.norm(u))
        if speed > 0.2 and self.use_anisotropic_opp:
            h = u / (speed + 1e-6)  # unit heading (full 3D)
            ref = np.array([0.0, 0.0, 1.0]) if abs(h[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
            e2 = np.cross(ref, h)
            e2 /= np.linalg.norm(e2) + 1e-9
            e3 = np.cross(h, e2)
            mat = np.stack([h, e2, e3], axis=-1)  # columns = local x/y/z axes
            semi = np.array([self.opp_axial, self.opp_lateral, self.opp_lateral])
            return o, semi, mat.flatten(), True
        safe_dist = self.initial_info["experiment"]["env"]["drone_radius"] * 2.5
        return o, np.array([safe_dist] * 3), np.eye(3).flatten(), False

    def _draw_opp_bubble(self, sim: Sim):
        """Draw the anisotropic keep-out zone around the opponent as a 3D ellipsoid.

        changedPractical: replaces the flat line outline. Three styles via `bubble_style`:
          - "gaussian": nested translucent ellipsoid shells whose alpha follows the actual
            cost bump exp(-r^2) at each shell's normalized radius r -> a soft glow that IS
            the Gaussian penalty field (brightest core, fading rim).
          - "solid": a single translucent ellipsoid at the d_aniso=1 level set.
          - "line": the old horizontal ellipse outline (cheap fallback).
        """
        if sim.viewer is None or getattr(self, "_opp_pos_host", None) is None:
            return
        o, semi, mat, aniso = self._opp_bubble_frame()
        rgb = (1.0, 0.5, 0.0) if aniso else (1.0, 0.0, 0.0)  # orange aniso / red isotropic

        if self.bubble_style == "line":
            n = 48
            phi = np.linspace(0.0, 2.0 * np.pi, n)
            axes = mat.reshape(3, 3)
            ring = (
                o[None, :]
                + semi[0] * np.cos(phi)[:, None] * axes[:, 0][None, :]
                + semi[1] * np.sin(phi)[:, None] * axes[:, 1][None, :]
            )
            draw_line(sim, ring, rgba=(*rgb, 1.0))
            return

        viewer = sim.viewer.viewer
        if self.bubble_style == "gaussian":
            # Outer shells first (painter's order) so the bright core reads on top.
            # r = normalized radius; alpha = exp(-r^2) is exactly the cost value there.
            radii = np.linspace(1.6, 0.35, self.bubble_shells)
            for r in radii:
                alpha = 0.35 * float(np.exp(-(r**2)))
                viewer.add_marker(
                    type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                    pos=o,
                    size=semi * r,
                    mat=mat,
                    rgba=np.array([*rgb, alpha]),
                )
        else:  # "solid"
            viewer.add_marker(
                type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                pos=o,
                size=semi,
                mat=mat,
                rgba=np.array([*rgb, 0.3]),
            )
