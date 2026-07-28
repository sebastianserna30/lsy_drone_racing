"""Strict configuration parsing for the MPPI controller.

Every parameter is required. There are no defaults, and unknown keys are rejected. This is a
deliberate reversal of the previous ``cfg.get(key, default)`` style, which silently hid four
cost weights that no code read any more (``pos``, ``vel``, ``obstacle_exp``, ``opp_drone``) and
would have flown a differently-tuned controller had a key ever been misspelled.

The resulting dataclasses are frozen and hold only scalars, so they are hashable and can be
closed over by (or passed as static arguments to) jitted functions. That matters: a closed-over
Python float is folded into the JAX trace exactly the way a ``self.`` attribute is, which is what
lets the weights stay compile-time constants.

TOML layout::

    [controller]                    ctrl_freq              (file/type belong to the framework)
    [controller.mppi]               horizon, sampling, v_theta, action_ema, agents
    [controller.mppi.cost]          ego cost weights
    [controller.mppi.opponent]      opponent model, prediction, keep-out, downwash
    [controller.mppi.spline]        reference-path generation
    [controller.mppi.geometry]      physical radii
"""

from __future__ import annotations

import dataclasses
import difflib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import Mapping

T = TypeVar("T")

OPPONENT_MODELS = ("mppi", "spline_progress", "const_vel")


class ConfigError(ValueError):
    """Raised when a config section is missing keys, has unknown keys, or has a bad value."""


def _coerce(value: Any, field: dataclasses.Field, path: str) -> Any:
    """Coerce a raw TOML value to the field's declared type, or raise.

    TOML cannot distinguish ``1`` from ``1.0``, so ints are accepted for float fields (and
    integral floats for int fields). Bools are checked strictly, because accepting ``1`` for a
    feature flag is exactly the kind of silence this module exists to remove.
    """
    kind = field.type if isinstance(field.type, type) else _unwrap(field.type)
    if kind is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path}.{field.name}: expected true/false, got {value!r}")
        return value
    if kind is int:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path}.{field.name}: expected an integer, got {value!r}")
        if int(value) != value:
            raise ConfigError(f"{path}.{field.name}: expected an integer, got {value!r}")
        return int(value)
    if kind is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path}.{field.name}: expected a number, got {value!r}")
        return float(value)
    if kind is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path}.{field.name}: expected a string, got {value!r}")
        return value
    return value


def _unwrap(annotation: Any) -> Any:
    """Resolve an annotation to a builtin type.

    ``from __future__ import annotations`` makes every field type a string, so this works on
    text. Optional wrappers are stripped so ``int | None`` coerces like ``int``.
    """
    text = str(annotation).replace("Optional[", "").replace("]", "")
    text = text.replace("| None", "").replace("None |", "").strip()
    return {"bool": bool, "int": int, "float": float, "str": str}.get(text, object)


def _suggest(key: str, known: set[str]) -> str:
    """Render a 'did you mean' hint for an unknown key, if a near match exists."""
    matches = difflib.get_close_matches(key, sorted(known), n=1, cutoff=0.6)
    return f" (did you mean '{matches[0]}'?)" if matches else ""


def build_section(
    raw: Mapping[str, Any], path: str, cls: type[T], nested: frozenset[str] = frozenset()
) -> T:
    """Build a frozen dataclass from one TOML table, requiring exactly its fields.

    Args:
        raw: the table as loaded from TOML.
        path: dotted path of the table, used in error messages.
        cls: the dataclass to build. Every field is required.
        nested: keys of the table that are sub-tables handled elsewhere, and so must not be
            reported as unknown.

    Raises:
        ConfigError: on any missing key, unknown key, or value of the wrong type.
    """
    fields = {f.name: f for f in dataclasses.fields(cls)}
    present = set(raw.keys()) - nested
    # A field is optional only if the dataclass gives it a default. Every section below except
    # AgentOverride declares its fields without defaults, so everything is required.
    required = {
        name
        for name, f in fields.items()
        if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
    }

    if missing := sorted(required - present):
        raise ConfigError(f"{path}: missing required key(s) {missing}")
    if unknown := sorted(present - set(fields)):
        hints = "".join(f"\n  unknown key '{k}'{_suggest(k, set(fields))}" for k in unknown)
        raise ConfigError(f"{path}: {len(unknown)} unknown key(s){hints}")

    return cls(
        **{
            name: _coerce(raw[name], field, path)
            for name, field in fields.items()
            if name in present
        }
    )


@dataclass(frozen=True, slots=True)
class Geometry:
    """Physical radii used for collision costs, in metres."""

    drone_radius: float
    obstacle_radius: float
    gate_frame_radius: float


@dataclass(frozen=True, slots=True)
class SplineConfig:
    """Reference-path generation (see ``control.spline_planner.SplinePlanner``)."""

    t_total: float
    curvature_weight: float
    clearance: float
    lut_samples: int


@dataclass(frozen=True, slots=True)
class CostWeights:
    """Per-drone cost weights. Mirrors the terms returned by ``single_drone_terms``."""

    contour: float  # off-path (perpendicular) error
    lag: float  # ahead/behind-of-theta error along the path tangent
    progress: float  # reward for advancing theta; the "go faster" term
    z: float  # extra altitude penalty
    ang_vel: float
    ang_acc: float
    tilt: float
    thrust: float
    yaw: float
    obstacle: float  # per-hit penalty, shared by pillars and gate frames
    floor: float
    floor_z: float  # height below which the floor penalty applies


@dataclass(frozen=True, slots=True)
class OpponentConfig:
    """Opponent modelling: which predictor, how it is built, and the keep-out it induces."""

    model: str  # one of OPPONENT_MODELS
    pred_offset_tau: float  # decay (s) of the opponent's off-spline offset in spline_progress
    anchor_fwd: float  # forward slack (m) of the theta anchor search window
    anchor_reacquire_dist: float  # residual above which the anchor is re-acquired globally

    yields: bool  # model opponents as avoiding us too (only meaningful for model="mppi")
    drone_exp: float  # weight of the smooth exponential collision cost

    use_anisotropic: bool  # elongate the keep-out along the opponent's heading
    axial: float  # fore/aft keep-out half-length (m)
    lateral: float  # sideways keep-out half-width (m)
    core_radius: float  # hard isotropic core the ellipse may never admit closer than

    vel_ema: float  # EMA low-pass on the observed opponent velocity
    blend_v0: float  # iso<->aniso blend centre (m/s)
    blend_width: float  # blend width (m/s)

    use_behind_contour: bool  # relax contour cost when an opponent is ahead, to allow passing
    behind_radius: float
    contour_behind_scale: float

    downwash: float  # below-opponent wake penalty; the sim does NOT model downwash
    downwash_radius: float
    downwash_dz: float

    stale_inflate_rate: float  # keep-out inflation per second of tracking staleness
    stale_max: float  # staleness cap (s)


@dataclass(frozen=True, slots=True)
class AgentOverride:
    """Per-agent overrides for a ragged multi-agent setup.

    The one place defaults are allowed: an override table is partial by nature, and an absent
    entry means "use the shared value". Unknown keys are still rejected.
    """

    K: int | None = None
    t_total: float | None = None
    curvature_weight: float | None = None
    clearance: float | None = None


@dataclass(frozen=True, slots=True)
class MPPIConfig:
    """Everything the MPPI controller needs, validated."""

    # horizon
    N: int
    T: float
    # sampling
    n_samples: int
    K: int
    noise_sigma: float
    temperature: float
    elite_percentage: float
    beta: float
    alpha: float
    min_variance: float
    # progress channel (v_theta), the 5th sampled control
    v_theta_init: float
    v_theta_max: float
    v_theta_sigma: float
    # execution
    action_ema: float

    cost: CostWeights
    opponent: OpponentConfig
    spline: SplineConfig
    geometry: Geometry
    agents: tuple[AgentOverride, ...]

    # sourced from outside [controller.mppi]
    ctrl_freq: int
    drone_model: str

    @property
    def ctrl_dt(self) -> float:
        """Control period in seconds."""
        return 1.0 / self.ctrl_freq

    @property
    def dt(self) -> float:
        """Planning step of the rollout horizon, in seconds."""
        return self.T / self.N

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> MPPIConfig:
        """Build and validate from a loaded race config.

        Expects ``config["controller"]`` to be this drone's controller table (the multi-agent
        wrapper narrows the per-rank list before the base controller sees it).

        Raises:
            ConfigError: on any missing key, unknown key, or invalid value.
        """
        try:
            controller = config["controller"]
            raw = controller["mppi"]
        except KeyError as e:
            raise ConfigError(f"config is missing the {e} section") from e

        nested = frozenset({"cost", "opponent", "spline", "geometry", "agents"})
        for name in nested:
            if name not in raw:
                raise ConfigError(f"controller.mppi: missing required section '{name}'")

        agents = tuple(
            build_section(entry, f"controller.mppi.agents[{i}]", AgentOverride)
            for i, entry in enumerate(raw["agents"])
        )
        cfg = build_section(
            raw,
            "controller.mppi",
            _MPPICore,
            nested=nested,
        )
        opponent = build_section(raw["opponent"], "controller.mppi.opponent", OpponentConfig)
        if opponent.model not in OPPONENT_MODELS:
            raise ConfigError(
                f"controller.mppi.opponent.model: {opponent.model!r} must be one of "
                f"{list(OPPONENT_MODELS)}"
            )

        if "ctrl_freq" not in controller:
            raise ConfigError("controller: missing required key 'ctrl_freq'")
        if "sim" not in config or "drone_model" not in config["sim"]:
            raise ConfigError("sim: missing required key 'drone_model'")

        if cfg.n_samples % cfg.K:
            raise ConfigError(
                f"controller.mppi: n_samples ({cfg.n_samples}) must be divisible by K ({cfg.K})"
            )
        # N / T is used as an integer sim frequency. Checked on the quotient rather than with
        # `N % T`, which is unreliable on floats: 30 % 0.6 == 0.5999999999999979.
        rollout_freq = cfg.N / cfg.T
        if abs(rollout_freq - round(rollout_freq)) > 1e-9:
            raise ConfigError(
                f"controller.mppi: N ({cfg.N}) / T ({cfg.T}) = {rollout_freq} must be a whole "
                "number, since it is the rollout sim frequency"
            )

        return cls(
            **dataclasses.asdict(cfg),
            cost=build_section(raw["cost"], "controller.mppi.cost", CostWeights),
            opponent=opponent,
            spline=build_section(raw["spline"], "controller.mppi.spline", SplineConfig),
            geometry=build_section(raw["geometry"], "controller.mppi.geometry", Geometry),
            agents=agents,
            ctrl_freq=int(controller["ctrl_freq"]),
            drone_model=str(config["sim"]["drone_model"]),
        )


@dataclass(frozen=True, slots=True)
class _MPPICore:
    """The scalar fields of [controller.mppi], split out so one strict build call covers them."""

    N: int
    T: float
    n_samples: int
    K: int
    noise_sigma: float
    temperature: float
    elite_percentage: float
    beta: float
    alpha: float
    min_variance: float
    v_theta_init: float
    v_theta_max: float
    v_theta_sigma: float
    action_ema: float
