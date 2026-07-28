"""Tests for the strict MPPI config layer.

The point of the layer is that it fails loudly, so most of these are negative tests. The last
one parses every real config that names an MPPI controller, which is what keeps the seven TOMLs
from drifting apart as keys are added.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import toml

from lsy_drone_racing.control.mppi import ConfigError, MPPIConfig

CONFIG_DIR = Path(__file__).parents[3] / "config"


def mppi_configs() -> list[Path]:
    """Every race config in config/ that loads an MPPI controller.

    Discovery is deliberately dynamic: a new MPPI config is covered the moment it is added,
    which is what stops the shipped TOMLs drifting apart as keys change.

    ``config_MPPI.toml`` and ``config_MPPI_original.toml`` carry an ``[controller.mppi]`` table
    but no ``[env]`` section and no controller ``file``, so the race framework cannot load them
    at all. They are leftovers from the crazyflow_experiments framework; the ``file`` check
    below skips them.
    """
    found = []
    for path in sorted(CONFIG_DIR.glob("*.toml")):
        controllers = toml.load(path).get("controller")
        entries = controllers if isinstance(controllers, list) else [controllers or {}]
        if any("mppi" in (e or {}) and "file" in (e or {}) for e in entries):
            found.append(path)
    return found


def narrow(raw: dict) -> dict:
    """Narrow a race config to the MPPI controller, the way the multi-agent wrapper does."""
    controller = raw["controller"]
    if isinstance(controller, list):
        controller = next(c for c in controller if "mppi" in c)
    return dict(raw) | {"controller": controller}


@pytest.fixture
def raw() -> dict:
    """A known-good config, narrowed and ready to mutate."""
    return narrow(toml.load(CONFIG_DIR / "multi_level0.toml"))


@pytest.mark.unit
@pytest.mark.parametrize("path", mppi_configs(), ids=lambda p: p.name)
def test_every_mppi_config_parses(path: Path):
    """Every shipped MPPI config must satisfy the strict schema."""
    cfg = MPPIConfig.from_config(narrow(toml.load(path)))
    assert cfg.N > 0
    assert cfg.n_samples % cfg.K == 0


@pytest.mark.unit
def test_missing_key_is_rejected(raw: dict):
    """A dropped key must raise at construction, not silently fall back to a default."""
    del raw["controller"]["mppi"]["cost"]["contour"]
    with pytest.raises(ConfigError, match="missing required key.*contour"):
        MPPIConfig.from_config(raw)


@pytest.mark.unit
def test_unknown_key_is_rejected_with_a_suggestion(raw: dict):
    """A misspelled key must be caught, and near misses should be pointed out."""
    raw["controller"]["mppi"]["cost"]["progres"] = 1.0
    with pytest.raises(ConfigError) as excinfo:
        MPPIConfig.from_config(raw)
    assert "progres" in str(excinfo.value)
    assert "did you mean 'progress'" in str(excinfo.value)


@pytest.mark.unit
def test_dead_weights_are_rejected(raw: dict):
    """The four weights no code reads any more must not silently reappear in a config."""
    for dead in ("pos", "vel", "obstacle_exp", "opp_drone"):
        cfg = copy.deepcopy(raw)
        cfg["controller"]["mppi"]["cost"][dead] = 1.0
        with pytest.raises(ConfigError, match="unknown key"):
            MPPIConfig.from_config(cfg)


@pytest.mark.unit
def test_bad_opponent_model_is_rejected(raw: dict):
    """The model selector is an enum, and a plausible-looking typo must not pass."""
    raw["controller"]["mppi"]["opponent"]["model"] = "spline"
    with pytest.raises(ConfigError, match="must be one of"):
        MPPIConfig.from_config(raw)


@pytest.mark.unit
def test_non_bool_flag_is_rejected(raw: dict):
    """A feature flag given as 1 rather than true is exactly the silence this layer removes."""
    raw["controller"]["mppi"]["opponent"]["use_anisotropic"] = 1
    with pytest.raises(ConfigError, match="expected true/false"):
        MPPIConfig.from_config(raw)


@pytest.mark.unit
def test_n_samples_must_divide_by_k(raw: dict):
    """Worlds are paired 1:1 with modes, so an indivisible sample count is a hard error."""
    raw["controller"]["mppi"]["n_samples"] = 999
    with pytest.raises(ConfigError, match="divisible"):
        MPPIConfig.from_config(raw)


@pytest.mark.unit
def test_ints_accepted_for_float_fields(raw: dict):
    """TOML cannot distinguish 1 from 1.0, so a float field must accept an int."""
    raw["controller"]["mppi"]["cost"]["contour"] = 20
    assert MPPIConfig.from_config(raw).cost.contour == pytest.approx(20.0)


@pytest.mark.unit
def test_agent_overrides_are_partial_but_still_checked(raw: dict):
    """Override tables may be partial, but a typo inside one must still be caught."""
    raw["controller"]["mppi"]["agents"] = [{"K": 3}]
    assert MPPIConfig.from_config(raw).agents[0].K == 3
    assert MPPIConfig.from_config(raw).agents[0].clearance is None

    raw["controller"]["mppi"]["agents"] = [{"KK": 3}]
    with pytest.raises(ConfigError, match="unknown key"):
        MPPIConfig.from_config(raw)


@pytest.mark.unit
def test_derived_timing(raw: dict):
    """Timing is derived from N/T and ctrl_freq, so it cannot drift out of sync."""
    cfg = MPPIConfig.from_config(raw)
    assert cfg.dt == pytest.approx(cfg.T / cfg.N)
    assert cfg.ctrl_dt == pytest.approx(1.0 / cfg.ctrl_freq)
