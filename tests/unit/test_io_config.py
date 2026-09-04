# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math

import pytest

from microperco.exceptions import ConfigurationError
from microperco.io import dumps_json, loads_config

BASE_CONFIG = """
schema_version: 1
domain:
  size: [10, 12, 14]
  periodic: [true, false, true]
particles:
  - name: fiber
    shape: cylinder
    radius: 0.2
    length: 2.0
    count: 5
    material:
      name: carbon
      cost_per_volume: 3.0
  - name: bead
    shape: sphere
    radius: 0.3
    count: 2
contact:
  type: threshold
  threshold: 0.1
percolation:
  axis: x
  mode: face_to_face
simulation:
  trials: 8
  seed: [4, 5]
  confidence: 0.95
  neighbor_backend: cell_list
critical:
  population: fiber
  counts: [0, 2, 4]
  target_probability: 0.8
  search_trials: 5
  certification_trials: 10
  strategy: pava
optimization:
  count_bounds:
    fiber: [0, 2]
    bead: [0, 1]
  target_probability: 0.8
  search_trials: 4
  certification_trials: 8
  max_candidates: 20
benchmark:
  population: bead
  particle_counts: [2, 4]
  repeats: 1
  warmup: 0
  seed: 7
"""


def test_loads_complete_config() -> None:
    config = loads_config(BASE_CONFIG)
    assert config.schema_version == 1
    assert config.domain.size == (10.0, 12.0, 14.0)
    assert config.domain.periodic == (True, False, True)
    assert config.particles[0].to_spec().cost > 0.0
    assert config.populations()[1].count == 2
    assert config.simulation.seed == (4, 5)
    config.validate_for("critical")
    config.validate_for("optimize")
    config.validate_for("benchmark")


@pytest.mark.parametrize(
    "replacement, message",
    [
        ("schema_version: 1", "domain and particles"),
        (BASE_CONFIG.replace("schema_version: 1", "schema_version: 2"), "schema_version"),
        (BASE_CONFIG + "mystery: true\n", "unknown key"),
        (BASE_CONFIG.replace("radius: 0.2", "radius: -0.2"), "positive"),
        (BASE_CONFIG.replace("counts: [0, 2, 4]", "counts: [0, 4, 2]"), "increasing"),
        (BASE_CONFIG.replace("confidence: 0.95", "confidence: 1.0"), "confidence"),
        (BASE_CONFIG.replace("name: bead", "name: fiber"), "unique"),
    ],
)
def test_config_rejects_invalid_documents(replacement: str, message: str) -> None:
    with pytest.raises(ConfigurationError, match=message):
        loads_config(replacement)


@pytest.mark.parametrize(
    "document",
    [
        BASE_CONFIG + "schema_version: 1\n",
        BASE_CONFIG.replace("  trials: 8\n", "  trials: 8\n  trials: 9\n"),
        BASE_CONFIG.replace("    radius: 0.2\n", "    radius: 0.2\n    radius: 0.3\n"),
    ],
)
def test_config_rejects_duplicate_mapping_keys_at_every_depth(document: str) -> None:
    with pytest.raises(ConfigurationError, match="duplicate mapping key"):
        loads_config(document)


@pytest.mark.parametrize("version", ["true", "1.0"])
def test_schema_version_requires_an_actual_integer(version: str) -> None:
    document = BASE_CONFIG.replace("schema_version: 1", f"schema_version: {version}")
    with pytest.raises(ConfigurationError, match="schema_version must be 1"):
        loads_config(document)


@pytest.mark.parametrize("value", ["9" * 400, ".inf", "-.inf", ".nan"])
def test_real_fields_reject_values_that_are_not_finite(value: str) -> None:
    document = BASE_CONFIG.replace("radius: 0.2", f"radius: {value}")
    with pytest.raises(ConfigurationError, match="finite real number"):
        loads_config(document)


def test_yaml_numeric_conversion_failure_is_a_configuration_error() -> None:
    document = BASE_CONFIG.replace("radius: 0.2", f"radius: {'9' * 5000}")
    with pytest.raises(ConfigurationError, match="configuration is not valid YAML"):
        loads_config(document)


def test_periodic_wrap_configuration_requires_axis_periodicity() -> None:
    invalid = BASE_CONFIG.replace("axis: x\n  mode: face_to_face", "axis: y\n  mode: periodic_wrap")
    with pytest.raises(ConfigurationError, match="periodic analysis axis"):
        loads_config(invalid, operation="simulate")


def test_operation_requires_matching_section() -> None:
    document = BASE_CONFIG.split("critical:", maxsplit=1)[0]
    with pytest.raises(ConfigurationError, match="critical section"):
        loads_config(document, operation="critical")


def test_json_serializer_is_deterministic_and_rejects_nonfinite() -> None:
    first = dumps_json({"b": 2, "a": (1, 3)})
    second = dumps_json({"a": [1, 3], "b": 2})
    assert first == second
    assert first.endswith("\n")
    with pytest.raises(ConfigurationError, match="non-JSON"):
        dumps_json({"value": math.nan})
