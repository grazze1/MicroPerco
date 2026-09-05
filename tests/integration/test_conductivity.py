# SPDX-License-Identifier: Apache-2.0
"""Geometry, transverse PBC, sampling, and CLI transport workflows."""

import json
import math
from itertools import product
from pathlib import Path

import numpy as np
import pytest

from microperco import (
    ConstantConductanceModel,
    Cylinder,
    Domain,
    PopulationSpec,
    Sphere,
    SphereSpec,
    TunnelingConductanceModel,
    analyze_conductivity,
    analyze_directional_conductivity,
    estimate_conductivity,
    generate_microstructure,
)
from microperco.cli import main
from microperco.exceptions import ConfigurationError
from microperco.io import dumps_json, loads_config


def chain() -> tuple[Sphere, ...]:
    return tuple(Sphere((x, 0, 0), 1) for x in (-4, -2, 0, 2, 4))


def test_chain_includes_two_electrode_resistors_and_geometry_factor() -> None:
    result = analyze_conductivity(chain(), Domain((10, 4, 6), False), ConstantConductanceModel(2))
    assert result.percolates
    assert result.effective_conductance == pytest.approx(2 / 6)
    assert result.effective_conductivity == pytest.approx((2 / 6) * 10 / 24)
    assert result.solution.node_voltages[:5] == pytest.approx((5 / 6, 4 / 6, 3 / 6, 2 / 6, 1 / 6))
    assert len(result.geometry.junctions) == len(result.geometry.network.edges) == 6


def test_anisotropic_chain_uses_one_realization_for_three_directions() -> None:
    result = analyze_directional_conductivity(chain(), Domain(10, True))
    assert result.sigma_x == pytest.approx(1 / 60)
    assert result.sigma_y == result.sigma_z == 0
    assert tuple(item.axis for item in result.results) == ("x", "y", "z")
    assert result.results[0].geometry.domain.periodic == (False, True, True)


def test_exponential_particle_gap_with_independent_electrode_contacts() -> None:
    particles = (Sphere((-1.25, 0, 0), 1), Sphere((1.25, 0, 0), 1))
    result = analyze_conductivity(
        particles,
        Domain((4.5, 3, 3), False),
        TunnelingConductanceModel(2, 0.5, 0.5),
        electrode_model=ConstantConductanceModel(10),
    )
    expected = 1 / (0.1 + math.exp(2) / 2 + 0.1)
    assert result.effective_conductance == pytest.approx(expected)
    assert [j.gap for j in result.geometry.junctions] == [0.5, 0, 0]


def test_cylinder_crossing_box_uses_flat_finite_electrodes() -> None:
    particle = Cylinder((0, 0, 0), (1, 0, 0), 4, 0.2)
    result = analyze_conductivity((particle,), Domain((4, 2, 2), False))
    assert result.effective_conductivity == pytest.approx(0.5)
    outside = particle.translated((0, 4, 0))
    assert not analyze_conductivity((outside,), Domain((4, 2, 2), False)).percolates


def test_equal_parent_fragments_are_equipotential_and_electrodes_not_duplicated() -> None:
    particles = (
        Sphere((-4, 0, 0), 1, parent_id="a"),
        Sphere((-4, 0, 0), 1, parent_id="a"),
        Sphere((4, 0, 0), 1, parent_id="a"),
    )
    result = analyze_conductivity(particles, Domain(10, False))
    assert result.geometry.particle_nodes == (0, 0, 0)
    assert len(result.geometry.network.edges) == 2
    assert result.effective_conductance == pytest.approx(0.5)


@pytest.mark.parametrize("periodic", list(product((False, True), repeat=3)))
def test_search_backends_agree_for_all_periodic_axes(periodic: tuple[bool, ...]) -> None:
    domain = Domain(4, periodic)
    particles = generate_microstructure(domain, [PopulationSpec(SphereSpec(0.8), 15)], seed=7)
    model = TunnelingConductanceModel(1, 0.3, 0.6)
    for axis in "xyz":
        optimized = analyze_conductivity(particles, domain, model, axis=axis)
        reference = analyze_conductivity(
            particles, domain, model, axis=axis, neighbor_backend="bruteforce"
        )
        assert optimized.geometry.network == reference.geometry.network
        assert optimized.geometry.junctions == reference.geometry.junctions
        assert optimized.effective_conductivity == pytest.approx(reference.effective_conductivity)


def test_transverse_image_junctions_act_in_parallel() -> None:
    particles = (Sphere((-0.75, -0.5, 0), 0.75), Sphere((0.75, 0.5, 0), 0.75))
    model = ConstantConductanceModel(1, 0.31)
    electrodes = ConstantConductanceModel(1)
    periodic = analyze_conductivity(
        particles, Domain((3, 2, 4), (False, True, False)), model, electrode_model=electrodes
    )
    opened = analyze_conductivity(
        particles, Domain((3, 2, 4), False), model, electrode_model=electrodes
    )
    assert len([j for j in periodic.geometry.junctions if j.kind == "particle"]) == 2
    assert periodic.effective_conductance == pytest.approx(1 / 2.5)
    assert opened.effective_conductance == pytest.approx(1 / 3)
    shifted = (particles[0], particles[1].translated((0, 10, 0)))
    translated = analyze_conductivity(
        shifted, Domain((3, 2, 4), (False, True, False)), model, electrode_model=electrodes
    )
    assert translated.effective_conductance == pytest.approx(periodic.effective_conductance)


def test_empty_and_isolated_systems_have_zero_conductivity_and_json_null_potentials() -> None:
    empty = analyze_conductivity((), Domain(10, False))
    assert empty.effective_conductivity == 0
    isolated = analyze_conductivity((Sphere((0, 0, 0), 1),), Domain(10, False))
    payload = json.loads(dumps_json(isolated))
    assert payload["solution"]["node_voltages"] == [None, 1, 0]


def test_seeded_monte_carlo_and_axis_order_reproducibility() -> None:
    domain = Domain(3, False)
    populations = (PopulationSpec(SphereSpec(1), 6),)
    model = ConstantConductanceModel(1, 0.2)
    first = estimate_conductivity(domain, populations, model, trials=4, seed=12)
    second = estimate_conductivity(domain, populations, model, trials=4, seed=12, axes=("z", "x"))
    assert first.estimates[0] == second.estimates[1]
    assert first.estimates[2] == second.estimates[0]
    sample = first.estimates[0]
    assert sample.mean == pytest.approx(np.mean(sample.samples))
    assert sample.standard_error == pytest.approx(np.std(sample.samples, ddof=1) / 2)
    root = np.random.SeedSequence(12)
    assert estimate_conductivity(domain, populations, model, trials=4, seed=root).estimates == (
        first.estimates
    )
    assert root.n_children_spawned == 0


def test_random_seed_provenance_can_replay_and_single_trial_has_no_standard_error() -> None:
    domain = Domain(3, False)
    populations = (PopulationSpec(SphereSpec(1), 6),)
    first = estimate_conductivity(domain, populations, trials=1, seed=None)
    assert first.seed is not None
    state = first.seed
    replay_seed = np.random.SeedSequence(
        state.entropy, spawn_key=state.spawn_key, pool_size=state.pool_size
    )
    assert estimate_conductivity(domain, populations, trials=1, seed=replay_seed) == first
    assert all(item.standard_error is None for item in first.estimates)


@pytest.mark.parametrize("axes", [(), ("x", "x"), ("bad",), (True,)])
def test_invalid_axes_fail_early(axes: tuple) -> None:
    with pytest.raises(ValueError):
        estimate_conductivity(Domain(3, False), [PopulationSpec(SphereSpec(1), 1)], axes=axes)


CONFIG = """schema_version: 1
domain:
  size: [3, 3, 3]
  periodic: true
particles:
  - name: beads
    shape: sphere
    radius: 1
    count: 4
simulation:
  trials: 2
  seed: 12
conductivity:
  axes: [x, y, z]
  model:
    type: tunneling
    contact_conductance: 1
    decay_length: 0.2
    cutoff: 0.6
"""


def test_conductivity_cli_writes_reproducible_json(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(CONFIG)
    output = tmp_path / "result.json"
    assert main(["conductivity", str(config), "--output", str(output)]) == 0
    first = output.read_text()
    payload = json.loads(first)
    assert payload["trials"] == 2
    assert payload["seed"] == 12
    assert [e["axis"] for e in payload["estimates"]] == ["x", "y", "z"]
    assert main(["conductivity", str(config), "--output", str(output)]) == 0
    assert output.read_text() == first


@pytest.mark.parametrize(
    "old,new",
    [
        ("axes: [x, y, z]", "axes: [x, x]"),
        ("axes: [x, y, z]", "axes: [x, invalid]"),
        ("axes: [x, y, z]", "axes: []"),
        ("axes: [x, y, z]", "applied_voltage: 0"),
        ("type: tunneling", "type: invalid"),
        ("type: tunneling", "type: constant"),
        ("decay_length: 0.2", "decay_length: -1"),
        ("cutoff: 0.6", "unknown: 0.6"),
        ("cutoff: 0.6", "cutoff: true"),
        ("cutoff: 0.6", "cutoff: .inf"),
        ("cutoff: 0.6", "cutoff: -1"),
    ],
)
def test_transport_config_is_strict(old: str, new: str) -> None:
    with pytest.raises(ConfigurationError):
        loads_config(CONFIG.replace(old, new), operation="conductivity")


def test_cli_requires_transport_section_and_finite_electrode_mode() -> None:
    with pytest.raises(ConfigurationError, match="section is required"):
        loads_config(CONFIG.split("conductivity:")[0], operation="conductivity")
    for option in ("mode: periodic_wrap", "wrapped_parent: true"):
        with pytest.raises(ConfigurationError, match="face_to_face"):
            loads_config(CONFIG + f"percolation:\n  {option}\n", operation="conductivity")
