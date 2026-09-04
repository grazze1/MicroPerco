# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import pytest

from microperco import Cylinder, Domain, Sphere, ThresholdContactModel, analyze_percolation


def _axis_chain(axis: int) -> tuple[Sphere, ...]:
    particles = []
    for index, coordinate in enumerate((-4.0, -2.0, 0.0, 2.0, 4.0)):
        center = np.zeros(3)
        center[axis] = coordinate
        particles.append(Sphere(center, 1.0, index))
    return tuple(particles)


@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("solver", ["union_find", "bfs"])
@pytest.mark.parametrize("search", ["cell_list", "bruteforce"])
def test_known_chain_percolates_in_each_direction(axis: int, solver: str, search: str) -> None:
    result = analyze_percolation(
        _axis_chain(axis),
        Domain((10.0, 10.0, 10.0), False),
        axis=axis,
        solver=solver,
        search=search,
    )
    assert result.percolates
    assert result.spanning_path == (0, 1, 2, 3, 4)
    assert result.component_count == 1
    assert result.edge_count == 4


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_broken_chain_does_not_percolate(axis: str) -> None:
    particles = list(_axis_chain("xyz".index(axis)))
    particles.pop(2)
    result = analyze_percolation(particles, Domain(10.0, False), axis=axis)
    assert not result.percolates
    assert result.spanning_path == ()
    assert result.component_count == 2


def test_empty_system_has_zero_components() -> None:
    result = analyze_percolation((), Domain(10.0, False))
    assert not result.percolates
    assert result.particle_count == 0
    assert result.component_count == 0


def test_finite_electrode_prevents_false_plane_contact() -> None:
    particles = (Sphere((-4.8, 8.0, 0.0), 0.5), Sphere((4.8, 8.0, 0.0), 0.5))
    result = analyze_percolation(
        particles,
        Domain(10.0, False),
        ThresholdContactModel(9.0),
        axis="x",
    )
    assert result.lower_electrode_particles == (0,)
    assert result.upper_electrode_particles == (1,)
    assert result.percolates
    strict = analyze_percolation(particles, Domain(10.0, False), axis="x")
    assert strict.lower_electrode_particles == ()
    assert strict.upper_electrode_particles == ()


@pytest.mark.parametrize("search", ["bruteforce", "cell_list"])
def test_unrelated_long_domain_axis_does_not_create_false_contacts(search: str) -> None:
    particles = tuple(
        Sphere((0.0, coordinate, 0.0), 1.0, index)
        for index, coordinate in enumerate((-4.0, -1.0, 2.0, 4.0))
    )
    result = analyze_percolation(
        particles,
        Domain((1.0e12, 10.0, 10.0), False),
        axis="y",
        search=search,
    )
    assert not result.percolates
    assert [(edge.i, edge.j) for edge in result.contact_edges] == [(2, 3)]


@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("solver", ["union_find", "bfs"])
def test_periodic_winding_cycle(axis: int, solver: str) -> None:
    positions = (-4.0, 0.0, 4.0)
    particles = []
    for index, coordinate in enumerate(positions):
        center = np.zeros(3)
        center[axis] = coordinate
        particles.append(Sphere(center, 1.0, index))
    periodic = [False, False, False]
    periodic[axis] = True
    result = analyze_percolation(
        particles,
        Domain(10.0, periodic),
        ThresholdContactModel(2.0),
        axis=axis,
        mode="periodic_wrap",
        solver=solver,
    )
    assert result.percolates
    assert result.winding is not None
    assert result.winding[axis] == 1


@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("solver", ["union_find", "bfs"])
@pytest.mark.parametrize("search", ["cell_list", "bruteforce"])
def test_single_cylinder_can_wrap_through_its_physical_images(
    axis: int, solver: str, search: str
) -> None:
    direction = np.zeros(3)
    direction[axis] = 1.0
    periodic = [False, False, False]
    periodic[axis] = True
    result = analyze_percolation(
        (Cylinder(np.zeros(3), direction, 10.0, 0.2),),
        Domain(10.0, periodic),
        axis=axis,
        mode="periodic_wrap",
        solver=solver,
        search=search,
    )
    assert result.percolates
    assert result.contact_edges == ()
    assert result.winding is not None and result.winding[axis] == 1
    assert {edge.lattice_shift[axis] for edge in result.topology_edges} == {1}
    assert {edge.kind for edge in result.topology_edges} == {"periodic_self_image"}


@pytest.mark.parametrize("radius, expected", [(4.9, False), (5.0, True), (5.1, True)])
def test_single_sphere_wrap_requires_physical_self_image_intersection(
    radius: float, expected: bool
) -> None:
    result = analyze_percolation(
        (Sphere((0.0, 0.0, 0.0), radius),),
        Domain(10.0, (True, False, False)),
        axis="x",
        mode="periodic_wrap",
    )
    assert result.percolates is expected


def test_contact_threshold_does_not_create_single_particle_self_contact() -> None:
    result = analyze_percolation(
        (Sphere((0.0, 0.0, 0.0), 4.9),),
        Domain(10.0, (True, False, False)),
        ThresholdContactModel(0.25),
        axis="x",
        mode="periodic_wrap",
    )
    assert not result.percolates
    assert result.topology_edges == ()


def test_long_single_cylinder_records_all_intersecting_image_shifts() -> None:
    result = analyze_percolation(
        (Cylinder((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 32.0, 0.2),),
        Domain(10.0, (True, False, False)),
        axis="x",
        mode="periodic_wrap",
    )
    assert result.percolates
    assert {edge.lattice_shift for edge in result.topology_edges} == {
        (1, 0, 0),
        (2, 0, 0),
        (3, 0, 0),
    }


def test_canonical_same_parent_seam_fragments_do_not_form_a_false_loop() -> None:
    particles = (
        Sphere((-4.8, 0.0, 0.0), 0.5, 0, "parent"),
        Sphere((4.8, 0.0, 0.0), 0.5, 1, "parent"),
    )
    result = analyze_percolation(
        particles,
        Domain(10.0, (True, False, False)),
        mode="periodic_wrap",
    )
    assert not result.percolates
    assert result.contact_edges == ()
    assert result.topology_edges == ()


@pytest.mark.parametrize("gauge", [0, 7, -10**30])
def test_explicit_fragment_offsets_are_lattice_gauge_invariant(gauge: int) -> None:
    particles = (
        Sphere((-4.8, 0.0, 0.0), 0.5, 0, "parent", (gauge, 0, 0)),
        Sphere((4.8, 0.0, 0.0), 0.5, 1, "parent", (gauge - 1, 0, 0)),
    )
    result = analyze_percolation(
        particles,
        Domain(10.0, (True, False, False)),
        mode="periodic_wrap",
    )
    assert not result.percolates
    assert result.topology_edges == ()


def test_zero_raw_shift_can_encode_nonzero_parent_image_winding() -> None:
    particles = (
        Sphere((0.0, 0.0, 0.0), 1.0, 0, "parent", (0, 0, 0)),
        Sphere((0.0, 0.0, 0.0), 1.0, 1, "parent", (1, 0, 0)),
    )
    result = analyze_percolation(
        particles,
        Domain(10.0, (True, False, False)),
        mode="periodic_wrap",
    )
    assert result.percolates
    assert result.winding == (1, 0, 0)
    assert len(result.topology_edges) == 1
    assert result.topology_edges[0].lattice_shift == (1, 0, 0)


def test_periodic_parent_offsets_must_be_complete_and_axis_valid() -> None:
    domain = Domain(10.0, (True, False, False))
    mixed = (
        Sphere((-1.0, 0.0, 0.0), 0.5, 0, "parent", (0, 0, 0)),
        Sphere((1.0, 0.0, 0.0), 0.5, 1, "parent"),
    )
    with pytest.raises(ValueError, match="all fragments"):
        analyze_percolation(mixed, domain, mode="periodic_wrap")
    invalid_axis = (Sphere((0.0, 0.0, 0.0), 0.5, image_offset=(0, 1, 0)),)
    with pytest.raises(ValueError, match="non-periodic"):
        analyze_percolation(invalid_axis, domain, mode="periodic_wrap")


def test_periodic_wrap_requires_periodic_analysis_axis() -> None:
    with pytest.raises(ValueError, match="requires"):
        analyze_percolation((), Domain(10.0, False), mode="periodic_wrap")


def test_same_parent_fragments_are_always_one_logical_component() -> None:
    particles = (
        Sphere((-3.0, 0.0, 0.0), 0.5, 0, "a"),
        Sphere((3.0, 0.0, 0.0), 0.5, 1, "a"),
    )
    domain = Domain(10.0, False)
    default = analyze_percolation(particles, domain)
    historical_face_mode = analyze_percolation(particles, domain, wrapped_parent=True)
    assert default.component_count == 1
    assert historical_face_mode.component_count == 1


def test_same_parent_fragments_bridge_faces_without_a_self_contact_edge() -> None:
    particles = (
        Sphere((-4.8, 0.0, 0.0), 0.2, 0, "parent"),
        Sphere((4.8, 0.0, 0.0), 0.2, 1, "parent"),
    )
    result = analyze_percolation(particles, Domain(10.0, False), wrapped_parent=False)
    assert result.percolates
    assert result.lower_electrode_particles == (0,)
    assert result.upper_electrode_particles == (1,)
    assert result.contact_edges == ()


def test_solver_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="solver"):
        analyze_percolation((), Domain(10.0, False), solver="dijkstra")
