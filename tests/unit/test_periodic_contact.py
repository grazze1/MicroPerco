# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np
import pytest

from microperco import Cylinder, Domain, Sphere, ThresholdContactModel
from microperco.contact import bruteforce_contacts, cell_list_contacts, find_contacts
from microperco.exceptions import GeometryError
from microperco.geometry import candidate_lattice_shifts, face_gaps, periodic_distance

PERIODIC_FLAGS = tuple(product((False, True), repeat=3))


@pytest.mark.parametrize("periodic", PERIODIC_FLAGS)
def test_cell_list_matches_bruteforce_for_every_pbc_combination(
    periodic: tuple[bool, bool, bool],
) -> None:
    seed = 8142 + sum(1 << index for index, value in enumerate(periodic) if value)
    rng = np.random.default_rng(seed)
    particles = []
    for index in range(18):
        center = rng.uniform(-4.8, 4.8, 3)
        if index % 3:
            particles.append(Sphere(center, 0.32 + 0.04 * (index % 2), index))
        else:
            particles.append(Cylinder(center, rng.normal(size=3), 1.1, 0.24, index))
    domain = Domain(10.0, periodic)
    model = ThresholdContactModel(0.18)
    brute = bruteforce_contacts(particles, domain, model)
    fast = cell_list_contacts(particles, domain, model)
    brute_signature = [(edge.i, edge.j, edge.lattice_shift) for edge in brute.edges]
    fast_signature = [(edge.i, edge.j, edge.lattice_shift) for edge in fast.edges]
    assert fast_signature == brute_signature
    np.testing.assert_allclose(
        [edge.distance for edge in fast.edges],
        [edge.distance for edge in brute.edges],
        rtol=0.0,
        atol=1.0e-10,
    )


@pytest.mark.parametrize("axis", range(3))
def test_periodic_distance_wraps_each_axis(axis: int) -> None:
    periodic = [False, False, False]
    periodic[axis] = True
    domain = Domain(10.0, periodic)
    lower = np.zeros(3)
    upper = np.zeros(3)
    lower[axis] = -4.7
    upper[axis] = 4.7
    result = periodic_distance(Sphere(lower, 0.4), Sphere(upper, 0.4), domain)
    assert result.distance == 0.0
    expected = [0, 0, 0]
    expected[axis] = -1
    assert result.lattice_shift == tuple(expected)


def test_corner_crossing_uses_multi_axis_image() -> None:
    domain = Domain(10.0, True)
    first = Sphere((-4.8, -4.8, -4.8), 0.5)
    second = Sphere((4.8, 4.8, 4.8), 0.5)
    result = periodic_distance(first, second, domain)
    assert result.distance == 0.0
    assert result.lattice_shift == (-1, -1, -1)


def test_candidate_images_include_multiple_relevant_periods_for_long_cylinders() -> None:
    domain = Domain((2.0, 8.0, 8.0), (True, False, False))
    first = Cylinder((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), 7.0, 0.2)
    second = Cylinder((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), 7.0, 0.2)
    shifts = candidate_lattice_shifts(first, second, domain, 2.0)
    assert {shift[0] for shift in shifts} >= {-3, -2, -1, 0, 1, 2, 3}


@pytest.mark.parametrize("method", ["bruteforce", "cell_list"])
def test_near_axis_long_cylinder_contact_is_not_pruned(method: str) -> None:
    domain = Domain((10_000.0, 10.0, 10.0), (True, False, False))
    particles = (
        Cylinder((0.0, 0.0, 0.0), (1.0, 1.0e-8, 0.0), 20_000.0, 0.5),
        Sphere((0.0, 1.0, 0.0), 0.5),
    )
    result = find_contacts(particles, domain, method=method)
    assert {(edge.i, edge.j) for edge in result.edges} == {(0, 1)}


def test_bruteforce_and_cell_list_share_long_particle_tolerance_padding() -> None:
    domain = Domain((10_000.0, 10.0, 10.0), (True, False, False))
    particles = (
        Cylinder((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 20_000.0, 1.0),
        Cylinder((0.0, 2.0 + 5.0e-9, 0.0), (1.0, 0.0, 0.0), 20_000.0, 1.0),
    )
    brute = find_contacts(particles, domain, method="bruteforce")
    fast = find_contacts(particles, domain, method="cell_list")
    assert [(edge.i, edge.j, edge.lattice_shift) for edge in fast.edges] == [
        (edge.i, edge.j, edge.lattice_shift) for edge in brute.edges
    ]
    assert len(brute.edges) == 5


def test_periodic_distance_tie_is_not_biased_by_unrelated_domain_scale() -> None:
    domain = Domain((10.0, 1.0e15, 10.0), (True, False, False))
    result = periodic_distance(Sphere((0.0, 0.0, 0.0), 1.0), Sphere((4.0, 0.0, 0.0), 1.0), domain)
    assert result.distance == pytest.approx(2.0)
    assert result.lattice_shift == (0, 0, 0)


def test_contact_search_is_invariant_under_large_integer_lattice_gauge() -> None:
    domain = Domain((1.0e-8, 1.0, 1.0), (True, False, False))
    first = Sphere((0.0, 0.0, 0.0), 2.0e-9)
    second = Sphere((0.0, 0.0, 0.0), 2.0e-9)
    gauge = 1_000_000_000_000_000
    shifted = second.translated(domain.lattice_vector((gauge, 0, 0)))

    base_distance = periodic_distance(first, second, domain)
    shifted_distance = periodic_distance(first, shifted, domain)
    assert shifted_distance.distance == base_distance.distance
    assert shifted_distance.lattice_shift == (
        base_distance.lattice_shift[0] - gauge,
        0,
        0,
    )

    for backend in (bruteforce_contacts, cell_list_contacts):
        base = backend((first, second), domain)
        translated = backend((first, shifted), domain)
        assert translated.candidate_pairs == base.candidate_pairs == 1
        assert [edge.distance for edge in translated.edges] == [
            edge.distance for edge in base.edges
        ]
        assert [edge.lattice_shift for edge in translated.edges] == [
            (edge.lattice_shift[0] - gauge, 0, 0) for edge in base.edges
        ]


def test_reported_shift_beyond_int64_can_be_reapplied() -> None:
    domain = Domain((1.0, 10_000.0, 10_000.0), (True, False, False))
    first = Sphere((0.0, 0.0, 0.0), 2048.0)
    second = Sphere((float(2**63 - 1024), 0.0, 0.0), 2048.0)
    result = periodic_distance(first, second, domain)
    assert abs(result.lattice_shift[0]) > 2**63
    assert np.all(np.isfinite(domain.lattice_vector(result.lattice_shift)))
    assert ThresholdContactModel().in_contact(first, second, domain)


def test_periodic_contact_does_not_flip_from_large_gauge_rounding() -> None:
    length = 0.04896433081287396
    domain = Domain((length, 1.0, 1.0), (True, False, False))
    first = Sphere((0.0, 0.0, 0.0), 0.001)
    local = Sphere((0.01904142015551394, 0.0, 0.0), 0.001)
    shifted = Sphere((4_412_295_474_029.962, 0.0, 0.0), 0.001)
    local_result = periodic_distance(first, local, domain)
    shifted_result = periodic_distance(first, shifted, domain)
    assert shifted_result.distance == pytest.approx(local_result.distance, abs=1.0e-18)
    assert shifted_result.distance == pytest.approx(0.017041420155513937, abs=1.0e-18)
    model = ThresholdContactModel(0.0168)
    assert not model.in_contact(first, local, domain)
    assert not model.in_contact(first, shifted, domain)


def test_candidate_image_enumeration_fails_before_pathological_allocation() -> None:
    domain = Domain(10.0, (True, False, False))
    huge = Cylinder((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 1.0e8, 0.1)
    with pytest.raises(GeometryError, match="safety limit"):
        candidate_lattice_shifts(huge, huge, domain, 0.0)


def test_candidate_image_limit_handles_counts_beyond_platform_range_length() -> None:
    domain = Domain((1.0e-6, 1.0, 1.0), (True, False, False))
    huge = Cylinder((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 1.0e20, 1.0e-10)
    with pytest.raises(GeometryError, match="safety limit"):
        candidate_lattice_shifts(huge, huge, domain, 0.0)
    with pytest.raises(GeometryError, match="safety limit"):
        cell_list_contacts((huge, huge.translated((0.0, 0.5, 0.0))), domain)


def test_face_image_enumeration_fails_before_pathological_iteration() -> None:
    domain = Domain((1.0, 1.0e-8, 1.0), (False, True, False))
    with pytest.raises(GeometryError, match="face-image count .* safety limit"):
        face_gaps(Sphere((0.0, 0.0, 0.0), 0.1), domain, "x")


def test_face_image_limit_handles_counts_beyond_platform_range_length() -> None:
    domain = Domain((1.0, 1.0e-6, 1.0), (False, True, False))
    huge = Cylinder((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), 1.0e20, 1.0e-10)
    with pytest.raises(GeometryError, match="face-image count .* safety limit"):
        face_gaps(huge, domain, "x")


def test_same_parent_fragments_do_not_create_contact_edge() -> None:
    domain = Domain(10.0, True)
    particles = (
        Sphere((-4.8, 0.0, 0.0), 0.5, "left", "parent"),
        Sphere((4.8, 0.0, 0.0), 0.5, "right", "parent"),
    )
    assert find_contacts(particles, domain).edges == ()


def test_exact_threshold_is_accepted() -> None:
    particles = (Sphere((0.0, 0.0, 0.0), 1.0), Sphere((3.8, 0.0, 0.0), 1.0))
    result = find_contacts(
        particles,
        Domain(10.0, False),
        ThresholdContactModel(1.8),
        method="bruteforce",
    )
    assert len(result.edges) == 1
    assert result.edges[0].distance == pytest.approx(1.8)


def test_threshold_just_outside_tolerance_is_rejected() -> None:
    particles = (
        Sphere((0.0, 0.0, 0.0), 1.0),
        Sphere((3.8 + 2.0e-8, 0.0, 0.0), 1.0),
    )
    result = find_contacts(
        particles,
        Domain(10.0, False),
        ThresholdContactModel(1.8),
        method="bruteforce",
    )
    assert result.edges == ()


def test_finite_face_gap_penalizes_particle_beyond_face_edge() -> None:
    domain = Domain(10.0, False)
    sphere = Sphere((5.0, 8.0, 0.0), 1.0)
    lower, upper = face_gaps(sphere, domain, "x")
    assert lower > 9.0
    assert upper == pytest.approx(2.0)


def test_transverse_periodicity_tiles_finite_face() -> None:
    domain = Domain(10.0, (False, True, False))
    sphere = Sphere((5.0, 8.0, 0.0), 1.0)
    _, upper = face_gaps(sphere, domain, "x")
    assert upper == 0.0


def test_face_gap_work_is_invariant_under_large_transverse_lattice_gauge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import microperco.geometry.periodic as periodic_module

    domain = Domain((200.0, 1.0, 300.0), (False, True, False))
    base = Sphere((0.0, 0.23, 0.0), 10.0)
    shifted = base.translated((0.0, 1.0e16, 0.0))
    original = periodic_module._rectangle_distance
    calls = 0

    def counted(*args: Any, **kwargs: Any) -> float:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(periodic_module, "_rectangle_distance", counted)
    base_gaps = face_gaps(base, domain, "x")
    base_calls = calls
    calls = 0
    shifted_gaps = face_gaps(shifted, domain, "x")
    assert shifted_gaps == pytest.approx(base_gaps)
    assert calls <= base_calls + 4


def test_wrapped_parent_face_mode_is_explicit_opt_in() -> None:
    domain = Domain(10.0, (True, False, False))
    sphere = Sphere((4.8, 0.0, 0.0), 0.5, 0, "parent")
    default = face_gaps(sphere, domain, "x")
    historical = face_gaps(sphere, domain, "x", wrapped_parent=True)
    assert default[0] > 9.0 and default[1] == 0.0
    assert historical == (0.0, 0.0)


def test_search_dispatch_aliases_and_errors() -> None:
    particles = (Sphere((0.0, 0.0, 0.0), 1.0), Sphere((1.0, 0.0, 0.0), 1.0))
    domain = Domain(10.0, False)
    assert find_contacts(particles, domain, method="spatial-hash").method == "cell_list"
    assert find_contacts(particles, domain, method="brute-force").method == "bruteforce"
    with pytest.raises(ValueError):
        find_contacts(particles, domain, method="octree")
