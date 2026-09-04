# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math

import numpy as np
import pytest

from microperco import Cylinder, NumericalPolicy, Sphere, distance
from microperco.exceptions import GeometryError
from microperco.geometry import (
    cylinder_cylinder_distance,
    cylinder_rectangle_distance,
    point_cylinder_distance,
    point_rectangle_distance,
    segment_segment_distance,
    sphere_cylinder_distance,
    sphere_rectangle_distance,
    sphere_sphere_distance,
)


@pytest.mark.parametrize(
    "separation, expected",
    [(0.0, 0.0), (1.5, 0.0), (2.0, 0.0), (2.75, 0.75), (10.0, 8.0)],
)
def test_sphere_sphere_analytic(separation: float, expected: float) -> None:
    first = Sphere((0.0, 0.0, 0.0), 1.0)
    second = Sphere((separation, 0.0, 0.0), 1.0)
    assert sphere_sphere_distance(first, second) == pytest.approx(expected)


@pytest.mark.parametrize(
    "point, expected",
    [
        ((0.0, 0.0, 0.0), 0.0),
        ((1.5, 0.0, 0.0), 0.5),
        ((0.0, 2.0, 0.0), 1.0),
        ((2.0, 2.0, 0.0), math.sqrt(2.0)),
    ],
)
def test_point_to_flat_cylinder(point: tuple[float, float, float], expected: float) -> None:
    cylinder = Cylinder((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 2.0, 1.0)
    assert point_cylinder_distance(point, cylinder) == pytest.approx(expected)


def test_sphere_cylinder_endcap_side_and_overlap() -> None:
    cylinder = Cylinder((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 2.0, 1.0)
    assert sphere_cylinder_distance(Sphere((2.0, 0.0, 0.0), 0.25), cylinder) == pytest.approx(0.75)
    assert sphere_cylinder_distance(Sphere((0.0, 2.0, 0.0), 0.25), cylinder) == pytest.approx(0.75)
    assert sphere_cylinder_distance(Sphere((0.0, 0.0, 0.0), 0.25), cylinder) == 0.0


@pytest.mark.parametrize(
    "second, expected",
    [
        (Cylinder((3.0, 0.0, 0.0), (1.0, 0.0, 0.0), 2.0, 0.5), 1.0),
        (Cylinder((0.0, 2.0, 0.0), (1.0, 0.0, 0.0), 2.0, 0.5), 1.0),
        (Cylinder((0.0, 0.0, 2.0), (0.0, 1.0, 0.0), 2.0, 0.5), 1.0),
        (Cylinder((0.5, 0.0, 0.0), (0.0, 1.0, 0.0), 2.0, 0.5), 0.0),
    ],
)
def test_cylinder_cylinder_known_cases(second: Cylinder, expected: float) -> None:
    first = Cylinder((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 2.0, 0.5)
    assert cylinder_cylinder_distance(first, second) == pytest.approx(expected, abs=2.0e-8)


def test_gjk_nonconvergence_is_explicit_instead_of_returning_an_uncertified_gap() -> None:
    first = Cylinder((0.0, 0.0, 0.0), (1.0, 2.0, 3.0), 4.0, 0.7)
    second = Cylinder((5.0, -1.0, 2.0), (2.0, -3.0, 1.0), 3.0, 0.4)
    with pytest.raises(GeometryError, match="did not converge"):
        cylinder_cylinder_distance(
            first,
            second,
            policy=NumericalPolicy(gjk_max_iterations=1),
        )


def test_stalled_cylinder_gjk_uses_certified_convex_fallback() -> None:
    # This well-separated, highly asymmetric pair exhausts distance GJK on
    # some BLAS/LAPACK builds with a bracket only slightly wider than the
    # requested tolerance.  The convex fallback must close that bracket,
    # rather than report failure or turn the positive gap into contact.
    first = Cylinder(
        (0.7248418432697861, 8.622982566915287, -4.648414180819842),
        (-3.1471082392708456, 1.9963589691646537, -0.5644930923300792),
        43.33608402233546,
        3.6035169208885045,
    )
    second = Cylinder(
        (-1.5564711889034744, -3.338283612451292, 3.347031487556766),
        (-0.08734053269774296, 0.2101618470363272, -0.21614630700643458),
        0.13651796074641298,
        1.490716805296276,
    )
    policy = NumericalPolicy(gjk_max_iterations=32)
    expected = 8.6842039296
    assert cylinder_cylinder_distance(first, second, policy=policy) == pytest.approx(
        expected, abs=2.0e-9
    )
    assert cylinder_cylinder_distance(second, first, policy=policy) == pytest.approx(
        expected, abs=2.0e-9
    )


def test_nearly_parallel_cylinders_are_symmetric() -> None:
    first = Cylinder((0.0, 0.0, 0.0), (1.0, 1.0e-10, 0.0), 4.0, 0.4)
    second = Cylinder((0.2, 2.0, 0.1), (1.0, -1.0e-10, 0.0), 3.0, 0.3)
    forward = distance(first, second)
    reverse = distance(second, first)
    assert forward >= 0.0
    assert forward == pytest.approx(reverse, abs=2.0e-8)


def test_long_nearly_parallel_cylinders_keep_their_transverse_gap() -> None:
    first = Cylinder((0.0, 0.0, 0.0), (1.0, 1.0e-16, 0.0), 1.0e9, 0.1)
    second = Cylinder((0.0, 1.0, 0.0), (1.0, -1.0e-16, 0.0), 1.0e9, 0.1)
    assert cylinder_cylinder_distance(first, second) == pytest.approx(0.8, abs=2.0e-7)


def test_high_aspect_parallel_cylinders_detect_side_overlap() -> None:
    first = Cylinder((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), 1.0e9, 1.0)
    second = Cylinder((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 1.0e9, 1.0)
    assert cylinder_cylinder_distance(first, second) == 0.0


def test_asymmetric_high_aspect_parallel_cylinders_detect_overlap() -> None:
    first = Cylinder((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 1.0e9, 0.012)
    second = Cylinder((-4.6e8, 0.019, 0.0), (1.0, 0.0, 0.0), 25.0, 0.046)
    assert cylinder_cylinder_distance(first, second) == 0.0


def test_asymmetric_high_aspect_nearly_parallel_cylinders_detect_overlap() -> None:
    first = Cylinder((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 1.0e9, 0.012)
    second = Cylinder((-4.6e8, 0.019, 0.0), (1.0, 1.0e-12, 0.0), 25.0, 0.046)
    assert cylinder_cylinder_distance(first, second) == 0.0


def test_nearly_parallel_long_cylinder_end_to_side_gap_is_resolved() -> None:
    first = Cylinder((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 1.0e9, 0.1)
    second = Cylinder((0.0, 1.0, 0.0), (1.0, 1.0e-9, 0.0), 1.0e9, 0.1)
    expected = 0.3
    assert cylinder_cylinder_distance(first, second) == pytest.approx(expected, abs=2.0e-9)
    assert cylinder_cylinder_distance(second, first) == pytest.approx(expected, abs=2.0e-9)


def test_nearly_parallel_stalled_gjk_retains_positive_planar_gap() -> None:
    first = Cylinder(
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        19774.661668169934,
        0.5078052434857245,
    )
    angle = -3.2945711659308436e-8
    second = Cylinder(
        (2822.4967337691805, -0.6698702553178393, 0.0),
        (math.cos(angle), math.sin(angle), 0.0),
        15204.071003818848,
        0.0029307186913366708,
    )
    expected = 0.15888383867110834
    assert cylinder_cylinder_distance(first, second) == pytest.approx(expected, abs=2.0e-9)
    assert cylinder_cylinder_distance(second, first) == pytest.approx(expected, abs=2.0e-9)


def test_nearly_parallel_long_cylinder_centerlines_cross_inside_segments() -> None:
    first = Cylinder((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 2.0e9, 0.1)
    second = Cylinder((0.0, 50.0, 0.0), (1.0, 1.0e-7, 0.0), 2.0e9, 0.1)
    assert cylinder_cylinder_distance(first, second) == 0.0


def test_asymmetric_nearly_parallel_cylinder_axis_crossing_is_detected() -> None:
    angle = 9.415647683409536e-8
    first = Cylinder(
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        2_627_489.7277399124,
        0.00030650085566899125,
    )
    second = Cylinder(
        (-1_629_856.0003364112, -0.1152231000938434, 0.0),
        (math.cos(angle), math.sin(angle), 0.0),
        9_667_314.350534419,
        0.0044824308959150955,
    )
    assert cylinder_cylinder_distance(first, second) == 0.0
    assert cylinder_cylinder_distance(second, first) == 0.0


@pytest.mark.parametrize(
    "transverse_center, expected",
    [(1.25, 0.0), (1.3, 0.03507672021902364)],
)
def test_high_aspect_end_disk_sidewall_shallow_intersection(
    transverse_center: float, expected: float
) -> None:
    angle = -2.6503808368947373e-7
    first = Cylinder(
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        2.0 * 8_428_916.25680924,
        0.02572304788186505,
    )
    second = Cylinder(
        (-6_167_615.677445408, transverse_center, 0.0),
        (math.cos(angle), math.sin(angle), 0.0),
        2.0 * 4_328_826.019916564,
        0.09189647895531027,
    )
    assert cylinder_cylinder_distance(first, second) == pytest.approx(expected, abs=2.0e-9)


def test_parallel_cylinder_closed_form_combines_axial_and_radial_gaps() -> None:
    first = Cylinder((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 4.0, 1.0)
    second = Cylinder((4.0, 3.0, 0.0), (-1.0, 0.0, 0.0), 2.0, 0.5)
    assert cylinder_cylinder_distance(first, second) == pytest.approx(math.hypot(1.0, 1.5))


def test_high_aspect_coaxial_cylinders_detect_end_overlap() -> None:
    first = Cylinder((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 3.0, 1.0e8)
    second = Cylinder((0.0, 0.0, 1.8), (0.0, 0.0, -1.0), 1.0, 1.0)
    assert cylinder_cylinder_distance(first, second) == 0.0


@pytest.mark.parametrize("offset", [(0.0, 0.0, 0.0), (10.0, -7.0, 4.0), (-1.0e4, 2.0e4, 3.0)])
def test_distance_translation_invariance(offset: tuple[float, float, float]) -> None:
    first = Cylinder((1.0, 2.0, 3.0), (1.0, 2.0, 3.0), 4.0, 0.7)
    second = Sphere((5.0, -1.0, 2.0), 0.8)
    translated_first = first.translated(offset)
    translated_second = second.translated(offset)
    assert distance(translated_first, translated_second) == pytest.approx(
        distance(first, second), abs=2.0e-9
    )


def test_unified_dispatch_is_symmetric_for_all_shape_pairs() -> None:
    particles = (
        Sphere((0.0, 0.0, 0.0), 0.5),
        Sphere((2.0, 0.0, 0.0), 0.2),
        Cylinder((0.0, 2.0, 0.0), (0.0, 0.0, 1.0), 1.5, 0.4),
    )
    for first in particles:
        for second in particles:
            assert distance(first, second) == pytest.approx(distance(second, first), abs=2.0e-8)


def test_segment_distance_crossing_parallel_and_points() -> None:
    assert segment_segment_distance((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0)) == pytest.approx(
        0.0
    )
    assert segment_segment_distance((0, 0, 0), (1, 0, 0), (0, 2, 0), (1, 2, 0)) == pytest.approx(
        2.0
    )
    assert segment_segment_distance((0, 0, 0), (0, 0, 0), (3, 4, 0), (3, 4, 0)) == pytest.approx(
        5.0
    )


def test_segment_distance_does_not_scale_degeneracy_by_global_separation() -> None:
    result = segment_segment_distance(
        (0.0, 0.0, 0.0),
        (10.0, 0.0, 0.0),
        (1.0e9, 0.0, 0.0),
        (1.0e9 + 10.0, 0.0, 0.0),
    )
    assert result == pytest.approx(1.0e9 - 10.0)


@pytest.mark.parametrize("angle", [1.0e-12, 1.0e-170, 1.0e-300])
def test_segment_distance_resolves_crossings_at_extreme_angles(angle: float) -> None:
    result = segment_segment_distance(
        (-1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (-1.0, -angle, 0.0),
        (1.0, angle, 0.0),
    )
    assert result == pytest.approx(0.0, abs=angle * 1.0e-6)


def test_segment_distance_resolves_crossing_with_huge_coordinate_scale() -> None:
    length = 1.0e16
    result = segment_segment_distance(
        (-length / 2.0, 0.0, 0.0),
        (length / 2.0, 0.0, 0.0),
        (-3.0e15, -0.4, 0.0),
        (7.0e15, 0.6, 0.0),
    )
    assert result == pytest.approx(0.0, abs=1.0e-70)


def test_segment_distance_is_exact_across_full_float_exponent_range() -> None:
    first = Cylinder((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 1.0e100, 1.0)
    angle = 1.0e-98
    second = Cylinder(
        (3.0e99, 10.0, 0.0),
        (math.cos(angle), math.sin(angle), 0.0),
        3.0e99,
        1.0,
    )
    assert segment_segment_distance(*first.endpoints, *second.endpoints) == 0.0


def test_near_axis_cylinder_aabb_and_support_are_stable() -> None:
    cylinder = Cylinder((0.0, 0.0, 0.0), (1.0, 1.0e-8, 0.0), 2.0e8, 100.0)
    axis = cylinder.axis
    expected_x = cylinder.half_length * abs(float(axis[0])) + cylinder.radius * math.hypot(
        float(axis[1]), float(axis[2])
    )
    assert cylinder.aabb_extent[0] == pytest.approx(expected_x, rel=2.0e-15)
    support = cylinder.support((1.0, 0.0, 0.0))
    assert float(np.dot(support, (1.0, 0.0, 0.0))) == pytest.approx(expected_x)


def test_cylinder_support_uses_center_for_axial_ties() -> None:
    cylinder = Cylinder((1.0, 2.0, 3.0), (1.0, 0.0, 0.0), 10.0, 2.0)
    np.testing.assert_allclose(cylinder.support((0.0, 1.0, 0.0)), (1.0, 4.0, 3.0))
    np.testing.assert_allclose(cylinder.support((1.0, 0.0, 0.0)), (6.0, 2.0, 3.0))


def test_cylinder_support_preserves_tiny_nonzero_axial_projection() -> None:
    cylinder = Cylinder((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 2.0e15, 1.0)
    direction = np.array((1.0e-15, 1.0, 0.0))
    direction /= np.linalg.norm(direction)
    support_value = float(np.dot(cylinder.support(direction), direction))
    expected = cylinder.half_length * abs(float(np.dot(cylinder.axis, direction)))
    expected += cylinder.radius * np.linalg.norm(np.cross(direction, cylinder.axis))
    assert support_value == pytest.approx(expected, rel=2.0e-15)


def test_rectangle_distances_are_finite_face_not_plane() -> None:
    center = np.zeros(3)
    half_extents = np.array((0.0, 1.0, 1.0))
    assert point_rectangle_distance((0.0, 3.0, 0.0), center, half_extents) == pytest.approx(2.0)
    assert sphere_rectangle_distance(
        Sphere((0.0, 3.0, 0.0), 0.5), center, half_extents
    ) == pytest.approx(1.5)
    cylinder = Cylinder((0.0, 3.0, 0.0), (0.0, 0.0, 1.0), 1.0, 0.5)
    assert cylinder_rectangle_distance(cylinder, center, half_extents) == pytest.approx(
        1.5, abs=2.0e-8
    )


def test_stalled_cylinder_rectangle_gjk_uses_certified_fallback() -> None:
    cylinder = Cylinder(
        (-0.6908793931556856, 1.6318062047259545, -4.871595176331934),
        (-0.6623148384847501, -0.6109325308300845, -0.4337055423861532),
        4.0,
        0.3,
    )
    rectangle_center = (-5.0, 10.0, 0.0)
    half_extents = (0.0, 5.0, 5.0)
    assert cylinder_rectangle_distance(
        cylinder,
        rectangle_center,
        half_extents,
        policy=NumericalPolicy(gjk_max_iterations=32),
    ) == pytest.approx(5.1040657795, abs=2.0e-9)
    with pytest.raises(GeometryError, match="did not converge"):
        cylinder_rectangle_distance(
            cylinder,
            rectangle_center,
            half_extents,
            policy=NumericalPolicy(gjk_max_iterations=1),
        )


def test_cylinder_rectangle_handles_huge_transverse_face_extent() -> None:
    cylinder = Cylinder((4.0, 0.0, 0.0), (1.0, 1.0e-12, 0.0), 2.0, 0.5)
    face_center = np.zeros(3)
    half_extents = np.array((0.0, 5.0e14, 5.0))
    assert cylinder_rectangle_distance(cylinder, face_center, half_extents) == pytest.approx(
        3.0, abs=2.0e-9
    )


def test_high_aspect_cylinder_detects_finite_face_overlap() -> None:
    cylinder = Cylinder((-4.0, 0.0, 0.0), (0.0, 1.0, 0.0), 1.0e9, 1.0)
    assert cylinder_rectangle_distance(
        cylinder,
        (-5.0, 0.0, 0.0),
        (0.0, 5.0, 5.0),
    ) == pytest.approx(0.0, abs=1.0e-12)


def test_nearly_parallel_cylinder_axis_piercing_face_is_detected() -> None:
    cylinder = Cylinder(
        (0.0, 0.0, 0.0),
        (2.749535427687764e-8, 0.9999999999999996, 0.0),
        363_109.40038509946,
        1.0860553037416517e-6,
    )
    assert (
        cylinder_rectangle_distance(
            cylinder,
            (-0.0035105373709489155, -119_941.05118569263, 0.0),
            (0.0, 26_534.44157111919, 0.002),
        )
        == 0.0
    )


def test_wide_cylinder_detects_small_rectangle_inside_end_cap() -> None:
    cylinder = Cylinder((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 3.0, 1.0e8)
    assert cylinder_rectangle_distance(
        cylinder,
        (0.0, 0.0, 1.4),
        (1.0, 1.0, 0.0),
    ) == pytest.approx(0.0, abs=1.0e-12)


def test_off_center_wide_cylinder_detects_shared_rectangle_point() -> None:
    axis = np.array((2.0, 3.0, 4.0))
    axis /= np.linalg.norm(axis)
    transverse = np.cross(axis, (1.0, 0.0, 0.0))
    transverse /= np.linalg.norm(transverse)
    cylinder = Cylinder(-0.0009 * axis - 1000.0 * transverse, axis, 0.002, 1.0e6)
    assert cylinder_rectangle_distance(
        cylinder,
        (0.0, 0.0, 0.0),
        (0.01, 0.01, 0.0),
    ) == pytest.approx(0.0, abs=1.0e-12)
