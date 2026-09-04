# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math

import numpy as np
import pytest

from microperco import Cylinder, Domain, Sphere
from microperco.exceptions import ConfigurationError, GeometryError
from microperco.generation import CylinderSpec, MaterialSpec, PopulationSpec, SphereSpec
from microperco.numerics import NumericalPolicy, positive_finite_product


def test_domain_scalar_size_and_periodicity() -> None:
    domain = Domain(4.0, True)
    assert domain.size == (4.0, 4.0, 4.0)
    assert domain.periodic == (True, True, True)
    assert domain.lower == (-2.0, -2.0, -2.0)
    assert domain.upper == (2.0, 2.0, 2.0)
    assert domain.volume == 64.0


def test_domain_centered_bounds_and_axes() -> None:
    domain = Domain((2.0, 4.0, 6.0), (True, False, True), (1.0, 2.0, 3.0))
    assert domain.lower == (0.0, 0.0, 0.0)
    assert domain.upper == (2.0, 4.0, 6.0)
    assert domain.periodic_axes == (0, 2)
    assert domain.with_periodic(False).periodic == (False, False, False)


def test_domain_rejects_bounds_that_collapse_at_large_coordinates() -> None:
    with pytest.raises(ValueError, match="collapse"):
        Domain(10.0, False, (1.0e308, 0.0, 0.0))


@pytest.mark.parametrize(
    "sizes,expected",
    [
        ((1.0e200, 1.0e200, 1.0e-200), 1.0e200),
        ((1.0e-200, 1.0e-200, 1.0e200), 1.0e-200),
    ],
)
def test_domain_volume_avoids_intermediate_range_errors(
    sizes: tuple[float, float, float], expected: float
) -> None:
    assert Domain(sizes, False).volume == pytest.approx(expected, rel=2.0e-15, abs=0.0)


@pytest.mark.parametrize("sizes", [(1.0e308, 1.0e308, 1.0), (1.0e-300, 1.0e-300, 1.0)])
def test_domain_rejects_truly_unrepresentable_volume(sizes: tuple[float, float, float]) -> None:
    with pytest.raises(ValueError, match="domain volume"):
        Domain(sizes, False)


@pytest.mark.parametrize(
    "value",
    [0.0, -1.0, math.inf, math.nan, (1.0, 2.0), (1.0, 2.0, math.inf)],
)
def test_domain_rejects_bad_sizes(value: object) -> None:
    with pytest.raises(ValueError):
        Domain(value, False)  # type: ignore[arg-type]


@pytest.mark.parametrize("periodic", [(True, False), (1, 0, 1), "yes", None])
def test_domain_rejects_bad_periodicity(periodic: object) -> None:
    with pytest.raises(ValueError):
        Domain(1.0, periodic)  # type: ignore[arg-type]


def test_domain_wrap_only_enabled_axes() -> None:
    domain = Domain((10.0, 8.0, 6.0), (True, False, True))
    wrapped = domain.wrap((6.0, 7.0, -4.0))
    np.testing.assert_allclose(wrapped, (-4.0, 7.0, 2.0))


def test_domain_wrap_vectorized() -> None:
    domain = Domain(10.0, (True, True, False))
    wrapped = domain.wrap(((5.0, -5.0, 8.0), (15.2, 26.0, -8.0)))
    np.testing.assert_allclose(wrapped, ((-5.0, -5.0, 8.0), (-4.8, -4.0, -8.0)))


def test_domain_wrap_uses_one_rounding_for_large_lattice_gauge() -> None:
    length = 0.04896433081287396
    domain = Domain((length, 1.0, 1.0), (True, False, False))
    wrapped = domain.wrap((4_412_295_474_029.962, 0.0, 0.0))
    assert wrapped[0] == pytest.approx(0.01904142015551394, abs=1.0e-18)


def test_minimum_image_and_lattice_vector() -> None:
    domain = Domain(10.0, (True, False, False))
    displacement, shift = domain.minimum_image_displacement((-4.0, 1.0, 0.0), (4.0, 3.0, 0.0))
    np.testing.assert_allclose(displacement, (-2.0, 2.0, 0.0))
    assert shift == (-1, 0, 0)
    np.testing.assert_allclose(domain.lattice_vector(shift), (-10.0, 0.0, 0.0))


def test_lattice_indices_are_not_limited_to_platform_integer_width() -> None:
    domain = Domain((1.0e-100, 1.0, 1.0), (True, False, False))
    huge_shift = 10**100
    vector = domain.lattice_vector((huge_shift, 0, 0))
    assert vector[0] == pytest.approx(1.0)
    assert domain.canonical_shift((1.0, 0.0, 0.0))[0] < -(2**63)


@pytest.mark.parametrize("axis, expected", [(0, 0), (1, 1), (2, 2), ("X", 0), ("y", 1), ("z", 2)])
def test_normalize_axis(axis: int | str, expected: int) -> None:
    from microperco.domain import normalize_axis

    assert normalize_axis(axis) == expected


def test_sphere_properties_support_and_translation() -> None:
    sphere = Sphere((1.0, 2.0, 3.0), 2.0, "s", "p")
    assert sphere.id == "s"
    assert sphere.parent_id == "p"
    assert sphere.volume == pytest.approx(4.0 * math.pi * 8.0 / 3.0)
    np.testing.assert_allclose(sphere.aabb_extent, (2.0, 2.0, 2.0))
    np.testing.assert_allclose(sphere.support((3.0, 0.0, 0.0)), (3.0, 2.0, 3.0))
    np.testing.assert_allclose(sphere.translated((1.0, -2.0, 0.5)).center, (2.0, 0.0, 3.5))


def test_particle_inputs_are_copied_and_read_only() -> None:
    center = np.array((1.0, 2.0, 3.0))
    sphere = Sphere(center, 1.0)
    center[:] = 9.0
    np.testing.assert_allclose(sphere.center, (1.0, 2.0, 3.0))
    with pytest.raises(ValueError):
        sphere.center[0] = 4.0


@pytest.mark.parametrize("radius", [0.0, -1.0, math.inf, math.nan])
def test_sphere_rejects_invalid_radius(radius: float) -> None:
    with pytest.raises(GeometryError):
        Sphere((0.0, 0.0, 0.0), radius)


def test_sphere_rejects_collapsed_large_coordinate_bounds() -> None:
    with pytest.raises(GeometryError, match="collapses"):
        Sphere((1.0e308, 0.0, 0.0), 1.0)


def test_sphere_volume_can_round_to_the_smallest_subnormal() -> None:
    sphere = Sphere((0.0, 0.0, 0.0), 1.0e-108)
    assert sphere.volume == np.nextafter(0.0, 1.0)


@pytest.mark.parametrize("radius", [1.0e103, 1.0e-110])
def test_sphere_rejects_truly_unrepresentable_volume(radius: float) -> None:
    with pytest.raises(GeometryError, match="sphere volume"):
        Sphere((0.0, 0.0, 0.0), radius)


def test_cylinder_normalizes_axis_and_has_exact_endpoints() -> None:
    cylinder = Cylinder((1.0, 2.0, 3.0), (10.0, 0.0, 0.0), 4.0, 0.5)
    np.testing.assert_allclose(cylinder.axis, (1.0, 0.0, 0.0))
    start, end = cylinder.endpoints
    np.testing.assert_allclose(start, (-1.0, 2.0, 3.0))
    np.testing.assert_allclose(end, (3.0, 2.0, 3.0))
    np.testing.assert_allclose(cylinder.aabb_extent, (2.0, 0.5, 0.5))


def test_cylinder_from_endpoints_and_support() -> None:
    cylinder = Cylinder.from_endpoints((0.0, 0.0, -2.0), (0.0, 0.0, 2.0), 1.0)
    assert cylinder.length == 4.0
    np.testing.assert_allclose(cylinder.center, (0.0, 0.0, 0.0))
    np.testing.assert_allclose(cylinder.support((1.0, 0.0, 1.0)), (1.0, 0.0, 2.0))


def test_particle_image_offsets_are_validated_and_preserved() -> None:
    sphere = Sphere((0.0, 0.0, 0.0), 1.0, "s", "parent", (2, -1, 0))
    assert sphere.image_offset == (2, -1, 0)
    assert sphere.translated((1.0, 0.0, 0.0)).image_offset == (2, -1, 0)

    cylinder = Cylinder.from_endpoints(
        (-1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        0.2,
        particle_id="c",
        parent_id="parent",
        image_offset=(-3, 0, 1),
    )
    assert cylinder.image_offset == (-3, 0, 1)
    assert cylinder.translated((0.0, 1.0, 0.0)).image_offset == (-3, 0, 1)


@pytest.mark.parametrize("offset", [(1, 2), (1.0, 0, 0), (True, 0, 0)])
def test_particle_rejects_invalid_image_offset(offset: object) -> None:
    with pytest.raises(GeometryError, match="image_offset"):
        Sphere((0.0, 0.0, 0.0), 1.0, image_offset=offset)  # type: ignore[arg-type]


def test_cylinder_rejects_collapsed_large_coordinate_bounds() -> None:
    with pytest.raises(GeometryError, match="collapses"):
        Cylinder((1.0e308, 0.0, 0.0), (1.0, 0.0, 0.0), 2.0, 1.0)


@pytest.mark.parametrize(
    "radius,length,expected",
    [(1.0e200, 1.0e-200, math.pi * 1.0e200), (1.0e-200, 1.0e200, math.pi * 1.0e-200)],
)
def test_cylinder_volume_avoids_intermediate_range_errors(
    radius: float, length: float, expected: float
) -> None:
    cylinder = Cylinder((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), length, radius)
    assert cylinder.volume == pytest.approx(expected, rel=2.0e-15, abs=0.0)


@pytest.mark.parametrize("radius,length", [(1.0e200, 1.0), (1.0e-200, 1.0)])
def test_cylinder_rejects_truly_unrepresentable_volume(radius: float, length: float) -> None:
    with pytest.raises(GeometryError, match="cylinder volume"):
        Cylinder((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), length, radius)


@pytest.mark.parametrize("axis", [(0.0, 0.0, 0.0), (math.inf, 0.0, 0.0), (math.nan, 1.0, 0.0)])
def test_cylinder_rejects_bad_axis(axis: tuple[float, float, float]) -> None:
    with pytest.raises(GeometryError):
        Cylinder((0.0, 0.0, 0.0), axis, 1.0, 1.0)


def test_extreme_axis_is_normalized_without_overflow() -> None:
    cylinder = Cylinder((0.0, 0.0, 0.0), (1.0e300, -1.0e300, 1.0e-300), 2.0, 1.0)
    assert math.hypot(*cylinder.axis) == pytest.approx(1.0)
    assert np.all(np.isfinite(cylinder.aabb_extent))


def test_specs_material_volume_and_cost() -> None:
    material = MaterialSpec("silver", 3.0)
    sphere = SphereSpec(2.0, material)
    cylinder = CylinderSpec(1.0, 4.0, material)
    assert sphere.cost == pytest.approx(3.0 * sphere.volume)
    assert cylinder.volume == pytest.approx(4.0 * math.pi)
    assert PopulationSpec(cylinder, 4, "fiber").cost == pytest.approx(4.0 * cylinder.cost)


@pytest.mark.parametrize(
    "spec,expected",
    [
        (CylinderSpec(1.0e200, 1.0e-200), math.pi * 1.0e200),
        (CylinderSpec(1.0e-200, 1.0e200), math.pi * 1.0e-200),
        (SphereSpec(1.0e-108), np.nextafter(0.0, 1.0)),
    ],
)
def test_particle_spec_volumes_avoid_intermediate_range_errors(
    spec: SphereSpec | CylinderSpec, expected: float
) -> None:
    if expected == np.nextafter(0.0, 1.0):
        assert spec.volume == expected
    else:
        assert spec.volume == pytest.approx(expected, rel=2.0e-15, abs=0.0)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: SphereSpec(1.0e103),
        lambda: SphereSpec(1.0e-110),
        lambda: CylinderSpec(1.0e200, 1.0),
        lambda: CylinderSpec(1.0e-200, 1.0),
    ],
)
def test_particle_specs_reject_truly_unrepresentable_volume(factory: object) -> None:
    with pytest.raises(ConfigurationError, match="volume"):
        factory()  # type: ignore[operator]


@pytest.mark.parametrize(
    "spec",
    [
        SphereSpec,
        lambda radius, material: CylinderSpec(radius, radius, material),
    ],
)
def test_particle_specs_reject_positive_cost_that_underflows_to_zero(spec: object) -> None:
    material = MaterialSpec("tiny", 1.0e-100)
    with pytest.raises(ConfigurationError, match="particle cost"):
        spec(1.0e-100, material)  # type: ignore[operator]


def test_particle_specs_still_allow_explicitly_zero_cost_material() -> None:
    assert SphereSpec(1.0e-100, MaterialSpec("free", 0.0)).cost == 0.0


@pytest.mark.parametrize("count", [-1, 1.2, True])
def test_population_rejects_bad_count(count: object) -> None:
    with pytest.raises(ConfigurationError):
        PopulationSpec(SphereSpec(1.0), count)  # type: ignore[arg-type]


def test_numerical_policy_scale_aware_tolerance() -> None:
    policy = NumericalPolicy(1.0e-9, 1.0e-6)
    assert policy.tolerance(1.0) == 1.0e-6
    assert policy.tolerance(1.0e4) == 1.0e-2
    assert policy.less_than_or_close(1.009, 1.0, scale=1.0e4)


def test_positive_finite_product_balances_extreme_factors_and_pi() -> None:
    large = positive_finite_product(math.pi, 1.0e200, 1.0e200, 1.0e-200)
    small = positive_finite_product(math.pi, 1.0e-200, 1.0e-200, 1.0e200)
    assert large == pytest.approx(math.pi * 1.0e200, rel=2.0e-15, abs=0.0)
    assert small == pytest.approx(math.pi * 1.0e-200, rel=2.0e-15, abs=0.0)


@pytest.mark.parametrize("iterations", [1.5, True, "3", 0])
def test_numerical_policy_rejects_non_integer_iteration_limits(iterations: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        NumericalPolicy(gjk_max_iterations=iterations)  # type: ignore[arg-type]


def test_numerical_policy_normalizes_numpy_integer_iteration_limit() -> None:
    policy = NumericalPolicy(gjk_max_iterations=np.int64(64))
    assert policy.gjk_max_iterations == 64
    assert isinstance(policy.gjk_max_iterations, int)
