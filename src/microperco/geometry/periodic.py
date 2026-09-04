# SPDX-License-Identifier: Apache-2.0
"""Periodic images and finite rectangular electrode-face distances."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import product

import numpy as np

from ..domain import Domain, _shifted_coordinate, normalize_axis
from ..exceptions import GeometryError
from ..numerics import DEFAULT_NUMERICAL_POLICY, NumericalPolicy
from ..particles import Cylinder, Particle, Sphere
from .distance import cylinder_rectangle_distance, distance, sphere_rectangle_distance

LatticeShift = tuple[int, int, int]
MAX_LATTICE_IMAGES = 1_000_000


@dataclass(frozen=True, slots=True)
class PeriodicDistance:
    """Minimum particle gap and lattice image applied to the second item."""

    distance: float
    lattice_shift: LatticeShift

    def __iter__(self) -> Iterator[float | LatticeShift]:
        yield self.distance
        yield self.lattice_shift


def aabb(
    particle: Particle,
    displacement: Sequence[float] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the tight axis-aligned bounding box, optionally translated."""

    center = np.asarray(particle.center, dtype=np.float64)
    if displacement is not None:
        vector = np.asarray(displacement, dtype=np.float64)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise ValueError("displacement must be a finite three-vector")
        center = center + vector
    extent = np.asarray(particle.aabb_extent)
    return center - extent, center + extent


def _canonical_particle(particle: Particle, domain: Domain) -> tuple[Particle, LatticeShift]:
    """Return a numerically local periodic representative and its lattice shift."""

    shift = domain.canonical_shift(particle.center)
    center = np.asarray(particle.center, dtype=np.float64).copy()
    try:
        for axis, enabled in enumerate(domain.periodic):
            if enabled:
                center[axis] = _shifted_coordinate(
                    float(center[axis]),
                    shift[axis],
                    domain.size[axis],
                )
    except (OverflowError, ValueError) as exc:
        raise GeometryError("canonical particle translation is not representable") from exc
    if not np.all(np.isfinite(center)):
        raise GeometryError("canonical particle translation is not representable")
    if isinstance(particle, Sphere):
        canonical: Particle = Sphere(
            center,
            particle.radius,
            particle.particle_id,
            particle.parent_id,
            particle.image_offset,
        )
    else:
        canonical = Cylinder(
            center,
            particle.axis,
            particle.length,
            particle.radius,
            particle.particle_id,
            particle.parent_id,
            particle.image_offset,
        )
    return canonical, shift


def _candidate_lattice_shifts_local(
    first: Particle,
    second: Particle,
    domain: Domain,
    max_distance: float,
    *,
    policy: NumericalPolicy,
) -> tuple[LatticeShift, ...]:
    """Enumerate shifts for particle centers already in the canonical cell."""

    limit = float(max_distance)
    if not math.isfinite(limit) or limit < 0.0:
        raise ValueError("max_distance must be finite and non-negative")
    delta = np.asarray(second.center) - np.asarray(first.center)
    extent = np.asarray(first.aabb_extent) + np.asarray(second.aabb_extent)
    sizes = np.asarray(domain.size)
    pair_scale = max(
        math.hypot(*(float(component) for component in delta)),
        float(np.max(extent)),
        limit,
        1.0,
    )
    interaction_tolerance = policy.tolerance(pair_scale)
    ranges: list[range] = []
    for axis, enabled in enumerate(domain.periodic):
        axis_scale = max(
            abs(float(delta[axis])),
            float(extent[axis]),
            limit,
            domain.size[axis] if enabled else 0.0,
            1.0,
        )
        tolerance = max(interaction_tolerance, policy.tolerance(axis_scale))
        bound = float(extent[axis] + limit + tolerance)
        if enabled:
            minimum = math.ceil((-bound - float(delta[axis])) / sizes[axis])
            maximum = math.floor((bound - float(delta[axis])) / sizes[axis])
            if minimum > maximum:
                return ()
            ranges.append(range(minimum, maximum + 1))
        else:
            if abs(float(delta[axis])) > bound:
                return ()
            ranges.append(range(0, 1))
    image_count = math.prod(max(0, values.stop - values.start) for values in ranges)
    if image_count > MAX_LATTICE_IMAGES:
        raise GeometryError(
            f"periodic image count {image_count} exceeds the safety limit "
            f"of {MAX_LATTICE_IMAGES}"
        )
    return tuple((int(x), int(y), int(z)) for x, y, z in product(*ranges))


def candidate_lattice_shifts(
    first: Particle,
    second: Particle,
    domain: Domain,
    max_distance: float,
    *,
    policy: NumericalPolicy = DEFAULT_NUMERICAL_POLICY,
) -> tuple[LatticeShift, ...]:
    """Enumerate every image whose AABBs can be within ``max_distance``."""

    canonical_first, first_shift = _canonical_particle(first, domain)
    canonical_second, second_shift = _canonical_particle(second, domain)
    local = _candidate_lattice_shifts_local(
        canonical_first,
        canonical_second,
        domain,
        max_distance,
        policy=policy,
    )
    return tuple(
        (
            shift[0] + second_shift[0] - first_shift[0],
            shift[1] + second_shift[1] - first_shift[1],
            shift[2] + second_shift[2] - first_shift[2],
        )
        for shift in local
    )


def periodic_distance(
    first: Particle,
    second: Particle,
    domain: Domain,
    *,
    policy: NumericalPolicy = DEFAULT_NUMERICAL_POLICY,
) -> PeriodicDistance:
    """Return globally minimum gap over relevant enabled periodic images."""

    canonical_first, first_shift = _canonical_particle(first, domain)
    canonical_second, second_shift = _canonical_particle(second, domain)
    _, nearest = domain.minimum_image_displacement(
        canonical_first.center,
        canonical_second.center,
    )
    nearest_particle = canonical_second.translated(domain.lattice_vector(nearest))
    upper_bound = distance(canonical_first, nearest_particle, policy=policy)
    shifts = set(
        _candidate_lattice_shifts_local(
            canonical_first,
            canonical_second,
            domain,
            upper_bound,
            policy=policy,
        )
    )
    shifts.add(nearest)
    best_distance = upper_bound
    best_shift = nearest
    for shift in sorted(shifts):
        if shift == nearest:
            continue
        shifted = canonical_second.translated(domain.lattice_vector(shift))
        value = distance(canonical_first, shifted, policy=policy)
        if value < best_distance or (value == best_distance and shift < best_shift):
            best_distance = value
            best_shift = shift
    reported_shift = (
        best_shift[0] + second_shift[0] - first_shift[0],
        best_shift[1] + second_shift[1] - first_shift[1],
        best_shift[2] + second_shift[2] - first_shift[2],
    )
    return PeriodicDistance(max(0.0, best_distance), reported_shift)


def _rectangle_distance(
    particle: Particle,
    rectangle_center: np.ndarray,
    half_extents: np.ndarray,
    policy: NumericalPolicy,
) -> float:
    if isinstance(particle, Sphere):
        return sphere_rectangle_distance(particle, rectangle_center, half_extents)
    if isinstance(particle, Cylinder):
        return cylinder_rectangle_distance(particle, rectangle_center, half_extents, policy=policy)
    raise TypeError("finite face distance supports Sphere and Cylinder only")


def _nearest_tile_index(coordinate: float, center: float, length: float) -> int:
    value = (coordinate - center) / length
    return math.floor(value + 0.5) if value >= 0.0 else math.ceil(value - 0.5)


def _face_gap(
    particle: Particle,
    domain: Domain,
    axis: int,
    coordinate: float,
    policy: NumericalPolicy,
) -> float:
    """Return minimum gap to a transversely tiled finite face."""

    transverse_periodic = tuple(
        enabled and index != axis for index, enabled in enumerate(domain.periodic)
    )
    evaluated = particle
    if any(transverse_periodic):
        evaluated, _ = _canonical_particle(
            particle,
            domain.with_periodic(transverse_periodic),
        )
    rectangle_center = np.asarray(domain.center, dtype=np.float64).copy()
    rectangle_center[axis] = coordinate
    half_extents = 0.5 * np.asarray(domain.size, dtype=np.float64)
    half_extents[axis] = 0.0

    nearest = [0, 0, 0]
    for transverse in range(3):
        if transverse != axis and domain.periodic[transverse]:
            nearest[transverse] = _nearest_tile_index(
                float(evaluated.center[transverse]),
                float(rectangle_center[transverse]),
                domain.size[transverse],
            )
    initial_center = rectangle_center + domain.lattice_vector(nearest)
    upper_bound = _rectangle_distance(evaluated, initial_center, half_extents, policy)
    delta = rectangle_center - np.asarray(evaluated.center)
    sizes = np.asarray(domain.size)
    extent_sum = np.asarray(particle.aabb_extent) + half_extents
    ranges: list[range] = []
    for transverse, enabled in enumerate(domain.periodic):
        if transverse == axis or not enabled:
            ranges.append(range(0, 1))
            continue
        scale = max(
            abs(float(delta[transverse])),
            float(extent_sum[transverse]),
            upper_bound,
            domain.size[transverse],
            1.0,
        )
        tolerance = policy.tolerance(scale)
        bound = float(extent_sum[transverse] + upper_bound + tolerance)
        minimum = math.ceil((-bound - float(delta[transverse])) / sizes[transverse])
        maximum = math.floor((bound - float(delta[transverse])) / sizes[transverse])
        ranges.append(range(minimum, maximum + 1))
    image_count = math.prod(max(0, values.stop - values.start) for values in ranges)
    if image_count > MAX_LATTICE_IMAGES:
        raise GeometryError(
            f"periodic face-image count {image_count} exceeds the safety limit "
            f"of {MAX_LATTICE_IMAGES}"
        )
    best = math.inf
    for shift in product(*ranges):
        shifted_center = rectangle_center + domain.lattice_vector(shift)
        best = min(best, _rectangle_distance(evaluated, shifted_center, half_extents, policy))
        if best == 0.0:
            break
    if not math.isfinite(best):
        raise GeometryError("particle-to-domain-face distance is not representable")
    return max(0.0, best)


def face_gaps(
    particle: Particle,
    domain: Domain,
    axis: int | str,
    *,
    wrapped_parent: bool = False,
    policy: NumericalPolicy = DEFAULT_NUMERICAL_POLICY,
) -> tuple[float, float]:
    """Return gaps to distinct finite lower and upper electrode rectangles.

    Enabled transverse axes tile each face. ``wrapped_parent=True`` applies the
    explicit historical interpretation that a parent crossing a periodic
    analysis seam also contacts the opposite electrode.
    """

    index = normalize_axis(axis)
    evaluated = particle
    if wrapped_parent and domain.periodic[index]:
        axis_periodic = tuple(
            enabled and current == index
            for current, enabled in enumerate(domain.periodic)
        )
        evaluated, _ = _canonical_particle(
            particle,
            domain.with_periodic(axis_periodic),
        )
    lower_gap = _face_gap(evaluated, domain, index, domain.lower[index], policy)
    upper_gap = _face_gap(evaluated, domain, index, domain.upper[index], policy)
    if wrapped_parent and domain.periodic[index]:
        tolerance = policy.tolerance(max(float(evaluated.aabb_extent[index]), 1.0))
        crosses_lower = lower_gap <= tolerance
        crosses_upper = upper_gap <= tolerance
        if crosses_upper:
            lower_gap = 0.0
        if crosses_lower:
            upper_gap = 0.0
    return max(0.0, lower_gap), max(0.0, upper_gap)
