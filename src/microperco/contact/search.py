# SPDX-License-Identifier: Apache-2.0
"""Brute-force oracle and periodic spatial-hash contact search."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from itertools import combinations, product

import numpy as np

from ..domain import Domain
from ..exceptions import GeometryError
from ..geometry.distance import distance
from ..geometry.periodic import (
    MAX_LATTICE_IMAGES,
    LatticeShift,
    _candidate_lattice_shifts_local,
    _canonical_particle,
    aabb,
)
from ..particles import Cylinder, Particle, Sphere
from .model import ContactEdge, ContactSearchResult, ThresholdContactModel

Candidate = tuple[int, int, LatticeShift]


def _validate_particles(particles: Sequence[Particle]) -> tuple[Particle, ...]:
    items = tuple(particles)
    if not all(isinstance(item, (Sphere, Cylinder)) for item in items):
        raise TypeError("particles must contain Sphere or Cylinder instances")
    return items


def _same_parent(first: Particle, second: Particle) -> bool:
    return (
        first.parent_id is not None
        and second.parent_id is not None
        and first.parent_id == second.parent_id
    )


def _evaluate_candidates(
    canonical_particles: tuple[Particle, ...],
    base_shifts: tuple[LatticeShift, ...],
    domain: Domain,
    model: ThresholdContactModel,
    candidates: Iterable[Candidate],
    *,
    method: str,
) -> ContactSearchResult:
    ordered = sorted(set(candidates))
    edges: list[ContactEdge] = []
    evaluations = 0
    for i, j, shift in ordered:
        first = canonical_particles[i]
        local_shift = (
            shift[0] - base_shifts[j][0] + base_shifts[i][0],
            shift[1] - base_shifts[j][1] + base_shifts[i][1],
            shift[2] - base_shifts[j][2] + base_shifts[i][2],
        )
        shifted = canonical_particles[j].translated(domain.lattice_vector(local_shift))
        evaluations += 1
        gap = distance(first, shifted, policy=model.numerical_policy)
        if model.accepts_pair(gap, first, shifted):
            edges.append(ContactEdge(i, j, shift, gap))
    return ContactSearchResult(tuple(edges), len(ordered), evaluations, method)


def bruteforce_contacts(
    particles: Sequence[Particle],
    domain: Domain,
    model: ThresholdContactModel | None = None,
) -> ContactSearchResult:
    """Exhaustively evaluate every base pair and all feasible contact images."""

    items = _validate_particles(particles)
    contact_model = ThresholdContactModel() if model is None else model
    if not isinstance(contact_model, ThresholdContactModel):
        raise TypeError("model must be a ThresholdContactModel")
    canonical_with_shifts = tuple(_canonical_particle(item, domain) for item in items)
    canonical = tuple(item for item, _ in canonical_with_shifts)
    base_shifts = tuple(shift for _, shift in canonical_with_shifts)
    candidates: list[Candidate] = []
    for i, j in combinations(range(len(items)), 2):
        if _same_parent(items[i], items[j]):
            continue
        _, nearest = domain.minimum_image_displacement(canonical[i].center, canonical[j].center)
        local_shifts = set(
            _candidate_lattice_shifts_local(
                canonical[i],
                canonical[j],
                domain,
                contact_model.threshold,
                policy=contact_model.numerical_policy,
            )
        )
        local_shifts.add(nearest)
        candidates.extend(
            (
                i,
                j,
                (
                    shift[0] + base_shifts[j][0] - base_shifts[i][0],
                    shift[1] + base_shifts[j][1] - base_shifts[i][1],
                    shift[2] + base_shifts[j][2] - base_shifts[i][2],
                ),
            )
            for shift in local_shifts
        )
    return _evaluate_candidates(
        canonical,
        base_shifts,
        domain,
        contact_model,
        candidates,
        method="bruteforce",
    )


def _image_shift_ranges(
    low: np.ndarray,
    high: np.ndarray,
    window_low: np.ndarray,
    window_high: np.ndarray,
    domain: Domain,
    padding: np.ndarray,
) -> tuple[range, range, range]:
    ranges: list[range] = []
    for axis, enabled in enumerate(domain.periodic):
        if enabled:
            length = domain.size[axis]
            minimum = math.ceil((window_low[axis] - high[axis] - padding[axis]) / length)
            maximum = math.floor((window_high[axis] - low[axis] + padding[axis]) / length)
            ranges.append(range(minimum, maximum + 1))
        else:
            ranges.append(range(0, 1))
    return ranges[0], ranges[1], ranges[2]


def cell_list_contacts(
    particles: Sequence[Particle],
    domain: Domain,
    model: ThresholdContactModel | None = None,
) -> ContactSearchResult:
    """Find contacts with an unfolded periodic-AABB spatial hash."""

    items = _validate_particles(particles)
    contact_model = ThresholdContactModel() if model is None else model
    if not isinstance(contact_model, ThresholdContactModel):
        raise TypeError("model must be a ThresholdContactModel")
    if len(items) < 2:
        return ContactSearchResult((), 0, 0, "cell_list")

    sizes = np.asarray(domain.size)

    base_shifts: list[LatticeShift] = []
    canonical_particles: list[Particle] = []
    base_lows: list[np.ndarray] = []
    base_highs: list[np.ndarray] = []
    extents: list[np.ndarray] = []
    for particle in items:
        canonical, shift = _canonical_particle(particle, domain)
        low, high = aabb(canonical)
        canonical_particles.append(canonical)
        base_shifts.append(shift)
        base_lows.append(low)
        base_highs.append(high)
        extents.append(np.asarray(particle.aabb_extent))

    stacked_bounds = np.stack(base_lows + base_highs)
    maximum_extent = np.max(np.stack(extents), axis=0)
    coordinate_scale = np.maximum.reduce(
        (
            np.max(np.abs(stacked_bounds), axis=0),
            2.0 * maximum_extent,
            np.full(3, contact_model.threshold),
            np.ones(3),
        )
    )
    tolerance = np.asarray(
        [contact_model.numerical_policy.tolerance(float(value)) for value in coordinate_scale]
    )
    interaction_tolerance = contact_model.numerical_policy.tolerance(
        max(float(np.max(2.0 * maximum_extent)), contact_model.threshold, 1.0)
    )
    tolerance = np.maximum(tolerance, interaction_tolerance)
    padding = 0.5 * (contact_model.threshold + tolerance)
    expanded_lows = [low - padding for low in base_lows]
    expanded_highs = [high + padding for high in base_highs]
    window_low = np.min(np.stack(expanded_lows), axis=0)
    window_high = np.max(np.stack(expanded_highs), axis=0)
    target_bins = max(1, int(math.ceil(len(items) ** (1.0 / 3.0))))
    cell_width = np.maximum(
        sizes / target_bins,
        2.0 * maximum_extent + contact_model.threshold + tolerance,
    )
    origin = window_low

    records: list[tuple[int, LatticeShift, np.ndarray, np.ndarray]] = []
    buckets: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for index, (base_shift, low, high) in enumerate(
        zip(base_shifts, base_lows, base_highs, strict=True)
    ):
        ranges = _image_shift_ranges(low, high, window_low, window_high, domain, padding)
        image_count = math.prod(max(0, values.stop - values.start) for values in ranges)
        if image_count > MAX_LATTICE_IMAGES:
            raise GeometryError(
                f"periodic image count {image_count} exceeds the safety limit "
                f"of {MAX_LATTICE_IMAGES}"
            )
        for extra in product(*ranges):
            absolute_shift: LatticeShift = (
                base_shift[0] + int(extra[0]),
                base_shift[1] + int(extra[1]),
                base_shift[2] + int(extra[2]),
            )
            displacement = np.asarray(extra, dtype=np.float64) * sizes
            expanded_low = low + displacement - padding
            expanded_high = high + displacement + padding
            if np.any(expanded_low > window_high + tolerance) or np.any(
                expanded_high < window_low - tolerance
            ):
                continue
            record_index = len(records)
            records.append((index, absolute_shift, expanded_low, expanded_high))
            first_key = np.floor((expanded_low - origin) / cell_width).astype(np.int64)
            last_key = np.floor((expanded_high - origin) / cell_width).astype(np.int64)
            for x, y, z in product(
                range(int(first_key[0]), int(last_key[0]) + 1),
                range(int(first_key[1]), int(last_key[1]) + 1),
                range(int(first_key[2]), int(last_key[2]) + 1),
            ):
                buckets[(x, y, z)].append(record_index)

    candidates: set[Candidate] = set()
    for record_indices in buckets.values():
        for left_record, right_record in combinations(record_indices, 2):
            left_index, left_shift, left_low, left_high = records[left_record]
            right_index, right_shift, right_low, right_high = records[right_record]
            if left_index == right_index or _same_parent(items[left_index], items[right_index]):
                continue
            if np.any(left_low > right_high + tolerance) or np.any(
                right_low > left_high + tolerance
            ):
                continue
            relative: LatticeShift = (
                right_shift[0] - left_shift[0],
                right_shift[1] - left_shift[1],
                right_shift[2] - left_shift[2],
            )
            if left_index < right_index:
                key = (left_index, right_index, relative)
            else:
                key = (
                    right_index,
                    left_index,
                    (-relative[0], -relative[1], -relative[2]),
                )
            candidates.add(key)
    return _evaluate_candidates(
        tuple(canonical_particles),
        tuple(base_shifts),
        domain,
        contact_model,
        candidates,
        method="cell_list",
    )


def find_contacts(
    particles: Sequence[Particle],
    domain: Domain,
    model: ThresholdContactModel | None = None,
    *,
    method: str = "cell_list",
    backend: str | None = None,
) -> ContactSearchResult:
    """Dispatch to the optimized search or exhaustive oracle."""

    if backend is not None:
        if method != "cell_list" and method.lower().replace("-", "_") != backend.lower().replace(
            "-", "_"
        ):
            raise ValueError("method and backend select different neighbor searches")
        method = backend
    normalized = method.lower().replace("-", "_")
    if normalized in {"cell_list", "spatial_hash"}:
        return cell_list_contacts(particles, domain, model)
    if normalized in {"bruteforce", "brute_force"}:
        return bruteforce_contacts(particles, domain, model)
    raise ValueError("method must be 'cell_list' or 'bruteforce'")
