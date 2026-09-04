# SPDX-License-Identifier: Apache-2.0
"""Threshold contact model and immutable search records."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..domain import Domain
from ..geometry.distance import distance
from ..geometry.periodic import LatticeShift, periodic_distance
from ..numerics import DEFAULT_NUMERICAL_POLICY, NumericalPolicy
from ..particles import Particle


@dataclass(frozen=True, slots=True)
class ThresholdContactModel:
    """Particles conduct when their surface gap is at most ``threshold``."""

    threshold: float = 0.0
    numerical_policy: NumericalPolicy = DEFAULT_NUMERICAL_POLICY

    def __post_init__(self) -> None:
        value = float(self.threshold)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("contact threshold must be finite and non-negative")
        if not isinstance(self.numerical_policy, NumericalPolicy):
            raise TypeError("numerical_policy must be a NumericalPolicy")
        object.__setattr__(self, "threshold", value)

    @property
    def policy(self) -> NumericalPolicy:
        return self.numerical_policy

    def accepts(self, gap: float, *, scale: float = 1.0) -> bool:
        value = float(gap)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("gap must be finite and non-negative")
        return self.numerical_policy.less_than_or_close(value, self.threshold, scale=scale)

    def pair_scale(self, first: Particle, second: Particle) -> float:
        """Return the local length scale for one concrete particle-image query."""

        delta = np.asarray(second.center) - np.asarray(first.center)
        separation = math.hypot(*(float(component) for component in delta))
        return max(
            separation,
            float(np.max(first.aabb_extent)),
            float(np.max(second.aabb_extent)),
            self.threshold,
            1.0,
        )

    def accepts_pair(self, gap: float, first: Particle, second: Particle) -> bool:
        """Apply the threshold using only the queried pair's local length scale."""

        return self.accepts(gap, scale=self.pair_scale(first, second))

    def in_contact(self, first: Particle, second: Particle, domain: Domain | None = None) -> bool:
        if domain is None:
            gap = distance(first, second, policy=self.numerical_policy)
            evaluated_second = second
        else:
            result = periodic_distance(first, second, domain, policy=self.numerical_policy)
            gap = result.distance
            evaluated_second = second.translated(domain.lattice_vector(result.lattice_shift))
        return self.accepts_pair(gap, first, evaluated_second)


@dataclass(frozen=True, slots=True, order=True)
class ContactEdge:
    """A contact edge; the lattice shift is applied to endpoint ``j``."""

    i: int
    j: int
    lattice_shift: LatticeShift
    distance: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.i, bool)
            or isinstance(self.j, bool)
            or not isinstance(self.i, int)
            or not isinstance(self.j, int)
            or self.i < 0
            or self.j <= self.i
        ):
            raise ValueError("contact edge endpoints must satisfy 0 <= i < j")
        shift = tuple(self.lattice_shift)
        if len(shift) != 3 or not all(isinstance(value, int) for value in shift):
            raise ValueError("lattice_shift must contain three integers")
        value = float(self.distance)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("contact distance must be finite and non-negative")
        object.__setattr__(self, "lattice_shift", (shift[0], shift[1], shift[2]))
        object.__setattr__(self, "distance", value)

    @property
    def source(self) -> int:
        return self.i

    @property
    def target(self) -> int:
        return self.j

    @property
    def gap(self) -> float:
        return self.distance


@dataclass(frozen=True, slots=True)
class ContactSearchResult:
    """Accepted edges and auditable broad-/narrow-phase counters."""

    edges: tuple[ContactEdge, ...]
    candidate_pairs: int
    distance_evaluations: int
    method: str

    @property
    def contact_count(self) -> int:
        return len(self.edges)

    @property
    def particle_pairs(self) -> int:
        return len({(edge.i, edge.j) for edge in self.edges})
