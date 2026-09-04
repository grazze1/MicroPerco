# SPDX-License-Identifier: Apache-2.0
"""Seeded random microstructure generation."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import overload

import numpy as np
from numpy.typing import NDArray

from ..domain import Domain
from ..exceptions import ConfigurationError
from ..particles import Cylinder, Particle, Sphere
from .specs import CylinderSpec, PopulationSpec, SphereSpec


@dataclass(frozen=True, slots=True)
class Microstructure(Sequence[Particle]):
    """Generated particles together with their declarative populations."""

    particles: tuple[Particle, ...]
    populations: tuple[PopulationSpec, ...]
    seed: int | tuple[int, ...] | None = None

    def __len__(self) -> int:
        return len(self.particles)

    def __iter__(self) -> Iterator[Particle]:
        return iter(self.particles)

    @overload
    def __getitem__(self, index: int) -> Particle: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[Particle, ...]: ...

    def __getitem__(self, index: int | slice) -> Particle | tuple[Particle, ...]:
        return self.particles[index]


def isotropic_directions(rng: np.random.Generator, count: int) -> NDArray[np.float64]:
    """Sample directions uniformly on the unit sphere."""

    if not isinstance(rng, np.random.Generator):
        raise ConfigurationError("rng must be a numpy.random.Generator")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ConfigurationError("count must be a non-negative integer")
    directions = rng.normal(size=(count, 3))
    norms = np.linalg.norm(directions, axis=1)
    while np.any(norms == 0.0):
        mask = norms == 0.0
        directions[mask] = rng.normal(size=(int(np.sum(mask)), 3))
        norms = np.linalg.norm(directions, axis=1)
    return np.asarray(directions / norms[:, None], dtype=np.float64)


def _seed_label(seed: int | Sequence[int] | None) -> int | tuple[int, ...] | None:
    if seed is None:
        return None
    if isinstance(seed, bool):
        raise ConfigurationError("seed must be a non-negative integer or integer sequence")
    if isinstance(seed, (int, np.integer)):
        if int(seed) < 0:
            raise ConfigurationError("seed must be non-negative")
        return int(seed)
    try:
        raw = tuple(seed)
    except TypeError as exc:
        raise ConfigurationError("seed must be an integer sequence") from exc
    if any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
        for value in raw
    ):
        raise ConfigurationError("seed must be an integer sequence")
    values = tuple(int(value) for value in raw)
    if not values or any(value < 0 for value in values):
        raise ConfigurationError("seed sequence must contain non-negative integers")
    return values


def generate_particles(
    domain: Domain,
    populations: Sequence[PopulationSpec],
    rng: np.random.Generator,
    *,
    first_particle_id: int = 0,
) -> tuple[Particle, ...]:
    """Generate particles with uniform centers and isotropic cylinder axes."""

    if not isinstance(domain, Domain):
        raise ConfigurationError("domain must be a Domain")
    if not isinstance(rng, np.random.Generator):
        raise ConfigurationError("rng must be a numpy.random.Generator")
    if (
        isinstance(first_particle_id, (bool, np.bool_))
        or not isinstance(first_particle_id, (int, np.integer))
        or int(first_particle_id) < 0
    ):
        raise ConfigurationError("first_particle_id must be non-negative")
    items = tuple(populations)
    if not all(isinstance(item, PopulationSpec) for item in items):
        raise ConfigurationError("populations must contain PopulationSpec values")
    lower = np.asarray(domain.lower)
    upper = np.asarray(domain.upper)
    particles: list[Particle] = []
    next_id = int(first_particle_id)
    for population in items:
        centers = rng.uniform(lower, upper, size=(population.count, 3))
        spec = population.particle_spec
        directions = (
            isotropic_directions(rng, population.count) if isinstance(spec, CylinderSpec) else None
        )
        for index, center in enumerate(centers):
            if isinstance(spec, SphereSpec):
                particle: Particle = Sphere(center, spec.radius, next_id, next_id)
            else:
                assert directions is not None
                particle = Cylinder(
                    center,
                    directions[index],
                    spec.length,
                    spec.radius,
                    next_id,
                    next_id,
                )
            particles.append(particle)
            next_id += 1
    return tuple(particles)


def generate_microstructure(
    domain: Domain,
    populations: Sequence[PopulationSpec],
    *,
    seed: int | Sequence[int] | None = None,
    rng: np.random.Generator | None = None,
) -> Microstructure:
    """Generate an immutable microstructure from an RNG or explicit seed."""

    if rng is not None and seed is not None:
        raise ConfigurationError("pass rng or seed, not both")
    label = _seed_label(seed)
    generator = np.random.default_rng(seed) if rng is None else rng
    population_tuple = tuple(populations)
    return Microstructure(
        generate_particles(domain, population_tuple, generator),
        population_tuple,
        label,
    )
