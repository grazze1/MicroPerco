# SPDX-License-Identifier: Apache-2.0
"""Internal validation and single-realization helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypeAlias

import numpy as np

from ..contact import ThresholdContactModel
from ..domain import Domain
from ..exceptions import ConfigurationError, SimulationError
from ..generation import (
    CylinderSpec,
    ParticleSpec,
    PopulationSpec,
    SphereSpec,
    generate_particles,
)
from ..particles import Particle
from ..seeding import SeedProvenance, SeedSequenceState

SeedLike: TypeAlias = int | Sequence[int] | np.random.SeedSequence | None


class TrialEvaluator(Protocol):
    def __call__(self, particles: Sequence[Particle], /) -> bool | object: ...


def validate_positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) <= 0:
        raise ConfigurationError(f"{name} must be a positive integer")
    return int(value)


def validate_nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 0:
        raise ConfigurationError(f"{name} must be a non-negative integer")
    return int(value)


def validate_probability(value: float, name: str, *, strict: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ConfigurationError(f"{name} must be a real probability")
    number = float(value)
    valid = 0.0 < number < 1.0 if strict else 0.0 <= number <= 1.0
    if not np.isfinite(number) or not valid:
        qualifier = "strictly " if strict else ""
        raise ConfigurationError(f"{name} must lie {qualifier}between zero and one")
    return number


def normalize_seed(
    seed: SeedLike,
) -> tuple[np.random.SeedSequence, SeedProvenance]:
    if isinstance(seed, np.random.SeedSequence):
        entropy = seed.entropy
        if isinstance(entropy, (int, np.integer)):
            normalized_entropy: int | tuple[int, ...] = int(entropy)
        elif entropy is None:
            raise SimulationError("SeedSequence did not expose reproducible entropy")
        else:
            normalized_entropy = tuple(int(value) for value in entropy)
        label = SeedSequenceState(
            normalized_entropy,
            tuple(int(value) for value in seed.spawn_key),
            int(seed.pool_size),
        )
        # SeedSequence.spawn mutates only its child counter. Clone the public
        # initialization state so accepting a SeedSequence remains a pure,
        # reproducible API operation even when the same object is reused.
        cloned = np.random.SeedSequence(
            entropy,
            spawn_key=seed.spawn_key,
            pool_size=seed.pool_size,
        )
        return cloned, label
    if seed is None:
        return np.random.SeedSequence(), None
    if isinstance(seed, bool):
        raise ConfigurationError("seed must be an integer or integer sequence")
    if isinstance(seed, (int, np.integer)):
        value = int(seed)
        if value < 0:
            raise ConfigurationError("seed integers must be non-negative")
        return np.random.SeedSequence(value), value
    try:
        raw = tuple(seed)
    except TypeError as exc:
        raise ConfigurationError("seed must be an integer or integer sequence") from exc
    if not raw or any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
        for value in raw
    ):
        raise ConfigurationError("seed sequences must contain non-negative integers")
    values = tuple(int(value) for value in raw)
    if any(value < 0 for value in values):
        raise ConfigurationError("seed sequences must contain non-negative integers")
    return np.random.SeedSequence(values), values


def normalize_populations(
    particle_specs: Sequence[ParticleSpec] | ParticleSpec | None,
    particle_counts: Sequence[int] | int | None,
    populations: Sequence[PopulationSpec] | None,
) -> tuple[PopulationSpec, ...]:
    if populations is not None:
        if particle_specs is not None or particle_counts is not None:
            raise ConfigurationError("pass populations or particle_specs/particle_counts, not both")
        result = tuple(populations)
        if not result or not all(isinstance(item, PopulationSpec) for item in result):
            raise ConfigurationError("populations must be a non-empty PopulationSpec sequence")
        return result
    if particle_specs is None or particle_counts is None:
        raise ConfigurationError(
            "particle_specs and particle_counts are required when populations is omitted"
        )
    specs = (
        (particle_specs,)
        if isinstance(particle_specs, (SphereSpec, CylinderSpec))
        else tuple(particle_specs)
    )
    counts = (
        (validate_nonnegative_integer(particle_counts, "particle_counts"),)
        if isinstance(particle_counts, (int, np.integer))
        else tuple(particle_counts)
    )
    if not specs or len(specs) != len(counts):
        raise ConfigurationError(
            "particle_specs and particle_counts must have equal non-zero length"
        )
    return tuple(
        PopulationSpec(
            spec,
            validate_nonnegative_integer(count, f"particle_counts[{index}]"),
        )
        for index, (spec, count) in enumerate(zip(specs, counts, strict=True))
    )


def make_evaluator(
    domain: Domain,
    contact_model: ThresholdContactModel | None,
    *,
    axis: int | str,
    mode: str,
    neighbor_backend: str,
    wrapped_parent: bool,
    evaluator: TrialEvaluator | None,
) -> TrialEvaluator:
    if evaluator is not None:
        if not callable(evaluator):
            raise ConfigurationError("evaluator must be callable")
        return evaluator

    def analyze(particles: Sequence[Particle]) -> object:
        from ..percolation import analyze_percolation

        return analyze_percolation(
            particles,
            domain,
            contact_model,
            axis=axis,
            mode=mode,
            search=neighbor_backend,
            wrapped_parent=wrapped_parent,
        )

    return analyze


def outcome(value: bool | object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if hasattr(value, "percolates"):
        return bool(value.percolates)
    if hasattr(value, "conductive"):
        return bool(value.conductive)
    raise SimulationError(
        "trial evaluator must return bool or an object exposing percolates/conductive"
    )


def run_one(
    domain: Domain,
    populations: Sequence[PopulationSpec],
    evaluator: TrialEvaluator,
    seed_sequence: np.random.SeedSequence,
) -> bool:
    particles = generate_particles(domain, populations, np.random.default_rng(seed_sequence))
    return outcome(evaluator(particles))
