# SPDX-License-Identifier: Apache-2.0
"""Monte Carlo percolation-probability estimation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..contact import ThresholdContactModel
from ..domain import Domain, normalize_axis
from ..exceptions import ConfigurationError
from ..generation import ParticleSpec, PopulationSpec
from ..statistics import binomial_estimate
from ._engine import (
    SeedLike,
    TrialEvaluator,
    make_evaluator,
    normalize_populations,
    normalize_seed,
    run_one,
    validate_positive_integer,
    validate_probability,
)
from .results import MonteCarloResult


def estimate_percolation_probability(
    domain: Domain,
    particle_specs: Sequence[ParticleSpec] | ParticleSpec | None = None,
    particle_counts: Sequence[int] | int | None = None,
    contact_model: ThresholdContactModel | None = None,
    *,
    populations: Sequence[PopulationSpec] | None = None,
    axis: int | str = 0,
    trials: int = 1000,
    seed: SeedLike = None,
    neighbor_backend: str = "cell_list",
    mode: str = "face_to_face",
    wrapped_parent: bool = False,
    confidence: float = 0.95,
    evaluator: TrialEvaluator | None = None,
) -> MonteCarloResult:
    """Estimate probability from independent child SeedSequence streams."""

    if not isinstance(domain, Domain):
        raise ConfigurationError("domain must be a Domain")
    population_tuple = normalize_populations(particle_specs, particle_counts, populations)
    trial_count = validate_positive_integer(trials, "trials")
    level = validate_probability(confidence, "confidence", strict=True)
    axis_index = normalize_axis(axis)
    if mode not in ("face_to_face", "periodic_wrap"):
        raise ConfigurationError("mode must be 'face_to_face' or 'periodic_wrap'")
    if not isinstance(neighbor_backend, str) or not neighbor_backend:
        raise ConfigurationError("neighbor_backend must be a non-empty string")
    root_seed, seed_label = normalize_seed(seed)
    classifier = make_evaluator(
        domain,
        contact_model,
        axis=axis_index,
        mode=mode,
        neighbor_backend=neighbor_backend,
        wrapped_parent=wrapped_parent,
        evaluator=evaluator,
    )
    successes = sum(
        run_one(domain, population_tuple, classifier, child)
        for child in root_seed.spawn(trial_count)
    )
    estimate = binomial_estimate(successes, trial_count, level)
    return MonteCarloResult(
        successes,
        trial_count,
        estimate.probability,
        level,
        estimate.wilson,
        estimate.clopper_pearson,
        tuple(population.count for population in population_tuple),
        seed_label,
        ("x", "y", "z")[axis_index],
        mode,
        neighbor_backend,
    )


@dataclass(frozen=True, slots=True, init=False)
class MonteCarloSimulator:
    """Configured facade around probability estimation."""

    domain: Domain
    particle_specs: tuple[ParticleSpec, ...]
    contact_model: ThresholdContactModel | None
    axis: int | str
    neighbor_backend: str
    mode: str
    wrapped_parent: bool

    def __init__(
        self,
        domain: Domain,
        particle_specs: Sequence[ParticleSpec],
        contact_model: ThresholdContactModel | None = None,
        *,
        axis: int | str = 0,
        neighbor_backend: str = "cell_list",
        mode: str = "face_to_face",
        wrapped_parent: bool = False,
    ) -> None:
        specs = tuple(particle_specs)
        if not specs:
            raise ConfigurationError("particle_specs must not be empty")
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "particle_specs", specs)
        object.__setattr__(self, "contact_model", contact_model)
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "neighbor_backend", neighbor_backend)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "wrapped_parent", wrapped_parent)

    def estimate_probability(
        self,
        particle_counts: Sequence[int] | int,
        *,
        trials: int = 1000,
        seed: SeedLike = None,
        confidence: float = 0.95,
        evaluator: TrialEvaluator | None = None,
    ) -> MonteCarloResult:
        return estimate_percolation_probability(
            self.domain,
            self.particle_specs,
            particle_counts,
            self.contact_model,
            axis=self.axis,
            trials=trials,
            seed=seed,
            neighbor_backend=self.neighbor_backend,
            mode=self.mode,
            wrapped_parent=self.wrapped_parent,
            confidence=confidence,
            evaluator=evaluator,
        )
