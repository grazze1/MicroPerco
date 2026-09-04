# SPDX-License-Identifier: Apache-2.0
"""Critical-loading estimation with nested search and independent certification."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np

from ..contact import ThresholdContactModel
from ..domain import Domain, normalize_axis
from ..exceptions import ConfigurationError
from ..generation import (
    CylinderSpec,
    ParticleSpec,
    PopulationSpec,
    SphereSpec,
    generate_particles,
    volume_fraction,
)
from ..statistics import (
    BinaryLinkModel,
    binomial_estimate,
    bonferroni_per_comparison_confidence,
    clopper_pearson_lower_bound,
    clopper_pearson_upper_bound,
    fit_logistic,
    fit_probit,
    pava,
)
from ._engine import (
    SeedLike,
    TrialEvaluator,
    make_evaluator,
    normalize_seed,
    outcome,
    validate_nonnegative_integer,
    validate_positive_integer,
    validate_probability,
)
from .results import CriticalLoadingResult, CriticalStatus, LoadingEstimate

CriticalStrategy = Literal["pava", "logistic", "probit"]
MAX_LOADING_GRID_POINTS = 100_000


def _loading_grid(
    min_count: int,
    max_count: int | None,
    loading_grid: Sequence[int] | None,
) -> tuple[int, ...]:
    minimum = validate_nonnegative_integer(min_count, "min_count")
    if loading_grid is None:
        if max_count is None:
            raise ConfigurationError("max_count is required when loading_grid is omitted")
        maximum = validate_nonnegative_integer(max_count, "max_count")
        if maximum < minimum:
            raise ConfigurationError("max_count must be at least min_count")
        point_count = maximum - minimum + 1
        if point_count > MAX_LOADING_GRID_POINTS:
            raise ConfigurationError(
                f"loading grid contains {point_count} points, exceeding the safety limit "
                f"of {MAX_LOADING_GRID_POINTS}"
            )
        return tuple(range(minimum, maximum + 1))
    validated_values: list[int] = []
    for index, value in enumerate(loading_grid):
        if index >= MAX_LOADING_GRID_POINTS:
            raise ConfigurationError(
                "loading_grid exceeds the safety limit of "
                f"{MAX_LOADING_GRID_POINTS} points"
            )
        validated_values.append(
            validate_nonnegative_integer(value, f"loading_grid[{index}]")
        )
    values = tuple(validated_values)
    if not values:
        raise ConfigurationError("loading_grid must not be empty")
    if values[0] < minimum:
        raise ConfigurationError("loading_grid values must be at least min_count")
    if any(right <= left for left, right in zip(values, values[1:], strict=False)):
        raise ConfigurationError("loading_grid must be strictly increasing")
    if max_count is not None and values[-1] > validate_nonnegative_integer(max_count, "max_count"):
        raise ConfigurationError("loading_grid values must not exceed max_count")
    return values


def _estimate(
    count: int,
    successes: int,
    trials: int,
    confidence: float,
    *,
    fitted_probability: float | None = None,
) -> LoadingEstimate:
    summary = binomial_estimate(successes, trials, confidence)
    return LoadingEstimate(
        count,
        successes,
        trials,
        summary.probability,
        summary.wilson,
        summary.clopper_pearson,
        clopper_pearson_lower_bound(successes, trials, confidence),
        clopper_pearson_upper_bound(successes, trials, confidence),
        fitted_probability,
    )


def _run_nested_search(
    domain: Domain,
    particle_spec: ParticleSpec,
    fixed_populations: tuple[PopulationSpec, ...],
    counts: tuple[int, ...],
    trials: int,
    root_seed: np.random.SeedSequence,
    evaluator: TrialEvaluator,
) -> np.ndarray:
    successes = np.zeros(len(counts), dtype=np.int64)
    variable_population = PopulationSpec(particle_spec, counts[-1])
    fixed_count = sum(population.count for population in fixed_populations)
    for child in root_seed.spawn(trials):
        rng = np.random.default_rng(child)
        fixed_particles = generate_particles(domain, fixed_populations, rng)
        variable_particles = generate_particles(
            domain,
            (variable_population,),
            rng,
            first_particle_id=fixed_count,
        )
        reached = False
        for index, count in enumerate(counts):
            if not reached:
                reached = outcome(evaluator(fixed_particles + variable_particles[:count]))
            if reached:
                successes[index] += 1
    return successes


def _run_independent_count(
    domain: Domain,
    particle_spec: ParticleSpec,
    fixed_populations: tuple[PopulationSpec, ...],
    count: int,
    trials: int,
    root_seed: np.random.SeedSequence,
    evaluator: TrialEvaluator,
) -> int:
    populations = fixed_populations + (PopulationSpec(particle_spec, count),)
    return sum(
        outcome(evaluator(generate_particles(domain, populations, np.random.default_rng(child))))
        for child in root_seed.spawn(trials)
    )


def estimate_critical_loading(
    domain: Domain,
    particle_spec: ParticleSpec,
    contact_model: ThresholdContactModel | None = None,
    *,
    min_count: int = 0,
    max_count: int | None = None,
    loading_grid: Sequence[int] | None = None,
    fixed_populations: Sequence[PopulationSpec] = (),
    target_probability: float = 0.9,
    search_trials: int = 1000,
    certification_trials: int = 5000,
    confidence: float = 0.95,
    strategy: CriticalStrategy = "pava",
    seed: SeedLike = None,
    axis: int | str = 0,
    neighbor_backend: str = "cell_list",
    mode: str = "face_to_face",
    wrapped_parent: bool = False,
    evaluator: TrialEvaluator | None = None,
) -> CriticalLoadingResult:
    """Estimate a discrete crossing and certify it at family-wise confidence.

    Search trials share nested particle prefixes. Certification is independent;
    its error is Bonferroni-allocated over the candidate lower-bound and, when
    present, predecessor upper-bound assertions.
    """

    if not isinstance(domain, Domain):
        raise ConfigurationError("domain must be a Domain")
    if not isinstance(particle_spec, (SphereSpec, CylinderSpec)):
        raise ConfigurationError("particle_spec must be a SphereSpec or CylinderSpec")
    counts = _loading_grid(min_count, max_count, loading_grid)
    fixed = tuple(fixed_populations)
    if not all(isinstance(population, PopulationSpec) for population in fixed):
        raise ConfigurationError("fixed_populations must contain PopulationSpec values")
    target = validate_probability(target_probability, "target_probability", strict=True)
    search_count = validate_positive_integer(search_trials, "search_trials")
    certification_count = validate_positive_integer(certification_trials, "certification_trials")
    level = validate_probability(confidence, "confidence", strict=True)
    if strategy not in ("pava", "logistic", "probit"):
        raise ConfigurationError("strategy must be 'pava', 'logistic', or 'probit'")
    classifier = make_evaluator(
        domain,
        contact_model,
        axis=normalize_axis(axis),
        mode=mode,
        neighbor_backend=neighbor_backend,
        wrapped_parent=wrapped_parent,
        evaluator=evaluator,
    )
    root_seed, seed_label = normalize_seed(seed)
    search_seed, certification_seed = root_seed.spawn(2)
    successes = _run_nested_search(
        domain,
        particle_spec,
        fixed,
        counts,
        search_count,
        search_seed,
        classifier,
    )
    raw_probability = successes.astype(np.float64) / search_count
    fitted_model: BinaryLinkModel | None = None
    if strategy == "pava":
        fitted_probability = pava(raw_probability, np.full(len(counts), search_count))
    elif strategy == "logistic":
        fitted_model = fit_logistic(counts, successes, search_count)
        fitted_probability = fitted_model.predict(counts)
    else:
        fitted_model = fit_probit(counts, successes, search_count)
        fitted_probability = fitted_model.predict(counts)
    search_estimates = tuple(
        _estimate(count, int(hit), search_count, level, fitted_probability=float(fitted))
        for count, hit, fitted in zip(counts, successes, fitted_probability, strict=True)
    )
    crossings = np.flatnonzero(fitted_probability >= target)
    candidate_index = int(crossings[0]) if crossings.size else None
    certification_counts: tuple[int, ...]
    if candidate_index is None:
        certification_counts = (counts[-1],)
    elif candidate_index == 0:
        certification_counts = (counts[0],)
    else:
        certification_counts = (counts[candidate_index - 1], counts[candidate_index])
    comparisons = len(certification_counts)
    per_comparison = bonferroni_per_comparison_confidence(level, comparisons)
    certification_estimates = tuple(
        _estimate(
            count,
            _run_independent_count(
                domain,
                particle_spec,
                fixed,
                count,
                certification_count,
                child,
                classifier,
            ),
            certification_count,
            per_comparison,
        )
        for count, child in zip(
            certification_counts,
            certification_seed.spawn(comparisons),
            strict=True,
        )
    )
    critical_count: int | None
    if candidate_index is None:
        critical_count = None
        status: CriticalStatus = (
            "NO_CROSSING" if certification_estimates[-1].upper_bound < target else "INCONCLUSIVE"
        )
    else:
        critical_count = counts[candidate_index]
        candidate_evidence = certification_estimates[-1]
        predecessor_excluded = (
            candidate_index == 0 or certification_estimates[0].upper_bound < target
        )
        status = (
            "CERTIFIED"
            if candidate_evidence.lower_bound >= target and predecessor_excluded
            else "INCONCLUSIVE"
        )
    fraction = (
        None
        if critical_count is None
        else volume_fraction(
            fixed + (PopulationSpec(particle_spec, critical_count),),
            domain,
        )
    )
    return CriticalLoadingResult(
        status,
        critical_count,
        fraction,
        target,
        level,
        per_comparison,
        comparisons,
        strategy,
        search_estimates,
        certification_estimates,
        fitted_model,
        seed_label,
    )
