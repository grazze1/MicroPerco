# SPDX-License-Identifier: Apache-2.0
"""Auditable bounded integer search for minimum-cost mixtures."""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import product

import numpy as np

from ..contact import ThresholdContactModel
from ..domain import Domain
from ..exceptions import ConfigurationError, OptimizationError
from ..generation import (
    CylinderSpec,
    ParticleSpec,
    PopulationSpec,
    SphereSpec,
    volume_fraction,
)
from ..numerics import COST_RELATIVE_TOLERANCE
from ..simulation._engine import (
    SeedLike,
    TrialEvaluator,
    normalize_seed,
    validate_nonnegative_integer,
    validate_positive_integer,
    validate_probability,
)
from ..simulation.monte_carlo import estimate_percolation_probability
from ..statistics import (
    bonferroni_per_comparison_confidence,
    clopper_pearson_lower_bound,
    clopper_pearson_upper_bound,
)
from .results import DesignEstimate, EvaluationPhase, OptimizationResult, OptimizationStatus


def _cost_tolerance(first: float, second: float) -> float:
    """Return a scale-free roundoff allowance for comparing represented costs."""

    return max(abs(first), abs(second)) * COST_RELATIVE_TOLERANCE


def _validated_candidates(
    particle_specs: Sequence[ParticleSpec],
    count_bounds: Sequence[tuple[int, int]],
    max_candidates: int,
) -> tuple[tuple[ParticleSpec, ...], list[tuple[tuple[int, ...], float]]]:
    specs = tuple(particle_specs)
    bounds = tuple(count_bounds)
    if not specs or not all(isinstance(spec, (SphereSpec, CylinderSpec)) for spec in specs):
        raise ConfigurationError("particle_specs must be a non-empty particle-spec sequence")
    if len(bounds) != len(specs):
        raise ConfigurationError("count_bounds must have one entry per particle specification")
    ranges: list[range] = []
    widths: list[int] = []
    for index, bound in enumerate(bounds):
        try:
            is_pair = isinstance(bound, Sequence) and len(bound) == 2
        except OverflowError:
            is_pair = False
        if not is_pair:
            raise ConfigurationError(f"count_bounds[{index}] must be an inclusive (low, high) pair")
        low = validate_nonnegative_integer(bound[0], f"count_bounds[{index}][0]")
        high = validate_nonnegative_integer(bound[1], f"count_bounds[{index}][1]")
        if high < low:
            raise ConfigurationError(f"count_bounds[{index}] has high below low")
        widths.append(high - low + 1)
        ranges.append(range(low, high + 1))
    limit = validate_positive_integer(max_candidates, "max_candidates")
    count = math.prod(widths)
    if count > limit:
        raise OptimizationError(
            f"bounded search contains {count} candidates, exceeding max_candidates={limit}"
        )
    candidates: list[tuple[tuple[int, ...], float]] = []
    for counts in product(*ranges):
        cost = float(sum(value * spec.cost for value, spec in zip(counts, specs, strict=True)))
        if not math.isfinite(cost):
            raise OptimizationError("candidate cost is not representable as a finite float")
        candidates.append((tuple(counts), cost))
    candidates.sort(key=lambda item: (item[1], item[0]))
    return specs, candidates


def _evaluate_design(
    domain: Domain,
    specs: tuple[ParticleSpec, ...],
    counts: tuple[int, ...],
    contact_model: ThresholdContactModel | None,
    *,
    trials: int,
    confidence: float,
    seed: np.random.SeedSequence,
    axis: int | str,
    neighbor_backend: str,
    mode: str,
    wrapped_parent: bool,
    evaluator: TrialEvaluator | None,
    phase: EvaluationPhase,
) -> DesignEstimate:
    result = estimate_percolation_probability(
        domain,
        specs,
        counts,
        contact_model,
        trials=trials,
        confidence=confidence,
        seed=seed,
        axis=axis,
        neighbor_backend=neighbor_backend,
        mode=mode,
        wrapped_parent=wrapped_parent,
        evaluator=evaluator,
    )
    material_costs = tuple(count * spec.cost for count, spec in zip(counts, specs, strict=True))
    populations = tuple(
        PopulationSpec(spec, count) for count, spec in zip(counts, specs, strict=True)
    )
    return DesignEstimate(
        counts,
        material_costs,
        float(sum(material_costs)),
        volume_fraction(populations, domain),
        result.successes,
        result.trials,
        result.probability,
        result.confidence_interval,
        result.exact_interval,
        clopper_pearson_lower_bound(result.successes, result.trials, confidence),
        clopper_pearson_upper_bound(result.successes, result.trials, confidence),
        phase,
    )


def optimize_mixture(
    domain: Domain,
    particle_specs: Sequence[ParticleSpec],
    count_bounds: Sequence[tuple[int, int]],
    contact_model: ThresholdContactModel | None = None,
    *,
    target_probability: float = 0.9,
    screening_trials: int = 250,
    certification_trials: int = 5000,
    confidence: float = 0.95,
    seed: SeedLike = None,
    axis: int | str = 0,
    neighbor_backend: str = "cell_list",
    mode: str = "face_to_face",
    wrapped_parent: bool = False,
    evaluator: TrialEvaluator | None = None,
    max_candidates: int = 100_000,
) -> OptimizationResult:
    """Search finite K-material counts with family-wise certification.

    Before sampling, error is allocated across both potential assertions for
    every bounded candidate. This makes cost-ordered adaptive stopping safe.
    """

    if not isinstance(domain, Domain):
        raise ConfigurationError("domain must be a Domain")
    specs, candidates = _validated_candidates(particle_specs, count_bounds, max_candidates)
    target = validate_probability(target_probability, "target_probability", strict=True)
    screening_count = validate_positive_integer(screening_trials, "screening_trials")
    certification_count = validate_positive_integer(certification_trials, "certification_trials")
    level = validate_probability(confidence, "confidence", strict=True)
    comparisons = 2 * len(candidates)
    per_comparison = bonferroni_per_comparison_confidence(level, comparisons)
    root_seed, seed_label = normalize_seed(seed)
    screening_seed, certification_seed = root_seed.spawn(2)

    screening_records: list[DesignEstimate] = []
    screening_cost_limit: float | None = None
    for (counts, cost), child in zip(
        candidates, screening_seed.spawn(len(candidates)), strict=True
    ):
        if screening_cost_limit is not None:
            tolerance = _cost_tolerance(cost, screening_cost_limit)
            if cost > screening_cost_limit + tolerance:
                break
        estimate = _evaluate_design(
            domain,
            specs,
            counts,
            contact_model,
            trials=screening_count,
            confidence=level,
            seed=child,
            axis=axis,
            neighbor_backend=neighbor_backend,
            mode=mode,
            wrapped_parent=wrapped_parent,
            evaluator=evaluator,
            phase="screening",
        )
        screening_records.append(estimate)
        if estimate.probability >= target and screening_cost_limit is None:
            screening_cost_limit = estimate.total_cost
    screening_estimates = tuple(screening_records)
    point_feasible = next(
        (estimate for estimate in screening_estimates if estimate.probability >= target),
        None,
    )

    certification_records: list[DesignEstimate] = []
    certified_cost_limit: float | None = None
    for (counts, cost), child in zip(
        candidates, certification_seed.spawn(len(candidates)), strict=True
    ):
        if certified_cost_limit is not None:
            tolerance = _cost_tolerance(cost, certified_cost_limit)
            if cost > certified_cost_limit + tolerance:
                break
        estimate = _evaluate_design(
            domain,
            specs,
            counts,
            contact_model,
            trials=certification_count,
            confidence=per_comparison,
            seed=child,
            axis=axis,
            neighbor_backend=neighbor_backend,
            mode=mode,
            wrapped_parent=wrapped_parent,
            evaluator=evaluator,
            phase="certification",
        )
        certification_records.append(estimate)
        if estimate.lower_bound >= target and certified_cost_limit is None:
            certified_cost_limit = estimate.total_cost
    certification_estimates = tuple(certification_records)
    certified_feasible = [
        estimate for estimate in certification_estimates if estimate.lower_bound >= target
    ]
    recommendation: DesignEstimate | None
    if not certified_feasible:
        status: OptimizationStatus = "NO_CERTIFIED_FEASIBLE"
        recommendation = point_feasible
    else:
        recommendation = min(
            certified_feasible,
            key=lambda estimate: (estimate.total_cost, estimate.particle_counts),
        )
        cheaper = [
            estimate
            for estimate in certification_estimates
            if estimate.total_cost
            < recommendation.total_cost
            - _cost_tolerance(estimate.total_cost, recommendation.total_cost)
        ]
        status = (
            "CERTIFIED_OPTIMAL"
            if all(estimate.upper_bound < target for estimate in cheaper)
            else "INCONCLUSIVE"
        )
    return OptimizationResult(
        status,
        None if recommendation is None else recommendation.particle_counts,
        None if recommendation is None else recommendation.total_cost,
        target,
        level,
        per_comparison,
        comparisons,
        max(len(screening_estimates), len(certification_estimates)),
        screening_estimates,
        certification_estimates,
        seed_label,
    )
