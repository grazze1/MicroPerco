# SPDX-License-Identifier: Apache-2.0
"""Immutable result objects for simulation workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..seeding import SeedProvenance
from ..statistics import BinaryLinkModel, ConfidenceInterval

CriticalStatus = Literal["CERTIFIED", "INCONCLUSIVE", "NO_CROSSING"]


@dataclass(frozen=True, slots=True)
class MonteCarloResult:
    successes: int
    trials: int
    probability: float
    confidence: float
    confidence_interval: ConfidenceInterval
    exact_interval: ConfidenceInterval
    particle_counts: tuple[int, ...]
    seed: SeedProvenance
    axis: str
    mode: str
    neighbor_backend: str

    @property
    def p_hat(self) -> float:
        return self.probability

    @property
    def percolation_probability(self) -> float:
        return self.probability


@dataclass(frozen=True, slots=True)
class LoadingEstimate:
    count: int
    successes: int
    trials: int
    probability: float
    confidence_interval: ConfidenceInterval
    exact_interval: ConfidenceInterval
    lower_bound: float
    upper_bound: float
    fitted_probability: float | None = None


@dataclass(frozen=True, slots=True)
class CriticalLoadingResult:
    """Estimated and, when possible, family-wise certified loading."""

    status: CriticalStatus
    critical_count: int | None
    critical_volume_fraction: float | None
    target_probability: float
    confidence: float
    per_comparison_confidence: float
    certification_comparisons: int
    strategy: str
    search_estimates: tuple[LoadingEstimate, ...]
    certification_estimates: tuple[LoadingEstimate, ...]
    fitted_model: BinaryLinkModel | None
    seed: SeedProvenance

    @property
    def certified(self) -> bool:
        return self.status == "CERTIFIED"

    @property
    def familywise_confidence(self) -> float:
        return self.confidence


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    problem_size: int
    backend: str
    repeats: int
    median_seconds: float
    first_quartile_seconds: float
    third_quartile_seconds: float
    spread_seconds: float
    candidate_pairs: int | None = None
    distance_evaluations: int | None = None
    speedup: float | None = None
