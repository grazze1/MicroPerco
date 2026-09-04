# SPDX-License-Identifier: Apache-2.0
"""Immutable inverse-design results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..seeding import SeedProvenance
from ..statistics import ConfidenceInterval

OptimizationStatus = Literal["CERTIFIED_OPTIMAL", "INCONCLUSIVE", "NO_CERTIFIED_FEASIBLE"]
EvaluationPhase = Literal["screening", "certification"]


@dataclass(frozen=True, slots=True)
class DesignEstimate:
    particle_counts: tuple[int, ...]
    material_costs: tuple[float, ...]
    total_cost: float
    nominal_volume_fraction: float
    successes: int
    trials: int
    probability: float
    confidence_interval: ConfidenceInterval
    exact_interval: ConfidenceInterval
    lower_bound: float
    upper_bound: float
    phase: EvaluationPhase

    @property
    def counts(self) -> tuple[int, ...]:
        return self.particle_counts

    @property
    def cost(self) -> float:
        return self.total_cost


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    """Result of family-wise controlled cost-ordered enumeration."""

    status: OptimizationStatus
    counts: tuple[int, ...] | None
    cost: float | None
    target_probability: float
    confidence: float
    per_comparison_confidence: float
    certification_comparisons: int
    candidates_evaluated: int
    screening_estimates: tuple[DesignEstimate, ...]
    certification_estimates: tuple[DesignEstimate, ...]
    seed: SeedProvenance

    @property
    def certified(self) -> bool:
        return self.status == "CERTIFIED_OPTIMAL"

    @property
    def familywise_confidence(self) -> float:
        return self.confidence

    @property
    def optimal_counts(self) -> tuple[int, ...] | None:
        return self.counts if self.certified else None

    @property
    def optimal_cost(self) -> float | None:
        return self.cost if self.certified else None
