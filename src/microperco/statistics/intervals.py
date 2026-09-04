# SPDX-License-Identifier: Apache-2.0
"""Binomial point estimates and confidence intervals."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Literal

from scipy.stats import beta, norm

from ..exceptions import ConfigurationError

IntervalSide = Literal["two-sided", "lower", "upper"]


def _validate_counts(successes: int, trials: int) -> None:
    if isinstance(successes, bool) or not isinstance(successes, int):
        raise ConfigurationError("successes must be an integer")
    if isinstance(trials, bool) or not isinstance(trials, int) or trials <= 0:
        raise ConfigurationError("trials must be a positive integer")
    if not 0 <= successes <= trials:
        raise ConfigurationError("successes must lie between zero and trials")


def _validate_confidence(confidence: float) -> float:
    value = float(confidence)
    if not isfinite(value) or not 0.0 < value < 1.0:
        raise ConfigurationError("confidence must lie strictly between zero and one")
    return value


def bonferroni_per_comparison_confidence(family_confidence: float, comparisons: int) -> float:
    """Return per-assertion confidence giving family-wise union-bound control."""

    level = _validate_confidence(family_confidence)
    if isinstance(comparisons, bool) or not isinstance(comparisons, int) or comparisons <= 0:
        raise ConfigurationError("comparisons must be a positive integer")
    result = 1.0 - (1.0 - level) / comparisons
    if result >= 1.0:
        raise ConfigurationError(
            "family confidence and comparison count exceed floating-point resolution"
        )
    return result


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    """A closed confidence interval on a probability."""

    lower: float
    upper: float
    confidence: float
    method: str
    side: IntervalSide = "two-sided"

    def __iter__(self) -> Iterator[float]:
        yield self.lower
        yield self.upper


@dataclass(frozen=True, slots=True)
class BinomialEstimate:
    """A point estimate with two-sided Wilson and exact intervals."""

    successes: int
    trials: int
    probability: float
    standard_error: float
    wilson: ConfidenceInterval
    clopper_pearson: ConfidenceInterval

    @property
    def p_hat(self) -> float:
        return self.probability


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> ConfidenceInterval:
    _validate_counts(successes, trials)
    level = _validate_confidence(confidence)
    estimate = successes / trials
    z = float(norm.ppf(0.5 + level / 2.0))
    z2 = z * z
    denominator = 1.0 + z2 / trials
    center = (estimate + z2 / (2.0 * trials)) / denominator
    half_width = (
        z * sqrt(estimate * (1.0 - estimate) / trials + z2 / (4.0 * trials * trials)) / denominator
    )
    return ConfidenceInterval(
        0.0 if successes == 0 else max(0.0, center - half_width),
        1.0 if successes == trials else min(1.0, center + half_width),
        level,
        "wilson",
    )


def clopper_pearson_interval(
    successes: int,
    trials: int,
    confidence: float = 0.95,
    *,
    side: IntervalSide = "two-sided",
) -> ConfidenceInterval:
    _validate_counts(successes, trials)
    level = _validate_confidence(confidence)
    if side not in ("two-sided", "lower", "upper"):
        raise ConfigurationError("side must be 'two-sided', 'lower', or 'upper'")
    tail = (1.0 - level) / 2.0 if side == "two-sided" else 1.0 - level
    lower = (
        0.0
        if successes == 0 or side == "upper"
        else float(beta.ppf(tail, successes, trials - successes + 1))
    )
    upper = (
        1.0
        if successes == trials or side == "lower"
        else float(beta.ppf(1.0 - tail, successes + 1, trials - successes))
    )
    return ConfidenceInterval(lower, upper, level, "clopper-pearson", side)


def clopper_pearson_lower_bound(successes: int, trials: int, confidence: float = 0.95) -> float:
    return clopper_pearson_interval(successes, trials, confidence, side="lower").lower


def clopper_pearson_upper_bound(successes: int, trials: int, confidence: float = 0.95) -> float:
    return clopper_pearson_interval(successes, trials, confidence, side="upper").upper


def binomial_estimate(successes: int, trials: int, confidence: float = 0.95) -> BinomialEstimate:
    _validate_counts(successes, trials)
    level = _validate_confidence(confidence)
    probability = successes / trials
    return BinomialEstimate(
        successes,
        trials,
        probability,
        sqrt(probability * (1.0 - probability) / trials),
        wilson_interval(successes, trials, level),
        clopper_pearson_interval(successes, trials, level),
    )
