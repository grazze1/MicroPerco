# SPDX-License-Identifier: Apache-2.0
"""Logistic and probit models for aggregated binomial observations."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize
from scipy.special import expit, ndtr, ndtri

from ..exceptions import ConfigurationError
from ..numerics import (
    INITIAL_PROBABILITY_CLIP,
    LIKELIHOOD_PROBABILITY_CLIP,
    MINIMUM_INITIAL_SLOPE,
)

LinkName = Literal["logistic", "probit"]


@dataclass(frozen=True, slots=True)
class BinaryLinkModel:
    """A fitted binomial response curve in an explicit standardized coordinate.

    ``intercept`` and ``slope`` parameterize ``z = (x - x_center) / x_scale``.
    Keeping that coordinate in the public model makes evaluation stable and every
    parameter finite even when coefficients in the original x units are not
    representable.
    """

    link: LinkName
    intercept: float
    slope: float
    log_likelihood: float
    aic: float
    bic: float
    converged: bool
    observations: int
    x_center: float = 0.0
    x_scale: float = 1.0

    def __post_init__(self) -> None:
        parameters = (
            self.intercept,
            self.slope,
            self.log_likelihood,
            self.aic,
            self.bic,
            self.x_center,
            self.x_scale,
        )
        if not all(isfinite(value) for value in parameters) or self.x_scale <= 0.0:
            raise ConfigurationError("model parameters must be finite and x_scale must be positive")

    def predict(self, x: ArrayLike) -> NDArray[np.float64]:
        values = np.asarray(x, dtype=np.float64)
        normalized = (values - self.x_center) / self.x_scale
        eta = self.intercept + self.slope * normalized
        function = expit if self.link == "logistic" else ndtr
        return np.asarray(function(eta), dtype=np.float64)

    def threshold(self, probability: float) -> float:
        target = float(probability)
        if not isfinite(target) or not 0.0 < target < 1.0:
            raise ConfigurationError("probability must lie strictly between zero and one")
        if self.slope <= 0.0:
            raise ConfigurationError("a non-positive fitted slope has no increasing threshold")
        transformed = (
            np.log(target / (1.0 - target)) if self.link == "logistic" else float(ndtri(target))
        )
        return float(self.x_center + self.x_scale * (transformed - self.intercept) / self.slope)


def _validated_observations(
    x: ArrayLike, successes: ArrayLike, trials: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    loading = np.asarray(x, dtype=np.float64)
    hit = np.asarray(successes, dtype=np.float64)
    total = np.asarray(trials, dtype=np.float64)
    if loading.ndim != 1 or loading.size < 2:
        raise ConfigurationError("x must be one-dimensional with at least two values")
    if hit.shape != loading.shape:
        raise ConfigurationError("successes must have the same shape as x")
    if total.ndim == 0:
        total = np.full_like(loading, float(total))
    if total.shape != loading.shape:
        raise ConfigurationError("trials must be scalar or have the same shape as x")
    if not np.all(np.isfinite(loading)) or np.min(loading) >= np.max(loading):
        raise ConfigurationError("x must be finite and contain distinct values")
    if not np.all(np.isfinite(hit)) or not np.all(np.isfinite(total)):
        raise ConfigurationError("binomial counts must be finite")
    if np.any(total <= 0.0) or np.any(hit < 0.0) or np.any(hit > total):
        raise ConfigurationError("binomial counts must satisfy 0 <= successes <= trials")
    if np.any(hit != np.floor(hit)) or np.any(total != np.floor(total)):
        raise ConfigurationError("binomial counts must be integers")
    return loading, hit, total


def fit_binomial_link(
    x: ArrayLike,
    successes: ArrayLike,
    trials: ArrayLike,
    *,
    link: LinkName = "logistic",
    monotone: bool = True,
) -> BinaryLinkModel:
    """Fit a logistic or probit aggregated-binomial likelihood."""

    if link not in ("logistic", "probit"):
        raise ConfigurationError("link must be 'logistic' or 'probit'")
    loading, hit, total = _validated_observations(x, successes, trials)
    lower = float(np.min(loading))
    upper = float(np.max(loading))
    center = lower / 2.0 + upper / 2.0
    scale = float(np.max(np.abs(loading - center)))
    normalized_loading = (loading - center) / scale
    smoothed = np.clip(
        (hit + 0.5) / (total + 1.0),
        INITIAL_PROBABILITY_CLIP,
        1.0 - INITIAL_PROBABILITY_CLIP,
    )
    transformed = np.log(smoothed / (1.0 - smoothed)) if link == "logistic" else ndtri(smoothed)
    initial_slope, initial_intercept = np.polyfit(normalized_loading, transformed, 1)
    if monotone:
        initial_slope = max(float(initial_slope), MINIMUM_INITIAL_SLOPE)

    def negative_log_likelihood(parameters: NDArray[np.float64]) -> float:
        eta = parameters[0] + parameters[1] * normalized_loading
        probability = expit(eta) if link == "logistic" else ndtr(eta)
        probability = np.clip(
            probability,
            LIKELIHOOD_PROBABILITY_CLIP,
            1.0 - LIKELIHOOD_PROBABILITY_CLIP,
        )
        return float(-np.sum(hit * np.log(probability) + (total - hit) * np.log1p(-probability)))

    result = minimize(
        negative_log_likelihood,
        np.array([initial_intercept, initial_slope], dtype=np.float64),
        method="L-BFGS-B",
        bounds=((None, None), (0.0, None)) if monotone else None,
    )
    scaled_intercept, scaled_slope = (float(value) for value in result.x)
    log_likelihood = -negative_log_likelihood(result.x)
    observations = int(np.sum(total))
    return BinaryLinkModel(
        link,
        scaled_intercept,
        scaled_slope,
        log_likelihood,
        4.0 - 2.0 * log_likelihood,
        2.0 * np.log(observations) - 2.0 * log_likelihood,
        bool(result.success),
        observations,
        center,
        scale,
    )


def fit_logistic(
    x: ArrayLike, successes: ArrayLike, trials: ArrayLike, *, monotone: bool = True
) -> BinaryLinkModel:
    return fit_binomial_link(x, successes, trials, link="logistic", monotone=monotone)


def fit_probit(
    x: ArrayLike, successes: ArrayLike, trials: ArrayLike, *, monotone: bool = True
) -> BinaryLinkModel:
    return fit_binomial_link(x, successes, trials, link="probit", monotone=monotone)
