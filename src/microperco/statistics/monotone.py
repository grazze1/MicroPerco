# SPDX-License-Identifier: Apache-2.0
"""Weighted pool-adjacent-violators isotonic regression."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..exceptions import ConfigurationError


def pava(values: ArrayLike, weights: ArrayLike | None = None) -> NDArray[np.float64]:
    """Return the non-decreasing weighted least-squares fit."""

    observations = np.asarray(values, dtype=np.float64)
    if observations.ndim != 1 or observations.size == 0 or not np.all(np.isfinite(observations)):
        raise ConfigurationError("values must be a non-empty finite one-dimensional array")
    if weights is None:
        weight_values = np.ones_like(observations)
    else:
        weight_values = np.asarray(weights, dtype=np.float64)
        if weight_values.shape != observations.shape:
            raise ConfigurationError("weights must have the same shape as values")
        if not np.all(np.isfinite(weight_values)) or np.any(weight_values <= 0.0):
            raise ConfigurationError("weights must be finite and positive")
        # A common weight factor does not affect the least-squares solution.  Scaling
        # before pooling prevents otherwise-valid finite weights from overflowing when
        # adjacent blocks are combined.  Preserve extremely small positive ratios as
        # the least subnormal float so every input retains a positive block weight.
        weight_values = weight_values / np.max(weight_values)
        weight_values = np.maximum(weight_values, np.nextafter(0.0, 1.0))

    means: list[float] = []
    sums: list[float] = []
    starts: list[int] = []
    ends: list[int] = []
    for index, (value, weight) in enumerate(zip(observations, weight_values, strict=True)):
        means.append(float(value))
        sums.append(float(weight))
        starts.append(index)
        ends.append(index + 1)
        while len(means) >= 2 and means[-2] > means[-1]:
            combined_weight = sums[-2] + sums[-1]
            left_fraction = sums[-2] / combined_weight
            right_fraction = sums[-1] / combined_weight
            combined_mean = means[-2] * left_fraction + means[-1] * right_fraction
            means[-2:] = [combined_mean]
            sums[-2:] = [combined_weight]
            ends[-2:] = [ends[-1]]
            starts.pop()
    result = np.empty_like(observations)
    for mean, start, end in zip(means, starts, ends, strict=True):
        result[start:end] = mean
    return result
