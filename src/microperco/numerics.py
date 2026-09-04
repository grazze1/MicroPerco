# SPDX-License-Identifier: Apache-2.0
"""Central numerical policies and named algorithmic safeguards."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

FLOAT_DTYPE = np.dtype(np.float64)
MACHINE_EPSILON = float(np.finfo(FLOAT_DTYPE).eps)
EPS = MACHINE_EPSILON

# These are statistical safeguards, not geometric contact tolerances.
INITIAL_PROBABILITY_CLIP = 1.0e-8
LIKELIHOOD_PROBABILITY_CLIP = 1.0e-14
MINIMUM_INITIAL_SLOPE = 1.0e-8
COST_RELATIVE_TOLERANCE = 64.0 * MACHINE_EPSILON


@dataclass(frozen=True, slots=True)
class NumericalPolicy:
    """Length-aware tolerances shared by geometry and contact decisions."""

    absolute_tolerance: float = 1.0e-9
    relative_tolerance: float = 1.0e-12
    gjk_max_iterations: int = 128

    def __post_init__(self) -> None:
        if isinstance(self.absolute_tolerance, (bool, np.bool_)) or isinstance(
            self.relative_tolerance, (bool, np.bool_)
        ):
            raise ValueError("numerical tolerances must be finite and non-negative")
        try:
            values = (float(self.absolute_tolerance), float(self.relative_tolerance))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("numerical tolerances must be finite and non-negative") from exc
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("numerical tolerances must be finite and non-negative")
        if (
            isinstance(self.gjk_max_iterations, (bool, np.bool_))
            or not isinstance(self.gjk_max_iterations, (int, np.integer))
            or int(self.gjk_max_iterations) < 1
        ):
            raise ValueError("gjk_max_iterations must be a positive integer")
        object.__setattr__(self, "absolute_tolerance", values[0])
        object.__setattr__(self, "relative_tolerance", values[1])
        object.__setattr__(self, "gjk_max_iterations", int(self.gjk_max_iterations))

    @property
    def dtype(self) -> np.dtype[np.float64]:
        return FLOAT_DTYPE

    def tolerance(self, scale: float = 1.0) -> float:
        if not math.isfinite(scale) or scale < 0.0:
            raise ValueError("scale must be finite and non-negative")
        return max(self.absolute_tolerance, self.relative_tolerance * max(scale, 1.0))

    def less_than_or_close(self, value: float, limit: float, *, scale: float = 1.0) -> bool:
        return value <= limit + self.tolerance(scale)


DEFAULT_NUMERICAL_POLICY = NumericalPolicy()
DEFAULT_POLICY = DEFAULT_NUMERICAL_POLICY


def positive_finite_product(*factors: float) -> float:
    """Multiply positive finite floats without avoidable intermediate range errors.

    Each factor is split into a bounded mantissa and a binary exponent.  Renormalizing
    after every multiplication keeps intermediates in range even when large and small
    factors compensate.  ``ArithmeticError`` means the mathematical product cannot be
    represented as a positive finite binary64 value.
    """

    mantissa = 1.0
    exponent = 0
    for factor in factors:
        value = float(factor)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("product factors must be positive finite floats")
        factor_mantissa, factor_exponent = math.frexp(value)
        mantissa *= factor_mantissa
        mantissa, adjustment = math.frexp(mantissa)
        exponent += factor_exponent + adjustment
    try:
        result = math.ldexp(mantissa, exponent)
    except OverflowError as exc:
        raise ArithmeticError("product is not representable as a positive finite float") from exc
    if result == 0.0 or not math.isfinite(result):
        raise ArithmeticError("product is not representable as a positive finite float")
    return result
