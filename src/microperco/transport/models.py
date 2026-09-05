# SPDX-License-Identifier: Apache-2.0
"""Validated, finite-range junction conductance laws."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from ..exceptions import ConfigurationError, SimulationError
from ..numerics import DEFAULT_NUMERICAL_POLICY, NumericalPolicy


def finite_real(value: float, name: str, *, positive: bool = True) -> float:
    if isinstance(value, bool):
        raise ConfigurationError(f"{name} must be a finite real number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConfigurationError(f"{name} must be a finite real number") from exc
    if not math.isfinite(number) or (number <= 0 if positive else number < 0):
        qualifier = "positive" if positive else "non-negative"
        raise ConfigurationError(f"{name} must be finite and {qualifier}")
    return number


class ConductanceModel(Protocol):
    @property
    def cutoff(self) -> float: ...

    @property
    def numerical_policy(self) -> NumericalPolicy: ...

    def conductance(self, gap: float) -> float: ...


@dataclass(frozen=True, slots=True)
class ConstantConductanceModel:
    """A constant junction conductance for surface gaps <= cutoff."""

    contact_conductance: float = 1.0
    cutoff: float = 0.0
    numerical_policy: NumericalPolicy = DEFAULT_NUMERICAL_POLICY

    def __post_init__(self) -> None:
        for name in ("contact_conductance", "cutoff"):
            object.__setattr__(
                self, name, finite_real(getattr(self, name), name, positive=name != "cutoff")
            )
        if not isinstance(self.numerical_policy, NumericalPolicy):
            raise ConfigurationError("numerical_policy must be a NumericalPolicy")

    def conductance(self, gap: float) -> float:
        gap = finite_real(gap, "gap", positive=False)
        return self.contact_conductance if gap <= self.cutoff else 0.0


@dataclass(frozen=True, slots=True)
class TunnelingConductanceModel:
    """g(d) = contact_conductance * exp(-2*d/decay_length), truncated at cutoff.

    Overlapping particles have zero surface gap. A zero computed conductance
    inside the cutoff raises rather than silently deleting a physical bond.
    """

    contact_conductance: float = 1.0
    decay_length: float = 0.1
    cutoff: float = 1.0
    numerical_policy: NumericalPolicy = DEFAULT_NUMERICAL_POLICY

    def __post_init__(self) -> None:
        for name in ("contact_conductance", "decay_length", "cutoff"):
            object.__setattr__(
                self, name, finite_real(getattr(self, name), name, positive=name != "cutoff")
            )
        if not isinstance(self.numerical_policy, NumericalPolicy):
            raise ConfigurationError("numerical_policy must be a NumericalPolicy")
        self.conductance(self.cutoff)

    def conductance(self, gap: float) -> float:
        gap = finite_real(gap, "gap", positive=False)
        if gap > self.cutoff:
            return 0.0
        exponent = 2.0 * (gap / self.decay_length)
        # Log space also handles large g0 combined with a very small exponential.
        result = (
            self.contact_conductance
            if gap == 0
            else math.exp(math.log(self.contact_conductance) - exponent)
        )
        if result <= 0.0 or not math.isfinite(result):
            raise SimulationError("tunneling conductance is not representable; reduce cutoff")
        return result
