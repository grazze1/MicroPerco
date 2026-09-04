# SPDX-License-Identifier: Apache-2.0
"""Declarative particle, material, and population specifications."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, pi
from typing import TypeAlias

from ..exceptions import ConfigurationError
from ..numerics import positive_finite_product


def _positive_finite(value: float, name: str) -> float:
    number = float(value)
    if not isfinite(number) or number <= 0.0:
        raise ConfigurationError(f"{name} must be a positive finite number")
    return number


def _require_derived(value: float, name: str, *, positive: bool = False) -> None:
    if not isfinite(value) or (positive and value <= 0.0):
        qualifier = "positive finite" if positive else "finite"
        raise ConfigurationError(f"{name} must be representable as a {qualifier} float")


@dataclass(frozen=True, slots=True)
class MaterialSpec:
    """Material metadata used by nominal-cost inverse design."""

    name: str = "material"
    cost_per_volume: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ConfigurationError("material name must be a non-empty string")
        cost = float(self.cost_per_volume)
        if not isfinite(cost) or cost < 0.0:
            raise ConfigurationError("cost_per_volume must be finite and non-negative")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "cost_per_volume", cost)


@dataclass(frozen=True, slots=True)
class SphereSpec:
    """Rule for sampling equal-radius spheres."""

    radius: float
    material: MaterialSpec = field(default_factory=MaterialSpec)

    def __post_init__(self) -> None:
        object.__setattr__(self, "radius", _positive_finite(self.radius, "radius"))
        if not isinstance(self.material, MaterialSpec):
            raise ConfigurationError("material must be a MaterialSpec")
        try:
            volume = self.volume
        except ArithmeticError as exc:
            raise ConfigurationError(
                "volume must be representable as a positive finite float"
            ) from exc
        _require_derived(volume, "volume", positive=True)
        _require_derived(
            self.cost,
            "particle cost",
            positive=self.material.cost_per_volume > 0.0,
        )

    @property
    def volume(self) -> float:
        return positive_finite_product(4.0 / 3.0, pi, *(self.radius,) * 3)

    @property
    def cost(self) -> float:
        return self.volume * self.material.cost_per_volume


@dataclass(frozen=True, slots=True)
class CylinderSpec:
    """Rule for sampling flat cylinders with isotropic axes."""

    radius: float
    length: float
    material: MaterialSpec = field(default_factory=MaterialSpec)

    def __post_init__(self) -> None:
        object.__setattr__(self, "radius", _positive_finite(self.radius, "radius"))
        object.__setattr__(self, "length", _positive_finite(self.length, "length"))
        if not isinstance(self.material, MaterialSpec):
            raise ConfigurationError("material must be a MaterialSpec")
        try:
            volume = self.volume
        except ArithmeticError as exc:
            raise ConfigurationError(
                "volume must be representable as a positive finite float"
            ) from exc
        _require_derived(volume, "volume", positive=True)
        _require_derived(
            self.cost,
            "particle cost",
            positive=self.material.cost_per_volume > 0.0,
        )

    @property
    def volume(self) -> float:
        return positive_finite_product(pi, self.radius, self.radius, self.length)

    @property
    def cost(self) -> float:
        return self.volume * self.material.cost_per_volume


ParticleSpec: TypeAlias = SphereSpec | CylinderSpec


@dataclass(frozen=True, slots=True)
class PopulationSpec:
    """A particle sampling rule paired with a non-negative count."""

    particle_spec: ParticleSpec
    count: int
    name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.particle_spec, (SphereSpec, CylinderSpec)):
            raise ConfigurationError("particle_spec must be a SphereSpec or CylinderSpec")
        if isinstance(self.count, bool) or not isinstance(self.count, int) or self.count < 0:
            raise ConfigurationError("population count must be a non-negative integer")
        if self.name is not None:
            if not isinstance(self.name, str) or not self.name.strip():
                raise ConfigurationError("population name must be non-empty when set")
            object.__setattr__(self, "name", self.name.strip())
        _require_derived(self.nominal_volume, "population nominal volume")
        _require_derived(self.cost, "population cost")

    @property
    def spec(self) -> ParticleSpec:
        return self.particle_spec

    @property
    def nominal_volume(self) -> float:
        return self.count * self.particle_spec.volume

    @property
    def cost(self) -> float:
        return self.count * self.particle_spec.cost
