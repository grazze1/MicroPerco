# SPDX-License-Identifier: Apache-2.0
"""Nominal particle-volume utilities."""

from __future__ import annotations

from collections.abc import Sequence
from decimal import ROUND_HALF_UP, Decimal, localcontext
from math import ceil, floor, isfinite
from typing import Any, Literal, cast

import numpy as np

from ..exceptions import ConfigurationError
from ..numerics import positive_finite_product
from ..particles import Cylinder, Sphere
from .specs import CylinderSpec, PopulationSpec, SphereSpec

RoundingMode = Literal["half_up", "ceil", "floor"]


def domain_volume(domain: object) -> float:
    try:
        size = np.asarray(cast(Any, domain).size, dtype=np.float64)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ConfigurationError("domain must expose a numeric size") from exc
    if size.shape != (3,) or not np.all(np.isfinite(size)) or np.any(size <= 0.0):
        raise ConfigurationError("domain size must contain three positive finite values")
    try:
        return positive_finite_product(*(float(value) for value in size))
    except ArithmeticError as exc:
        raise ConfigurationError("domain volume must be positive and finite") from exc


def particle_volume(particle: object) -> float:
    if isinstance(particle, (SphereSpec, CylinderSpec, Sphere, Cylinder)):
        return particle.volume
    try:
        radius = float(cast(Any, particle).radius)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ConfigurationError("particle must expose a numeric radius") from exc
    if not isfinite(radius) or radius <= 0.0:
        raise ConfigurationError("particle radius must be positive and finite")
    length_value = getattr(particle, "length", None)
    try:
        if length_value is None:
            return positive_finite_product(4.0 / 3.0, np.pi, radius, radius, radius)
        return positive_finite_product(np.pi, radius, radius, float(length_value))
    except (ArithmeticError, ValueError) as exc:
        raise ConfigurationError("particle volume must be positive and finite") from exc


def _volume_terms(
    particles_or_populations: Sequence[object],
) -> tuple[tuple[int, float], ...]:
    return tuple(
        (item.count, item.particle_spec.volume)
        if isinstance(item, PopulationSpec)
        else (1, particle_volume(item))
        for item in particles_or_populations
    )


def _weighted_volume_ratio(terms: Sequence[tuple[int, float]], denominator: float) -> float:
    """Return ``sum(count * volume) / denominator`` without range-limited intermediates."""

    with localcontext() as context:
        context.prec = 80
        divisor = Decimal.from_float(denominator)
        value = sum(
            (
                Decimal(count) * Decimal.from_float(volume) / divisor
                for count, volume in terms
            ),
            Decimal(0),
        )
        result = float(value)
    has_positive_volume = any(count > 0 for count, _ in terms)
    if not isfinite(result) or (has_positive_volume and result <= 0.0):
        raise ConfigurationError("derived volume ratio is not representable as a finite float")
    return result


def total_particle_volume(particles_or_populations: Sequence[object]) -> float:
    terms = _volume_terms(particles_or_populations)
    try:
        return _weighted_volume_ratio(terms, 1.0)
    except ConfigurationError as exc:
        raise ConfigurationError("total particle volume is not finite") from exc


def volume_fraction(particles_or_populations: Sequence[object], domain: object) -> float:
    return _weighted_volume_ratio(
        _volume_terms(particles_or_populations),
        domain_volume(domain),
    )


def particle_count_for_volume_fraction(
    target_fraction: float,
    particle_spec: SphereSpec | CylinderSpec,
    domain: object,
    *,
    rounding: RoundingMode = "half_up",
) -> int:
    fraction = float(target_fraction)
    if not isfinite(fraction) or fraction < 0.0:
        raise ConfigurationError("target_fraction must be finite and non-negative")
    if not isinstance(particle_spec, (SphereSpec, CylinderSpec)):
        raise ConfigurationError("particle_spec must be a SphereSpec or CylinderSpec")
    raw = fraction * domain_volume(domain) / particle_spec.volume
    if not isfinite(raw):
        raise ConfigurationError("requested particle count is not representable")
    if rounding == "half_up":
        return int(Decimal(str(raw)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    if rounding == "ceil":
        return int(ceil(raw))
    if rounding == "floor":
        return int(floor(raw))
    raise ConfigurationError("rounding must be 'half_up', 'ceil', or 'floor'")
