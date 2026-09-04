# SPDX-License-Identifier: Apache-2.0
"""Immutable sphere and finite flat-cylinder geometry."""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .exceptions import GeometryError
from .numerics import MACHINE_EPSILON, positive_finite_product

FloatArray = NDArray[np.float64]
ParticleId: TypeAlias = Hashable | None
ImageOffset: TypeAlias = tuple[int, int, int] | None


def _vector3(value: ArrayLike, name: str) -> FloatArray:
    try:
        array = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise GeometryError(f"{name} must be a finite three-vector") from exc
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise GeometryError(f"{name} must be a finite three-vector")
    array.setflags(write=False)
    return array


def _identifier(value: ParticleId, name: str) -> ParticleId:
    if value is not None:
        try:
            hash(value)
        except TypeError as exc:
            raise GeometryError(f"{name} must be hashable or None") from exc
    return value


def _image_offset(value: Sequence[int] | None) -> ImageOffset:
    if value is None:
        return None
    items = tuple(value)
    if len(items) != 3 or not all(
        isinstance(item, (int, np.integer)) and not isinstance(item, (bool, np.bool_))
        for item in items
    ):
        raise GeometryError("image_offset must be None or a length-three integer sequence")
    return int(items[0]), int(items[1]), int(items[2])


def _unit_vector(value: FloatArray, name: str) -> FloatArray:
    """Normalize a finite vector without overflow or underflow."""

    component_scale = max(abs(float(component)) for component in value)
    if component_scale == 0.0:
        raise GeometryError(f"{name} must be non-zero")
    scaled = value / component_scale
    scaled_norm = math.hypot(*(float(component) for component in scaled))
    normalized = np.array(scaled / scaled_norm, dtype=np.float64, copy=True)
    normalized_norm = math.hypot(*(float(component) for component in normalized))
    if not np.all(np.isfinite(normalized)) or not math.isclose(
        normalized_norm,
        1.0,
        rel_tol=8.0 * MACHINE_EPSILON,
        abs_tol=8.0 * MACHINE_EPSILON,
    ):
        raise GeometryError(f"{name} could not be normalized reliably")
    normalized.setflags(write=False)
    return normalized


def _support_direction(value: ArrayLike) -> FloatArray | None:
    vector = _vector3(value, "direction")
    if max(abs(float(component)) for component in vector) == 0.0:
        return None
    return _unit_vector(vector, "direction")


def _orthogonal_plane_basis(axis: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Return a stable orthonormal basis for the plane normal to ``axis``."""

    helper = np.zeros(3, dtype=np.float64)
    helper[int(np.argmin(np.abs(axis)))] = 1.0
    first = np.cross(axis, helper)
    first /= math.hypot(*(float(component) for component in first))
    second = np.cross(axis, first)
    second /= math.hypot(*(float(component) for component in second))
    return np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)


def _radial_aabb_factors(axis: FloatArray) -> FloatArray:
    """Compute ``sqrt(1 - axis[i]**2)`` without cancellation near axes."""

    return np.asarray(
        (
            math.hypot(float(axis[1]), float(axis[2])),
            math.hypot(float(axis[0]), float(axis[2])),
            math.hypot(float(axis[0]), float(axis[1])),
        ),
        dtype=np.float64,
    )


@dataclass(frozen=True, slots=True, eq=False, init=False)
class Sphere:
    """A closed sphere with strictly positive radius."""

    center: FloatArray
    radius: float
    particle_id: ParticleId = None
    parent_id: ParticleId = None
    image_offset: ImageOffset = None

    def __init__(
        self,
        center: ArrayLike,
        radius: float,
        particle_id: ParticleId = None,
        parent_id: ParticleId = None,
        image_offset: Sequence[int] | None = None,
    ) -> None:
        normalized_center = _vector3(center, "center")
        normalized_radius = float(radius)
        if not math.isfinite(normalized_radius) or normalized_radius <= 0.0:
            raise GeometryError("sphere radius must be finite and positive")
        try:
            positive_finite_product(4.0 / 3.0, math.pi, *(normalized_radius,) * 3)
        except ArithmeticError as exc:
            raise GeometryError(
                "sphere volume must be representable as a positive finite float"
            ) from exc
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            lower = normalized_center - normalized_radius
            upper = normalized_center + normalized_radius
        if not np.all(np.isfinite((lower, upper))):
            raise GeometryError("sphere bounding box must be representable as finite floats")
        if not np.all((lower < normalized_center) & (normalized_center < upper)):
            raise GeometryError("sphere bounding box collapses at the requested coordinate scale")
        object.__setattr__(self, "center", normalized_center)
        object.__setattr__(self, "radius", normalized_radius)
        object.__setattr__(self, "particle_id", _identifier(particle_id, "particle_id"))
        object.__setattr__(self, "parent_id", _identifier(parent_id, "parent_id"))
        object.__setattr__(self, "image_offset", _image_offset(image_offset))

    @property
    def id(self) -> ParticleId:
        return self.particle_id

    @property
    def volume(self) -> float:
        return positive_finite_product(4.0 / 3.0, math.pi, *(self.radius,) * 3)

    @property
    def aabb_extent(self) -> FloatArray:
        result = np.full(3, self.radius, dtype=np.float64)
        result.setflags(write=False)
        return result

    def support(self, direction: ArrayLike) -> FloatArray:
        unit = _support_direction(direction)
        if unit is None:
            return np.array(self.center, copy=True)
        return np.asarray(self.center + self.radius * unit, dtype=np.float64)

    def translated(self, displacement: ArrayLike) -> Sphere:
        vector = _vector3(displacement, "displacement")
        return Sphere(
            self.center + vector,
            self.radius,
            self.particle_id,
            self.parent_id,
            self.image_offset,
        )


@dataclass(frozen=True, slots=True, eq=False, init=False)
class Cylinder:
    """A closed finite right circular cylinder with flat end caps."""

    center: FloatArray
    axis: FloatArray
    length: float
    radius: float
    particle_id: ParticleId = None
    parent_id: ParticleId = None
    image_offset: ImageOffset = None

    def __init__(
        self,
        center: ArrayLike,
        axis: ArrayLike,
        length: float,
        radius: float,
        particle_id: ParticleId = None,
        parent_id: ParticleId = None,
        image_offset: Sequence[int] | None = None,
    ) -> None:
        normalized_center = _vector3(center, "center")
        normalized_axis = _unit_vector(_vector3(axis, "axis"), "cylinder axis")
        normalized_length = float(length)
        normalized_radius = float(radius)
        if not math.isfinite(normalized_length) or normalized_length <= 0.0:
            raise GeometryError("cylinder length must be finite and positive")
        if not math.isfinite(normalized_radius) or normalized_radius <= 0.0:
            raise GeometryError("cylinder radius must be finite and positive")
        half_length = 0.5 * normalized_length
        try:
            positive_finite_product(
                math.pi, normalized_radius, normalized_radius, normalized_length
            )
        except ArithmeticError as exc:
            raise GeometryError(
                "cylinder volume must be representable as a positive finite float"
            ) from exc
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            radial_extent = normalized_radius * _radial_aabb_factors(normalized_axis)
            extent = half_length * np.abs(normalized_axis) + radial_extent
            endpoint_offset = half_length * normalized_axis
            endpoints = (
                normalized_center - endpoint_offset,
                normalized_center + endpoint_offset,
            )
            bounds = (normalized_center - extent, normalized_center + extent)
        if not (
            np.all(np.isfinite(extent))
            and np.all(np.isfinite(endpoints))
            and np.all(np.isfinite(bounds))
        ):
            raise GeometryError("cylinder endpoints and bounding box must be finite")
        if not np.all((bounds[0] < normalized_center) & (normalized_center < bounds[1])):
            raise GeometryError("cylinder bounding box collapses at the requested coordinate scale")
        represented_length = math.hypot(
            *(float(component) for component in endpoints[1] - endpoints[0])
        )
        if represented_length == 0.0:
            raise GeometryError("cylinder endpoints collapse at the requested coordinate scale")
        object.__setattr__(self, "center", normalized_center)
        object.__setattr__(self, "axis", normalized_axis)
        object.__setattr__(self, "length", normalized_length)
        object.__setattr__(self, "radius", normalized_radius)
        object.__setattr__(self, "particle_id", _identifier(particle_id, "particle_id"))
        object.__setattr__(self, "parent_id", _identifier(parent_id, "parent_id"))
        object.__setattr__(self, "image_offset", _image_offset(image_offset))

    @classmethod
    def from_endpoints(
        cls,
        start: ArrayLike,
        end: ArrayLike,
        radius: float,
        *,
        particle_id: ParticleId = None,
        parent_id: ParticleId = None,
        image_offset: Sequence[int] | None = None,
    ) -> Cylinder:
        first = _vector3(start, "start")
        second = _vector3(end, "end")
        delta = second - first
        if not np.all(np.isfinite(delta)):
            raise GeometryError("cylinder endpoint difference must be finite")
        length = math.hypot(*(float(component) for component in delta))
        if not math.isfinite(length) or length == 0.0:
            raise GeometryError("cylinder endpoints must be distinct and representable")
        center = first + 0.5 * delta
        if not np.all(np.isfinite(center)):
            raise GeometryError("cylinder endpoint midpoint must be finite")
        return cls(center, delta, length, radius, particle_id, parent_id, image_offset)

    @property
    def id(self) -> ParticleId:
        return self.particle_id

    @property
    def half_length(self) -> float:
        return 0.5 * self.length

    @property
    def endpoints(self) -> tuple[FloatArray, FloatArray]:
        offset = self.half_length * self.axis
        return self.center - offset, self.center + offset

    @property
    def volume(self) -> float:
        return positive_finite_product(math.pi, self.radius, self.radius, self.length)

    @property
    def aabb_extent(self) -> FloatArray:
        radial = self.radius * _radial_aabb_factors(self.axis)
        result = self.half_length * np.abs(self.axis) + radial
        result.setflags(write=False)
        return result

    def support(self, direction: ArrayLike) -> FloatArray:
        unit = _support_direction(direction)
        if unit is None:
            return np.array(self.center, copy=True)
        first, second = _orthogonal_plane_basis(self.axis)
        first_component = math.fsum(
            float(left) * float(right) for left, right in zip(unit, first, strict=True)
        )
        second_component = math.fsum(
            float(left) * float(right) for left, right in zip(unit, second, strict=True)
        )
        perpendicular_norm = math.hypot(first_component, second_component)
        axial_projection = math.fsum(
            float(left) * float(right) for left, right in zip(unit, self.axis, strict=True)
        )
        if axial_projection > 0.0:
            axial_offset = self.half_length * self.axis
        elif axial_projection < 0.0:
            axial_offset = -self.half_length * self.axis
        else:
            axial_offset = np.zeros(3, dtype=np.float64)
        result = self.center + axial_offset
        if perpendicular_norm > 0.0:
            radial_direction = (
                first_component * first + second_component * second
            ) / perpendicular_norm
            result = result + self.radius * radial_direction
        return np.asarray(result, dtype=np.float64)

    def translated(self, displacement: ArrayLike) -> Cylinder:
        vector = _vector3(displacement, "displacement")
        return Cylinder(
            self.center + vector,
            self.axis,
            self.length,
            self.radius,
            self.particle_id,
            self.parent_id,
            self.image_offset,
        )


Particle: TypeAlias = Sphere | Cylinder
