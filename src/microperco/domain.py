# SPDX-License-Identifier: Apache-2.0
"""Orthorhombic simulation domain and periodic-coordinate utilities."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .numerics import MACHINE_EPSILON, positive_finite_product

FloatArray = NDArray[np.float64]


def _floor_scaled_difference(
    left: float,
    right: float,
    scale: float,
    *,
    half_offset: bool = False,
) -> int:
    """Floor an exact float ratio, using a fast path away from cell boundaries."""

    try:
        approximate = (left - right) / scale + (0.5 if half_offset else 0.0)
    except OverflowError:
        approximate = math.inf
    if math.isfinite(approximate):
        nearest = round(approximate)
        uncertainty = 16.0 * MACHINE_EPSILON * max(abs(approximate), 1.0)
        if abs(approximate - nearest) > uncertainty:
            return math.floor(approximate)
    exact = (
        Fraction.from_float(left) - Fraction.from_float(right)
    ) / Fraction.from_float(scale)
    if half_offset:
        exact += Fraction(1, 2)
    return exact.numerator // exact.denominator


def _scaled_lattice_component(shift: int, size: float) -> float:
    """Multiply an arbitrary Python lattice integer by a binary64 cell size."""

    try:
        result = float(shift) * size
    except OverflowError:
        result = math.inf
    if not math.isfinite(result):
        try:
            result = float(shift * Fraction.from_float(size))
        except OverflowError as exc:
            raise ValueError("lattice translation is not representable") from exc
    if not math.isfinite(result):
        raise ValueError("lattice translation is not representable")
    return result


def _shifted_coordinate(coordinate: float, shift: int, size: float) -> float:
    """Translate one float coordinate by an integer lattice shift with one rounding."""

    if shift == 0:
        return coordinate
    exact = Fraction.from_float(coordinate) + shift * Fraction.from_float(size)
    try:
        result = float(exact)
    except OverflowError as exc:
        raise ValueError("lattice translation is not representable") from exc
    if not math.isfinite(result):
        raise ValueError("lattice translation is not representable")
    return result


def _float3(value: float | Sequence[float], name: str) -> tuple[float, float, float]:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite scalar or length-three sequence") from exc
    if array.ndim == 0:
        array = np.repeat(array, 3)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite scalar or length-three sequence")
    return (float(array[0]), float(array[1]), float(array[2]))


def _bool3(value: bool | Sequence[bool]) -> tuple[bool, bool, bool]:
    if isinstance(value, (bool, np.bool_)):
        return (bool(value),) * 3
    try:
        items = tuple(value)
    except TypeError as exc:
        raise ValueError("periodic must be a bool or length-three bool sequence") from exc
    if len(items) != 3 or not all(isinstance(item, (bool, np.bool_)) for item in items):
        raise ValueError("periodic must be a bool or length-three bool sequence")
    return bool(items[0]), bool(items[1]), bool(items[2])


def normalize_axis(axis: int | str) -> int:
    """Normalize an integer or x/y/z axis label."""

    if isinstance(axis, str):
        try:
            return {"x": 0, "y": 1, "z": 2}[axis.lower()]
        except KeyError as exc:
            raise ValueError("axis must be 0, 1, 2, 'x', 'y', or 'z'") from exc
    if isinstance(axis, (int, np.integer)) and not isinstance(axis, (bool, np.bool_)):
        if int(axis) in (0, 1, 2):
            return int(axis)
    raise ValueError("axis must be 0, 1, 2, 'x', 'y', or 'z'")


@dataclass(frozen=True, slots=True, init=False)
class Domain:
    """A centered orthorhombic domain with independently periodic axes."""

    size: tuple[float, float, float]
    periodic: tuple[bool, bool, bool]
    center: tuple[float, float, float]

    def __init__(
        self,
        size: float | Sequence[float],
        periodic: bool | Sequence[bool],
        center: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> None:
        normalized_size = _float3(size, "size")
        if not all(value > 0.0 for value in normalized_size):
            raise ValueError("all domain sizes must be positive")
        normalized_center = _float3(center, "center")
        try:
            positive_finite_product(*normalized_size)
        except ArithmeticError as exc:
            raise ValueError(
                "domain volume must be representable as a positive finite float"
            ) from exc
        bounds = tuple(
            coordinate + sign * 0.5 * length
            for coordinate, length in zip(normalized_center, normalized_size, strict=True)
            for sign in (-1.0, 1.0)
        )
        if not all(math.isfinite(bound) for bound in bounds):
            raise ValueError("domain bounds must be representable as finite floats")
        if not all(
            bounds[2 * axis] < normalized_center[axis] < bounds[2 * axis + 1] for axis in range(3)
        ):
            raise ValueError("domain bounds collapse at the requested coordinate scale")
        object.__setattr__(self, "size", normalized_size)
        object.__setattr__(self, "periodic", _bool3(periodic))
        object.__setattr__(self, "center", normalized_center)

    @property
    def lower(self) -> tuple[float, float, float]:
        return (
            self.center[0] - 0.5 * self.size[0],
            self.center[1] - 0.5 * self.size[1],
            self.center[2] - 0.5 * self.size[2],
        )

    @property
    def upper(self) -> tuple[float, float, float]:
        return (
            self.center[0] + 0.5 * self.size[0],
            self.center[1] + 0.5 * self.size[1],
            self.center[2] + 0.5 * self.size[2],
        )

    @property
    def volume(self) -> float:
        return positive_finite_product(*self.size)

    @property
    def periodic_axes(self) -> tuple[int, ...]:
        return tuple(index for index, enabled in enumerate(self.periodic) if enabled)

    def with_periodic(self, periodic: bool | Sequence[bool]) -> Domain:
        return Domain(self.size, periodic, self.center)

    def lattice_vector(self, shift: Sequence[int]) -> FloatArray:
        try:
            lattice = tuple(shift)
        except TypeError as exc:
            raise ValueError("shift must be a length-three integer sequence") from exc
        if len(lattice) != 3 or not all(
            isinstance(item, (int, np.integer)) and not isinstance(item, (bool, np.bool_))
            for item in lattice
        ):
            raise ValueError("shift must be a length-three integer sequence")
        return np.asarray(
            [
                _scaled_lattice_component(int(item), size)
                for item, size in zip(lattice, self.size, strict=True)
            ],
            dtype=np.float64,
        )

    def canonical_shift(self, point: ArrayLike) -> tuple[int, int, int]:
        array = np.asarray(point, dtype=np.float64)
        if array.shape != (3,) or not np.all(np.isfinite(array)):
            raise ValueError("point must be a finite three-vector")
        lower = self.lower
        shift = [0, 0, 0]
        for axis, enabled in enumerate(self.periodic):
            if enabled:
                shift[axis] = -_floor_scaled_difference(
                    float(array[axis]),
                    lower[axis],
                    self.size[axis],
                )
        return shift[0], shift[1], shift[2]

    def wrap(self, points: ArrayLike) -> FloatArray:
        array = np.asarray(points, dtype=np.float64)
        if array.shape[-1:] != (3,) or not np.all(np.isfinite(array)):
            raise ValueError("points must be finite and have final dimension three")
        result = cast(FloatArray, array.copy())
        lower = self.lower
        upper = self.upper
        for axis, enabled in enumerate(self.periodic):
            if enabled:
                coordinates = result[..., axis].reshape(-1)
                for index in range(coordinates.size):
                    coordinate = float(coordinates[index])
                    shift = -_floor_scaled_difference(
                        coordinate,
                        lower[axis],
                        self.size[axis],
                    )
                    wrapped = _shifted_coordinate(coordinate, shift, self.size[axis])
                    if wrapped < lower[axis] or wrapped >= upper[axis]:
                        wrapped = lower[axis]
                    coordinates[index] = wrapped
        return result

    def minimum_image_displacement(
        self, first: ArrayLike, second: ArrayLike
    ) -> tuple[FloatArray, tuple[int, int, int]]:
        first_array = np.asarray(first, dtype=np.float64)
        second_array = np.asarray(second, dtype=np.float64)
        if (
            first_array.shape != (3,)
            or second_array.shape != (3,)
            or not np.all(np.isfinite(first_array))
            or not np.all(np.isfinite(second_array))
        ):
            raise ValueError("first and second must be finite three-vectors")
        with np.errstate(over="ignore", invalid="ignore"):
            raw_delta = second_array - first_array
        delta = np.array(raw_delta, copy=True)
        lattice = [0, 0, 0]
        for axis, enabled in enumerate(self.periodic):
            if enabled:
                first_value = float(first_array[axis])
                second_value = float(second_array[axis])
                approximate = float(raw_delta[axis]) / self.size[axis] + 0.5
                if math.isfinite(approximate) and abs(approximate) <= 2.0**32:
                    lattice[axis] = -math.floor(approximate)
                    delta[axis] = raw_delta[axis] + lattice[axis] * self.size[axis]
                else:
                    lattice[axis] = -_floor_scaled_difference(
                        second_value,
                        first_value,
                        self.size[axis],
                        half_offset=True,
                    )
                    exact_delta = (
                        Fraction.from_float(second_value)
                        - Fraction.from_float(first_value)
                        + lattice[axis] * Fraction.from_float(self.size[axis])
                    )
                    try:
                        delta[axis] = float(exact_delta)
                    except OverflowError as exc:
                        raise ValueError("point separation is not representable") from exc
        if not np.all(np.isfinite(delta)):
            raise ValueError("point separation is not representable")
        return delta, (lattice[0], lattice[1], lattice[2])
