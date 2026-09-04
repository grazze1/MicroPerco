# SPDX-License-Identifier: Apache-2.0
"""Optional independent geometry references used by validation."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.optimize import minimize

from ..exceptions import GeometryError
from ..particles import Cylinder


def _inside_constraints(point: np.ndarray, cylinder: Cylinder) -> tuple[float, float]:
    relative = point - cylinder.center
    axial = float(np.dot(relative, cylinder.axis))
    radial = relative - axial * cylinder.axis
    return (
        cylinder.half_length * cylinder.half_length - axial * axial,
        cylinder.radius * cylinder.radius - float(np.dot(radial, radial)),
    )


def cylinder_distance_scipy(first: Cylinder, second: Cylinder) -> float:
    """Reference distance from a constrained SciPy SLSQP calculation."""

    def objective(values: np.ndarray) -> float:
        difference = values[:3] - values[3:]
        return float(np.dot(difference, difference))

    def constraints(values: np.ndarray) -> np.ndarray:
        return np.asarray(
            _inside_constraints(values[:3], first) + _inside_constraints(values[3:], second),
            dtype=np.float64,
        )

    starts = [
        np.concatenate((first.center, second.center)),
        np.concatenate((first.endpoints[0], second.endpoints[0])),
        np.concatenate((first.endpoints[1], second.endpoints[1])),
        np.concatenate((first.endpoints[0], second.endpoints[1])),
        np.concatenate((first.endpoints[1], second.endpoints[0])),
    ]
    best = math.inf
    for start in starts:
        result = minimize(
            objective,
            start,
            method="SLSQP",
            constraints={"type": "ineq", "fun": constraints},
            options={"ftol": 1.0e-14, "maxiter": 1000},
        )
        if result.success and float(np.min(constraints(result.x))) >= -1.0e-7:
            best = min(best, max(0.0, float(result.fun)))
    if not math.isfinite(best):
        raise GeometryError("SciPy reference optimizer did not converge")
    return math.sqrt(best)


def _rotation_from_z(axis: np.ndarray) -> np.ndarray:
    helper = np.array((1.0, 0.0, 0.0)) if abs(float(axis[0])) < 0.9 else np.array((0.0, 1.0, 0.0))
    first = np.cross(helper, axis)
    first /= np.linalg.norm(first)
    second = np.cross(axis, first)
    return np.column_stack((first, second, axis))


def cylinder_distance_hppfcl(first: Cylinder, second: Cylinder) -> float:
    """Return an HPP-FCL distance when the optional package is installed."""

    try:
        import hppfcl  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError("hpp-fcl is required for this validation function") from exc
    shape_a = hppfcl.Cylinder(first.radius, first.length)
    shape_b = hppfcl.Cylinder(second.radius, second.length)
    transform_a = hppfcl.Transform3f(_rotation_from_z(first.axis), first.center)
    transform_b = hppfcl.Transform3f(_rotation_from_z(second.axis), second.center)
    request: Any = hppfcl.DistanceRequest()
    request.gjk_tolerance = 1.0e-10
    request.gjk_max_iterations = 1000
    result: Any = hppfcl.DistanceResult()
    return max(
        0.0,
        float(hppfcl.distance(shape_a, transform_a, shape_b, transform_b, request, result)),
    )
