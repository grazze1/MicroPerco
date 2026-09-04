# SPDX-License-Identifier: Apache-2.0
"""Deterministic independent-reference cylinder cases."""

from __future__ import annotations

import numpy as np

from microperco import Cylinder


def cylinder_cases(seed: int = 20250831, count: int = 24) -> tuple[tuple[Cylinder, Cylinder], ...]:
    """Return diverse separated, tangent, and overlapping cylinder pairs."""

    rng = np.random.default_rng(seed)
    result: list[tuple[Cylinder, Cylinder]] = [
        (
            Cylinder((0, 0, 0), (1, 0, 0), 2, 0.5),
            Cylinder((3, 0, 0), (1, 0, 0), 2, 0.5),
        ),
        (
            Cylinder((0, 0, 0), (1, 0, 0), 2, 0.5),
            Cylinder((0, 2, 0), (1, 0, 0), 2, 0.5),
        ),
        (
            Cylinder((0, 0, 0), (1, 0, 0), 2, 0.5),
            Cylinder((0, 0, 2), (0, 1, 0), 2, 0.5),
        ),
        (
            Cylinder((0, 0, 0), (1, 1.0e-9, 0), 4, 0.4),
            Cylinder((0.2, 1.4, 0.1), (1, -1.0e-9, 0), 3, 0.3),
        ),
        (
            Cylinder((0, 0, 0), (1, 0, 0), 2, 0.5),
            Cylinder((0, 0, 0), (0, 1, 0), 2, 0.5),
        ),
        (
            Cylinder((0, 0, 0), (1, 0, 0), 2, 0.5),
            Cylinder((0, 1, 0), (1, 0, 0), 2, 0.5),
        ),
    ]
    while len(result) < count:
        axis_a = rng.normal(size=3)
        axis_b = rng.normal(size=3)
        length_a, length_b = rng.uniform(0.6, 4.5, size=2)
        radius_a, radius_b = rng.uniform(0.15, 0.8, size=2)
        center_a = rng.uniform(-1.0, 1.0, size=3)
        displacement = rng.normal(size=3)
        displacement /= np.linalg.norm(displacement)
        bound_a = np.hypot(0.5 * length_a, radius_a)
        bound_b = np.hypot(0.5 * length_b, radius_b)
        center_b = center_a + displacement * (bound_a + bound_b + rng.uniform(0.0, 2.0))
        result.append(
            (
                Cylinder(center_a, axis_a, length_a, radius_a),
                Cylinder(center_b, axis_b, length_b, radius_b),
            )
        )
    return tuple(result)


__all__ = ["cylinder_cases"]
