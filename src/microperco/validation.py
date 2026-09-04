# SPDX-License-Identifier: Apache-2.0
"""Fast built-in correctness checks used by the CLI and release process."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contact import ContactSearchResult, ThresholdContactModel, find_contacts
from .domain import Domain
from .geometry import distance, periodic_distance
from .particles import Cylinder, Sphere
from .percolation import analyze_percolation


@dataclass(frozen=True, slots=True)
class ValidationCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class ValidationSummary:
    passed: bool
    checks: tuple[ValidationCheck, ...]


def _edge_signature(
    result: ContactSearchResult,
) -> tuple[tuple[int, int, tuple[int, int, int]], ...]:
    return tuple((edge.i, edge.j, edge.lattice_shift) for edge in result.edges)


def run_validation_suite() -> ValidationSummary:
    """Run deterministic analytic, PBC, search, and connectivity checks."""

    checks: list[ValidationCheck] = []

    first = Sphere((0.0, 0.0, 0.0), 1.0)
    second = Sphere((3.0, 0.0, 0.0), 1.0)
    sphere_gap = distance(first, second)
    checks.append(
        ValidationCheck(
            "analytic sphere gap",
            abs(sphere_gap - 1.0) <= 1.0e-12,
            f"computed={sphere_gap:.16g}, expected=1",
        )
    )

    cylinder = Cylinder((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 2.0, 0.5)
    endcap_sphere = Sphere((2.0, 0.0, 0.0), 0.5)
    cylinder_gap = distance(cylinder, endcap_sphere)
    checks.append(
        ValidationCheck(
            "cylinder end-cap gap",
            abs(cylinder_gap - 0.5) <= 1.0e-9,
            f"computed={cylinder_gap:.16g}, expected=0.5",
        )
    )

    periodic_domain = Domain((10.0, 10.0, 10.0), (True, False, False))
    periodic_gap = periodic_distance(
        Sphere((-4.6, 0.0, 0.0), 0.5),
        Sphere((4.6, 0.0, 0.0), 0.5),
        periodic_domain,
    )
    checks.append(
        ValidationCheck(
            "periodic minimum image",
            periodic_gap.distance == 0.0 and periodic_gap.lattice_shift == (-1, 0, 0),
            f"gap={periodic_gap.distance:.16g}, shift={periodic_gap.lattice_shift}",
        )
    )

    rng = np.random.default_rng(20250901)
    random_particles = tuple(Sphere(rng.uniform(-5.0, 5.0, 3), 0.35, index) for index in range(28))
    model = ThresholdContactModel(0.2)
    brute = find_contacts(random_particles, periodic_domain, model, method="bruteforce")
    fast = find_contacts(random_particles, periodic_domain, model, method="cell_list")
    parity = _edge_signature(brute) == _edge_signature(fast)
    checks.append(
        ValidationCheck(
            "optimized/reference contact parity",
            parity,
            f"edges={len(brute.edges)}, candidates={brute.candidate_pairs}->{fast.candidate_pairs}",
        )
    )

    chain_domain = Domain((10.0, 4.0, 4.0), False)
    chain = tuple(
        Sphere((coordinate, 0.0, 0.0), 1.0, index)
        for index, coordinate in enumerate((-4.0, -2.0, 0.0, 2.0, 4.0))
    )
    union_result = analyze_percolation(chain, chain_domain, axis="x", solver="union_find")
    bfs_result = analyze_percolation(chain, chain_domain, axis="x", solver="bfs")
    connected = union_result.percolates and bfs_result.percolates
    checks.append(
        ValidationCheck(
            "known spanning chain and solver parity",
            connected and union_result.spanning_path == bfs_result.spanning_path,
            f"path={union_result.spanning_path}",
        )
    )

    return ValidationSummary(all(check.passed for check in checks), tuple(checks))


__all__ = ["ValidationCheck", "ValidationSummary", "run_validation_suite"]
