# SPDX-License-Identifier: Apache-2.0
"""Run deterministic release validation beyond the unit-test suite."""

from __future__ import annotations

import argparse
import json
import platform
from itertools import product
from pathlib import Path

import numpy as np
import scipy
from geometry_cases import cylinder_cases

from microperco import (
    Cylinder,
    Domain,
    Sphere,
    ThresholdContactModel,
    analyze_percolation,
    generate_microstructure,
)
from microperco.contact import find_contacts
from microperco.generation import CylinderSpec, PopulationSpec, isotropic_directions
from microperco.geometry import distance
from microperco.geometry.reference import cylinder_distance_scipy
from microperco.validation import run_validation_suite


def _signature(result: object) -> list[tuple[int, int, tuple[int, int, int]]]:
    return [(edge.i, edge.j, edge.lattice_shift) for edge in result.edges]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    builtin = run_validation_suite()

    scipy_errors = [
        abs(distance(first, second) - cylinder_distance_scipy(first, second))
        for first, second in cylinder_cases()
    ]

    parity_cases = 0
    total_brute_candidates = 0
    total_fast_candidates = 0
    total_brute_distances = 0
    total_fast_distances = 0
    for system in range(3):
        rng = np.random.default_rng(1800 + system)
        particles = tuple(
            Sphere(rng.uniform(-5, 5, 3), 0.35, index)
            if index % 3
            else Cylinder(rng.uniform(-5, 5, 3), rng.normal(size=3), 1.4, 0.25, index)
            for index in range(18 + system * 4)
        )
        for periodic in product((False, True), repeat=3):
            domain = Domain(10.0, periodic)
            model = ThresholdContactModel(0.2)
            brute = find_contacts(particles, domain, model, method="bruteforce")
            fast = find_contacts(particles, domain, model, method="cell_list")
            if _signature(brute) != _signature(fast):
                raise AssertionError(f"search disagreement for system={system}, PBC={periodic}")
            parity_cases += 1
            total_brute_candidates += brute.candidate_pairs
            total_fast_candidates += fast.candidate_pairs
            total_brute_distances += brute.distance_evaluations
            total_fast_distances += fast.distance_evaluations

    solver_face_cases = 0
    for case in range(15):
        rng = np.random.default_rng(7100 + case)
        particles = tuple(Sphere(rng.uniform(-5, 5, 3), 0.8, index) for index in range(30))
        domain = Domain(10.0, (False, bool(case % 2), bool(case % 3)))
        union = analyze_percolation(
            particles, domain, ThresholdContactModel(0.2), solver="union_find"
        )
        bfs = analyze_percolation(particles, domain, ThresholdContactModel(0.2), solver="bfs")
        if union.percolates != bfs.percolates or union.spanning_path != bfs.spanning_path:
            raise AssertionError(f"face solver disagreement in case {case}")
        solver_face_cases += 1

    solver_wrap_cases = 0
    for axis in range(3):
        for threshold in (1.9, 2.0, 2.1):
            particles = []
            for index, coordinate in enumerate((-4.0, 0.0, 4.0)):
                center = np.zeros(3)
                center[axis] = coordinate
                particles.append(Sphere(center, 1.0, index))
            periodic = [False, False, False]
            periodic[axis] = True
            domain = Domain(10.0, periodic)
            model = ThresholdContactModel(threshold)
            union = analyze_percolation(
                particles, domain, model, axis=axis, mode="periodic_wrap", solver="union_find"
            )
            bfs = analyze_percolation(
                particles, domain, model, axis=axis, mode="periodic_wrap", solver="bfs"
            )
            if union.percolates != bfs.percolates or union.winding != bfs.winding:
                raise AssertionError(f"wrap solver disagreement for axis={axis}")
            solver_wrap_cases += 1

    directions = isotropic_directions(np.random.default_rng(20250902), 100_000)
    isotropy = {
        "count": len(directions),
        "mean": np.mean(directions, axis=0).tolist(),
        "second_moment": np.mean(directions * directions, axis=0).tolist(),
        "positive_fraction": np.mean(directions > 0.0, axis=0).tolist(),
    }

    generation_domain = Domain(12.0, True)
    population = (PopulationSpec(CylinderSpec(0.2, 2.0), 16),)
    first = generate_microstructure(generation_domain, population, seed=9001)
    second = generate_microstructure(generation_domain, population, seed=9001)
    rng_reproducible = all(
        np.array_equal(left.center, right.center) and np.array_equal(left.axis, right.axis)
        for left, right in zip(first, second, strict=True)
        if isinstance(left, Cylinder) and isinstance(right, Cylinder)
    )

    payload = {
        "passed": builtin.passed and max(scipy_errors) < 2.0e-6 and rng_reproducible,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "builtin_checks": [
            {"name": check.name, "passed": check.passed, "detail": check.detail}
            for check in builtin.checks
        ],
        "geometry_scipy": {
            "cases": len(scipy_errors),
            "max_absolute_error": max(scipy_errors),
            "mean_absolute_error": float(np.mean(scipy_errors)),
            "acceptance_limit": 2.0e-6,
        },
        "contact_search_parity": {
            "cases": parity_cases,
            "bruteforce_candidates": total_brute_candidates,
            "cell_list_candidates": total_fast_candidates,
            "bruteforce_distance_evaluations": total_brute_distances,
            "cell_list_distance_evaluations": total_fast_distances,
        },
        "solver_parity": {
            "face_to_face_cases": solver_face_cases,
            "periodic_wrap_cases": solver_wrap_cases,
        },
        "rng_reproducible": rng_reproducible,
        "isotropy": isotropy,
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.write_text(text, encoding="utf-8")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
