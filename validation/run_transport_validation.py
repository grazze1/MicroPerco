# SPDX-License-Identifier: Apache-2.0
"""Independent dense-circuit and mixed-geometry transport release checks."""

from __future__ import annotations

import argparse
import platform
from itertools import product
from pathlib import Path

import numpy as np
import scipy

from microperco import (
    CylinderSpec,
    Domain,
    PopulationSpec,
    ResistorEdge,
    ResistorNetwork,
    SphereSpec,
    TunnelingConductanceModel,
    __version__,
    analyze_conductivity,
    generate_microstructure,
    solve_resistor_network,
)
from microperco.io import dump_json
from microperco.validation import run_validation_suite


def validate() -> dict[str, object]:
    voltage_errors: list[float] = []
    conductance_errors: list[float] = []
    residuals: list[float] = []
    for seed in range(32):
        rng = np.random.default_rng(2000 + seed)
        size = 6 + seed
        edges = tuple(
            ResistorEdge(i, j, float(np.exp(rng.uniform(-4, 4))))
            for i in range(size)
            for j in range(i + 1, size)
            if j == i + 1 or rng.random() < 0.2
        )
        incidence = np.zeros((len(edges), size))
        weights = np.array([edge.conductance for edge in edges])
        for index, edge in enumerate(edges):
            incidence[index, edge.i] = 1
            incidence[index, edge.j] = -1
        laplacian = incidence.T @ (weights[:, None] * incidence)
        expected = np.zeros(size)
        expected[0] = 1
        expected[1:-1] = np.linalg.solve(laplacian[1:-1, 1:-1], -laplacian[1:-1, 0])
        reference_g = float((laplacian @ expected)[0])
        solved = solve_resistor_network(ResistorNetwork(size, edges, 0, size - 1))
        voltage_errors.append(float(np.max(np.abs(np.asarray(solved.node_voltages) - expected))))
        conductance_errors.append(abs(solved.effective_conductance / reference_g - 1))
        residuals.append(solved.relative_kirchhoff_residual)
    parity_cases = 0
    conducting_cases = 0
    maximum_sigma_difference = 0.0
    for periodic in product((False, True), repeat=3):
        domain = Domain(4, periodic)
        particles = generate_microstructure(
            domain,
            (
                PopulationSpec(SphereSpec(0.75), 8),
                PopulationSpec(CylinderSpec(0.4, 2), 6),
            ),
            seed=2026,
        )
        for axis in "xyz":
            model = TunnelingConductanceModel(1, 0.4, 0.6)
            fast = analyze_conductivity(particles, domain, model, axis=axis)
            brute = analyze_conductivity(
                particles, domain, model, axis=axis, neighbor_backend="bruteforce"
            )
            if fast.geometry.network != brute.geometry.network:
                raise AssertionError(f"network mismatch: periodic={periodic}, axis={axis}")
            maximum_sigma_difference = max(
                maximum_sigma_difference,
                abs(fast.effective_conductivity - brute.effective_conductivity),
            )
            parity_cases += 1
            conducting_cases += int(fast.percolates)
    builtin = run_validation_suite()
    return {
        "version": __version__,
        "passed": builtin.passed
        and max(voltage_errors) < 1e-10
        and max(conductance_errors) < 1e-10
        and max(residuals) < 1e-10
        and maximum_sigma_difference == 0,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
        "dense_oracle": {
            "cases": len(voltage_errors),
            "seeds": [2000, 2031],
            "max_absolute_voltage_error": max(voltage_errors),
            "max_relative_conductance_error": max(conductance_errors),
            "max_relative_kirchhoff_residual": max(residuals),
            "acceptance_limit": 1e-10,
        },
        "mixed_geometry_search_parity": {
            "cases": parity_cases,
            "conducting_cases": conducting_cases,
            "seed": 2026,
            "max_absolute_conductivity_difference": maximum_sigma_difference,
        },
        "builtin": builtin,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate()
    payload = dump_json(result, args.output)
    if args.output is None:
        print(payload, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
