# SPDX-License-Identifier: Apache-2.0
"""Executable contract for the README Quick Start."""

from microperco import (
    Domain,
    MonteCarloSimulator,
    PopulationSpec,
    SphereSpec,
    ThresholdContactModel,
    analyze_percolation,
    generate_microstructure,
)


def test_readme_quick_start() -> None:
    domain = Domain(size=(8.0, 8.0, 8.0), periodic=(False, True, True))
    simulator = MonteCarloSimulator(
        domain=domain,
        particle_specs=[SphereSpec(radius=0.75)],
        contact_model=ThresholdContactModel(0.10),
        axis="x",
    )
    result = simulator.estimate_probability(particle_counts=[24], trials=16, seed=42)
    assert result.trials == 16
    assert 0.0 <= result.probability <= 1.0
    assert result.confidence_interval.lower <= result.probability
    assert result.probability <= result.confidence_interval.upper


def test_readme_single_realization() -> None:
    domain = Domain(size=(8.0, 8.0, 8.0), periodic=(False, True, True))
    sample = generate_microstructure(
        domain,
        [PopulationSpec(SphereSpec(0.75), 24, "beads")],
        seed=42,
    )
    graph = analyze_percolation(
        sample.particles,
        domain,
        ThresholdContactModel(0.10),
        axis="x",
    )
    assert graph.particle_count == 24
