# SPDX-License-Identifier: Apache-2.0
"""Executable contract for the README Quick Start."""

from microperco import (
    ConstantConductanceModel,
    Domain,
    MonteCarloSimulator,
    PopulationSpec,
    Sphere,
    SphereSpec,
    ThresholdContactModel,
    TunnelingConductanceModel,
    analyze_directional_conductivity,
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


def test_readme_conductivity() -> None:
    import math

    particles = (Sphere((-1.25, 0, 0), 1), Sphere((1.25, 0, 0), 1))
    result = analyze_directional_conductivity(
        particles,
        Domain((4.5, 3.0, 3.0), False),
        TunnelingConductanceModel(contact_conductance=2.0, decay_length=0.5, cutoff=0.5),
        electrode_model=ConstantConductanceModel(contact_conductance=10.0),
    )
    assert math.isclose(result.sigma_x, 0.12838526097369987, rel_tol=1e-12)
    assert result.sigma_y == result.sigma_z == 0
