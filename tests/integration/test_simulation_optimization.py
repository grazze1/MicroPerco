# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Sequence
from math import pi

import numpy as np
import pytest

from microperco import (
    Domain,
    MaterialSpec,
    PopulationSpec,
    SeedSequenceState,
    SphereSpec,
    estimate_critical_loading,
    estimate_percolation_probability,
    optimize_mixture,
)
from microperco.exceptions import ConfigurationError, OptimizationError
from microperco.particles import Particle


def _at_least(required: int):
    def evaluate(particles: Sequence[Particle]) -> bool:
        return len(particles) >= required

    return evaluate


def test_monte_carlo_seed_reproducibility() -> None:
    kwargs = dict(
        domain=Domain(5.0, False),
        particle_specs=SphereSpec(0.5),
        particle_counts=3,
        trials=24,
        seed=(1, 2, 3),
        evaluator=_at_least(3),
    )
    first = estimate_percolation_probability(**kwargs)
    second = estimate_percolation_probability(**kwargs)
    assert first == second
    assert first.successes == 24
    assert first.particle_counts == (3,)


def test_reusing_seed_sequence_does_not_advance_caller_state() -> None:
    seed = np.random.SeedSequence(123)
    kwargs = dict(
        domain=Domain(5.0, False),
        particle_specs=SphereSpec(0.2),
        particle_counts=1,
        trials=31,
        seed=seed,
        evaluator=lambda particles: bool(particles[0].center[0] > 0.0),
    )
    first = estimate_percolation_probability(**kwargs)
    second = estimate_percolation_probability(**kwargs)
    assert first == second
    assert seed.n_children_spawned == 0


def test_seed_sequence_result_records_full_initialization_state() -> None:
    first_seed = np.random.SeedSequence(123, spawn_key=(1,))
    second_seed = np.random.SeedSequence(123, spawn_key=(2,))
    kwargs = dict(
        domain=Domain(5.0, False),
        particle_specs=SphereSpec(0.2),
        particle_counts=1,
        trials=20,
        evaluator=lambda particles: bool(particles[0].center[0] > 0.0),
    )
    first = estimate_percolation_probability(**kwargs, seed=first_seed)
    second = estimate_percolation_probability(**kwargs, seed=second_seed)
    assert first.seed == SeedSequenceState(123, (1,), first_seed.pool_size)
    assert second.seed == SeedSequenceState(123, (2,), second_seed.pool_size)
    assert first.seed != second.seed


def test_monte_carlo_custom_result_protocol() -> None:
    class Outcome:
        conductive = True

    result = estimate_percolation_probability(
        Domain(5.0, False),
        SphereSpec(0.5),
        1,
        trials=5,
        seed=10,
        evaluator=lambda _: Outcome(),
    )
    assert result.probability == 1.0


@pytest.mark.parametrize("strategy", ["pava", "logistic", "probit"])
def test_critical_loading_detects_and_certifies_step(strategy: str) -> None:
    result = estimate_critical_loading(
        Domain(5.0, False),
        SphereSpec(0.5),
        loading_grid=(0, 1, 2, 3),
        target_probability=0.5,
        search_trials=30,
        certification_trials=100,
        confidence=0.95,
        strategy=strategy,  # type: ignore[arg-type]
        seed=42,
        evaluator=_at_least(2),
    )
    assert result.status == "CERTIFIED"
    assert result.critical_count == 2
    assert result.certification_comparisons == 2
    assert result.per_comparison_confidence == pytest.approx(0.975)
    assert result.familywise_confidence == 0.95


def test_critical_volume_fraction_includes_fixed_populations() -> None:
    domain = Domain(5.0, False)
    fixed_spec = SphereSpec(0.25)
    variable_spec = SphereSpec(0.5)
    fixed_population = PopulationSpec(fixed_spec, 1)
    result = estimate_critical_loading(
        domain,
        variable_spec,
        loading_grid=(0, 1, 2, 3),
        fixed_populations=(fixed_population,),
        target_probability=0.5,
        search_trials=30,
        certification_trials=100,
        seed=42,
        evaluator=_at_least(3),
    )

    assert result.status == "CERTIFIED"
    assert result.critical_count == 2
    expected_fraction = (fixed_population.nominal_volume + 2 * variable_spec.volume) / 5.0**3
    assert result.critical_volume_fraction == pytest.approx(expected_fraction)


def test_critical_volume_fraction_avoids_overflowing_population_sum() -> None:
    domain = Domain((1.0e103, 1.0e103, 1.0e102), False)
    spec = SphereSpec(3.0e102)
    result = estimate_critical_loading(
        domain,
        spec,
        loading_grid=(1,),
        fixed_populations=(PopulationSpec(spec, 1),),
        target_probability=0.01,
        search_trials=1,
        certification_trials=1,
        evaluator=lambda _: True,
        seed=1,
    )
    assert result.critical_volume_fraction == pytest.approx(2.2619467105846507)


def test_critical_loading_no_crossing_is_certified() -> None:
    result = estimate_critical_loading(
        Domain(5.0, False),
        SphereSpec(0.5),
        loading_grid=(0, 1, 2),
        target_probability=0.5,
        search_trials=20,
        certification_trials=100,
        seed=2,
        evaluator=lambda _: False,
    )
    assert result.status == "NO_CROSSING"
    assert result.critical_count is None
    assert result.critical_volume_fraction is None
    assert result.certification_comparisons == 1


def test_nested_search_estimates_are_monotone_for_deterministic_evaluator() -> None:
    result = estimate_critical_loading(
        Domain(5.0, False),
        SphereSpec(0.2),
        loading_grid=(0, 1, 2, 3, 4),
        target_probability=0.6,
        search_trials=10,
        certification_trials=30,
        seed=9,
        evaluator=_at_least(3),
    )
    probabilities = [record.probability for record in result.search_estimates]
    assert probabilities == sorted(probabilities)


def test_optimization_certifies_minimum_cost_design() -> None:
    material = MaterialSpec("unit", 1.0)
    result = optimize_mixture(
        Domain(5.0, False),
        (SphereSpec(0.5, material),),
        ((0, 3),),
        target_probability=0.5,
        screening_trials=10,
        certification_trials=120,
        confidence=0.95,
        seed=7,
        evaluator=_at_least(1),
    )
    assert result.status == "CERTIFIED_OPTIMAL"
    assert result.optimal_counts == (1,)
    assert result.optimal_cost == pytest.approx(SphereSpec(0.5, material).cost)
    assert result.certification_comparisons == 8
    assert result.per_comparison_confidence == pytest.approx(0.99375)


def test_optimization_volume_fraction_avoids_overflowing_species_sum() -> None:
    domain = Domain((1.0e103, 1.0e103, 1.0e102), False)
    spec = SphereSpec(3.0e102)
    result = optimize_mixture(
        domain,
        (spec, spec),
        ((1, 1), (1, 1)),
        target_probability=0.01,
        screening_trials=1,
        certification_trials=1,
        evaluator=lambda _: True,
        seed=1,
    )
    assert result.certification_estimates[0].nominal_volume_fraction == pytest.approx(
        2.2619467105846507
    )


def test_optimization_does_not_round_tiny_real_cost_difference_to_a_tie() -> None:
    sphere_volume = 4.0 * pi * 0.1**3 / 3.0
    spec = SphereSpec(0.1, MaterialSpec("tiny", 1.0e-15 / sphere_volume))

    def evaluate(particles: Sequence[Particle]) -> bool:
        return len(particles) >= 2 or particles[0].center[0] > 1.5

    result = optimize_mixture(
        Domain(5.0, False),
        (spec,),
        ((1, 2),),
        target_probability=0.1,
        screening_trials=1,
        certification_trials=2,
        confidence=0.95,
        seed=2,
        evaluator=evaluate,
    )
    assert result.status == "INCONCLUSIVE"
    assert result.counts == (2,)
    cheaper, recommendation = result.certification_estimates
    assert cheaper.total_cost < recommendation.total_cost
    assert cheaper.upper_bound >= result.target_probability


def test_optimization_no_certified_feasible() -> None:
    result = optimize_mixture(
        Domain(5.0, False),
        (SphereSpec(0.5),),
        ((0, 2),),
        target_probability=0.8,
        screening_trials=5,
        certification_trials=30,
        evaluator=lambda _: False,
        seed=3,
    )
    assert result.status == "NO_CERTIFIED_FEASIBLE"
    assert result.optimal_counts is None


def test_optimization_rejects_excessive_search_space() -> None:
    with pytest.raises(OptimizationError, match="exceeding"):
        optimize_mixture(
            Domain(5.0, False),
            (SphereSpec(0.5), SphereSpec(0.4)),
            ((0, 100), (0, 100)),
            max_candidates=100,
            evaluator=lambda _: False,
        )


def test_optimization_rejects_search_space_larger_than_platform_range_length() -> None:
    with pytest.raises(OptimizationError, match="exceeding"):
        optimize_mixture(
            Domain(5.0, False),
            (SphereSpec(0.5),),
            ((0, 10**100),),
            max_candidates=10,
            evaluator=lambda _: False,
        )


def test_critical_loading_rejects_pathological_implicit_grid() -> None:
    with pytest.raises(ConfigurationError, match="loading grid .* safety limit"):
        estimate_critical_loading(
            Domain(5.0, False),
            SphereSpec(0.5),
            max_count=10**100,
            evaluator=lambda _: False,
        )
