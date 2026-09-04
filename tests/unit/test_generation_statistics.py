# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math

import numpy as np
import pytest

from microperco import Cylinder, Domain, Sphere
from microperco.exceptions import ConfigurationError
from microperco.generation import (
    CylinderSpec,
    PopulationSpec,
    SphereSpec,
    generate_microstructure,
    isotropic_directions,
    particle_count_for_volume_fraction,
    particle_volume,
    total_particle_volume,
    volume_fraction,
)
from microperco.statistics import (
    binomial_estimate,
    bonferroni_per_comparison_confidence,
    clopper_pearson_interval,
    fit_logistic,
    fit_probit,
    pava,
    wilson_interval,
)


def test_seeded_generation_is_reproducible() -> None:
    domain = Domain((7.0, 8.0, 9.0), True)
    populations = (
        PopulationSpec(SphereSpec(0.2), 4, "spheres"),
        PopulationSpec(CylinderSpec(0.15, 1.2), 5, "fibers"),
    )
    first = generate_microstructure(domain, populations, seed=1234)
    second = generate_microstructure(domain, populations, seed=1234)
    assert first.seed == 1234
    assert len(first) == 9
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left.center, right.center)
        if isinstance(left, Cylinder) and isinstance(right, Cylinder):
            np.testing.assert_array_equal(left.axis, right.axis)


def test_different_seeds_change_realization() -> None:
    population = (PopulationSpec(SphereSpec(0.2), 4),)
    first = generate_microstructure(Domain(5.0, False), population, seed=1)
    second = generate_microstructure(Domain(5.0, False), population, seed=2)
    assert not np.array_equal(first[0].center, second[0].center)


@pytest.mark.parametrize("seed", [[1.5], [True], [], [-1], "bad"])
def test_generation_rejects_bad_seed(seed: object) -> None:
    with pytest.raises(ConfigurationError):
        generate_microstructure(
            Domain(5.0, False),
            (PopulationSpec(SphereSpec(0.2), 1),),
            seed=seed,  # type: ignore[arg-type]
        )


def test_isotropic_direction_statistics() -> None:
    directions = isotropic_directions(np.random.default_rng(2025), 50_000)
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0, atol=2.0e-15)
    assert np.max(np.abs(np.mean(directions, axis=0))) < 0.01
    assert np.max(np.abs(np.mean(directions * directions, axis=0) - 1.0 / 3.0)) < 0.01
    positive_fraction = np.mean(directions > 0.0, axis=0)
    assert np.max(np.abs(positive_fraction - 0.5)) < 0.01


def test_volume_helpers_for_mixed_particles() -> None:
    domain = Domain(10.0, False)
    particles = (
        Sphere((0.0, 0.0, 0.0), 1.0),
        Cylinder((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 2.0, 1.0),
    )
    expected = 4.0 * math.pi / 3.0 + 2.0 * math.pi
    assert particle_volume(particles[0]) == pytest.approx(4.0 * math.pi / 3.0)
    assert total_particle_volume(particles) == pytest.approx(expected)
    assert volume_fraction(particles, domain) == pytest.approx(expected / 1000.0)


def test_volume_fraction_avoids_overflowing_the_unscaled_total() -> None:
    domain = Domain((1.0e103, 1.0e103, 1.0e102), False)
    spec = SphereSpec(3.0e102)
    populations = (PopulationSpec(spec, 1), PopulationSpec(spec, 1))
    with pytest.raises(ConfigurationError, match="total particle volume"):
        total_particle_volume(populations)
    assert volume_fraction(populations, domain) == pytest.approx(
        2.2619467105846507,
        rel=2.0e-15,
    )


@pytest.mark.parametrize("fraction", [0.0, 0.01, 0.5])
def test_particle_count_for_volume_fraction_uses_ceiling(fraction: float) -> None:
    spec = SphereSpec(1.0)
    expected = math.ceil(fraction * 1000.0 / spec.volume)
    assert (
        particle_count_for_volume_fraction(fraction, spec, Domain(10.0, False), rounding="ceil")
        == expected
    )


@pytest.mark.parametrize(
    "successes,trials,expected",
    [(0, 10, 0.0), (5, 10, 0.5), (10, 10, 1.0), (37, 100, 0.37)],
)
def test_binomial_estimates(successes: int, trials: int, expected: float) -> None:
    result = binomial_estimate(successes, trials)
    assert result.probability == expected
    assert result.wilson.lower <= expected <= result.wilson.upper
    assert result.clopper_pearson.lower <= expected <= result.clopper_pearson.upper


def test_wilson_known_value() -> None:
    interval = wilson_interval(50, 100, 0.95)
    assert interval.lower == pytest.approx(0.4038315303659956)
    assert interval.upper == pytest.approx(0.5961684696340044)


def test_clopper_pearson_boundary_cases() -> None:
    zero = clopper_pearson_interval(0, 20)
    all_success = clopper_pearson_interval(20, 20)
    assert zero.lower == 0.0 and zero.upper < 0.2
    assert all_success.upper == 1.0 and all_success.lower > 0.8


@pytest.mark.parametrize(
    "values,weights,expected",
    [
        ([0.1, 0.2, 0.3], None, [0.1, 0.2, 0.3]),
        ([0.3, 0.1, 0.2], None, [0.2, 0.2, 0.2]),
        ([0.0, 1.0, 0.0], [1.0, 1.0, 2.0], [0.0, 1.0 / 3.0, 1.0 / 3.0]),
    ],
)
def test_pava_known_solutions(
    values: list[float], weights: list[float] | None, expected: list[float]
) -> None:
    np.testing.assert_allclose(pava(values, weights), expected)


def test_pava_result_is_monotone_for_random_input() -> None:
    values = np.random.default_rng(42).uniform(size=100)
    fitted = pava(values)
    assert np.all(np.diff(fitted) >= 0.0)
    assert np.mean(fitted) == pytest.approx(np.mean(values))


@pytest.mark.parametrize("fit", [fit_logistic, fit_probit])
def test_binary_link_fits_increasing_data(fit: object) -> None:
    model = fit([0, 1, 2, 3, 4], [1, 5, 18, 42, 49], 50)  # type: ignore[operator]
    prediction = model.predict([0, 1, 2, 3, 4])
    assert model.converged
    assert model.slope >= 0.0
    assert np.all(np.diff(prediction) >= 0.0)
    assert 1.0 < model.threshold(0.5) < 3.0


def test_bonferroni_confidence_controls_family() -> None:
    level = bonferroni_per_comparison_confidence(0.95, 20)
    assert level == pytest.approx(0.9975)
    assert 1.0 - 20 * (1.0 - level) == pytest.approx(0.95)


@pytest.mark.parametrize(
    "call",
    [
        lambda: wilson_interval(-1, 10),
        lambda: wilson_interval(2, 0),
        lambda: pava([]),
        lambda: pava([1.0, 2.0], [1.0, 0.0]),
        lambda: bonferroni_per_comparison_confidence(1.0, 2),
    ],
)
def test_statistics_reject_invalid_inputs(call: object) -> None:
    with pytest.raises(ConfigurationError):
        call()  # type: ignore[operator]
