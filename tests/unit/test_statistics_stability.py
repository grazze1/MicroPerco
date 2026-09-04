# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest

from microperco.io import dumps_json
from microperco.statistics import BinaryLinkModel, fit_logistic, fit_probit, pava

LinkFit = Callable[..., BinaryLinkModel]


@pytest.mark.parametrize("fit", [fit_logistic, fit_probit])
def test_monotone_link_fit_is_translation_invariant(fit: LinkFit) -> None:
    x = np.array([0.0, 1.0])
    shifted_x = x + 1.0e10

    reference = fit(x, [1, 0], 1)
    shifted = fit(shifted_x, [1, 0], 1)

    np.testing.assert_allclose(shifted.predict(shifted_x), reference.predict(x), atol=2.0e-8)
    assert shifted.log_likelihood == pytest.approx(reference.log_likelihood, abs=1.0e-12)
    assert shifted.converged == reference.converged
    assert shifted.slope >= 0.0
    assert reference.slope >= 0.0


@pytest.mark.parametrize("fit", [fit_logistic, fit_probit])
def test_link_fit_is_translation_and_scale_invariant(fit: LinkFit) -> None:
    x = np.arange(5.0)
    successes = [1, 5, 18, 42, 49]
    transformed_x = 1.0e10 + 1.0e6 * x

    reference = fit(x, successes, 50)
    transformed = fit(transformed_x, successes, 50)

    np.testing.assert_allclose(
        transformed.predict(transformed_x), reference.predict(x), rtol=2.0e-7, atol=2.0e-8
    )
    assert transformed.log_likelihood == pytest.approx(reference.log_likelihood, rel=1.0e-10)
    assert transformed.converged == reference.converged
    assert transformed.slope == pytest.approx(reference.slope, rel=2.0e-7)
    assert transformed.intercept == pytest.approx(reference.intercept, rel=2.0e-7)
    assert transformed.x_center == 1.0e10 + 1.0e6 * reference.x_center
    assert transformed.x_scale == 1.0e6 * reference.x_scale
    assert transformed.threshold(0.5) == pytest.approx(
        1.0e10 + 1.0e6 * reference.threshold(0.5), rel=1.0e-12
    )


@pytest.mark.parametrize("fit", [fit_logistic, fit_probit])
def test_unconstrained_decreasing_fit_keeps_slope_semantics_after_translation(fit: LinkFit) -> None:
    x = np.array([0.0, 1.0])
    shifted_x = x + 1.0e10

    reference = fit(x, [1, 0], 1, monotone=False)
    shifted = fit(shifted_x, [1, 0], 1, monotone=False)

    assert reference.converged and shifted.converged
    assert reference.slope < 0.0 and shifted.slope < 0.0
    np.testing.assert_allclose(shifted.predict(shifted_x), reference.predict(x), atol=1.0e-12)
    assert shifted.log_likelihood == pytest.approx(reference.log_likelihood, abs=1.0e-12)


@pytest.mark.parametrize("fit", [fit_logistic, fit_probit])
def test_link_model_json_is_public_complete_and_round_trippable(fit: LinkFit) -> None:
    model = fit([0.0, 1.0, 2.0], [1, 5, 9], 10)
    payload = json.loads(dumps_json(model))

    assert not any(key.startswith("_") for key in payload)
    assert payload["x_center"] == model.x_center
    assert payload["x_scale"] == model.x_scale
    reconstructed = BinaryLinkModel(**payload)
    assert reconstructed == model
    np.testing.assert_array_equal(reconstructed.predict([0.0, 1.0]), model.predict([0.0, 1.0]))


@pytest.mark.parametrize("fit", [fit_logistic, fit_probit])
def test_model_equality_includes_the_coordinate_that_controls_prediction(fit: LinkFit) -> None:
    model = fit([0.0, 1.0, 2.0], [1, 5, 9], 10)
    shifted_coordinate = replace(model, x_center=model.x_center + 0.25)

    assert shifted_coordinate != model
    assert not np.array_equal(shifted_coordinate.predict([0.0, 1.0]), model.predict([0.0, 1.0]))


@pytest.mark.parametrize("fit", [fit_logistic, fit_probit])
def test_smallest_subnormal_x_scale_keeps_model_and_json_finite(fit: LinkFit) -> None:
    smallest = np.nextafter(0.0, 1.0)
    model = fit([0.0, smallest], [0, 1], 1)

    assert np.all(np.isfinite(model.predict([0.0, smallest])))
    assert np.isfinite(model.intercept)
    assert np.isfinite(model.slope)
    assert model.x_scale == smallest
    dumps_json(model)


def test_pava_handles_weights_whose_unscaled_sum_overflows() -> None:
    np.testing.assert_array_equal(pava([1.0, 0.0], [1.0e308, 1.0e308]), [0.5, 0.5])


def test_pava_is_invariant_to_large_common_weight_scale() -> None:
    values = [0.2, 0.9, 0.1, 0.8]
    weights = np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(pava(values, weights * 1.0e307), pava(values, weights))
