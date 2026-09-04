# SPDX-License-Identifier: Apache-2.0
"""Small end-to-end API smoke test used by release checks."""

from microperco import (
    Domain,
    MonteCarloSimulator,
    SphereSpec,
    ThresholdContactModel,
)

domain = Domain((8.0, 8.0, 8.0), (False, True, True))
simulator = MonteCarloSimulator(
    domain,
    [SphereSpec(radius=0.75)],
    ThresholdContactModel(0.1),
    axis="x",
)
result = simulator.estimate_probability([24], trials=8, seed=42)

assert result.trials == 8
assert 0.0 <= result.probability <= 1.0
print(f"p_hat={result.probability:.3f}; Wilson={tuple(result.confidence_interval)}")
