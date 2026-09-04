# SPDX-License-Identifier: Apache-2.0
"""Run a bounded two-material design example."""

from microperco import (
    Domain,
    MaterialSpec,
    SphereSpec,
    ThresholdContactModel,
    optimize_mixture,
)

cheap = SphereSpec(0.55, MaterialSpec("cheap", 0.2))
effective = SphereSpec(0.9, MaterialSpec("effective", 1.0))
result = optimize_mixture(
    Domain((7.0, 7.0, 7.0), (False, True, True)),
    (cheap, effective),
    ((0, 3), (0, 2)),
    ThresholdContactModel(0.15),
    target_probability=0.5,
    screening_trials=5,
    certification_trials=10,
    seed=11,
)

print(result.status, result.counts, result.cost)
print(f"family confidence={result.confidence:.3f}")
print(f"per-comparison confidence={result.per_comparison_confidence:.6f}")
