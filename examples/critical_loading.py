# SPDX-License-Identifier: Apache-2.0
"""Estimate a small illustrative critical-loading curve."""

from microperco import Domain, SphereSpec, ThresholdContactModel, estimate_critical_loading

result = estimate_critical_loading(
    Domain((8.0, 8.0, 8.0), (False, True, True)),
    SphereSpec(0.8),
    ThresholdContactModel(0.15),
    loading_grid=(20, 30, 40, 50, 60),
    target_probability=0.8,
    search_trials=20,
    certification_trials=40,
    seed=17,
)

print(result.status, result.critical_count)
for estimate in result.search_estimates:
    print(estimate.count, estimate.probability, tuple(estimate.confidence_interval))
