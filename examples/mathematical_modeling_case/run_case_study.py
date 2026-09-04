# SPDX-License-Identifier: Apache-2.0
"""Data-free Q1–Q4 workflow mapping; outputs are illustrative, not historical."""

from microperco import (
    Domain,
    MaterialSpec,
    Sphere,
    SphereSpec,
    ThresholdContactModel,
    analyze_percolation,
    estimate_critical_loading,
    estimate_percolation_probability,
    optimize_mixture,
)

domain = Domain((10.0, 10.0, 10.0), (False, True, True))
model = ThresholdContactModel(0.2)

# Q1: deterministic connectivity on an authorized/preprocessed realization.
chain = tuple(Sphere((x, 0.0, 0.0), 1.0, i) for i, x in enumerate((-4, -2, 0, 2, 4)))
q1 = analyze_percolation(chain, domain, model)

# Q2: forward probability for a declared loading.
spec = SphereSpec(1.0)
q2 = estimate_percolation_probability(domain, spec, 60, model, trials=12, seed=42)

# Q3: nested grid search plus independent certification.
q3 = estimate_critical_loading(
    domain,
    spec,
    model,
    loading_grid=(40, 50, 60, 70, 80),
    target_probability=0.8,
    search_trials=12,
    certification_trials=20,
    seed=43,
)

# Q4: bounded two-material design with explicit nonzero material costs.
# Real studies should use enough trials and wider bounds for certification.
cheap = SphereSpec(0.75, MaterialSpec("small_bead", 0.05))
effective = SphereSpec(1.0, MaterialSpec("large_bead", 0.20))
q4 = optimize_mixture(
    domain,
    (cheap, effective),
    ((0, 2), (56, 60)),
    model,
    target_probability=0.5,
    screening_trials=4,
    certification_trials=8,
    seed=44,
)

print("Q1", q1.percolates, q1.spanning_path)
print("Q2", q2.probability, tuple(q2.confidence_interval))
print("Q3", q3.status, q3.critical_count)
print("Q4", q4.status, q4.counts, q4.cost)
