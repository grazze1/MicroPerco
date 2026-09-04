# SPDX-License-Identifier: Apache-2.0
"""Generate one mixed realization and inspect its percolation graph."""

from microperco import (
    CylinderSpec,
    Domain,
    PopulationSpec,
    SphereSpec,
    ThresholdContactModel,
    analyze_percolation,
    generate_microstructure,
)

domain = Domain((10.0, 10.0, 10.0), (False, True, True))
populations = (
    PopulationSpec(CylinderSpec(radius=0.3, length=4.0), 25, "fibers"),
    PopulationSpec(SphereSpec(radius=0.45), 20, "beads"),
)
sample = generate_microstructure(domain, populations, seed=2026)
result = analyze_percolation(
    sample.particles,
    domain,
    ThresholdContactModel(0.15),
    axis="x",
)

print(f"percolates={result.percolates}")
print(f"contacts={result.edge_count}, components={result.component_count}")
print(f"candidate_pairs={result.candidate_pairs}, exact={result.distance_evaluations}")
