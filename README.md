# MicroPerco 2.0.0

[![Release v2.0.0](https://img.shields.io/badge/release-v2.0.0-blue.svg)](https://github.com/grazze1/MicroPerco/releases/tag/v2.0.0)
[![CI](https://github.com/grazze1/MicroPerco/actions/workflows/ci.yml/badge.svg)](https://github.com/grazze1/MicroPerco/actions/workflows/ci.yml)
[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10--3.12-3776AB.svg)](https://www.python.org/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

MicroPerco is a reusable scientific-computing framework for three-dimensional microstructure percolation, tunneling conductivity, Monte Carlo uncertainty quantification, critical-loading estimation, and bounded inverse design.

**Current release: [v2.0.0](https://github.com/grazze1/MicroPerco/releases/tag/v2.0.0), published September 5, 2026.** Resistor networks, distance-dependent tunneling, finite-electrode effective conductivity, and directional `sigma_x` / `sigma_y` / `sigma_z` are available now. Existing v1 APIs and configurations remain supported.

[Documentation guide](docs/README.md) · [Release notes](CHANGELOG.md) · [Validation](validation/VALIDATION_REPORT.md) · [Download packages](https://github.com/grazze1/MicroPerco/releases/tag/v2.0.0)

![A seeded mixed-particle MicroPerco realization with finite electrode faces](docs/assets/microstructure_3d.png)

## Overview

MicroPerco turns finite spheres and flat-ended cylinders into auditable contact graphs and resistor networks inside an orthorhombic domain. It calculates binary percolation and finite-electrode electrical transport, with independently periodic x/y/z boundaries, topological winding, an exhaustive geometry oracle, an optimized cell-list search, and seeded Monte Carlo trials. Critical-loading and mixture-design certification apply to percolation probability.

It is a library and CLI, not a hard-coded solution to one dataset. Units are user-defined but must be internally consistent.

## Why MicroPerco

Microstructure simulations are unusually sensitive to small geometry and boundary-condition shortcuts. Treating a cylinder as a capsule changes end-cap distances; treating a finite electrode as an infinite plane creates false contacts; searching only 27 periodic images can miss long-particle interactions; selecting an optimum from noisy point estimates can overstate confidence.

MicroPerco makes those choices explicit and records enough evidence to audit each result:

- exact accepted edges and their lattice shifts;
- broad-phase candidate and narrow-phase evaluation counts;
- deterministic representative spanning paths or winding vectors;
- point estimates, Wilson intervals, and exact intervals;
- declared family confidence, per-comparison confidence, and comparison counts;
- independent certification samples separated from search/screening samples;
- electrical junction gaps and conductances, node voltages, branch currents, Joule power, and conservation diagnostics;
- directional conductivity samples, means, standard errors, and reproducible seed provenance.

## Features

- Immutable, validated `Sphere`, finite `Cylinder`, `Domain`, material, and population models.
- Analytic sphere–sphere and sphere–cylinder gaps; support-map GJK with a certified convex fallback for flat-cylinder pair and electrode queries.
- Finite rectangular electrode faces, including transverse periodic tiling.
- All eight periodic-axis combinations and dynamically complete image ranges.
- Brute-force reference and periodic AABB cell-list contact searches.
- Union-Find plus independent BFS connectivity checks; weighted lattice winding detection.
- Uniform centers and rotation-invariant cylinder orientations from reproducible NumPy streams.
- Monte Carlo probability estimation with Wilson and Clopper–Pearson intervals.
- Nested PAVA/logistic/probit critical search with independent family-wise certification.
- Complete bounded integer mixture search with cost ordering and fixed-family error control.
- Sparse resistor networks with node voltages, branch currents, Joule power, and conservation checks.
- Constant or exponentially distance-dependent junction conductance, independent electrode laws, and directional conductivity.
- Seeded conductivity sampling with shared realizations across axes and recorded samples and standard errors.
- Strict schema-versioned YAML, stable JSON CLI output, plotting, validation, and benchmarks.

## Installation

MicroPerco requires Python 3.10 or newer; CI tests Python 3.10–3.12.

Install the published v2.0.0 wheel directly from GitHub:

```bash
python -m pip install https://github.com/grazze1/MicroPerco/releases/download/v2.0.0/microperco-2.0.0-py3-none-any.whl
microperco --version
```

The [release page](https://github.com/grazze1/MicroPerco/releases/tag/v2.0.0) also provides the source archive and SHA-256 checksums. To work from the current repository:

```bash
git clone https://github.com/grazze1/MicroPerco.git
cd MicroPerco
python -m pip install .
```

From the cloned repository, install plotting or development support when needed:

```bash
python -m pip install '.[plot]'
python -m pip install -e '.[dev]'
```

HPP-FCL is optional and used only as an external validation backend. It is not needed by the core package.

## Quick Start: Directional Conductivity

Measure all three directions on the same explicit microstructure:

```python
from microperco import (
    ConstantConductanceModel,
    Domain,
    Sphere,
    TunnelingConductanceModel,
    analyze_directional_conductivity,
)

particles = (Sphere((-1.25, 0, 0), 1), Sphere((1.25, 0, 0), 1))
conductivity = analyze_directional_conductivity(
    particles,
    Domain((4.5, 3.0, 3.0), False),
    TunnelingConductanceModel(
        contact_conductance=2.0, decay_length=0.5, cutoff=0.5,
    ),
    electrode_model=ConstantConductanceModel(contact_conductance=10.0),
)
print(conductivity.sigma_x)  # approximately 0.12838526
print(conductivity.sigma_y, conductivity.sigma_z)  # 0.0, 0.0
```

The tunneling law is `g(d) = g0 * exp(-2*d/decay_length)` up to an explicit
cutoff. Particles are equipotential; particle and electrode junctions have
finite resistance. Each measurement opens its axis and retains transverse
periodicity. The apparent conductivity `sigma = G*L/A` includes electrode
resistance. Lengths in metres and conductances in siemens give S/m.

`analyze_conductivity` returns one axis with the complete resistor network,
junction geometry, node voltages, branch currents, and conservation diagnostics.
`ResistorNetwork` and `solve_resistor_network` also work on explicit circuits.
Disconnected networks return zero conductivity; floating potentials are JSON
`null`. Numerically unresolved networks raise an error.

For Monte Carlo sampling, use `estimate_conductivity` or the CLI configuration
below. Samples share particles across axes, retain seed provenance, and report
means and standard errors. These statistics are not performance certification.
The three finite-electrode measurements are not a full homogenized tensor.

See [conductivity methodology and model limitations](docs/methodology/conductivity.md).

## Percolation Quick Start

This complete example is exercised by `tests/test_readme_examples.py`:

```python
from microperco import (
    Domain,
    SphereSpec,
    ThresholdContactModel,
    MonteCarloSimulator,
)

domain = Domain(
    size=(8.0, 8.0, 8.0),
    periodic=(False, True, True),
)

simulator = MonteCarloSimulator(
    domain=domain,
    particle_specs=[SphereSpec(radius=0.75)],
    contact_model=ThresholdContactModel(0.10),
    axis="x",
)

result = simulator.estimate_probability(
    particle_counts=[24],
    trials=16,
    seed=42,
)

print(result.probability)
print(result.confidence_interval)
```

For a single explicit realization:

```python
from microperco import PopulationSpec, analyze_percolation, generate_microstructure

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
print(graph.percolates, graph.spanning_path)
```

## Core Concepts

| Concept | Meaning |
|---|---|
| Domain | A finite orthorhombic box with independent periodic flags and a configurable center |
| Particle | A closed sphere or finite right circular cylinder with flat end caps |
| Contact | Exact non-negative surface gap no larger than a threshold plus scale-aware tolerance |
| Parent | Optional identity used to group explicitly reconstructed periodic fragments |
| Face percolation | A contact-graph path between distinct finite lower and upper electrode faces |
| Periodic wrapping | A graph cycle with non-zero integer lattice winding along the selected periodic axis |
| Nominal volume fraction | Sum of particle volumes divided by domain volume; overlaps are not subtracted |
| Electrical junction | A finite constant or exponentially distance-dependent conductance between equipotential particle nodes or an electrode |
| Directional conductivity | Apparent two-terminal `sigma = G*L/A` along x, y, or z, including finite electrode junction resistance |

MicroPerco never infers units. Domain dimensions, radii, lengths, and contact threshold must use the same length unit; material cost per volume must match the desired cost unit.

## Geometry Kernel

Sphere distances and the point-to-flat-cylinder primitive are analytic.
Cylinder–cylinder distance uses a scaled float64 support-map GJK algorithm. If
an ill-conditioned GJK iteration stalls, a convex primal/dual fallback returns
only after its global distance bracket meets the configured tolerance;
otherwise the query fails explicitly. The cell list performs conservative AABB
pruning, and both search backends use the same narrow-phase kernel.

The default `NumericalPolicy` uses a `1e-9` absolute tolerance, a `1e-12` relative tolerance, and at most 128 GJK iterations. Invalid radii, lengths, axes, derived volumes, AABBs, endpoints, and lattice translations fail explicitly.

See [geometry methodology](docs/methodology/geometry.md).

## Periodic Boundary Conditions

`Domain.periodic` is a three-boolean tuple. Relevant lattice translations are solved from AABB overlap intervals, so long bodies can require images beyond immediate neighbors.

Face analysis opens the analysis axis and retains transverse PBC. This preserves two distinct electrodes. Equal non-null `parent_id` values always provide fragment continuity without self-contact edges. For binary percolation, the historical seam-to-both-electrodes shortcut is enabled by `wrapped_parent=True`. Conductivity uses finite electrode contacts and does not enable this shortcut.

See [periodic-boundary methodology](docs/methodology/periodic_boundary.md).

## Percolation Analysis

```python
result = analyze_percolation(
    particles,
    domain,
    ThresholdContactModel(0.10),
    axis="x",
    mode="face_to_face",  # or "periodic_wrap"
    search="cell_list",  # or "bruteforce"
    solver="union_find",  # independently checked by BFS
)
```

`PercolationResult` is immutable and includes the decision, mode, axis,
ordinary contact edges, periodic topology evidence, electrode memberships,
representative path, winding, component count, and search-work counters.

![A deterministic face-to-face spanning chain highlighted in red](docs/assets/spanning_cluster.png)

## Monte Carlo Simulation

Every trial gets a child `SeedSequence`; the same seed and configuration reproduce the same stream. Results contain both a two-sided Wilson interval and an exact Clopper–Pearson interval. These quantify sampling error under the model, not model-form uncertainty.

The built-in generator uses independent uniform centers and isotropic cylinder axes. It does not impose excluded-volume packing or prevent a body centered inside a non-periodic domain from extending outside it.

See [Monte Carlo methodology](docs/methodology/monte_carlo.md).

## Critical Loading

```python
from microperco import estimate_critical_loading

critical = estimate_critical_loading(
    domain,
    SphereSpec(0.75),
    ThresholdContactModel(0.10),
    loading_grid=(20, 25, 30, 35, 40),
    target_probability=0.90,
    search_trials=1000,
    certification_trials=5000,
    confidence=0.95,
    seed=42,
)
```

Search trials share nested particle prefixes, then PAVA (or a monotone logistic/probit link) proposes the first crossing. New random streams certify the candidate lower bound and, where applicable, the predecessor upper bound. Bonferroni allocation controls the one- or two-assertion family. A result that lacks both required inequalities is `INCONCLUSIVE`.

![Recorded Monte Carlo estimates, asymmetric Wilson intervals, and monotone fit](docs/assets/percolation_curve.png)

See [critical-loading methodology](docs/methodology/critical_loading.md).

## Inverse Design

`optimize_mixture` enumerates the full Cartesian product of inclusive count bounds, orders candidates by nominal material cost, screens them, and independently certifies them. If the bounded family has `M` candidates, exact one-sided certification confidence is fixed at `1 - (1 - family_confidence) / (2M)` before sampling.

`CERTIFIED_OPTIMAL` means feasibility was certified and every strictly cheaper evaluated design was excluded at the declared family-wise level. It means optimal only within the provided finite bounds and stochastic model.

See [inverse-design methodology](docs/methodology/inverse_design.md).

## CLI

All computational commands emit standards-compliant JSON (`NaN` and infinity are rejected):

```bash
microperco simulate configs/example.yaml
microperco conductivity configs/conductivity.yaml
microperco critical configs/example.yaml
microperco optimize configs/example.yaml
microperco validate
microperco benchmark configs/benchmark.yaml
```

Add `--output result.json` to write JSON to a file. Use `microperco COMMAND --help` for command-specific arguments.

## Configuration

Every YAML document starts with `schema_version: 1`. Unknown keys are errors. The main sections are `domain`, `particles`, `contact`, `percolation`, `simulation`, and optional `critical`, `optimization`, `benchmark`, and `conductivity` blocks. Existing v1 configurations remain valid. The new [conductivity example](configs/conductivity.yaml) declares particle and electrode conductance laws explicitly.

See [the complete generic example](configs/example.yaml). Configuration validation checks shapes, finite physical values, unique population names, modes/backends, count grids and bounds, probability levels, and operation-specific requirements.

## Examples

- [`examples/smoke_test.py`](examples/smoke_test.py): minimal end-to-end Monte Carlo run.
- [`examples/basic_simulation.py`](examples/basic_simulation.py): one mixed realization and graph audit.
- [`examples/critical_loading.py`](examples/critical_loading.py): nested critical-loading workflow.
- [`examples/inverse_design.py`](examples/inverse_design.py): bounded two-material design.
- [`examples/conductivity.py`](examples/conductivity.py): analytic two-particle tunneling and directional conductivity.
- [`examples/mathematical_modeling_case/`](examples/mathematical_modeling_case/): data-free Q1–Q4 mapping; restricted source attachments are not redistributed.

## Validation

The v2.0.0 release passed **368 automated tests**, including all 303 original regression tests and 65 new transport/README cases. [Release-tag CI](https://github.com/grazze1/MicroPerco/actions/runs/33956868659) passed on Ubuntu and Windows with Python 3.10, 3.11, and 3.12; Ruff, strict mypy, and package builds passed. The [release artifact workflow](https://github.com/grazze1/MicroPerco/actions/runs/33956868682) also passed metadata checks and installed-wheel validation.

- 32 resistor networks versus an independent dense solver: maximum relative conductance difference `5.80e-14`.
- 24 mixed sphere/cylinder transport cases across all eight PBC combinations and three measurement axes: identical optimized/reference resistor networks and conductivity.
- Geometry regression rerun: 24 cylinder pairs versus SciPy with maximum absolute gap error `6.73e-10`; 24 contact-search comparisons and 24 face/wrapping solver cases agreed.

Full commands, evidence, and limitations are in [the validation report](validation/VALIDATION_REPORT.md), [transport validation](validation/TRANSPORT_VALIDATION.md), and [release readiness](RELEASE_READINESS.md). These results use synthetic models and do not constitute experimental validation of a material's conductivity.

## Benchmarks

The following contact-search measurements were recorded for v1.0 on September 4, 2026 and retained as a historical baseline. v2.0 does not introduce a new transport-performance benchmark.

On the recorded Linux/Python 3.11.15 environment, a sparse periodic sphere benchmark used seed 42, one warmup, and five timed repeats per backend. At `N=1000`, brute force evaluated 499,500 pairs in a median `19.764 s`; the cell list evaluated 52 pairs in `0.04088 s`, a `483.4×` observed speedup for this realization. This is an empirical sparse-case result, not a universal complexity guarantee.

![Runtime quartiles and search work counters for the recorded benchmark](docs/assets/benchmark.png)

See [the benchmark report](benchmarks/BENCHMARK_REPORT.md) and machine-readable [`benchmark_results.json`](benchmarks/benchmark_results.json).

## Reproducibility

```bash
python -m pytest
python validation/run_validation.py
python validation/run_transport_validation.py
python validation/run_external_geometry.py --backend scipy
python benchmarks/run_benchmark.py
```

Fixed seeds, exact configuration, quartiles, dependency versions, and immutable result records are retained. Search and certification use independent root-seed branches. The source-integrity and environment records are in [`docs/provenance`](docs/provenance/) and [`docs/development`](docs/development/).

## Project Structure

```text
src/microperco/       geometry, contact, graph, transport, simulation, statistics, I/O, CLI, plots
tests/                unit, integration, regression, and golden cases
validation/           independent geometry and cross-backend validation
benchmarks/           repeatable performance measurements
configs/              generic schema-versioned YAML examples
examples/             runnable API workflows and a data-free case study
docs/                 methodology, provenance, legal, development, and generated assets
.github/               CI, release-artifact preparation, and contribution templates
```

## Current Scope and Roadmap

v2.0.0 implements resistor networks, distance-dependent tunneling, effective conductivity, and directional x/y/z measurements. The current transport model treats particles as equipotential and includes finite electrode resistance.

The following remain future work:

- **v2.x:** additional particle distributions, parallel Monte Carlo, intrinsic particle resistance, and fully periodic conductivity homogenization.
- **Future:** multi-objective design, surrogate models, Bayesian optimization, and GPU acceleration.

## Citation

Use the metadata in [`CITATION.cff`](CITATION.cff). Until a DOI-backed archive exists, cite the software title, version 2.0.0, repository URL, and access date.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for development gates and numerical-evidence expectations. Conduct is governed by [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md); private vulnerability reports follow [`SECURITY.md`](SECURITY.md).

## License

Copyright 2026 MicroPerco contributors. Licensed under the [Apache License 2.0](LICENSE). Dependency and asset provenance is documented in [`docs/legal/third_party.md`](docs/legal/third_party.md).
