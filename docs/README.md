# MicroPerco 2.0.0 documentation

Current release: [v2.0.0](https://github.com/grazze1/MicroPerco/releases/tag/v2.0.0), published September 5, 2026. The repository homepage and current guides describe this release. YAML configuration remains `schema_version: 1`; that value is independent of the package version.

## Start here

- [Repository overview, installation, and quick starts](../README.md)
- [Changes in v2.0.0](../CHANGELOG.md)
- [Conductivity Python example](../examples/conductivity.py) and [CLI configuration](../configs/conductivity.yaml)
- [Percolation and inverse-design configuration](../configs/example.yaml)
- [Release status and reproducible checks](../RELEASE_READINESS.md)

## Current capabilities

| Workflow | API / CLI | Guide |
|---|---|---|
| Explicit resistor circuits | `ResistorNetwork`, `solve_resistor_network` | [Kirchhoff solution](methodology/conductivity.md#kirchhoff-solution-and-numerical-checks) |
| Constant or exponential tunneling junctions | `ConstantConductanceModel`, `TunnelingConductanceModel` | [Laws and units](methodology/conductivity.md#physical-model-and-units) |
| Finite-electrode conductivity | `analyze_conductivity`, `analyze_directional_conductivity` | [Conductivity](methodology/conductivity.md) |
| Conductivity sampling | `estimate_conductivity`, `microperco conductivity` | [Monte Carlo](methodology/monte_carlo.md) |
| Binary face/wrapping connectivity | `analyze_percolation` | [Percolation](methodology/percolation.md) |
| Probability sampling | `estimate_percolation_probability`, `microperco simulate` | [Monte Carlo](methodology/monte_carlo.md) |
| Critical percolation loading | `estimate_critical_loading`, `microperco critical` | [Critical loading](methodology/critical_loading.md) |
| Bounded mixture cost optimization | `optimize_mixture`, `microperco optimize` | [Inverse design](methodology/inverse_design.md) |

Shared foundations: [sphere and flat-cylinder geometry](methodology/geometry.md) and [periodic boundaries](methodology/periodic_boundary.md).

The transport model treats particles as equipotential, includes finite electrode junction resistance, and reports three directional two-terminal measurements. Intrinsic particle resistance, full periodic conductivity tensors, parallel Monte Carlo, and conductivity-constrained inverse design remain future work. Percolation certification does not certify conductivity.

## Validation and development

- [Current validation report](../validation/VALIDATION_REPORT.md): 368 tests and release-tag CI on Ubuntu/Windows with Python 3.10–3.12.
- [Transport validation](../validation/TRANSPORT_VALIDATION.md): 32 independent circuit comparisons and 24 mixed-geometry electrical-network comparisons.
- [Contribution workflow](../CONTRIBUTING.md), [development environment](development/environment_audit.md), and [security support](../SECURITY.md).
- [Case-study scaffold](../examples/mathematical_modeling_case/README.md): retained Q1–Q4 percolation workflow and a separate v2 conductivity extension.
- [Citation metadata](../CITATION.cff), [license](../LICENSE), and [third-party dependency audit](legal/third_party.md).

## Historical evidence

These records retain the versions, dates, and measurements from their original runs. They are not new v2 transport measurements.

| Record | Scope |
|---|---|
| [v1 release audit](development/release_readiness_v1.md) | Archived 1.0.0 release and original source-workspace integrity checks |
| [Source snapshot](provenance/source_snapshot.md) and [algorithm audit](provenance/source_algorithm_audit.md) | Provenance of the v1 rebuild; the current transport extension is documented separately |
| [Contact-search benchmark](../benchmarks/BENCHMARK_REPORT.md) | September 4, 2026 sparse-sphere performance baseline |
| [Figure contract](FIGURE_CONTRACT.md) | Existing geometry, binary connectivity, probability, and benchmark figures |

The [current validation report](../validation/VALIDATION_REPORT.md) distinguishes the v2 geometry regression rerun from the retained v1 external HPP-FCL comparison.
