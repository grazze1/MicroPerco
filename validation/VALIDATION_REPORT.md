# MicroPerco 2.0.0 validation report

Current release validation date: 2026-09-05. The v2 results are [v2_regression_results.json](v2_regression_results.json) and [transport_results.json](transport_results.json). The original `validation_results.json`, `scipy_results.json`, and `hppfcl_results.json` retain the September 4 v1 baseline.

## Current release status

**PASS:** 368 automated tests, repository-wide Ruff, strict mypy, package builds, and installed-wheel checks. [Release-tag CI](https://github.com/grazze1/MicroPerco/actions/runs/33956868659) passed on Ubuntu and Windows with Python 3.10, 3.11, and 3.12. The [artifact workflow](https://github.com/grazze1/MicroPerco/actions/runs/33956868682) passed metadata validation and fresh-environment installation. See [release readiness](../RELEASE_READINESS.md) for the published release and complete gate evidence.

The v2 transport checks add 32 circuits compared against an independent dense solver (maximum relative conductance difference `5.80e-14`) and 24 conducting mixed sphere/cylinder networks across eight periodic flag combinations and three electrode directions (identical optimized/reference networks and conductivity). Analytic series/parallel/bridge circuits, floating components, tunneling laws, and seeded sampling are covered in the [transport validation report](TRANSPORT_VALIDATION.md).

The geometry, search, connectivity, and RNG checks below were rerun for v2 and are recorded in `v2_regression_results.json`. The external HPP-FCL and fixed-threshold comparison results are retained v1 evidence, explicitly identified below.

## Environment

- Linux 6.8.0-138-generic, x86-64
- Python 3.11.15
- NumPy 2.4.6 and SciPy 1.17.1 for primary validation
- Retained v1 external check: HPP-FCL 2.4.4, NumPy 1.26.0, and SciPy 1.15.3 in an existing isolated environment
- IEEE-754 float64 production geometry

## Automated tests

Command:

```bash
python -m pytest
```

Result: **368 passed** locally on CPython 3.11.15 and in the six release-tag CI environments above. This includes all 303 v1 regression tests and 65 new transport/README cases. Tests cover domain and particle
validation, geometry semantics, all eight PBC flag combinations, finite
electrode faces, parent fragments, contact thresholds, brute-force/cell-list
parity, face and winding connectivity, seeded generation, isotropy, stable
volume arithmetic, Wilson and exact intervals, PAVA, logistic/probit links,
critical loading, family-wise mixture design, YAML/JSON, CLI, visualization
exports, golden cases, and executable README examples, plus resistor circuits, finite-electrode conductivity, exponential tunneling, electrical conservation, axis/seed reproducibility, and conductivity CLI configuration.

## Geometry regression

Deterministic analytic cases passed for:

- sphere separation, tangency, and overlap;
- cylinder coaxial end caps and parallel sidewalls;
- perpendicular and near-parallel cylinders;
- sphere contacts at cylinder side and end cap;
- exact threshold acceptance;
- symmetry, non-negativity, and translation invariance;
- point/sphere/cylinder gaps to finite rectangles.

Extreme finite axis components (`1e300` and `1e-300`) normalize without
overflow. Exact-rational segment predicates, single-rounding large-gauge
periodic translations, and certified primal/dual fallback cases are covered.
Invalid or unrepresentable radii, lengths, volumes, AABBs, endpoints, bounds,
and lattice translations are rejected.

## Independent cylinder geometry

Twenty-four deterministic flat-cylinder pairs include analytic, nearly parallel, touching/overlapping, and seeded separated configurations.

| Reference | Cases | Mean absolute difference | Maximum absolute difference | Fixed-threshold decisions |
|---|---:|---:|---:|---:|
| SciPy SLSQP constrained distance | 24 | `1.976e-10` | `6.731e-10` | 120/120 agree |
| HPP-FCL 2.4.4 | 24 | `1.745e-6` | `2.000e-5` | 120/120 agree |

The table preserves the v1 external-reference comparison, including its fixed-threshold decisions. The v2 regression rerun independently repeated the SciPy distance comparison with maximum absolute error `6.731e-10`; it did not rerun HPP-FCL or regenerate the external fixed-threshold records.

Fixed thresholds were 0, 0.2, 1.0, 1.8, and 3.0, using the same scale-aware contact decision policy. HPP-FCL's largest difference occurs for exactly axis-aligned cylinder support queries and remains visible in the raw record; it is not used as MicroPerco's production backend. SciPy provides the tighter continuous optimization cross-check.

Acceptance criterion for the SciPy comparison was maximum absolute error below `2e-6`; observed error was more than three orders of magnitude smaller.

## Periodic boundaries and contact search

All x/y/z combinations—none, X, Y, Z, XY, XZ, YZ, and XYZ—are parameterized in automated tests. Dedicated cases cover single-axis seams, corner crossings, multi-axis images, multiple lattice periods for long cylinders, transverse electrode tiling, and same-parent self-edge exclusion.

The release validation constructed three independent mixed sphere/cylinder systems and evaluated each under all eight periodic flag combinations: **24 optimized/reference edge sets agreed exactly**.

| Aggregate work over 24 comparisons | Brute force | Cell list |
|---|---:|---:|
| Candidate tuples | 5,672 | 80 |
| Exact distance evaluations | 5,672 | 80 |

These counts demonstrate conservative pruning for the sampled sparse systems. Equality of accepted edges, not pair reduction, is the correctness requirement.

## Connectivity

- Known five-sphere chains span finite electrode faces along x, y, and z.
- Broken chains do not span.
- Union-Find and BFS agreed in 15 seeded face-to-face systems.
- Weighted Union-Find and BFS lattice potentials agreed in 9 periodic-wrapping systems.
- Winding vectors are oriented canonically with a positive component on the analysis axis.
- Face-to-face and periodic-wrapping interpretations remain separate.

## RNG reproducibility and isotropy

Repeated generation with seed 9001 produced byte-identical centers and axes. The 100,000-direction audit produced:

| Statistic | x | y | z |
|---|---:|---:|---:|
| Mean | 0.001110 | -0.001447 | 0.002054 |
| Second moment | 0.333092 | 0.333197 | 0.333712 |
| Positive fraction | 0.50179 | 0.50059 | 0.50053 |

The means are close to zero, second moments to `1/3`, and signs to one half, consistent with rotation-invariant sampling without imposing flaky tail thresholds.

## Statistical routines

Wilson and Clopper–Pearson intervals pass known values and boundary cases. Weighted PAVA passes closed-form blocks and preserves means for unweighted data. Logistic and probit fits are non-decreasing on increasing aggregated data. Deterministic synthetic trial evaluators verify critical-loading statuses, independent certification, and bounded-design outcomes.

For critical loading, the per-comparison confidence is derived from the actual one- or two-assertion final family. For optimization, it is fixed from all `2M` potential lower/upper assertions before any sampling. Result objects expose both family and per-comparison levels.

## Limitations of this validation

- HPP-FCL is optional and was exercised on Linux only.
- SciPy constrained optimization is an independent numerical reference, not a symbolic proof.
- Random parity tests cover many small systems but cannot exhaust all floating-point inputs.
- Geometry is binary64. If both GJK and its convex primal/dual fallback cannot
  close a distance bracket to the configured tolerance for a severely
  ill-conditioned valid query, MicroPerco raises `GeometryError` rather than
  silently returning an uncertified contact decision.
- Image enumeration and implicit loading grids have documented safety limits;
  configurations beyond them fail before pathological allocation or iteration.
- The built-in generator permits overlap and does not create equilibrated hard-particle packings.
- Statistical coverage guarantees assume the declared Bernoulli model and independent certification streams.
- Transport tests validate numerical solutions of the declared synthetic junction model, not agreement with experimental material conductivity. Conductivity includes finite electrode resistance and an explicit tunneling cutoff; its sample standard errors are descriptive, not performance certification.

Overall validation status: **PASS**.
