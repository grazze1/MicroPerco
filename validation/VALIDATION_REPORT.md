# MicroPerco 1.0 validation report

Validation date: 2026-09-04. The machine-readable records are `validation_results.json`, `scipy_results.json`, and `hppfcl_results.json` in this directory.

## Environment

- Linux 6.8.0-138-generic, x86-64
- Python 3.11.15
- NumPy 2.4.6 and SciPy 1.17.1 for primary validation
- HPP-FCL 2.4.4, NumPy 1.26.0, and SciPy 1.15.3 in an existing isolated environment
- IEEE-754 float64 production geometry

## Automated tests

Command:

```bash
python -m pytest
```

Result: **303 passed** on CPython 3.11.15. The same 303 tests also passed on
CPython 3.10.12 with the user site disabled and on CPython 3.12.14. Tests cover domain and particle
validation, geometry semantics, all eight PBC flag combinations, finite
electrode faces, parent fragments, contact thresholds, brute-force/cell-list
parity, face and winding connectivity, seeded generation, isotropy, stable
volume arithmetic, Wilson and exact intervals, PAVA, logistic/probit links,
critical loading, family-wise mixture design, YAML/JSON, CLI, visualization
exports, golden cases, and executable README examples.

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

Overall validation status: **PASS**.
