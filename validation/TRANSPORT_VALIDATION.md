# v2.0 transport validation

Recorded on 2026-09-05 with Python 3.11.15, NumPy 2.4.6, and SciPy 1.17.1
on Linux x86_64. The exact machine-readable results are
[transport_results.json](transport_results.json).

This is the transport part of the [current v2.0.0 validation](VALIDATION_REPORT.md). [Release-tag CI](https://github.com/grazze1/MicroPerco/actions/runs/33956868659) passed all 368 tests on Ubuntu/Windows and Python 3.10–3.12; the [release artifact workflow](https://github.com/grazze1/MicroPerco/actions/runs/33956868682) passed package and installation checks.

```bash
python validation/run_transport_validation.py --output validation/transport_results.json
python -m pytest tests/unit/test_transport_network.py tests/integration/test_conductivity.py
```

## Independent circuit oracle

For 32 seeds (2000–2031), the validation creates connected random resistor
graphs with 6–37 nodes and log-uniform junction conductances from `exp(-4)` to
`exp(4)`. The reference constructs the dense Laplacian as `B.T @ diag(g) @ B`
from a signed incidence matrix, solves the interior Dirichlet problem with
NumPy, and computes terminal current directly. The production solver assembles
a sparse free-node matrix from edges and computes conductance from energy.

| Quantity | Maximum error | Acceptance |
|---|---:|---:|
| Absolute unit-voltage node difference | `6.61e-15` | `<1e-10` |
| Relative effective conductance difference | `5.80e-14` | `<1e-10` |
| Relative free-node Kirchhoff residual | `2.95e-14` | `<1e-10` |

Analytic tests independently cover series, parallel, a balanced Wheatstone
bridge, dangling branches, isolated components, voltage scaling, and uniform
conductance scaling by factors down to `1e-250` and up to `1e250`. Failure tests
exercise unrepresentable tunneling, network dynamic range, and result overflow.

## Geometric electrical-network parity

For each of eight periodic flag combinations, seed 2026 generates eight spheres
and six flat-ended cylinders in a side-4 domain. All three electrode directions
are measured with `g0=1`, `decay_length=0.4`, and `cutoff=0.6`. The optimized
cell list and brute-force oracle produce identical resistor networks in all
24 cases. All 24 conduct; maximum conductivity difference is exactly zero.

Integration tests additionally cover the finite-electrode geometry factor,
an analytic two-sphere tunneling circuit, a spanning flat cylinder, off-face
rejection, multiple transverse image channels in parallel, periodic translation,
equal-parent equipotential fragments, empty systems, retained seed provenance,
axis-order invariance, and strict CLI configuration/JSON behavior.

## Scope of the evidence

These are numerical correctness checks on synthetic models. The independent
linear solver does not validate the physical junction law against experiment.
Geometry backend parity tests share the established narrow-phase kernel;
the separate [v2 regression rerun](v2_regression_results.json) also checks that
kernel against the existing independent SciPy geometry oracle. The original
[geometry validation report](VALIDATION_REPORT.md) supplies the broader v1
evidence. No transport speedup or universal accuracy claim is made.
