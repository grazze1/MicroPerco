# Development environment audit

## v2.0 development and release checks

Updated 2026-09-05. The v2.0.0 implementation reused the existing `mm` environment for 368 tests, Ruff, strict mypy, independent transport validation, geometry regression, and package builds. A temporary virtual environment sharing those runtime dependencies was created for the local installed-wheel smoke test. GitHub Actions separately passed fresh-environment package installation and all six Ubuntu/Windows, Python 3.10–3.12 test jobs; see [release readiness](../../RELEASE_READINESS.md).

The recorded v2 runtime versions are Python 3.11.15, NumPy 2.4.6, and SciPy 1.17.1; see [transport results](../../validation/transport_results.json). No new runtime dependency was added. HPP-FCL was not rerun for v2.0.

## Original environment selection (v1.0)

Audit date: 2026-09-04. The statements in this section describe the original environment selection before the v1 release.

Existing Conda environments were inspected before v1 development: `base`, `coco`, `isaaclab`, `lerobot`, `mm`, and `px4`. No environment was created or modified during that selection audit.

### Selected environments

The existing `mm` environment satisfied implementation, tests, plotting, static analysis, validation, benchmarking, and package building:

| Component | Version |
|---|---:|
| Python | 3.11.15 |
| NumPy | 2.4.6 |
| SciPy | 1.17.1 |
| PyYAML | 6.0.3 |
| Matplotlib | 3.11.1 |
| pytest | 8.4.2 |
| Ruff | 0.16.1 |
| mypy | 1.20.2 |
| build | 1.5.0 |
| setuptools | 83.0.0 |
| wheel | 0.47.0 |

The existing `isaaclab` environment was used only for optional HPP-FCL cross-validation:

| Component | Version |
|---|---:|
| Python | 3.11.15 |
| NumPy | 1.26.0 |
| SciPy | 1.15.3 |
| hpp-fcl | 2.4.4 |

## Commands

```bash
python -m pytest
ruff check .
mypy src/microperco
python validation/run_validation.py
python validation/run_transport_validation.py
python validation/run_external_geometry.py --backend hppfcl
python benchmarks/run_benchmark.py
python -m build
```

GitHub Actions uses ordinary CPython and pip rather than relying on either local Conda environment.

The external HPP-FCL and benchmark commands reproduce the retained historical workloads; they were not part of the v2 rerun. The transport validation command was added for v2.0.
