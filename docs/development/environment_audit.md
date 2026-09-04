# Development environment audit

Audit date: 2026-09-04.

Existing Conda environments were inspected before development: `base`, `coco`, `isaaclab`, `lerobot`, `mm`, and `px4`. No environment was created and no existing environment was modified.

## Selected environments

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
python validation/run_external_geometry.py --backend hppfcl
python benchmarks/run_benchmark.py
python -m build
```

GitHub Actions uses ordinary CPython and pip rather than relying on either local Conda environment.
