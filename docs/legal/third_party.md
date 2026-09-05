# Third-party and license audit

## Repository content

No third-party source is vendored. No code, figures, problem statements, spreadsheets, or datasets from the unlicensed source workspace are distributed. All repository-authored code is Apache-2.0 and carries SPDX identifiers where the file format supports comments.

## Runtime dependencies

| Dependency | Role | Upstream license summary | Distributed here? |
|---|---|---|---|
| NumPy | float64 arrays and random generators | BSD-3-Clause with compatible bundled notices | No |
| SciPy | statistical distributions, fitted link models, geometry fallback/validation, and v2 sparse resistor-network solves | BSD-3-Clause with compatible binary-runtime notices | No |
| PyYAML | configuration parsing | MIT | No |

## Optional/development dependencies

Matplotlib is used for figures under its PSF/BSD-compatible license. pytest, coverage, Ruff, mypy, build, setuptools, and wheel are development tools and are not vendored. HPP-FCL 2.4.4 is an optional BSD-3-Clause validation backend and is not a runtime dependency.

Python packaging resolves dependencies separately; their upstream license files and binary notices remain the responsibility of their distributions. MicroPerco does not copy or relabel them.

## Assets

The four README figures are generated solely from MicroPerco code and recorded synthetic simulations with fixed seeds. PNG, SVG, and PDF exports share the repository's Apache-2.0 license. They contain no external icons, fonts embedded as source assets, or proprietary data.

Audit result: no incompatible or unknown-license material is included.
