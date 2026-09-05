# MicroPerco 2.0.0 release readiness

Assessment date: 2026-09-05. Scope: the README v2.0 roadmap (resistor
networks, distance-dependent tunneling, effective conductivity, and directional
x/y/z measurements), compatible v1 workflows, and GitHub release artifacts.

## Local release gates

- 368 automated tests passed on Python 3.11.15, including 65 new transport and
  README cases and all 303 v1 regression tests.
- Ruff passed repository-wide; strict mypy passed all 48 package source files.
- The independent transport validation passed 32 random circuits against a
  dense incidence-matrix oracle. Maximum relative conductance disagreement was
  `5.80e-14` (acceptance `1e-10`).
- Mixed sphere/cylinder electrical networks matched the exhaustive geometry
  backend in all 24 combinations of eight periodic flags and three measurement
  axes. All 24 were conducting, with zero conductivity disagreement.
- The existing geometry/PBC/connectivity/RNG validation was rerun successfully.
  All 24 cylinder distances matched the SciPy oracle within `6.74e-10`;
  24 contact-search systems and 24 graph-solver cases retained parity.
- `python -m build --no-isolation` produced the v2.0.0 wheel and source archive
  using the existing development environment.
- The built wheel was installed with `--no-deps` into a temporary virtual
  environment sharing the already-validated runtime dependencies. From outside
  the checkout and without `PYTHONPATH`, its version, built-in validation, and
  the complete conductivity CLI example succeeded. This is an installed-package
  smoke test, not a fresh dependency-resolution test.
- Eight built-in validation checks passed, including the new analytic series
  circuit, finite-electrode chain conductivity, and exponential tunneling law.

GitHub Actions remains configured for Ubuntu and Windows with Python
3.10–3.12. CI builds distributions; tag builds also check metadata with Twine
and install the wheel in a fresh virtual environment. The local results above
do not stand in for those remote runs.

## Reproduce

```bash
python -m pip install -e '.[dev]'
python -m pytest
ruff check .
mypy src/microperco
python validation/run_transport_validation.py --output validation/transport_results.json
python validation/run_validation.py --output validation/v2_regression_results.json
python -m build
microperco --version
microperco validate
microperco conductivity configs/conductivity.yaml --output conductivity.json
```

See [transport validation](validation/TRANSPORT_VALIDATION.md),
[machine-readable transport results](validation/transport_results.json), and
[v2 regression results](validation/v2_regression_results.json).

## Model scope and compatibility

Particles are equipotential nodes; intrinsic particle resistance and contact
area effects are not modeled. The reported conductivity includes finite
electrode junction resistance. Three directional two-terminal measurements do
not constitute a full periodic homogenized tensor. Tunneling has an explicit
finite cutoff that users must check for convergence. Very ill-conditioned
networks fail explicitly when numerical conservation cannot be resolved.
See the full [conductivity methodology](docs/methodology/conductivity.md).

The optional conductivity configuration extends YAML schema version 1. All
existing v1 APIs and commands are retained. The generator still permits overlap;
the original geometric and statistical model limitations remain applicable.
No GPU, parallel Monte Carlo, or electrical-performance inverse-design claims
are introduced by this release. No new runtime dependency is required.

The [v1 release audit](docs/development/release_readiness_v1.md), including the
original source-workspace provenance, remains available as a historical record.
PyPI publishing remains outside the GitHub artifact-preparation workflow.
