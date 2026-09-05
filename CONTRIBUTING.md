# Contributing to MicroPerco

Thank you for helping improve MicroPerco. Open an issue before a large API or numerical-method change so that scope and validation can be agreed early.

## Development setup

Use Python 3.10–3.12. A dedicated virtual environment or an existing compatible Conda environment is recommended:

```bash
python -m pip install -e '.[dev]'
```

Do not add generated caches, local environments, credentials, or proprietary datasets.

The current release is v2.0.0. Start with the [documentation guide](docs/README.md) for current APIs, electrical-model scope, and validation evidence. Preserve compatibility with existing YAML schema version 1 configurations.

## Quality gates

Run these commands from the repository root:

```bash
python -m pytest
ruff check .
mypy src/microperco
python -m build
```

Run `python validation/run_validation.py` after geometry, PBC, graph, RNG, or statistical changes. Run `python benchmarks/run_benchmark.py` when a pull request makes a performance claim. Include fixed seeds, warmups, repeat counts, environment details, quartiles, and correctness parity with the reference backend.

Run `python validation/run_transport_validation.py` after conductance-law, resistor-network, electrode-boundary, or conductivity changes. Check analytic circuits, independent dense-solver agreement, Kirchhoff and power residuals, and optimized/reference network parity. Conductivity samples report descriptive standard errors; do not apply the percolation certification claims to electrical performance.

## Pull requests

Describe the behavior change, tests, numerical effect, performance effect, and API compatibility. New optimizations need an unoptimized oracle comparison. Geometry changes need analytic edge cases and an independent reference where practical. Statistical claims must state the interval, confidence interpretation, comparison family, and independent certification design.

Contributions are submitted under the repository's Apache-2.0 license.
