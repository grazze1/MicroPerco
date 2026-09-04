# Contributing to MicroPerco

Thank you for helping improve MicroPerco. Open an issue before a large API or numerical-method change so that scope and validation can be agreed early.

## Development setup

Use Python 3.10–3.12. A dedicated virtual environment or an existing compatible Conda environment is recommended:

```bash
python -m pip install -e '.[dev]'
```

Do not add generated caches, local environments, credentials, or proprietary datasets.

## Quality gates

Run these commands from the repository root:

```bash
python -m pytest
ruff check .
mypy src/microperco
python -m build
```

Run `python validation/run_validation.py` after geometry, PBC, graph, RNG, or statistical changes. Run `python benchmarks/run_benchmark.py` when a pull request makes a performance claim. Include fixed seeds, warmups, repeat counts, environment details, quartiles, and correctness parity with the reference backend.

## Pull requests

Describe the behavior change, tests, numerical effect, performance effect, and API compatibility. New optimizations need an unoptimized oracle comparison. Geometry changes need analytic edge cases and an independent reference where practical. Statistical claims must state the interval, confidence interpretation, comparison family, and independent certification design.

Contributions are submitted under the repository's Apache-2.0 license.
