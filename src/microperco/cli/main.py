# SPDX-License-Identifier: Apache-2.0
"""MicroPerco command-line interface."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .._version import __version__
from ..benchmarking import benchmark_contact_search
from ..contact import ThresholdContactModel
from ..domain import Domain
from ..exceptions import MicroPercoError
from ..generation import SphereSpec
from ..io import dump_json, load_config
from ..optimization import optimize_mixture
from ..simulation import estimate_critical_loading, estimate_percolation_probability
from ..validation import run_validation_suite


def _write_result(result: object, output: Path | None) -> None:
    payload = dump_json(result, output)
    if output is None:
        sys.stdout.write(payload)


def _simulate(config_path: Path) -> object:
    config = load_config(config_path, operation="simulate")
    return estimate_percolation_probability(
        config.domain.to_domain(),
        populations=config.populations(),
        contact_model=config.contact.to_model(),
        axis=config.percolation.axis,
        trials=config.simulation.trials,
        seed=config.simulation.seed,
        neighbor_backend=config.simulation.neighbor_backend,
        mode=config.percolation.mode,
        wrapped_parent=config.percolation.wrapped_parent,
        confidence=config.simulation.confidence,
    )


def _critical(config_path: Path) -> object:
    config = load_config(config_path, operation="critical")
    assert config.critical is not None
    variable = config.particle_named(config.critical.population)
    fixed = tuple(
        particle.to_population() for particle in config.particles if particle.name != variable.name
    )
    return estimate_critical_loading(
        config.domain.to_domain(),
        variable.to_spec(),
        config.contact.to_model(),
        loading_grid=config.critical.counts,
        fixed_populations=fixed,
        target_probability=config.critical.target_probability,
        search_trials=config.critical.search_trials,
        certification_trials=config.critical.certification_trials,
        confidence=config.simulation.confidence,
        strategy=config.critical.strategy,
        seed=config.simulation.seed,
        axis=config.percolation.axis,
        neighbor_backend=config.simulation.neighbor_backend,
        mode=config.percolation.mode,
        wrapped_parent=config.percolation.wrapped_parent,
    )


def _optimize(config_path: Path) -> object:
    config = load_config(config_path, operation="optimize")
    assert config.optimization is not None
    bounds_by_name = {name: (low, high) for name, low, high in config.optimization.count_bounds}
    return optimize_mixture(
        config.domain.to_domain(),
        tuple(particle.to_spec() for particle in config.particles),
        tuple(bounds_by_name[particle.name] for particle in config.particles),
        config.contact.to_model(),
        target_probability=config.optimization.target_probability,
        screening_trials=config.optimization.search_trials,
        certification_trials=config.optimization.certification_trials,
        confidence=config.simulation.confidence,
        seed=config.simulation.seed,
        axis=config.percolation.axis,
        neighbor_backend=config.simulation.neighbor_backend,
        mode=config.percolation.mode,
        wrapped_parent=config.percolation.wrapped_parent,
        max_candidates=config.optimization.max_candidates,
    )


def _benchmark(config_path: Path | None) -> object:
    if config_path is None:
        return benchmark_contact_search(
            Domain((50.0, 50.0, 50.0), True),
            SphereSpec(0.5),
            ThresholdContactModel(0.2),
        )
    config = load_config(config_path, operation="benchmark")
    assert config.benchmark is not None
    selected = (
        config.particles[0]
        if config.benchmark.population is None
        else config.particle_named(config.benchmark.population)
    )
    return benchmark_contact_search(
        config.domain.to_domain(),
        selected.to_spec(),
        config.contact.to_model(),
        config.benchmark.particle_counts,
        repeats=config.benchmark.repeats,
        warmup=config.benchmark.warmup,
        seed=config.benchmark.seed,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="microperco",
        description="3D microstructure percolation simulation and inverse design",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name, help_text in (
        ("simulate", "estimate percolation probability"),
        ("critical", "estimate and certify a critical loading"),
        ("optimize", "search a bounded minimum-cost mixture"),
    ):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("config", type=Path)
        command.add_argument("--output", type=Path, help="write JSON to this path")

    validate = subparsers.add_parser("validate", help="run fast built-in checks")
    validate.add_argument("--output", type=Path, help="write JSON to this path")

    benchmark = subparsers.add_parser("benchmark", help="benchmark contact search")
    benchmark.add_argument("config", nargs="?", type=Path)
    benchmark.add_argument("--output", type=Path, help="write JSON to this path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process status code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    failed_validation = False
    try:
        if args.command == "simulate":
            result = _simulate(args.config)
        elif args.command == "critical":
            result = _critical(args.config)
        elif args.command == "optimize":
            result = _optimize(args.config)
        elif args.command == "validate":
            result = run_validation_suite()
            failed_validation = not result.passed
        else:
            result = _benchmark(args.config)
        _write_result(result, args.output)
        return 1 if failed_validation else 0
    except (MicroPercoError, OSError, ValueError) as exc:
        parser.exit(2, f"microperco: error: {exc}\n")


__all__ = ["build_parser", "main"]
