# SPDX-License-Identifier: Apache-2.0
"""Repeatable contact-search benchmarks with robust timing summaries."""

from __future__ import annotations

import platform
import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .contact import ContactSearchResult, ThresholdContactModel, find_contacts
from .domain import Domain
from .exceptions import ConfigurationError
from .generation import ParticleSpec, PopulationSpec, generate_particles
from .simulation import BenchmarkResult


@dataclass(frozen=True, slots=True)
class BenchmarkSuite:
    """Benchmark records plus the environment needed to interpret them."""

    seed: int
    warmup: int
    repeats: int
    python: str
    platform: str
    numpy: str
    results: tuple[BenchmarkResult, ...]


def benchmark_contact_search(
    domain: Domain,
    particle_spec: ParticleSpec,
    contact_model: ThresholdContactModel,
    particle_counts: Sequence[int] = (10, 50, 100, 500, 1000),
    *,
    repeats: int = 5,
    warmup: int = 1,
    seed: int = 42,
) -> BenchmarkSuite:
    """Compare exhaustive and cell-list search on identical realizations."""

    if not isinstance(domain, Domain):
        raise ConfigurationError("domain must be a Domain")
    if not isinstance(contact_model, ThresholdContactModel):
        raise ConfigurationError("contact_model must be a ThresholdContactModel")
    counts = tuple(particle_counts)
    if not counts or any(
        isinstance(count, bool) or not isinstance(count, int) or count < 2 for count in counts
    ):
        raise ConfigurationError("particle_counts must contain integers >= 2")
    if len(set(counts)) != len(counts):
        raise ConfigurationError("particle_counts must be unique")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ConfigurationError("repeats must be a positive integer")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ConfigurationError("warmup must be a non-negative integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ConfigurationError("seed must be a non-negative integer")

    records: list[BenchmarkResult] = []
    root = np.random.SeedSequence(seed)
    for count, child in zip(counts, root.spawn(len(counts)), strict=True):
        particles = generate_particles(
            domain,
            (PopulationSpec(particle_spec, count, "benchmark"),),
            np.random.default_rng(child),
        )
        backend_data: dict[str, tuple[np.ndarray, ContactSearchResult]] = {}
        for backend in ("bruteforce", "cell_list"):
            for _ in range(warmup):
                find_contacts(particles, domain, contact_model, method=backend)
            elapsed: list[float] = []
            result = find_contacts(particles, domain, contact_model, method=backend)
            for _ in range(repeats):
                start = time.perf_counter()
                result = find_contacts(particles, domain, contact_model, method=backend)
                elapsed.append(time.perf_counter() - start)
            values = np.asarray(elapsed, dtype=np.float64)
            backend_data[backend] = (values, result)
        brute_median = float(np.median(backend_data["bruteforce"][0]))
        for backend in ("bruteforce", "cell_list"):
            values, raw_result = backend_data[backend]
            result = raw_result
            median = float(np.median(values))
            first_quartile, third_quartile = (
                float(value) for value in np.quantile(values, (0.25, 0.75))
            )
            records.append(
                BenchmarkResult(
                    count,
                    backend,
                    repeats,
                    median,
                    first_quartile,
                    third_quartile,
                    third_quartile - first_quartile,
                    result.candidate_pairs,
                    result.distance_evaluations,
                    1.0 if backend == "bruteforce" else brute_median / median,
                )
            )
    return BenchmarkSuite(
        seed,
        warmup,
        repeats,
        platform.python_version(),
        platform.platform(),
        np.__version__,
        tuple(records),
    )


__all__ = ["BenchmarkSuite", "benchmark_contact_search"]
