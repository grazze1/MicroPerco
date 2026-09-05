# SPDX-License-Identifier: Apache-2.0
"""Reproducible conductivity samples and descriptive sampling uncertainty."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..domain import Domain, normalize_axis
from ..exceptions import ConfigurationError, SimulationError
from ..generation import PopulationSpec, generate_particles
from ..seeding import SeedProvenance
from ..simulation._engine import SeedLike, normalize_seed, validate_positive_integer
from .conductivity import JunctionModel, _validate_model, analyze_conductivity
from .models import ConstantConductanceModel, finite_real


@dataclass(frozen=True, slots=True)
class ConductivityEstimate:
    axis: str
    samples: tuple[float, ...]
    mean: float
    standard_error: float | None
    conducting_trials: int


@dataclass(frozen=True, slots=True)
class ConductivityMonteCarloResult:
    trials: int
    seed: SeedProvenance
    domain: Domain
    populations: tuple[PopulationSpec, ...]
    conductance_model: JunctionModel
    electrode_model: JunctionModel
    applied_voltage: float
    neighbor_backend: str
    estimates: tuple[ConductivityEstimate, ...]


def estimate_conductivity(
    domain: Domain,
    populations: Sequence[PopulationSpec],
    conductance_model: JunctionModel | None = None,
    *,
    axes: Sequence[int | str] = ("x", "y", "z"),
    trials: int = 100,
    seed: SeedLike = None,
    electrode_model: JunctionModel | None = None,
    neighbor_backend: str = "cell_list",
    applied_voltage: float = 1.0,
) -> ConductivityMonteCarloResult:
    """Use a fresh child seed per trial and shared particles across its axes.

    Standard error is sample standard deviation / sqrt(trials), or null for
    one trial. It is descriptive and is not a probability certification.
    """

    count = validate_positive_integer(trials, "trials")
    selected = tuple("xyz"[normalize_axis(axis)] for axis in axes)
    if not selected or len(set(selected)) != len(selected):
        raise ConfigurationError("axes must be non-empty and unique")
    items = tuple(populations)
    if not items or not all(isinstance(item, PopulationSpec) for item in items):
        raise ConfigurationError("populations must be a non-empty PopulationSpec sequence")
    model = ConstantConductanceModel() if conductance_model is None else conductance_model
    electrodes = model if electrode_model is None else electrode_model
    _validate_model(model)
    _validate_model(electrodes)
    voltage = finite_real(applied_voltage, "applied_voltage")
    root, label = normalize_seed(seed)
    # Retain entropy even when the caller did not supply a seed.
    if label is None:
        _, label = normalize_seed(root)
    samples: dict[str, list[float]] = {axis: [] for axis in selected}
    for child in root.spawn(count):
        particles = generate_particles(domain, items, np.random.default_rng(child))
        for axis in selected:
            result = analyze_conductivity(
                particles,
                domain,
                model,
                axis=axis,
                electrode_model=electrodes,
                neighbor_backend=neighbor_backend,
                applied_voltage=voltage,
            )
            samples[axis].append(result.effective_conductivity)
    estimates: list[ConductivityEstimate] = []
    for axis, values in samples.items():
        scale = max(values)
        normalized = np.asarray(values) / scale if scale > 0 else np.zeros(count)
        mean = float(np.mean(normalized)) * scale
        error = None if count == 1 else float(np.std(normalized, ddof=1) / np.sqrt(count)) * scale
        if not np.isfinite(mean) or (error is not None and not np.isfinite(error)):
            raise SimulationError("conductivity sample statistics exceed float64")
        estimates.append(
            ConductivityEstimate(
                axis,
                tuple(values),
                mean,
                error,
                sum(value > 0 for value in values),
            )
        )
    return ConductivityMonteCarloResult(
        count,
        label,
        domain,
        items,
        model,
        electrodes,
        voltage,
        neighbor_backend,
        tuple(estimates),
    )
