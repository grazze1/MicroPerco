# SPDX-License-Identifier: Apache-2.0
"""Strict schema-versioned YAML configuration."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode

from ..contact import ThresholdContactModel
from ..domain import Domain
from ..exceptions import ConfigurationError
from ..generation import (
    CylinderSpec,
    MaterialSpec,
    ParticleSpec,
    PopulationSpec,
    SphereSpec,
)
from ..transport import ConstantConductanceModel, TunnelingConductanceModel


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects ambiguous mapping definitions."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    loader.flatten_mapping(node)
    result: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate mapping key {key!r}",
                key_node.start_mark,
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ConfigurationError(f"{name} must be a mapping with string keys")
    return cast(Mapping[str, object], value)


def _reject_unknown(mapping: Mapping[str, object], allowed: set[str], name: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ConfigurationError(f"unknown key(s) in {name}: {', '.join(unknown)}")


def _sequence(value: object, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConfigurationError(f"{name} must be a sequence")
    return cast(Sequence[object], value)


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigurationError(f"{name} must be an integer >= {minimum}")
    return value


def _real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ConfigurationError(f"{name} must be a finite real number") from exc
    if not math.isfinite(result):
        raise ConfigurationError(f"{name} must be a finite real number")
    return result


def _seed(value: object) -> int | tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        if value < 0:
            raise ConfigurationError("simulation.seed must be non-negative")
        return value
    values = _sequence(value, "simulation.seed")
    result = tuple(_integer(item, "simulation.seed item") for item in values)
    if not result:
        raise ConfigurationError("simulation.seed sequence must not be empty")
    return result


@dataclass(frozen=True, slots=True)
class DomainConfig:
    size: tuple[float, float, float]
    periodic: tuple[bool, bool, bool]
    center: tuple[float, float, float]

    def to_domain(self) -> Domain:
        return Domain(self.size, self.periodic, self.center)


@dataclass(frozen=True, slots=True)
class ParticleConfig:
    name: str
    shape: Literal["sphere", "cylinder"]
    radius: float
    length: float | None
    count: int
    material: MaterialSpec

    def to_spec(self) -> ParticleSpec:
        if self.shape == "sphere":
            return SphereSpec(self.radius, self.material)
        assert self.length is not None
        return CylinderSpec(self.radius, self.length, self.material)

    def to_population(self, count: int | None = None) -> PopulationSpec:
        return PopulationSpec(self.to_spec(), self.count if count is None else count, self.name)


@dataclass(frozen=True, slots=True)
class ContactConfig:
    threshold: float

    def to_model(self) -> ThresholdContactModel:
        return ThresholdContactModel(self.threshold)


@dataclass(frozen=True, slots=True)
class PercolationConfig:
    axis: Literal["x", "y", "z"] = "x"
    mode: Literal["face_to_face", "periodic_wrap"] = "face_to_face"
    wrapped_parent: bool = False


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    trials: int = 1000
    seed: int | tuple[int, ...] | None = None
    confidence: float = 0.95
    neighbor_backend: Literal["cell_list", "bruteforce"] = "cell_list"


@dataclass(frozen=True, slots=True)
class CriticalConfig:
    population: str
    counts: tuple[int, ...]
    target_probability: float = 0.9
    search_trials: int = 1000
    certification_trials: int = 5000
    strategy: Literal["pava", "logistic", "probit"] = "pava"


@dataclass(frozen=True, slots=True)
class OptimizationConfig:
    count_bounds: tuple[tuple[str, int, int], ...]
    target_probability: float = 0.9
    search_trials: int = 250
    certification_trials: int = 5000
    max_candidates: int = 100_000


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    population: str | None
    particle_counts: tuple[int, ...]
    repeats: int = 5
    warmup: int = 1
    seed: int = 42


@dataclass(frozen=True, slots=True)
class ConductivityConfig:
    model: ConstantConductanceModel | TunnelingConductanceModel
    electrode_model: ConstantConductanceModel | TunnelingConductanceModel | None = None
    axes: tuple[str, ...] = ("x", "y", "z")
    applied_voltage: float = 1.0


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    schema_version: int
    domain: DomainConfig
    particles: tuple[ParticleConfig, ...]
    contact: ContactConfig
    percolation: PercolationConfig
    simulation: SimulationConfig
    critical: CriticalConfig | None = None
    optimization: OptimizationConfig | None = None
    benchmark: BenchmarkConfig | None = None
    conductivity: ConductivityConfig | None = None

    def populations(self) -> tuple[PopulationSpec, ...]:
        return tuple(particle.to_population() for particle in self.particles)

    def particle_named(self, name: str) -> ParticleConfig:
        try:
            return next(particle for particle in self.particles if particle.name == name)
        except StopIteration as exc:
            raise ConfigurationError(f"unknown particle population {name!r}") from exc

    def validate_for(self, operation: str) -> None:
        self.domain.to_domain()
        for particle in self.particles:
            particle.to_spec()
        if self.percolation.mode == "periodic_wrap":
            axis = "xyz".index(self.percolation.axis)
            if not self.domain.periodic[axis]:
                raise ConfigurationError("periodic_wrap requires a periodic analysis axis")
        if operation == "conductivity":
            if self.conductivity is None:
                raise ConfigurationError("conductivity section is required")
            if self.percolation.mode != "face_to_face" or self.percolation.wrapped_parent:
                raise ConfigurationError(
                    "conductivity requires face_to_face mode with wrapped_parent disabled"
                )
        elif operation == "critical":
            if self.critical is None:
                raise ConfigurationError("critical section is required")
            self.particle_named(self.critical.population)
        elif operation == "optimize":
            if self.optimization is None:
                raise ConfigurationError("optimization section is required")
            names = {name for name, _, _ in self.optimization.count_bounds}
            expected = {particle.name for particle in self.particles}
            if names != expected:
                raise ConfigurationError("optimization.count_bounds must name every population")
        elif operation == "benchmark":
            if self.benchmark is None:
                raise ConfigurationError("benchmark section is required")
            if self.benchmark.population is not None:
                self.particle_named(self.benchmark.population)
        elif operation not in {"simulate", "validate"}:
            raise ConfigurationError(f"unknown operation {operation!r}")


def _triple_real(value: object, name: str) -> tuple[float, float, float]:
    values = _sequence(value, name)
    if len(values) != 3:
        raise ConfigurationError(f"{name} must contain three values")
    return tuple(_real(item, name) for item in values)  # type: ignore[return-value]


def _parse_domain(value: object) -> DomainConfig:
    data = _mapping(value, "domain")
    _reject_unknown(data, {"size", "periodic", "center"}, "domain")
    if "size" not in data:
        raise ConfigurationError("domain.size is required")
    size = _triple_real(data["size"], "domain.size")
    periodic_raw = data.get("periodic", False)
    if isinstance(periodic_raw, bool):
        periodic = (periodic_raw,) * 3
    else:
        entries = _sequence(periodic_raw, "domain.periodic")
        if len(entries) != 3 or not all(isinstance(item, bool) for item in entries):
            raise ConfigurationError("domain.periodic must contain three booleans")
        periodic = bool(entries[0]), bool(entries[1]), bool(entries[2])
    center = _triple_real(data.get("center", (0.0, 0.0, 0.0)), "domain.center")
    config = DomainConfig(size, periodic, center)
    config.to_domain()
    return config


def _parse_particle(value: object, index: int) -> ParticleConfig:
    name = f"particles[{index}]"
    data = _mapping(value, name)
    _reject_unknown(data, {"name", "shape", "radius", "length", "count", "material"}, name)
    for required in ("name", "shape", "radius", "count"):
        if required not in data:
            raise ConfigurationError(f"{name}.{required} is required")
    population_name = data["name"]
    shape = data["shape"]
    if not isinstance(population_name, str) or not population_name.strip():
        raise ConfigurationError(f"{name}.name must be non-empty")
    if shape not in ("sphere", "cylinder"):
        raise ConfigurationError(f"{name}.shape must be sphere or cylinder")
    material_data = _mapping(data.get("material", {}), f"{name}.material")
    _reject_unknown(material_data, {"name", "cost_per_volume"}, f"{name}.material")
    material = MaterialSpec(
        str(material_data.get("name", population_name)),
        _real(material_data.get("cost_per_volume", 0.0), f"{name}.material.cost_per_volume"),
    )
    length = None
    if shape == "cylinder":
        if "length" not in data:
            raise ConfigurationError(f"{name}.length is required for cylinders")
        length = _real(data["length"], f"{name}.length")
    elif "length" in data:
        raise ConfigurationError(f"{name}.length is not valid for spheres")
    result = ParticleConfig(
        population_name.strip(),
        shape,
        _real(data["radius"], f"{name}.radius"),
        length,
        _integer(data["count"], f"{name}.count"),
        material,
    )
    result.to_spec()
    return result


def _parse_percolation(value: object) -> PercolationConfig:
    data = _mapping(value, "percolation")
    _reject_unknown(data, {"axis", "mode", "wrapped_parent"}, "percolation")
    axis = data.get("axis", "x")
    mode = data.get("mode", "face_to_face")
    wrapped = data.get("wrapped_parent", False)
    if axis not in ("x", "y", "z"):
        raise ConfigurationError("percolation.axis must be x, y, or z")
    if mode not in ("face_to_face", "periodic_wrap"):
        raise ConfigurationError("percolation.mode is invalid")
    if not isinstance(wrapped, bool):
        raise ConfigurationError("percolation.wrapped_parent must be boolean")
    return PercolationConfig(axis, mode, wrapped)


def _parse_simulation(value: object) -> SimulationConfig:
    data = _mapping(value, "simulation")
    _reject_unknown(data, {"trials", "seed", "confidence", "neighbor_backend"}, "simulation")
    backend = data.get("neighbor_backend", "cell_list")
    if backend not in ("cell_list", "bruteforce"):
        raise ConfigurationError("simulation.neighbor_backend is invalid")
    confidence = _real(data.get("confidence", 0.95), "simulation.confidence")
    if not 0.0 < confidence < 1.0:
        raise ConfigurationError("simulation.confidence must lie between zero and one")
    return SimulationConfig(
        _integer(data.get("trials", 1000), "simulation.trials", minimum=1),
        _seed(data.get("seed")),
        confidence,
        backend,
    )


def _parse_critical(value: object) -> CriticalConfig:
    data = _mapping(value, "critical")
    allowed = {
        "population",
        "counts",
        "target_probability",
        "search_trials",
        "certification_trials",
        "strategy",
    }
    _reject_unknown(data, allowed, "critical")
    if "population" not in data or "counts" not in data:
        raise ConfigurationError("critical.population and critical.counts are required")
    population = data["population"]
    if not isinstance(population, str) or not population:
        raise ConfigurationError("critical.population must be non-empty")
    counts = tuple(
        _integer(item, "critical.counts item")
        for item in _sequence(data["counts"], "critical.counts")
    )
    if not counts or any(right <= left for left, right in zip(counts, counts[1:], strict=False)):
        raise ConfigurationError("critical.counts must be non-empty and strictly increasing")
    strategy = data.get("strategy", "pava")
    if strategy not in ("pava", "logistic", "probit"):
        raise ConfigurationError("critical.strategy is invalid")
    target = _real(data.get("target_probability", 0.9), "critical.target_probability")
    if not 0.0 < target < 1.0:
        raise ConfigurationError("critical.target_probability must lie between zero and one")
    return CriticalConfig(
        population,
        counts,
        target,
        _integer(data.get("search_trials", 1000), "critical.search_trials", minimum=1),
        _integer(
            data.get("certification_trials", 5000),
            "critical.certification_trials",
            minimum=1,
        ),
        strategy,
    )


def _parse_optimization(value: object) -> OptimizationConfig:
    data = _mapping(value, "optimization")
    allowed = {
        "count_bounds",
        "target_probability",
        "search_trials",
        "certification_trials",
        "max_candidates",
    }
    _reject_unknown(data, allowed, "optimization")
    bounds_data = _mapping(data.get("count_bounds"), "optimization.count_bounds")
    bounds: list[tuple[str, int, int]] = []
    for name, raw in bounds_data.items():
        pair = _sequence(raw, f"optimization.count_bounds.{name}")
        if len(pair) != 2:
            raise ConfigurationError(f"optimization.count_bounds.{name} must contain two integers")
        low = _integer(pair[0], f"optimization.count_bounds.{name}[0]")
        high = _integer(pair[1], f"optimization.count_bounds.{name}[1]")
        if high < low:
            raise ConfigurationError(f"optimization.count_bounds.{name} is reversed")
        bounds.append((name, low, high))
    if not bounds:
        raise ConfigurationError("optimization.count_bounds must not be empty")
    target = _real(data.get("target_probability", 0.9), "optimization.target_probability")
    if not 0.0 < target < 1.0:
        raise ConfigurationError("optimization.target_probability must lie between zero and one")
    return OptimizationConfig(
        tuple(bounds),
        target,
        _integer(data.get("search_trials", 250), "optimization.search_trials", minimum=1),
        _integer(
            data.get("certification_trials", 5000),
            "optimization.certification_trials",
            minimum=1,
        ),
        _integer(data.get("max_candidates", 100_000), "optimization.max_candidates", minimum=1),
    )


def _parse_benchmark(value: object) -> BenchmarkConfig:
    data = _mapping(value, "benchmark")
    _reject_unknown(
        data,
        {"population", "particle_counts", "repeats", "warmup", "seed"},
        "benchmark",
    )
    raw_population = data.get("population")
    if raw_population is not None and (not isinstance(raw_population, str) or not raw_population):
        raise ConfigurationError("benchmark.population must be non-empty when set")
    counts = tuple(
        _integer(item, "benchmark.particle_counts item", minimum=2)
        for item in _sequence(
            data.get("particle_counts", (10, 50, 100, 500, 1000)),
            "benchmark.particle_counts",
        )
    )
    if len(set(counts)) != len(counts):
        raise ConfigurationError("benchmark.particle_counts must be unique")
    return BenchmarkConfig(
        raw_population,
        counts,
        _integer(data.get("repeats", 5), "benchmark.repeats", minimum=1),
        _integer(data.get("warmup", 1), "benchmark.warmup"),
        _integer(data.get("seed", 42), "benchmark.seed"),
    )


def _parse_conductance_model(
    value: object,
    name: str,
) -> ConstantConductanceModel | TunnelingConductanceModel:
    data = _mapping(value, name)
    kind = data.get("type", "tunneling")
    if kind not in ("constant", "tunneling"):
        raise ConfigurationError(f"{name}.type must be constant or tunneling")
    allowed = {"type", "contact_conductance", "cutoff"}
    if kind == "tunneling":
        allowed.add("decay_length")
    _reject_unknown(data, allowed, name)
    if "cutoff" not in data:
        raise ConfigurationError(f"{name}.cutoff is required")
    g0 = _real(data.get("contact_conductance", 1.0), f"{name}.contact_conductance")
    cutoff = _real(data["cutoff"], f"{name}.cutoff")
    if kind == "constant":
        return ConstantConductanceModel(g0, cutoff)
    if "decay_length" not in data:
        raise ConfigurationError(f"{name}.decay_length is required")
    return TunnelingConductanceModel(
        g0,
        _real(data["decay_length"], f"{name}.decay_length"),
        cutoff,
    )


def _parse_conductivity(value: object) -> ConductivityConfig:
    data = _mapping(value, "conductivity")
    _reject_unknown(data, {"model", "electrode_model", "axes", "applied_voltage"}, "conductivity")
    axes = tuple(_sequence(data.get("axes", ("x", "y", "z")), "conductivity.axes"))
    if not axes or any(axis not in ("x", "y", "z") for axis in axes):
        raise ConfigurationError("conductivity.axes must contain x, y, or z")
    if len(set(axes)) != len(axes):
        raise ConfigurationError("conductivity.axes must be unique")
    voltage = _real(data.get("applied_voltage", 1.0), "conductivity.applied_voltage")
    if voltage <= 0:
        raise ConfigurationError("conductivity.applied_voltage must be positive")
    return ConductivityConfig(
        _parse_conductance_model(data.get("model"), "conductivity.model"),
        None
        if "electrode_model" not in data
        else _parse_conductance_model(
            data["electrode_model"],
            "conductivity.electrode_model",
        ),
        cast(tuple[str, ...], axes),
        voltage,
    )


def loads_config(text: str, *, operation: str | None = None) -> ProjectConfig:
    try:
        raw = yaml.load(text, Loader=_UniqueKeyLoader)
    except (yaml.YAMLError, OverflowError, ValueError) as exc:
        detail = getattr(exc, "problem", None)
        suffix = f": {detail}" if detail else ""
        raise ConfigurationError(f"configuration is not valid YAML{suffix}") from exc
    data = _mapping(raw, "root")
    allowed = {
        "schema_version",
        "domain",
        "particles",
        "contact",
        "percolation",
        "simulation",
        "critical",
        "optimization",
        "benchmark",
        "conductivity",
    }
    _reject_unknown(data, allowed, "root")
    schema_version = data.get("schema_version")
    if type(schema_version) is not int or schema_version != 1:
        raise ConfigurationError("schema_version must be 1")
    if "domain" not in data or "particles" not in data:
        raise ConfigurationError("domain and particles are required")
    particles = tuple(
        _parse_particle(item, index)
        for index, item in enumerate(_sequence(data["particles"], "particles"))
    )
    if not particles:
        raise ConfigurationError("particles must not be empty")
    names = [particle.name for particle in particles]
    if len(set(names)) != len(names):
        raise ConfigurationError("particle population names must be unique")
    contact_data = _mapping(data.get("contact", {}), "contact")
    _reject_unknown(contact_data, {"type", "threshold"}, "contact")
    if contact_data.get("type", "threshold") != "threshold":
        raise ConfigurationError("contact.type must be threshold")
    config = ProjectConfig(
        1,
        _parse_domain(data["domain"]),
        particles,
        ContactConfig(_real(contact_data.get("threshold", 0.0), "contact.threshold")),
        _parse_percolation(data.get("percolation", {})),
        _parse_simulation(data.get("simulation", {})),
        None if "critical" not in data else _parse_critical(data["critical"]),
        None if "optimization" not in data else _parse_optimization(data["optimization"]),
        None if "benchmark" not in data else _parse_benchmark(data["benchmark"]),
        None if "conductivity" not in data else _parse_conductivity(data["conductivity"]),
    )
    config.contact.to_model()
    if operation is not None:
        config.validate_for(operation)
    return config


def load_config(path: str | Path, *, operation: str | None = None) -> ProjectConfig:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigurationError(f"could not read configuration: {exc}") from exc
    return loads_config(text, operation=operation)
