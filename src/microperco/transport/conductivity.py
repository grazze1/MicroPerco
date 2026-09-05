# SPDX-License-Identifier: Apache-2.0
"""Build junction networks from geometry and measure directional conductivity."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from ..contact import ContactSearchResult, ThresholdContactModel, find_contacts
from ..domain import Domain, normalize_axis
from ..exceptions import ConfigurationError, SimulationError
from ..geometry import face_gaps
from ..numerics import positive_finite_product
from ..particles import Cylinder, Particle, Sphere
from ..percolation import _node_mapping
from .models import ConstantConductanceModel, TunnelingConductanceModel
from .network import NetworkSolution, ResistorEdge, ResistorNetwork, solve_resistor_network

JunctionModel = ConstantConductanceModel | TunnelingConductanceModel


@dataclass(frozen=True, slots=True)
class Junction:
    """Geometry provenance for the resistor at the same index in network.edges."""

    kind: str
    source_particle: int
    target_particle: int | None
    lattice_shift: tuple[int, int, int]
    gap: float


@dataclass(frozen=True, slots=True)
class ConductivityNetwork:
    axis: str
    domain: Domain
    particle_nodes: tuple[int, ...]
    network: ResistorNetwork
    junctions: tuple[Junction, ...]
    contact_search: ContactSearchResult
    conductance_model: JunctionModel
    electrode_model: JunctionModel


@dataclass(frozen=True, slots=True)
class ConductivityResult:
    axis: str
    effective_conductivity: float
    electrode_separation: float
    cross_section_area: float
    geometry: ConductivityNetwork
    solution: NetworkSolution

    @property
    def effective_conductance(self) -> float:
        return self.solution.effective_conductance

    @property
    def percolates(self) -> bool:
        return self.solution.connected


@dataclass(frozen=True, slots=True)
class DirectionalConductivityResult:
    sigma_x: float
    sigma_y: float
    sigma_z: float
    results: tuple[ConductivityResult, ...]


def _validate_model(model: JunctionModel) -> None:
    if not isinstance(model, (ConstantConductanceModel, TunnelingConductanceModel)):
        raise ConfigurationError("model must be a constant or tunneling conductance model")


def build_conductivity_network(
    particles: Sequence[Particle],
    domain: Domain,
    conductance_model: JunctionModel | None = None,
    *,
    axis: int | str = "x",
    electrode_model: JunctionModel | None = None,
    neighbor_backend: str = "cell_list",
) -> ConductivityNetwork:
    """Open the measurement axis and retain transverse periodic boundaries.

    Particles and equal-parent fragments are equipotential nodes. Distinct
    particle-image junctions are parallel resistors. Each logical node has at
    most one junction to each electrode, using its minimum fragment face gap.
    Transverse self-image loops have zero voltage drop and are omitted.
    """

    if not isinstance(domain, Domain):
        raise ConfigurationError("domain must be a Domain")
    items = tuple(particles)
    if not all(isinstance(item, (Sphere, Cylinder)) for item in items):
        raise ConfigurationError("particles must contain Sphere or Cylinder instances")
    model = ConstantConductanceModel() if conductance_model is None else conductance_model
    electrodes = model if electrode_model is None else electrode_model
    _validate_model(model)
    _validate_model(electrodes)
    index = normalize_axis(axis)
    periodic = tuple(enabled and i != index for i, enabled in enumerate(domain.periodic))
    opened = domain.with_periodic(periodic)
    search = find_contacts(
        items,
        opened,
        ThresholdContactModel(model.cutoff, model.numerical_policy),
        method=neighbor_backend,
    )
    nodes, representatives = _node_mapping(items)
    source, sink = len(representatives), len(representatives) + 1
    edges: list[ResistorEdge] = []
    junctions: list[Junction] = []
    for edge in search.edges:
        # The search includes a tolerance halo; the electrical cutoff is exact.
        value = model.conductance(edge.distance)
        if value > 0.0 and nodes[edge.i] != nodes[edge.j]:
            edges.append(ResistorEdge(nodes[edge.i], nodes[edge.j], value))
            junctions.append(
                Junction("particle", edge.i, edge.j, edge.lattice_shift, edge.distance)
            )
    closest: dict[tuple[int, int], tuple[float, int]] = {}
    for i, particle in enumerate(items):
        gaps = face_gaps(particle, opened, index, policy=electrodes.numerical_policy)
        for terminal, gap in zip((source, sink), gaps, strict=True):
            key = nodes[i], terminal
            candidate = (gap, i)
            if key not in closest or candidate < closest[key]:
                closest[key] = candidate
    for (node, terminal), (gap, particle_index) in sorted(closest.items()):
        value = electrodes.conductance(gap)
        if value > 0.0:
            edges.append(ResistorEdge(node, terminal, value))
            kind = "lower_electrode" if terminal == source else "upper_electrode"
            junctions.append(Junction(kind, particle_index, None, (0, 0, 0), gap))
    return ConductivityNetwork(
        "xyz"[index],
        opened,
        tuple(nodes),
        ResistorNetwork(sink + 1, tuple(edges), source, sink),
        tuple(junctions),
        search,
        model,
        electrodes,
    )


def analyze_conductivity(
    particles: Sequence[Particle],
    domain: Domain,
    conductance_model: JunctionModel | None = None,
    *,
    axis: int | str = "x",
    electrode_model: JunctionModel | None = None,
    neighbor_backend: str = "cell_list",
    applied_voltage: float = 1.0,
) -> ConductivityResult:
    """Compute sigma = G*L/A with finite electrode junction resistances.

    Conductances in siemens and geometry in metres yield conductivity in S/m.
    This is an apparent two-terminal conductivity, not a full periodic tensor.
    """

    geometry = build_conductivity_network(
        particles,
        domain,
        conductance_model,
        axis=axis,
        electrode_model=electrode_model,
        neighbor_backend=neighbor_backend,
    )
    solution = solve_resistor_network(geometry.network, applied_voltage=applied_voltage)
    index = normalize_axis(axis)
    length = domain.size[index]
    try:
        area = positive_finite_product(*(size for i, size in enumerate(domain.size) if i != index))
        sigma = (
            0.0
            if not solution.connected
            else positive_finite_product(solution.effective_conductance, length, 1.0 / area)
        )
    except (ArithmeticError, OverflowError) as exc:
        raise SimulationError("conductivity geometry factor is not representable") from exc
    if not math.isfinite(sigma):
        raise SimulationError("effective conductivity is not representable")
    return ConductivityResult(geometry.axis, sigma, length, area, geometry, solution)


def analyze_directional_conductivity(
    particles: Sequence[Particle],
    domain: Domain,
    conductance_model: JunctionModel | None = None,
    *,
    electrode_model: JunctionModel | None = None,
    neighbor_backend: str = "cell_list",
    applied_voltage: float = 1.0,
) -> DirectionalConductivityResult:
    """Measure x/y/z on the same realization, opening each axis in turn."""

    items = tuple(particles)
    results = tuple(
        analyze_conductivity(
            items,
            domain,
            conductance_model,
            axis=axis,
            electrode_model=electrode_model,
            neighbor_backend=neighbor_backend,
            applied_voltage=applied_voltage,
        )
        for axis in "xyz"
    )
    return DirectionalConductivityResult(
        results[0].effective_conductivity,
        results[1].effective_conductivity,
        results[2].effective_conductivity,
        results,
    )
