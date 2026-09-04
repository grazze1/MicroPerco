# SPDX-License-Identifier: Apache-2.0
"""Face-to-face and periodic-wrapping percolation analysis."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np

from .contact import ContactEdge, ThresholdContactModel, find_contacts
from .domain import Domain, normalize_axis
from .geometry import distance, face_gaps
from .geometry.periodic import _candidate_lattice_shifts_local, _canonical_particle
from .graph import UnionFind, WeightedUnionFind
from .particles import Cylinder, Particle, Sphere

PercolationMode = Literal["face_to_face", "periodic_wrap"]
AxisName = Literal["x", "y", "z"]
TopologyEdgeKind = Literal["inter_group", "periodic_self_image"]


@dataclass(frozen=True, slots=True, order=True)
class PeriodicTopologyEdge:
    """A lattice-labelled edge used only for periodic winding analysis."""

    node_i: int
    node_j: int
    lattice_shift: tuple[int, int, int]
    source_particle: int
    target_particle: int
    distance: float
    kind: TopologyEdgeKind

    def __post_init__(self) -> None:
        if self.node_i < 0 or self.node_j < self.node_i:
            raise ValueError("topology edge nodes must satisfy 0 <= node_i <= node_j")
        if self.source_particle < 0 or self.target_particle < 0:
            raise ValueError("topology edge particle indices must be non-negative")
        shift = tuple(self.lattice_shift)
        if len(shift) != 3 or not all(isinstance(value, int) for value in shift):
            raise ValueError("topology lattice_shift must contain three integers")
        if self.kind == "inter_group":
            if self.node_i == self.node_j:
                raise ValueError("inter-group topology edges require distinct nodes")
        elif self.kind == "periodic_self_image":
            if self.node_i != self.node_j or shift == (0, 0, 0):
                raise ValueError("self-image topology edges require a non-zero loop shift")
        else:
            raise ValueError("unknown periodic topology edge kind")
        if not np.isfinite(self.distance) or self.distance < 0.0:
            raise ValueError("topology edge distance must be finite and non-negative")
        object.__setattr__(self, "lattice_shift", (shift[0], shift[1], shift[2]))
        object.__setattr__(self, "distance", float(self.distance))


@dataclass(frozen=True, slots=True)
class PercolationResult:
    """Immutable, auditable result of one connectivity analysis."""

    percolates: bool
    mode: PercolationMode
    axis: AxisName
    particle_count: int
    component_count: int
    contact_edges: tuple[ContactEdge, ...]
    lower_electrode_particles: tuple[int, ...]
    upper_electrode_particles: tuple[int, ...]
    spanning_path: tuple[int, ...]
    winding: tuple[int, int, int] | None
    candidate_pairs: int
    distance_evaluations: int
    topology_edges: tuple[PeriodicTopologyEdge, ...] = ()
    topology_candidate_pairs: int = 0
    topology_distance_evaluations: int = 0

    @property
    def conductive(self) -> bool:
        return self.percolates

    @property
    def edge_count(self) -> int:
        return len(self.contact_edges)

    @property
    def path(self) -> tuple[int, ...]:
        return self.spanning_path

    @property
    def contacts(self) -> tuple[ContactEdge, ...]:
        return self.contact_edges


def _normalize_mode(mode: str) -> PercolationMode:
    value = mode.lower().replace("-", "_")
    if value not in {"face_to_face", "periodic_wrap"}:
        raise ValueError("mode must be 'face_to_face' or 'periodic_wrap'")
    return cast(PercolationMode, value)


def _node_mapping(particles: tuple[Particle, ...]) -> tuple[list[int], list[int]]:
    keys: dict[tuple[str, object], int] = {}
    particle_nodes: list[int] = []
    representatives: list[int] = []
    for index, particle in enumerate(particles):
        key: tuple[str, object]
        if particle.parent_id is not None:
            key = ("parent", particle.parent_id)
        else:
            key = ("particle", index)
        if key not in keys:
            keys[key] = len(keys)
            representatives.append(index)
        particle_nodes.append(keys[key])
    return particle_nodes, representatives


def _axis_name(axis: int) -> AxisName:
    return ("x", "y", "z")[axis]


def _add_lattice(
    first: tuple[int, int, int], second: tuple[int, int, int]
) -> tuple[int, int, int]:
    return first[0] + second[0], first[1] + second[1], first[2] + second[2]


def _subtract_lattice(
    first: tuple[int, int, int], second: tuple[int, int, int]
) -> tuple[int, int, int]:
    return first[0] - second[0], first[1] - second[1], first[2] - second[2]


def _orient_winding(winding: tuple[int, int, int] | None, axis: int) -> tuple[int, int, int] | None:
    if winding is None or winding[axis] >= 0:
        return winding
    return -winding[0], -winding[1], -winding[2]


def _canonical_loop_shift(shift: tuple[int, int, int]) -> tuple[int, int, int]:
    for value in shift:
        if value > 0:
            return shift
        if value < 0:
            return -shift[0], -shift[1], -shift[2]
    raise ValueError("loop shifts must be non-zero")


def _offset(particle: Particle) -> tuple[int, int, int]:
    return (0, 0, 0) if particle.image_offset is None else particle.image_offset


def _logical_shift(
    shift: tuple[int, int, int], source: Particle, target: Particle
) -> tuple[int, int, int]:
    source_offset = _offset(source)
    target_offset = _offset(target)
    return (
        shift[0] + source_offset[0] - target_offset[0],
        shift[1] + source_offset[1] - target_offset[1],
        shift[2] + source_offset[2] - target_offset[2],
    )


def _intrinsic_topology_edges(
    particles: tuple[Particle, ...],
    particle_nodes: list[int],
    domain: Domain,
    contact_model: ThresholdContactModel,
) -> tuple[tuple[PeriodicTopologyEdge, ...], int, int]:
    """Find non-zero image contacts inside each logical particle group."""

    groups: dict[int, list[int]] = {}
    for particle_index, node in enumerate(particle_nodes):
        groups.setdefault(node, []).append(particle_index)
    accepted: dict[tuple[int, tuple[int, int, int]], PeriodicTopologyEdge] = {}
    candidate_count = 0
    evaluation_count = 0
    canonical_with_shifts = tuple(_canonical_particle(particle, domain) for particle in particles)
    for node, indices in groups.items():
        for offset, first_index in enumerate(indices):
            for second_index in indices[offset:]:
                first = particles[first_index]
                second = particles[second_index]
                if (
                    first_index != second_index
                    and (first.image_offset is None or second.image_offset is None)
                ):
                    continue
                canonical_first, first_base_shift = canonical_with_shifts[first_index]
                canonical_second, second_base_shift = canonical_with_shifts[second_index]
                shifts = _candidate_lattice_shifts_local(
                    canonical_first,
                    canonical_second,
                    domain,
                    0.0,
                    policy=contact_model.numerical_policy,
                )
                for shift in shifts:
                    candidate_count += 1
                    shifted = canonical_second.translated(domain.lattice_vector(shift))
                    gap = distance(
                        canonical_first,
                        shifted,
                        policy=contact_model.numerical_policy,
                    )
                    evaluation_count += 1
                    if not contact_model.numerical_policy.less_than_or_close(
                        gap,
                        0.0,
                        scale=contact_model.pair_scale(canonical_first, shifted),
                    ):
                        continue
                    reported_shift = (
                        shift[0] + second_base_shift[0] - first_base_shift[0],
                        shift[1] + second_base_shift[1] - first_base_shift[1],
                        shift[2] + second_base_shift[2] - first_base_shift[2],
                    )
                    logical_shift = _logical_shift(reported_shift, first, second)
                    if logical_shift == (0, 0, 0):
                        continue
                    canonical_shift = _canonical_loop_shift(logical_shift)
                    edge = PeriodicTopologyEdge(
                        node,
                        node,
                        canonical_shift,
                        first_index,
                        second_index,
                        gap,
                        "periodic_self_image",
                    )
                    key = node, canonical_shift
                    previous = accepted.get(key)
                    if previous is None or (
                        edge.distance,
                        edge.source_particle,
                        edge.target_particle,
                    ) < (
                        previous.distance,
                        previous.source_particle,
                        previous.target_particle,
                    ):
                        accepted[key] = edge
    return tuple(sorted(accepted.values())), candidate_count, evaluation_count


def _validate_image_offsets(particles: tuple[Particle, ...], domain: Domain) -> None:
    parent_offsets: dict[object, list[bool]] = {}
    for particle in particles:
        if particle.image_offset is not None and any(
            value != 0 and not domain.periodic[axis]
            for axis, value in enumerate(particle.image_offset)
        ):
            raise ValueError("image_offset must be zero along non-periodic domain axes")
        if particle.parent_id is not None:
            parent_offsets.setdefault(particle.parent_id, []).append(
                particle.image_offset is not None
            )
    if any(any(flags) and not all(flags) for flags in parent_offsets.values()):
        raise ValueError("each parent group must provide image_offset for all fragments or none")


def _inter_group_topology_edges(
    contact_edges: tuple[ContactEdge, ...],
    particles: tuple[Particle, ...],
    particle_nodes: list[int],
) -> tuple[PeriodicTopologyEdge, ...]:
    result: list[PeriodicTopologyEdge] = []
    for edge in contact_edges:
        first = particle_nodes[edge.i]
        second = particle_nodes[edge.j]
        shift = _logical_shift(edge.lattice_shift, particles[edge.i], particles[edge.j])
        if first == second:
            continue
        if first < second:
            result.append(
                PeriodicTopologyEdge(
                    first,
                    second,
                    shift,
                    edge.i,
                    edge.j,
                    edge.distance,
                    "inter_group",
                )
            )
        else:
            result.append(
                PeriodicTopologyEdge(
                    second,
                    first,
                    (
                        -shift[0],
                        -shift[1],
                        -shift[2],
                    ),
                    edge.j,
                    edge.i,
                    edge.distance,
                    "inter_group",
                )
            )
    return tuple(sorted(result))


def _open_axis_domain(domain: Domain, axis: int) -> Domain:
    periodic = list(domain.periodic)
    periodic[axis] = False
    return domain.with_periodic(periodic)


def _face_to_face(
    particles: tuple[Particle, ...],
    domain: Domain,
    contact_model: ThresholdContactModel,
    axis: int,
    search: str,
    wrapped_parent: bool,
    solver: str,
) -> PercolationResult:
    search_result = find_contacts(
        particles, _open_axis_domain(domain, axis), contact_model, method=search
    )
    particle_nodes, representatives = _node_mapping(particles)
    node_count = len(representatives)
    lower_node = node_count
    upper_node = node_count + 1
    union_find = UnionFind(node_count + 2)
    particle_union_find = UnionFind(node_count)
    adjacency: list[set[int]] = [set() for _ in range(node_count + 2)]

    def connect(first: int, second: int) -> None:
        union_find.union(first, second)
        adjacency[first].add(second)
        adjacency[second].add(first)

    for edge in search_result.edges:
        first = particle_nodes[edge.i]
        second = particle_nodes[edge.j]
        if first != second:
            connect(first, second)
            particle_union_find.union(first, second)

    lower_particles: list[int] = []
    upper_particles: list[int] = []
    for index, particle in enumerate(particles):
        lower_gap, upper_gap = face_gaps(
            particle,
            domain,
            axis,
            wrapped_parent=wrapped_parent,
            policy=contact_model.numerical_policy,
        )
        node = particle_nodes[index]
        face_scale = max(
            float(particle.aabb_extent[axis]),
            contact_model.threshold,
            1.0,
        )
        if contact_model.accepts(lower_gap, scale=face_scale):
            lower_particles.append(index)
            connect(lower_node, node)
        if contact_model.accepts(upper_gap, scale=face_scale):
            upper_particles.append(index)
            connect(upper_node, node)

    previous: dict[int, int | None] = {lower_node: None}
    queue: deque[int] = deque((lower_node,))
    while queue and upper_node not in previous:
        current = queue.popleft()
        for neighbour in sorted(adjacency[current]):
            if neighbour not in previous:
                previous[neighbour] = current
                queue.append(neighbour)
    union_decision = node_count > 0 and union_find.connected(lower_node, upper_node)
    bfs_decision = node_count > 0 and upper_node in previous
    normalized_solver = solver.lower().replace("-", "_")
    if normalized_solver == "union_find":
        percolates = union_decision
    elif normalized_solver == "bfs":
        percolates = bfs_decision
    else:
        raise ValueError("solver must be 'union_find' or 'bfs'")
    if union_decision != bfs_decision:
        raise RuntimeError("connectivity solvers disagreed")
    spanning_path: tuple[int, ...] = ()
    if percolates:
        nodes: list[int] = []
        cursor: int | None = upper_node
        while cursor is not None:
            if cursor < node_count:
                nodes.append(representatives[cursor])
            cursor = previous[cursor]
        spanning_path = tuple(reversed(nodes))
    roots = {particle_union_find.find(node) for node in range(node_count)}
    return PercolationResult(
        percolates,
        "face_to_face",
        _axis_name(axis),
        len(particles),
        len(roots),
        search_result.edges,
        tuple(lower_particles),
        tuple(upper_particles),
        spanning_path,
        None,
        search_result.candidate_pairs,
        search_result.distance_evaluations,
    )


def _periodic_wrap(
    particles: tuple[Particle, ...],
    domain: Domain,
    contact_model: ThresholdContactModel,
    axis: int,
    search: str,
    wrapped_parent: bool,
    solver: str,
) -> PercolationResult:
    if not domain.periodic[axis]:
        raise ValueError("periodic_wrap requires the analysis axis to be periodic")
    _validate_image_offsets(particles, domain)
    search_result = find_contacts(particles, domain, contact_model, method=search)
    particle_nodes, representatives = _node_mapping(particles)
    weighted = WeightedUnionFind(len(representatives))
    intrinsic_edges, topology_candidates, topology_evaluations = _intrinsic_topology_edges(
        particles,
        particle_nodes,
        domain,
        contact_model,
    )
    topology_edges = (
        _inter_group_topology_edges(search_result.edges, particles, particle_nodes)
        + intrinsic_edges
    )
    topology_edges = tuple(sorted(topology_edges))
    union_winding: tuple[int, int, int] | None = None
    adjacency: list[list[tuple[int, tuple[int, int, int]]]] = [
        [] for _ in representatives
    ]
    for edge in topology_edges:
        first = edge.node_i
        second = edge.node_j
        delta = edge.lattice_shift
        adjacency[first].append((second, delta))
        adjacency[second].append((first, (-delta[0], -delta[1], -delta[2])))
        residual: tuple[int, int, int] | None
        if first == second:
            residual = delta
        else:
            residual = weighted.union(first, second, delta)
        if residual is not None and int(residual[axis]) != 0 and union_winding is None:
            union_winding = residual

    potentials: dict[int, tuple[int, int, int]] = {}
    bfs_winding: tuple[int, int, int] | None = None
    for start in range(len(representatives)):
        if start in potentials:
            continue
        potentials[start] = (0, 0, 0)
        queue: deque[int] = deque((start,))
        while queue:
            current = queue.popleft()
            for neighbour, delta in adjacency[current]:
                proposed = _add_lattice(potentials[current], delta)
                if neighbour not in potentials:
                    potentials[neighbour] = proposed
                    queue.append(neighbour)
                else:
                    bfs_residual = _subtract_lattice(proposed, potentials[neighbour])
                    if int(bfs_residual[axis]) != 0 and bfs_winding is None:
                        bfs_winding = bfs_residual

    union_winding = _orient_winding(union_winding, axis)
    bfs_winding = _orient_winding(bfs_winding, axis)
    normalized_solver = solver.lower().replace("-", "_")
    if normalized_solver == "union_find":
        winding = union_winding
    elif normalized_solver == "bfs":
        winding = bfs_winding
    else:
        raise ValueError("solver must be 'union_find' or 'bfs'")
    if (union_winding is None) != (bfs_winding is None):
        raise RuntimeError("winding solvers disagreed")
    return PercolationResult(
        winding is not None,
        "periodic_wrap",
        _axis_name(axis),
        len(particles),
        weighted.component_count,
        search_result.edges,
        (),
        (),
        (),
        winding,
        search_result.candidate_pairs,
        search_result.distance_evaluations,
        topology_edges,
        topology_candidates,
        topology_evaluations,
    )


def analyze_percolation(
    particles: Sequence[Particle],
    domain: Domain,
    contact_model: ThresholdContactModel | None = None,
    *,
    axis: int | str = 0,
    mode: str = "face_to_face",
    search: str = "cell_list",
    wrapped_parent: bool = False,
    solver: str = "union_find",
    backend: str | None = None,
) -> PercolationResult:
    """Analyze connectivity under an explicit boundary interpretation."""

    if not isinstance(domain, Domain):
        raise TypeError("domain must be a Domain")
    items = tuple(particles)
    if not all(isinstance(item, (Sphere, Cylinder)) for item in items):
        raise TypeError("particles must contain Sphere or Cylinder instances")
    model = ThresholdContactModel() if contact_model is None else contact_model
    if not isinstance(model, ThresholdContactModel):
        raise TypeError("contact_model must be a ThresholdContactModel")
    if backend is not None:
        if search != "cell_list" and search.lower().replace("-", "_") != backend.lower().replace(
            "-", "_"
        ):
            raise ValueError("search and backend select different neighbor searches")
        search = backend
    index = normalize_axis(axis)
    mode_value = _normalize_mode(mode)
    if mode_value == "face_to_face":
        return _face_to_face(items, domain, model, index, search, wrapped_parent, solver)
    return _periodic_wrap(items, domain, model, index, search, wrapped_parent, solver)


__all__ = [
    "AxisName",
    "PercolationMode",
    "PercolationResult",
    "PeriodicTopologyEdge",
    "TopologyEdgeKind",
    "analyze_percolation",
]
