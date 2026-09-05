# SPDX-License-Identifier: Apache-2.0
"""Sparse two-terminal resistor networks with auditable Kirchhoff solutions."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import MatrixRankWarning, spsolve

from ..exceptions import ConfigurationError, SimulationError
from ..graph import UnionFind
from .models import finite_real


@dataclass(frozen=True, slots=True)
class ResistorEdge:
    """An undirected resistor; positive current is from i to j."""

    i: int
    j: int
    conductance: float

    def __post_init__(self) -> None:
        if any(type(v) is not int or v < 0 for v in (self.i, self.j)) or self.i == self.j:
            raise ConfigurationError("resistor endpoints must be distinct non-negative integers")
        object.__setattr__(self, "conductance", finite_real(self.conductance, "conductance"))


@dataclass(frozen=True, slots=True)
class ResistorNetwork:
    """A multigraph with finite resistors and two distinct voltage terminals."""

    node_count: int
    edges: tuple[ResistorEdge, ...]
    source: int
    sink: int

    def __post_init__(self) -> None:
        if type(self.node_count) is not int or self.node_count < 2:
            raise ConfigurationError("node_count must be an integer >= 2")
        if (
            any(
                type(v) is not int or not 0 <= v < self.node_count for v in (self.source, self.sink)
            )
            or self.source == self.sink
        ):
            raise ConfigurationError("source and sink must be distinct valid nodes")
        edges = tuple(self.edges)
        if not all(isinstance(edge, ResistorEdge) for edge in edges):
            raise ConfigurationError("edges must contain ResistorEdge values")
        if any(max(edge.i, edge.j) >= self.node_count for edge in edges):
            raise ConfigurationError("resistor endpoint exceeds node_count")
        object.__setattr__(self, "edges", edges)


@dataclass(frozen=True, slots=True)
class NetworkSolution:
    connected: bool
    applied_voltage: float
    effective_conductance: float
    total_current: float
    dissipated_power: float
    node_voltages: tuple[float | None, ...]
    edge_currents: tuple[float, ...]
    floating_nodes: tuple[int, ...]
    relative_kirchhoff_residual: float
    relative_current_imbalance: float
    relative_power_error: float


def solve_resistor_network(
    network: ResistorNetwork, *, applied_voltage: float = 1.0
) -> NetworkSolution:
    """Solve at unit voltage, then scale; floating components have null voltages.

    Source-only/sink-only components are exactly equipotential. Singular or
    insufficiently resolved spanning systems fail explicitly. No leakage or
    diagonal regularization is added to disconnected components.
    """

    if not isinstance(network, ResistorNetwork):
        raise ConfigurationError("network must be a ResistorNetwork")
    voltage = finite_real(applied_voltage, "applied_voltage")
    groups = UnionFind(network.node_count)
    for edge in network.edges:
        groups.union(edge.i, edge.j)
    source_root = groups.find(network.source)
    sink_root = groups.find(network.sink)
    connected = source_root == sink_root
    roots = [groups.find(i) for i in range(network.node_count)]
    floating = tuple(i for i, root in enumerate(roots) if root not in {source_root, sink_root})
    potentials = np.zeros(network.node_count, dtype=np.float64)
    potentials[[i for i, root in enumerate(roots) if root == source_root]] = 1.0
    conductance = residual = imbalance = power_error = 0.0
    unit_currents = np.zeros(len(network.edges), dtype=np.float64)
    if connected:
        active_edges = [
            (index, edge)
            for index, edge in enumerate(network.edges)
            if roots[edge.i] == source_root
        ]
        scale = max(edge.conductance for _, edge in active_edges)
        unknown = [
            i
            for i, root in enumerate(roots)
            if root == source_root and i not in {network.source, network.sink}
        ]
        positions = {node: i for i, node in enumerate(unknown)}
        rows: list[int] = []
        cols: list[int] = []
        values: list[float] = []
        rhs = np.zeros(len(unknown), dtype=np.float64)
        for _, edge in active_edges:
            weight = edge.conductance / scale
            if weight == 0.0:
                raise SimulationError("network conductance dynamic range exceeds float64")
            for i, j in ((edge.i, edge.j), (edge.j, edge.i)):
                if i not in positions:
                    continue
                row = positions[i]
                rows.append(row)
                cols.append(row)
                values.append(weight)
                if j in positions:
                    rows.append(row)
                    cols.append(positions[j])
                    values.append(-weight)
                elif j == network.source:
                    rhs[row] += weight
        potentials[network.sink] = 0.0
        if unknown:
            matrix = coo_matrix((values, (rows, cols)), shape=(len(unknown), len(unknown))).tocsc()
            with warnings.catch_warnings():
                warnings.simplefilter("error", MatrixRankWarning)
                try:
                    solved = np.asarray(spsolve(matrix, rhs), dtype=np.float64)
                except (MatrixRankWarning, RuntimeError, ValueError) as exc:
                    raise SimulationError("resistor network solve failed") from exc
            if (
                not np.all(np.isfinite(solved))
                or np.any(solved < -1e-10)
                or np.any(solved > 1.0 + 1e-10)
            ):
                raise SimulationError("network voltages violate the maximum principle")
            potentials[unknown] = np.clip(solved, 0.0, 1.0)
        contributions: list[list[float]] = [[] for _ in roots]
        energies: list[float] = []
        for index, edge in active_edges:
            drop = float(potentials[edge.i] - potentials[edge.j])
            current = (edge.conductance / scale) * drop
            contributions[edge.i].append(current)
            contributions[edge.j].append(-current)
            energies.append(current * drop)
            unit_currents[index] = current
        net = [math.fsum(items) for items in contributions]
        energy = math.fsum(energies)
        if not math.isfinite(energy) or energy <= 0.0:
            raise SimulationError("connected network conductance is not numerically resolved")
        residual = max((abs(net[i]) for i in unknown), default=0.0) / energy
        imbalance = abs(net[network.source] + net[network.sink]) / energy
        power_error = abs(net[network.source] - energy) / energy
        if max(residual, imbalance, power_error) > 1e-7:
            raise SimulationError(
                "network conservation checks failed; conductances are ill-conditioned"
            )
        conductance = energy * scale
        unit_currents *= scale
    total_current = conductance * voltage
    power = total_current * voltage
    scaled_currents = tuple(float(value) * voltage for value in unit_currents)
    if not all(
        math.isfinite(value) for value in (conductance, total_current, power, *scaled_currents)
    ):
        raise SimulationError("network result exceeds float64 range")
    if connected and min(conductance, total_current, power) <= 0.0:
        raise SimulationError("network result underflows float64")
    floating_set = set(floating)
    return NetworkSolution(
        connected,
        voltage,
        conductance,
        total_current,
        power,
        tuple(None if i in floating_set else float(p) * voltage for i, p in enumerate(potentials)),
        scaled_currents,
        floating,
        residual,
        imbalance,
        power_error,
    )
