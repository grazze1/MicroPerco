# SPDX-License-Identifier: Apache-2.0
"""Analytic circuits and an independent dense Kirchhoff oracle."""

import math

import numpy as np
import pytest

from microperco import (
    ConstantConductanceModel,
    ResistorEdge,
    ResistorNetwork,
    TunnelingConductanceModel,
    solve_resistor_network,
)
from microperco.exceptions import ConfigurationError, SimulationError


def test_series_resistors_and_voltage_scaling() -> None:
    network = ResistorNetwork(3, (ResistorEdge(0, 1, 2), ResistorEdge(1, 2, 3)), 0, 2)
    result = solve_resistor_network(network, applied_voltage=5)
    assert result.effective_conductance == pytest.approx(1.2)
    assert result.node_voltages == pytest.approx((5, 2, 0))
    assert result.edge_currents == pytest.approx((6, 6))
    assert result.total_current == pytest.approx(6)
    assert result.dissipated_power == pytest.approx(30)
    assert result.relative_kirchhoff_residual < 1e-14


def test_parallel_resistors_and_reversed_edge_orientation() -> None:
    network = ResistorNetwork(2, (ResistorEdge(0, 1, 2), ResistorEdge(1, 0, 3)), 0, 1)
    result = solve_resistor_network(network)
    assert result.effective_conductance == 5
    assert result.edge_currents == (2, -3)


def test_balanced_wheatstone_bridge_and_dangling_branch() -> None:
    edges = tuple(
        ResistorEdge(i, j, 1)
        for i, j in (
            (0, 1),
            (1, 3),
            (0, 2),
            (2, 3),
            (1, 2),
            (1, 4),
        )
    )
    result = solve_resistor_network(ResistorNetwork(5, edges, 0, 3))
    assert result.effective_conductance == pytest.approx(1)
    assert result.node_voltages == pytest.approx((1, 0.5, 0.5, 0, 0.5))
    assert result.edge_currents[-2:] == pytest.approx((0, 0), abs=1e-15)


def test_disconnected_and_floating_components() -> None:
    network = ResistorNetwork(
        7, (ResistorEdge(0, 1, 1), ResistorEdge(2, 3, 1), ResistorEdge(4, 5, 1)), 0, 3
    )
    result = solve_resistor_network(network, applied_voltage=2)
    assert not result.connected
    assert result.effective_conductance == result.total_current == result.dissipated_power == 0
    assert result.node_voltages == (2, 2, 0, 0, None, None, None)
    assert result.floating_nodes == (4, 5, 6)
    assert result.edge_currents == (0, 0, 0)


@pytest.mark.parametrize("scale", [1e-250, 1.0, 1e250])
def test_uniform_conductance_scaling(scale: float) -> None:
    network = ResistorNetwork(3, (ResistorEdge(0, 1, scale), ResistorEdge(1, 2, scale)), 0, 2)
    result = solve_resistor_network(network)
    assert result.effective_conductance / scale == pytest.approx(0.5)
    assert result.node_voltages == (1, 0.5, 0)


@pytest.mark.parametrize("seed", range(10))
def test_sparse_solver_matches_independent_dense_oracle(seed: int) -> None:
    rng = np.random.default_rng(seed)
    size = 12
    raw = [(i, i + 1, float(rng.uniform(0.1, 10))) for i in range(size - 1)]
    raw += [
        (i, j, float(rng.uniform(0.1, 10)))
        for i in range(size)
        for j in range(i + 2, size)
        if rng.random() < 0.3
    ]
    edges = tuple(ResistorEdge(*edge) for edge in raw)
    incidence = np.zeros((len(edges), size))
    for k, edge in enumerate(edges):
        incidence[k, edge.i] = 1
        incidence[k, edge.j] = -1
    weights = np.array([edge.conductance for edge in edges])
    laplacian = incidence.T @ np.diag(weights) @ incidence
    expected = np.zeros(size)
    expected[0] = 1
    expected[1:-1] = np.linalg.solve(laplacian[1:-1, 1:-1], -laplacian[1:-1, 0])
    currents = weights * (incidence @ expected)
    result = solve_resistor_network(ResistorNetwork(size, edges, 0, size - 1))
    assert result.node_voltages == pytest.approx(expected, abs=1e-13)
    assert result.edge_currents == pytest.approx(currents, abs=1e-12)
    assert result.effective_conductance == pytest.approx((laplacian @ expected)[0])
    assert result.relative_current_imbalance < 1e-12
    assert result.relative_power_error < 1e-12


def test_adding_an_edge_cannot_reduce_conductance() -> None:
    edges = (ResistorEdge(0, 1, 1), ResistorEdge(1, 2, 1))
    before = solve_resistor_network(ResistorNetwork(3, edges, 0, 2))
    after = solve_resistor_network(ResistorNetwork(3, (*edges, ResistorEdge(0, 2, 0.3)), 0, 2))
    assert after.effective_conductance == pytest.approx(before.effective_conductance + 0.3)


def test_models_obey_cutoff_and_tunneling_law() -> None:
    constant = ConstantConductanceModel(4, 0.5)
    tunneling = TunnelingConductanceModel(4, 0.2, 0.5)
    assert constant.conductance(0.5) == 4
    assert constant.conductance(math.nextafter(0.5, 1)) == 0
    assert tunneling.conductance(0) == 4
    assert tunneling.conductance(0.2) == pytest.approx(4 * math.exp(-2))
    assert tunneling.conductance(0.5) > 0
    assert tunneling.conductance(math.nextafter(0.5, 1)) == 0


@pytest.mark.parametrize("invalid", [-1, 0, float("nan"), float("inf"), True])
def test_invalid_conductances_and_voltages(invalid: float) -> None:
    with pytest.raises(ConfigurationError):
        ResistorEdge(0, 1, invalid)
    with pytest.raises(ConfigurationError):
        ConstantConductanceModel(invalid)
    with pytest.raises(ConfigurationError):
        TunnelingConductanceModel(decay_length=invalid)
    with pytest.raises(ConfigurationError):
        solve_resistor_network(ResistorNetwork(2, (), 0, 1), applied_voltage=invalid)


@pytest.mark.parametrize(
    "args", [(2, (), 0, 0), (2, (), -1, 1), (True, (), 0, 1), (2, (ResistorEdge(0, 2, 1),), 0, 1)]
)
def test_invalid_networks(args: tuple) -> None:
    with pytest.raises(ConfigurationError):
        ResistorNetwork(*args)


def test_unrepresentable_tunneling_and_network_results_fail_explicitly() -> None:
    with pytest.raises(SimulationError, match="representable"):
        TunnelingConductanceModel(decay_length=1e-10, cutoff=1)
    with pytest.raises(SimulationError, match="range"):
        solve_resistor_network(
            ResistorNetwork(2, (ResistorEdge(0, 1, 1e308),), 0, 1), applied_voltage=10
        )
    with pytest.raises(SimulationError):
        solve_resistor_network(
            ResistorNetwork(
                3,
                (
                    ResistorEdge(0, 1, 1e300),
                    ResistorEdge(1, 2, 1e-300),
                ),
                0,
                2,
            )
        )


def test_extreme_tunneling_product_is_evaluated_in_log_space() -> None:
    model = TunnelingConductanceModel(1e300, 1, 500)
    assert model.conductance(500) == pytest.approx(math.exp(math.log(1e300) - 1000), rel=1e-12)
