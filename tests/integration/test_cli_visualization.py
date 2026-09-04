# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from microperco import Domain, Sphere, analyze_percolation
from microperco.cli import main
from microperco.simulation import BenchmarkResult, LoadingEstimate
from microperco.statistics import ConfidenceInterval
from microperco.visualization import (
    export_figure,
    plot_benchmark,
    plot_microstructure,
    plot_probability_curve,
    plot_spanning_cluster,
)

CLI_CONFIG = """
schema_version: 1
domain:
  size: [6, 6, 6]
  periodic: [false, false, false]
particles:
  - name: bead
    shape: sphere
    radius: 0.5
    count: 2
contact:
  threshold: 0.1
percolation:
  axis: x
simulation:
  trials: 2
  seed: 42
  confidence: 0.9
critical:
  population: bead
  counts: [0, 1, 2]
  target_probability: 0.5
  search_trials: 2
  certification_trials: 2
optimization:
  count_bounds:
    bead: [0, 1]
  target_probability: 0.5
  search_trials: 1
  certification_trials: 1
benchmark:
  population: bead
  particle_counts: [2, 3]
  repeats: 1
  warmup: 0
  seed: 42
"""


@pytest.fixture
def cli_config(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(CLI_CONFIG, encoding="utf-8")
    return path


@pytest.mark.parametrize("command", ["simulate", "critical", "optimize", "benchmark"])
def test_cli_workflows_write_standard_json(command: str, cli_config: Path, tmp_path: Path) -> None:
    output = tmp_path / f"{command}.json"
    assert main([command, str(cli_config), "--output", str(output)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)


def test_cli_validate_and_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["validate"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["passed"] is True
    with pytest.raises(SystemExit) as exception:
        main(["--version"])
    assert exception.value.code == 0


def _loading(count: int, probability: float) -> LoadingEstimate:
    interval = ConfidenceInterval(
        max(0.0, probability - 0.1),
        min(1.0, probability + 0.15),
        0.95,
        "wilson",
    )
    return LoadingEstimate(
        count,
        int(probability * 20),
        20,
        probability,
        interval,
        interval,
        interval.lower,
        interval.upper,
        probability,
    )


def test_visualizations_export_all_supported_formats(tmp_path: Path) -> None:
    domain = Domain(10.0, False)
    particles = tuple(
        Sphere((x, 0.0, 0.0), 1.0, index) for index, x in enumerate((-4.0, -2.0, 0.0, 2.0, 4.0))
    )
    result = analyze_percolation(particles, domain)
    figures = [
        plot_microstructure(particles, domain)[0],
        plot_spanning_cluster(particles, domain, result)[0],
        plot_probability_curve(
            (_loading(1, 0.1), _loading(2, 0.45), _loading(3, 0.9)),
            target_probability=0.8,
            critical_count=3,
        )[0],
    ]
    benchmark = tuple(
        BenchmarkResult(
            size,
            backend,
            5,
            runtime,
            runtime * 0.8,
            runtime * 1.3,
            runtime * 0.5,
            candidates,
            candidates,
            speedup,
        )
        for size, runtime, candidates, speedup in ((10, 0.01, 45, 1.0), (100, 0.2, 4950, 1.0))
        for backend in ("bruteforce", "cell_list")
    )
    figures.append(plot_benchmark(benchmark)[0])
    suffixes = (".png", ".svg", ".pdf", ".png")
    for index, (figure, suffix) in enumerate(zip(figures, suffixes, strict=True)):
        destination = export_figure(figure, tmp_path / f"figure_{index}{suffix}")
        assert destination.stat().st_size > 100
        plt.close(figure)


def test_export_rejects_unknown_format(tmp_path: Path) -> None:
    figure = plt.figure()
    with pytest.raises(ValueError, match="extension"):
        export_figure(figure, tmp_path / "figure.bmp")
    plt.close(figure)
