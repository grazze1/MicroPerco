# SPDX-License-Identifier: Apache-2.0
"""Generate README figures exclusively with Python/Matplotlib."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from figure_qa import require_matplotlib_panel_alignment
from matplotlib.transforms import ScaledTranslation

from microperco import (
    Cylinder,
    CylinderSpec,
    Domain,
    PopulationSpec,
    Sphere,
    SphereSpec,
    ThresholdContactModel,
    analyze_percolation,
    estimate_critical_loading,
    generate_microstructure,
)
from microperco.io import to_jsonable
from microperco.simulation import BenchmarkResult
from microperco.visualization import (
    plot_benchmark,
    plot_microstructure,
    plot_probability_curve,
    plot_spanning_cluster,
)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)
plt.rcParams["font.size"] = 7
plt.rcParams["axes.titlesize"] = 8
plt.rcParams["axes.labelsize"] = 7
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8
plt.rcParams["legend.fontsize"] = 7
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.linewidth"] = 0.7
plt.rcParams["legend.frameon"] = False

fig_width_mm = 183.0
demo_mode = True  # All generated microstructures are explicitly documented demonstrations.
EXPORT_SUFFIXES = (".svg", ".pdf", ".png")


def _panel_label(axis: object, label: str) -> None:
    offset = ScaledTranslation(-4 / 72, 3 / 72, axis.figure.dpi_scale_trans)
    axis.text(
        0,
        1,
        label,
        transform=axis.transAxes + offset,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def _save(figure: object, name: str, output: Path, *, panels: tuple[object, ...]) -> None:
    qa = output / "qa"
    qa.mkdir(parents=True, exist_ok=True)
    options: dict[str, object] = {}
    if len(panels) > 1:
        options.update(
            axes=list(panels),
            panel_ids=[chr(ord("a") + index) for index in range(len(panels))],
            row_groups=[[chr(ord("a") + index) for index in range(len(panels))]],
            require_panel_labels=True,
        )
    require_matplotlib_panel_alignment(
        figure,
        json_out=qa / f"{name}.alignment.json",
        overlay_svg=qa / f"{name}.alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        strict=True,
        **options,
    )
    for suffix in EXPORT_SUFFIXES:
        figure.savefig(
            output / f"{name}{suffix}",
            dpi=600,
            facecolor="white",
        )
    plt.close(figure)


def _benchmark_records(path: Path) -> tuple[BenchmarkResult, ...]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return tuple(BenchmarkResult(**record) for record in raw["results"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "assets")
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    mixed_domain = Domain((10.0, 10.0, 10.0), (False, True, True))
    mixed = generate_microstructure(
        mixed_domain,
        (
            PopulationSpec(CylinderSpec(0.22, 3.4), 32, "fibers"),
            PopulationSpec(SphereSpec(0.38), 18, "beads"),
        ),
        seed=20260904,
    )
    micro_figure, micro_axis = plot_microstructure(mixed.particles, mixed_domain)
    micro_figure.set_size_inches(6.7, 4.4)
    micro_axis.set_title("Seeded mixed-particle microstructure")
    _save(micro_figure, "microstructure_3d", output, panels=(micro_axis,))

    chain_domain = Domain((10.0, 6.0, 6.0), False)
    chain = [
        Sphere((coordinate, 0.0, 0.0), 1.0, index)
        for index, coordinate in enumerate((-4.0, -2.0, 0.0, 2.0, 4.0))
    ]
    rng = np.random.default_rng(2701)
    background = [
        Sphere(
            (
                rng.uniform(-4.5, 4.5),
                rng.choice((-2.4, 2.4)),
                rng.uniform(-2.4, 2.4),
            ),
            0.2,
            index + 5,
        )
        for index in range(15)
    ]
    spanning_particles = tuple(chain + background)
    spanning_result = analyze_percolation(spanning_particles, chain_domain)
    spanning_figure, spanning_axis = plot_spanning_cluster(
        spanning_particles, chain_domain, spanning_result
    )
    spanning_figure.set_size_inches(6.7, 4.4)
    _save(spanning_figure, "spanning_cluster", output, panels=(spanning_axis,))

    critical = estimate_critical_loading(
        Domain((10.0, 10.0, 10.0), (False, True, True)),
        SphereSpec(1.0),
        ThresholdContactModel(0.2),
        loading_grid=(40, 50, 60, 70, 80, 90, 100),
        target_probability=0.8,
        search_trials=100,
        certification_trials=160,
        confidence=0.95,
        seed=314159,
    )
    probability_figure, probability_axis = plot_probability_curve(
        critical.search_estimates,
        target_probability=critical.target_probability,
        critical_count=critical.critical_count,
    )
    probability_figure.set_size_inches(3.5, 2.6)
    _save(
        probability_figure,
        "percolation_curve",
        output,
        panels=(probability_axis,),
    )

    benchmark_path = Path(__file__).parents[1] / "benchmarks" / "benchmark_results.json"
    benchmark_records = _benchmark_records(benchmark_path)
    benchmark_figure, benchmark_axes = plot_benchmark(benchmark_records)
    benchmark_figure.set_size_inches(7.2, 2.4)
    for axis, label in zip(benchmark_axes, ("a", "b", "c"), strict=True):
        _panel_label(axis, label)
    _save(benchmark_figure, "benchmark", output, panels=benchmark_axes)

    figure_data = {
        "microstructure": {
            "seed": 20260904,
            "domain": to_jsonable(mixed_domain),
            "particles": [
                {
                    "shape": "cylinder" if isinstance(particle, Cylinder) else "sphere",
                    "center": particle.center.tolist(),
                    "radius": particle.radius,
                    **(
                        {"axis": particle.axis.tolist(), "length": particle.length}
                        if isinstance(particle, Cylinder)
                        else {}
                    ),
                }
                for particle in mixed.particles
            ],
        },
        "spanning_cluster": {
            "background_seed": 2701,
            "particles": [to_jsonable(particle.center) for particle in spanning_particles],
            "spanning_path": spanning_result.spanning_path,
        },
        "percolation_curve": {
            "seed": 314159,
            "nested_trials_per_count": 100,
            "result": to_jsonable(critical),
        },
        "benchmark": json.loads(benchmark_path.read_text(encoding="utf-8")),
    }
    (output / "figure_data.json").write_text(
        json.dumps(figure_data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
