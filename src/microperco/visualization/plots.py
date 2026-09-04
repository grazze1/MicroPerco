# SPDX-License-Identifier: Apache-2.0
"""Publication-aware static plots for MicroPerco results."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.axes3d import Axes3D

from ..domain import Domain, normalize_axis
from ..particles import Cylinder, Particle, Sphere
from ..percolation import PercolationResult
from ..simulation import BenchmarkResult, LoadingEstimate

COLORS = {
    "all": "#8FA3B8",
    "highlight": "#D1495B",
    "electrode": "#2A9D8F",
    "bruteforce": "#6C757D",
    "cell_list": "#0072B2",
    "target": "#E69F00",
}


def _new_axes_3d(ax: Axes3D | None) -> tuple[Figure, Axes3D]:
    if ax is not None:
        return cast(Figure, ax.figure), ax
    figure = plt.figure(figsize=(7.2, 5.4), constrained_layout=True)
    return figure, cast(Axes3D, figure.add_subplot(111, projection="3d"))


def _box_edges(domain: Domain) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    lower = np.asarray(domain.lower)
    upper = np.asarray(domain.upper)
    corners = np.asarray(
        [
            (x, y, z)
            for x in (lower[0], upper[0])
            for y in (lower[1], upper[1])
            for z in (lower[2], upper[2])
        ]
    )
    for first in range(8):
        for second in range(first + 1, 8):
            if np.count_nonzero(corners[first] != corners[second]) == 1:
                yield corners[first], corners[second]


def _face_vertices(domain: Domain, axis: int, upper_face: bool) -> list[np.ndarray]:
    lower = np.asarray(domain.lower)
    upper = np.asarray(domain.upper)
    coordinate = upper[axis] if upper_face else lower[axis]
    transverse = [index for index in range(3) if index != axis]
    vertices: list[np.ndarray] = []
    for first, second in ((0, 0), (1, 0), (1, 1), (0, 1)):
        point = np.asarray(domain.center, dtype=np.float64)
        point[axis] = coordinate
        point[transverse[0]] = (lower, upper)[first][transverse[0]]
        point[transverse[1]] = (lower, upper)[second][transverse[1]]
        vertices.append(point)
    return vertices


def plot_microstructure(
    particles: Sequence[Particle],
    domain: Domain,
    *,
    highlighted: Sequence[int] = (),
    electrode_axis: int | str = "x",
    show_electrodes: bool = True,
    ax: Axes3D | None = None,
) -> tuple[Figure, Axes3D]:
    """Plot the finite domain, particles, and optional electrode faces."""

    items = tuple(particles)
    highlights = set(highlighted)
    if not all(isinstance(item, (Sphere, Cylinder)) for item in items):
        raise TypeError("particles must contain Sphere or Cylinder instances")
    if any(index < 0 or index >= len(items) for index in highlights):
        raise IndexError("highlighted particle index is out of range")
    axis = normalize_axis(electrode_axis)
    figure, axes = _new_axes_3d(ax)
    for start, end in _box_edges(domain):
        axes.plot(*zip(start, end, strict=True), color="#495057", linewidth=0.7, alpha=0.55)
    if show_electrodes:
        for upper_face in (False, True):
            collection = Poly3DCollection(
                [_face_vertices(domain, axis, upper_face)],
                facecolor=COLORS["electrode"],
                edgecolor="none",
                alpha=0.11,
            )
            axes.add_collection3d(collection)
    scale = 900.0 / max(domain.size)
    for index, particle in enumerate(items):
        color = COLORS["highlight"] if index in highlights else COLORS["all"]
        alpha = 0.95 if index in highlights else 0.55
        if isinstance(particle, Sphere):
            axes.scatter(
                *particle.center,
                s=max(7.0, (particle.radius * scale) ** 2),
                color=color,
                alpha=alpha,
                edgecolors="none",
            )
        else:
            start, end = particle.endpoints
            axes.plot(
                *zip(start, end, strict=True),
                color=color,
                alpha=alpha,
                linewidth=max(0.8, particle.radius * scale * 0.08),
                solid_capstyle="round",
            )
    padding = tuple(0.04 * length for length in domain.size)
    axes.set(
        xlim=(domain.lower[0] - padding[0], domain.upper[0] + padding[0]),
        ylim=(domain.lower[1] - padding[1], domain.upper[1] + padding[1]),
        zlim=(domain.lower[2] - padding[2], domain.upper[2] + padding[2]),
        xlabel="x",
        ylabel="y",
        zlabel="z",
    )
    axes.tick_params(pad=6)
    axes.xaxis.labelpad = 3
    axes.yaxis.labelpad = 5
    axes.zaxis.labelpad = 5
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_zticks([])
    axes.set_box_aspect(domain.size)
    axes.view_init(elev=21, azim=-58)
    return figure, axes


def plot_spanning_cluster(
    particles: Sequence[Particle],
    domain: Domain,
    result: PercolationResult,
    *,
    ax: Axes3D | None = None,
) -> tuple[Figure, Axes3D]:
    """Highlight the representative particle path in a percolating sample."""

    if not isinstance(result, PercolationResult):
        raise TypeError("result must be a PercolationResult")
    figure, axes = plot_microstructure(
        particles,
        domain,
        highlighted=result.spanning_path,
        electrode_axis=result.axis,
        ax=ax,
    )
    axes.set_title("Spanning cluster" if result.percolates else "No spanning cluster")
    return figure, axes


def plot_probability_curve(
    estimates: Sequence[LoadingEstimate],
    *,
    target_probability: float,
    critical_count: int | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot Monte Carlo points with their actual asymmetric Wilson intervals."""

    records = tuple(estimates)
    if not records:
        raise ValueError("estimates must not be empty")
    if not 0.0 <= target_probability <= 1.0:
        raise ValueError("target_probability must lie between zero and one")
    if ax is None:
        figure, axes = plt.subplots(figsize=(6.7, 4.2), constrained_layout=True)
    else:
        figure, axes = cast(Figure, ax.figure), ax
    x = np.asarray([record.count for record in records])
    y = np.asarray([record.probability for record in records])
    lower = np.asarray([record.confidence_interval.lower for record in records])
    upper = np.asarray([record.confidence_interval.upper for record in records])
    axes.errorbar(
        x,
        y,
        yerr=np.vstack((y - lower, upper - y)),
        marker="o",
        linestyle="none",
        color=COLORS["cell_list"],
        capsize=3,
        label="Monte Carlo estimate (Wilson CI)",
    )
    fitted = [record.fitted_probability for record in records]
    if all(value is not None for value in fitted):
        axes.plot(
            x,
            [float(value) for value in fitted if value is not None],
            color="#4D4D4D",
            linewidth=1.4,
            label="Monotone fit",
        )
    axes.axhline(
        target_probability,
        color=COLORS["target"],
        linestyle="--",
        linewidth=1.2,
        label="Target probability",
    )
    if critical_count is not None:
        axes.axvline(
            critical_count,
            color=COLORS["highlight"],
            linestyle=":",
            linewidth=1.4,
            label="Estimated critical count",
        )
    axes.set(xlabel="Particle count", ylabel="Percolation probability", ylim=(-0.03, 1.03))
    axes.grid(alpha=0.22)
    axes.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=1,
    )
    return figure, axes


def plot_benchmark(
    results: Sequence[BenchmarkResult],
) -> tuple[Figure, tuple[Axes, Axes, Axes]]:
    """Plot runtime quartiles and broad-/narrow-phase work counters."""

    records = tuple(results)
    if not records:
        raise ValueError("results must not be empty")
    figure, raw_axes = plt.subplots(1, 3, figsize=(12.0, 3.7), constrained_layout=True)
    axes = cast(tuple[Axes, Axes, Axes], tuple(raw_axes))
    for backend in sorted({record.backend for record in records}):
        group = sorted(
            (record for record in records if record.backend == backend),
            key=lambda record: record.problem_size,
        )
        x = np.asarray([record.problem_size for record in group])
        median = np.asarray([record.median_seconds for record in group])
        q1 = np.asarray([record.first_quartile_seconds for record in group])
        q3 = np.asarray([record.third_quartile_seconds for record in group])
        color = COLORS.get(backend, "#333333")
        display_label = {
            "bruteforce": "Brute force",
            "cell_list": "Cell list",
        }.get(backend, backend)
        axes[0].errorbar(
            x,
            median,
            yerr=np.vstack((median - q1, q3 - median)),
            marker="o",
            capsize=3,
            color=color,
            label=display_label,
        )
        axes[1].plot(
            x,
            [
                np.nan if record.candidate_pairs is None else record.candidate_pairs
                for record in group
            ],
            marker="o",
            color=color,
            label=display_label,
        )
        axes[2].plot(
            x,
            [
                np.nan if record.distance_evaluations is None else record.distance_evaluations
                for record in group
            ],
            marker="o",
            color=color,
            label=display_label,
        )
    for index, (current, ylabel) in enumerate(
        zip(
            axes,
            ("Median runtime (s)", "Candidate pairs", "Exact distance evaluations"),
            strict=True,
        )
    ):
        current.set(xlabel="Particle count", ylabel=ylabel, xscale="log")
        if index == 0:
            current.set_yscale("log")
        else:
            current.set_yscale("symlog", linthresh=1.0)
            current.set_ylim(bottom=0.0)
        current.grid(alpha=0.2, which="both")
    axes[0].legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=2,
    )
    return figure, axes


def export_figure(figure: Figure, path: str | Path, *, dpi: int = 240) -> Path:
    """Export a figure as PNG, SVG, or PDF."""

    destination = Path(path)
    if destination.suffix.lower() not in {".png", ".svg", ".pdf"}:
        raise ValueError("figure extension must be .png, .svg, or .pdf")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=dpi, bbox_inches="tight", facecolor="white")
    return destination
