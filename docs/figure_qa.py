# SPDX-License-Identifier: Apache-2.0
"""Small, repository-local alignment audit used by ``create_assets.py``.

The helper deliberately depends only on Matplotlib.  It records panel bounds,
checks row alignment and gutter consistency, verifies requested panel labels,
and writes a lightweight SVG overlay for human inspection.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from xml.sax.saxutils import escape

from matplotlib.axes import Axes
from matplotlib.figure import Figure


def _panel_bounds_pt(figure: Figure, axis: Axes) -> tuple[float, float, float, float]:
    width, height = (value * 72.0 for value in figure.get_size_inches())
    position = axis.get_position()
    return (
        position.x0 * width,
        position.y0 * height,
        position.x1 * width,
        position.y1 * height,
    )


def _label_anchor_pt(figure: Figure, axis: Axes, label: str) -> tuple[float, float] | None:
    for artist in axis.texts:
        if artist.get_text() == label:
            display = artist.get_transform().transform(artist.get_position())
            scale = 72.0 / figure.dpi
            return float(display[0] * scale), float(display[1] * scale)
    return None


def _write_overlay(
    destination: Path,
    width: float,
    height: float,
    panels: Sequence[dict[str, object]],
) -> None:
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}pt" height="{height}pt" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    for panel in panels:
        left, bottom, right, top = panel["bbox_pt"]  # type: ignore[misc]
        panel_id = escape(str(panel["id"]))
        elements.append(
            f'<rect x="{left}" y="{height - top}" width="{right - left}" '
            f'height="{top - bottom}" fill="none" stroke="#D1495B" stroke-width="0.7"/>'
        )
        elements.append(
            f'<text x="{left + 2}" y="{height - top + 9}" font-family="sans-serif" '
            f'font-size="7" fill="#D1495B">{panel_id}</text>'
        )
    elements.append("</svg>")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(elements) + "\n", encoding="utf-8")


def require_matplotlib_panel_alignment(
    figure: Figure,
    *,
    json_out: Path,
    overlay_svg: Path,
    tolerance_pt: float = 1.5,
    gutter_tolerance_pt: float = 1.5,
    strict: bool = True,
    axes: Sequence[Axes] | None = None,
    panel_ids: Sequence[str] | None = None,
    row_groups: Sequence[Sequence[str]] | None = None,
    require_panel_labels: bool = False,
) -> dict[str, object]:
    """Audit a Matplotlib panel row and persist machine-readable evidence."""

    figure.canvas.draw()
    selected_axes = tuple(figure.axes if axes is None else axes)
    identifiers = (
        tuple(chr(ord("a") + index) for index in range(len(selected_axes)))
        if panel_ids is None
        else tuple(panel_ids)
    )
    if len(identifiers) != len(selected_axes):
        raise ValueError("panel_ids must contain one identifier per axis")

    width, height = (float(value * 72.0) for value in figure.get_size_inches())
    panels: list[dict[str, object]] = []
    bounds_by_id: dict[str, tuple[float, float, float, float]] = {}
    findings: list[dict[str, object]] = []
    for index, (axis, panel_id) in enumerate(zip(selected_axes, identifiers, strict=True)):
        bounds = _panel_bounds_pt(figure, axis)
        bounds_by_id[panel_id] = bounds
        panel: dict[str, object] = {
            "id": panel_id,
            "bbox_pt": list(bounds),
            "grid_id": "matplotlib-grid-1",
            "row_start": 0,
            "row_stop": 1,
            "col_start": index,
            "col_stop": index + 1,
        }
        anchor = _label_anchor_pt(figure, axis, panel_id)
        if anchor is not None:
            panel["panel_label"] = panel_id
            panel["panel_label_anchor_pt"] = list(anchor)
        elif require_panel_labels:
            findings.append(
                {
                    "severity": "FAIL",
                    "check": "panel_label",
                    "panels": [panel_id],
                    "detail": f"panel {panel_id!r} has no matching label",
                }
            )
        panels.append(panel)

    groups = tuple(tuple(group) for group in (row_groups or (identifiers,)))
    comparisons = 0
    for group in groups:
        if len(group) < 2:
            continue
        group_bounds = [bounds_by_id[panel_id] for panel_id in group]
        for boundary_name, coordinate in (("bottom", 1), ("top", 3)):
            comparisons += 1
            spread = max(item[coordinate] for item in group_bounds) - min(
                item[coordinate] for item in group_bounds
            )
            if spread > tolerance_pt:
                findings.append(
                    {
                        "severity": "FAIL",
                        "check": f"row_{boundary_name}_alignment",
                        "panels": list(group),
                        "detail": f"spread {spread:.3f} pt exceeds {tolerance_pt:.3f} pt",
                    }
                )
        if len(group_bounds) > 2:
            comparisons += 1
            gutters = [
                group_bounds[index + 1][0] - group_bounds[index][2]
                for index in range(len(group_bounds) - 1)
            ]
            spread = max(gutters) - min(gutters)
            if spread > gutter_tolerance_pt:
                findings.append(
                    {
                        "severity": "FAIL",
                        "check": "row_gutter_consistency",
                        "panels": list(group),
                        "detail": (
                            f"gutter spread {spread:.3f} pt exceeds "
                            f"{gutter_tolerance_pt:.3f} pt"
                        ),
                    }
                )

    fail_count = sum(item["severity"] == "FAIL" for item in findings)
    applicable = len(selected_axes) > 1
    payload: dict[str, object] = {
        "schema_version": 1,
        "applicable": applicable,
        "auditable": applicable,
        "verdict": "PASS" if applicable and fail_count == 0 else "NOT APPLICABLE",
        "backend": "python-matplotlib",
        "summary": {
            "fail": fail_count,
            "warn": 0,
            "comparisons": comparisons,
            "exemptions": 0,
        },
        "tolerances": {
            "alignment_pt": tolerance_pt,
            "gutter_pt": gutter_tolerance_pt,
        },
        "layout": {
            "schema_version": 1,
            "backend": "python-matplotlib",
            "figure": {"width_pt": width, "height_pt": height},
            "panels": panels,
            "row_groups": [
                {"id": f"row-{index}", "panels": list(group)}
                for index, group in enumerate(groups, start=1)
                if len(group) > 1
            ],
            "column_groups": [],
            "boundary_groups": [],
            "exemptions": [],
        },
        "findings": findings,
    }
    if applicable and fail_count:
        payload["verdict"] = "FAIL"
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_overlay(overlay_svg, width, height, panels)
    if strict and fail_count:
        raise RuntimeError(f"panel alignment audit failed with {fail_count} finding(s)")
    return payload
