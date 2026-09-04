# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path

import pytest

from microperco import Cylinder, Domain, Sphere, distance, periodic_distance


def test_golden_geometry_cases() -> None:
    path = Path(__file__).parents[1] / "data" / "golden_geometry.json"
    golden = json.loads(path.read_text(encoding="utf-8"))
    assert distance(Sphere((0, 0, 0), 1), Sphere((3, 0, 0), 1)) == pytest.approx(
        golden["sphere_gap"]
    )
    cylinder = Cylinder((0, 0, 0), (1, 0, 0), 2, 0.5)
    assert distance(cylinder, Sphere((2, 0, 0), 0.5)) == pytest.approx(
        golden["cylinder_endcap_gap"]
    )
    periodic = periodic_distance(
        Sphere((-4.6, 0, 0), 0.5),
        Sphere((4.6, 0, 0), 0.5),
        Domain(10, (True, False, False)),
    )
    assert list(periodic.lattice_shift) == golden["periodic_shift"]
