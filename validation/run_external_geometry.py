# SPDX-License-Identifier: Apache-2.0
"""Compare the in-package GJK kernel with an independent geometry backend."""

from __future__ import annotations

import argparse
import json
import platform
from collections.abc import Callable
from pathlib import Path

import numpy as np
from geometry_cases import cylinder_cases

from microperco import ThresholdContactModel
from microperco.geometry import cylinder_cylinder_distance
from microperco.geometry.reference import cylinder_distance_hppfcl, cylinder_distance_scipy


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("scipy", "hppfcl"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    reference: Callable[..., float]
    if args.backend == "scipy":
        reference = cylinder_distance_scipy
        dependency_version = __import__("scipy").__version__
    else:
        reference = cylinder_distance_hppfcl
        dependency_version = __import__("hppfcl").__version__
    errors = []
    records = []
    thresholds = (0.0, 0.2, 1.0, 1.8, 3.0)
    decision_disagreements = 0
    for index, (first, second) in enumerate(cylinder_cases()):
        computed = cylinder_cylinder_distance(first, second)
        expected = reference(first, second)
        error = abs(computed - expected)
        errors.append(error)
        scale = max(
            first.length,
            second.length,
            first.radius,
            second.radius,
            float(np.linalg.norm(second.center - first.center)),
            1.0,
        )
        disagreements = sum(
            ThresholdContactModel(threshold).accepts(computed, scale=scale)
            != ThresholdContactModel(threshold).accepts(expected, scale=scale)
            for threshold in thresholds
        )
        decision_disagreements += disagreements
        records.append(
            {
                "case": index,
                "computed": computed,
                "reference": expected,
                "absolute_error": error,
                "fixed_threshold_disagreements": disagreements,
            }
        )
    payload = {
        "backend": args.backend,
        "backend_version": dependency_version,
        "case_count": len(records),
        "max_absolute_error": max(errors),
        "mean_absolute_error": float(np.mean(errors)),
        "fixed_thresholds": thresholds,
        "decision_comparisons": len(records) * len(thresholds),
        "decision_disagreements": decision_disagreements,
        "python": platform.python_version(),
        "records": records,
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
