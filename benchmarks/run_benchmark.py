# SPDX-License-Identifier: Apache-2.0
"""Release benchmark for exhaustive and optimized contact search."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from microperco import Domain, SphereSpec, ThresholdContactModel
from microperco.benchmarking import benchmark_contact_search
from microperco.io import to_jsonable


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    suite = benchmark_contact_search(
        Domain((50.0, 50.0, 50.0), True),
        SphereSpec(0.5),
        ThresholdContactModel(0.2),
        (10, 50, 100, 500, 1000),
        repeats=args.repeats,
        warmup=1,
        seed=42,
    )
    text = json.dumps(to_jsonable(suite), indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
