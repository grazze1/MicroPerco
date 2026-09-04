# SPDX-License-Identifier: Apache-2.0
"""Connectivity graph utilities."""

from __future__ import annotations

from typing import Any

from .union_find import UnionFind, WeightedUnionFind


def analyze_percolation(*args: Any, **kwargs: Any) -> Any:
    """Lazily forward to avoid a contact/graph import cycle."""

    from ..percolation import analyze_percolation as implementation

    return implementation(*args, **kwargs)


__all__ = ["UnionFind", "WeightedUnionFind", "analyze_percolation"]
