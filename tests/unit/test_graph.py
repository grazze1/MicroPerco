# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from microperco.graph import WeightedUnionFind


def test_weighted_union_find_preserves_arbitrary_precision_lattice_vectors() -> None:
    huge = 10**30
    graph = WeightedUnionFind(3)
    assert graph.union(0, 1, (huge, -huge, 1)) is None
    assert graph.union(1, 2, (huge, 2, -huge)) is None
    assert graph.potential(2) == (2 * huge, -huge + 2, 1 - huge)
    assert graph.union(0, 2, (2 * huge, -huge + 2, 1 - huge)) == (0, 0, 0)
    assert graph.union(0, 2, (2 * huge + 1, -huge + 2, 1 - huge)) == (-1, 0, 0)
