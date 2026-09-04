# SPDX-License-Identifier: Apache-2.0
"""Disjoint-set structures for connectivity and periodic winding."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class UnionFind:
    """Union by size with path compression."""

    def __init__(self, size: int) -> None:
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError("size must be a non-negative integer")
        self._parent = list(range(size))
        self._size = [1] * size

    def __len__(self) -> int:
        return len(self._parent)

    def _check(self, item: int) -> None:
        if isinstance(item, bool) or not isinstance(item, int) or not 0 <= item < len(self):
            raise IndexError("union-find index out of range")

    def find(self, item: int) -> int:
        self._check(item)
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[item] != item:
            parent = self._parent[item]
            self._parent[item] = root
            item = parent
        return root

    def union(self, first: int, second: int) -> bool:
        root_a = self.find(first)
        root_b = self.find(second)
        if root_a == root_b:
            return False
        if self._size[root_a] < self._size[root_b]:
            root_a, root_b = root_b, root_a
        self._parent[root_b] = root_a
        self._size[root_a] += self._size[root_b]
        return True

    def connected(self, first: int, second: int) -> bool:
        return self.find(first) == self.find(second)

    @property
    def component_count(self) -> int:
        return sum(self.find(item) == item for item in range(len(self)))


class WeightedUnionFind:
    """Union-find with integer lattice-vector potentials."""

    def __init__(self, size: int) -> None:
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError("size must be a non-negative integer")
        self._parent = list(range(size))
        self._size = [1] * size
        self._weight = [(0, 0, 0) for _ in range(size)]

    def __len__(self) -> int:
        return len(self._parent)

    def _check(self, item: int) -> None:
        if isinstance(item, bool) or not isinstance(item, int) or not 0 <= item < len(self):
            raise IndexError("weighted union-find index out of range")

    @staticmethod
    def _add(
        first: tuple[int, int, int], second: tuple[int, int, int]
    ) -> tuple[int, int, int]:
        return (
            first[0] + second[0],
            first[1] + second[1],
            first[2] + second[2],
        )

    @staticmethod
    def _subtract(
        first: tuple[int, int, int], second: tuple[int, int, int]
    ) -> tuple[int, int, int]:
        return (
            first[0] - second[0],
            first[1] - second[1],
            first[2] - second[2],
        )

    @staticmethod
    def _negate(value: tuple[int, int, int]) -> tuple[int, int, int]:
        return -value[0], -value[1], -value[2]

    def _find_with_weight(self, item: int) -> tuple[int, tuple[int, int, int]]:
        self._check(item)
        if self._parent[item] == item:
            return item, (0, 0, 0)
        parent = self._parent[item]
        root, parent_weight = self._find_with_weight(parent)
        self._weight[item] = self._add(self._weight[item], parent_weight)
        self._parent[item] = root
        return root, self._weight[item]

    def find(self, item: int) -> int:
        return self._find_with_weight(item)[0]

    def potential(self, item: int) -> tuple[int, int, int]:
        _, value = self._find_with_weight(item)
        return value

    def union(self, first: int, second: int, delta: Sequence[int]) -> tuple[int, int, int] | None:
        """Enforce ``potential[second] - potential[first] = delta``."""

        values = tuple(delta)
        if len(values) != 3 or not all(
            isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
            for value in values
        ):
            raise ValueError("delta must be a length-three integer sequence")
        vector = int(values[0]), int(values[1]), int(values[2])
        root_a, weight_a = self._find_with_weight(first)
        root_b, weight_b = self._find_with_weight(second)
        if root_a == root_b:
            return self._subtract(self._subtract(weight_b, weight_a), vector)
        if self._size[root_a] >= self._size[root_b]:
            self._parent[root_b] = root_a
            self._weight[root_b] = self._subtract(self._add(vector, weight_a), weight_b)
            self._size[root_a] += self._size[root_b]
        else:
            self._parent[root_a] = root_b
            self._weight[root_a] = self._add(
                self._subtract(self._negate(vector), weight_a), weight_b
            )
            self._size[root_b] += self._size[root_a]
        return None

    @property
    def component_count(self) -> int:
        return sum(self.find(item) == item for item in range(len(self)))
