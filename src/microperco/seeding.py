# SPDX-License-Identifier: Apache-2.0
"""Serializable provenance for externally constructed NumPy seed sequences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

SeedEntropy: TypeAlias = int | tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SeedSequenceState:
    """Initialization state needed to reconstruct a NumPy ``SeedSequence``."""

    entropy: SeedEntropy
    spawn_key: tuple[int, ...]
    pool_size: int


SeedProvenance: TypeAlias = int | tuple[int, ...] | SeedSequenceState | None
