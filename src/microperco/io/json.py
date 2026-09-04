# SPDX-License-Identifier: Apache-2.0
"""Standards-compliant deterministic JSON serialization."""

from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from pathlib import Path

import numpy as np

from ..exceptions import ConfigurationError


def to_jsonable(value: object) -> object:
    """Recursively convert result objects and NumPy scalars to JSON values."""

    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: to_jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"cannot serialize {type(value).__name__} to JSON")


def dumps_json(value: object) -> str:
    """Return stable JSON and reject NaN/Infinity."""

    try:
        return (
            json.dumps(
                to_jsonable(value),
                allow_nan=False,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            + "\n"
        )
    except (TypeError, ValueError) as exc:
        raise ConfigurationError("result contains a non-JSON value") from exc


def dump_json(value: object, path: Path | None = None) -> str:
    """Return JSON and optionally write it to ``path``."""

    payload = dumps_json(value)
    if path is not None:
        path.write_text(payload, encoding="utf-8")
    return payload
