# SPDX-License-Identifier: Apache-2.0
"""Shared test configuration."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "microperco-mpl-test"))
