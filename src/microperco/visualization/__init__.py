# SPDX-License-Identifier: Apache-2.0
"""Optional Matplotlib visualizations.

Install ``microperco[plot]`` before importing this module.
"""

from .plots import (
    export_figure,
    plot_benchmark,
    plot_microstructure,
    plot_probability_curve,
    plot_spanning_cluster,
)

__all__ = [
    "export_figure",
    "plot_benchmark",
    "plot_microstructure",
    "plot_probability_curve",
    "plot_spanning_cluster",
]
