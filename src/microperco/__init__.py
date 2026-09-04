# SPDX-License-Identifier: Apache-2.0
"""MicroPerco: 3D microstructure percolation and inverse design."""

from ._version import __version__
from .contact import ContactEdge, ContactSearchResult, ThresholdContactModel, find_contacts
from .domain import Domain
from .exceptions import (
    ConfigurationError,
    GeometryError,
    MicroPercoError,
    OptimizationError,
    SimulationError,
)
from .generation import (
    CylinderSpec,
    MaterialSpec,
    Microstructure,
    ParticleSpec,
    PopulationSpec,
    SphereSpec,
    generate_microstructure,
    particle_count_for_volume_fraction,
    particle_volume,
    volume_fraction,
)
from .geometry import Cylinder, ImageOffset, Particle, Sphere, distance, periodic_distance
from .numerics import (
    DEFAULT_NUMERICAL_POLICY,
    DEFAULT_POLICY,
    EPS,
    FLOAT_DTYPE,
    MACHINE_EPSILON,
    NumericalPolicy,
)
from .optimization import DesignEstimate, OptimizationResult, optimize_mixture
from .percolation import PercolationResult, PeriodicTopologyEdge, analyze_percolation
from .simulation import (
    CriticalLoadingResult,
    LoadingEstimate,
    MonteCarloResult,
    MonteCarloSimulator,
    SeedProvenance,
    SeedSequenceState,
    estimate_critical_loading,
    estimate_percolation_probability,
)
from .statistics import (
    BinomialEstimate,
    ConfidenceInterval,
    fit_logistic,
    fit_probit,
    pava,
    wilson_interval,
)

__all__ = [
    "DEFAULT_NUMERICAL_POLICY",
    "DEFAULT_POLICY",
    "EPS",
    "FLOAT_DTYPE",
    "MACHINE_EPSILON",
    "BinomialEstimate",
    "ConfigurationError",
    "ConfidenceInterval",
    "ContactEdge",
    "ContactSearchResult",
    "CriticalLoadingResult",
    "Cylinder",
    "CylinderSpec",
    "DesignEstimate",
    "Domain",
    "GeometryError",
    "ImageOffset",
    "LoadingEstimate",
    "MaterialSpec",
    "MicroPercoError",
    "Microstructure",
    "MonteCarloResult",
    "MonteCarloSimulator",
    "NumericalPolicy",
    "OptimizationError",
    "OptimizationResult",
    "Particle",
    "ParticleSpec",
    "PercolationResult",
    "PeriodicTopologyEdge",
    "PopulationSpec",
    "SeedProvenance",
    "SeedSequenceState",
    "SimulationError",
    "Sphere",
    "SphereSpec",
    "ThresholdContactModel",
    "__version__",
    "analyze_percolation",
    "distance",
    "estimate_critical_loading",
    "estimate_percolation_probability",
    "find_contacts",
    "fit_logistic",
    "fit_probit",
    "generate_microstructure",
    "optimize_mixture",
    "particle_count_for_volume_fraction",
    "particle_volume",
    "pava",
    "periodic_distance",
    "volume_fraction",
    "wilson_interval",
]
