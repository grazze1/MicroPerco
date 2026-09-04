# SPDX-License-Identifier: Apache-2.0
"""Public exception hierarchy."""


class MicroPercoError(Exception):
    """Base class for expected MicroPerco failures."""


class GeometryError(MicroPercoError, ValueError):
    """Invalid or numerically unrepresentable geometry."""


class ConfigurationError(MicroPercoError, ValueError):
    """Invalid user configuration."""


class SimulationError(MicroPercoError):
    """A simulation could not produce a valid result."""


class OptimizationError(MicroPercoError):
    """An inverse-design request could not be evaluated safely."""
