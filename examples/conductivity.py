# SPDX-License-Identifier: Apache-2.0
"""Finite-electrode conductance and x/y/z conductivity on explicit particles."""

from microperco import (
    ConstantConductanceModel,
    Domain,
    Sphere,
    TunnelingConductanceModel,
    analyze_directional_conductivity,
)
from microperco.io import dumps_json

domain = Domain((4.5, 3.0, 3.0), False)
particles = (Sphere((-1.25, 0, 0), 1), Sphere((1.25, 0, 0), 1))
result = analyze_directional_conductivity(
    particles,
    domain,
    TunnelingConductanceModel(contact_conductance=2, decay_length=0.5, cutoff=0.5),
    electrode_model=ConstantConductanceModel(contact_conductance=10),
)
print(dumps_json({"sigma_x": result.sigma_x, "sigma_y": result.sigma_y, "sigma_z": result.sigma_z}))
