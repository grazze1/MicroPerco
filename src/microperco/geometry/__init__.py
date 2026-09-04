# SPDX-License-Identifier: Apache-2.0
"""Public geometry API."""

from ..particles import Cylinder, ImageOffset, Particle, Sphere
from .distance import (
    cylinder_cylinder_distance,
    cylinder_distance_gjk,
    cylinder_rectangle_distance,
    cylinder_sphere_gap,
    distance,
    point_cylinder_distance,
    point_rectangle_distance,
    segment_segment_distance,
    sphere_cylinder_distance,
    sphere_rectangle_distance,
    sphere_sphere_distance,
    sphere_sphere_gap,
)
from .periodic import (
    LatticeShift,
    PeriodicDistance,
    aabb,
    candidate_lattice_shifts,
    face_gaps,
    periodic_distance,
)
from .reference import cylinder_distance_hppfcl, cylinder_distance_scipy

__all__ = [
    "Cylinder",
    "ImageOffset",
    "LatticeShift",
    "Particle",
    "PeriodicDistance",
    "Sphere",
    "aabb",
    "candidate_lattice_shifts",
    "cylinder_cylinder_distance",
    "cylinder_distance_gjk",
    "cylinder_distance_hppfcl",
    "cylinder_distance_scipy",
    "cylinder_rectangle_distance",
    "cylinder_sphere_gap",
    "distance",
    "face_gaps",
    "periodic_distance",
    "point_cylinder_distance",
    "point_rectangle_distance",
    "segment_segment_distance",
    "sphere_cylinder_distance",
    "sphere_rectangle_distance",
    "sphere_sphere_distance",
    "sphere_sphere_gap",
]
