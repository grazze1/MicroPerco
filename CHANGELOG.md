# Changelog

All notable changes follow the principles of [Keep a Changelog](https://keepachangelog.com/).

## [1.0.0] - 2026-09-04

### Added

- Immutable sphere, finite flat-cylinder, material, population, and orthorhombic-domain models.
- Exact sphere distances and a support-map GJK kernel for cylinder pairs and finite electrode faces.
- Independently configurable periodic axes, complete image enumeration, and explicit face-to-face versus periodic-wrapping semantics.
- Auditable brute-force and periodic cell-list contact search backends.
- Union-Find and independent BFS connectivity solvers, including lattice-winding detection.
- Seeded random microstructures with isotropic cylinder orientations.
- Monte Carlo probability estimates with Wilson and exact Clopper-Pearson intervals.
- Nested critical-loading search with independent family-wise certification.
- Bounded, cost-ordered mixture optimization with family-wise certification.
- Strict YAML configuration, stable JSON output, CLI workflows, static visualizations, validation, and repeatable benchmarks.
