# Changelog

All notable changes follow the principles of [Keep a Changelog](https://keepachangelog.com/).

## [2.0.0] - 2026-09-05

### Added

- Validated immutable resistor multigraphs and a sparse Dirichlet Kirchhoff solver with branch currents, node voltages, Joule power, and conservation diagnostics.
- Constant and exponentially distance-dependent tunneling junction laws with explicit finite cutoffs and independent electrode contact parameters.
- Geometry-to-network conversion for spheres and flat-ended cylinders, transverse periodic image channels, and equipotential parent fragments.
- Finite-electrode effective conductivity and x/y/z measurements on the same realization.
- Seeded conductivity Monte Carlo sampling, retained samples and provenance, means, and descriptive standard errors.
- `microperco conductivity`, strict optional YAML configuration, runnable API and CLI examples, and physical/numerical methodology.
- Analytic circuit tests, independent dense-solver comparisons, all-axis periodic backend parity, and built-in transport validation checks.

### Compatibility and scope

- Existing v1 APIs, commands, and `schema_version: 1` configurations remain supported; the conductivity block is additive.
- Conductivity includes finite electrode junction resistance. Particles are equipotential, and directional measurements are not a full periodic conductivity tensor.
- Disconnected networks return zero; floating voltages serialize as `null`. Unrepresentable or unresolved electrical solves fail explicitly.

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
