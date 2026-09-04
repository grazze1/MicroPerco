# Source algorithm audit

## Scope

The audited source solved a specific conductive-microstructure modeling problem. It mixed reusable ideas with case-specific units, filenames, data repairs, fixed geometry, and Q1–Q4 reporting. MicroPerco retains the general algorithms and deliberately excludes the problem-specific data layer.

## Reusable algorithmic ideas identified

| Area | Audited behavior | MicroPerco 1.0 treatment |
|---|---|---|
| Particle geometry | Finite flat cylinders and spheres | General immutable `Cylinder` and `Sphere` types |
| Narrow phase | Support-map GJK for cylinder pairs; analytic mixed/sphere gaps | Independently implemented float64 kernel with named numerical policy |
| Broad phase | Conservative capsule/AABB filtering | Periodic AABB spatial hash; capsule check is rejection-only |
| Boundary conditions | Periodic images and parent fragments | Arbitrary orthorhombic domains, all eight PBC flag combinations, explicit parent grouping |
| Connectivity | Union-Find with BFS checks | Both solvers available; agreement asserted internally |
| Sampling | Uniform centers and isotropic axes | Seeded `Generator`/`SeedSequence` implementation |
| Probability | Repeated Bernoulli trials | Monte Carlo result with Wilson and exact intervals |
| Critical loading | Nested common random prefixes and monotone fitting | PAVA/logistic/probit search plus independent certification |
| Mixture design | Integer search under a cost model | Complete bounded, cost-ordered enumeration with auditable records |

## Corrections and strengthened semantics

- Electrode contact is measured against the finite rectangular domain face, not an infinite plane. Enabled transverse periodic axes tile that rectangle.
- Face-to-face conduction and non-zero periodic winding are separate modes. Periodic analysis does not silently turn an electrode face into its opposite face.
- The historical “wrapped parent touches both electrodes” interpretation is an explicit `wrapped_parent=True` opt-in.
- Periodic image ranges are derived dynamically from body extents and threshold; they are not restricted to a fixed 27-image stencil.
- Cylinder axis normalization, derived volumes, endpoints, support directions, AABBs, and domain bounds reject overflow or underflow that would make results unrepresentable.
- Optimization certification allocates error to a fixed family of `2M` potential one-sided assertions before sampling. Critical-loading certification allocates error only to its one or two declared final assertions.

## Excluded material

Competition labels, repaired attachment columns, fixed nanometre dimensions, hard-coded costs, historical numerical answers, original plots, PDF/XLSX files, and generated result tables are not part of the core library. A data-free case-study scaffold documents how users may map an authorized dataset into the generic API.

## Provenance conclusion

All shipped Python source, tests, configuration, documentation, and plotting code were authored for this repository. No vendored code or unlicensed snippet was retained.
