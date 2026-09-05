# Geometry kernel

MicroPerco models closed spheres and closed, finite right circular cylinders with flat end caps. All calculations use NumPy `float64`; invalid or unrepresentable derived geometry raises an exception instead of silently producing NaN or infinity.

## Analytic distances

For spheres with centers $c_1,c_2$ and radii $r_1,r_2$, the non-negative surface gap is

$$
d_{SS}=\max\left(0,\lVert c_2-c_1\rVert-r_1-r_2\right).
$$

The point-to-cylinder query decomposes a point into axial and radial coordinates. The Euclidean norm of positive axial and radial excess gives the exact distance to the closed flat cylinder. Sphere–cylinder distance subtracts the sphere radius and clamps at zero.

## Flat-cylinder GJK

Cylinder–cylinder and cylinder–rectangle gaps use a support-map form of the Gilbert–Johnson–Keerthi distance algorithm. For direction $d$, cylinder axis $u$, half-length $h$, radius $r$, and center $c$, the support point is

$$
s(d)=c+\operatorname{sign}(d\cdot u)hu+r\frac{d-(d\cdot u)u}{\lVert d-(d\cdot u)u\rVert},
$$

with the radial term omitted when its denominator is zero. This is a flat-cylinder support map; a capsule approximation is never accepted as the final distance.

Queries are translated and scaled before simplex iteration. The closest simplex
point is obtained by enumerating active faces and solving the constrained affine
projection. Exact-rational segment predicates provide additional safe
intersection witnesses for high-aspect cases. Termination uses a scale-aware
absolute/relative `NumericalPolicy`, duplicate-support detection, and a bounded
iteration count.

If a cylinder-pair or cylinder–rectangle GJK query stalls under a normal
iteration budget, a convex fallback evaluates the dual support problem. A
feasible point of the Minkowski difference is a global upper bound and any
feasible unit dual direction supplies a global lower bound. The cylinder-pair
path also evaluates a direct closest-point primal problem. A distance is
returned only when the resulting bracket closes to the requested policy
tolerance. If a severely ill-conditioned query cannot be certified,
`GeometryError` is raised instead of reporting an unsupported gap. Deliberately
setting an insufficient GJK iteration budget retains the same explicit
non-convergence behavior.

## Electrode geometry

An electrode is the finite rectangle forming a selected domain face. Sphere distance is point-to-rectangle minus radius; cylinder distance is a GJK query against the zero-thickness rectangle. This avoids false contacts for particles that are close to the face's infinite plane but lie beyond a transverse edge.

## Validation

Analytic cases cover end caps, sidewalls, perpendicular axes, near-parallel axes, overlaps, exact thresholds, sphere–cylinder, and sphere–sphere pairs. Cylinder pairs were checked against SciPy constrained optimization in the v2.0 regression rerun; the HPP-FCL comparison remains the recorded v1 baseline. See [the validation report](../../validation/VALIDATION_REPORT.md).

The v2.0 [transport module](conductivity.md) reuses these particle and finite-electrode gaps to assign junction conductances. Electrical cutoff and conductance-law semantics are documented separately from geometric distance accuracy.
