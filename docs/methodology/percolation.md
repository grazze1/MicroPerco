# Percolation analysis

Each logical particle (or explicitly grouped parent) is a graph node. An undirected edge joins two nodes when their exact periodic surface gap is at most the threshold, including the configured numerical tolerance.

## Face-to-face mode

Two additional graph nodes represent the distinct finite lower and upper electrode rectangles. A sample percolates when those nodes share a connected component. The returned `PercolationResult` records accepted contact edges, electrode-touching particle indices, a deterministic representative spanning path, graph component count, candidate count, and exact-distance evaluation count.

Union-Find is the primary connectivity implementation. A separately constructed sorted-adjacency BFS computes the same decision and path; any decision disagreement raises an internal error. The public `solver` selection controls which recorded decision is used without removing the cross-check.

## Periodic-wrapping mode

Weighted Union-Find assigns each graph node an integer lattice potential. An edge constrains the potential difference by its image translation. Adding an edge within an existing component exposes a residual cycle vector. A non-zero component along the analysis axis certifies topological winding. Independent BFS potentials cross-check whether such a residual exists.

## Interpretation limits

`analyze_percolation` reports geometric, binary connectivity. For resistance and distance-dependent tunneling, v2.0 provides the separate `analyze_conductivity` and `analyze_directional_conductivity` APIs; see [electrical transport](conductivity.md). A connected threshold graph alone does not determine a conductivity value: junction laws, electrode contacts, and geometry factors are also required.

The transport model uses equipotential particles and finite junction resistances. Contact area, intrinsic particle resistance, excluded-volume relaxation, and mechanical deformation remain outside the current model. Nominal volume fraction sums particle volumes and can exceed occupied union volume when bodies overlap.
