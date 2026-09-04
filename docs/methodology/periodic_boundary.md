# Periodic boundary conditions

The domain is orthorhombic, centered at an arbitrary point, and periodic independently along x, y, and z. Coordinates may remain unwrapped; the geometry layer derives lattice translations explicitly.

## Image completeness

For a pair of particle AABBs, MicroPerco solves an integer interval on each enabled axis for every lattice image that could lie within the contact threshold. Disabled axes admit only zero translation. The Cartesian product of those intervals is complete for the conservative AABB test and can contain translations outside $\{-1,0,1\}^3$ when particles are long relative to the box.

The brute-force oracle evaluates every base particle pair and every feasible contact image. The optimized backend hashes unfolded, padded periodic AABBs into cells. Duplicate candidate tuples are removed; exact geometry decides every retained edge.

Lattice indices are Python integers rather than fixed-width integers. Canonicalization,
minimum-image differences, and wrapping form `coordinate + index * box_length`
exactly from the input binary64 values and round only the final coordinate. This avoids
large-gauge cancellation changing a contact decision. Any particle-pair or transversely
tiled face query that would enumerate more than 1,000,000 images raises `GeometryError`
before materializing or iterating the Cartesian product.

## Parent fragments

Particles with equal non-null `parent_id` never create an inter-particle contact edge, but they always map to the same logical graph node. This preserves explicitly reconstructed parent continuity without manufacturing an ordinary self-contact edge. By default, fragment coordinates for one parent describe one coherent, common unwrapped lift. Independently wrapped fragments can declare an integer `image_offset` on every member of the parent group; mixed declared/undeclared offsets fail fast, as do offsets on disabled axes.

## Face semantics

Face-to-face mode opens the selected analysis axis while preserving periodicity on transverse axes. Finite electrode rectangles are tiled on those transverse axes. Periodicity along the analysis axis therefore does not implicitly merge the electrodes.

`wrapped_parent=True` additionally enables a documented historical interpretation in which a parent crossing the analysis seam contacts both electrode sides. It is opt-in because it can strongly change the physical result.

## Wrapping mode

`mode="periodic_wrap"` has no electrodes. It detects a graph cycle with non-zero integer lattice winding along the selected periodic axis. Ordinary inter-group contacts and separately recorded non-zero physical self-image intersections form the topology graph; zero-shift self-contact and threshold-only tunneling to the same particle remain excluded. Cross-fragment self-image evidence is evaluated only when a parent group supplies explicit `image_offset` values. `PercolationResult.topology_edges` is the auditable winding evidence, while `contact_edges` retains its ordinary `i < j` meaning. Union-Find lattice potentials and an independent BFS potential assignment must agree on the decision.
