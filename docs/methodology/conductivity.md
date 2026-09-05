# Resistor networks and directional conductivity (v2.0)

Available in the published [v2.0.0 release](https://github.com/grazze1/MicroPerco/releases/tag/v2.0.0). See the [documentation guide](../README.md) for the other workflows and [current validation](../../validation/VALIDATION_REPORT.md) for the completed release checks.

## Physical model and units

Each sphere or flat-ended cylinder is an equipotential conductor. Separate
particles are connected by finite junction conductances. The surrounding matrix
is insulating. Intrinsic particle resistance, contact area, temperature, field
dependence, and material-specific band structure are not modeled. Calibrate the
junction parameters for the system of interest; the default values are examples.

`ConstantConductanceModel(g0, cutoff)` assigns `g0` to gaps `d <= cutoff`.
`TunnelingConductanceModel(g0, decay_length, cutoff)` uses

\[
g(d)=\begin{cases}g_0\exp(-2d/\xi), & 0\le d\le d_\mathrm{cut},\\
0, & d>d_\mathrm{cut}.\end{cases}
\]

Here `decay_length` is the localization length `xi`, including the explicit
factor of two in the exponent. Overlap and physical contact both have `d=0` and
finite conductance `g0`. The distance-dependent junction model follows the
tunneling-network formulation discussed by
[Ambrosetti et al., 2010](https://arxiv.org/abs/1004.4728).
The finite cutoff is a computational/modeling approximation to a global
tunneling network: increase it and check convergence for your own observable.
The package does not claim a universal cutoff error bound.

Lengths must share a unit; decay length and cutoff use that same unit. With
junction conductances in siemens and lengths in metres, effective conductivity
is in S/m. A geometry expressed in micrometres produces S/micrometre, which must
be multiplied by `1e6` to report S/m. Applied voltage must be positive and finite.

## Geometry, boundaries, and fragment identities

`build_conductivity_network` opens the measurement axis and retains the supplied
transverse periodic flags. The two finite rectangular electrode faces are held
at `applied_voltage` (lower face) and zero (upper face). Their surface gaps use
the existing finite-face geometry with transverse periodic tiling. The optional
`electrode_model` can differ from the particle junction model; otherwise the
same law applies to both. Electrode junctions are finite resistors, so a single
particle spanning the box has two electrode resistances in series.

The optimized cell list and brute-force oracle share the established geometry
kernel. Both enumerate relevant periodic image contacts, including multiple
images of a particle pair. Each distinct particle-image junction is a parallel
resistor. Transverse self-image loops connect an equipotential node to itself
and carry zero current, so they are omitted.

Equal non-null `parent_id` values collapse reconstructed fragments into one
equipotential node, matching the v1 continuity convention. Different fragment
pair junctions count as separate channels. Do not provide duplicate geometric
fragments if this would count a physical junction twice. Each logical node has
at most one junction to each electrode, evaluated at the minimum gap over its
fragments. Electrode area or multiplicity is not an additional conductance
factor. The historical `wrapped_parent` seam shortcut is not used.

The contact search admits a numerical tolerance halo, but the electrical law
applies an exact cutoff to the reported non-negative gap. Consequently a v1
threshold edge infinitesimally outside the cutoff can be absent from the
electrical network. Junction records retain particle indices, gaps, and lattice
shifts; their order matches the resistor and current records. A face gap records
the minimum distance to the tiled electrode; its lattice-shift field is zero
because the electrode is a single terminal.

## Kirchhoff solution and numerical checks

`ResistorNetwork` is a reusable two-terminal multigraph independent of geometry.
`solve_resistor_network` solves the Dirichlet graph Laplacian using
[SciPy's sparse direct solver](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.spsolve.html).
The free-node equations are

\[
\sum_j g_{ij}(v_i-v_j)=0.
\]

Voltages are solved at a unit terminal difference, and conductances are scaled
by the largest conductance in the terminal-spanning component. Source-only and
sink-only components are assigned their terminal potential exactly. Components
touching neither terminal have undefined absolute potential: their voltages
are `None` (`null` in JSON), their currents are zero. No artificial leakage or
diagonal regularization is introduced. Disconnected terminals return exactly
zero effective conductance and conductivity.

The solution records oriented branch currents `g_ij * (v_i-v_j)` and Joule
power. Unit-voltage energy determines effective conductance. Free-node current
balance, source/sink current balance, and source-current/energy agreement are
checked relative to this conductance with a `1e-7` limit. Voltages must obey the
maximum principle within `1e-10` before roundoff clipping. These are diagnostic
acceptance limits, not certified forward-error bounds. Singular systems,
unresolved conservation, unrepresentable tunneling weights, and overflow or
underflow of the total result raise `SimulationError`; they do not become a
false insulating result. Very large ratios of junction strengths can exceed
binary64 resolution even when every junction individually fits in a float.

## Conductivity and sampling

For separation `L` and transverse area `A`, the reported apparent conductivity
is `sigma = G * L / A`, where `G = I / V`. It includes the finite electrode
junction resistance. It is not an intrinsic bulk conductivity with electrode
resistance removed.

`analyze_directional_conductivity` runs three independent boundary-value
problems on the same particles and returns `sigma_x`, `sigma_y`, and `sigma_z`.
These are directional two-terminal measurements, not the full homogenized
conductivity tensor: off-diagonal responses and an imposed affine field in a
fully periodic cell are not implemented in v2.0.

`estimate_conductivity` spawns one NumPy child seed per trial and shares that
trial's particles across all requested axes. Reordering/subsetting axes does
not change samples. Returned provenance includes geometry, populations, laws,
voltage, search backend, and seed initialization state, including generated
entropy when no seed was supplied. Each axis retains all conductivity samples,
their arithmetic mean, conducting-trial count, and sample-standard-deviation
divided by `sqrt(trials)`. For one trial the standard error is `null`. These are
descriptive Monte Carlo statistics, not a binomial interval or a certification
of material performance. Individual detailed networks are available through
`analyze_conductivity`; the Monte Carlo result keeps compact samples.

## API and CLI

See [the explicit two-sphere example](../../examples/conductivity.py) and
[the complete YAML configuration](../../configs/conductivity.yaml).

```bash
microperco conductivity configs/conductivity.yaml --output conductivity.json
```

The optional `conductivity` block adds `model`, `electrode_model`, `axes`, and
`applied_voltage`. Each law declares `type`, `contact_conductance`, and an
explicit `cutoff`; tunneling also requires `decay_length`. Sampling uses
`simulation.trials`, `simulation.seed`, and `simulation.neighbor_backend`.
The CLI requires `face_to_face` mode and `wrapped_parent: false`. It rejects
unknown and duplicate keys. `contact.threshold` remains a v1 percolation
setting; it does not override the electrical model's cutoff. Neither
`percolation.axis` nor `simulation.confidence` selects or certifies transport
results: use `conductivity.axes`, and interpret the reported standard error as
described above.

The YAML schema remains version 1 with an additive optional block. Existing
v1 configurations, result types, and commands remain supported. v1 binaries
will reject the new block as an unknown key. Material `cost_per_volume` remains
an economic parameter and is never interpreted as electrical conductance.

## Verification

Tests cover series, parallel, balanced bridge, dangling, isolated, and extreme
scale circuits; a dense incidence-matrix oracle; finite-electrode sphere and
cylinder cases; analytic tunneling; all eight periodic flag combinations;
multiple image channels; parent continuity; seed and axis-order reproducibility;
strict configuration; and CLI JSON. The built-in `microperco validate` includes
series-circuit, finite-electrode conductivity, and exponential-law checks.
