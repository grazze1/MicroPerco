# Figure contract and QA record

This contract records the existing figures created for the v1 release and retained in the v2.0 documentation. They illustrate geometry, binary connectivity, percolation probability, and the historical contact-search benchmark. They are not electrical-current or conductivity plots. Current transport evidence is in [transport validation](../validation/TRANSPORT_VALIDATION.md).

Backend: Python/Matplotlib, persisted as the selected `nature-figure` backend. All previews and exports use that backend. Target: GitHub documentation with Nature-family publication discipline, not a journal submission. Text remains editable in SVG/PDF; PNG is the web preview.

## Figure sequence

### Microstructure overview

- Core conclusion: MicroPerco represents mixed finite particles, a bounded 3D domain, and distinct electrode faces in one reusable model.
- Results-level question: What physical objects and boundaries does the simulator operate on?
- Archetype: representative image / setup.
- Final size: 170 × 112 mm.
- Evidence: one seeded realization with all 50 generated particles; no rows excluded.
- Statistics: none; this is a representative geometry view.
- Integrity note: centers and cylinder axes come directly from seed 20260904. Screen-space cylinder widths and sphere marker areas aid visibility and are not quantitative radii.
- Reviewer risk: perspective and glyph scaling can exaggerate depth or radius; axes and domain edges provide context, and the caption does not claim a packing fraction from the rendering.

### Spanning cluster

- Core conclusion: a finite-face percolation decision exposes a traceable representative path rather than only a Boolean.
- Results-level question: Which particles carry the demonstrated electrode-to-electrode connection?
- Archetype: representative image / primary qualitative evidence.
- Final size: 170 × 112 mm.
- Evidence: an exact five-sphere chain plus 15 seeded small background spheres; red particles are exactly `PercolationResult.spanning_path`.
- Statistics: none; deterministic constructed regression case.
- Integrity note: no particles are hidden. The background is deliberately non-decisive and retained in `figure_data.json`.
- Reviewer risk: one representative path is not the union of every particle in a spanning component; wording consistently says “representative path.”

### Percolation curve

- Core conclusion: physical Monte Carlo simulations recover an increasing transition with uncertainty visible at every sampled loading.
- Results-level question: Where does the target probability first appear to be crossed on the declared count grid?
- Archetype: quantitative single panel.
- Final size: 89 × 66 mm.
- Evidence: nested, seed-314159 search trials from the real geometry/connectivity engine for counts 40–100; all grid points shown.
- Statistics: n = 100 nested Bernoulli trials per count; points are proportions, bars are two-sided 95% Wilson intervals, line is PAVA, horizontal target is 0.8. Independent certification records are retained but not drawn because the panel visualizes search evidence.
- Source data: `docs/assets/figure_data.json`.
- Reviewer risk: nested counts are correlated across x positions and should not be interpreted as independent groups; the caption states the sampling design.

### Contact-search benchmark

- Core conclusion: conservative cell-list pruning sharply reduces measured work and runtime in the recorded sparse system while retaining an exact narrow phase.
- Results-level question: Is the observed runtime gain explained by fewer candidate and exact-distance queries across scale?
- Archetype: quantitative-grid validation envelope.
- Final size: 183 × 61 mm.
- Panel map:
  - a — primary comparison: median wall-clock runtime with actual Q1/Q3 asymmetric error bars.
  - b — mechanism: number of broad-phase candidate tuples.
  - c — bounded computational consequence: number of exact distance evaluations.
- Evidence hierarchy: panel a is the outcome; b and c explain where it comes from. Correctness parity is orthogonal evidence in the validation report, not redundantly plotted.
- Statistics: n = 5 timed repeats per backend/size after one warmup; center is median; interval spans first to third quartile. The realization is fixed by seed 42.
- Source data: `benchmarks/benchmark_results.json`, mirrored in `figure_data.json`.
- Reviewer risk: this sparse sphere workload does not imply a universal speedup or asymptotic complexity; the report names dense and long-cylinder failure regimes.

## Export and audit contract

- White background; sans-serif 7 pt body-text target, 5 pt hard minimum, and 8 pt bold panel labels.
- Restrained neutral/blue/signal-red palette; method identity is consistent across panels.
- Primary editable SVG plus PDF and 600 dpi PNG for each figure.
- Multi-panel benchmark alignment must pass at 1.5 pt tolerance. Single panels record “not applicable.”
- Every final PDF is audited for rendered glyph size and text/stroke collisions, followed by panel-by-panel PNG inspection at final size.
- No external images, private data, local absolute paths, selective image processing, or AI-generated visual content.

## Panel-by-panel final audit

This table is completed after rendering and recorded alongside the generated QA JSON.

| Figure/panel | Unique claim | Center/spread | Replicate unit | Alignment | Collision review | Status |
|---|---|---|---|---|---|---|
| Microstructure | Model vocabulary | none | one seeded realization | N/A | 0 fail, 1 reviewed warn | PASS |
| Spanning path | Traceable route | none | deterministic system | N/A | 0 fail, 1 reviewed warn | PASS |
| Probability | Loading transition | proportion / 95% Wilson CI | 100 nested trials | N/A | 0 fail, 0 warn | PASS |
| Benchmark a | Runtime gain | median / Q1–Q3 | 5 timed repeats | row a–c PASS | 0 fail, 0 warn | PASS |
| Benchmark b | Candidate pruning | exact count | one fixed realization per N | row a–c PASS | 0 fail, 0 warn | PASS |
| Benchmark c | Narrow-phase reduction | exact count | one fixed realization per N | row a–c PASS | 0 fail, 0 warn | PASS |

Rendered PDF text audits found minima of 7.0 pt for both 3D figures and the probability curve, and 5.6 pt for the benchmark. All exceed the 5 pt contract. The collision audit reported one fill-edge warning in each 3D figure where an axis label approaches a translucent electrode-face polygon; final-size PNG inspection confirmed that both labels remain unobstructed and inside the canvas, so the warnings were accepted. The probability and benchmark figures passed without findings. Electrode faces, the highlighted path, uncertainty bars, benchmark identities, zero-count points, and panel labels remain legible at the declared dimensions. Source preflight recorded 19 passes, two reviewed warnings (PNG is intentionally the web raster rather than TIFF; seeded simulated demonstrations are explicitly isolated), and no failures.
