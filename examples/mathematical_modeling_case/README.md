# Mathematical modeling case-study scaffold

This directory shows how a historical Q1–Q4 conductive-microstructure problem maps onto the generic MicroPerco API:

- Q1: load authorized particle coordinates, construct `Cylinder`/`Sphere` objects, and call `analyze_percolation`.
- Q2: use `estimate_percolation_probability` at each declared loading.
- Q3: use nested search and independent certification through `estimate_critical_loading`.
- Q4: declare material costs and finite bounds, then call `optimize_mixture`.

The original PDF and spreadsheet are intentionally absent because redistribution rights could not be confirmed. Place an authorized local dataset outside the repository and write an adapter; do not add private or competition-restricted data to commits.

`case_config.yaml` preserves the physical scale only as an example configuration. It does not claim to reproduce historical answers without the original preprocessing assumptions and data.

## Conductivity extension in v2.0

The Q1–Q4 scaffold above still evaluates binary connectivity and percolation probability. Once an authorized dataset has been converted into particles, use `analyze_conductivity` or `analyze_directional_conductivity` to evaluate electrical transport with explicitly calibrated junction and electrode parameters. For synthetic conductivity sampling, use `estimate_conductivity` or `microperco conductivity configs/conductivity.yaml` from the repository root.

See [the runnable conductivity example](../conductivity.py), [the separate conductivity configuration](../../configs/conductivity.yaml), and [model/units documentation](../../docs/methodology/conductivity.md). The existing case configuration does not contain a conductivity section; `optimize_mixture` continues to certify percolation probability, not electrical performance.
