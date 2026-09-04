# Mathematical modeling case-study scaffold

This directory shows how a historical Q1–Q4 conductive-microstructure problem maps onto the generic MicroPerco API:

- Q1: load authorized particle coordinates, construct `Cylinder`/`Sphere` objects, and call `analyze_percolation`.
- Q2: use `estimate_percolation_probability` at each declared loading.
- Q3: use nested search and independent certification through `estimate_critical_loading`.
- Q4: declare material costs and finite bounds, then call `optimize_mixture`.

The original PDF and spreadsheet are intentionally absent because redistribution rights could not be confirmed. Place an authorized local dataset outside the repository and write an adapter; do not add private or competition-restricted data to commits.

`case_config.yaml` preserves the physical scale only as an example configuration. It does not claim to reproduce historical answers without the original preprocessing assumptions and data.
