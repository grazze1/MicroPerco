# Critical loading

The critical count is the smallest count on a declared increasing grid whose percolation probability reaches a target. A naive noisy binary search can violate monotonicity and repeatedly select a favorable random fluctuation, so MicroPerco uses a two-phase procedure.

## Search

Each search trial generates the maximum variable population once. Smaller grid counts use nested prefixes, so every realization is sample-wise monotone whenever adding particles cannot destroy connectivity. Fixed populations share that realization. Raw proportions are fitted with weighted PAVA by default; monotone logistic and probit links are optional descriptive alternatives.

Implicit or explicit loading grids are limited to 100,000 points and rejected before
allocation when they exceed that resource-safety bound. Logistic/probit coefficients
are stored in their fitted standardized coordinate together with its center and scale,
so predictions and serialized provenance remain finite and reconstructable across
extreme but representable loading coordinates.

## Independent certification

Certification uses new random streams. If search proposes an interior candidate, the final family contains two one-sided assertions:

1. the candidate's lower confidence bound is at least the target;
2. its predecessor's upper confidence bound is below the target.

At the first grid point or when no crossing is proposed, only one assertion is required. For family confidence $1-\alpha$ and $m\in\{1,2\}$ assertions, each exact bound uses confidence $1-\alpha/m$. The result records both levels and the actual comparison count.

`CERTIFIED` means both required inequalities hold. `NO_CROSSING` means the largest grid point is independently excluded. Otherwise the result is `INCONCLUSIVE`; the API does not relabel an uncertain estimate as certified.
