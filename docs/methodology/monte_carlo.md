# Monte Carlo simulation

For each trial, particle centers are independently uniform within the domain bounds. Cylinder directions are normalized three-component Gaussian draws, yielding the rotation-invariant distribution on the unit sphere. The generator permits overlap and lets finite bodies extend beyond a non-periodic boundary when their centers are inside; users needing hard-particle packing should provide their own realizations.

## Reproducibility

An integer or integer sequence initializes NumPy `SeedSequence`. Every trial receives its own spawned child sequence. The same configuration, dependency behavior, and seed reproduce the same realization stream and result; child streams avoid accidental reuse between trials. Critical search and certification, and optimization screening and certification, are split at the root seed.

When callers pass an existing `SeedSequence`, result provenance records its entropy,
spawn key, and pool size. The input object is cloned before children are spawned, so
reusing it neither advances caller-owned state nor hides distinct child streams behind
the same entropy label.

## Estimates

With $k$ percolating trials from $n$ independent trials, $\hat p=k/n$. Results include the two-sided Wilson interval

$$
\frac{\hat p+z^2/(2n)\pm z\sqrt{\hat p(1-\hat p)/n+z^2/(4n^2)}}{1+z^2/n}
$$

and the exact Clopper–Pearson interval. Exact one-sided Clopper–Pearson bounds are used for certification assertions.

Monte Carlo error quantifies sampling uncertainty under the configured stochastic model; it does not cover model misspecification, geometry-unit errors, or dependency defects.
