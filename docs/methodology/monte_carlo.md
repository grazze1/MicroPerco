# Monte Carlo simulation: probability and conductivity

For each trial, particle centers are independently uniform within the domain bounds. Cylinder directions are normalized three-component Gaussian draws, yielding the rotation-invariant distribution on the unit sphere. The generator permits overlap and lets finite bodies extend beyond a non-periodic boundary when their centers are inside; users needing hard-particle packing should provide their own realizations.

## Reproducibility

An integer or integer sequence initializes NumPy `SeedSequence`. Every trial receives its own spawned child sequence. The same configuration, dependency behavior, and seed reproduce the same realization stream and result; child streams avoid accidental reuse between trials. Critical search and certification, and optimization screening and certification, are split at the root seed.

When callers pass an existing `SeedSequence`, result provenance records its entropy,
spawn key, and pool size. The input object is cloned before children are spawned, so
reusing it neither advances caller-owned state nor hides distinct child streams behind
the same entropy label.

## Percolation probability estimates

With $k$ percolating trials from $n$ independent trials, $\hat p=k/n$. Results include the two-sided Wilson interval

$$
\frac{\hat p+z^2/(2n)\pm z\sqrt{\hat p(1-\hat p)/n+z^2/(4n^2)}}{1+z^2/n}
$$

and the exact Clopper–Pearson interval. Exact one-sided Clopper–Pearson bounds are used for certification assertions.

Monte Carlo error quantifies sampling uncertainty under the configured stochastic model; it does not cover model misspecification, geometry-unit errors, or dependency defects.

## Conductivity estimates (v2.0)

`estimate_conductivity` uses one child seed per trial and shares that trial's particles across the requested measurement axes. It retains per-axis conductivity samples, their arithmetic mean, conducting-trial count, and sample standard deviation divided by `sqrt(trials)`. The standard error is `None` (`null` in JSON) for one trial. These are descriptive statistics, not Wilson/Clopper–Pearson intervals or electrical-performance certification.

Conductivity sampling also records generated seed entropy when the caller passes `seed=None`, permitting replay. See [transport sampling and model scope](conductivity.md#conductivity-and-sampling) and [the CLI configuration](../../configs/conductivity.yaml).
