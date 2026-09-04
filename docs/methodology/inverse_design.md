# Inverse design

MicroPerco searches finite integer bounds for one or more particle populations. Candidate cost is

$$
C(n)=\sum_k n_k V_k c_k,
$$

where $n_k$ is count, $V_k$ is single-particle nominal volume, and $c_k$ is material cost per volume. This is a relative design objective unless the supplied costs carry calibrated physical units.

## Auditable enumeration

The Cartesian product of inclusive count bounds is generated before sampling and sorted by `(total_cost, counts)`. `max_candidates` guards accidental combinatorial explosion. Screening uses modest Monte Carlo samples only to prioritize work; it never supplies formal feasibility evidence.

Candidate cardinality is computed with arbitrary-precision integers before any range
is materialized, so even bounds larger than the platform's native range length fail
through the documented `OptimizationError` safety path.

## Family-wise certification

Let $M$ be the full bounded candidate count. Before sampling, the algorithm fixes a family of `2M` potential one-sided assertions: a lower feasibility bound and an upper infeasibility bound for every candidate. With requested family confidence $1-\alpha$, each exact bound uses confidence

$$
1-\frac{\alpha}{2M}.
$$

Cost-ordered certification can stop above an already certified cost, but the error allocation remains tied to the original full family. A result is `CERTIFIED_OPTIMAL` only when a candidate's lower bound reaches the target and every evaluated strictly cheaper candidate has an upper bound below it. Otherwise it is explicitly inconclusive or has no certified feasible design.

The method is complete only inside the supplied finite bounds and under the configured stochastic model. It does not assert global optimality over unbounded compositions or unmodeled material physics.
