# MicroPerco 1.0 benchmark report

Benchmark date: 2026-09-04. Raw results are in `benchmark_results.json`; `run_benchmark.py` reproduces the workload.

## Environment

- CPU: Intel Core i7-14650HX, 16 physical cores / 24 logical CPUs, up to 5.2 GHz
- OS: Linux 6.8.0-138-generic, x86-64
- Python: 3.11.15
- NumPy: 2.4.6
- Execution: one Python process; no explicit parallel workers

Wall-clock measurements can vary with frequency scaling, thermal state, background work, allocator behavior, and dependency versions.

## Workload and protocol

- Periodic box: `50 × 50 × 50`
- Particles: equal spheres, radius 0.5
- Contact threshold: 0.2
- Counts: 10, 50, 100, 500, 1000
- Seed: 42
- Identical realization passed to both backends at each count
- One untimed warmup and five timed repeats per backend
- Statistic: median; first and third quartiles retained separately

The brute-force backend is the correctness oracle and evaluates every base pair plus any feasible periodic images. The cell-list backend hashes padded periodic AABBs and performs exact distance queries only on retained candidates.

## Results

| N | Backend | Q1 (s) | Median (s) | Q3 (s) | Candidates | Exact distances | Relative speedup |
|---:|---|---:|---:|---:|---:|---:|---:|
| 10 | brute force | 0.001845 | 0.001852 | 0.001878 | 45 | 45 | 1.00× |
| 10 | cell list | 0.000378 | 0.000378 | 0.000379 | 0 | 0 | 4.89× |
| 50 | brute force | 0.045983 | 0.046010 | 0.046329 | 1,225 | 1,225 | 1.00× |
| 50 | cell list | 0.001683 | 0.001703 | 0.001707 | 0 | 0 | 27.02× |
| 100 | brute force | 0.193733 | 0.194516 | 0.195424 | 4,950 | 4,950 | 1.00× |
| 100 | cell list | 0.003734 | 0.003737 | 0.003741 | 1 | 1 | 52.05× |
| 500 | brute force | 4.854974 | 4.905378 | 4.924999 | 124,750 | 124,750 | 1.00× |
| 500 | cell list | 0.019504 | 0.019531 | 0.019647 | 16 | 16 | 251.15× |
| 1000 | brute force | 19.718427 | 19.763878 | 19.809286 | 499,500 | 499,500 | 1.00× |
| 1000 | cell list | 0.040826 | 0.040883 | 0.041035 | 52 | 52 | 483.43× |

At N=1000, candidate work fell from 499,500 to 52 tuples (99.9896% reduction) for this sparse realization. The median runtime fell from 19.7639 s to 40.88 ms.

## Interpretation and limitations

This benchmark demonstrates the value of broad-phase pruning in a sparse, monodisperse sphere system. It does not establish an asymptotic bound or a universal speedup. Dense systems, large thresholds, long cylinders, small periodic boxes, and strongly overlapping AABBs produce more candidates. Cylinder narrow-phase queries are also more expensive than sphere queries.

Correctness is guarded separately: the validation suite compares complete accepted edge signatures between backends across mixed particles and all PBC combinations. Performance changes must keep that oracle parity.

The benchmark plot uses the actual asymmetric distances from median to Q1 and Q3; it does not replace them with a symmetric half-IQR.
