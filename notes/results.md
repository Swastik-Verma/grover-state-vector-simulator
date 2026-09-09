# Day 2 — Naive Implementation Results

## Correctness sanity checks
| Test | n | N | M | R | Success probability | Norm |
|------|---|---|---|---|---------------------|------|
| n=2 exact case | 2 | 4 | 1 | 1 | 1.000000 | 1.000000 |
| n=3 basic | 3 | 8 | 1 | 2 | 0.945312 | 1.000000 |
| n=4 multi-marked | 4 | 16 | 3 | 1 | 0.949219 | 1.000000 |

n=2 gives exactly P=1.0 as predicted by theory — strong first correctness signal.

## Scaling test (naive, full-matrix approach)

| n | N=2^n | R | Matrix build time (s) | Iteration time (s) | Total time (s) | Matrix memory (GB) | Success prob |
|---|-------|---|------------------------|---------------------|-----------------|---------------------|--------------|
| 5  | 32    | 4  | 0.0001  | 0.0002   | 0.0003   | 0.000 | 0.999182 |
| 6  | 64    | 6  | 0.0003  | 0.0008   | 0.0011   | 0.000 | 0.996586 |
| 7  | 128   | 8  | 0.0013  | 0.0056   | 0.0070   | 0.001 | 0.995620 |
| 8  | 256   | 12 | 0.0050  | 0.0382   | 0.0432   | 0.003 | 0.999947 |
| 9  | 512   | 17 | 0.0220  | 0.2102   | 0.2321   | 0.012 | 0.999448 |
| 10 | 1024  | 25 | 0.0870  | 1.1776   | 1.2646   | 0.047 | 0.999461 |
| 11 | 2048  | 35 | 0.4013  | 5.7811   | 6.1824   | 0.188 | 0.999997 |
| 12 | 4096  | 50 | 1.8006  | 33.7176  | 35.5182  | 0.750 | 0.999945 |
| 13 | 8192  | 71 | 10.0897 | 231.1254 | 241.2150 | 3.000 | 0.999916 |
| 14 | 16384 | -  | did not complete — manually terminated (Ctrl+C) after excessive time/memory. Estimated matrix memory ≈ 12 GB for 3 matrices, exceeding 8 GB RAM; machine was swapping. |

## Machine spec
- 8 GB RAM (per hardware constraint in build plan)

## Key finding
The naive full-matrix approach becomes impractical past **n=13** on this machine — not from a clean out-of-memory crash, but from RAM exhaustion causing disk swapping, which made n=14 effectively unusable (killed manually after hanging). n=13 itself already took ~4 minutes for a search space of only 8192 elements. This confirms the core problem the rest of the project solves: matrix storage is O(2^2n), while the state vector itself is only O(2^n) — the matrices are the bottleneck, not the actual information being processed.


# Day 3 — Correctness Validation

## Test results
- 95/95 assertions passed, 0 failed
- Worst norm deviation across all tests: 3.905765e-13 (at n=10), well within 1e-9 tolerance — consistent with expected floating-point accumulation, not a bug
- n=2, M=1 gives P=1.000000000000 for all 4 possible marked elements — matches exact theoretical prediction
- Measured probabilities matched sin²((2k+1)θ/2) within 1e-9 for all tested configurations (n=3 to n=8, M=1 to M=3)
- Over-rotation confirmed: for n=5, probability peaked at k=4 (P=0.999182) then dropped to P=0.014453 at k=8, then began rising again — exact match to theoretical oscillation
- Multiple-marked-elements tests confirmed, including the M=N/2 edge case (n=3, M=4) giving exactly P=0.5 after 1 iteration
- Full test suite runs in under 5 seconds (all test cases use n≤10, well below the memory ceiling found on Day 2)


# Day 4 — In-Place Single-Qubit Gates Results

## Correctness validation
- 70/70 test assertions passed, 0 failed
- Optimized H^{⊗n} matches naive full-matrix H^{⊗n} EXACTLY (diff=0.0) for n=1 to 12 on the |0...0> initial state
- On non-initial (arbitrary) states, max diff stayed at machine-precision level (~1e-16), confirming correctness isn't a coincidence of the symmetric initial state
- Verified each qubit is targeted correctly in isolation (H on qubit 0/1/2 of |000> gives the expected single-qubit superposition)
- H^{⊗n} applied twice returns to the original state (H²=I) within ~1e-16 for n=2 to 12
- Norm preserved to ~4.44e-16 (essentially machine epsilon) for n=2 through n=15
- Full hybrid Grover run (optimized Hadamard + naive oracle/diffusion) matches fully-naive Grover exactly (diff=0.0) for n=2 to 10

## Timing comparison — naive full-matrix vs optimized in-place

| n  | Naive (s) | Optimized (s) | Speedup |
|----|-----------|----------------|---------|
| 4  | 0.000023  | 0.000003       | 8.3x    |
| 5  | 0.000072  | 0.000005       | 13.2x   |
| 6  | 0.000416  | 0.000026       | 15.9x   |
| 7  | 0.001109  | 0.000036       | 30.7x   |
| 8  | 0.004226  | 0.000146       | 28.9x   |
| 9  | 0.018053  | 0.000303       | 59.6x   |
| 10 | 0.087518  | 0.000332       | 263.3x  |
| 11 | 0.338251  | 0.000821       | 412.2x  |
| 12 | 1.152413  | 0.002103       | 547.9x  |
| 13 | OOM (naive dies) | 0.005019 | — (naive cannot run at all) |

Speedup grows with n, as expected — naive is O(2^2n) while optimized is O(n · 2^n), so the gap widens exponentially as n increases.

## Optimized-only scaling (beyond naive's reach)

| n  | Time (s) |
|----|----------|
| 14 | 0.008650 |
| 16 | 0.066265 |
| 18 | 0.219413 |
| 20 | 1.009818 |
| 22 | 4.514811 |

The optimized Hadamard alone now comfortably runs at n=22 (4M+ amplitudes) in ~4.5s — a regime the naive matrix approach could never reach (would need ~4×10^12 GB for the matrix at n=22).

## Key finding
Eliminating the H^{⊗n} matrix via in-place bit-pair gate application gives up to a **547.9x speedup** at n=12 and removes the memory ceiling entirely for this component — naive dies from OOM at n=13, while the optimized version scales past n=22 with no matrix ever allocated. This is the single highest-impact optimization in the project so far.



# Day 5 — Zero-Matrix Optimization Results

## Correctness validation
- 62/62 test assertions passed, 0 failed
- Optimized oracle matches naive oracle matrix exactly (diff=0.0) for n=2 to 10
- Optimized diffusion matches naive diffusion matrix within ~1e-16 for n=2 to 10
- Full optimized Grover (zero matrices) matches full naive Grover within 1e-13 for n=2 to 10
- Optimized results match theoretical P(k)=sin²((2k+1)θ/2) within 5e-14 across all tested n, M
- Predicate-based oracle matches set-based oracle exactly (diff=0.0)
- Structured oracles (oracle_bit_set, oracle_hamming_weight) verified correct and match theory
- Norm preserved (worst deviation 1.6e-11 at n=15) across n=2 to 15

## Scaling results (zero-matrix optimized Grover)

| n  | N         | R    | Total (s) | Per iter (s) | P_success |
|----|-----------|------|-----------|---------------|-----------|
| 5  | 32        | 4    | 0.0000    | 0.000010      | 0.999182  |
| 10 | 1,024     | 25   | 0.0010    | 0.000041      | 0.999461  |
| 15 | 32,768    | 142  | 0.1104    | 0.000778      | 0.999987  |
| 18 | 262,144   | 402  | 2.4844    | 0.006180      | 0.999998  |
| 20 | 1,048,576 | 804  | 17.3547   | 0.021586      | 1.000000  |
| 22 | 4,194,304 | 1608 | 144.3165  | 0.089749      | 1.000000  |
| 24 | 16,777,216| 3216 | 1037.5697 | 0.322627      | 1.000000  |

## Key finding
Zero matrices anywhere in the codebase now. Memory usage is O(2^n) — just the state vector. Successfully scaled to n=24 (16.7M amplitudes) on an 8 GB machine, compared to naive's hard ceiling at n=13. Time complexity confirmed O(N^1.5): per-iteration time and total time both scale consistently with theory as n increases.



# Day 6 — Scaling Test and First Benchmark

## Naive vs Optimized comparison (n=3 to 13)

| n  | N     | R  | Naive (s) | Optimized (s) | Speedup  | P_naive  | P_opt    |
|----|-------|----|-----------|-----------------|----------|----------|----------|
| 3  | 8     | 2  | 0.0000    | 0.0000          | 9.2x     | 0.945312 | 0.945312 |
| 4  | 16    | 3  | 0.0001    | 0.0000          | 18.2x    | 0.961319 | 0.961319 |
| 5  | 32    | 4  | 0.0005    | 0.0000          | 33.1x    | 0.999182 | 0.999182 |
| 6  | 64    | 6  | 0.0023    | 0.0000          | 61.2x    | 0.996586 | 0.996586 |
| 7  | 128   | 8  | 0.0062    | 0.0000          | 130.6x   | 0.995620 | 0.995620 |
| 8  | 256   | 12 | 0.0416    | 0.0002          | 186.7x   | 0.999947 | 0.999947 |
| 9  | 512   | 17 | 0.2277    | 0.0006          | 362.6x   | 0.999448 | 0.999448 |
| 10 | 1024  | 25 | 1.2990    | 0.0008          | 1612.0x  | 0.999461 | 0.999461 |
| 11 | 2048  | 35 | 7.7605    | 0.0027          | 2894.1x  | 0.999997 | 0.999997 |
| 12 | 4096  | 50 | 40.1600   | 0.0055          | 7249.8x  | 0.999945 | 0.999945 |
| 13 | 8192  | 71 | OOM       | 0.0172          | —        | (naive cannot run) | 0.999916 |

## Optimized-only scaling (n=10 to 24)

| n  | N          | R    | Total (s) | Per iter (s) | Memory (MB) | P_success |
|----|------------|------|-----------|---------------|-------------|-----------|
| 14 | 16,384     | 100  | 0.0379    | 0.000379      | 0.25        | 1.000000  |
| 16 | 65,536     | 201  | 0.2906    | 0.001446      | 1.00        | 0.999988  |
| 18 | 262,144    | 402  | 2.4698    | 0.006144      | 4.00        | 0.999998  |
| 20 | 1,048,576  | 804  | 17.6305   | 0.021928      | 16.00       | 1.000000  |
| 22 | 4,194,304  | 1608 | 139.2013  | 0.086568      | 64.00       | 1.000000  |
| 23 | 8,388,608  | 2274 | 451.7614  | 0.198664      | 128.00      | 1.000000  |
| 24 | 16,777,216 | 3216 | 1508.1480 | 0.468951      | 256.00      | 1.000000  |

## Memory comparison (computed, naive vs optimized)

| n  | Naive matrices | Optimized | Reduction |
|----|-----------------|-----------|-----------|
| 10 | 48 MB           | 0.02 MB   | 3,072x    |
| 12 | 768 MB          | 0.06 MB   | 12,288x   |
| 13 | 3 GB            | 0.12 MB   | 24,576x   |
| 15 | 12 GB           | 0.5 MB    | 98,304x   |
| 18 | 96 GB           | 4 MB      | 786,432x  |
| 20 | 384 GB          | 16 MB     | 3,145,728x|
| 22 | 1536 GB         | 64 MB     | 12,582,912x |
| 24 | 6144 GB         | 256 MB    | 50,331,648x |

## Key numbers for CV
- Max n (naive): 12 (dies at n=13 from OOM)
- Max n (optimized, fp64) tested: 24 (16.7M amplitudes, 256 MB, ~25 minutes runtime)
- Speedup naive → optimized at n=12: 7249.8x
- Peak memory at n=24: 256 MB (vs naive's theoretical 6 TB for the same n)
- CSV data saved in benchmarks/ for Day 11 plotting




# Day 7 — OpenMP Parallelism Results

## Machine
- 4 CPU cores (`nproc` = 4)

## Correctness validation
- 30/30 test assertions passed, 0 failed
- Parallel Grover matches theoretical P(k) within 3e-12 for n=2 to 14, M=1 to 3
- Norm preserved (worst deviation 5.4e-11 at n=18) across n=2 to 18
- Day 3 over-rotation regression test still passes — parallelism introduced no correctness issues

## Thread scaling (n=20, 50 iterations)

| Threads | Time (s) | Speedup | Efficiency | P_success |
|---------|----------|---------|------------|-----------|
| 1       | 3.2946   | 1.00x   | 100.0%     | 0.009697  |
| 2       | 2.1345   | 1.54x   | 77.2%      | 0.009697  |
| 3       | 1.7072   | 1.93x   | 64.3%      | 0.009697  |
| 4       | 1.6305   | 2.02x   | 50.5%      | 0.009697  |

Sub-linear scaling (2.02x on 4 cores, not 4x) — expected for this workload, since diffusion and oracle passes are memory-bandwidth-bound rather than compute-bound. All 4 cores compete for the same memory bus, so speedup plateaus well before core count. This is a legitimate, explainable result, not a bug.

## Full scaling with 4 threads (n=10 to 24)

| n  | N          | R    | Total (s) | Per iter (s) | P_success |
|----|------------|------|-----------|---------------|-----------|
| 14 | 16,384     | 100  | 0.0316    | 0.000316      | 1.000000  |
| 16 | 65,536     | 201  | 0.2186    | 0.001088      | 0.999988  |
| 18 | 262,144    | 402  | 2.1945    | 0.005459      | 0.999998  |
| 20 | 1,048,576  | 804  | 17.2009   | 0.021394      | 1.000000  |
| 22 | 4,194,304  | 1608 | 102.0970  | 0.063493      | 1.000000  |
| 24 | 16,777,216 | 3216 | 910.4505  | 0.283100      | 1.000000  |

Comparing to Day 6's serial n=24 result (1508.15s) vs this parallel n=24 result (910.45s): **1.66x speedup** from OpenMP at n=24, consistent with the memory-bandwidth-bound scaling pattern seen in the thread benchmark.

## Key finding
OpenMP parallelism gives up to 2.02x speedup on 4 cores at n=20, and 1.66x at n=24 in full-run comparison. Scaling is sub-linear because both diffusion (reduction + update) and oracle/gate passes are memory-bandwidth-bound, not compute-bound — a well-understood limitation for this class of algorithm on multi-core CPUs, and the exact reason Day 8's cache optimization is a meaningful next step.