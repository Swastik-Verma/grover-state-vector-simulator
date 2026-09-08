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