# Grover State-Vector Simulator

A high-performance C++ simulator for Grover's quantum search algorithm, built from scratch with a focus on progressive optimization and rigorous validation.

Starting from a deliberately naive O(2²ⁿ) matrix-based implementation, each component was systematically optimized — eliminating all matrices, parallelizing with OpenMP, and tuning memory layout — with every change validated against exact theoretical results and an independent external simulator (Qiskit). The commit history traces each optimization as a separate, measured step.

## Results at a Glance

| Metric | Value |
|--------|-------|
| Max n (naive baseline) | 12 (dies at n=13, ~3 GB matrix memory) |
| Max n (optimized, fp64) | 24 tested, 27 allocatable (8 GB machine) |
| Max n (optimized, fp32) | 27 allocatable (half the memory of fp64) |
| Speedup: naive → optimized (n=12) | **7,250x** |
| OpenMP scaling (4 cores, n=20) | 2.02x |
| Qiskit cross-validation | Passed for n ≤ 12 |
| Correctness tests | 95+ assertions against exact theory |

## How Grover's Algorithm Works (Linear Algebra, No Physics)

The state of n qubits is a unit vector in ℂ^(2ⁿ). The 2ⁿ computational basis states |0⟩, |1⟩, ..., |2ⁿ−1⟩ are the standard basis vectors e₀, e₁, ..., e_{N−1}.

Grover's algorithm searches an unstructured space of N = 2ⁿ elements for M marked elements using only O(√(N/M)) oracle queries — a quadratic speedup over classical search.

**The algorithm:**

1. Start with |0⟩⊗ⁿ and apply Hadamard to all qubits, producing the uniform superposition |ψ⟩ = (1/√N) Σ|x⟩.

2. Repeat R ≈ (π/4)√(N/M) times: apply the oracle Uᶠ (flips the sign of marked elements' amplitudes), then apply the diffusion operator D = 2|ψ⟩⟨ψ| − I (reflects all amplitudes around their mean).

3. Measure. The marked elements have high probability.

**Why it works:** The oracle makes marked amplitudes negative (below the mean). The diffusion operator reflects every amplitude around the mean, pushing the formerly-negative marked amplitudes to *above* average. Each iteration rotates the state vector by angle θ toward the marked subspace. After R iterations, the state nearly aligns with the marked elements, giving success probability sin²((2R+1)θ/2) ≈ 1.

## The Optimization Story

### Phase 1: Naive Baseline (Days 1–3)

The naive implementation stores three full 2ⁿ × 2ⁿ matrices (Hadamard, oracle, diffusion) and does explicit matrix-vector multiplication. This requires O(2²ⁿ) memory and O(√N · N²) time.

At n=12, the three matrices consume 768 MB. At n=13, they need ~3 GB and the program takes over 4 minutes. At n=14 it exceeds 8 GB RAM and becomes unusable. This is the deliberate baseline — it establishes the problem the optimizations solve.

### Phase 2: Matrix Elimination (Days 4–5)

**In-place single-qubit gates (Day 4):** Instead of building the full H⊗ⁿ matrix via Kronecker products, apply the 2×2 Hadamard to each qubit individually by iterating over index pairs that differ in one bit. This reduces H⊗ⁿ from O(2²ⁿ) memory to zero extra memory, and from O(2²ⁿ) time to O(n·2ⁿ). Speedup at n=12: **548x** (rising to **7,250x** in the full pipeline).

**O(N) oracle and diffusion (Day 5):** The oracle becomes a direct sign-flip on marked indices (O(M) or O(N) with a predicate). The diffusion becomes "compute mean, then reflect each amplitude around it" — two O(N) passes. After this, zero matrices exist anywhere in the codebase. Memory usage drops to exactly 16 bytes × 2ⁿ (the state vector alone).

### Phase 3: Performance Engineering (Days 7–9)

**OpenMP parallelism (Day 7):** All gate loops, oracle, and diffusion parallelized. The diffusion's sum reduction uses split real/imaginary accumulators (OpenMP doesn't natively reduce std::complex). Achieved 2.02x speedup on 4 cores — sub-linear because the workload is memory-bandwidth-bound, not compute-bound.

**Cache-aware blocking (Day 8):** Implemented blocked iteration for high-index qubits where the stride exceeds cache line size. Measured modest gains (up to 1.53x on individual high-stride qubits) — the memory-bandwidth bottleneck limits the benefit on this hardware. Documented as a legitimate neutral result.

**fp32 precision (Day 9):** Made the amplitude type configurable. fp32 halves memory (confirmed at every n), enabling +1 qubit on the same hardware. Accuracy cost: ~1e-6 error vs theory (compared to fp64's ~1e-9), well within acceptable bounds for success probability. No consistent speed advantage observed on this laptop.

### Phase 4: Validation and Documentation (Days 10–12)

**Qiskit cross-validation (Day 10):** Independently verified against IBM's Qiskit Aer statevector simulator for n=2 to 12. Both implementations converge to the same theoretical success probability to within floating-point tolerance.

## Correctness

This simulator is validated more thoroughly than most:

- **Norm preservation:** Verified after every gate application (unitary matrices preserve vector norms). Worst deviation: ~4e-16 for optimized gates, ~5e-11 for full runs at n=18.
- **Exact theoretical match:** Success probability at every iteration matches P(k) = sin²((2k+1)θ/2) to within 1e-9 across all tested configurations.
- **Over-rotation:** Running past optimal iterations produces the exact sine-squared oscillation pattern predicted by theory — probability peaks, drops, and rises again.
- **n=2 exact case:** Theory predicts P=1.0 exactly after 1 iteration. Simulator matches to 12 decimal places for all 4 possible marked elements.
- **Cross-validation:** Qiskit Aer independently confirms the same results for n ≤ 12.
- **95+ automated assertions** across 7 test categories, run after every optimization.

## Benchmark Plots

See `benchmarks/plots/` for the full set:

- **Runtime: naive vs optimized** — log-scale comparison showing the naive approach's exponential wall vs the optimized version scaling to n=24
- **OpenMP thread scaling** — measured vs ideal linear speedup on 4 cores
- **Memory usage** — naive's O(2²ⁿ) matrix storage vs optimized O(2ⁿ) state vector, with 8 GB RAM line
- **Measured vs theoretical probability** — simulator output overlaid on the exact sin²((2k+1)θ/2) curve, demonstrating over-rotation
- **Speedup growth** — optimization factor vs problem size (reaches 7,250x at n=12)
- **fp32 vs fp64 precision** — error growth comparison with machine epsilon reference lines

## Build

Requires: C++17 compiler (GCC 9+), CMake 3.16+, OpenMP.

```bash
mkdir build && cd build
cmake ..
make
./grover           # main program
./test_basic       # Day 1 basic tests
./test_correctness # Day 3 full correctness suite
```

## Project Structure

├── include/
│ ├── state_vector.h # Core state vector class + timer
│ ├── naive_grover.h # Naive full-matrix implementation (baseline)
│ ├── optimized_gates.h # In-place gates, O(N) oracle/diffusion, full runner
│ └── state_vector_float.h # fp32 variant for precision experiments
├── src/
│ └── main.cpp
├── tests/
│ ├── test_basic.cpp # Day 1 — data structure tests
│ ├── test_correctness.cpp # Day 3 — 7 categories, 95+ assertions
│ ├── test_optimized_gates.cpp# Day 4 — in-place gate validation
│ ├── test_day5_optimized.cpp # Day 5 — zero-matrix validation
│ ├── test_day7_openmp.cpp # Day 7 — parallel correctness + scaling
│ ├── test_day8_cache.cpp # Day 8 — cache blocking validation
│ └── test_day9_precision.cpp # Day 9 — fp32 vs fp64 comparison
├── benchmarks/
│ ├── benchmark.cpp # Day 6 — comprehensive benchmark suite
│ ├── cross_validate.py # Day 10 — Qiskit cross-validation
│ ├── generate_plots.py # Day 11 — plot generation from CSVs
│ ├── plots/ # Generated benchmark plots (6 PNGs)
│ └── *.csv # Raw benchmark data (11 CSVs)
├── notes/
│ └── results.md # Running log of all measured results
├── CMakeLists.txt
└── README.md



## Hardware

Developed and benchmarked on an 8 GB RAM, 4-core laptop running Ubuntu 24.04.

## Background

This project grew out of a research internship on quantum computation under Prof. Abhranil Chatterjee, studying Grover's algorithm through the lens of linear algebra — no quantum physics prerequisites, following the approach in Nielsen & Chuang's textbook. The simulator makes that mathematical framework concrete and measurable, and is designed so that future research (e.g., applying Grover's algorithm to the Longest Path Problem) can plug in new oracle predicates without modifying the simulation infrastructure.