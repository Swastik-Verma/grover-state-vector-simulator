# Grover State-Vector Simulator

A C++ state-vector simulator that executes Grover's quantum search algorithm, built from scratch with a focus on performance engineering.

## Project Status

Currently in active development (Phase 1 — Foundation).

- [x] Day 1: Project setup, CMake build system, StateVector class
- [x] Day 2: Naive full-matrix baseline implementation
- [x] Day 3: Correctness validation suite
- [x] Day 4: In-place single-qubit gates (eliminated H⊗ⁿ matrix)
- [ ] Days 4–5: Matrix-free optimizations (in-place gates, O(N) diffusion)
- [ ] Days 6–9: Performance engineering (OpenMP, cache optimization, precision tuning)
- [ ] Days 10–12: Cross-validation, benchmarks, documentation

## What This Project Does

Simulates Grover's search algorithm using classical state-vector simulation. The project starts with a deliberately naive O(2^2n) matrix-based approach, then progressively optimizes it to O(2^n) memory and O(√N · N) time, measuring the impact of each optimization along the way.

## Build

```bash
mkdir build && cd build
cmake ..
make
./grover
./test_basic
```

## Hardware

Developed and benchmarked on an 8 GB RAM machine running Ubuntu.


## Correctness

The naive baseline is validated against exact theoretical results, not just spot-checked:

- Norm preservation verified after every gate application (unitary matrices must preserve vector norm)
- Measured success probability matches the exact formula P(k) = sin²((2k+1)θ/2) to within 1e-9, checked at every iteration across multiple qubit counts and marked-element counts
- Over-rotation behavior confirmed: probability oscillates exactly as predicted when running past the optimal iteration count
- The n=2, M=1 special case (theory predicts exactly P=1.0) passes to 12 decimal places
- 95/95 automated test assertions passing (`tests/test_correctness.cpp`)

This test suite is the reference every subsequent optimization is checked against.