# Grover State-Vector Simulator

A C++ state-vector simulator that executes Grover's quantum search algorithm, built from scratch with a focus on performance engineering.

## Project Status

Currently in active development (Phase 1 — Foundation).

- [x] Day 1: Project setup, CMake build system, StateVector class
- [x] Day 2: Naive full-matrix baseline implementation
- [ ] Day 3: Correctness validation suite
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