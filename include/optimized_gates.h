#ifndef OPTIMIZED_GATES_H
#define OPTIMIZED_GATES_H

#include "state_vector.h"
#include <complex>
#include <cstdint>
#include <cmath>
#include <functional>
#include <array>
#include <omp.h>

using Complex = std::complex<double>;

// ============================================================
// GENERIC SINGLE-QUBIT GATE (IN-PLACE)
// ============================================================
//
// Applies an arbitrary 2×2 unitary gate to qubit `target` of
// the state vector, in-place. No matrix is built or stored.
//
// The gate is specified as a 2×2 array:
//   gate = [[g00, g01],
//           [g10, g11]]
//
// For each pair of indices (i, j) that differ only in bit `target`:
//   new_sv[i] = g00 * sv[i] + g01 * sv[j]
//   new_sv[j] = g10 * sv[i] + g11 * sv[j]
//
// where i has bit `target` = 0 and j has bit `target` = 1.

using Gate2x2 = std::array<std::array<Complex, 2>, 2>;

void apply_single_qubit_gate(StateVector& sv, int target, const Gate2x2& gate) {
    int n = sv.num_qubits();
    uint64_t N = sv.dimension();

    // Number of pairs = N/2 = 2^(n-1)
    uint64_t num_pairs = N >> 1;

    uint64_t step = 1ULL << target;       // 2^target
    uint64_t low_mask = step - 1;          // bits below target: 0...01...1

    #pragma omp parallel for schedule(static)
    for (uint64_t k = 0; k < num_pairs; ++k) {
        // Construct index i with bit `target` = 0
        // Lower bits: k & low_mask (the bits of k below position target)
        // Upper bits: (k >> target) << (target + 1) (bits of k at/above target, shifted up by 1)
        uint64_t i = ((k >> target) << (target + 1)) | (k & low_mask);
        uint64_t j = i | step;  // same as i but with bit `target` set to 1

        // Read current amplitudes
        Complex amp_i = sv[i];
        Complex amp_j = sv[j];

        // Apply the 2x2 gate
        sv[i] = gate[0][0] * amp_i + gate[0][1] * amp_j;
        sv[j] = gate[1][0] * amp_i + gate[1][1] * amp_j;
    }
}

// ============================================================
// PREDEFINED GATES
// ============================================================

// Hadamard gate: (1/√2) * [[1, 1], [1, -1]]
Gate2x2 hadamard_gate() {
    double h = 1.0 / std::sqrt(2.0);
    Gate2x2 H;
    H[0][0] = Complex(h, 0.0);
    H[0][1] = Complex(h, 0.0);
    H[1][0] = Complex(h, 0.0);
    H[1][1] = Complex(-h, 0.0);
    return H;
}

// Pauli-X gate (NOT gate): [[0, 1], [1, 0]]
Gate2x2 pauli_x_gate() {
    Gate2x2 X;
    X[0][0] = Complex(0.0, 0.0);
    X[0][1] = Complex(1.0, 0.0);
    X[1][0] = Complex(1.0, 0.0);
    X[1][1] = Complex(0.0, 0.0);
    return X;
}

// Pauli-Z gate: [[1, 0], [0, -1]]
Gate2x2 pauli_z_gate() {
    Gate2x2 Z;
    Z[0][0] = Complex(1.0, 0.0);
    Z[0][1] = Complex(0.0, 0.0);
    Z[1][0] = Complex(0.0, 0.0);
    Z[1][1] = Complex(-1.0, 0.0);
    return Z;
}

// Phase gate: [[1, 0], [0, e^{iφ}]]
Gate2x2 phase_gate(double phi) {
    Gate2x2 P;
    P[0][0] = Complex(1.0, 0.0);
    P[0][1] = Complex(0.0, 0.0);
    P[1][0] = Complex(0.0, 0.0);
    P[1][1] = std::polar(1.0, phi);  // e^{iφ}
    return P;
}

// ============================================================
// APPLY H^{⊗n} IN-PLACE
// ============================================================
// Instead of building the full 2^n × 2^n matrix H^{⊗n},
// apply H to each qubit sequentially. The result is identical
// because gates on different qubits commute.

void apply_hadamard_all(StateVector& sv) {
    Gate2x2 H = hadamard_gate();
    for (int q = 0; q < sv.num_qubits(); ++q) {
        apply_single_qubit_gate(sv, q, H);
    }
}

// ============================================================
// OPTIMIZED ORACLE (IN-PLACE, NO MATRIX)
// ============================================================
//
// The oracle flips the sign of marked elements:
//   amplitude[x] *= -1  if is_marked(x)
//
// Two interfaces:
//   1. Predicate-based (general): takes a function bool(uint64_t)
//   2. Set-based (convenience): takes a vector of marked indices
//
// The predicate interface is the general one — it supports any
// oracle, including structured oracles where "is_marked" is a
// computation (e.g., "does this index encode a valid long path
// in a graph?"), not just a lookup in a precomputed set.

// Predicate-based oracle: the most general form
void apply_oracle_predicate(StateVector& sv,
                            const std::function<bool(uint64_t)>& is_marked) {
    uint64_t N = sv.dimension();

    #pragma omp parallel for schedule(static)
    for (uint64_t x = 0; x < N; ++x) {
        if (is_marked(x)) {
            sv[x] = -sv[x];
        }
    }
}

// Set-based oracle: convenience wrapper for known marked indices
void apply_oracle_set(StateVector& sv, const std::vector<uint64_t>& marked) {
    for (uint64_t m : marked) {
        sv[m] = -sv[m];
    }
}

// ============================================================
// OPTIMIZED DIFFUSION (IN-PLACE, NO MATRIX)
// ============================================================
//
// D = 2|ψ⟩⟨ψ| - I performs inversion about the mean:
//   α'_i = -α_i + 2 * mean(α)
//
// Implementation:
//   Pass 1: compute sum of all amplitudes → mean = sum / N
//   Pass 2: for each i, α_i → -α_i + 2·mean
//
// Time: O(N).  Extra memory: O(1) (just one Complex for the sum).
//
// Note on numerical precision:
// Summing N complex numbers accumulates rounding error of roughly
// O(√N · ε) where ε ≈ 2.2e-16 (double machine epsilon). For
// N = 2^25 (~33M), this gives ~1.3e-12 error, which is acceptable.
// If higher precision were needed, Kahan compensated summation
// could be used — it reduces error to O(ε) regardless of N, at
// the cost of 4 extra floating-point ops per addition. For our
// use case, naive summation is sufficient (verified by tests).

void apply_diffusion(StateVector& sv) {
    uint64_t N = sv.dimension();

    // Pass 1: compute sum of all amplitudes (reduction)
    double sum_real = 0.0;
    double sum_imag = 0.0;

    #pragma omp parallel for reduction(+:sum_real,sum_imag) schedule(static)
    for (uint64_t i = 0; i < N; ++i) {
        sum_real += sv[i].real();
        sum_imag += sv[i].imag();
    }

    Complex two_mean = Complex(2.0 * sum_real / static_cast<double>(N),
                               2.0 * sum_imag / static_cast<double>(N));

    // Pass 2: inversion about the mean
    #pragma omp parallel for schedule(static)
    for (uint64_t i = 0; i < N; ++i) {
        sv[i] = two_mean - sv[i];
    }
}

// ============================================================
// COMPLETE OPTIMIZED GROVER RUNNER (ZERO MATRICES)
// ============================================================
//
// This is the fully optimized Grover simulator:
//   - H^{⊗n} via in-place single-qubit gates (Day 4)
//   - Oracle via predicate/set (no matrix)
//   - Diffusion via inversion-about-mean (no matrix)
//
// Total memory: just the state vector (16 bytes × 2^n)
// Time per iteration: O(N) for oracle + O(N) for diffusion + O(n·N) for H = O(n·N)
// Total time: O(R · n · N) where R ≈ (π/4)·√N

struct OptimizedGroverResult {
    int num_qubits;
    int num_iterations;
    double success_probability;
    double total_time_seconds;
    double time_per_iteration;
};

// Version with set of marked indices
OptimizedGroverResult run_optimized_grover(
    int num_qubits,
    const std::vector<uint64_t>& marked,
    int iterations = -1)
{
    OptimizedGroverResult result;
    result.num_qubits = num_qubits;

    uint64_t N = 1ULL << num_qubits;
    int M = marked.size();

    if (iterations < 0) {
        iterations = compute_optimal_iterations(num_qubits, M);
    }
    result.num_iterations = iterations;

    // Initialize state to |0...0>
    StateVector sv(num_qubits);

    result.total_time_seconds = Timer::measure([&]() {
        // Step 1: Apply H^{⊗n} to get uniform superposition
        apply_hadamard_all(sv);

        // Step 2: Grover iterations
        for (int r = 0; r < iterations; ++r) {
            // Oracle: flip marked elements
            apply_oracle_set(sv, marked);

            // Diffusion: inversion about the mean
            apply_diffusion(sv);
        }
    });

    result.time_per_iteration = (iterations > 0)
        ? result.total_time_seconds / iterations
        : 0.0;

    // Compute success probability
    result.success_probability = 0.0;
    for (uint64_t m : marked) {
        result.success_probability += sv.probability(m);
    }

    return result;
}

// Version with predicate oracle (for future research use)
OptimizedGroverResult run_optimized_grover_predicate(
    int num_qubits,
    const std::function<bool(uint64_t)>& is_marked,
    int num_marked,
    int iterations = -1)
{
    OptimizedGroverResult result;
    result.num_qubits = num_qubits;

    uint64_t N = 1ULL << num_qubits;

    if (iterations < 0) {
        iterations = compute_optimal_iterations(num_qubits, num_marked);
    }
    result.num_iterations = iterations;

    StateVector sv(num_qubits);

    result.total_time_seconds = Timer::measure([&]() {
        apply_hadamard_all(sv);

        for (int r = 0; r < iterations; ++r) {
            apply_oracle_predicate(sv, is_marked);
            apply_diffusion(sv);
        }
    });

    result.time_per_iteration = (iterations > 0)
        ? result.total_time_seconds / iterations
        : 0.0;

    // Can't easily compute success probability without knowing
    // which indices are marked — caller should check
    result.success_probability = -1.0;  // sentinel: not computed

    return result;
}

// ============================================================
// EXAMPLE STRUCTURED ORACLES
// ============================================================
// These demonstrate how different "problems" plug into the
// predicate interface. Each is just a function bool(uint64_t).

// Oracle 1: Single marked index (the basic case)
std::function<bool(uint64_t)> oracle_single_index(uint64_t target) {
    return [target](uint64_t x) { return x == target; };
}

// Oracle 2: All indices where bit k is set
// "Find an element with property: bit k = 1"
std::function<bool(uint64_t)> oracle_bit_set(int bit_position) {
    return [bit_position](uint64_t x) {
        return (x >> bit_position) & 1;
    };
}

// Oracle 3: All indices where the number of set bits is exactly k
// "Find an element with exactly k bits set" (Hamming weight)
std::function<bool(uint64_t)> oracle_hamming_weight(int target_weight) {
    return [target_weight](uint64_t x) {
        return __builtin_popcountll(x) == target_weight;
    };
}

// Oracle 4: Indices in a given range [lo, hi)
// "Find an element in the range"
std::function<bool(uint64_t)> oracle_range(uint64_t lo, uint64_t hi) {
    return [lo, hi](uint64_t x) {
        return x >= lo && x < hi;
    };
}


#endif // OPTIMIZED_GATES_H