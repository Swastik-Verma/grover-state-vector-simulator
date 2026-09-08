#ifndef OPTIMIZED_GATES_H
#define OPTIMIZED_GATES_H

#include "state_vector.h"
#include <complex>
#include <cstdint>
#include <cmath>
#include <functional>
#include <array>

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

#endif // OPTIMIZED_GATES_H