#ifndef STATE_VECTOR_FLOAT_H
#define STATE_VECTOR_FLOAT_H

#include <vector>
#include <complex>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <stdexcept>
#include <string>
#include <functional>
#include <array>
#include <omp.h>

// ============================================================
// FLOAT (fp32) STATE VECTOR
// ============================================================

class StateVectorF {
public:
    using Amplitude = std::complex<float>;

    explicit StateVectorF(int num_qubits)
        : n_qubits(num_qubits), amplitudes(1ULL << num_qubits, Amplitude(0.0f, 0.0f))
    {
        if (num_qubits < 1 || num_qubits > 30) {
            throw std::invalid_argument(
                "num_qubits must be between 1 and 30. Got: " + std::to_string(num_qubits)
            );
        }
        amplitudes[0] = Amplitude(1.0f, 0.0f);
    }

    int num_qubits() const { return n_qubits; }
    uint64_t dimension() const { return 1ULL << n_qubits; }

    Amplitude& operator[](uint64_t index) { return amplitudes[index]; }
    const Amplitude& operator[](uint64_t index) const { return amplitudes[index]; }

    double probability(uint64_t index) const {
        return static_cast<double>(std::norm(amplitudes[index]));
    }

    double norm_squared() const {
        double sum = 0.0;
        for (uint64_t i = 0; i < dimension(); ++i) {
            sum += static_cast<double>(std::norm(amplitudes[i]));
        }
        return sum;
    }

private:
    int n_qubits;
    std::vector<Amplitude> amplitudes;
};

// ============================================================
// FP32 GATES
// ============================================================

using ComplexF = std::complex<float>;
using Gate2x2F = std::array<std::array<ComplexF, 2>, 2>;

Gate2x2F hadamard_gate_f() {
    float h = 1.0f / std::sqrt(2.0f);
    Gate2x2F H;
    H[0][0] = ComplexF(h, 0.0f);
    H[0][1] = ComplexF(h, 0.0f);
    H[1][0] = ComplexF(h, 0.0f);
    H[1][1] = ComplexF(-h, 0.0f);
    return H;
}

void apply_single_qubit_gate_f(StateVectorF& sv, int target, const Gate2x2F& gate) {
    uint64_t N = sv.dimension();
    uint64_t num_pairs = N >> 1;
    uint64_t step = 1ULL << target;
    uint64_t low_mask = step - 1;

    #pragma omp parallel for schedule(static)
    for (uint64_t k = 0; k < num_pairs; ++k) {
        uint64_t i = ((k >> target) << (target + 1)) | (k & low_mask);
        uint64_t j = i | step;
        ComplexF amp_i = sv[i];
        ComplexF amp_j = sv[j];
        sv[i] = gate[0][0] * amp_i + gate[0][1] * amp_j;
        sv[j] = gate[1][0] * amp_i + gate[1][1] * amp_j;
    }
}

void apply_hadamard_all_f(StateVectorF& sv) {
    Gate2x2F H = hadamard_gate_f();
    for (int q = 0; q < sv.num_qubits(); ++q) {
        apply_single_qubit_gate_f(sv, q, H);
    }
}

void apply_oracle_set_f(StateVectorF& sv, const std::vector<uint64_t>& marked) {
    for (uint64_t m : marked) {
        sv[m] = -sv[m];
    }
}

void apply_diffusion_f(StateVectorF& sv) {
    uint64_t N = sv.dimension();

    double sum_real = 0.0;
    double sum_imag = 0.0;

    #pragma omp parallel for reduction(+:sum_real,sum_imag) schedule(static)
    for (uint64_t i = 0; i < N; ++i) {
        sum_real += static_cast<double>(sv[i].real());
        sum_imag += static_cast<double>(sv[i].imag());
    }

    float two_mean_real = static_cast<float>(2.0 * sum_real / static_cast<double>(N));
    float two_mean_imag = static_cast<float>(2.0 * sum_imag / static_cast<double>(N));
    ComplexF two_mean(two_mean_real, two_mean_imag);

    #pragma omp parallel for schedule(static)
    for (uint64_t i = 0; i < N; ++i) {
        sv[i] = two_mean - sv[i];
    }
}

// ============================================================
// FP32 GROVER RUNNER
// ============================================================

struct Fp32GroverResult {
    int num_qubits;
    int num_iterations;
    double success_probability;
    double total_time_seconds;
    double norm_squared;
};

Fp32GroverResult run_grover_fp32(int num_qubits, const std::vector<uint64_t>& marked,
                                  int iterations) {
    Fp32GroverResult result;
    result.num_qubits = num_qubits;
    result.num_iterations = iterations;

    StateVectorF sv(num_qubits);

    result.total_time_seconds = Timer::measure([&]() {
        apply_hadamard_all_f(sv);
        for (int r = 0; r < iterations; ++r) {
            apply_oracle_set_f(sv, marked);
            apply_diffusion_f(sv);
        }
    });

    result.success_probability = 0.0;
    for (uint64_t m : marked) {
        result.success_probability += sv.probability(m);
    }
    result.norm_squared = sv.norm_squared();

    return result;
}

#endif // STATE_VECTOR_FLOAT_H