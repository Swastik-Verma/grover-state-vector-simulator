#ifndef NAIVE_GROVER_H
#define NAIVE_GROVER_H

#include "state_vector.h"
#include <vector>
#include <complex>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <functional>

// A full matrix of complex numbers, stored as a vector of rows
// Matrix[i][j] = element at row i, column j
using Complex = std::complex<double>;
using Matrix = std::vector<std::vector<Complex>>;

// ============================================================
// MATRIX UTILITIES
// ============================================================

// Create a zero matrix of size rows x cols
Matrix create_matrix(uint64_t rows, uint64_t cols) {
    return Matrix(rows, std::vector<Complex>(cols, Complex(0.0, 0.0)));
}

// Create an identity matrix of size n x n
Matrix identity_matrix(uint64_t n) {
    Matrix I = create_matrix(n, n);
    for (uint64_t i = 0; i < n; ++i) {
        I[i][i] = Complex(1.0, 0.0);
    }
    return I;
}

// Matrix-vector multiply: result = M * v
// M is a dim x dim matrix, v is a dim-length vector (the state vector)
void mat_vec_multiply(const Matrix& M, StateVector& sv) {
    uint64_t dim = sv.dimension();
    std::vector<Complex> result(dim, Complex(0.0, 0.0));

    for (uint64_t i = 0; i < dim; ++i) {
        for (uint64_t j = 0; j < dim; ++j) {
            result[i] += M[i][j] * sv[j];
        }
    }

    // Copy result back into state vector
    for (uint64_t i = 0; i < dim; ++i) {
        sv[i] = result[i];
    }
}

// Kronecker product of two matrices: A ⊗ B
// If A is m×n and B is p×q, result is (m*p) × (n*q)
Matrix kronecker_product(const Matrix& A, const Matrix& B) {
    uint64_t m = A.size();
    uint64_t n = A[0].size();
    uint64_t p = B.size();
    uint64_t q = B[0].size();

    Matrix result = create_matrix(m * p, n * q);

    for (uint64_t i = 0; i < m; ++i) {
        for (uint64_t j = 0; j < n; ++j) {
            for (uint64_t k = 0; k < p; ++k) {
                for (uint64_t l = 0; l < q; ++l) {
                    result[i * p + k][j * q + l] = A[i][j] * B[k][l];
                }
            }
        }
    }

    return result;
}

// ============================================================
// BUILDING THE MATRICES FOR GROVER'S ALGORITHM
// ============================================================

// Build the single-qubit Hadamard matrix:
//     1/√2 * [ 1   1 ]
//             [ 1  -1 ]
Matrix hadamard_single() {
    double h = 1.0 / std::sqrt(2.0);
    Matrix H = create_matrix(2, 2);
    H[0][0] = Complex(h, 0.0);
    H[0][1] = Complex(h, 0.0);
    H[1][0] = Complex(h, 0.0);
    H[1][1] = Complex(-h, 0.0);
    return H;
}

// Build H^{⊗n} by repeated Kronecker product
// H^{⊗1} = H
// H^{⊗2} = H ⊗ H
// H^{⊗n} = H^{⊗(n-1)} ⊗ H
Matrix build_hadamard_full(int n) {
    Matrix result = hadamard_single();
    for (int i = 1; i < n; ++i) {
        result = kronecker_product(result, hadamard_single());
    }
    return result;
}

// Build the oracle matrix for given marked elements
// Uf = I - 2 * sum_over_marked( |x><x| )
// Which is a diagonal matrix:
//   Uf[x][x] = -1 if x is marked
//   Uf[x][x] = +1 if x is not marked
Matrix build_oracle_matrix(int num_qubits, const std::vector<uint64_t>& marked) {
    uint64_t N = 1ULL << num_qubits;
    Matrix oracle = identity_matrix(N);

    for (uint64_t m : marked) {
        oracle[m][m] = Complex(-1.0, 0.0);
    }

    return oracle;
}

// Build the diffusion matrix D = 2|ψ><ψ| - I
// |ψ> = (1/√N)(1, 1, ..., 1)^T (the uniform superposition)
// So |ψ><ψ| is the matrix where every entry is 1/N
// D[i][j] = 2/N - δ_{ij}
//         = 2/N     if i ≠ j
//         = 2/N - 1 if i = j
Matrix build_diffusion_matrix(int num_qubits) {
    uint64_t N = 1ULL << num_qubits;
    Matrix D = create_matrix(N, N);

    double two_over_N = 2.0 / static_cast<double>(N);

    for (uint64_t i = 0; i < N; ++i) {
        for (uint64_t j = 0; j < N; ++j) {
            if (i == j) {
                D[i][j] = Complex(two_over_N - 1.0, 0.0);
            } else {
                D[i][j] = Complex(two_over_N, 0.0);
            }
        }
    }

    return D;
}

// ============================================================
// COMPUTE OPTIMAL ITERATIONS
// ============================================================

// R = floor( (π/4) * √(N/M) )
int compute_optimal_iterations(int num_qubits, int num_marked) {
    uint64_t N = 1ULL << num_qubits;
    double ratio = static_cast<double>(N) / static_cast<double>(num_marked);
    int R = static_cast<int>(std::floor((M_PI / 4.0) * std::sqrt(ratio)));
    return std::max(R, 1);  // At least 1 iteration
}

// ============================================================
// THE FULL NAIVE GROVER RUNNER
// ============================================================

struct GroverResult {
    int num_qubits;
    int num_iterations;
    std::vector<uint64_t> marked_elements;
    double success_probability;  // sum of probabilities of marked elements
    double total_time_seconds;
    double matrix_build_time;
    double iteration_time;
};

GroverResult run_naive_grover(int num_qubits, const std::vector<uint64_t>& marked,
                               int iterations = -1) {
    GroverResult result;
    result.num_qubits = num_qubits;
    result.marked_elements = marked;

    uint64_t N = 1ULL << num_qubits;
    int M = marked.size();

    // Compute number of iterations if not specified
    if (iterations < 0) {
        iterations = compute_optimal_iterations(num_qubits, M);
    }
    result.num_iterations = iterations;

    std::cout << "Naive Grover: n=" << num_qubits
              << ", N=" << N
              << ", M=" << M
              << ", R=" << iterations << "\n";

    // Step 1: Build all matrices (this is the expensive part memory-wise)
    double build_time = Timer::measure([&]() {
        std::cout << "  Building H^{⊗n} (" << N << " x " << N << " matrix)...\n";
    });

    Matrix H_full;
    Matrix oracle;
    Matrix diffusion;

    result.matrix_build_time = Timer::measure([&]() {
        H_full = build_hadamard_full(num_qubits);
        oracle = build_oracle_matrix(num_qubits, marked);
        diffusion = build_diffusion_matrix(num_qubits);
    });

    std::cout << "  Matrices built in " << std::fixed << std::setprecision(4)
              << result.matrix_build_time << "s\n";

    // Step 2: Initialize state to |0...0> (already done by constructor)
    StateVector sv(num_qubits);

    // Step 3: Apply H^{⊗n} to get uniform superposition
    result.iteration_time = Timer::measure([&]() {
        mat_vec_multiply(H_full, sv);

        // Step 4: Apply Grover iterations
        for (int r = 0; r < iterations; ++r) {
            // Apply oracle
            mat_vec_multiply(oracle, sv);
            // Apply diffusion
            mat_vec_multiply(diffusion, sv);
        }
    });

    std::cout << "  Iterations completed in " << std::fixed << std::setprecision(4)
              << result.iteration_time << "s\n";

    result.total_time_seconds = result.matrix_build_time + result.iteration_time;

    // Compute success probability
    result.success_probability = 0.0;
    for (uint64_t m : marked) {
        result.success_probability += sv.probability(m);
    }

    std::cout << "  Success probability: " << std::fixed << std::setprecision(6)
              << result.success_probability << "\n";
    std::cout << "  Norm: " << std::sqrt(sv.norm_squared()) << "\n";

    return result;
}

// ============================================================
// STEP-BY-STEP GROVER (for testing/validation)
// ============================================================

// Runs Grover's algorithm step by step, checking norm after each operation.
// Returns the success probability after exactly `iterations` iterations.
// If `check_norm_each_step` is true, asserts norm ≈ 1 after every gate.
// Populates `probability_history` with the success probability after each iteration.
struct StepByStepResult {
    double final_success_probability;
    std::vector<double> probability_after_each_iteration;  // P(marked) after iteration 0, 1, 2, ...
    bool norm_ok;       // true if norm stayed ≈1 throughout
    double worst_norm;  // the norm value furthest from 1.0
};

StepByStepResult run_grover_step_by_step(
    int num_qubits,
    const std::vector<uint64_t>& marked,
    int iterations)
{
    StepByStepResult result;
    result.norm_ok = true;
    result.worst_norm = 1.0;

    uint64_t N = 1ULL << num_qubits;

    // Build matrices
    Matrix H_full = build_hadamard_full(num_qubits);
    Matrix oracle = build_oracle_matrix(num_qubits, marked);
    Matrix diffusion = build_diffusion_matrix(num_qubits);

    // Initialize state to |0...0>
    StateVector sv(num_qubits);

    // Helper lambda: check the norm and track the worst deviation
    auto check_norm = [&](const std::string& label) {
        double ns = sv.norm_squared();
        double deviation = std::abs(ns - 1.0);
        if (deviation > std::abs(result.worst_norm - 1.0)) {
            result.worst_norm = ns;
        }
        if (deviation > 1e-9) {
            std::cerr << "  NORM DRIFT at " << label
                      << ": norm²=" << ns << " (off by " << deviation << ")\n";
            result.norm_ok = false;
        }
    };

    // Helper lambda: compute success probability
    auto success_prob = [&]() {
        double p = 0.0;
        for (uint64_t m : marked) {
            p += sv.probability(m);
        }
        return p;
    };

    // Apply H^{⊗n}
    mat_vec_multiply(H_full, sv);
    check_norm("after H^n");

    // Record probability at iteration 0 (before any Grover iteration)
    result.probability_after_each_iteration.push_back(success_prob());

    // Apply Grover iterations
    for (int r = 0; r < iterations; ++r) {
        mat_vec_multiply(oracle, sv);
        check_norm("after oracle, iteration " + std::to_string(r + 1));

        mat_vec_multiply(diffusion, sv);
        check_norm("after diffusion, iteration " + std::to_string(r + 1));

        result.probability_after_each_iteration.push_back(success_prob());
    }

    result.final_success_probability = success_prob();
    return result;
}

// Compute theoretical success probability after k iterations
// P(k) = sin²((2k+1) * θ/2)  where sin(θ/2) = √(M/N)
double theoretical_success_probability(int num_qubits, int num_marked, int k) {
    uint64_t N = 1ULL << num_qubits;
    double theta = 2.0 * std::asin(std::sqrt(
        static_cast<double>(num_marked) / static_cast<double>(N)
    ));
    double angle = (2.0 * k + 1.0) * theta / 2.0;
    return std::sin(angle) * std::sin(angle);
}

#endif // NAIVE_GROVER_H