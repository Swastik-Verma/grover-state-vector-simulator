#include "state_vector.h"
#include "naive_grover.h"
#include "optimized_gates.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <iomanip>

int tests_passed = 0;
int tests_failed = 0;

void ASSERT_NEAR(double val, double expected, double tol, const std::string& msg) {
    if (std::abs(val - expected) > tol) {
        std::cerr << "FAIL: " << msg << "\n"
                  << "  Expected: " << std::setprecision(12) << expected << "\n"
                  << "  Got:      " << std::setprecision(12) << val << "\n"
                  << "  Diff:     " << std::scientific << std::abs(val - expected) << "\n";
        tests_failed++;
    } else {
        tests_passed++;
    }
}

void ASSERT_TRUE(bool cond, const std::string& msg) {
    if (!cond) {
        std::cerr << "FAIL: " << msg << "\n";
        tests_failed++;
    } else {
        tests_passed++;
    }
}

void print_section(const std::string& name) {
    std::cout << "\n========================================\n";
    std::cout << "TEST: " << name << "\n";
    std::cout << "========================================\n";
}

// ============================================================
// TEST 1: OPTIMIZED H^{⊗n} vs NAIVE H^{⊗n}
// ============================================================
// Both methods should produce identical state vectors when
// applied to |0...0>. This is the core correctness check for
// the bit-manipulation approach.

void test_hadamard_matches_naive() {
    print_section("Optimized H^{⊗n} matches naive H^{⊗n}");

    for (int n = 1; n <= 12; ++n) {
        uint64_t N = 1ULL << n;

        // Method 1: Naive — build full matrix, multiply
        StateVector sv_naive(n);
        Matrix H_full = build_hadamard_full(n);
        mat_vec_multiply(H_full, sv_naive);

        // Method 2: Optimized — in-place qubit-by-qubit
        StateVector sv_opt(n);
        apply_hadamard_all(sv_opt);

        // Compare every amplitude
        double max_diff = 0.0;
        for (uint64_t i = 0; i < N; ++i) {
            double diff = std::abs(sv_naive[i] - sv_opt[i]);
            max_diff = std::max(max_diff, diff);
        }

        ASSERT_TRUE(max_diff < 1e-12,
            "n=" + std::to_string(n) + ": max amplitude diff should be < 1e-12");

        std::cout << "  n=" << std::setw(2) << n
                  << ": max diff = " << std::scientific << max_diff
                  << (max_diff < 1e-12 ? "  ✓" : "  ✗") << "\n";
    }
}

// ============================================================
// TEST 2: OPTIMIZED H^{⊗n} ON NON-INITIAL STATES
// ============================================================
// H^{⊗n} applied to |0...0> always gives uniform superposition,
// which is a special case. We also test on non-trivial states
// to make sure the gate works generally, not just on |0...0>.

void test_hadamard_on_non_initial_states() {
    print_section("Optimized H^{⊗n} on non-initial states");

    for (int n = 2; n <= 10; ++n) {
        uint64_t N = 1ULL << n;

        // Create a state with varied amplitudes
        StateVector sv_naive(n);
        StateVector sv_opt(n);

        // Set both to the same non-trivial state
        for (uint64_t i = 0; i < N; ++i) {
            double real_part = std::sin(static_cast<double>(i) * 0.7);
            double imag_part = std::cos(static_cast<double>(i) * 1.3);
            Complex amp(real_part, imag_part);
            sv_naive[i] = amp;
            sv_opt[i] = amp;
        }

        // Normalize both
        double norm = std::sqrt(sv_naive.norm_squared());
        for (uint64_t i = 0; i < N; ++i) {
            sv_naive[i] /= Complex(norm, 0.0);
            sv_opt[i] /= Complex(norm, 0.0);
        }

        // Apply H^{⊗n} using both methods
        Matrix H_full = build_hadamard_full(n);
        mat_vec_multiply(H_full, sv_naive);
        apply_hadamard_all(sv_opt);

        // Compare
        double max_diff = 0.0;
        for (uint64_t i = 0; i < N; ++i) {
            double diff = std::abs(sv_naive[i] - sv_opt[i]);
            max_diff = std::max(max_diff, diff);
        }

        ASSERT_TRUE(max_diff < 1e-10,
            "n=" + std::to_string(n) + " non-initial: max diff should be < 1e-10");

        std::cout << "  n=" << std::setw(2) << n
                  << ": max diff = " << std::scientific << max_diff
                  << (max_diff < 1e-10 ? "  ✓" : "  ✗") << "\n";
    }
}

// ============================================================
// TEST 3: SINGLE-QUBIT GATE ON SPECIFIC QUBITS
// ============================================================
// Test that applying H to qubit q only affects the pairs
// that differ in bit q, and leaves everything else untouched.

void test_single_qubit_targeting() {
    print_section("Single-qubit gate targets correct qubit");

    int n = 3;
    uint64_t N = 8;
    Gate2x2 H = hadamard_gate();

    // Apply H only to qubit 0
    {
        StateVector sv(n);  // starts as |000>
        apply_single_qubit_gate(sv, 0, H);

        // H on qubit 0 of |000> should give:
        //   (1/√2)|000> + (1/√2)|001>
        double h = 1.0 / std::sqrt(2.0);
        ASSERT_NEAR(sv[0].real(), h, 1e-12, "H on q0: amp[000]");
        ASSERT_NEAR(sv[1].real(), h, 1e-12, "H on q0: amp[001]");
        for (uint64_t i = 2; i < N; ++i) {
            ASSERT_NEAR(std::abs(sv[i]), 0.0, 1e-12,
                "H on q0: amp[" + std::to_string(i) + "] should be 0");
        }
        std::cout << "  H on qubit 0 of |000>: (1/√2)|000> + (1/√2)|001> ✓\n";
    }

    // Apply H only to qubit 1
    {
        StateVector sv(n);  // starts as |000>
        apply_single_qubit_gate(sv, 1, H);

        // H on qubit 1 of |000> should give:
        //   (1/√2)|000> + (1/√2)|010>
        double h = 1.0 / std::sqrt(2.0);
        ASSERT_NEAR(sv[0].real(), h, 1e-12, "H on q1: amp[000]");
        ASSERT_NEAR(sv[2].real(), h, 1e-12, "H on q1: amp[010]");
        ASSERT_NEAR(std::abs(sv[1]), 0.0, 1e-12, "H on q1: amp[001] should be 0");
        std::cout << "  H on qubit 1 of |000>: (1/√2)|000> + (1/√2)|010> ✓\n";
    }

    // Apply H only to qubit 2
    {
        StateVector sv(n);  // starts as |000>
        apply_single_qubit_gate(sv, 2, H);

        // H on qubit 2 of |000> should give:
        //   (1/√2)|000> + (1/√2)|100>
        double h = 1.0 / std::sqrt(2.0);
        ASSERT_NEAR(sv[0].real(), h, 1e-12, "H on q2: amp[000]");
        ASSERT_NEAR(sv[4].real(), h, 1e-12, "H on q2: amp[100]");
        ASSERT_NEAR(std::abs(sv[1]), 0.0, 1e-12, "H on q2: amp[001] should be 0");
        ASSERT_NEAR(std::abs(sv[2]), 0.0, 1e-12, "H on q2: amp[010] should be 0");
        std::cout << "  H on qubit 2 of |000>: (1/√2)|000> + (1/√2)|100> ✓\n";
    }
}

// ============================================================
// TEST 4: H APPLIED TWICE = IDENTITY
// ============================================================
// H² = I (Hadamard is its own inverse). Applying H^{⊗n} twice
// should return the state vector to its original state.

void test_hadamard_self_inverse() {
    print_section("H^{⊗n} applied twice = identity");

    for (int n = 2; n <= 12; ++n) {
        uint64_t N = 1ULL << n;

        StateVector sv(n);
        // Set a non-trivial state
        for (uint64_t i = 0; i < N; ++i) {
            sv[i] = Complex(std::sin(i * 0.3), std::cos(i * 0.9));
        }
        double norm = std::sqrt(sv.norm_squared());
        for (uint64_t i = 0; i < N; ++i) {
            sv[i] /= Complex(norm, 0.0);
        }

        // Save original
        std::vector<Complex> original(N);
        for (uint64_t i = 0; i < N; ++i) {
            original[i] = sv[i];
        }

        // Apply H^{⊗n} twice
        apply_hadamard_all(sv);
        apply_hadamard_all(sv);

        // Should be back to original
        double max_diff = 0.0;
        for (uint64_t i = 0; i < N; ++i) {
            double diff = std::abs(sv[i] - original[i]);
            max_diff = std::max(max_diff, diff);
        }

        ASSERT_TRUE(max_diff < 1e-10,
            "n=" + std::to_string(n) + ": H^{⊗n} twice should return to original");

        std::cout << "  n=" << std::setw(2) << n
                  << ": max diff from original = " << std::scientific << max_diff
                  << (max_diff < 1e-10 ? "  ✓" : "  ✗") << "\n";
    }
}

// ============================================================
// TEST 5: NORM PRESERVATION
// ============================================================

void test_norm_preservation() {
    print_section("Optimized gates preserve norm");

    Gate2x2 H = hadamard_gate();

    for (int n = 2; n <= 15; ++n) {
        StateVector sv(n);

        // Apply H to each qubit and check norm after each
        double worst_deviation = 0.0;
        for (int q = 0; q < n; ++q) {
            apply_single_qubit_gate(sv, q, H);
            double dev = std::abs(sv.norm_squared() - 1.0);
            worst_deviation = std::max(worst_deviation, dev);
        }

        ASSERT_TRUE(worst_deviation < 1e-10,
            "n=" + std::to_string(n) + ": norm should be preserved");

        std::cout << "  n=" << std::setw(2) << n
                  << ": worst norm deviation = " << std::scientific
                  << worst_deviation << "\n";
    }
}

// ============================================================
// TEST 6: TIMING COMPARISON
// ============================================================
// Show how much faster the optimized version is.

void test_timing_comparison() {
    print_section("Timing: naive matrix vs optimized in-place");

    std::cout << "  " << std::setw(4) << "n"
              << std::setw(14) << "Naive (s)"
              << std::setw(14) << "Optimized (s)"
              << std::setw(12) << "Speedup"
              << "\n";

    for (int n = 4; n <= 13; ++n) {
        double naive_time = 0.0;
        double opt_time = 0.0;

        // Naive: build full matrix and multiply
        // Skip if n > 12 (too much memory)
        if (n <= 12) {
            naive_time = Timer::measure([n]() {
                StateVector sv(n);
                Matrix H_full = build_hadamard_full(n);
                mat_vec_multiply(H_full, sv);
            });
        }

        // Optimized: in-place
        opt_time = Timer::measure([n]() {
            StateVector sv(n);
            apply_hadamard_all(sv);
        });

        if (n <= 12) {
            double speedup = naive_time / opt_time;
            std::cout << "  " << std::setw(4) << n
                      << std::setw(14) << std::fixed << std::setprecision(6) << naive_time
                      << std::setw(14) << opt_time
                      << std::setw(11) << std::setprecision(1) << speedup << "x"
                      << "\n";
        } else {
            std::cout << "  " << std::setw(4) << n
                      << std::setw(14) << "OOM"
                      << std::setw(14) << std::fixed << std::setprecision(6) << opt_time
                      << std::setw(12) << "(naive dies)"
                      << "\n";
        }
    }

    // Push optimized further — beyond where naive can go
    std::cout << "\n  Optimized-only (beyond naive's reach):\n";
    for (int n = 14; n <= 22; n += 2) {
        double opt_time = Timer::measure([n]() {
            StateVector sv(n);
            apply_hadamard_all(sv);
        });
        std::cout << "  n=" << std::setw(2) << n << ": "
                  << std::fixed << std::setprecision(6) << opt_time << "s\n";
    }
}

// ============================================================
// TEST 7: FULL GROVER USING OPTIMIZED HADAMARD
// ============================================================
// Run a full Grover simulation using the optimized Hadamard
// but still using naive oracle and diffusion matrices.
// The result should match the fully naive version exactly.

void test_grover_with_optimized_hadamard() {
    print_section("Full Grover with optimized H^{⊗n} matches naive");

    for (int n = 2; n <= 10; ++n) {
        uint64_t N = 1ULL << n;
        std::vector<uint64_t> marked = {N / 3};  // arbitrary marked element
        int R = compute_optimal_iterations(n, 1);

        // Naive: everything is matrices
        StateVector sv_naive(n);
        Matrix H_full = build_hadamard_full(n);
        Matrix oracle = build_oracle_matrix(n, marked);
        Matrix diffusion = build_diffusion_matrix(n);

        mat_vec_multiply(H_full, sv_naive);
        for (int r = 0; r < R; ++r) {
            mat_vec_multiply(oracle, sv_naive);
            mat_vec_multiply(diffusion, sv_naive);
        }

        // Hybrid: optimized Hadamard, naive oracle & diffusion
        StateVector sv_hybrid(n);
        apply_hadamard_all(sv_hybrid);  // <-- optimized!
        for (int r = 0; r < R; ++r) {
            mat_vec_multiply(oracle, sv_hybrid);
            mat_vec_multiply(diffusion, sv_hybrid);
        }

        // Compare success probabilities
        double prob_naive = 0.0, prob_hybrid = 0.0;
        for (uint64_t m : marked) {
            prob_naive += sv_naive.probability(m);
            prob_hybrid += sv_hybrid.probability(m);
        }

        double diff = std::abs(prob_naive - prob_hybrid);
        ASSERT_TRUE(diff < 1e-10,
            "n=" + std::to_string(n) + ": hybrid Grover should match naive");

        std::cout << "  n=" << std::setw(2) << n
                  << ": naive P=" << std::fixed << std::setprecision(8) << prob_naive
                  << "  hybrid P=" << prob_hybrid
                  << "  diff=" << std::scientific << diff
                  << (diff < 1e-10 ? "  ✓" : "  ✗") << "\n";
    }
}

// ============================================================
// MAIN
// ============================================================

int main() {
    std::cout << "=== Day 4: Optimized Gates Validation ===\n";

    test_hadamard_matches_naive();
    test_hadamard_on_non_initial_states();
    test_single_qubit_targeting();
    test_hadamard_self_inverse();
    test_norm_preservation();
    test_timing_comparison();
    test_grover_with_optimized_hadamard();

    std::cout << "\n========================================\n";
    std::cout << "RESULTS: " << tests_passed << " passed, "
              << tests_failed << " failed out of "
              << (tests_passed + tests_failed) << " total\n";
    std::cout << "========================================\n";

    return (tests_failed > 0) ? 1 : 0;
}