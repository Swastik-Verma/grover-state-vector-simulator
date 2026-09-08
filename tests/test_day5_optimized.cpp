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
// TEST 1: OPTIMIZED ORACLE vs NAIVE ORACLE MATRIX
// ============================================================

void test_oracle_matches_naive() {
    print_section("Optimized oracle matches naive oracle matrix");

    for (int n = 2; n <= 10; ++n) {
        uint64_t N = 1ULL << n;
        std::vector<uint64_t> marked = {N / 4, N / 2, N - 1};

        // Create two identical states (non-trivial)
        StateVector sv_naive(n);
        StateVector sv_opt(n);

        // Start from uniform superposition
        apply_hadamard_all(sv_naive);
        apply_hadamard_all(sv_opt);

        // Apply naive oracle (matrix multiply)
        Matrix oracle_matrix = build_oracle_matrix(n, marked);
        mat_vec_multiply(oracle_matrix, sv_naive);

        // Apply optimized oracle (in-place)
        apply_oracle_set(sv_opt, marked);

        // Compare
        double max_diff = 0.0;
        for (uint64_t i = 0; i < N; ++i) {
            double diff = std::abs(sv_naive[i] - sv_opt[i]);
            max_diff = std::max(max_diff, diff);
        }

        ASSERT_TRUE(max_diff < 1e-15,
            "n=" + std::to_string(n) + ": optimized oracle should match naive");

        std::cout << "  n=" << std::setw(2) << n
                  << ": max diff = " << std::scientific << max_diff
                  << (max_diff < 1e-15 ? "  ✓" : "  ✗") << "\n";
    }
}

// ============================================================
// TEST 2: OPTIMIZED DIFFUSION vs NAIVE DIFFUSION MATRIX
// ============================================================

void test_diffusion_matches_naive() {
    print_section("Optimized diffusion matches naive diffusion matrix");

    for (int n = 2; n <= 10; ++n) {
        uint64_t N = 1ULL << n;

        // Create two identical non-trivial states
        StateVector sv_naive(n);
        StateVector sv_opt(n);

        // Start from a post-oracle state (uniform superposition with one sign flipped)
        apply_hadamard_all(sv_naive);
        apply_hadamard_all(sv_opt);
        sv_naive[0] = -sv_naive[0];  // simulate oracle
        sv_opt[0] = -sv_opt[0];

        // Apply naive diffusion (matrix multiply)
        Matrix diff_matrix = build_diffusion_matrix(n);
        mat_vec_multiply(diff_matrix, sv_naive);

        // Apply optimized diffusion (inversion about mean)
        apply_diffusion(sv_opt);

        // Compare
        double max_diff = 0.0;
        for (uint64_t i = 0; i < N; ++i) {
            double diff = std::abs(sv_naive[i] - sv_opt[i]);
            max_diff = std::max(max_diff, diff);
        }

        ASSERT_NEAR(max_diff, 0.0, 1e-12,
            "n=" + std::to_string(n) + ": optimized diffusion should match naive");

        std::cout << "  n=" << std::setw(2) << n
                  << ": max diff = " << std::scientific << max_diff
                  << (max_diff < 1e-12 ? "  ✓" : "  ✗") << "\n";
    }
}

// ============================================================
// TEST 3: FULL OPTIMIZED GROVER vs FULL NAIVE GROVER
// ============================================================

void test_full_optimized_vs_naive() {
    print_section("Full optimized Grover matches full naive Grover");

    for (int n = 2; n <= 10; ++n) {
        uint64_t N = 1ULL << n;
        std::vector<uint64_t> marked = {N / 3};
        int M = marked.size();
        int R = compute_optimal_iterations(n, M);

        // Naive: all matrices
        StateVector sv_naive(n);
        Matrix H_full = build_hadamard_full(n);
        Matrix oracle = build_oracle_matrix(n, marked);
        Matrix diffusion = build_diffusion_matrix(n);
        mat_vec_multiply(H_full, sv_naive);
        for (int r = 0; r < R; ++r) {
            mat_vec_multiply(oracle, sv_naive);
            mat_vec_multiply(diffusion, sv_naive);
        }

        // Optimized: zero matrices
        StateVector sv_opt(n);
        apply_hadamard_all(sv_opt);
        for (int r = 0; r < R; ++r) {
            apply_oracle_set(sv_opt, marked);
            apply_diffusion(sv_opt);
        }

        // Compare every amplitude
        double max_diff = 0.0;
        for (uint64_t i = 0; i < N; ++i) {
            double diff = std::abs(sv_naive[i] - sv_opt[i]);
            max_diff = std::max(max_diff, diff);
        }

        double prob_naive = 0.0, prob_opt = 0.0;
        for (uint64_t m : marked) {
            prob_naive += sv_naive.probability(m);
            prob_opt += sv_opt.probability(m);
        }

        ASSERT_TRUE(max_diff < 1e-9,
            "n=" + std::to_string(n) + ": full optimized should match full naive");

        std::cout << "  n=" << std::setw(2) << n << ", R=" << std::setw(2) << R
                  << ": max amp diff = " << std::scientific << max_diff
                  << "  P_naive=" << std::fixed << std::setprecision(8) << prob_naive
                  << "  P_opt=" << prob_opt
                  << (max_diff < 1e-9 ? "  ✓" : "  ✗") << "\n";
    }
}

// ============================================================
// TEST 4: OPTIMIZED GROVER MATCHES THEORETICAL PROBABILITY
// ============================================================

void test_optimized_matches_theory() {
    print_section("Optimized Grover matches theoretical P(k) = sin²((2k+1)θ/2)");

    struct TestCase {
        int n;
        std::vector<uint64_t> marked;
    };

    std::vector<TestCase> cases = {
        {2, {3}},
        {3, {5}},
        {4, {10}},
        {5, {17}},
        {6, {42}},
        {8, {200}},
        {10, {500}},
        {4, {3, 12}},
        {5, {1, 10, 25}},
    };

    for (auto& tc : cases) {
        int M = tc.marked.size();
        int R = compute_optimal_iterations(tc.n, M);

        auto res = run_optimized_grover(tc.n, tc.marked, R);
        double theory = theoretical_success_probability(tc.n, M, R);
        double diff = std::abs(res.success_probability - theory);

        ASSERT_NEAR(res.success_probability, theory, 1e-9,
            "n=" + std::to_string(tc.n) + " M=" + std::to_string(M) +
            ": optimized should match theory");

        std::cout << "  n=" << std::setw(2) << tc.n
                  << " M=" << M << " R=" << std::setw(2) << R
                  << ": P_opt=" << std::fixed << std::setprecision(8) << res.success_probability
                  << "  P_theory=" << theory
                  << "  diff=" << std::scientific << diff
                  << (diff < 1e-9 ? "  ✓" : "  ✗") << "\n";
    }
}

// ============================================================
// TEST 5: PREDICATE ORACLE MATCHES SET ORACLE
// ============================================================

void test_predicate_oracle() {
    print_section("Predicate oracle matches set oracle");

    for (int n = 3; n <= 10; ++n) {
        uint64_t N = 1ULL << n;
        std::vector<uint64_t> marked = {N / 4, N / 2};

        // Set-based
        StateVector sv_set(n);
        apply_hadamard_all(sv_set);
        apply_oracle_set(sv_set, marked);

        // Predicate-based (lambda that checks membership)
        StateVector sv_pred(n);
        apply_hadamard_all(sv_pred);
        auto is_marked = [&marked](uint64_t x) {
            for (uint64_t m : marked) {
                if (x == m) return true;
            }
            return false;
        };
        apply_oracle_predicate(sv_pred, is_marked);

        // Compare
        double max_diff = 0.0;
        for (uint64_t i = 0; i < N; ++i) {
            double diff = std::abs(sv_set[i] - sv_pred[i]);
            max_diff = std::max(max_diff, diff);
        }

        ASSERT_TRUE(max_diff == 0.0,
            "n=" + std::to_string(n) + ": predicate oracle should exactly match set oracle");

        std::cout << "  n=" << std::setw(2) << n
                  << ": max diff = " << std::scientific << max_diff
                  << (max_diff == 0.0 ? "  ✓" : "  ✗") << "\n";
    }
}

// ============================================================
// TEST 6: STRUCTURED ORACLES WORK CORRECTLY
// ============================================================

void test_structured_oracles() {
    print_section("Structured oracles (bit_set, hamming_weight)");

    // Test oracle_bit_set: marks all x where bit k is set
    // For n=4 (N=16), bit 2 set → indices 4,5,6,7,12,13,14,15 → M=8
    {
        int n = 4;
        uint64_t N = 16;
        int bit = 2;
        auto pred = oracle_bit_set(bit);

        // Count how many are marked
        int count = 0;
        for (uint64_t x = 0; x < N; ++x) {
            if (pred(x)) count++;
        }

        ASSERT_TRUE(count == 8, "oracle_bit_set(2) should mark 8 of 16 elements");
        std::cout << "  oracle_bit_set(2) for n=4: marks " << count << "/16 elements ✓\n";

        // Run optimized Grover with this oracle
        int R = compute_optimal_iterations(n, count);
        auto res = run_optimized_grover_predicate(n, pred, count, R);

        // Check against theory
        // success_probability isn't computed by predicate runner,
        // so run it manually
        StateVector sv(n);
        apply_hadamard_all(sv);
        for (int r = 0; r < R; ++r) {
            apply_oracle_predicate(sv, pred);
            apply_diffusion(sv);
        }
        double prob = 0.0;
        for (uint64_t x = 0; x < N; ++x) {
            if (pred(x)) prob += sv.probability(x);
        }
        double theory = theoretical_success_probability(n, count, R);

        ASSERT_NEAR(prob, theory, 1e-9,
            "oracle_bit_set Grover should match theory");
        std::cout << "  Grover with oracle_bit_set: P=" << std::fixed << std::setprecision(6)
                  << prob << " (theory: " << theory << ") ✓\n";
    }

    // Test oracle_hamming_weight: marks all x with exactly k bits set
    // For n=5 (N=32), hamming weight 2 → C(5,2) = 10 elements
    {
        int n = 5;
        uint64_t N = 32;
        int weight = 2;
        auto pred = oracle_hamming_weight(weight);

        int count = 0;
        for (uint64_t x = 0; x < N; ++x) {
            if (pred(x)) count++;
        }

        ASSERT_TRUE(count == 10, "oracle_hamming_weight(2) for n=5 should mark C(5,2)=10");
        std::cout << "  oracle_hamming_weight(2) for n=5: marks " << count << "/32 elements ✓\n";

        int R = compute_optimal_iterations(n, count);
        StateVector sv(n);
        apply_hadamard_all(sv);
        for (int r = 0; r < R; ++r) {
            apply_oracle_predicate(sv, pred);
            apply_diffusion(sv);
        }
        double prob = 0.0;
        for (uint64_t x = 0; x < N; ++x) {
            if (pred(x)) prob += sv.probability(x);
        }
        double theory = theoretical_success_probability(n, count, R);

        ASSERT_NEAR(prob, theory, 1e-9,
            "oracle_hamming_weight Grover should match theory");
        std::cout << "  Grover with oracle_hamming_weight: P=" << std::fixed << std::setprecision(6)
                  << prob << " (theory: " << theory << ") ✓\n";
    }
}

// ============================================================
// TEST 7: NORM PRESERVATION WITH OPTIMIZED OPERATIONS
// ============================================================

void test_norm_preservation() {
    print_section("Norm preserved through full optimized Grover");

    for (int n = 2; n <= 15; ++n) {
        uint64_t N = 1ULL << n;
        std::vector<uint64_t> marked = {0};
        int R = compute_optimal_iterations(n, 1);

        StateVector sv(n);
        double worst_dev = 0.0;

        apply_hadamard_all(sv);
        worst_dev = std::max(worst_dev, std::abs(sv.norm_squared() - 1.0));

        for (int r = 0; r < R; ++r) {
            apply_oracle_set(sv, marked);
            worst_dev = std::max(worst_dev, std::abs(sv.norm_squared() - 1.0));

            apply_diffusion(sv);
            worst_dev = std::max(worst_dev, std::abs(sv.norm_squared() - 1.0));
        }

        ASSERT_TRUE(worst_dev < 1e-9,
            "n=" + std::to_string(n) + ": norm should stay ≈1 throughout");

        std::cout << "  n=" << std::setw(2) << n << ", R=" << std::setw(3) << R
                  << ": worst norm deviation = " << std::scientific << worst_dev
                  << (worst_dev < 1e-9 ? "  ✓" : "  ✗") << "\n";
    }
}

// ============================================================
// TEST 8: TIMING AND SCALING
// ============================================================

void test_timing_scaling() {
    print_section("Timing: full optimized Grover scaling");

    std::cout << "  " << std::setw(4) << "n"
              << std::setw(10) << "N"
              << std::setw(6) << "R"
              << std::setw(14) << "Total (s)"
              << std::setw(14) << "Per iter (s)"
              << std::setw(10) << "P_success"
              << "\n";

    for (int n = 5; n <= 24; ++n) {
        uint64_t N = 1ULL << n;

        // Check memory won't exceed ~6 GB (leave headroom for 8 GB system)
        double memory_gb = (N * sizeof(std::complex<double>)) / (1024.0*1024.0*1024.0);
        if (memory_gb > 6.0) {
            std::cout << "  n=" << n << ": would need " << std::fixed << std::setprecision(1)
                      << memory_gb << " GB — skipping (8 GB RAM limit)\n";
            break;
        }

        std::vector<uint64_t> marked = {0};
        auto res = run_optimized_grover(n, marked);

        std::cout << "  " << std::setw(4) << n
                  << std::setw(10) << N
                  << std::setw(6) << res.num_iterations
                  << std::setw(14) << std::fixed << std::setprecision(4) << res.total_time_seconds
                  << std::setw(14) << std::setprecision(6) << res.time_per_iteration
                  << std::setw(10) << std::setprecision(6) << res.success_probability
                  << "\n";
    }
}

// ============================================================
// MAIN
// ============================================================

int main() {
    std::cout << "=== Day 5: Zero-Matrix Optimized Grover Validation ===\n";

    test_oracle_matches_naive();
    test_diffusion_matches_naive();
    test_full_optimized_vs_naive();
    test_optimized_matches_theory();
    test_predicate_oracle();
    test_structured_oracles();
    test_norm_preservation();
    test_timing_scaling();

    std::cout << "\n========================================\n";
    std::cout << "RESULTS: " << tests_passed << " passed, "
              << tests_failed << " failed out of "
              << (tests_passed + tests_failed) << " total\n";
    std::cout << "========================================\n";

    return (tests_failed > 0) ? 1 : 0;
}