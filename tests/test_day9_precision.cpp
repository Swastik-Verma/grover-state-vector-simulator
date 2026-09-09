#include "state_vector.h"
#include "naive_grover.h"
#include "optimized_gates.h"
#include "state_vector_float.h"
#include <omp.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>

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
// TEST 1: FP32 vs FP64 PROBABILITY ACCURACY
// ============================================================

void test_fp32_vs_fp64_accuracy() {
    print_section("fp32 vs fp64 success probability accuracy");

    std::cout << std::setw(4) << "n" << std::setw(6) << "R"
              << std::setw(14) << "P_fp64" << std::setw(14) << "P_fp32"
              << std::setw(14) << "P_theory" << std::setw(14) << "fp64_err"
              << std::setw(14) << "fp32_err" << "\n";

    std::ofstream csv("fp32_vs_fp64_accuracy.csv");
    csv << "n,R,P_fp64,P_fp32,P_theory,fp64_err,fp32_err,norm_fp64,norm_fp32\n";

    for (int n = 2; n <= 22; ++n) {
        uint64_t N = 1ULL << n;
        double memory_gb = (N * 16.0) / (1024.0 * 1024.0 * 1024.0);
        if (memory_gb > 3.0) break;  // leave room for both fp32 and fp64

        std::vector<uint64_t> marked = {0};
        int R = compute_optimal_iterations(n, 1);
        double theory = theoretical_success_probability(n, 1, R);

        // fp64
        auto res64 = run_optimized_grover(n, marked, R);

        // fp32
        auto res32 = run_grover_fp32(n, marked, R);

        double err64 = std::abs(res64.success_probability - theory);
        double err32 = std::abs(res32.success_probability - theory);

        std::cout << std::setw(4) << n << std::setw(6) << R
                  << std::setw(14) << std::fixed << std::setprecision(8) << res64.success_probability
                  << std::setw(14) << res32.success_probability
                  << std::setw(14) << theory
                  << std::setw(14) << std::scientific << err64
                  << std::setw(14) << err32 << "\n";

        csv << n << "," << R << "," << res64.success_probability << ","
            << res32.success_probability << "," << theory << ","
            << err64 << "," << err32 << ","
            << std::sqrt(res64.success_probability > 0 ? 1.0 : 0.0) << ","  // placeholder
            << res32.norm_squared << "\n";

        ASSERT_TRUE(err32 < 0.01,
            "n=" + std::to_string(n) + ": fp32 error should be < 1%");
    }
}

// ============================================================
// TEST 2: FP32 NORM DRIFT
// ============================================================

void test_fp32_norm_drift() {
    print_section("fp32 norm drift over Grover iterations");

    std::cout << std::setw(4) << "n" << std::setw(6) << "R"
              << std::setw(16) << "fp64 norm²"
              << std::setw(16) << "fp32 norm²"
              << std::setw(16) << "fp32 drift" << "\n";

    std::ofstream csv("fp32_norm_drift.csv");
    csv << "n,R,norm_fp64,norm_fp32,drift_fp32\n";

    for (int n = 2; n <= 22; ++n) {
        uint64_t N = 1ULL << n;
        double memory_gb = (N * 16.0) / (1024.0 * 1024.0 * 1024.0);
        if (memory_gb > 3.0) break;

        std::vector<uint64_t> marked = {0};
        int R = compute_optimal_iterations(n, 1);

        // fp64
        StateVector sv64(n);
        apply_hadamard_all(sv64);
        for (int r = 0; r < R; ++r) {
            apply_oracle_set(sv64, marked);
            apply_diffusion(sv64);
        }
        double norm64 = sv64.norm_squared();

        // fp32
        auto res32 = run_grover_fp32(n, marked, R);
        double norm32 = res32.norm_squared;

        double drift32 = std::abs(norm32 - 1.0);

        std::cout << std::setw(4) << n << std::setw(6) << R
                  << std::setw(16) << std::fixed << std::setprecision(12) << norm64
                  << std::setw(16) << norm32
                  << std::setw(16) << std::scientific << drift32 << "\n";

        csv << n << "," << R << "," << norm64 << "," << norm32 << "," << drift32 << "\n";
    }

    csv.close();
    std::cout << "  Saved: fp32_norm_drift.csv\n";
}

// ============================================================
// TEST 3: TIMING AND MEMORY — fp32 vs fp64
// ============================================================

void test_timing_comparison() {
    print_section("Timing and memory: fp32 vs fp64");

    std::cout << std::setw(4) << "n" << std::setw(6) << "R"
              << std::setw(12) << "fp64 (s)" << std::setw(12) << "fp32 (s)"
              << std::setw(10) << "Speedup"
              << std::setw(12) << "fp64 (MB)" << std::setw(12) << "fp32 (MB)" << "\n";

    std::ofstream csv("fp32_vs_fp64_timing.csv");
    csv << "n,R,time_fp64,time_fp32,speedup,memory_fp64_mb,memory_fp32_mb\n";

    for (int n = 10; n <= 24; ++n) {
        uint64_t N = 1ULL << n;
        double mem64_mb = (N * sizeof(std::complex<double>)) / (1024.0 * 1024.0);
        double mem32_mb = (N * sizeof(std::complex<float>)) / (1024.0 * 1024.0);
        double mem64_gb = mem64_mb / 1024.0;
        double mem32_gb = mem32_mb / 1024.0;

        // Skip if fp64 would exceed ~5 GB (leave headroom)
        if (mem64_gb > 5.0) {
            // Can still try fp32 if it fits
            if (mem32_gb > 5.0) break;

            std::vector<uint64_t> marked = {0};
            int R = compute_optimal_iterations(n, 1);

            auto res32 = run_grover_fp32(n, marked, R);

            std::cout << std::setw(4) << n << std::setw(6) << R
                      << std::setw(12) << "OOM"
                      << std::setw(12) << std::fixed << std::setprecision(4) << res32.total_time_seconds
                      << std::setw(10) << "—"
                      << std::setw(12) << std::setprecision(1) << mem64_mb
                      << std::setw(12) << mem32_mb << "\n";

            csv << n << "," << R << ",OOM," << res32.total_time_seconds << ",NA,"
                << mem64_mb << "," << mem32_mb << "\n";
            continue;
        }

        std::vector<uint64_t> marked = {0};
        int R = compute_optimal_iterations(n, 1);

        auto res64 = run_optimized_grover(n, marked, R);
        auto res32 = run_grover_fp32(n, marked, R);

        double speedup = res64.total_time_seconds / res32.total_time_seconds;

        std::cout << std::setw(4) << n << std::setw(6) << R
                  << std::setw(12) << std::fixed << std::setprecision(4) << res64.total_time_seconds
                  << std::setw(12) << res32.total_time_seconds
                  << std::setw(9) << std::setprecision(2) << speedup << "x"
                  << std::setw(12) << std::setprecision(1) << mem64_mb
                  << std::setw(12) << mem32_mb << "\n";

        csv << n << "," << R << "," << res64.total_time_seconds << ","
            << res32.total_time_seconds << "," << speedup << ","
            << mem64_mb << "," << mem32_mb << "\n";
    }

    csv.close();
    std::cout << "  Saved: fp32_vs_fp64_timing.csv\n";
}

// ============================================================
// TEST 4: FIND TRUE MAX N FOR BOTH PRECISIONS
// ============================================================

void test_max_n() {
    print_section("Maximum n for fp64 and fp32");

    std::cout << "  Finding max n (allocation only, no Grover run):\n";

    int max_n_64 = 0, max_n_32 = 0;

    for (int n = 20; n <= 30; ++n) {
        uint64_t N = 1ULL << n;
        double mem64_gb = (N * sizeof(std::complex<double>)) / (1024.0 * 1024.0 * 1024.0);
        double mem32_gb = (N * sizeof(std::complex<float>)) / (1024.0 * 1024.0 * 1024.0);

        if (mem64_gb <= 6.5 && max_n_64 < n) {
            try {
                StateVector sv(n);
                volatile auto x = sv[0];
                (void)x;
                max_n_64 = n;
                std::cout << "  fp64 n=" << n << ": " << std::fixed << std::setprecision(2)
                          << mem64_gb << " GB — OK\n";
            } catch (...) {
                std::cout << "  fp64 n=" << n << ": " << std::fixed << std::setprecision(2)
                          << mem64_gb << " GB — FAILED\n";
            }
        }

        if (mem32_gb <= 6.5 && max_n_32 < n) {
            try {
                StateVectorF sv(n);
                volatile auto x = sv[0];
                (void)x;
                max_n_32 = n;
                std::cout << "  fp32 n=" << n << ": " << std::fixed << std::setprecision(2)
                          << mem32_gb << " GB — OK\n";
            } catch (...) {
                std::cout << "  fp32 n=" << n << ": " << std::fixed << std::setprecision(2)
                          << mem32_gb << " GB — FAILED\n";
            }
        }
    }

    std::cout << "\n  Max n (fp64): " << max_n_64 << "\n";
    std::cout << "  Max n (fp32): " << max_n_32 << "\n";
    std::cout << "  fp32 gains +" << (max_n_32 - max_n_64) << " extra qubit(s)\n";
}

// ============================================================
// MAIN
// ============================================================

int main() {
    std::cout << "=== Day 9: Memory and Precision Tuning ===\n";
    std::cout << "OpenMP threads: " << omp_get_max_threads() << "\n";

    test_fp32_vs_fp64_accuracy();
    test_fp32_norm_drift();
    test_timing_comparison();
    test_max_n();

    std::cout << "\n========================================\n";
    std::cout << "RESULTS: " << tests_passed << " passed, "
              << tests_failed << " failed out of "
              << (tests_passed + tests_failed) << " total\n";
    std::cout << "========================================\n";

    return (tests_failed > 0) ? 1 : 0;
}