#include "state_vector.h"
#include "naive_grover.h"
#include "optimized_gates.h"
#include <omp.h>
#include <cassert>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>

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
// TEST 1: CORRECTNESS — OpenMP results match serial
// ============================================================

void test_correctness_vs_theory() {
    print_section("OpenMP Grover matches theory (correctness check)");

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
        {12, {1000}},
        {14, {5000}},
        {4, {3, 12}},
        {5, {1, 10, 25}},
    };

    for (auto& tc : cases) {
        int M = tc.marked.size();
        int R = compute_optimal_iterations(tc.n, M);

        auto res = run_optimized_grover(tc.n, tc.marked, R);
        double theory = theoretical_success_probability(tc.n, M, R);
        double diff = std::abs(res.success_probability - theory);

        ASSERT_NEAR(res.success_probability, theory, 1e-8,
            "n=" + std::to_string(tc.n) + " M=" + std::to_string(M));

        std::cout << "  n=" << std::setw(2) << tc.n
                  << " M=" << M << " R=" << std::setw(3) << R
                  << ": P=" << std::fixed << std::setprecision(8) << res.success_probability
                  << "  theory=" << theory
                  << "  diff=" << std::scientific << diff
                  << (diff < 1e-8 ? "  ✓" : "  ✗") << "\n";
    }
}

// ============================================================
// TEST 2: NORM PRESERVATION WITH OpenMP
// ============================================================

void test_norm_preservation() {
    print_section("Norm preserved with OpenMP");

    for (int n = 2; n <= 18; ++n) {
        uint64_t N = 1ULL << n;
        double memory_gb = (N * 16.0) / (1024.0*1024.0*1024.0);
        if (memory_gb > 2.0) break;

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

        ASSERT_TRUE(worst_dev < 1e-8,
            "n=" + std::to_string(n) + ": norm preserved");

        std::cout << "  n=" << std::setw(2) << n << ", R=" << std::setw(3) << R
                  << ": worst deviation = " << std::scientific << worst_dev
                  << (worst_dev < 1e-8 ? "  ✓" : "  ✗") << "\n";
    }
}

// ============================================================
// TEST 3: THREAD SCALING BENCHMARK
// ============================================================

void test_thread_scaling() {
    print_section("Thread scaling benchmark");

    int max_threads = omp_get_max_threads();
    std::cout << "  Max threads available: " << max_threads << "\n\n";

    // Test at n=20 (16 MB state vector — large enough for parallelism to matter)
    int test_n = 20;
    uint64_t N = 1ULL << test_n;
    double memory_gb = (N * 16.0) / (1024.0*1024.0*1024.0);

    // If n=20 would use too much memory, fall back to n=18
    if (memory_gb > 4.0) {
        test_n = 18;
        N = 1ULL << test_n;
    }

    std::vector<uint64_t> marked = {0};
    int R = compute_optimal_iterations(test_n, 1);

    // Limit iterations so each run doesn't take too long
    int bench_iters = std::min(R, 50);

    std::cout << "  Benchmarking at n=" << test_n << " (N=" << N
              << "), " << bench_iters << " iterations\n\n";

    std::cout << std::setw(10) << "Threads"
              << std::setw(14) << "Time (s)"
              << std::setw(12) << "Speedup"
              << std::setw(14) << "Efficiency"
              << std::setw(12) << "P_success" << "\n";

    double baseline_time = 0.0;

    // Build list of thread counts to test
    std::vector<int> thread_counts = {1};
    for (int t = 2; t <= max_threads; t++) {
        thread_counts.push_back(t);
    }

    std::ofstream csv("thread_scaling.csv");
    csv << "threads,n,iterations,time_s,speedup,efficiency,success_prob\n";

    for (int num_threads : thread_counts) {
        omp_set_num_threads(num_threads);

        StateVector sv(test_n);
        double prob = 0.0;

        double elapsed = Timer::measure([&]() {
            apply_hadamard_all(sv);
            for (int r = 0; r < bench_iters; ++r) {
                apply_oracle_set(sv, marked);
                apply_diffusion(sv);
            }
        });

        prob = sv.probability(0);

        if (num_threads == 1) {
            baseline_time = elapsed;
        }

        double speedup = baseline_time / elapsed;
        double efficiency = speedup / num_threads * 100.0;

        std::cout << std::setw(10) << num_threads
                  << std::setw(14) << std::fixed << std::setprecision(4) << elapsed
                  << std::setw(11) << std::setprecision(2) << speedup << "x"
                  << std::setw(12) << std::setprecision(1) << efficiency << "%"
                  << std::setw(12) << std::setprecision(6) << prob << "\n";

        csv << num_threads << "," << test_n << "," << bench_iters << ","
            << elapsed << "," << speedup << "," << efficiency << "," << prob << "\n";
    }

    csv.close();
    std::cout << "\n  Saved: thread_scaling.csv\n";

    // Restore max threads
    omp_set_num_threads(max_threads);
}

// ============================================================
// TEST 4: FULL GROVER SCALING WITH ALL THREADS
// ============================================================

void test_full_scaling_parallel() {
    print_section("Full Grover scaling with OpenMP (all threads)");

    int max_threads = omp_get_max_threads();
    omp_set_num_threads(max_threads);
    std::cout << "  Using " << max_threads << " threads\n\n";

    std::cout << std::setw(4) << "n" << std::setw(12) << "N"
              << std::setw(6) << "R" << std::setw(14) << "Total (s)"
              << std::setw(14) << "Per iter (s)"
              << std::setw(12) << "P_success" << "\n";

    std::ofstream csv("parallel_scaling.csv");
    csv << "threads,n,N,R,total_s,per_iter_s,success_prob\n";

    for (int n = 10; n <= 24; ++n) {
        uint64_t N = 1ULL << n;
        double memory_gb = (N * 16.0) / (1024.0*1024.0*1024.0);
        if (memory_gb > 6.0) {
            std::cout << "  n=" << n << ": needs " << std::fixed << std::setprecision(1)
                      << memory_gb << " GB — stopping\n";
            break;
        }

        std::vector<uint64_t> marked = {0};
        auto res = run_optimized_grover(n, marked);

        std::cout << std::setw(4) << n << std::setw(12) << N
                  << std::setw(6) << res.num_iterations
                  << std::setw(14) << std::fixed << std::setprecision(4) << res.total_time_seconds
                  << std::setw(14) << std::setprecision(6) << res.time_per_iteration
                  << std::setw(12) << std::setprecision(6) << res.success_probability << "\n";

        csv << max_threads << "," << n << "," << N << "," << res.num_iterations << ","
            << res.total_time_seconds << "," << res.time_per_iteration << ","
            << res.success_probability << "\n";
    }

    csv.close();
    std::cout << "  Saved: parallel_scaling.csv\n";
}

// ============================================================
// TEST 5: VERIFY Day 3 TESTS STILL PASS
// ============================================================

void test_day3_regression() {
    print_section("Day 3 regression — over-rotation still correct");

    int n = 5;
    int R_opt = compute_optimal_iterations(n, 1);
    int run_to = R_opt * 3;

    // Run step-by-step (uses naive matrices, not affected by OpenMP)
    auto result = run_grover_step_by_step(n, {0}, run_to);

    bool all_match = true;
    for (int k = 0; k <= run_to; ++k) {
        double measured = result.probability_after_each_iteration[k];
        double theory = theoretical_success_probability(n, 1, k);
        double diff = std::abs(measured - theory);
        if (diff > 1e-9) {
            all_match = false;
            std::cout << "  k=" << k << ": MISMATCH diff=" << diff << "\n";
        }
    }

    ASSERT_TRUE(all_match, "Over-rotation pattern still matches theory");
    ASSERT_TRUE(result.norm_ok, "Norm preserved in step-by-step run");

    std::cout << "  Over-rotation pattern: " << (all_match ? "PASSED ✓" : "FAILED ✗") << "\n";
    std::cout << "  Norm check: " << (result.norm_ok ? "PASSED ✓" : "FAILED ✗") << "\n";
}

// ============================================================
// MAIN
// ============================================================

int main() {
    std::cout << "=== Day 7: OpenMP Parallelism ===\n";
    std::cout << "OpenMP max threads: " << omp_get_max_threads() << "\n";

    test_correctness_vs_theory();
    test_norm_preservation();
    test_thread_scaling();
    test_full_scaling_parallel();
    test_day3_regression();

    std::cout << "\n========================================\n";
    std::cout << "RESULTS: " << tests_passed << " passed, "
              << tests_failed << " failed out of "
              << (tests_passed + tests_failed) << " total\n";
    std::cout << "========================================\n";

    return (tests_failed > 0) ? 1 : 0;
}