#include "state_vector.h"
#include "naive_grover.h"
#include "optimized_gates.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>

int main() {
    std::cout << "=== Day 6: Comprehensive Benchmark Suite ===\n\n";

    // -------------------------------------------------------
    // Benchmark 1: Naive vs Optimized (overlapping range)
    // -------------------------------------------------------
    {
        std::ofstream csv("naive_vs_optimized.csv");
        csv << "n,N,R,naive_total_s,opt_total_s,speedup,naive_matrix_mb,opt_memory_mb,naive_prob,opt_prob\n";

        std::cout << "=== Benchmark 1: Naive vs Optimized ===\n";
        std::cout << std::setw(4) << "n" << std::setw(10) << "N" << std::setw(6) << "R"
                  << std::setw(14) << "Naive (s)" << std::setw(14) << "Optimized (s)"
                  << std::setw(10) << "Speedup" << std::setw(12) << "P_naive"
                  << std::setw(12) << "P_opt" << "\n";

        for (int n = 3; n <= 13; ++n) {
            uint64_t N = 1ULL << n;
            std::vector<uint64_t> marked = {0};
            int R = compute_optimal_iterations(n, 1);

            double naive_time = 0.0;
            double naive_prob = 0.0;
            double naive_matrix_mb = 0.0;

            if (n <= 12) {
                // Run naive
                StateVector sv_naive(n);
                Matrix H_full, oracle, diffusion;

                naive_time = Timer::measure([&]() {
                    H_full = build_hadamard_full(n);
                    oracle = build_oracle_matrix(n, marked);
                    diffusion = build_diffusion_matrix(n);
                    mat_vec_multiply(H_full, sv_naive);
                    for (int r = 0; r < R; ++r) {
                        mat_vec_multiply(oracle, sv_naive);
                        mat_vec_multiply(diffusion, sv_naive);
                    }
                });

                naive_prob = sv_naive.probability(0);
                naive_matrix_mb = (3.0 * N * N * sizeof(Complex)) / (1024.0 * 1024.0);
            }

            // Run optimized
            auto opt_res = run_optimized_grover(n, marked, R);
            double opt_memory_mb = (N * sizeof(Complex)) / (1024.0 * 1024.0);

            double speedup = (n <= 12 && naive_time > 0) ? naive_time / opt_res.total_time_seconds : 0.0;

            std::cout << std::setw(4) << n << std::setw(10) << N << std::setw(6) << R;
            if (n <= 12) {
                std::cout << std::setw(14) << std::fixed << std::setprecision(4) << naive_time;
            } else {
                std::cout << std::setw(14) << "OOM";
            }
            std::cout << std::setw(14) << std::fixed << std::setprecision(4) << opt_res.total_time_seconds;
            if (n <= 12) {
                std::cout << std::setw(9) << std::setprecision(1) << speedup << "x";
            } else {
                std::cout << std::setw(10) << "—";
            }
            std::cout << std::setw(12) << std::setprecision(6) << (n <= 12 ? naive_prob : 0.0)
                      << std::setw(12) << opt_res.success_probability << "\n";

            csv << n << "," << N << "," << R << ","
                << (n <= 12 ? std::to_string(naive_time) : "OOM") << ","
                << opt_res.total_time_seconds << ","
                << (n <= 12 ? std::to_string(speedup) : "NA") << ","
                << naive_matrix_mb << "," << opt_memory_mb << ","
                << (n <= 12 ? std::to_string(naive_prob) : "NA") << ","
                << opt_res.success_probability << "\n";
        }

        csv.close();
        std::cout << "  Saved: naive_vs_optimized.csv\n\n";
    }

    // -------------------------------------------------------
    // Benchmark 2: Optimized-only scaling to max n
    // -------------------------------------------------------
    {
        std::ofstream csv("optimized_scaling.csv");
        csv << "n,N,R,total_s,per_iter_s,memory_mb,success_prob\n";

        std::cout << "=== Benchmark 2: Optimized Scaling (push to max n) ===\n";
        std::cout << std::setw(4) << "n" << std::setw(12) << "N" << std::setw(6) << "R"
                  << std::setw(14) << "Total (s)" << std::setw(14) << "Per iter (s)"
                  << std::setw(12) << "Memory (MB)" << std::setw(12) << "P_success" << "\n";

        for (int n = 10; n <= 24; ++n) {
            uint64_t N = 1ULL << n;
            double memory_mb = (N * sizeof(Complex)) / (1024.0 * 1024.0);
            double memory_gb = memory_mb / 1024.0;

            if (memory_gb > 6.0) {
                std::cout << "  n=" << n << ": needs " << std::fixed << std::setprecision(1)
                          << memory_gb << " GB — stopping (8 GB RAM limit)\n";
                break;
            }

            std::vector<uint64_t> marked = {0};
            auto res = run_optimized_grover(n, marked);

            std::cout << std::setw(4) << n << std::setw(12) << N << std::setw(6) << res.num_iterations
                      << std::setw(14) << std::fixed << std::setprecision(4) << res.total_time_seconds
                      << std::setw(14) << std::setprecision(6) << res.time_per_iteration
                      << std::setw(12) << std::setprecision(2) << memory_mb
                      << std::setw(12) << std::setprecision(6) << res.success_probability << "\n";

            csv << n << "," << N << "," << res.num_iterations << ","
                << res.total_time_seconds << "," << res.time_per_iteration << ","
                << memory_mb << "," << res.success_probability << "\n";
        }

        csv.close();
        std::cout << "  Saved: optimized_scaling.csv\n\n";
    }

    // -------------------------------------------------------
    // Benchmark 3: Memory comparison
    // -------------------------------------------------------
    {
        std::ofstream csv("memory_comparison.csv");
        csv << "n,N,naive_matrix_mb,optimized_mb,reduction_factor\n";

        std::cout << "=== Benchmark 3: Memory Usage Comparison ===\n";
        std::cout << std::setw(4) << "n" << std::setw(16) << "Naive matrices"
                  << std::setw(16) << "Optimized" << std::setw(16) << "Reduction" << "\n";

        for (int n = 5; n <= 28; ++n) {
            uint64_t N = 1ULL << n;
            double naive_mb = (3.0 * N * N * sizeof(Complex)) / (1024.0 * 1024.0);
            double opt_mb = (N * sizeof(Complex)) / (1024.0 * 1024.0);
            double reduction = naive_mb / opt_mb;

            if (opt_mb / 1024.0 > 6.0) break;

            std::string naive_str, opt_str;
            if (naive_mb > 1024.0) {
                naive_str = std::to_string(static_cast<int>(naive_mb / 1024.0)) + " GB";
            } else {
                naive_str = std::to_string(static_cast<int>(naive_mb)) + " MB";
            }
            if (opt_mb > 1024.0) {
                opt_str = std::to_string(static_cast<int>(opt_mb / 1024.0)) + " GB";
            } else if (opt_mb >= 1.0) {
                opt_str = std::to_string(static_cast<int>(opt_mb)) + " MB";
            } else {
                opt_str = std::to_string(static_cast<int>(opt_mb * 1024.0)) + " KB";
            }

            std::cout << std::setw(4) << n << std::setw(16) << naive_str
                      << std::setw(16) << opt_str
                      << std::setw(15) << std::fixed << std::setprecision(0) << reduction << "x\n";

            csv << n << "," << N << "," << naive_mb << "," << opt_mb << "," << reduction << "\n";
        }

        csv.close();
        std::cout << "  Saved: memory_comparison.csv\n";
    }

    std::cout << "\n=== All benchmarks complete. CSVs saved in build directory. ===\n";
    std::cout << "Copy CSVs to benchmarks/ folder:\n";
    std::cout << "  cp *.csv ../benchmarks/\n";
    return 0;
}