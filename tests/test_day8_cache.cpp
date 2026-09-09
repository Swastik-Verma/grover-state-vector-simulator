#include "state_vector.h"
#include "naive_grover.h"
#include "optimized_gates.h"
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
                  << "  Got:      " << std::setprecision(12) << val << "\n";
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
// BLOCKED SINGLE-QUBIT GATE (cache-aware version)
// ============================================================

void apply_single_qubit_gate_blocked(StateVector& sv, int target, const Gate2x2& gate,
                                      uint64_t block_size = 1024) {
    int n = sv.num_qubits();
    uint64_t N = sv.dimension();
    uint64_t num_pairs = N >> 1;
    uint64_t step = 1ULL << target;
    uint64_t low_mask = step - 1;

    // For low-index qubits (small stride), blocking doesn't help — pairs are already adjacent
    // For high-index qubits (large stride), process in blocks so both elements of a pair
    // are likely in cache when accessed
    if (target <= 10) {
        // Small stride — original loop is already cache-friendly
        #pragma omp parallel for schedule(static)
        for (uint64_t k = 0; k < num_pairs; ++k) {
            uint64_t i = ((k >> target) << (target + 1)) | (k & low_mask);
            uint64_t j = i | step;
            Complex amp_i = sv[i];
            Complex amp_j = sv[j];
            sv[i] = gate[0][0] * amp_i + gate[0][1] * amp_j;
            sv[j] = gate[1][0] * amp_i + gate[1][1] * amp_j;
        }
    } else {
        // Large stride — use blocking
        // Process the array in chunks of block_size
        // Within each block, handle all pairs that have their i-index in this block
        uint64_t num_blocks = (N + block_size - 1) / block_size;

        #pragma omp parallel for schedule(dynamic)
        for (uint64_t b = 0; b < num_blocks; ++b) {
            uint64_t block_start = b * block_size;
            uint64_t block_end = std::min(block_start + block_size, N);

            for (uint64_t i = block_start; i < block_end; ++i) {
                // Only process indices where bit `target` is 0
                if ((i & step) == 0) {
                    uint64_t j = i | step;
                    if (j < N) {
                        Complex amp_i = sv[i];
                        Complex amp_j = sv[j];
                        sv[i] = gate[0][0] * amp_i + gate[0][1] * amp_j;
                        sv[j] = gate[1][0] * amp_i + gate[1][1] * amp_j;
                    }
                }
            }
        }
    }
}

void apply_hadamard_all_blocked(StateVector& sv, uint64_t block_size = 1024) {
    Gate2x2 H = hadamard_gate();
    for (int q = 0; q < sv.num_qubits(); ++q) {
        apply_single_qubit_gate_blocked(sv, q, H, block_size);
    }
}

void apply_diffusion_blocked(StateVector& sv) {
    // Diffusion is already cache-friendly (sequential access), no change needed
    apply_diffusion(sv);
}

// ============================================================
// TEST 1: MEASURE THE PROBLEM — gate time vs qubit index
// ============================================================

void test_measure_problem() {
    print_section("Gate time vs qubit index (showing the cache problem)");

    int n = 22;
    uint64_t N = 1ULL << n;
    double memory_gb = (N * 16.0) / (1024.0 * 1024.0 * 1024.0);
    if (memory_gb > 6.0) {
        n = 20;
        N = 1ULL << n;
    }

    StateVector sv(n);
    apply_hadamard_all(sv);  // put in non-trivial state

    Gate2x2 H = hadamard_gate();

    std::cout << "  n=" << n << " (" << N << " amplitudes, "
              << std::fixed << std::setprecision(0)
              << (N * 16.0) / (1024.0 * 1024.0) << " MB)\n\n";

    std::cout << std::setw(8) << "Qubit" << std::setw(10) << "Stride"
              << std::setw(14) << "Original (s)" << std::setw(14) << "Blocked (s)"
              << std::setw(10) << "Speedup" << "\n";

    std::ofstream csv("cache_qubit_timing.csv");
    csv << "qubit,stride,original_s,blocked_s,speedup\n";

    for (int q = 0; q < n; q += (n > 15 ? 2 : 1)) {
        uint64_t stride = 1ULL << q;

        // Benchmark original
        double orig_time = 0.0;
        {
            StateVector sv_copy(n);
            for (uint64_t i = 0; i < N; ++i) sv_copy[i] = sv[i];

            orig_time = Timer::measure([&]() {
                apply_single_qubit_gate(sv_copy, q, H);
            });
        }

        // Benchmark blocked
        double blocked_time = 0.0;
        {
            StateVector sv_copy(n);
            for (uint64_t i = 0; i < N; ++i) sv_copy[i] = sv[i];

            blocked_time = Timer::measure([&]() {
                apply_single_qubit_gate_blocked(sv_copy, q, H);
            });
        }

        double speedup = orig_time / blocked_time;

        std::cout << std::setw(8) << q << std::setw(10) << stride
                  << std::setw(14) << std::fixed << std::setprecision(6) << orig_time
                  << std::setw(14) << blocked_time
                  << std::setw(9) << std::setprecision(2) << speedup << "x\n";

        csv << q << "," << stride << "," << orig_time << "," << blocked_time << "," << speedup << "\n";
    }

    csv.close();
    std::cout << "  Saved: cache_qubit_timing.csv\n";
}

// ============================================================
// TEST 2: BLOCK SIZE TUNING
// ============================================================

void test_block_size_tuning() {
    print_section("Block size tuning for high-index qubit");

    int n = 22;
    uint64_t N = 1ULL << n;
    double memory_gb = (N * 16.0) / (1024.0 * 1024.0 * 1024.0);
    if (memory_gb > 6.0) {
        n = 20;
        N = 1ULL << n;
    }

    StateVector sv(n);
    apply_hadamard_all(sv);

    Gate2x2 H = hadamard_gate();
    int high_qubit = n - 2;  // second-highest qubit — large stride

    std::cout << "  Testing qubit " << high_qubit << " (stride " << (1ULL << high_qubit)
              << ") at n=" << n << "\n\n";

    std::cout << std::setw(14) << "Block size" << std::setw(14) << "Time (s)" << "\n";

    std::ofstream csv("block_size_tuning.csv");
    csv << "block_size,qubit,time_s\n";

    std::vector<uint64_t> block_sizes = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};

    for (uint64_t bs : block_sizes) {
        StateVector sv_copy(n);
        for (uint64_t i = 0; i < N; ++i) sv_copy[i] = sv[i];

        double t = Timer::measure([&]() {
            apply_single_qubit_gate_blocked(sv_copy, high_qubit, H, bs);
        });

        std::cout << std::setw(14) << bs << std::setw(14) << std::fixed << std::setprecision(6) << t << "\n";
        csv << bs << "," << high_qubit << "," << t << "\n";
    }

    csv.close();
    std::cout << "  Saved: block_size_tuning.csv\n";
}

// ============================================================
// TEST 3: CORRECTNESS — blocked matches original
// ============================================================

void test_blocked_correctness() {
    print_section("Blocked gate correctness vs original");

    Gate2x2 H = hadamard_gate();

    for (int n = 2; n <= 18; ++n) {
        uint64_t N = 1ULL << n;
        double memory_gb = (N * 16.0) / (1024.0 * 1024.0 * 1024.0);
        if (memory_gb > 2.0) break;

        // Create non-trivial state
        StateVector sv_orig(n);
        StateVector sv_blocked(n);
        for (uint64_t i = 0; i < N; ++i) {
            Complex amp(std::sin(i * 0.3), std::cos(i * 0.9));
            sv_orig[i] = amp;
            sv_blocked[i] = amp;
        }
        double norm = std::sqrt(sv_orig.norm_squared());
        for (uint64_t i = 0; i < N; ++i) {
            sv_orig[i] /= Complex(norm, 0.0);
            sv_blocked[i] /= Complex(norm, 0.0);
        }

        // Apply to every qubit using both methods
        for (int q = 0; q < n; ++q) {
            apply_single_qubit_gate(sv_orig, q, H);
            apply_single_qubit_gate_blocked(sv_blocked, q, H);
        }

        double max_diff = 0.0;
        for (uint64_t i = 0; i < N; ++i) {
            double diff = std::abs(sv_orig[i] - sv_blocked[i]);
            max_diff = std::max(max_diff, diff);
        }

        ASSERT_TRUE(max_diff < 1e-12,
            "n=" + std::to_string(n) + ": blocked matches original");

        std::cout << "  n=" << std::setw(2) << n
                  << ": max diff = " << std::scientific << max_diff
                  << (max_diff < 1e-12 ? "  ✓" : "  ✗") << "\n";
    }
}

// ============================================================
// TEST 4: FULL GROVER with blocked gates vs original
// ============================================================

void test_full_grover_blocked() {
    print_section("Full Grover: blocked vs original");

    for (int n = 5; n <= 18; ++n) {
        uint64_t N = 1ULL << n;
        double memory_gb = (N * 16.0) / (1024.0 * 1024.0 * 1024.0);
        if (memory_gb > 1.0) break;

        std::vector<uint64_t> marked = {0};
        int R = compute_optimal_iterations(n, 1);

        // Original
        StateVector sv_orig(n);
        apply_hadamard_all(sv_orig);
        for (int r = 0; r < R; ++r) {
            apply_oracle_set(sv_orig, marked);
            apply_diffusion(sv_orig);
        }

        // Blocked
        StateVector sv_blocked(n);
        apply_hadamard_all_blocked(sv_blocked);
        for (int r = 0; r < R; ++r) {
            apply_oracle_set(sv_blocked, marked);
            apply_diffusion_blocked(sv_blocked);
        }

        double prob_orig = sv_orig.probability(0);
        double prob_blocked = sv_blocked.probability(0);
        double diff = std::abs(prob_orig - prob_blocked);
        double theory = theoretical_success_probability(n, 1, R);

        ASSERT_NEAR(prob_blocked, theory, 1e-8,
            "n=" + std::to_string(n) + ": blocked Grover matches theory");

        std::cout << "  n=" << std::setw(2) << n << " R=" << std::setw(3) << R
                  << ": P_orig=" << std::fixed << std::setprecision(8) << prob_orig
                  << "  P_blocked=" << prob_blocked
                  << "  diff=" << std::scientific << diff
                  << (diff < 1e-8 ? "  ✓" : "  ✗") << "\n";
    }
}

// ============================================================
// TEST 5: TIMING — full H^{⊗n} original vs blocked
// ============================================================

void test_hadamard_timing() {
    print_section("Full H^{⊗n} timing: original vs blocked");

    std::cout << std::setw(4) << "n" << std::setw(14) << "Original (s)"
              << std::setw(14) << "Blocked (s)" << std::setw(10) << "Speedup" << "\n";

    std::ofstream csv("hadamard_blocked_timing.csv");
    csv << "n,original_s,blocked_s,speedup\n";

    for (int n = 12; n <= 24; ++n) {
        uint64_t N = 1ULL << n;
        double memory_gb = (N * 16.0) / (1024.0 * 1024.0 * 1024.0);
        if (memory_gb > 6.0) break;

        double orig_time = Timer::measure([n]() {
            StateVector sv(n);
            apply_hadamard_all(sv);
        });

        double blocked_time = Timer::measure([n]() {
            StateVector sv(n);
            apply_hadamard_all_blocked(sv);
        });

        double speedup = orig_time / blocked_time;

        std::cout << std::setw(4) << n
                  << std::setw(14) << std::fixed << std::setprecision(6) << orig_time
                  << std::setw(14) << blocked_time
                  << std::setw(9) << std::setprecision(2) << speedup << "x\n";

        csv << n << "," << orig_time << "," << blocked_time << "," << speedup << "\n";
    }

    csv.close();
    std::cout << "  Saved: hadamard_blocked_timing.csv\n";
}

// ============================================================
// MAIN
// ============================================================

int main() {
    std::cout << "=== Day 8: Cache-Aware Access ===\n";
    std::cout << "OpenMP threads: " << omp_get_max_threads() << "\n";

    test_measure_problem();
    test_block_size_tuning();
    test_blocked_correctness();
    test_full_grover_blocked();
    test_hadamard_timing();

    std::cout << "\n========================================\n";
    std::cout << "RESULTS: " << tests_passed << " passed, "
              << tests_failed << " failed out of "
              << (tests_passed + tests_failed) << " total\n";
    std::cout << "========================================\n";

    return (tests_failed > 0) ? 1 : 0;
}