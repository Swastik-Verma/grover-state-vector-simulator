#include "state_vector.h"
#include "naive_grover.h"

int main() {
    std::cout << "=== Grover State-Vector Simulator — Day 2: Naive Implementation ===\n\n";

    // -------------------------------------------------------
    // Test 1: Small case — n=3, 1 marked element
    // -------------------------------------------------------
    std::cout << "=== Test 1: n=3, marked={5} ===\n";
    {
        GroverResult res = run_naive_grover(3, {5});
        std::cout << "\n";
    }

    // -------------------------------------------------------
    // Test 2: n=2, M=1 — special case
    // With n=2 (N=4) and M=1, one iteration should give
    // probability 1.0 (this is a known exact result)
    // -------------------------------------------------------
    std::cout << "=== Test 2: n=2, marked={3} (should give P≈1.0) ===\n";
    {
        GroverResult res = run_naive_grover(2, {3});
        std::cout << "\n";
    }

    // -------------------------------------------------------
    // Test 3: n=4, multiple marked elements
    // -------------------------------------------------------
    std::cout << "=== Test 3: n=4, marked={2, 7, 11} ===\n";
    {
        GroverResult res = run_naive_grover(4, {2, 7, 11});
        std::cout << "\n";
    }

    // -------------------------------------------------------
    // Scaling test: push n upward until it dies
    // -------------------------------------------------------
    std::cout << "=== Scaling test: finding where naive dies ===\n";
    for (int n = 5; n <= 15; ++n) {
        std::cout << "\n--- n = " << n << " ---\n";

        // Mark element 0 (arbitrary choice, doesn't matter which)
        std::vector<uint64_t> marked = {0};

        try {
            GroverResult res = run_naive_grover(n, marked);
            std::cout << "  Total time: " << std::fixed << std::setprecision(4)
                      << res.total_time_seconds << "s\n";

            // Estimate memory used by the three matrices
            uint64_t N = 1ULL << n;
            double matrix_memory_gb = (3.0 * N * N * sizeof(Complex)) / (1024.0*1024.0*1024.0);
            double sv_memory_mb = (N * sizeof(Complex)) / (1024.0*1024.0);
            std::cout << "  Matrix memory: " << std::fixed << std::setprecision(3)
                      << matrix_memory_gb << " GB"
                      << "  |  State vector: " << sv_memory_mb << " MB\n";

        } catch (const std::bad_alloc& e) {
            std::cout << "  *** OUT OF MEMORY at n=" << n << " ***\n";
            std::cout << "  This is expected. The naive approach stores full 2^n x 2^n matrices.\n";
            uint64_t N = 1ULL << n;
            double would_need_gb = (3.0 * N * N * sizeof(Complex)) / (1024.0*1024.0*1024.0);
            std::cout << "  Would need ~" << std::fixed << std::setprecision(1)
                      << would_need_gb << " GB just for matrices.\n";
            break;  // Stop trying larger n
        }
    }

    std::cout << "\n=== Day 2 complete. Record the max-n and timings in notes/results.md ===\n";
    return 0;
}