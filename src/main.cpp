#include "state_vector.h"

int main() {
    std::cout << "=== Grover State-Vector Simulator ===\n\n";

    // Test 1: Create a small state vector and print it
    std::cout << "--- Test: 3-qubit state vector (initialized to |000>) ---\n";
    StateVector sv(3);
    sv.print();

    std::cout << "\n--- Test: Manually set some amplitudes ---\n";
    // Set to an equal superposition manually (what H^{⊗n}|0> would produce)
    // For 3 qubits, that's 1/sqrt(8) for each of 8 basis states
    uint64_t N = sv.dimension();
    double amp = 1.0 / std::sqrt(static_cast<double>(N));
    for (uint64_t i = 0; i < N; ++i) {
        sv[i] = StateVector::Amplitude(amp, 0.0);
    }
    sv.print();

    // Test 2: Verify norm is 1
    std::cout << "\nNorm squared: " << sv.norm_squared()
              << " (should be 1.0)\n";

    // Test 3: Memory sizing for various n
    std::cout << "\n--- Memory requirements ---\n";
    for (int n = 5; n <= 30; n += 5) {
        uint64_t dim = 1ULL << n;
        double memory_mb = (dim * sizeof(StateVector::Amplitude)) / (1024.0 * 1024.0);
        double memory_gb = memory_mb / 1024.0;
        if (memory_gb >= 1.0) {
            std::cout << "n = " << n << ": 2^" << n << " = " << dim
                      << " amplitudes, " << std::fixed << std::setprecision(2)
                      << memory_gb << " GB\n";
        } else {
            std::cout << "n = " << n << ": 2^" << n << " = " << dim
                      << " amplitudes, " << std::fixed << std::setprecision(2)
                      << memory_mb << " MB\n";
        }
    }

    // Test 4: Timing the allocation
    std::cout << "\n--- Allocation timing ---\n";
    for (int n = 10; n <= 25; n += 5) {
        double elapsed = Timer::measure_and_print(
            "Allocate n=" + std::to_string(n),
            [n]() {
                StateVector temp(n);
                // Force the memory to actually be touched (prevent lazy allocation)
                volatile auto x = temp[0];
                (void)x;
            }
        );
    }

    std::cout << "\nSetup complete. Ready for Day 2.\n";
    return 0;
}