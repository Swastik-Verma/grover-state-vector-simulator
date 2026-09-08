#include "state_vector.h"
#include <cassert>
#include <cmath>

// Simple assertion with a message
#define ASSERT_NEAR(val, expected, tol, msg) \
    do { \
        if (std::abs((val) - (expected)) > (tol)) { \
            std::cerr << "FAIL: " << (msg) << "\n" \
                      << "  Expected: " << (expected) << "\n" \
                      << "  Got:      " << (val) << "\n" \
                      << "  Tolerance: " << (tol) << "\n"; \
            return 1; \
        } \
    } while(0)

int main() {
    int failures = 0;

    // Test 1: Initial state is |0...0>
    {
        StateVector sv(3);
        ASSERT_NEAR(sv.probability(0), 1.0, 1e-12,
                     "Initial state should have P(|000>) = 1");
        for (uint64_t i = 1; i < sv.dimension(); ++i) {
            ASSERT_NEAR(sv.probability(i), 0.0, 1e-12,
                         "Initial state should have P(other) = 0");
        }
        ASSERT_NEAR(sv.norm_squared(), 1.0, 1e-12,
                     "Initial state norm should be 1");
        std::cout << "PASS: Initial state\n";
    }

    // Test 2: Norm preserved after manual uniform superposition
    {
        StateVector sv(4);
        double amp = 1.0 / std::sqrt(16.0);
        for (uint64_t i = 0; i < 16; ++i) {
            sv[i] = StateVector::Amplitude(amp, 0.0);
        }
        ASSERT_NEAR(sv.norm_squared(), 1.0, 1e-12,
                     "Uniform superposition norm should be 1");
        ASSERT_NEAR(sv.probability(0), 1.0/16.0, 1e-12,
                     "Each state should have probability 1/16");
        std::cout << "PASS: Uniform superposition\n";
    }

    // Test 3: Dimension is correct
    {
        StateVector sv(5);
        assert(sv.dimension() == 32);
        assert(sv.num_qubits() == 5);
        std::cout << "PASS: Dimensions\n";
    }

    // Test 4: Edge case - 1 qubit
    {
        StateVector sv(1);
        assert(sv.dimension() == 2);
        ASSERT_NEAR(sv.probability(0), 1.0, 1e-12,
                     "1-qubit |0> state");
        std::cout << "PASS: 1 qubit\n";
    }

    std::cout << "\nAll Day 1 tests passed!\n";
    return 0;
}