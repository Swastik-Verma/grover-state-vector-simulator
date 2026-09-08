#ifndef STATE_VECTOR_H
#define STATE_VECTOR_H

#include <vector>
#include <complex>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <stdexcept>
#include <string>
#include <chrono>
#include <functional>

class StateVector {
public:
    using Amplitude = std::complex<double>;

    // Constructor: creates a state vector for n qubits, initialized to |0...0>
    explicit StateVector(int num_qubits)
        : n_qubits(num_qubits), amplitudes(1ULL << num_qubits, Amplitude(0.0, 0.0))
    {
        if (num_qubits < 1 || num_qubits > 30) {
            throw std::invalid_argument(
                "num_qubits must be between 1 and 30. Got: " + std::to_string(num_qubits)
            );
        }
        // Initialize to |0...0>: the first basis state has amplitude 1
        amplitudes[0] = Amplitude(1.0, 0.0);
    }

    // Number of qubits
    int num_qubits() const { return n_qubits; }

    // Dimension of the state space (2^n)
    uint64_t dimension() const { return 1ULL << n_qubits; }

    // Access amplitudes (read/write)
    Amplitude& operator[](uint64_t index) { return amplitudes[index]; }
    const Amplitude& operator[](uint64_t index) const { return amplitudes[index]; }

    // Compute the probability of measuring a given basis state
    double probability(uint64_t index) const {
        return std::norm(amplitudes[index]);  // |amplitude|^2
    }

    // Compute the squared norm of the entire state vector (should be 1.0)
    double norm_squared() const {
        double sum = 0.0;
        for (uint64_t i = 0; i < dimension(); ++i) {
            sum += std::norm(amplitudes[i]);
        }
        return sum;
    }

    // Print the state vector (useful for debugging small cases)
    void print(int precision = 4) const {
        uint64_t N = dimension();
        std::cout << "State vector (" << n_qubits << " qubits, "
                  << N << " amplitudes):\n";

        // For large state vectors, only print non-negligible amplitudes
        uint64_t print_limit = (N > 64) ? 64 : N;
        bool truncated = (N > 64);

        for (uint64_t i = 0; i < print_limit; ++i) {
            if (std::abs(amplitudes[i]) > 1e-10) {
                std::cout << "  |" << to_binary(i, n_qubits) << "> ("
                          << std::setw(3) << i << "): "
                          << std::fixed << std::setprecision(precision)
                          << amplitudes[i]
                          << "  (prob: " << probability(i) << ")\n";
            }
        }

        if (truncated) {
            std::cout << "  ... (" << N << " total amplitudes, showing first 64 non-zero)\n";
        }

        std::cout << "  Norm: " << std::sqrt(norm_squared()) << "\n";
    }

private:
    int n_qubits;
    std::vector<Amplitude> amplitudes;

    // Helper: convert index to binary string for pretty printing
    static std::string to_binary(uint64_t val, int width) {
        std::string result(width, '0');
        for (int i = width - 1; i >= 0; --i) {
            if (val & 1) result[i] = '1';
            val >>= 1;
        }
        return result;
    }
};

class Timer {
public:
    // Measures the execution time of a block of code
    // Returns elapsed time in seconds
    static double measure(std::function<void()> func) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        return elapsed.count();
    }

    // Measures and prints
    static double measure_and_print(const std::string& label, std::function<void()> func) {
        double elapsed = measure(func);
        std::cout << label << ": " << std::fixed << std::setprecision(6)
                  << elapsed << " seconds\n";
        return elapsed;
    }
};

#endif // STATE_VECTOR_H