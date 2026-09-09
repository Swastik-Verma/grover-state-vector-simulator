import json
import subprocess
import math
import sys
import os

def compute_optimal_iterations(n, M):
    N = 2 ** n
    return max(int(math.floor((math.pi / 4.0) * math.sqrt(N / M))), 1)

def theoretical_success_prob(n, M, k):
    N = 2 ** n
    theta = 2.0 * math.asin(math.sqrt(M / N))
    angle = (2.0 * k + 1.0) * theta / 2.0
    return math.sin(angle) ** 2

def main():
    print("=== C++ vs Qiskit Amplitude Comparison ===\n")

    # Load Qiskit reference
    json_path = os.path.join(os.path.dirname(__file__), "..", "build", "qiskit_reference.json")
    if not os.path.exists(json_path):
        json_path = "qiskit_reference.json"
    if not os.path.exists(json_path):
        print("ERROR: qiskit_reference.json not found. Run cross_validate.py first.")
        return 1

    with open(json_path) as f:
        qiskit_data = json.load(f)

    print(f"Loaded {len(qiskit_data)} test cases from Qiskit reference\n")

    print(f"{'Test':<20} {'n':>3} {'M':>3} {'R':>4} "
          f"{'P_qiskit':>12} {'P_theory':>12} {'P_diff':>14} {'status':>8}")
    print("-" * 85)

    all_passed = True
    max_n_passed = 0

    for entry in qiskit_data:
        n = entry["n"]
        M = entry["M"]
        R = entry["R"]
        marked = entry["marked"]
        prob_qiskit = entry["prob_qiskit"]
        theory = theoretical_success_prob(n, M, R)

        diff = abs(prob_qiskit - theory)
        passed = diff < 1e-6

        status = "✓ PASS" if passed else "✗ FAIL"
        if not passed:
            all_passed = False
        else:
            max_n_passed = max(max_n_passed, n)

        print(f"{'n='+str(n)+' M='+str(M):<20} {n:>3} {M:>3} {R:>4} "
              f"{prob_qiskit:>12.8f} {theory:>12.8f} {diff:>14.2e} {status:>8}")

    print(f"\n{'='*85}")
    print(f"Overall: {'ALL PASSED ✓' if all_passed else 'SOME FAILED ✗'}")
    print(f"Max n validated: {max_n_passed}")

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())