import numpy as np
import json
import sys
import math
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

def build_grover_circuit(num_qubits, marked_elements, num_iterations):
    """Build a Grover circuit for the given parameters."""
    n = num_qubits
    N = 2 ** n
    qc = QuantumCircuit(n)

    # Step 1: Apply H to all qubits → uniform superposition
    qc.h(range(n))

    for _ in range(num_iterations):
        # Oracle: flip phase of marked elements
        for target in marked_elements:
            # Convert target to binary, apply multi-controlled Z
            binary = format(target, f'0{n}b')
            # Apply X to qubits that are 0 in the target
            for i in range(n):
                if binary[n - 1 - i] == '0':
                    qc.x(i)
            # Multi-controlled Z = H on last, MCX, H on last
            if n == 1:
                qc.z(0)
            elif n == 2:
                qc.cz(0, 1)
            else:
                qc.h(n - 1)
                qc.mcx(list(range(n - 1)), n - 1)
                qc.h(n - 1)
            # Undo X gates
            for i in range(n):
                if binary[n - 1 - i] == '0':
                    qc.x(i)

        # Diffusion: 2|ψ><ψ| - I = H^n (2|0><0| - I) H^n
        qc.h(range(n))
        # 2|0><0| - I: flip phase of all states except |0>
        qc.x(range(n))
        if n == 1:
            qc.z(0)
        elif n == 2:
            qc.cz(0, 1)
        else:
            qc.h(n - 1)
            qc.mcx(list(range(n - 1)), n - 1)
            qc.h(n - 1)
        qc.x(range(n))
        qc.h(range(n))

    return qc


def get_statevector(qc):
    """Run the circuit on the statevector simulator and return amplitudes."""
    sim = AerSimulator(method='statevector')
    qc_save = qc.copy()
    qc_save.save_statevector()
    result = sim.run(qc_save).result()
    sv = result.get_statevector(qc_save)
    return np.array(sv)


def compute_optimal_iterations(num_qubits, num_marked):
    N = 2 ** num_qubits
    R = int(math.floor((math.pi / 4.0) * math.sqrt(N / num_marked)))
    return max(R, 1)


def theoretical_success_prob(num_qubits, num_marked, k):
    N = 2 ** num_qubits
    theta = 2.0 * math.asin(math.sqrt(num_marked / N))
    angle = (2.0 * k + 1.0) * theta / 2.0
    return math.sin(angle) ** 2


def main():
    print("=== Day 10: Qiskit Cross-Validation ===\n")

    test_cases = [
        {"n": 2, "marked": [3], "label": "n=2, M=1 (exact case)"},
        {"n": 3, "marked": [5], "label": "n=3, M=1"},
        {"n": 4, "marked": [10], "label": "n=4, M=1"},
        {"n": 5, "marked": [17], "label": "n=5, M=1"},
        {"n": 6, "marked": [42], "label": "n=6, M=1"},
        {"n": 8, "marked": [200], "label": "n=8, M=1"},
        {"n": 10, "marked": [500], "label": "n=10, M=1"},
        {"n": 12, "marked": [1000], "label": "n=12, M=1"},
        {"n": 4, "marked": [3, 12], "label": "n=4, M=2"},
        {"n": 5, "marked": [1, 10, 25], "label": "n=5, M=3"},
    ]

    results = []
    all_passed = True

    print(f"{'Test':<25} {'n':>3} {'M':>3} {'R':>4} "
          f"{'P_qiskit':>12} {'P_theory':>12} {'diff':>14} {'status':>8}")
    print("-" * 90)

    for tc in test_cases:
        n = tc["n"]
        marked = tc["marked"]
        M = len(marked)
        R = compute_optimal_iterations(n, M)
        theory = theoretical_success_prob(n, M, R)

        try:
            qc = build_grover_circuit(n, marked, R)
            sv = get_statevector(qc)

            # Compute success probability from Qiskit statevector
            prob_qiskit = sum(abs(sv[m]) ** 2 for m in marked)
            diff = abs(prob_qiskit - theory)
            passed = diff < 1e-6

            status = "✓ PASS" if passed else "✗ FAIL"
            if not passed:
                all_passed = False

            print(f"{tc['label']:<25} {n:>3} {M:>3} {R:>4} "
                  f"{prob_qiskit:>12.8f} {theory:>12.8f} {diff:>14.2e} {status:>8}")

            results.append({
                "label": tc["label"],
                "n": n, "M": M, "R": R,
                "prob_qiskit": prob_qiskit,
                "prob_theory": theory,
                "diff": diff,
                "passed": passed
            })

        except Exception as e:
            print(f"{tc['label']:<25} {n:>3} {M:>3} {R:>4}  ERROR: {e}")
            all_passed = False

    # Now compare Qiskit statevectors with our C++ simulator output
    # We'll do amplitude-level comparison for small cases
    print("\n\n=== Amplitude-level comparison (Qiskit vs theory) ===\n")
    print(f"{'Test':<25} {'n':>3} {'R':>4} {'max_amp_diff':>16} {'norm_qiskit':>14} {'status':>8}")
    print("-" * 80)

    for tc in test_cases:
        n = tc["n"]
        marked = tc["marked"]
        M = len(marked)
        R = compute_optimal_iterations(n, M)

        if n > 12:
            continue  # skip large cases for amplitude comparison

        try:
            qc = build_grover_circuit(n, marked, R)
            sv = get_statevector(qc)

            norm_qiskit = sum(abs(a) ** 2 for a in sv)
            norm_diff = abs(norm_qiskit - 1.0)

            status = "✓ PASS" if norm_diff < 1e-10 else "✗ FAIL"

            print(f"{tc['label']:<25} {n:>3} {R:>4} {'(vs C++ later)':>16} "
                  f"{norm_qiskit:>14.10f} {status:>8}")

        except Exception as e:
            print(f"{tc['label']:<25}  ERROR: {e}")

    # Save results for comparison with C++ output
    print("\n\n=== Generating C++ comparison data ===\n")

    comparison_data = []
    for tc in test_cases:
        n = tc["n"]
        marked = tc["marked"]
        M = len(marked)
        R = compute_optimal_iterations(n, M)

        if n > 14:
            continue

        try:
            qc = build_grover_circuit(n, marked, R)
            sv = get_statevector(qc)

            entry = {
                "n": n, "M": M, "R": R,
                "marked": marked,
                "prob_qiskit": float(sum(abs(sv[m]) ** 2 for m in marked)),
                "amplitudes_real": [float(a.real) for a in sv],
                "amplitudes_imag": [float(a.imag) for a in sv],
            }
            comparison_data.append(entry)
            print(f"  Generated data for n={n}, M={M}")

        except Exception as e:
            print(f"  Error for n={n}: {e}")

    with open("qiskit_reference.json", "w") as f:
        json.dump(comparison_data, f, indent=2)
    print("\n  Saved: qiskit_reference.json")

    # Summary
    print(f"\n\n=== SUMMARY ===")
    passed_count = sum(1 for r in results if r["passed"])
    total_count = len(results)
    print(f"  {passed_count}/{total_count} probability checks passed")
    print(f"  Overall: {'ALL PASSED ✓' if all_passed else 'SOME FAILED ✗'}")
    print(f"\n  Max n validated against Qiskit: {max(r['n'] for r in results if r['passed'])}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())