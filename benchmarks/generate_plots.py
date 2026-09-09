import matplotlib
matplotlib.use('Agg')  # non-interactive backend — saves to file, no display needed
import matplotlib.pyplot as plt
import csv
import math
import os
import sys

# Ensure output directory exists
os.makedirs("plots", exist_ok=True)

# ============================================================
# HELPER: read CSV safely
# ============================================================

def read_csv(filepath):
    """Read a CSV file and return list of dicts."""
    if not os.path.exists(filepath):
        print(f"  WARNING: {filepath} not found, skipping.")
        return None
    with open(filepath) as f:
        reader = csv.DictReader(f)
        return list(reader)


# ============================================================
# PLOT 1: Runtime vs n (naive vs optimized, log scale)
# ============================================================

def plot_naive_vs_optimized():
    print("Generating Plot 1: Naive vs Optimized runtime...")

    data = read_csv("naive_vs_optimized.csv")
    if not data:
        return

    naive_n = []
    naive_t = []
    opt_n = []
    opt_t = []

    for row in data:
        n = int(row["n"])
        opt_time = float(row["opt_total_s"])
        opt_n.append(n)
        opt_t.append(opt_time)

        if row["naive_total_s"] != "OOM":
            naive_n.append(n)
            naive_t.append(float(row["naive_total_s"]))

    # Add optimized-only scaling data if available
    data2 = read_csv("optimized_scaling.csv")
    if data2:
        existing_n = set(opt_n)
        for row in data2:
            n = int(row["n"])
            if n not in existing_n:
                opt_n.append(n)
                opt_t.append(float(row["total_s"]))
                existing_n.add(n)

    # Sort by n
    opt_pairs = sorted(zip(opt_n, opt_t))
    opt_n, opt_t = zip(*opt_pairs) if opt_pairs else ([], [])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(naive_n, naive_t, 'ro-', label='Naive (full matrices)', markersize=8, linewidth=2)
    ax.semilogy(opt_n, opt_t, 'b^-', label='Optimized (zero matrices)', markersize=8, linewidth=2)

    # Mark where naive dies
    if naive_n:
        max_naive = max(naive_n)
        ax.axvline(x=max_naive + 0.5, color='red', linestyle='--', alpha=0.5)
        ax.text(max_naive + 0.7, max(naive_t) * 0.5, 'Naive OOM\nlimit',
                color='red', fontsize=10, alpha=0.7)

    ax.set_xlabel('Number of qubits (n)', fontsize=12)
    ax.set_ylabel('Total runtime (seconds, log scale)', fontsize=12)
    ax.set_title("Grover's Algorithm: Naive vs Optimized Runtime", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(min(naive_n + list(opt_n)), max(list(opt_n)) + 1, 2))

    plt.tight_layout()
    plt.savefig("plots/runtime_naive_vs_optimized.png", dpi=150)
    plt.close()
    print("  Saved: plots/runtime_naive_vs_optimized.png")


# ============================================================
# PLOT 2: Thread scaling
# ============================================================

def plot_thread_scaling():
    print("Generating Plot 2: Thread scaling...")

    data = read_csv("thread_scaling.csv")
    if not data:
        return

    threads = [int(row["threads"]) for row in data]
    speedups = [float(row["speedup"]) for row in data]
    max_t = max(threads)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(threads, speedups, 'go-', label='Measured speedup', markersize=10, linewidth=2)
    ax.plot([1, max_t], [1, max_t], 'k--', alpha=0.4, label='Ideal linear scaling')

    ax.set_xlabel('Number of threads', fontsize=12)
    ax.set_ylabel('Speedup vs 1 thread', fontsize=12)
    ax.set_title('OpenMP Thread Scaling (n=20, 50 iterations)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(threads)
    ax.set_yticks(range(1, max_t + 1))

    plt.tight_layout()
    plt.savefig("plots/thread_scaling.png", dpi=150)
    plt.close()
    print("  Saved: plots/thread_scaling.png")


# ============================================================
# PLOT 3: Memory comparison (naive vs optimized)
# ============================================================

def plot_memory():
    print("Generating Plot 3: Memory comparison...")

    data = read_csv("memory_comparison.csv")
    if not data:
        return

    ns = []
    naive_mb = []
    opt_mb = []

    for row in data:
        n = int(row["n"])
        nm = float(row["naive_matrix_mb"])
        om = float(row["optimized_mb"])
        if n <= 24 and nm > 0 and om > 0:
            ns.append(n)
            naive_mb.append(nm)
            opt_mb.append(om)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(ns, naive_mb, 'rs-', label='Naive (3 full matrices)', markersize=8, linewidth=2)
    ax.semilogy(ns, opt_mb, 'b^-', label='Optimized (state vector only)', markersize=8, linewidth=2)

    # Add 8 GB RAM line
    ax.axhline(y=8192, color='gray', linestyle='--', alpha=0.5)
    ax.text(min(ns) + 0.3, 9500, '8 GB RAM limit', color='gray', fontsize=10)

    ax.set_xlabel('Number of qubits (n)', fontsize=12)
    ax.set_ylabel('Memory usage (MB, log scale)', fontsize=12)
    ax.set_title('Memory Usage: Naive vs Optimized', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("plots/memory_comparison.png", dpi=150)
    plt.close()
    print("  Saved: plots/memory_comparison.png")


# ============================================================
# PLOT 4: Measured vs theoretical success probability
# ============================================================

def plot_theory_vs_measured():
    print("Generating Plot 4: Measured vs theoretical success probability...")

    # Generate data: for n=5 (clear oscillation), plot P vs k
    n = 5
    N = 2 ** n
    M = 1
    theta = 2.0 * math.asin(math.sqrt(M / N))

    max_k = 15
    ks = list(range(max_k + 1))
    theory_p = [math.sin((2 * k + 1) * theta / 2) ** 2 for k in ks]

    # Read from optimized scaling or accuracy CSV if available
    # For this plot, we generate the theoretical curve and overlay our known data points
    # from the Day 3 over-rotation test output

    # Known data points from Day 3 over-rotation test (n=5, M=1)
    measured_k = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    measured_p = [0.031250, 0.258301, 0.602425, 0.896937, 0.999182,
                  0.859637, 0.545892, 0.209918, 0.014453, 0.054175,
                  0.309843, 0.657618, 0.929048]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ks, theory_p, 'b-', label='Theory: sin²((2k+1)θ/2)', linewidth=2, alpha=0.7)
    ax.plot(measured_k, measured_p, 'ro', label='Measured (C++ simulator)', markersize=8)

    # Mark optimal R
    R_opt = int(math.floor((math.pi / 4.0) * math.sqrt(N / M)))
    ax.axvline(x=R_opt, color='green', linestyle='--', alpha=0.5)
    ax.text(R_opt + 0.2, 0.95, f'Optimal R={R_opt}', color='green', fontsize=10)

    ax.set_xlabel('Grover iterations (k)', fontsize=12)
    ax.set_ylabel('Success probability P(k)', fontsize=12)
    ax.set_title(f"Grover's Algorithm: Measured vs Theoretical (n={n}, M={M})", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.1)

    plt.tight_layout()
    plt.savefig("plots/theory_vs_measured.png", dpi=150)
    plt.close()
    print("  Saved: plots/theory_vs_measured.png")


# ============================================================
# PLOT 5: Speedup growth (naive/optimized ratio vs n)
# ============================================================

def plot_speedup_growth():
    print("Generating Plot 5: Speedup growth vs n...")

    data = read_csv("naive_vs_optimized.csv")
    if not data:
        return

    ns = []
    speedups = []

    for row in data:
        if row["speedup"] != "NA" and row["naive_total_s"] != "OOM":
            ns.append(int(row["n"]))
            speedups.append(float(row["speedup"]))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(ns, speedups, 'mo-', markersize=10, linewidth=2)

    ax.set_xlabel('Number of qubits (n)', fontsize=12)
    ax.set_ylabel('Speedup (naive time / optimized time, log scale)', fontsize=12)
    ax.set_title('Speedup from Optimization vs Problem Size', fontsize=14)
    ax.grid(True, alpha=0.3)

    # Annotate max speedup
    max_idx = speedups.index(max(speedups))
    ax.annotate(f'{speedups[max_idx]:.0f}x at n={ns[max_idx]}',
                xy=(ns[max_idx], speedups[max_idx]),
                xytext=(ns[max_idx] - 2, speedups[max_idx] * 2),
                fontsize=11, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='purple'),
                color='purple')

    plt.tight_layout()
    plt.savefig("plots/speedup_growth.png", dpi=150)
    plt.close()
    print("  Saved: plots/speedup_growth.png")


# ============================================================
# PLOT 6: fp32 vs fp64 accuracy
# ============================================================

def plot_precision_comparison():
    print("Generating Plot 6: fp32 vs fp64 accuracy...")

    data = read_csv("fp32_vs_fp64_accuracy.csv")
    if not data:
        return

    ns = []
    err64 = []
    err32 = []

    for row in data:
        ns.append(int(row["n"]))
        err64.append(float(row["fp64_err"]))
        err32.append(float(row["fp32_err"]))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(ns, err64, 'b^-', label='fp64 error vs theory', markersize=8, linewidth=2)
    ax.semilogy(ns, err32, 'rs-', label='fp32 error vs theory', markersize=8, linewidth=2)

    ax.axhline(y=1e-7, color='red', linestyle=':', alpha=0.4)
    ax.text(min(ns) + 0.3, 1.5e-7, 'fp32 machine epsilon', color='red', fontsize=9, alpha=0.6)
    ax.axhline(y=2.2e-16, color='blue', linestyle=':', alpha=0.4)
    ax.text(min(ns) + 0.3, 3.5e-16, 'fp64 machine epsilon', color='blue', fontsize=9, alpha=0.6)

    ax.set_xlabel('Number of qubits (n)', fontsize=12)
    ax.set_ylabel('Absolute error in success probability (log scale)', fontsize=12)
    ax.set_title('Precision: fp32 vs fp64 Error Growth', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("plots/precision_fp32_vs_fp64.png", dpi=150)
    plt.close()
    print("  Saved: plots/precision_fp32_vs_fp64.png")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=== Day 11: Generating Benchmark Plots ===\n")

    # Change to benchmarks directory where CSVs live
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    plot_naive_vs_optimized()
    plot_thread_scaling()
    plot_memory()
    plot_theory_vs_measured()
    plot_speedup_growth()
    plot_precision_comparison()

    print(f"\n=== All plots saved to {os.path.join(script_dir, 'plots')}/ ===")
    print("Generated 6 plots:")
    print("  1. runtime_naive_vs_optimized.png — Runtime comparison (log scale)")
    print("  2. thread_scaling.png — OpenMP thread scaling")
    print("  3. memory_comparison.png — Memory usage comparison (log scale)")
    print("  4. theory_vs_measured.png — Measured vs theoretical success probability")
    print("  5. speedup_growth.png — Speedup factor vs problem size")
    print("  6. precision_fp32_vs_fp64.png — fp32 vs fp64 error growth")


if __name__ == "__main__":
    main()