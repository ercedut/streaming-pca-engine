# Author: Emrullah Erce Dutkan
"""
Command-line interface for streaming PCA experiments.

Provides three main modes:
1. simulate: Run a real-time streaming simulation with visualization
2. benchmark: Run systematic benchmarks and generate reports
3. real: Run on a real dataset (e.g., digits)

Usage examples:
    python -m src.cli --mode simulate --d 50 --k 5 --n-steps 20000 --eta-schedule invsqrt --eta0 0.5 --plot
    python -m src.cli --mode benchmark --d 100 --k 10 --steps 30000 --quant-bits 4,6,8 --sketch-size 40 --reports reports/
    python -m src.cli --mode real --dataset digits --k 10 --eta-schedule invsqrt --plot
"""

import argparse
import os
import sys
import time
from typing import List, Optional

import numpy as np

# Add src directory to path for imports
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from streaming_pca.oja import OjaPCA
from streaming_pca.quant import QuantizedOjaPCA
from streaming_pca.sketch import SketchPCA
from streaming_pca.stream import SyntheticStream, DatasetStream
from streaming_pca.datasets import load_dataset, split_train_test
from streaming_pca.metrics import (
    batch_pca_reference,
    subspace_distance,
    explained_variance_ratio,
    reconstruction_error
)
from streaming_pca.benchmark import (
    BenchmarkConfig,
    run_benchmark,
    format_results_table,
    results_to_dict
)
from streaming_pca.viz import (
    plot_metrics_over_time,
    plot_memory_accuracy_tradeoff,
    create_report_figures
)
from streaming_pca.io import (
    save_results_csv,
    save_history_csv,
    MetricsLogger,
    ensure_dir,
    format_memory,
    create_summary_report
)
from streaming_pca.config import (
    ExperimentConfig,
    DEFAULT_REPORTS_DIR,
    DEFAULT_LOGS_DIR
)


def parse_quant_bits(bits_str: str) -> List[int]:
    """Parse comma-separated bit widths."""
    return [int(b.strip()) for b in bits_str.split(",")]


def run_simulate_mode(args: argparse.Namespace) -> None:
    """
    Run real-time streaming simulation.

    Simulates data arriving over time and updates models incrementally,
    displaying progress and metrics.
    """
    print("=" * 60)
    print("Streaming PCA Simulation")
    print("=" * 60)
    print(f"Dimensions: d={args.d}, k={args.k}")
    print(f"Steps: {args.n_steps}")
    print(f"Learning rate: eta0={args.eta0}, schedule={args.eta_schedule}")
    print(f"Seed: {args.seed}")
    print()

    # Create output directories
    ensure_dir(DEFAULT_LOGS_DIR)
    ensure_dir(DEFAULT_REPORTS_DIR)

    # Create stream
    drift_interval = args.drift_interval if args.drift else None
    stream = SyntheticStream(
        d=args.d,
        rank=args.k * 2,
        noise_std=0.1,
        seed=args.seed,
        drift_interval=drift_interval,
        drift_angle=0.1
    )

    # Generate test data
    test_stream = SyntheticStream(
        d=args.d,
        rank=args.k * 2,
        noise_std=0.1,
        seed=args.seed + 1000
    )
    X_test = test_stream.get_batch(1000)

    # Compute batch PCA reference
    ref_stream = SyntheticStream(
        d=args.d,
        rank=args.k * 2,
        noise_std=0.1,
        seed=args.seed + 2000
    )
    X_ref = ref_stream.get_batch(args.n_steps)
    W_ref = batch_pca_reference(X_ref, args.k)

    # Initialize models
    models = {
        "oja_fp": OjaPCA(
            d=args.d,
            k=args.k,
            eta0=args.eta0,
            eta_schedule=args.eta_schedule,
            seed=args.seed + 100
        ),
        "oja_quant_b8": QuantizedOjaPCA(
            d=args.d,
            k=args.k,
            bits=8,
            error_feedback=True,
            eta0=args.eta0,
            eta_schedule=args.eta_schedule,
            seed=args.seed + 200
        ),
        "sketch_fd": SketchPCA(
            d=args.d,
            k=args.k,
            sketch_size=args.k * 2,
            seed=args.seed + 300
        )
    }

    # Initialize loggers
    loggers = {
        name: MetricsLogger(
            os.path.join(DEFAULT_LOGS_DIR, f"metrics_{name}.csv"),
            name
        )
        for name in models
    }

    # History for plotting
    history = {name: {"t": [], "subspace_error": [], "explained_variance": []}
               for name in models}

    log_interval = max(1, args.n_steps // 50)

    print("Starting simulation...")
    print("-" * 60)

    start_time = time.time()

    for t, x in enumerate(stream.stream(args.n_steps), 1):
        # Update all models
        for name, model in models.items():
            model.partial_fit(x)

        # Log metrics periodically
        if t % log_interval == 0 or t == args.n_steps:
            elapsed = time.time() - start_time
            print(f"\nStep {t}/{args.n_steps} ({elapsed:.1f}s)")

            for name, model in models.items():
                W_est = model.components_
                sub_err = subspace_distance(W_est, W_ref)
                exp_var = explained_variance_ratio(X_test, W_est)
                recon_err = reconstruction_error(X_test, W_est)
                memory = model.get_memory_bytes()

                history[name]["t"].append(t)
                history[name]["subspace_error"].append(sub_err)
                history[name]["explained_variance"].append(exp_var)

                bits = getattr(model, "bits", None)
                loggers[name].log(
                    t=t,
                    subspace_error=sub_err,
                    explained_variance=exp_var,
                    reconstruction_error=recon_err,
                    memory_bytes=memory,
                    bits=bits,
                    eta_schedule=args.eta_schedule
                )

                print(f"  {name:<20} SubErr={sub_err:.6f}  ExpVar={exp_var:.4f}  "
                      f"Mem={format_memory(memory)}")

    # Close loggers
    for logger in loggers.values():
        logger.close()

    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"Simulation complete in {total_time:.1f}s")
    print("=" * 60)

    # Final summary
    print("\nFinal Metrics:")
    print("-" * 60)
    for name, model in models.items():
        W_est = model.components_
        sub_err = subspace_distance(W_est, W_ref)
        exp_var = explained_variance_ratio(X_test, W_est)
        recon_err = reconstruction_error(X_test, W_est)
        memory = model.get_memory_bytes()

        print(f"{name}:")
        print(f"  Subspace error: {sub_err:.6f}")
        print(f"  Explained variance: {exp_var:.4f}")
        print(f"  Reconstruction error: {recon_err:.6f}")
        print(f"  Memory: {format_memory(memory)}")

        bits = getattr(model, "bits", None)
        if bits is not None:
            print(f"  Quantization: {bits} bits, error_feedback=True")

    print(f"\nLogs saved to: {DEFAULT_LOGS_DIR}/")

    # Plot if requested
    if args.plot:
        print("\nGenerating plots...")

        # Create simple result objects for plotting
        from streaming_pca.benchmark import MethodResult
        plot_results = []
        for name, model in models.items():
            W_est = model.components_
            result = MethodResult(
                method=name,
                final_subspace_error=subspace_distance(W_est, W_ref),
                final_explained_variance=explained_variance_ratio(X_test, W_est),
                final_reconstruction_error=reconstruction_error(X_test, W_est),
                memory_bytes=model.get_memory_bytes(),
                runtime_seconds=0,
                history=history[name]
            )
            plot_results.append(result)

        plot_metrics_over_time(
            plot_results,
            "subspace_error",
            title="Subspace Error Over Time (Simulation)",
            save_path=os.path.join(DEFAULT_REPORTS_DIR, "simulate_convergence.png"),
            show=True
        )


def run_benchmark_mode(args: argparse.Namespace) -> None:
    """
    Run systematic benchmark and generate reports.

    Compares multiple methods across configurations and produces
    CSV reports and visualization plots.
    """
    print("=" * 60)
    print("Streaming PCA Benchmark")
    print("=" * 60)
    print(f"Dimensions: d={args.d}, k={args.k}")
    print(f"Steps: {args.steps}")
    print(f"Learning rate: eta0={args.eta0}, schedule={args.eta_schedule}")
    print(f"Quantization bits: {args.quant_bits}")
    print(f"Sketch size: {args.sketch_size}")
    print(f"Output directory: {args.reports}")
    print()

    # Create output directory
    ensure_dir(args.reports)
    ensure_dir(DEFAULT_LOGS_DIR)

    # Parse quant bits
    quant_bits = parse_quant_bits(args.quant_bits)

    # Create benchmark config
    config = BenchmarkConfig(
        d=args.d,
        k=args.k,
        n_steps=args.steps,
        eta0=args.eta0,
        eta_schedule=args.eta_schedule,
        quant_bits=quant_bits,
        sketch_size=args.sketch_size,
        noise_std=0.1,
        log_interval=max(1, args.steps // 40),
        test_size=1000,
        seed=args.seed
    )

    print("Running benchmark...")
    start_time = time.time()

    results = run_benchmark(config)

    total_time = time.time() - start_time
    print(f"\nBenchmark complete in {total_time:.1f}s")
    print()

    # Print results table
    print("Results Summary:")
    print("-" * 80)
    print(format_results_table(results))
    print()

    # Save results
    csv_path = os.path.join(args.reports, "tradeoff.csv")
    save_results_csv(results, csv_path)
    print(f"Results saved to: {csv_path}")

    # Save history
    history_path = os.path.join(DEFAULT_LOGS_DIR, "metrics.csv")
    save_history_csv(results, history_path)
    print(f"Metrics history saved to: {history_path}")

    # Generate plots
    print("\nGenerating plots...")
    figure_paths = create_report_figures(results, args.reports)
    for name, path in figure_paths.items():
        print(f"  {name}: {path}")

    # Print memory notes
    print("\nMemory Usage Notes:")
    print("-" * 60)
    for r in results:
        bits_info = f", {r.bits} bits" if r.bits else ""
        ef_info = ", error_feedback=True" if r.error_feedback else ""
        print(f"  {r.method}: {format_memory(r.memory_bytes)}{bits_info}{ef_info}")

    print(f"\nAll reports saved to: {args.reports}/")


def run_real_mode(args: argparse.Namespace) -> None:
    """
    Run streaming PCA on a real dataset.

    Loads a dataset and processes it as a stream, comparing
    streaming methods against batch PCA.
    """
    print("=" * 60)
    print(f"Streaming PCA on {args.dataset} Dataset")
    print("=" * 60)

    # Load dataset
    print(f"Loading {args.dataset} dataset...")
    X, y = load_dataset(args.dataset, seed=args.seed)
    n, d = X.shape
    print(f"Loaded {n} samples with {d} features")

    # Split data
    X_train, X_test = split_train_test(X, test_ratio=0.2, seed=args.seed)
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"Components: k={args.k}")
    print(f"Learning rate: eta0={args.eta0}, schedule={args.eta_schedule}")
    print()

    # Compute batch PCA reference
    W_ref = batch_pca_reference(X_train, args.k)
    ref_exp_var = explained_variance_ratio(X_test, W_ref)
    print(f"Batch PCA reference explained variance: {ref_exp_var:.4f}")
    print()

    # Create output directories
    ensure_dir(DEFAULT_LOGS_DIR)
    ensure_dir(DEFAULT_REPORTS_DIR)

    # Create dataset stream
    stream = DatasetStream(X_train, shuffle=True, seed=args.seed)

    # Determine number of steps (allow multiple passes)
    n_steps = min(args.n_steps, len(X_train) * 3)

    # Initialize models
    models = {
        "oja_fp": OjaPCA(
            d=d,
            k=args.k,
            eta0=args.eta0,
            eta_schedule=args.eta_schedule,
            seed=args.seed + 100
        ),
        "oja_quant_b8": QuantizedOjaPCA(
            d=d,
            k=args.k,
            bits=8,
            error_feedback=True,
            eta0=args.eta0,
            eta_schedule=args.eta_schedule,
            seed=args.seed + 200
        ),
        "oja_quant_b4": QuantizedOjaPCA(
            d=d,
            k=args.k,
            bits=4,
            error_feedback=True,
            eta0=args.eta0,
            eta_schedule=args.eta_schedule,
            seed=args.seed + 201
        ),
        "sketch_fd": SketchPCA(
            d=d,
            k=args.k,
            sketch_size=args.k * 2,
            seed=args.seed + 300
        )
    }

    # History for plotting
    history = {name: {"t": [], "subspace_error": [], "explained_variance": []}
               for name in models}

    log_interval = max(1, n_steps // 40)

    print(f"Running {n_steps} streaming steps...")
    print("-" * 60)

    start_time = time.time()

    for t, x in enumerate(stream.stream(n_steps), 1):
        for name, model in models.items():
            model.partial_fit(x)

        if t % log_interval == 0 or t == n_steps:
            print(f"\nStep {t}/{n_steps}")
            for name, model in models.items():
                W_est = model.components_
                sub_err = subspace_distance(W_est, W_ref)
                exp_var = explained_variance_ratio(X_test, W_est)

                history[name]["t"].append(t)
                history[name]["subspace_error"].append(sub_err)
                history[name]["explained_variance"].append(exp_var)

                print(f"  {name:<20} SubErr={sub_err:.6f}  ExpVar={exp_var:.4f}")

    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"Complete in {total_time:.1f}s")
    print("=" * 60)

    # Final summary
    print("\nFinal Metrics (vs Batch PCA Reference):")
    print("-" * 60)
    print(f"Batch PCA explained variance: {ref_exp_var:.4f}")
    print()

    for name, model in models.items():
        W_est = model.components_
        sub_err = subspace_distance(W_est, W_ref)
        exp_var = explained_variance_ratio(X_test, W_est)
        recon_err = reconstruction_error(X_test, W_est)
        memory = model.get_memory_bytes()

        print(f"{name}:")
        print(f"  Subspace error: {sub_err:.6f}")
        print(f"  Explained variance: {exp_var:.4f} (ref: {ref_exp_var:.4f})")
        print(f"  Reconstruction error: {recon_err:.6f}")
        print(f"  Memory: {format_memory(memory)}")

        bits = getattr(model, "bits", None)
        if bits is not None:
            print(f"  Quantization: {bits} bits, error_feedback=True")

    # Plot if requested
    if args.plot:
        print("\nGenerating plots...")

        from streaming_pca.benchmark import MethodResult
        plot_results = []
        for name, model in models.items():
            W_est = model.components_
            result = MethodResult(
                method=name,
                final_subspace_error=subspace_distance(W_est, W_ref),
                final_explained_variance=explained_variance_ratio(X_test, W_est),
                final_reconstruction_error=reconstruction_error(X_test, W_est),
                memory_bytes=model.get_memory_bytes(),
                runtime_seconds=0,
                history=history[name]
            )
            plot_results.append(result)

        plot_metrics_over_time(
            plot_results,
            "subspace_error",
            title=f"Subspace Error Over Time ({args.dataset})",
            save_path=os.path.join(DEFAULT_REPORTS_DIR, f"{args.dataset}_convergence.png"),
            show=True
        )


def main() -> None:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Streaming PCA Engine: Oja, Quantized Oja, and Sketching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Simulation mode:
    python -m src.cli --mode simulate --d 50 --k 5 --n-steps 20000 --eta-schedule invsqrt --eta0 0.5 --plot

  Benchmark mode:
    python -m src.cli --mode benchmark --d 100 --k 10 --steps 30000 --quant-bits 4,6,8 --sketch-size 40 --reports reports/

  Real dataset mode:
    python -m src.cli --mode real --dataset digits --k 10 --eta-schedule invsqrt --plot
        """
    )

    parser.add_argument(
        "--mode",
        choices=["simulate", "benchmark", "real"],
        required=True,
        help="Operation mode"
    )

    # Common arguments
    parser.add_argument("--d", type=int, default=100, help="Data dimensionality")
    parser.add_argument("--k", type=int, default=10, help="Number of principal components")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Learning rate arguments
    parser.add_argument("--eta0", type=float, default=0.5, help="Initial learning rate")
    parser.add_argument(
        "--eta-schedule",
        choices=["constant", "invsqrt", "invt"],
        default="invsqrt",
        help="Learning rate schedule"
    )

    # Simulation arguments
    parser.add_argument("--n-steps", type=int, default=20000, help="Number of streaming steps (simulate/real mode)")
    parser.add_argument("--plot", action="store_true", help="Show plots")
    parser.add_argument("--drift", action="store_true", help="Enable concept drift (simulate mode)")
    parser.add_argument("--drift-interval", type=int, default=10000, help="Steps between drift events")

    # Benchmark arguments
    parser.add_argument("--steps", type=int, default=30000, help="Number of steps (benchmark mode)")
    parser.add_argument("--quant-bits", type=str, default="4,6,8", help="Comma-separated quantization bits")
    parser.add_argument("--sketch-size", type=int, default=20, help="Sketch size for Frequent Directions")
    parser.add_argument("--reports", type=str, default="reports/", help="Output directory for reports")

    # Real dataset arguments
    parser.add_argument(
        "--dataset",
        choices=["digits", "synthetic", "random"],
        default="digits",
        help="Dataset to use (real mode)"
    )

    args = parser.parse_args()

    if args.mode == "simulate":
        run_simulate_mode(args)
    elif args.mode == "benchmark":
        run_benchmark_mode(args)
    elif args.mode == "real":
        run_real_mode(args)


if __name__ == "__main__":
    main()
