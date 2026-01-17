# Author: Emrullah Erce Dutkan
"""
Benchmarking framework for streaming PCA algorithms.

This module provides tools for running systematic comparisons of
streaming PCA methods across different configurations:
- Oja's algorithm (full precision)
- Quantized Oja (various bit widths)
- Frequent Directions sketching

The benchmark measures:
- Subspace error (vs batch PCA reference)
- Explained variance ratio
- Reconstruction error
- Memory usage
- Runtime

Results are saved to CSV files and can be visualized with the viz module.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import time
import numpy as np

from .oja import OjaPCA, LearningRateSchedule
from .quant import QuantizedOjaPCA
from .sketch import SketchPCA
from .metrics import (
    subspace_distance,
    explained_variance_ratio,
    reconstruction_error,
    batch_pca_reference
)
from .stream import SyntheticStream, DatasetStream


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    d: int = 100
    k: int = 10
    n_steps: int = 20000
    eta0: float = 0.5
    eta_schedule: LearningRateSchedule = "invsqrt"
    quant_bits: List[int] = field(default_factory=lambda: [4, 6, 8])
    sketch_size: int = 20
    noise_std: float = 0.1
    log_interval: int = 500
    test_size: int = 1000
    seed: int = 42
    drift_interval: Optional[int] = None
    drift_angle: float = 0.1


@dataclass
class MethodResult:
    """Results for a single method run."""
    method: str
    final_subspace_error: float
    final_explained_variance: float
    final_reconstruction_error: float
    memory_bytes: int
    runtime_seconds: float
    bits: Optional[int] = None
    eta_schedule: str = ""
    k: int = 0
    sketch_size: Optional[int] = None
    error_feedback: bool = False
    history: Dict[str, List[float]] = field(default_factory=dict)


def run_single_method(
    method_name: str,
    model: Any,
    stream: Any,
    n_steps: int,
    k: int,
    log_interval: int,
    X_test: np.ndarray,
    W_ref: np.ndarray
) -> MethodResult:
    """
    Run a single streaming PCA method and collect metrics.

    Args:
        method_name: Name for logging.
        model: Streaming PCA model with partial_fit method.
        stream: Data stream generator.
        n_steps: Number of streaming steps.
        k: Number of components.
        log_interval: Log metrics every N steps.
        X_test: Test data for evaluation.
        W_ref: Reference PCA components.

    Returns:
        MethodResult with final metrics and history.
    """
    history = {
        "t": [],
        "subspace_error": [],
        "explained_variance": [],
        "reconstruction_error": []
    }

    start_time = time.time()

    for t, x in enumerate(stream.stream(n_steps), 1):
        model.partial_fit(x)

        if t % log_interval == 0 or t == n_steps:
            W_est = model.components_

            sub_err = subspace_distance(W_est, W_ref)
            exp_var = explained_variance_ratio(X_test, W_est)
            recon_err = reconstruction_error(X_test, W_est)

            history["t"].append(t)
            history["subspace_error"].append(sub_err)
            history["explained_variance"].append(exp_var)
            history["reconstruction_error"].append(recon_err)

    runtime = time.time() - start_time

    # Final metrics
    W_final = model.components_
    final_sub_err = subspace_distance(W_final, W_ref)
    final_exp_var = explained_variance_ratio(X_test, W_final)
    final_recon_err = reconstruction_error(X_test, W_final)
    memory = model.get_memory_bytes()

    # Extract config info
    bits = getattr(model, "bits", None)
    eta_schedule = getattr(model, "eta_schedule", "")
    sketch_size = getattr(model, "sketch_size", None)
    error_feedback = getattr(model, "error_feedback", False)

    return MethodResult(
        method=method_name,
        final_subspace_error=final_sub_err,
        final_explained_variance=final_exp_var,
        final_reconstruction_error=final_recon_err,
        memory_bytes=memory,
        runtime_seconds=runtime,
        bits=bits,
        eta_schedule=eta_schedule,
        k=k,
        sketch_size=sketch_size,
        error_feedback=error_feedback,
        history=history
    )


def run_benchmark(config: BenchmarkConfig) -> List[MethodResult]:
    """
    Run full benchmark suite comparing multiple methods.

    Args:
        config: Benchmark configuration.

    Returns:
        List of MethodResult for each method.
    """
    results = []

    # Create synthetic stream
    stream = SyntheticStream(
        d=config.d,
        rank=config.k * 2,  # True rank higher than k
        noise_std=config.noise_std,
        seed=config.seed,
        drift_interval=config.drift_interval,
        drift_angle=config.drift_angle
    )

    # Generate test data (separate from stream)
    test_stream = SyntheticStream(
        d=config.d,
        rank=config.k * 2,
        noise_std=config.noise_std,
        seed=config.seed + 1000
    )
    X_test = test_stream.get_batch(config.test_size)

    # Compute batch PCA reference
    ref_stream = SyntheticStream(
        d=config.d,
        rank=config.k * 2,
        noise_std=config.noise_std,
        seed=config.seed + 2000
    )
    X_ref = ref_stream.get_batch(config.n_steps)
    W_ref = batch_pca_reference(X_ref, config.k)

    # Method 1: Full precision Oja
    stream.reset(seed=config.seed)
    model_fp = OjaPCA(
        d=config.d,
        k=config.k,
        eta0=config.eta0,
        eta_schedule=config.eta_schedule,
        seed=config.seed + 100
    )
    result_fp = run_single_method(
        "oja_fp",
        model_fp,
        stream,
        config.n_steps,
        config.k,
        config.log_interval,
        X_test,
        W_ref
    )
    results.append(result_fp)

    # Method 2: Quantized Oja at various bit widths
    for bits in config.quant_bits:
        stream.reset(seed=config.seed)
        model_q = QuantizedOjaPCA(
            d=config.d,
            k=config.k,
            bits=bits,
            error_feedback=True,
            eta0=config.eta0,
            eta_schedule=config.eta_schedule,
            seed=config.seed + 200 + bits
        )
        result_q = run_single_method(
            f"oja_quant_b{bits}",
            model_q,
            stream,
            config.n_steps,
            config.k,
            config.log_interval,
            X_test,
            W_ref
        )
        results.append(result_q)

    # Method 3: Frequent Directions sketching
    stream.reset(seed=config.seed)
    model_fd = SketchPCA(
        d=config.d,
        k=config.k,
        sketch_size=config.sketch_size,
        seed=config.seed + 300
    )
    result_fd = run_single_method(
        "sketch_fd",
        model_fd,
        stream,
        config.n_steps,
        config.k,
        config.log_interval,
        X_test,
        W_ref
    )
    results.append(result_fd)

    return results


def run_memory_sweep(
    config: BenchmarkConfig,
    sketch_sizes: List[int]
) -> List[MethodResult]:
    """
    Run benchmark sweeping sketch sizes for memory-accuracy tradeoff.

    Args:
        config: Base benchmark configuration.
        sketch_sizes: List of sketch sizes to test.

    Returns:
        List of MethodResult.
    """
    results = []

    # Create synthetic stream
    stream = SyntheticStream(
        d=config.d,
        rank=config.k * 2,
        noise_std=config.noise_std,
        seed=config.seed
    )

    # Test and reference data
    test_stream = SyntheticStream(
        d=config.d,
        rank=config.k * 2,
        noise_std=config.noise_std,
        seed=config.seed + 1000
    )
    X_test = test_stream.get_batch(config.test_size)

    ref_stream = SyntheticStream(
        d=config.d,
        rank=config.k * 2,
        noise_std=config.noise_std,
        seed=config.seed + 2000
    )
    X_ref = ref_stream.get_batch(config.n_steps)
    W_ref = batch_pca_reference(X_ref, config.k)

    for sketch_size in sketch_sizes:
        if sketch_size < config.k:
            continue

        stream.reset(seed=config.seed)
        model = SketchPCA(
            d=config.d,
            k=config.k,
            sketch_size=sketch_size,
            seed=config.seed + 300 + sketch_size
        )
        result = run_single_method(
            f"sketch_fd_l{sketch_size}",
            model,
            stream,
            config.n_steps,
            config.k,
            config.log_interval,
            X_test,
            W_ref
        )
        results.append(result)

    return results


def run_dataset_benchmark(
    X: np.ndarray,
    k: int,
    n_steps: int,
    config: BenchmarkConfig
) -> List[MethodResult]:
    """
    Run benchmark on a real dataset.

    Args:
        X: Data matrix.
        k: Number of components.
        n_steps: Number of streaming steps.
        config: Benchmark configuration.

    Returns:
        List of MethodResult.
    """
    from .datasets import split_train_test

    results = []
    d = X.shape[1]

    # Split into train and test
    X_train, X_test = split_train_test(X, test_ratio=0.2, seed=config.seed)

    # Compute batch PCA reference on full training data
    W_ref = batch_pca_reference(X_train, k)

    # Create dataset stream
    stream = DatasetStream(X_train, shuffle=True, seed=config.seed)

    # Limit steps to available data (allow multiple passes)
    effective_steps = min(n_steps, len(X_train) * 3)

    # Full precision Oja
    stream.reset(seed=config.seed)
    model_fp = OjaPCA(
        d=d,
        k=k,
        eta0=config.eta0,
        eta_schedule=config.eta_schedule,
        seed=config.seed + 100
    )
    result_fp = run_single_method(
        "oja_fp",
        model_fp,
        stream,
        effective_steps,
        k,
        config.log_interval,
        X_test,
        W_ref
    )
    results.append(result_fp)

    # Quantized Oja
    for bits in config.quant_bits:
        stream.reset(seed=config.seed)
        model_q = QuantizedOjaPCA(
            d=d,
            k=k,
            bits=bits,
            error_feedback=True,
            eta0=config.eta0,
            eta_schedule=config.eta_schedule,
            seed=config.seed + 200 + bits
        )
        result_q = run_single_method(
            f"oja_quant_b{bits}",
            model_q,
            stream,
            effective_steps,
            k,
            config.log_interval,
            X_test,
            W_ref
        )
        results.append(result_q)

    # Frequent Directions
    stream.reset(seed=config.seed)
    model_fd = SketchPCA(
        d=d,
        k=k,
        sketch_size=config.sketch_size,
        seed=config.seed + 300
    )
    result_fd = run_single_method(
        "sketch_fd",
        model_fd,
        stream,
        effective_steps,
        k,
        config.log_interval,
        X_test,
        W_ref
    )
    results.append(result_fd)

    return results


def results_to_dict(results: List[MethodResult]) -> List[Dict[str, Any]]:
    """Convert results to list of dictionaries for CSV export."""
    rows = []
    for r in results:
        row = {
            "method": r.method,
            "subspace_error": r.final_subspace_error,
            "explained_variance": r.final_explained_variance,
            "reconstruction_error": r.final_reconstruction_error,
            "memory_bytes": r.memory_bytes,
            "runtime_seconds": r.runtime_seconds,
            "bits": r.bits if r.bits is not None else "",
            "eta_schedule": r.eta_schedule,
            "k": r.k,
            "sketch_size": r.sketch_size if r.sketch_size is not None else "",
            "error_feedback": r.error_feedback
        }
        rows.append(row)
    return rows


def format_results_table(results: List[MethodResult]) -> str:
    """Format results as a text table for console output."""
    lines = []
    header = (
        f"{'Method':<20} {'SubErr':>10} {'ExpVar':>10} "
        f"{'ReconErr':>12} {'Memory':>12} {'Time':>8}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for r in results:
        mem_str = f"{r.memory_bytes:,}"
        line = (
            f"{r.method:<20} {r.final_subspace_error:>10.6f} "
            f"{r.final_explained_variance:>10.4f} "
            f"{r.final_reconstruction_error:>12.6f} "
            f"{mem_str:>12} {r.runtime_seconds:>8.2f}s"
        )
        lines.append(line)

    return "\n".join(lines)
