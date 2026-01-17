# Author: Emrullah Erce Dutkan
"""
Input/Output utilities for streaming PCA experiments.

This module provides functions for:
- Saving and loading benchmark results
- Logging metrics during streaming
- Creating report files
"""

from typing import List, Dict, Any, Optional
import os
import csv
import json
from datetime import datetime

from .benchmark import MethodResult, results_to_dict


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def save_results_csv(
    results: List[MethodResult],
    path: str
) -> None:
    """
    Save benchmark results to CSV.

    Args:
        results: List of MethodResult.
        path: Output CSV path.
    """
    ensure_dir(os.path.dirname(path) or ".")

    rows = results_to_dict(results)
    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_history_csv(
    results: List[MethodResult],
    path: str
) -> None:
    """
    Save metric history over time to CSV.

    Creates a long-format CSV with columns: t, method, metric, value

    Args:
        results: List of MethodResult with history.
        path: Output CSV path.
    """
    ensure_dir(os.path.dirname(path) or ".")

    rows = []
    for r in results:
        if not r.history:
            continue

        t_vals = r.history.get("t", [])
        for metric in ["subspace_error", "explained_variance", "reconstruction_error"]:
            if metric in r.history:
                for t, val in zip(t_vals, r.history[metric]):
                    rows.append({
                        "t": t,
                        "method": r.method,
                        "metric": metric,
                        "value": val
                    })

    if not rows:
        return

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["t", "method", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def load_results_csv(path: str) -> List[Dict[str, Any]]:
    """
    Load benchmark results from CSV.

    Args:
        path: Input CSV path.

    Returns:
        List of result dictionaries.
    """
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)


class MetricsLogger:
    """
    Logger for streaming metrics to a CSV file.

    Logs metrics incrementally during a streaming run, writing
    to disk periodically for fault tolerance.
    """

    def __init__(
        self,
        path: str,
        method: str,
        buffer_size: int = 100
    ):
        """
        Initialize the logger.

        Args:
            path: Output CSV path.
            method: Method name for this run.
            buffer_size: Write to disk every N entries.
        """
        self.path = path
        self.method = method
        self.buffer_size = buffer_size

        self.buffer: List[Dict[str, Any]] = []
        self.fieldnames = [
            "timestamp", "t", "method", "subspace_error",
            "explained_variance", "reconstruction_error",
            "memory_bytes", "bits", "eta_schedule"
        ]

        ensure_dir(os.path.dirname(path) or ".")

        # Write header if file doesn't exist
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(
        self,
        t: int,
        subspace_error: float,
        explained_variance: float,
        reconstruction_error: float,
        memory_bytes: int,
        bits: Optional[int] = None,
        eta_schedule: str = ""
    ) -> None:
        """
        Log a single metric entry.

        Args:
            t: Current step.
            subspace_error: Subspace error value.
            explained_variance: Explained variance value.
            reconstruction_error: Reconstruction error value.
            memory_bytes: Memory usage.
            bits: Quantization bits (if applicable).
            eta_schedule: Learning rate schedule name.
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "t": t,
            "method": self.method,
            "subspace_error": subspace_error,
            "explained_variance": explained_variance,
            "reconstruction_error": reconstruction_error,
            "memory_bytes": memory_bytes,
            "bits": bits if bits is not None else "",
            "eta_schedule": eta_schedule
        }
        self.buffer.append(entry)

        if len(self.buffer) >= self.buffer_size:
            self.flush()

    def flush(self) -> None:
        """Write buffered entries to disk."""
        if not self.buffer:
            return

        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerows(self.buffer)

        self.buffer = []

    def close(self) -> None:
        """Flush remaining entries and close."""
        self.flush()


def save_config(config: Dict[str, Any], path: str) -> None:
    """
    Save configuration to JSON.

    Args:
        config: Configuration dictionary.
        path: Output JSON path.
    """
    ensure_dir(os.path.dirname(path) or ".")

    with open(path, "w") as f:
        json.dump(config, f, indent=2, default=str)


def load_config(path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON.

    Args:
        path: Input JSON path.

    Returns:
        Configuration dictionary.
    """
    with open(path, "r") as f:
        return json.load(f)


def format_memory(nbytes: int) -> str:
    """
    Format memory size in human-readable form.

    Args:
        nbytes: Number of bytes.

    Returns:
        Formatted string (e.g., "1.5 KB", "2.3 MB").
    """
    if nbytes < 1024:
        return f"{nbytes} B"
    elif nbytes < 1024 ** 2:
        return f"{nbytes / 1024:.1f} KB"
    elif nbytes < 1024 ** 3:
        return f"{nbytes / 1024**2:.1f} MB"
    else:
        return f"{nbytes / 1024**3:.1f} GB"


def create_summary_report(
    results: List[MethodResult],
    config: Dict[str, Any],
    output_dir: str
) -> str:
    """
    Create a summary report as text.

    Args:
        results: Benchmark results.
        config: Configuration used.
        output_dir: Directory where results are saved.

    Returns:
        Summary text.
    """
    lines = [
        "Streaming PCA Benchmark Report",
        "=" * 40,
        "",
        "Configuration:",
        f"  Dimensionality (d): {config.get('d', 'N/A')}",
        f"  Components (k): {config.get('k', 'N/A')}",
        f"  Streaming steps: {config.get('n_steps', 'N/A')}",
        f"  Learning rate (eta0): {config.get('eta0', 'N/A')}",
        f"  Schedule: {config.get('eta_schedule', 'N/A')}",
        f"  Seed: {config.get('seed', 'N/A')}",
        "",
        "Results:",
        "-" * 40
    ]

    for r in results:
        lines.append(f"\nMethod: {r.method}")
        lines.append(f"  Subspace Error: {r.final_subspace_error:.6f}")
        lines.append(f"  Explained Variance: {r.final_explained_variance:.4f}")
        lines.append(f"  Reconstruction Error: {r.final_reconstruction_error:.6f}")
        lines.append(f"  Memory: {format_memory(r.memory_bytes)}")
        lines.append(f"  Runtime: {r.runtime_seconds:.2f}s")

        if r.bits is not None:
            lines.append(f"  Quantization: {r.bits} bits")
        if r.error_feedback:
            lines.append("  Error feedback: enabled")

    lines.extend([
        "",
        "-" * 40,
        f"Output files in: {output_dir}",
        "  - tradeoff.csv: Summary metrics",
        "  - metrics.csv: Metrics over time",
        "  - tradeoff_accuracy_vs_memory.png: Memory-accuracy plot",
        "  - metric_over_time.png: Convergence plot"
    ])

    return "\n".join(lines)
