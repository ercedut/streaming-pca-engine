# Author: Emrullah Erce Dutkan
"""
Visualization utilities for streaming PCA experiments.

This module provides plotting functions for:
- Metrics over time (convergence curves)
- Memory vs accuracy tradeoffs
- Method comparisons
"""

from typing import List, Optional, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from .benchmark import MethodResult


def setup_style() -> None:
    """Configure matplotlib style for clean plots."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "lines.linewidth": 1.5,
        "lines.markersize": 6
    })


def plot_metrics_over_time(
    results: List[MethodResult],
    metric: str = "subspace_error",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True
) -> Figure:
    """
    Plot a metric over time for multiple methods.

    Args:
        results: List of MethodResult with history.
        metric: Which metric to plot ("subspace_error", "explained_variance",
                "reconstruction_error").
        title: Plot title.
        save_path: Path to save figure.
        show: Whether to display the plot.

    Returns:
        Matplotlib figure.
    """
    setup_style()
    fig, ax = plt.subplots()

    for result in results:
        if metric in result.history and result.history[metric]:
            t = result.history["t"]
            values = result.history[metric]
            ax.plot(t, values, label=result.method, marker="o", markersize=3)

    ax.set_xlabel("Samples processed")

    metric_labels = {
        "subspace_error": "Subspace Error (mean sin theta)",
        "explained_variance": "Explained Variance Ratio",
        "reconstruction_error": "Reconstruction Error (MSE)"
    }
    ax.set_ylabel(metric_labels.get(metric, metric))

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"{metric_labels.get(metric, metric)} vs Samples")

    ax.legend(loc="best")

    if metric == "subspace_error" or metric == "reconstruction_error":
        ax.set_yscale("log")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_memory_accuracy_tradeoff(
    results: List[MethodResult],
    metric: str = "subspace_error",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True
) -> Figure:
    """
    Plot memory vs accuracy tradeoff.

    Args:
        results: List of MethodResult.
        metric: Which accuracy metric to use.
        title: Plot title.
        save_path: Path to save figure.
        show: Whether to display.

    Returns:
        Matplotlib figure.
    """
    setup_style()
    fig, ax = plt.subplots()

    # Group by method type
    method_groups = {}
    for r in results:
        base_method = r.method.split("_")[0]
        if base_method not in method_groups:
            method_groups[base_method] = {"memory": [], "accuracy": [], "labels": []}
        method_groups[base_method]["memory"].append(r.memory_bytes)

        if metric == "subspace_error":
            method_groups[base_method]["accuracy"].append(r.final_subspace_error)
        elif metric == "explained_variance":
            method_groups[base_method]["accuracy"].append(r.final_explained_variance)
        else:
            method_groups[base_method]["accuracy"].append(r.final_reconstruction_error)

        method_groups[base_method]["labels"].append(r.method)

    markers = ["o", "s", "^", "D", "v", "<", ">"]
    colors = plt.cm.tab10.colors

    for i, (method_type, data) in enumerate(method_groups.items()):
        ax.scatter(
            data["memory"],
            data["accuracy"],
            label=method_type,
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            s=80
        )

        # Add labels
        for mem, acc, label in zip(data["memory"], data["accuracy"], data["labels"]):
            ax.annotate(
                label,
                (mem, acc),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8
            )

    ax.set_xlabel("Memory (bytes)")

    metric_labels = {
        "subspace_error": "Subspace Error",
        "explained_variance": "Explained Variance",
        "reconstruction_error": "Reconstruction Error"
    }
    ax.set_ylabel(metric_labels.get(metric, metric))

    if title:
        ax.set_title(title)
    else:
        ax.set_title("Memory vs Accuracy Tradeoff")

    ax.set_xscale("log")
    if metric in ("subspace_error", "reconstruction_error"):
        ax.set_yscale("log")

    ax.legend(loc="best")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_comparison_bars(
    results: List[MethodResult],
    metrics: List[str] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True
) -> Figure:
    """
    Plot bar chart comparing methods on multiple metrics.

    Args:
        results: List of MethodResult.
        metrics: List of metrics to compare.
        title: Plot title.
        save_path: Path to save figure.
        show: Whether to display.

    Returns:
        Matplotlib figure.
    """
    if metrics is None:
        metrics = ["subspace_error", "explained_variance"]

    setup_style()

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    methods = [r.method for r in results]
    x = np.arange(len(methods))
    width = 0.6

    metric_attrs = {
        "subspace_error": "final_subspace_error",
        "explained_variance": "final_explained_variance",
        "reconstruction_error": "final_reconstruction_error",
        "memory_bytes": "memory_bytes",
        "runtime_seconds": "runtime_seconds"
    }

    metric_labels = {
        "subspace_error": "Subspace Error",
        "explained_variance": "Explained Variance",
        "reconstruction_error": "Reconstruction Error",
        "memory_bytes": "Memory (bytes)",
        "runtime_seconds": "Runtime (s)"
    }

    for ax, metric in zip(axes, metrics):
        values = [getattr(r, metric_attrs.get(metric, metric)) for r in results]
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(methods)))

        bars = ax.bar(x, values, width, color=colors)
        ax.set_ylabel(metric_labels.get(metric, metric))
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha="right")

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(
                f"{val:.4f}" if val < 100 else f"{val:.0f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8
            )

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_convergence_comparison(
    results: List[MethodResult],
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True
) -> Figure:
    """
    Plot convergence curves for all metrics in subplots.

    Args:
        results: List of MethodResult with history.
        title: Overall title.
        save_path: Path to save figure.
        show: Whether to display.

    Returns:
        Matplotlib figure.
    """
    setup_style()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    metrics = [
        ("subspace_error", "Subspace Error", True),
        ("explained_variance", "Explained Variance", False),
        ("reconstruction_error", "Reconstruction Error", True)
    ]

    for ax, (metric, label, use_log) in zip(axes[:3], metrics):
        for result in results:
            if metric in result.history and result.history[metric]:
                t = result.history["t"]
                values = result.history[metric]
                ax.plot(t, values, label=result.method, marker="o", markersize=2)

        ax.set_xlabel("Samples")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(loc="best", fontsize=8)

        if use_log:
            ax.set_yscale("log")

    # Use fourth subplot for memory comparison
    methods = [r.method for r in results]
    memories = [r.memory_bytes for r in results]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(methods)))

    axes[3].barh(methods, memories, color=colors)
    axes[3].set_xlabel("Memory (bytes)")
    axes[3].set_title("Memory Usage")

    if title:
        fig.suptitle(title, fontsize=14, y=1.02)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def create_report_figures(
    results: List[MethodResult],
    output_dir: str
) -> Dict[str, str]:
    """
    Create all report figures and save them.

    Args:
        results: Benchmark results.
        output_dir: Directory to save figures.

    Returns:
        Dictionary mapping figure names to file paths.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    paths = {}

    # Convergence plot
    path = os.path.join(output_dir, "metric_over_time.png")
    plot_metrics_over_time(results, "subspace_error", save_path=path, show=False)
    paths["metric_over_time"] = path

    # Memory-accuracy tradeoff
    path = os.path.join(output_dir, "tradeoff_accuracy_vs_memory.png")
    plot_memory_accuracy_tradeoff(results, save_path=path, show=False)
    paths["memory_accuracy"] = path

    # Full convergence comparison
    path = os.path.join(output_dir, "convergence_comparison.png")
    plot_convergence_comparison(results, save_path=path, show=False)
    paths["convergence"] = path

    return paths
