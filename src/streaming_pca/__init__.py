# Author: Emrullah Erce Dutkan
"""
Streaming PCA Engine

A library for streaming principal component analysis using:
- Oja's algorithm for online learning of top-k eigenvectors
- Quantized Oja for low-precision streaming PCA
- Frequent Directions sketching as a memory-efficient baseline

This package provides implementations, benchmarking tools, and evaluation
metrics for comparing streaming PCA methods in edge and streaming settings.
"""

from .oja import OjaPCA, oja_pca, OnlineMeanEstimator
from .quant import QuantizedOjaPCA, quantized_oja_pca, uniform_quantize
from .sketch import SketchPCA, FrequentDirections, frequent_directions_pca
from .metrics import (
    subspace_distance,
    explained_variance_ratio,
    reconstruction_error,
    normalized_reconstruction_error,
    compute_all_metrics,
    batch_pca_reference,
    principal_angles
)
from .stream import (
    SyntheticStream,
    HeavyTailedStream,
    DatasetStream,
    create_stream
)
from .datasets import load_dataset, load_digits, load_synthetic
from .benchmark import BenchmarkConfig, run_benchmark, MethodResult
from .config import ExperimentConfig, get_default_config

__version__ = "0.1.0"
__author__ = "Emrullah Erce Dutkan"

__all__ = [
    # Core algorithms
    "OjaPCA",
    "QuantizedOjaPCA",
    "SketchPCA",
    "FrequentDirections",
    # Convenience functions
    "oja_pca",
    "quantized_oja_pca",
    "frequent_directions_pca",
    # Utilities
    "OnlineMeanEstimator",
    "uniform_quantize",
    # Metrics
    "subspace_distance",
    "explained_variance_ratio",
    "reconstruction_error",
    "normalized_reconstruction_error",
    "compute_all_metrics",
    "batch_pca_reference",
    "principal_angles",
    # Streams
    "SyntheticStream",
    "HeavyTailedStream",
    "DatasetStream",
    "create_stream",
    # Datasets
    "load_dataset",
    "load_digits",
    "load_synthetic",
    # Benchmarking
    "BenchmarkConfig",
    "run_benchmark",
    "MethodResult",
    # Configuration
    "ExperimentConfig",
    "get_default_config",
]
