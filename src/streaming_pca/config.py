# Author: Emrullah Erce Dutkan
"""
Configuration management for streaming PCA experiments.

This module provides dataclasses and utilities for managing
experiment configurations, with sensible defaults that allow
the system to run out of the box.
"""

from typing import List, Optional, Literal
from dataclasses import dataclass, field, asdict


LearningRateSchedule = Literal["constant", "invsqrt", "invt"]
StreamType = Literal["gaussian", "drifting", "heavy_tailed"]


@dataclass
class OjaConfig:
    """Configuration for Oja's algorithm."""
    eta0: float = 0.5
    eta_schedule: LearningRateSchedule = "invsqrt"
    ortho_interval: int = 1
    grad_clip: Optional[float] = None
    normalize_by_variance: bool = False


@dataclass
class QuantConfig:
    """Configuration for quantized Oja."""
    bits: int = 8
    quant_target: str = "update"
    stochastic_rounding: bool = True
    error_feedback: bool = True


@dataclass
class SketchConfig:
    """Configuration for Frequent Directions sketching."""
    sketch_size: Optional[int] = None  # If None, uses 2*k


@dataclass
class StreamConfig:
    """Configuration for synthetic data streams."""
    stream_type: StreamType = "gaussian"
    noise_std: float = 0.1
    decay: str = "exponential"
    decay_rate: float = 0.5
    drift_interval: Optional[int] = None
    drift_angle: float = 0.1
    heavy_tail_df: float = 3.0


@dataclass
class ExperimentConfig:
    """
    Complete configuration for a streaming PCA experiment.

    This dataclass holds all parameters needed to run a benchmark
    or simulation. Defaults are chosen to produce reasonable results
    without manual tuning.
    """
    # Data dimensions
    d: int = 100
    k: int = 10

    # Streaming parameters
    n_steps: int = 20000
    log_interval: int = 500
    test_size: int = 1000

    # Method configurations
    oja: OjaConfig = field(default_factory=OjaConfig)
    quant: QuantConfig = field(default_factory=QuantConfig)
    sketch: SketchConfig = field(default_factory=SketchConfig)

    # Stream configuration
    stream: StreamConfig = field(default_factory=StreamConfig)

    # Benchmark settings
    quant_bits_list: List[int] = field(default_factory=lambda: [4, 6, 8])

    # Reproducibility
    seed: int = 42

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        """Create from dictionary."""
        # Handle nested configs
        if "oja" in d and isinstance(d["oja"], dict):
            d["oja"] = OjaConfig(**d["oja"])
        if "quant" in d and isinstance(d["quant"], dict):
            d["quant"] = QuantConfig(**d["quant"])
        if "sketch" in d and isinstance(d["sketch"], dict):
            d["sketch"] = SketchConfig(**d["sketch"])
        if "stream" in d and isinstance(d["stream"], dict):
            d["stream"] = StreamConfig(**d["stream"])
        return cls(**d)


def get_default_config() -> ExperimentConfig:
    """Get default experiment configuration."""
    return ExperimentConfig()


def get_quick_config() -> ExperimentConfig:
    """Get configuration for quick testing (smaller scale)."""
    return ExperimentConfig(
        d=50,
        k=5,
        n_steps=5000,
        log_interval=200,
        test_size=500
    )


def get_benchmark_config() -> ExperimentConfig:
    """Get configuration for thorough benchmarking."""
    return ExperimentConfig(
        d=100,
        k=10,
        n_steps=30000,
        log_interval=500,
        test_size=2000,
        quant_bits_list=[4, 6, 8],
        stream=StreamConfig(noise_std=0.1)
    )


def get_drift_config() -> ExperimentConfig:
    """Get configuration for concept drift experiments."""
    return ExperimentConfig(
        d=100,
        k=10,
        n_steps=40000,
        log_interval=500,
        test_size=1000,
        stream=StreamConfig(
            stream_type="drifting",
            drift_interval=10000,
            drift_angle=0.2
        )
    )


# Default output paths
DEFAULT_REPORTS_DIR = "reports"
DEFAULT_LOGS_DIR = "logs"
DEFAULT_METRICS_FILE = "logs/metrics.csv"
DEFAULT_TRADEOFF_FILE = "reports/tradeoff.csv"
