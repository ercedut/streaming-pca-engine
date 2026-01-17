# Author: Emrullah Erce Dutkan
"""
Quantized Oja's algorithm for low-precision streaming PCA.

This module implements a quantized variant of Oja's algorithm that reduces
memory bandwidth and storage requirements by quantizing updates and/or
weights to low bit-widths (e.g., 4, 6, or 8 bits).

Quantization strategy:
- Master weights W are kept in full precision (float64) for accumulation
- The gradient update delta_W can be quantized before accumulation
- Stochastic rounding is supported for unbiased quantization
- Error feedback (residual accumulation) compensates for quantization error

This is useful for:
- Edge devices with limited memory bandwidth
- Distributed streaming where communication is expensive
- Studying the robustness of Oja's algorithm to precision loss
"""

from typing import Optional, Literal, Tuple
import numpy as np

from .oja import OjaPCA, OnlineMeanEstimator, compute_learning_rate, LearningRateSchedule


QuantizationTarget = Literal["update", "input", "both"]


def uniform_quantize(
    x: np.ndarray,
    bits: int,
    scale: Optional[float] = None,
    stochastic: bool = True,
    rng: Optional[np.random.Generator] = None
) -> Tuple[np.ndarray, float]:
    """
    Uniform symmetric quantization to b bits.

    Maps values to integers in [-2^(b-1), 2^(b-1) - 1], then back to floats.

    Args:
        x: Array to quantize.
        bits: Number of bits (e.g., 4, 6, 8).
        scale: Quantization scale. If None, uses max absolute value.
        stochastic: If True, use stochastic rounding (unbiased).
        rng: Random generator for stochastic rounding.

    Returns:
        Tuple of (quantized array, scale used).

    The quantization scheme is:
        q = round(x / scale * (2^(b-1) - 1))
        x_hat = q * scale / (2^(b-1) - 1)

    Stochastic rounding:
        Instead of deterministic rounding, we round up with probability
        equal to the fractional part, making E[round(x)] = x.
    """
    if bits < 1 or bits > 16:
        raise ValueError(f"bits must be in [1, 16], got {bits}")

    # Number of quantization levels (symmetric around zero)
    n_levels = (1 << (bits - 1)) - 1  # 2^(b-1) - 1

    # Compute scale from max absolute value if not provided
    if scale is None:
        max_abs = np.max(np.abs(x))
        if max_abs < 1e-10:
            # All zeros or near-zero, return as-is
            return x.copy(), 1.0
        scale = max_abs

    # Scale to [-n_levels, n_levels]
    scaled = x / scale * n_levels

    if stochastic and rng is not None:
        # Stochastic rounding: round up with probability = fractional part
        floor_val = np.floor(scaled)
        frac = scaled - floor_val
        # Generate uniform random values
        rand = rng.random(scaled.shape)
        # Round up if rand < frac, else round down
        quantized = np.where(rand < frac, floor_val + 1, floor_val)
    else:
        # Deterministic rounding
        quantized = np.round(scaled)

    # Clip to valid range
    quantized = np.clip(quantized, -n_levels, n_levels)

    # Dequantize back to float
    dequantized = quantized * scale / n_levels

    return dequantized, scale


def compute_quantization_memory(d: int, k: int, bits: int) -> int:
    """
    Compute memory for quantized weight storage.

    If weights were actually stored in low precision:
        d * k * bits / 8 bytes

    Note: In this implementation, we keep master weights in float64
    for numerical stability, so actual memory is higher. This function
    reports the theoretical low-precision storage requirement.
    """
    return int(np.ceil(d * k * bits / 8))


class QuantizedOjaPCA(OjaPCA):
    """
    Quantized variant of Oja's streaming PCA.

    This class extends OjaPCA with quantization of the gradient update.
    The key insight is that Oja's update W <- W + eta * x * (x^T W)
    involves adding a rank-1 update to W. We can quantize this update
    to reduce memory bandwidth in distributed or memory-constrained settings.

    Quantization targets:
    - "update": Quantize the gradient delta_W before adding to W
    - "input": Quantize the input sample x before computing the update
    - "both": Quantize both input and update

    Error feedback mechanism:
    When error_feedback=True, we accumulate the quantization residual:
        r <- r + delta_W - Q(delta_W + r)
        W <- W + Q(delta_W + r)

    This ensures that quantization errors are eventually corrected,
    reducing bias at the cost of slight additional computation.

    Stability notes:
    - Master weights are always kept in float64 for numerical stability
    - Quantization is applied to the update path, not the stored weights
    - Very low bit widths (< 4) may cause instability without error feedback
    - Stochastic rounding is recommended for unbiased gradients
    """

    def __init__(
        self,
        d: int,
        k: int,
        bits: int = 8,
        quant_target: QuantizationTarget = "update",
        stochastic_rounding: bool = True,
        error_feedback: bool = True,
        eta0: float = 0.1,
        eta_schedule: LearningRateSchedule = "invsqrt",
        ortho_interval: int = 1,
        grad_clip: Optional[float] = None,
        normalize_by_variance: bool = False,
        seed: Optional[int] = None
    ):
        """
        Initialize quantized Oja's PCA.

        Args:
            d: Data dimensionality.
            k: Number of principal components.
            bits: Quantization bit width (e.g., 4, 6, 8).
            quant_target: What to quantize ("update", "input", "both").
            stochastic_rounding: Use stochastic rounding for unbiased quantization.
            error_feedback: Accumulate and correct quantization residuals.
            eta0: Initial learning rate.
            eta_schedule: Learning rate schedule.
            ortho_interval: Re-orthonormalize every N steps.
            grad_clip: Gradient clipping threshold.
            normalize_by_variance: Normalize by running variance.
            seed: Random seed.
        """
        super().__init__(
            d=d,
            k=k,
            eta0=eta0,
            eta_schedule=eta_schedule,
            ortho_interval=ortho_interval,
            grad_clip=grad_clip,
            normalize_by_variance=normalize_by_variance,
            seed=seed
        )

        self.bits = bits
        self.quant_target = quant_target
        self.stochastic_rounding = stochastic_rounding
        self.error_feedback = error_feedback

        # Error feedback residual (accumulated quantization error)
        self.residual = np.zeros((d, k), dtype=np.float64)

        # Track quantization statistics
        self.quant_scale_history = []

    def _quantize(self, x: np.ndarray) -> np.ndarray:
        """Apply quantization to an array."""
        q, scale = uniform_quantize(
            x,
            bits=self.bits,
            stochastic=self.stochastic_rounding,
            rng=self.rng
        )
        self.quant_scale_history.append(scale)
        return q

    def partial_fit(self, x: np.ndarray) -> "QuantizedOjaPCA":
        """
        Update the model with a single sample using quantized updates.

        Args:
            x: Sample vector of shape (d,).

        Returns:
            self, for method chaining.
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.shape[0] != self.d:
            raise ValueError(f"Expected dimension {self.d}, got {x.shape[0]}")

        # Update mean estimator (always full precision)
        self.mean_estimator.update(x)

        # Center the sample
        x_centered = x - self.mean_estimator.get_mean()

        # Optionally normalize by variance
        if self.normalize_by_variance and self.mean_estimator.count >= 2:
            std = self.mean_estimator.get_std()
            std = np.maximum(std, 1e-8)
            x_centered = x_centered / std

        # Quantize input if requested
        if self.quant_target in ("input", "both"):
            x_centered = self._quantize(x_centered)

        # Increment step counter
        self.t += 1

        # Compute learning rate
        eta = compute_learning_rate(self.t, self.eta0, self.eta_schedule)

        # Compute Oja's update: delta_W = eta * x * (x^T W)
        projection = x_centered @ self.W  # shape (k,)
        delta_W = eta * np.outer(x_centered, projection)  # shape (d, k)

        # Optional gradient clipping (before quantization)
        if self.grad_clip is not None:
            grad_norm = np.linalg.norm(delta_W, "fro")
            if grad_norm > self.grad_clip:
                delta_W = delta_W * (self.grad_clip / grad_norm)

        # Quantize update if requested
        if self.quant_target in ("update", "both"):
            if self.error_feedback:
                # Error feedback: add residual before quantization
                delta_W_with_residual = delta_W + self.residual
                delta_W_quantized = self._quantize(delta_W_with_residual)
                # Update residual: what we wanted minus what we got
                self.residual = delta_W_with_residual - delta_W_quantized
                delta_W = delta_W_quantized
            else:
                delta_W = self._quantize(delta_W)

        # Apply update to master weights (always float64)
        self.W += delta_W

        # Periodic re-orthonormalization
        if self.t % self.ortho_interval == 0:
            self._orthonormalize()

        return self

    def get_memory_bytes(self, report_quantized: bool = False) -> int:
        """
        Estimate memory usage in bytes.

        Args:
            report_quantized: If True, report theoretical quantized memory.
                              If False, report actual memory (float64 master).

        Returns:
            Memory in bytes.
        """
        if report_quantized:
            # Theoretical quantized storage
            w_bytes = compute_quantization_memory(self.d, self.k, self.bits)
        else:
            # Actual storage (master weights in float64)
            w_bytes = self.W.nbytes

        # Mean estimator always full precision
        mean_bytes = self.mean_estimator.mean.nbytes + self.mean_estimator.M2.nbytes

        # Error feedback residual
        if self.error_feedback:
            residual_bytes = self.residual.nbytes
        else:
            residual_bytes = 0

        return w_bytes + mean_bytes + residual_bytes

    def get_effective_bits(self) -> int:
        """Return the effective bit width used for quantization."""
        return self.bits

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset the model to initial state."""
        super().reset(seed)
        self.residual = np.zeros((self.d, self.k), dtype=np.float64)
        self.quant_scale_history = []


def quantized_oja_pca(
    X: np.ndarray,
    k: int,
    bits: int = 8,
    error_feedback: bool = True,
    eta0: float = 0.1,
    eta_schedule: LearningRateSchedule = "invsqrt",
    seed: Optional[int] = None
) -> Tuple[np.ndarray, QuantizedOjaPCA]:
    """
    Convenience function to run quantized Oja's PCA on a dataset.

    Args:
        X: Data matrix of shape (n_samples, d).
        k: Number of principal components.
        bits: Quantization bit width.
        error_feedback: Use error feedback for residual accumulation.
        eta0: Initial learning rate.
        eta_schedule: Learning rate schedule.
        seed: Random seed.

    Returns:
        Tuple of (components, model) where components has shape (k, d).
    """
    X = np.asarray(X, dtype=np.float64)
    d = X.shape[1]
    model = QuantizedOjaPCA(
        d, k,
        bits=bits,
        error_feedback=error_feedback,
        eta0=eta0,
        eta_schedule=eta_schedule,
        seed=seed
    )
    model.fit(X)
    return model.components_, model
