# Streaming PCA Engine

A Python library for streaming principal component analysis, implementing Oja's algorithm, a quantized low-precision variant, and Frequent Directions sketching for memory-efficient comparison.

## Overview

This repository provides implementations of streaming PCA algorithms designed for edge computing and online learning scenarios where data arrives sequentially and storage is limited. The key algorithms are:

1. Oja's algorithm: An online learning rule that incrementally updates an estimate of the top-k principal components
2. Quantized Oja: A variant that reduces precision of updates for memory and bandwidth efficiency
3. Frequent Directions: A deterministic sketching algorithm that maintains a low-rank approximation of the data covariance

The library includes benchmarking tools, evaluation metrics, and visualization utilities for comparing these methods.

## Streaming PCA Problem and Constraints

In streaming PCA, data samples arrive one at a time, and the goal is to maintain an estimate of the top-k principal components without storing all the data. Constraints include:

- Single-pass or limited-pass processing of the data
- Bounded memory independent of the number of samples seen
- Ability to update estimates incrementally as each sample arrives
- Numerical stability over long streams

These constraints arise in applications such as real-time sensor data analysis, network traffic monitoring, and edge computing where resources are limited.

## Oja Algorithm

Oja's rule updates a weight matrix W in R^{d x k} representing the top-k eigenvector estimates. For each incoming sample x_t (after centering):

    W <- W + eta_t * x_t * (x_t^T * W)

where eta_t is the learning rate at step t.

### Update Rule Details

The algorithm maintains W with k columns representing approximations to the top-k eigenvectors of the data covariance. The update adds a rank-1 correction proportional to the outer product of the sample with its projection onto the current subspace.

### Orthonormalization

After each update (or periodically), W is re-orthonormalized via QR decomposition:

    Q, R = qr(W)
    W = Q * sign(diag(R))

This ensures the columns remain orthonormal, preventing drift and improving numerical stability. The sign correction ensures consistent orientation.

### Learning Rate Schedules

Three learning rate schedules are supported:

- constant: eta_t = eta0
- invsqrt: eta_t = eta0 / sqrt(t), standard choice for SGD convergence
- invt: eta_t = eta0 / t, faster decay

The invsqrt schedule is recommended as a default. It balances convergence speed with stability.

### Centering Strategy

Data must be centered before processing. This implementation uses Welford's online algorithm to maintain a running mean estimate:

    count <- count + 1
    delta <- x - mean
    mean <- mean + delta / count

Welford's method is numerically stable and converges to the true mean for stationary distributions. For non-stationary streams where the mean drifts, an exponential moving average may be more appropriate, but this implementation uses Welford by default.

## Quantized Oja

The quantized variant applies low-precision quantization to reduce memory and communication costs.

### What Is Quantized

In this implementation, quantization is applied to:

- The gradient update delta_W = eta_t * x_t * (x_t^T * W) before adding to W
- Optionally, the input sample x_t can also be quantized

The master weight matrix W is always kept in float64 for numerical stability. Quantization affects only the update path.

### Bit Widths

Uniform symmetric quantization with configurable bit width b (typically 4, 6, or 8 bits):

- 8 bits: Minimal accuracy loss, good default choice
- 6 bits: Moderate compression with some accuracy reduction
- 4 bits: Aggressive compression, may require error feedback for stability

The quantization maps values to integers in [-2^{b-1}+1, 2^{b-1}-1], then scales back to floats.

### Error Feedback

Error feedback (also called residual accumulation) compensates for quantization error:

    r <- r + delta_W
    delta_W_q = quantize(r)
    r <- r - delta_W_q
    W <- W + delta_W_q

This ensures that quantization errors accumulate in the residual and are eventually corrected, reducing bias at the cost of additional memory for the residual buffer.

### Stochastic Rounding

Stochastic rounding rounds to the nearest quantization level with probability proportional to the fractional part:

    P(round up) = fractional part
    P(round down) = 1 - fractional part

This makes E[quantize(x)] = x, providing unbiased gradient estimates.

### Stability Notes

- Very low bit widths (below 4) may cause instability without error feedback
- Stochastic rounding is recommended for unbiased gradients
- Gradient clipping can help with heavy-tailed data
- The master weights are never quantized, only the updates

## Sketching Baseline

### Frequent Directions

Frequent Directions (Liberty, 2013) maintains a sketch matrix B of size (l x d) where l >= k:

1. Insert incoming samples as rows in B
2. When B is full (l rows), shrink via SVD:
   - Compute B = U * S * V^T
   - Shrink: S' = sqrt(max(0, S^2 - sigma_l^2 * I))
   - Update B = S' * V^T

The shrinkage step reduces the sketch back to l-1 non-zero rows, making room for new samples.

Principal components are extracted from the SVD of the sketch matrix. The algorithm is deterministic and provides provable approximation guarantees.

Memory complexity: O(l * d) floats, independent of the number of samples processed.

## Metrics

### Subspace Error

The subspace distance between estimated W_est and reference W_ref is computed via principal angles:

1. Compute M = W_ref^T * W_est
2. Compute SVD: M = U * S * V^T
3. Principal angles: theta_i = arccos(s_i)
4. Report mean(sin(theta_i))

A value of 0 indicates identical subspaces; larger values indicate greater disagreement.

### Explained Variance Ratio

The fraction of total variance captured by the principal components:

    EV = mean(||W^T * x||^2) / mean(||x||^2)

where x are centered samples. Computed on a held-out test set.

### Reconstruction Error

Mean squared error of projection and reconstruction:

    MSE = mean(||x - W * W^T * x||^2)

This measures how well the subspace captures the data.

### Batch PCA Reference

The batch PCA reference is computed using numpy SVD on a fixed dataset:

    U, S, V^T = svd(X_centered)
    W_ref = V^T[:k]

This serves as a baseline for evaluating streaming methods. Note that batch PCA requires storing all data and is not a streaming algorithm; it is used only for evaluation.

## How to Run

### Installation

    pip install -r requirements.txt

### CLI Examples

Simulation mode (real-time streaming visualization):

    python -m src.cli --mode simulate --d 50 --k 5 --n-steps 20000 --eta-schedule invsqrt --eta0 0.5 --plot

Simulation with concept drift:

    python -m src.cli --mode simulate --d 50 --k 5 --n-steps 40000 --drift --drift-interval 10000 --plot

Benchmark mode (systematic comparison and reports):

    python -m src.cli --mode benchmark --d 100 --k 10 --steps 30000 --quant-bits 4,6,8 --sketch-size 40 --reports reports/

Real dataset mode (digits dataset):

    python -m src.cli --mode real --dataset digits --k 10 --eta-schedule invsqrt --plot

### Python API

    from streaming_pca import OjaPCA, QuantizedOjaPCA, SketchPCA

    # Initialize Oja PCA
    model = OjaPCA(d=100, k=10, eta0=0.5, eta_schedule="invsqrt")

    # Process samples one at a time
    for x in data_stream:
        model.partial_fit(x)

    # Get components
    components = model.components_  # shape (k, d)

    # Transform new data
    projected = model.transform(X_new)

## Memory-Accuracy Tradeoff Report

The benchmark mode produces reports comparing memory usage against accuracy:

### Output Files

- reports/tradeoff.csv: Summary metrics for each method
  - method, subspace_error, explained_variance, reconstruction_error, memory_bytes, runtime_seconds, bits, etc.

- reports/tradeoff_accuracy_vs_memory.png: Scatter plot of accuracy vs memory

- reports/metric_over_time.png: Convergence curves showing subspace error over streaming steps

- logs/metrics.csv: Detailed metrics logged over time during the run

### What Is Measured

The benchmark measures:

- Final subspace error compared to batch PCA reference
- Final explained variance ratio on held-out test data
- Reconstruction error on held-out test data
- Memory usage in bytes (computed from array sizes)
- Runtime in seconds

Memory is computed explicitly from the stored matrices:
- Oja: W (d x k x 8 bytes) + mean estimator (2 x d x 8 bytes)
- Quantized Oja: Same as Oja plus residual buffer if error feedback enabled
- Frequent Directions: B (l x d x 8 bytes) + mean (d x 8 bytes)

## Limitations and When Results Can Mislead

### Sensitivity to Learning Rate

Oja's algorithm is sensitive to the choice of eta0 and the learning rate schedule:
- Too large eta0 can cause divergence or oscillation
- Too small eta0 leads to slow convergence
- The invsqrt schedule works well in many cases but may need tuning
- Heavy-tailed data may require gradient clipping

### Non-Stationary Streams

The algorithms assume the underlying data distribution is approximately stationary:
- Concept drift (changing principal subspace) causes lag in adaptation
- Welford mean estimation converges slowly after a shift
- Exponential moving average may track drift better but has different tradeoffs
- The drift simulation mode can be used to study adaptation behavior

### Quantization Instability

Quantization can cause instability if misconfigured:
- Very low bit widths (2-3 bits) often fail without error feedback
- Input quantization combined with update quantization compounds errors
- Error feedback requires additional memory for the residual
- Stochastic rounding is important for avoiding systematic bias

### Metric Interpretation

- Subspace error depends on the reference, which is computed from a finite sample
- Explained variance on test data may differ from training statistics
- Early stopping may show good metrics that degrade with more samples
- Results on synthetic data may not transfer to real applications

### Comparison Fairness

- Frequent Directions uses more memory per sample but fewer computations
- Oja is incremental but requires tuning the learning rate
- Batch PCA is not a streaming algorithm and serves only as a reference baseline
- Memory estimates are approximate and exclude Python overhead

## Repository Structure

    streaming-pca-engine/
    ├── README.md
    ├── requirements.txt
    ├── src/
    │   ├── cli.py
    │   └── streaming_pca/
    │       ├── __init__.py
    │       ├── oja.py          # Oja algorithm implementation
    │       ├── quant.py        # Quantized Oja implementation
    │       ├── sketch.py       # Frequent Directions sketching
    │       ├── metrics.py      # Evaluation metrics
    │       ├── stream.py       # Synthetic data stream generators
    │       ├── datasets.py     # Dataset loading utilities
    │       ├── benchmark.py    # Benchmarking framework
    │       ├── viz.py          # Visualization utilities
    │       ├── io.py           # Input/output utilities
    │       └── config.py       # Configuration management
    ├── reports/                # Generated reports (created at runtime)
    └── logs/                   # Metric logs (created at runtime)

## Dependencies

- numpy: Core numerical operations
- matplotlib: Plotting and visualization
- pandas: Report tables (optional)
- scipy: Additional linear algebra (optional)
- scikit-learn: Dataset loading and batch PCA baseline (optional)

## Author

Emrullah Erce Dutkan
