# Author: Emrullah Erce Dutkan
"""
Oja's algorithm for streaming PCA.

This module implements Oja's online learning rule for extracting the top-k
principal components from a data stream in a single pass. The algorithm
maintains an orthonormal basis W in R^{d x k} and updates it incrementally
as each sample arrives.

Key features:
- Multiple learning rate schedules (constant, 1/sqrt(t), 1/t)
- Online mean estimation via Welford's algorithm
- Periodic QR re-orthonormalization for numerical stability
- Optional gradient clipping for heavy-tailed distributions
- Optional covariance normalization using running variance
"""

from typing import Optional, Literal, Tuple
import numpy as np


LearningRateSchedule = Literal["constant", "invsqrt", "invt"]


class OnlineMeanEstimator:
    """
    Welford's online algorithm for computing running mean and variance.

    This is numerically stable and computes the mean incrementally without
    storing all samples. For streaming PCA, we need to center data using
    an estimate of the population mean.

    Welford's algorithm is preferred over exponential moving average (EMA)
    because:
    1. It converges to the true mean for stationary distributions
    2. It is numerically stable for large sample counts
    3. It provides unbiased variance estimates

    For non-stationary streams where the mean drifts, EMA may be preferable.
    This implementation uses Welford by default.
    """

    def __init__(self, d: int):
        """
        Initialize the online mean estimator.

        Args:
            d: Dimensionality of the data.
        """
        self.d = d
        self.mean = np.zeros(d, dtype=np.float64)
        self.M2 = np.zeros(d, dtype=np.float64)  # Sum of squared deviations
        self.count = 0

    def update(self, x: np.ndarray) -> None:
        """
        Update the running mean and variance with a new sample.

        Args:
            x: New sample vector of shape (d,).
        """
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def get_mean(self) -> np.ndarray:
        """Return the current mean estimate."""
        return self.mean.copy()

    def get_variance(self) -> np.ndarray:
        """
        Return the current variance estimate.

        Uses Bessel's correction (n-1) for unbiased estimation.
        Returns zeros if count < 2.
        """
        if self.count < 2:
            return np.ones(self.d, dtype=np.float64)
        return self.M2 / (self.count - 1)

    def get_std(self) -> np.ndarray:
        """Return the current standard deviation estimate."""
        return np.sqrt(self.get_variance())

    def reset(self) -> None:
        """Reset the estimator to initial state."""
        self.mean = np.zeros(self.d, dtype=np.float64)
        self.M2 = np.zeros(self.d, dtype=np.float64)
        self.count = 0


def compute_learning_rate(
    t: int,
    eta0: float,
    schedule: LearningRateSchedule
) -> float:
    """
    Compute the learning rate at step t.

    Args:
        t: Current step (1-indexed).
        eta0: Initial learning rate.
        schedule: One of "constant", "invsqrt", or "invt".

    Returns:
        Learning rate eta_t.

    The schedules are:
    - constant: eta_t = eta0
    - invsqrt: eta_t = eta0 / sqrt(t), standard for SGD convergence
    - invt: eta_t = eta0 / t, faster decay
    """
    if schedule == "constant":
        return eta0
    elif schedule == "invsqrt":
        return eta0 / np.sqrt(t)
    elif schedule == "invt":
        return eta0 / t
    else:
        raise ValueError(f"Unknown schedule: {schedule}")


class OjaPCA:
    """
    Streaming PCA using Oja's algorithm.

    Oja's rule updates the weight matrix W as:
        W <- W + eta_t * x_t * (x_t^T W)

    where x_t is the centered sample at time t and eta_t is the learning rate.

    After the update, W is periodically re-orthonormalized via QR decomposition
    to maintain numerical stability. The columns of W converge to the top-k
    principal eigenvectors of the data covariance matrix.

    Attributes:
        d: Data dimensionality.
        k: Number of principal components.
        W: Weight matrix of shape (d, k), columns are approximate eigenvectors.
        components_: Transposed weight matrix (k, d), compatible with sklearn API.
    """

    def __init__(
        self,
        d: int,
        k: int,
        eta0: float = 0.1,
        eta_schedule: LearningRateSchedule = "invsqrt",
        ortho_interval: int = 1,
        grad_clip: Optional[float] = None,
        normalize_by_variance: bool = False,
        seed: Optional[int] = None
    ):
        """
        Initialize Oja's PCA algorithm.

        Args:
            d: Dimensionality of input data.
            k: Number of principal components to extract.
            eta0: Initial learning rate. Typical values: 0.1-1.0 for invsqrt.
            eta_schedule: Learning rate schedule ("constant", "invsqrt", "invt").
            ortho_interval: Re-orthonormalize every N steps. 1 = every step.
            grad_clip: If set, clip gradient norm to this value.
            normalize_by_variance: If True, normalize samples by running variance.
            seed: Random seed for reproducibility.
        """
        self.d = d
        self.k = k
        self.eta0 = eta0
        self.eta_schedule = eta_schedule
        self.ortho_interval = ortho_interval
        self.grad_clip = grad_clip
        self.normalize_by_variance = normalize_by_variance

        self.rng = np.random.default_rng(seed)

        # Initialize W with random orthonormal columns
        self.W = self._init_weights()

        # Online mean/variance estimator (Welford's algorithm)
        self.mean_estimator = OnlineMeanEstimator(d)

        # Step counter
        self.t = 0

    def _init_weights(self) -> np.ndarray:
        """Initialize weight matrix with random orthonormal columns."""
        # Random Gaussian initialization
        W = self.rng.standard_normal((self.d, self.k))
        # Orthonormalize via QR
        Q, _ = np.linalg.qr(W)
        return Q.astype(np.float64)

    def _orthonormalize(self) -> None:
        """Re-orthonormalize W via QR decomposition."""
        Q, R = np.linalg.qr(self.W)
        # Ensure consistent sign (positive diagonal in R)
        signs = np.sign(np.diag(R))
        signs[signs == 0] = 1
        self.W = Q * signs

    def partial_fit(self, x: np.ndarray) -> "OjaPCA":
        """
        Update the model with a single sample.

        Args:
            x: Sample vector of shape (d,).

        Returns:
            self, for method chaining.
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.shape[0] != self.d:
            raise ValueError(f"Expected dimension {self.d}, got {x.shape[0]}")

        # Update mean estimator
        self.mean_estimator.update(x)

        # Center the sample
        x_centered = x - self.mean_estimator.get_mean()

        # Optionally normalize by variance
        if self.normalize_by_variance and self.mean_estimator.count >= 2:
            std = self.mean_estimator.get_std()
            # Avoid division by zero
            std = np.maximum(std, 1e-8)
            x_centered = x_centered / std

        # Increment step counter
        self.t += 1

        # Compute learning rate
        eta = compute_learning_rate(self.t, self.eta0, self.eta_schedule)

        # Oja's update: W <- W + eta * x * (x^T W)
        # This is an outer product update
        projection = x_centered @ self.W  # shape (k,)
        gradient = np.outer(x_centered, projection)  # shape (d, k)

        # Optional gradient clipping
        if self.grad_clip is not None:
            grad_norm = np.linalg.norm(gradient, "fro")
            if grad_norm > self.grad_clip:
                gradient = gradient * (self.grad_clip / grad_norm)

        # Apply update
        self.W += eta * gradient

        # Periodic re-orthonormalization
        if self.t % self.ortho_interval == 0:
            self._orthonormalize()

        return self

    def fit(self, X: np.ndarray) -> "OjaPCA":
        """
        Fit the model to a batch of samples (processes them as a stream).

        Args:
            X: Data matrix of shape (n_samples, d).

        Returns:
            self, for method chaining.
        """
        X = np.asarray(X, dtype=np.float64)
        for i in range(X.shape[0]):
            self.partial_fit(X[i])
        return self

    @property
    def components_(self) -> np.ndarray:
        """
        Return the principal components as row vectors.

        Shape: (k, d), compatible with sklearn PCA API.
        """
        return self.W.T.copy()

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project data onto the principal components.

        Args:
            X: Data matrix of shape (n_samples, d) or (d,).

        Returns:
            Projected data of shape (n_samples, k) or (k,).
        """
        X = np.asarray(X, dtype=np.float64)
        single = X.ndim == 1
        if single:
            X = X.reshape(1, -1)

        # Center using learned mean
        X_centered = X - self.mean_estimator.get_mean()

        if self.normalize_by_variance and self.mean_estimator.count >= 2:
            std = self.mean_estimator.get_std()
            std = np.maximum(std, 1e-8)
            X_centered = X_centered / std

        result = X_centered @ self.W

        if single:
            return result.ravel()
        return result

    def inverse_transform(self, Y: np.ndarray) -> np.ndarray:
        """
        Reconstruct data from the projected representation.

        Args:
            Y: Projected data of shape (n_samples, k) or (k,).

        Returns:
            Reconstructed data of shape (n_samples, d) or (d,).
        """
        Y = np.asarray(Y, dtype=np.float64)
        single = Y.ndim == 1
        if single:
            Y = Y.reshape(1, -1)

        result = Y @ self.W.T

        # Add back the mean
        result = result + self.mean_estimator.get_mean()

        if self.normalize_by_variance and self.mean_estimator.count >= 2:
            std = self.mean_estimator.get_std()
            std = np.maximum(std, 1e-8)
            result = result * std

        if single:
            return result.ravel()
        return result

    def get_memory_bytes(self) -> int:
        """
        Estimate memory usage in bytes.

        Counts:
        - W matrix: d * k * 8 bytes (float64)
        - Mean estimator: 2 * d * 8 bytes
        - Scalar state: negligible
        """
        w_bytes = self.W.nbytes
        mean_bytes = self.mean_estimator.mean.nbytes + self.mean_estimator.M2.nbytes
        return w_bytes + mean_bytes

    def reset(self, seed: Optional[int] = None) -> None:
        """
        Reset the model to initial state.

        Args:
            seed: New random seed. If None, use original seed.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.W = self._init_weights()
        self.mean_estimator.reset()
        self.t = 0


def oja_pca(
    X: np.ndarray,
    k: int,
    eta0: float = 0.1,
    eta_schedule: LearningRateSchedule = "invsqrt",
    seed: Optional[int] = None
) -> Tuple[np.ndarray, OjaPCA]:
    """
    Convenience function to run Oja's PCA on a dataset.

    Args:
        X: Data matrix of shape (n_samples, d).
        k: Number of principal components.
        eta0: Initial learning rate.
        eta_schedule: Learning rate schedule.
        seed: Random seed.

    Returns:
        Tuple of (components, model) where components has shape (k, d).
    """
    X = np.asarray(X, dtype=np.float64)
    d = X.shape[1]
    model = OjaPCA(d, k, eta0=eta0, eta_schedule=eta_schedule, seed=seed)
    model.fit(X)
    return model.components_, model
