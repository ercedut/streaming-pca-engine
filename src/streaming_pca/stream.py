# Author: Emrullah Erce Dutkan
"""
Streaming data generators for testing streaming PCA algorithms.

This module provides synthetic data streams with controllable properties:
- Low-rank Gaussian data with specified eigenvalue decay
- Concept drift (rotating subspace over time)
- Heavy-tailed distributions for robustness testing
- Stationary and non-stationary scenarios

These generators yield samples one at a time, simulating a real streaming
scenario where data arrives sequentially.
"""

from typing import Iterator, Optional, Tuple, Literal
import numpy as np


EigenvalueDecay = Literal["linear", "exponential", "polynomial"]


def generate_covariance_matrix(
    d: int,
    rank: int,
    noise_std: float = 0.1,
    decay: EigenvalueDecay = "exponential",
    decay_rate: float = 0.5,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a covariance matrix with specified rank and eigenvalue decay.

    Args:
        d: Dimensionality.
        rank: Effective rank (number of dominant eigenvalues).
        noise_std: Standard deviation of isotropic noise added to all dimensions.
        decay: Type of eigenvalue decay ("linear", "exponential", "polynomial").
        decay_rate: Controls decay speed.
        seed: Random seed.

    Returns:
        Tuple of:
        - Covariance matrix (d, d)
        - Eigenvectors (d, d), columns are eigenvectors
        - Eigenvalues (d,), sorted descending
    """
    rng = np.random.default_rng(seed)

    # Generate random orthonormal basis
    Q = rng.standard_normal((d, d))
    Q, _ = np.linalg.qr(Q)

    # Generate eigenvalues with decay
    if decay == "linear":
        # Linear decay from 1 to near 0
        lambdas = np.zeros(d)
        lambdas[:rank] = 1 - decay_rate * np.arange(rank) / rank
        lambdas[rank:] = noise_std ** 2
    elif decay == "exponential":
        # Exponential decay
        lambdas = np.zeros(d)
        lambdas[:rank] = np.exp(-decay_rate * np.arange(rank))
        lambdas[rank:] = noise_std ** 2
    elif decay == "polynomial":
        # Polynomial decay: 1/(i+1)^decay_rate
        lambdas = np.zeros(d)
        lambdas[:rank] = 1 / (np.arange(1, rank + 1) ** decay_rate)
        lambdas[rank:] = noise_std ** 2
    else:
        raise ValueError(f"Unknown decay type: {decay}")

    # Ensure non-negative
    lambdas = np.maximum(lambdas, 1e-10)

    # Construct covariance: Sigma = Q @ diag(lambdas) @ Q^T
    Sigma = Q @ np.diag(lambdas) @ Q.T

    # Sort eigenvectors by eigenvalue (descending)
    idx = np.argsort(lambdas)[::-1]
    lambdas = lambdas[idx]
    Q = Q[:, idx]

    return Sigma, Q, lambdas


def rotation_matrix(d: int, angle: float, plane: Tuple[int, int]) -> np.ndarray:
    """
    Generate a rotation matrix that rotates in a 2D plane.

    Args:
        d: Dimensionality.
        angle: Rotation angle in radians.
        plane: Tuple (i, j) specifying the plane of rotation.

    Returns:
        Rotation matrix (d, d).
    """
    R = np.eye(d)
    i, j = plane
    c, s = np.cos(angle), np.sin(angle)
    R[i, i] = c
    R[j, j] = c
    R[i, j] = -s
    R[j, i] = s
    return R


class SyntheticStream:
    """
    Generator for synthetic streaming data with controllable properties.

    This class generates samples from a multivariate Gaussian with a
    specified covariance structure. It supports concept drift via
    gradual rotation of the principal subspace.

    Attributes:
        d: Dimensionality.
        rank: Effective rank of the covariance.
        current_eigenvectors: Current principal directions (may drift).
    """

    def __init__(
        self,
        d: int,
        rank: int,
        noise_std: float = 0.1,
        decay: EigenvalueDecay = "exponential",
        decay_rate: float = 0.5,
        mean: Optional[np.ndarray] = None,
        drift_interval: Optional[int] = None,
        drift_angle: float = 0.1,
        seed: Optional[int] = None
    ):
        """
        Initialize the synthetic stream.

        Args:
            d: Dimensionality of each sample.
            rank: Number of dominant principal components.
            noise_std: Noise standard deviation.
            decay: Eigenvalue decay type.
            decay_rate: Eigenvalue decay rate.
            mean: Mean vector. If None, uses zeros.
            drift_interval: If set, rotate subspace every N samples.
            drift_angle: Rotation angle per drift event (radians).
            seed: Random seed for reproducibility.
        """
        self.d = d
        self.rank = rank
        self.noise_std = noise_std
        self.decay = decay
        self.decay_rate = decay_rate
        self.drift_interval = drift_interval
        self.drift_angle = drift_angle
        self.seed = seed

        self.rng = np.random.default_rng(seed)

        # Generate initial covariance structure
        self.Sigma, self.eigenvectors, self.eigenvalues = generate_covariance_matrix(
            d, rank, noise_std, decay, decay_rate,
            seed=self.rng.integers(0, 2**31)
        )

        # Current eigenvectors (may drift)
        self.current_eigenvectors = self.eigenvectors.copy()

        # Mean
        self.mean = mean if mean is not None else np.zeros(d)

        # Cholesky decomposition for efficient sampling
        self._update_cholesky()

        # Sample counter
        self.n_samples = 0
        self.n_drifts = 0

    def _update_cholesky(self) -> None:
        """Update Cholesky decomposition for current covariance."""
        # Reconstruct covariance from current eigenvectors
        Sigma = self.current_eigenvectors @ np.diag(self.eigenvalues) @ self.current_eigenvectors.T
        # Add small regularization for numerical stability
        Sigma += 1e-10 * np.eye(self.d)
        self.L = np.linalg.cholesky(Sigma)

    def _apply_drift(self) -> None:
        """Apply rotation to the principal subspace."""
        # Rotate in a random 2D plane involving the top components
        if self.d >= 2:
            # Choose a plane involving the first principal component
            i = 0
            j = self.rng.integers(1, min(self.rank + 1, self.d))
            R = rotation_matrix(self.d, self.drift_angle, (i, j))
            self.current_eigenvectors = R @ self.current_eigenvectors
            self._update_cholesky()
            self.n_drifts += 1

    def sample(self) -> np.ndarray:
        """
        Generate a single sample.

        Returns:
            Sample vector of shape (d,).
        """
        # Check for drift
        if self.drift_interval is not None and self.n_samples > 0:
            if self.n_samples % self.drift_interval == 0:
                self._apply_drift()

        # Generate sample: x = mean + L @ z, where z ~ N(0, I)
        z = self.rng.standard_normal(self.d)
        x = self.mean + self.L @ z

        self.n_samples += 1
        return x

    def stream(self, n: int) -> Iterator[np.ndarray]:
        """
        Generate a stream of n samples.

        Args:
            n: Number of samples to generate.

        Yields:
            Sample vectors of shape (d,).
        """
        for _ in range(n):
            yield self.sample()

    def get_batch(self, n: int) -> np.ndarray:
        """
        Generate a batch of n samples.

        Args:
            n: Number of samples.

        Returns:
            Data matrix of shape (n, d).
        """
        return np.array([self.sample() for _ in range(n)])

    def get_true_components(self, k: int) -> np.ndarray:
        """
        Get the current true principal components.

        Args:
            k: Number of components.

        Returns:
            Components of shape (k, d).
        """
        return self.current_eigenvectors[:, :k].T.copy()

    def get_initial_components(self, k: int) -> np.ndarray:
        """
        Get the initial principal components (before any drift).

        Args:
            k: Number of components.

        Returns:
            Components of shape (k, d).
        """
        return self.eigenvectors[:, :k].T.copy()

    def reset(self, seed: Optional[int] = None) -> None:
        """
        Reset the stream to initial state.

        Args:
            seed: New seed. If None, uses original seed.
        """
        if seed is not None:
            self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.current_eigenvectors = self.eigenvectors.copy()
        self._update_cholesky()
        self.n_samples = 0
        self.n_drifts = 0


class HeavyTailedStream:
    """
    Stream generator with heavy-tailed (Student-t) noise.

    Useful for testing robustness of streaming PCA algorithms
    to outliers and heavy tails.
    """

    def __init__(
        self,
        d: int,
        rank: int,
        df: float = 3.0,
        noise_std: float = 0.1,
        seed: Optional[int] = None
    ):
        """
        Initialize heavy-tailed stream.

        Args:
            d: Dimensionality.
            rank: Effective rank.
            df: Degrees of freedom for t-distribution. Lower = heavier tails.
            noise_std: Noise scale.
            seed: Random seed.
        """
        self.d = d
        self.rank = rank
        self.df = df
        self.seed = seed

        self.rng = np.random.default_rng(seed)

        # Generate covariance structure
        _, self.eigenvectors, self.eigenvalues = generate_covariance_matrix(
            d, rank, noise_std, "exponential", 0.5,
            seed=self.rng.integers(0, 2**31)
        )

        self.n_samples = 0

    def sample(self) -> np.ndarray:
        """Generate a single sample from heavy-tailed distribution."""
        # Generate t-distributed sample via:
        # x = sqrt(df/chi2) * z, where z ~ N(0, Sigma)
        z = self.rng.standard_normal(self.d)
        chi2 = self.rng.chisquare(self.df)
        scale = np.sqrt(self.df / chi2)

        # Transform by covariance structure
        L = self.eigenvectors @ np.diag(np.sqrt(self.eigenvalues))
        x = scale * (L @ z)

        self.n_samples += 1
        return x

    def stream(self, n: int) -> Iterator[np.ndarray]:
        """Generate a stream of n samples."""
        for _ in range(n):
            yield self.sample()

    def get_batch(self, n: int) -> np.ndarray:
        """Generate a batch of n samples."""
        return np.array([self.sample() for _ in range(n)])

    def get_true_components(self, k: int) -> np.ndarray:
        """Get true principal components."""
        return self.eigenvectors[:, :k].T.copy()


class DatasetStream:
    """
    Stream wrapper for existing datasets.

    Wraps a data matrix and yields samples in random or sequential order,
    simulating a stream from a fixed dataset.
    """

    def __init__(
        self,
        X: np.ndarray,
        shuffle: bool = True,
        seed: Optional[int] = None
    ):
        """
        Initialize dataset stream.

        Args:
            X: Data matrix of shape (n_samples, d).
            shuffle: If True, shuffle order of samples.
            seed: Random seed for shuffling.
        """
        self.X = np.asarray(X, dtype=np.float64)
        self.n, self.d = self.X.shape
        self.shuffle = shuffle
        self.seed = seed

        self.rng = np.random.default_rng(seed)

        # Generate order
        self._generate_order()

        self.idx = 0
        self.n_samples = 0

    def _generate_order(self) -> None:
        """Generate sample order."""
        self.order = np.arange(self.n)
        if self.shuffle:
            self.rng.shuffle(self.order)

    def sample(self) -> np.ndarray:
        """Get next sample from the dataset."""
        if self.idx >= self.n:
            # Wrap around and reshuffle
            self._generate_order()
            self.idx = 0

        x = self.X[self.order[self.idx]]
        self.idx += 1
        self.n_samples += 1
        return x

    def stream(self, n: int) -> Iterator[np.ndarray]:
        """Generate a stream of n samples."""
        for _ in range(n):
            yield self.sample()

    def get_batch(self, n: int) -> np.ndarray:
        """Get a batch of n samples."""
        return np.array([self.sample() for _ in range(n)])

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset stream to beginning."""
        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed)
        self._generate_order()
        self.idx = 0
        self.n_samples = 0


def create_stream(
    stream_type: str,
    d: int,
    rank: int,
    seed: Optional[int] = None,
    **kwargs
) -> SyntheticStream:
    """
    Factory function to create streams.

    Args:
        stream_type: One of "gaussian", "drifting", "heavy_tailed".
        d: Dimensionality.
        rank: Effective rank.
        seed: Random seed.
        **kwargs: Additional arguments for specific stream types.

    Returns:
        Stream object.
    """
    if stream_type == "gaussian":
        return SyntheticStream(
            d=d, rank=rank, seed=seed,
            noise_std=kwargs.get("noise_std", 0.1),
            decay=kwargs.get("decay", "exponential"),
            decay_rate=kwargs.get("decay_rate", 0.5)
        )
    elif stream_type == "drifting":
        return SyntheticStream(
            d=d, rank=rank, seed=seed,
            noise_std=kwargs.get("noise_std", 0.1),
            drift_interval=kwargs.get("drift_interval", 5000),
            drift_angle=kwargs.get("drift_angle", 0.1)
        )
    elif stream_type == "heavy_tailed":
        return HeavyTailedStream(
            d=d, rank=rank, seed=seed,
            df=kwargs.get("df", 3.0),
            noise_std=kwargs.get("noise_std", 0.1)
        )
    else:
        raise ValueError(f"Unknown stream type: {stream_type}")
