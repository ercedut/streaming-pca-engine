# Author: Emrullah Erce Dutkan
"""
Sketching baseline for streaming PCA: Frequent Directions algorithm.

Frequent Directions (FD) is a deterministic streaming algorithm that maintains
a low-rank sketch of the data covariance matrix. It provides strong theoretical
guarantees on the quality of the approximation.

The algorithm maintains a sketch matrix B of size (l x d) where l >= k
(typically l = 2k). For each incoming sample, it appends the sample to B
and periodically shrinks B via SVD to maintain the size constraint.

Key properties:
- Deterministic (no randomization needed)
- Space complexity: O(ld) where l is the sketch size
- Per-sample update: O(ld) amortized
- Provides (1 + epsilon) approximation guarantees for covariance

Reference:
Liberty, E. (2013). Simple and deterministic matrix sketching.
Proceedings of the 19th ACM SIGKDD.
"""

from typing import Optional, Tuple
import numpy as np


class FrequentDirections:
    """
    Frequent Directions algorithm for streaming PCA.

    The algorithm maintains a sketch B of the data stream such that
    B^T B approximates the covariance matrix. Principal components
    can be extracted from the SVD of B.

    Algorithm:
    1. Maintain sketch B of size (l x d), initialized to zeros
    2. For each sample x:
       - Insert x as a row in B (replacing a zero row)
       - If B is full, shrink via SVD:
         a. Compute SVD: B = U S V^T
         b. Shrink singular values: S' = sqrt(S^2 - sigma_l^2 * I)
         c. Update B = S' V^T (keeping only top l-1 rows, one zero row)

    The shrinkage step ensures that the sketch doesn't grow unboundedly
    while preserving information about the dominant directions.
    """

    def __init__(
        self,
        d: int,
        l: int,
        seed: Optional[int] = None
    ):
        """
        Initialize Frequent Directions sketch.

        Args:
            d: Data dimensionality.
            l: Sketch size (number of rows in B). Should be >= 2k
               where k is the desired number of principal components.
            seed: Random seed (not used in this deterministic algorithm,
                  but kept for API consistency).
        """
        self.d = d
        self.l = l

        # Sketch matrix B of size (l x d)
        # We maintain at most l-1 non-zero rows at any time
        self.B = np.zeros((l, d), dtype=np.float64)

        # Number of non-zero rows currently in B
        self.n_rows = 0

        # Online mean estimator for centering
        self.mean = np.zeros(d, dtype=np.float64)
        self.count = 0

        # Step counter
        self.t = 0

    def _shrink(self) -> None:
        """
        Shrink the sketch via SVD when full.

        Computes SVD of B, subtracts sigma_l^2 from all squared singular values,
        and reconstructs B with the reduced singular values.
        """
        # Compute SVD of current sketch
        try:
            U, s, Vt = np.linalg.svd(self.B, full_matrices=False)
        except np.linalg.LinAlgError:
            # SVD failed, skip shrinkage this time
            return

        # Get the smallest singular value to subtract
        if len(s) >= self.l:
            delta = s[self.l - 1] ** 2
        else:
            delta = s[-1] ** 2 if len(s) > 0 else 0.0

        # Shrink singular values: s' = sqrt(max(0, s^2 - delta))
        s_squared = s ** 2 - delta
        s_squared = np.maximum(s_squared, 0)  # Numerical safety
        s_new = np.sqrt(s_squared)

        # Count non-zero singular values
        nonzero_mask = s_new > 1e-10
        n_nonzero = np.sum(nonzero_mask)

        # Reconstruct B with shrunk singular values
        # B = S_new @ Vt, but only keep rows with non-zero singular values
        self.B[:] = 0  # Reset
        if n_nonzero > 0:
            # Keep at most l-1 rows to leave room for new sample
            n_keep = min(int(n_nonzero), self.l - 1)
            for i in range(n_keep):
                if s_new[i] > 1e-10:
                    self.B[i] = s_new[i] * Vt[i]

        self.n_rows = min(int(n_nonzero), self.l - 1)

    def partial_fit(self, x: np.ndarray) -> "FrequentDirections":
        """
        Update the sketch with a single sample.

        Args:
            x: Sample vector of shape (d,).

        Returns:
            self, for method chaining.
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.shape[0] != self.d:
            raise ValueError(f"Expected dimension {self.d}, got {x.shape[0]}")

        # Update running mean (Welford-style)
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count

        # Center the sample
        x_centered = x - self.mean

        self.t += 1

        # Insert sample into sketch
        if self.n_rows < self.l:
            # Find first zero row
            self.B[self.n_rows] = x_centered
            self.n_rows += 1
        else:
            # Sketch is full, need to shrink first
            self._shrink()
            # Now insert
            if self.n_rows < self.l:
                self.B[self.n_rows] = x_centered
                self.n_rows += 1

        # If sketch is full after insertion, shrink
        if self.n_rows >= self.l:
            self._shrink()

        return self

    def fit(self, X: np.ndarray) -> "FrequentDirections":
        """
        Fit the sketch to a batch of samples.

        Args:
            X: Data matrix of shape (n_samples, d).

        Returns:
            self, for method chaining.
        """
        X = np.asarray(X, dtype=np.float64)
        for i in range(X.shape[0]):
            self.partial_fit(X[i])
        return self

    def get_components(self, k: int) -> np.ndarray:
        """
        Extract the top-k principal components from the sketch.

        Args:
            k: Number of components to extract.

        Returns:
            Components array of shape (k, d).
        """
        if k > self.l:
            raise ValueError(f"k={k} > sketch size l={self.l}")

        # Compute SVD of sketch
        try:
            U, s, Vt = np.linalg.svd(self.B, full_matrices=False)
        except np.linalg.LinAlgError:
            # Return whatever we have
            return self.B[:k].copy()

        # Return top-k right singular vectors
        return Vt[:k].copy()

    @property
    def components_(self) -> np.ndarray:
        """
        Return principal components (all available from sketch).

        Shape: (l, d) but only meaningful rows are non-zero.
        Use get_components(k) for specific k.
        """
        try:
            U, s, Vt = np.linalg.svd(self.B, full_matrices=False)
            # Return rows corresponding to non-trivial singular values
            n_components = np.sum(s > 1e-10)
            return Vt[:n_components].copy()
        except np.linalg.LinAlgError:
            return self.B.copy()

    def transform(self, X: np.ndarray, k: Optional[int] = None) -> np.ndarray:
        """
        Project data onto principal components.

        Args:
            X: Data matrix of shape (n_samples, d) or (d,).
            k: Number of components. If None, uses all available.

        Returns:
            Projected data.
        """
        X = np.asarray(X, dtype=np.float64)
        single = X.ndim == 1
        if single:
            X = X.reshape(1, -1)

        # Center
        X_centered = X - self.mean

        # Get components
        if k is None:
            V = self.components_
        else:
            V = self.get_components(k)

        result = X_centered @ V.T

        if single:
            return result.ravel()
        return result

    def get_memory_bytes(self) -> int:
        """
        Estimate memory usage in bytes.

        Counts:
        - Sketch B: l * d * 8 bytes
        - Mean: d * 8 bytes
        """
        return self.B.nbytes + self.mean.nbytes

    def reset(self) -> None:
        """Reset the sketch to initial state."""
        self.B = np.zeros((self.l, self.d), dtype=np.float64)
        self.n_rows = 0
        self.mean = np.zeros(self.d, dtype=np.float64)
        self.count = 0
        self.t = 0


class SketchPCA:
    """
    Wrapper class for Frequent Directions with sklearn-like API.

    This provides a consistent interface matching OjaPCA for benchmarking.
    """

    def __init__(
        self,
        d: int,
        k: int,
        sketch_size: Optional[int] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize sketch-based PCA.

        Args:
            d: Data dimensionality.
            k: Number of principal components to extract.
            sketch_size: Size of the sketch (l). If None, uses 2*k.
            seed: Random seed (for API consistency).
        """
        self.d = d
        self.k = k
        self.sketch_size = sketch_size if sketch_size is not None else 2 * k

        if self.sketch_size < k:
            raise ValueError(f"sketch_size ({self.sketch_size}) must be >= k ({k})")

        self.fd = FrequentDirections(d, self.sketch_size, seed=seed)
        self.t = 0

    def partial_fit(self, x: np.ndarray) -> "SketchPCA":
        """Update with a single sample."""
        self.fd.partial_fit(x)
        self.t = self.fd.t
        return self

    def fit(self, X: np.ndarray) -> "SketchPCA":
        """Fit to a batch of samples."""
        self.fd.fit(X)
        self.t = self.fd.t
        return self

    @property
    def components_(self) -> np.ndarray:
        """Return top-k principal components as (k, d) array."""
        return self.fd.get_components(self.k)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project data onto principal components."""
        return self.fd.transform(X, k=self.k)

    def get_memory_bytes(self) -> int:
        """Estimate memory usage in bytes."""
        return self.fd.get_memory_bytes()

    def reset(self) -> None:
        """Reset to initial state."""
        self.fd.reset()
        self.t = 0


def frequent_directions_pca(
    X: np.ndarray,
    k: int,
    sketch_size: Optional[int] = None
) -> Tuple[np.ndarray, SketchPCA]:
    """
    Convenience function to run Frequent Directions PCA on a dataset.

    Args:
        X: Data matrix of shape (n_samples, d).
        k: Number of principal components.
        sketch_size: Sketch size. If None, uses 2*k.

    Returns:
        Tuple of (components, model) where components has shape (k, d).
    """
    X = np.asarray(X, dtype=np.float64)
    d = X.shape[1]
    model = SketchPCA(d, k, sketch_size=sketch_size)
    model.fit(X)
    return model.components_, model
