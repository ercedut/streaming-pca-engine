# Author: Emrullah Erce Dutkan
"""
Evaluation metrics for streaming PCA algorithms.

This module provides metrics for comparing streaming PCA estimates
against a batch PCA reference:

1. Subspace distance: Measures angle between estimated and true subspaces
2. Explained variance ratio: Fraction of variance captured by components
3. Reconstruction error: Mean squared error of projection-reconstruction

These metrics are computed using standard linear algebra operations
and do not depend on any streaming PCA libraries.
"""

from typing import Optional, Tuple
import numpy as np


def principal_angles(
    W1: np.ndarray,
    W2: np.ndarray
) -> np.ndarray:
    """
    Compute principal angles between two subspaces.

    The principal angles theta_1, ..., theta_k between subspaces spanned
    by columns of W1 and W2 are defined via:
        cos(theta_i) = sigma_i(W1^T W2)

    where sigma_i are the singular values of W1^T W2.

    Args:
        W1: Matrix with orthonormal columns, shape (d, k1).
        W2: Matrix with orthonormal columns, shape (d, k2).

    Returns:
        Array of principal angles in radians, length min(k1, k2).
    """
    # Ensure column matrices
    if W1.ndim == 1:
        W1 = W1.reshape(-1, 1)
    if W2.ndim == 1:
        W2 = W2.reshape(-1, 1)

    # Compute W1^T W2
    M = W1.T @ W2

    # SVD to get singular values (cosines of principal angles)
    try:
        _, s, _ = np.linalg.svd(M, full_matrices=False)
    except np.linalg.LinAlgError:
        # Fallback: return worst case
        k = min(W1.shape[1], W2.shape[1])
        return np.full(k, np.pi / 2)

    # Clip to [0, 1] for numerical stability
    s = np.clip(s, 0, 1)

    # Principal angles
    angles = np.arccos(s)
    return angles


def subspace_distance(
    W_est: np.ndarray,
    W_ref: np.ndarray,
    method: str = "sin"
) -> float:
    """
    Compute distance between estimated and reference subspaces.

    Args:
        W_est: Estimated principal components, shape (d, k) or (k, d).
        W_ref: Reference principal components, shape (d, k) or (k, d).
        method: Distance measure:
            - "sin": Mean of sin(theta) for principal angles (default)
            - "sin_max": Maximum sin(theta)
            - "grassmann": Grassmann distance sqrt(sum(theta^2))
            - "projection": 1 - mean(cos(theta))

    Returns:
        Subspace distance (0 = identical, larger = more different).
    """
    # Ensure column matrices (d, k)
    if W_est.ndim == 1:
        W_est = W_est.reshape(-1, 1)
    if W_ref.ndim == 1:
        W_ref = W_ref.reshape(-1, 1)

    # If components are row vectors (k, d), transpose
    if W_est.shape[0] < W_est.shape[1]:
        W_est = W_est.T
    if W_ref.shape[0] < W_ref.shape[1]:
        W_ref = W_ref.T

    # Orthonormalize (in case they aren't exactly orthonormal)
    W_est, _ = np.linalg.qr(W_est)
    W_ref, _ = np.linalg.qr(W_ref)

    # Compute principal angles
    angles = principal_angles(W_est, W_ref)

    if method == "sin":
        return float(np.mean(np.sin(angles)))
    elif method == "sin_max":
        return float(np.max(np.sin(angles)))
    elif method == "grassmann":
        return float(np.sqrt(np.sum(angles ** 2)))
    elif method == "projection":
        return float(1 - np.mean(np.cos(angles)))
    else:
        raise ValueError(f"Unknown method: {method}")


def explained_variance_ratio(
    X: np.ndarray,
    W: np.ndarray,
    center: bool = True
) -> float:
    """
    Compute explained variance ratio.

    The explained variance ratio is:
        EV = mean(||W^T x||^2) / mean(||x||^2)

    where x are centered samples and W has orthonormal columns.

    Args:
        X: Data matrix of shape (n_samples, d).
        W: Principal components, shape (d, k) or (k, d).
        center: If True, center X before computing (default True).

    Returns:
        Explained variance ratio in [0, 1].
    """
    X = np.asarray(X, dtype=np.float64)

    # Ensure W is (d, k)
    if W.ndim == 1:
        W = W.reshape(-1, 1)
    if W.shape[0] < W.shape[1]:
        W = W.T

    # Center if requested
    if center:
        X = X - np.mean(X, axis=0)

    # Total variance: mean(||x||^2)
    total_var = np.mean(np.sum(X ** 2, axis=1))

    if total_var < 1e-10:
        return 1.0  # All zeros, trivially explained

    # Projected variance: mean(||W^T x||^2) = mean(||X @ W||^2)
    projected = X @ W
    proj_var = np.mean(np.sum(projected ** 2, axis=1))

    return float(proj_var / total_var)


def reconstruction_error(
    X: np.ndarray,
    W: np.ndarray,
    center: bool = True,
    mean: Optional[np.ndarray] = None
) -> float:
    """
    Compute mean squared reconstruction error.

    The reconstruction error is:
        MSE = mean(||x - W W^T x||^2)

    This measures how well the principal components capture the data.

    Args:
        X: Data matrix of shape (n_samples, d).
        W: Principal components, shape (d, k) or (k, d).
        center: If True, center X before computing.
        mean: Optional mean vector for centering.

    Returns:
        Mean squared reconstruction error.
    """
    X = np.asarray(X, dtype=np.float64)

    # Ensure W is (d, k)
    if W.ndim == 1:
        W = W.reshape(-1, 1)
    if W.shape[0] < W.shape[1]:
        W = W.T

    # Center if requested
    if center:
        if mean is not None:
            X_centered = X - mean
        else:
            X_centered = X - np.mean(X, axis=0)
    else:
        X_centered = X

    # Project and reconstruct: x_hat = W W^T x
    # X_centered @ W gives projection (n, k)
    # Then @ W.T gives reconstruction (n, d)
    X_proj = X_centered @ W
    X_recon = X_proj @ W.T

    # Reconstruction error
    error = X_centered - X_recon
    mse = np.mean(np.sum(error ** 2, axis=1))

    return float(mse)


def normalized_reconstruction_error(
    X: np.ndarray,
    W: np.ndarray,
    center: bool = True
) -> float:
    """
    Compute normalized reconstruction error.

    Normalized by total variance:
        NMSE = mean(||x - W W^T x||^2) / mean(||x||^2)

    This equals 1 - explained_variance_ratio for orthonormal W.

    Args:
        X: Data matrix of shape (n_samples, d).
        W: Principal components, shape (d, k) or (k, d).
        center: If True, center X before computing.

    Returns:
        Normalized MSE in [0, 1].
    """
    X = np.asarray(X, dtype=np.float64)

    if center:
        X_centered = X - np.mean(X, axis=0)
    else:
        X_centered = X

    total_var = np.mean(np.sum(X_centered ** 2, axis=1))

    if total_var < 1e-10:
        return 0.0

    mse = reconstruction_error(X, W, center=center)

    return float(mse / total_var)


def compute_all_metrics(
    W_est: np.ndarray,
    W_ref: np.ndarray,
    X_test: np.ndarray,
    center: bool = True
) -> dict:
    """
    Compute all evaluation metrics.

    Args:
        W_est: Estimated principal components.
        W_ref: Reference principal components (e.g., from batch PCA).
        X_test: Test data for explained variance and reconstruction error.
        center: If True, center X_test.

    Returns:
        Dictionary with keys:
        - subspace_error: Mean sin of principal angles
        - subspace_error_max: Max sin of principal angles
        - explained_variance: Explained variance ratio of W_est
        - explained_variance_ref: Explained variance ratio of W_ref
        - reconstruction_error: MSE of W_est
        - reconstruction_error_ref: MSE of W_ref
        - normalized_recon_error: Normalized MSE of W_est
    """
    metrics = {}

    # Subspace distances
    metrics["subspace_error"] = subspace_distance(W_est, W_ref, method="sin")
    metrics["subspace_error_max"] = subspace_distance(W_est, W_ref, method="sin_max")

    # Explained variance
    metrics["explained_variance"] = explained_variance_ratio(X_test, W_est, center=center)
    metrics["explained_variance_ref"] = explained_variance_ratio(X_test, W_ref, center=center)

    # Reconstruction error
    metrics["reconstruction_error"] = reconstruction_error(X_test, W_est, center=center)
    metrics["reconstruction_error_ref"] = reconstruction_error(X_test, W_ref, center=center)
    metrics["normalized_recon_error"] = normalized_reconstruction_error(X_test, W_est, center=center)

    return metrics


def batch_pca_reference(
    X: np.ndarray,
    k: int,
    center: bool = True
) -> np.ndarray:
    """
    Compute batch PCA reference using SVD.

    This provides a ground truth for evaluating streaming methods.
    Uses numpy SVD directly (not sklearn) to minimize dependencies.

    Args:
        X: Data matrix of shape (n_samples, d).
        k: Number of principal components.
        center: If True, center X first.

    Returns:
        Principal components of shape (k, d), as row vectors.
    """
    X = np.asarray(X, dtype=np.float64)

    if center:
        X = X - np.mean(X, axis=0)

    # SVD: X = U S V^T
    # Principal components are rows of V (right singular vectors)
    try:
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
    except np.linalg.LinAlgError:
        # Fallback to randomized
        from scipy.linalg import svd
        _, _, Vt = svd(X, full_matrices=False)

    return Vt[:k].copy()
