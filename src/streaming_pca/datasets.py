# Author: Emrullah Erce Dutkan
"""
Dataset loading utilities for streaming PCA experiments.

This module provides loaders for standard datasets that can be used
for benchmarking streaming PCA algorithms. It wraps sklearn datasets
for convenience but keeps the dependency optional.

Available datasets:
- digits: 8x8 handwritten digit images (1797 samples, 64 features)
- mnist_subset: Downsampled MNIST if available
- synthetic: Generated low-rank data

For streaming experiments, these datasets are typically shuffled and
processed sample-by-sample to simulate a stream.
"""

from typing import Tuple, Optional
import numpy as np

from .stream import DatasetStream


def load_digits() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the digits dataset from sklearn.

    Returns:
        Tuple of (X, y) where:
        - X: Feature matrix of shape (1797, 64)
        - y: Labels of shape (1797,)

    Raises:
        ImportError: If sklearn is not available.
    """
    try:
        from sklearn.datasets import load_digits as sklearn_load_digits
    except ImportError:
        raise ImportError(
            "sklearn is required for loading the digits dataset. "
            "Install with: pip install scikit-learn"
        )

    data = sklearn_load_digits()
    return data.data.astype(np.float64), data.target


def load_synthetic(
    n: int = 10000,
    d: int = 100,
    rank: int = 10,
    noise_std: float = 0.1,
    seed: Optional[int] = 42
) -> Tuple[np.ndarray, None]:
    """
    Generate a synthetic low-rank dataset.

    Args:
        n: Number of samples.
        d: Dimensionality.
        rank: Effective rank.
        noise_std: Noise standard deviation.
        seed: Random seed.

    Returns:
        Tuple of (X, None) where X has shape (n, d).
    """
    from .stream import SyntheticStream

    stream = SyntheticStream(
        d=d, rank=rank, noise_std=noise_std, seed=seed
    )
    X = stream.get_batch(n)
    return X, None


def load_random(
    n: int = 10000,
    d: int = 100,
    seed: Optional[int] = 42
) -> Tuple[np.ndarray, None]:
    """
    Generate random Gaussian data (full rank).

    Args:
        n: Number of samples.
        d: Dimensionality.
        seed: Random seed.

    Returns:
        Tuple of (X, None) where X has shape (n, d).
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    return X, None


def load_dataset(
    name: str,
    n: Optional[int] = None,
    d: Optional[int] = None,
    seed: Optional[int] = 42
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load a dataset by name.

    Args:
        name: Dataset name ("digits", "synthetic", "random").
        n: Number of samples (for synthetic datasets).
        d: Dimensionality (for synthetic datasets).
        seed: Random seed.

    Returns:
        Tuple of (X, y) where y may be None.
    """
    if name == "digits":
        return load_digits()
    elif name == "synthetic":
        n = n or 10000
        d = d or 100
        return load_synthetic(n=n, d=d, rank=min(10, d // 2), seed=seed)
    elif name == "random":
        n = n or 10000
        d = d or 100
        return load_random(n=n, d=d, seed=seed)
    else:
        raise ValueError(f"Unknown dataset: {name}")


def create_dataset_stream(
    name: str,
    shuffle: bool = True,
    seed: Optional[int] = 42,
    **kwargs
) -> DatasetStream:
    """
    Create a DatasetStream from a named dataset.

    Args:
        name: Dataset name.
        shuffle: Whether to shuffle samples.
        seed: Random seed.
        **kwargs: Additional arguments for dataset loading.

    Returns:
        DatasetStream object.
    """
    X, _ = load_dataset(name, seed=seed, **kwargs)
    return DatasetStream(X, shuffle=shuffle, seed=seed)


def split_train_test(
    X: np.ndarray,
    test_ratio: float = 0.2,
    seed: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split data into train and test sets.

    Args:
        X: Data matrix.
        test_ratio: Fraction for test set.
        seed: Random seed.

    Returns:
        Tuple of (X_train, X_test).
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    n_test = int(n * test_ratio)

    indices = np.arange(n)
    rng.shuffle(indices)

    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    return X[train_indices], X[test_indices]
