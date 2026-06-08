from __future__ import annotations

from typing import Generator

import numpy as np


def check_X(X: np.ndarray, name: str = "X") -> np.ndarray:
    """Validate and coerce a feature matrix to a 2-D float array.

    Parameters
    ----------
    X : array-like
        Input feature matrix.
    name : str
        Variable name used in error messages.

    Returns
    -------
    np.ndarray
        2-D float array of shape (n_samples, n_features).

    Raises
    ------
    ValueError
        If X is not 2-D or is empty.

    Time complexity: O(n)
    """
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(
            f"{name} must be 2-D (n_samples, n_features); got {arr.ndim}-D "
            f"with shape {arr.shape}."
        )
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    return arr


def check_y(y: np.ndarray, name: str = "y") -> np.ndarray:
    """Validate and coerce a label vector to a 1-D array.

    Parameters
    ----------
    y : array-like
        Input label vector.
    name : str
        Variable name used in error messages.

    Returns
    -------
    np.ndarray
        1-D array of shape (n_samples,).

    Raises
    ------
    ValueError
        If y is not 1-D or is empty.

    Time complexity: O(n)
    """
    arr = np.asarray(y)
    if arr.ndim != 1:
        raise ValueError(
            f"{name} must be 1-D (n_samples,); got {arr.ndim}-D "
            f"with shape {arr.shape}."
        )
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    return arr


def check_X_y(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate X and y together and ensure matching sample counts.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    y : array-like, shape (n_samples,)

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        Validated (X, y).

    Raises
    ------
    ValueError
        If the number of samples in X and y differ.

    Time complexity: O(n)
    """
    X = check_X(X)
    y = check_y(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y have inconsistent sample counts: "
            f"X has {X.shape[0]} rows, y has {y.shape[0]}."
        )
    return X, y


def iter_chunks(
    X: np.ndarray,
    y: np.ndarray | None = None,
    chunk_size: int = 100,
    shuffle: bool = False,
    random_state: int | None = None,
) -> Generator[tuple[np.ndarray, np.ndarray | None], None, None]:
    """Yield successive (X, y) chunks to simulate a data stream.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray or None, shape (n_samples,)
        Label vector. If None, y chunks are yielded as None.
    chunk_size : int
        Number of samples per chunk. Must be >= 1.
    shuffle : bool
        If True, shuffle row order before chunking.
    random_state : int or None
        Seed for reproducible shuffling.

    Yields
    ------
    tuple of (np.ndarray, np.ndarray or None)
        (X_chunk, y_chunk).

    Raises
    ------
    ValueError
        If chunk_size < 1.

    Time complexity: O(n) total across all chunks.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}.")
    n = X.shape[0]
    order = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(order)
    for start in range(0, n, chunk_size):
        idx = order[start : start + chunk_size]
        yield X[idx], (y[idx] if y is not None else None)


def gini_impurity(counts: np.ndarray) -> float:
    """Gini impurity from class counts: ``1 - sum(p_i^2)``.

    Parameters
    ----------
    counts : np.ndarray, shape (n_classes,)
        Count of samples per class.

    Returns
    -------
    float
        Gini impurity. Returns 0.0 for empty input.

    Time complexity: O(C)
    """
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    return float(1.0 - np.sum(p ** 2))


def entropy(counts: np.ndarray) -> float:
    """Shannon entropy (base 2) from class counts.

    ``-sum(p_i * log2(p_i))`` ignoring zero-probability classes.

    Parameters
    ----------
    counts : np.ndarray, shape (n_classes,)
        Count of samples per class.

    Returns
    -------
    float
        Entropy in bits. Returns 0.0 for empty or pure input.

    Time complexity: O(C)
    """
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax (max-shifted).

    Parameters
    ----------
    x : np.ndarray
        Input logits.
    axis : int
        Axis along which softmax is computed.

    Returns
    -------
    np.ndarray
        Same shape as x, summing to 1 along ``axis``.

    Time complexity: O(n)
    """
    x = np.asarray(x, dtype=float)
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(x_shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)
