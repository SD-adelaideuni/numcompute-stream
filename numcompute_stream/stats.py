from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Batch functions (carried over from Assignment 1, unchanged behaviour)
# ---------------------------------------------------------------------------

def mean(arr, axis=None):
    """Arithmetic mean, ignoring NaN values.

    Parameters
    ----------
    arr : array-like
        Input data of any shape.
    axis : int or None
        Axis along which the mean is computed. None flattens first.

    Returns
    -------
    np.ndarray or float
        Mean value(s). Shape is ``arr.shape`` with ``axis`` removed.

    Time complexity: O(n)
    """
    arr = np.asarray(arr, dtype=float)
    return np.nanmean(arr, axis=axis)


def median(arr, axis=None):
    """Median, ignoring NaN values.

    Parameters
    ----------
    arr : array-like
        Input data of any shape.
    axis : int or None
        Axis along which the median is computed.

    Returns
    -------
    np.ndarray or float

    Time complexity: O(n log n)
    """
    arr = np.asarray(arr, dtype=float)
    return np.nanmedian(arr, axis=axis)


def std(arr, axis=None, ddof=0):
    """Standard deviation, ignoring NaN values.

    Parameters
    ----------
    arr : array-like
        Input data of any shape.
    axis : int or None
        Reduction axis.
    ddof : int
        Delta degrees of freedom. 0 for population, 1 for sample.

    Returns
    -------
    np.ndarray or float

    Time complexity: O(n)
    """
    arr = np.asarray(arr, dtype=float)
    return np.nanstd(arr, axis=axis, ddof=ddof)


def var(arr, axis=None, ddof=0):
    """Variance, ignoring NaN values.

    Parameters
    ----------
    arr : array-like
        Input data of any shape.
    axis : int or None
        Reduction axis.
    ddof : int
        Delta degrees of freedom.

    Returns
    -------
    np.ndarray or float

    Time complexity: O(n)
    """
    arr = np.asarray(arr, dtype=float)
    return np.nanvar(arr, axis=axis, ddof=ddof)


def minimum(arr, axis=None):
    """Minimum value, ignoring NaN entries.

    Parameters
    ----------
    arr : array-like
        Input data.
    axis : int or None
        Reduction axis.

    Returns
    -------
    np.ndarray or float

    Time complexity: O(n)
    """
    arr = np.asarray(arr, dtype=float)
    return np.nanmin(arr, axis=axis)


def maximum(arr, axis=None):
    """Maximum value, ignoring NaN entries.

    Parameters
    ----------
    arr : array-like
        Input data.
    axis : int or None
        Reduction axis.

    Returns
    -------
    np.ndarray or float

    Time complexity: O(n)
    """
    arr = np.asarray(arr, dtype=float)
    return np.nanmax(arr, axis=axis)


def summary(arr, axis=None):
    """Dictionary of descriptive statistics for quick exploration.

    Parameters
    ----------
    arr : array-like
        Input data.
    axis : int or None
        Axis along which stats are computed. None reduces the whole array.

    Returns
    -------
    dict
        Keys: ``mean``, ``median``, ``std``, ``min``, ``max``, ``count_nan``.

    Time complexity: O(n)
    """
    arr = np.asarray(arr, dtype=float)
    return {
        "mean":      np.nanmean(arr, axis=axis),
        "median":    np.nanmedian(arr, axis=axis),
        "std":       np.nanstd(arr, axis=axis),
        "min":       np.nanmin(arr, axis=axis),
        "max":       np.nanmax(arr, axis=axis),
        "count_nan": int(np.sum(np.isnan(arr))),
    }


def welford_update(existing_aggregate, new_value):
    """One-step Welford online variance update.

    Takes ``(count, mean, M2)`` and a new scalar, returns updated
    ``(count, mean, M2)``.

    Parameters
    ----------
    existing_aggregate : tuple of (int, float, float)
        Current state ``(count, mean, M2)``.
    new_value : float
        Next observation.

    Returns
    -------
    tuple of (int, float, float)
        Updated ``(count, mean, M2)``.

    Time complexity: O(1)
    """
    count, mean_val, m2 = existing_aggregate
    count += 1
    delta = new_value - mean_val
    mean_val += delta / count
    delta2 = new_value - mean_val
    m2 += delta * delta2
    return count, mean_val, m2


def welford_finalize(count, mean_val, m2, ddof=0):
    """Finalise Welford accumulator into ``(mean, variance)``.

    Parameters
    ----------
    count : int
        Total observations processed.
    mean_val : float
        Running mean.
    m2 : float
        Sum of squared deviations.
    ddof : int
        0 for population, 1 for sample variance.

    Returns
    -------
    tuple of (float, float)
        ``(mean, variance)``. Returns ``(nan, nan)`` when ``count <= ddof``.

    Time complexity: O(1)
    """
    if count == 0 or count <= ddof:
        return float("nan"), float("nan")
    variance = m2 / (count - ddof)
    return mean_val, variance


def histogram(arr, bins=10, range=None):
    """Compute a histogram, ignoring NaN values.

    Parameters
    ----------
    arr : array-like
        Input data (flattened internally).
    bins : int
        Number of equal-width bins. Must be >= 1.
    range : tuple of (float, float) or None
        Bin range. None uses ``(nanmin, nanmax)``.

    Returns
    -------
    counts : np.ndarray, shape ``(bins,)``
    bin_edges : np.ndarray, shape ``(bins + 1,)``

    Raises
    ------
    ValueError
        If ``bins < 1``.

    Time complexity: O(n)
    """
    if bins < 1:
        raise ValueError(f"bins must be >= 1, got {bins}.")
    arr = np.asarray(arr, dtype=float)
    clean = arr[~np.isnan(arr)].ravel()
    return np.histogram(clean, bins=bins, range=range)


def quantile(arr, q, axis=None):
    """Compute quantile(s) along a given axis, ignoring NaN values.

    Parameters
    ----------
    arr : array-like
        Input data.
    q : float or array-like
        Quantile(s) in ``[0, 1]``.
    axis : int or None
        Reduction axis.

    Returns
    -------
    np.ndarray or float

    Raises
    ------
    ValueError
        If any value in ``q`` is outside ``[0, 1]``.

    Time complexity: O(n log n)
    """
    arr = np.asarray(arr, dtype=float)
    q = np.asarray(q, dtype=float)
    if np.any((q < 0) | (q > 1)):
        raise ValueError("All quantile values must be in [0, 1].")
    return np.nanpercentile(arr, q * 100, axis=axis)


# ---------------------------------------------------------------------------
# Streaming statistics (new for Assignment 2.2)
# ---------------------------------------------------------------------------

class StreamingStats:
    """Maintain per-feature descriptive statistics over a data stream.

    Tracks running count, mean, M2 (for variance), min, and max per feature
    using the chunked/parallel form of Welford's algorithm. NaNs are ignored
    per feature. Optionally keeps a bounded reservoir per feature to estimate
    quantiles online.

    Parameters
    ----------
    track_quantiles : bool
        If True, maintain a bounded reservoir per feature to support
        ``quantile`` queries. Default False.
    reservoir_size : int
        Max values kept per feature when ``track_quantiles`` is True.

    Attributes
    ----------
    n_samples_seen_ : np.ndarray, shape (n_features,)
        Per-feature count of non-NaN values seen.
    mean_ : np.ndarray, shape (n_features,)
        Running per-feature mean.
    min_ : np.ndarray, shape (n_features,)
    max_ : np.ndarray, shape (n_features,)
    """

    def __init__(
        self,
        track_quantiles: bool = False,
        reservoir_size: int = 10_000,
    ) -> None:
        self.track_quantiles = bool(track_quantiles)
        self.reservoir_size = int(reservoir_size)
        self.n_samples_seen_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self._m2_: np.ndarray | None = None
        self.min_: np.ndarray | None = None
        self.max_: np.ndarray | None = None
        self._reservoir_: list[np.ndarray] | None = None

    def update_stats(self, X_chunk: np.ndarray) -> "StreamingStats":
        """Update running statistics with a new chunk.

        Parameters
        ----------
        X_chunk : np.ndarray, shape (n_chunk, n_features)
            New data chunk. NaNs are ignored per feature.

        Returns
        -------
        self : StreamingStats

        Raises
        ------
        ValueError
            If X_chunk is not 2-D, or feature count changes between chunks.

        Time complexity: O(n_chunk * n_features)
        """
        X = np.asarray(X_chunk, dtype=float)
        if X.ndim != 2:
            raise ValueError("X_chunk must be 2-D (n_samples, n_features).")
        n_features = X.shape[1]

        if self.n_samples_seen_ is None:
            self.n_samples_seen_ = np.zeros(n_features, dtype=float)
            self.mean_ = np.zeros(n_features, dtype=float)
            self._m2_ = np.zeros(n_features, dtype=float)
            self.min_ = np.full(n_features, np.inf)
            self.max_ = np.full(n_features, -np.inf)
            if self.track_quantiles:
                self._reservoir_ = [np.array([], dtype=float) for _ in range(n_features)]
        elif self.mean_.shape[0] != n_features:
            raise ValueError(
                f"X_chunk has {n_features} features but tracker was using "
                f"{self.mean_.shape[0]} features."
            )

        mask = ~np.isnan(X)
        n_b = mask.sum(axis=0).astype(float)
        safe_n_b = np.where(n_b == 0, 1.0, n_b)
        mean_b = np.nansum(X, axis=0) / safe_n_b
        deviations = np.where(mask, X - mean_b, 0.0)
        m2_b = np.sum(deviations ** 2, axis=0)

        n_a = self.n_samples_seen_
        mean_a = self.mean_
        m2_a = self._m2_
        n_ab = n_a + n_b
        safe_n_ab = np.where(n_ab == 0, 1.0, n_ab)
        delta = mean_b - mean_a
        new_mean = mean_a + delta * (n_b / safe_n_ab)
        new_m2 = m2_a + m2_b + (delta ** 2) * (n_a * n_b / safe_n_ab)

        update = n_b > 0
        self.mean_ = np.where(update, new_mean, mean_a)
        self._m2_ = np.where(update, new_m2, m2_a)
        self.n_samples_seen_ = np.where(update, n_ab, n_a)

        # Min / max ignoring NaN; columns with no data this chunk stay put.
        chunk_min = np.where(mask, X, np.inf).min(axis=0)
        chunk_max = np.where(mask, X, -np.inf).max(axis=0)
        self.min_ = np.minimum(self.min_, chunk_min)
        self.max_ = np.maximum(self.max_, chunk_max)

        if self.track_quantiles:
            for j in range(n_features):
                col = X[:, j]
                col = col[~np.isnan(col)]
                if col.size:
                    combined = np.concatenate([self._reservoir_[j], col])
                    if combined.size > self.reservoir_size:
                        combined = combined[-self.reservoir_size :]
                    self._reservoir_[j] = combined
        return self

    def mean(self) -> np.ndarray:
        """Return the running per-feature mean.

        Returns
        -------
        np.ndarray, shape (n_features,)

        Raises
        ------
        RuntimeError
            If no data has been seen yet.
        """
        self._check_seen()
        return self.mean_.copy()

    def variance(self, ddof: int = 0) -> np.ndarray:
        """Return the running per-feature variance.

        Parameters
        ----------
        ddof : int
            Delta degrees of freedom. 0 = population, 1 = sample.

        Returns
        -------
        np.ndarray, shape (n_features,)
            Variance per feature. NaN where count <= ddof.

        Raises
        ------
        RuntimeError
            If no data has been seen yet.
        """
        self._check_seen()
        denom = self.n_samples_seen_ - ddof
        with np.errstate(invalid="ignore", divide="ignore"):
            v = np.where(denom > 0, self._m2_ / denom, np.nan)
        return v

    def std(self, ddof: int = 0) -> np.ndarray:
        """Return the running per-feature standard deviation.

        Parameters
        ----------
        ddof : int
            Delta degrees of freedom.

        Returns
        -------
        np.ndarray, shape (n_features,)
        """
        return np.sqrt(self.variance(ddof=ddof))

    def minimum(self) -> np.ndarray:
        """Return the running per-feature minimum."""
        self._check_seen()
        return self.min_.copy()

    def maximum(self) -> np.ndarray:
        """Return the running per-feature maximum."""
        self._check_seen()
        return self.max_.copy()

    def quantile(self, q) -> np.ndarray:
        """Estimate per-feature quantile(s) from the reservoir.

        Parameters
        ----------
        q : float or array-like
            Quantile(s) in [0, 1].

        Returns
        -------
        np.ndarray
            Estimated quantiles per feature.

        Raises
        ------
        RuntimeError
            If quantile tracking was not enabled or no data seen.
        ValueError
            If any q is outside [0, 1].
        """
        if not self.track_quantiles or self._reservoir_ is None:
            raise RuntimeError(
                "Quantile tracking is disabled. "
                "Construct StreamingStats(track_quantiles=True)."
            )
        q = np.asarray(q, dtype=float)
        scalar_q = q.ndim == 0
        if np.any((q < 0) | (q > 1)):
            raise ValueError("All quantile values must be in [0, 1].")
        q_flat = np.atleast_1d(q)
        cols = []
        for r in self._reservoir_:
            if r.size:
                cols.append(np.percentile(r, q_flat * 100))
            else:
                cols.append(np.full(q_flat.shape, np.nan))
        # result[:, j] is the quantile vector for feature j
        out = np.array(cols).T  # shape (len(q), n_features)
        if scalar_q:
            # one quantile -> return per-feature vector of shape (n_features,)
            return out[0]
        return out

    def result(self) -> dict:
        """Return a dictionary snapshot of all tracked statistics.

        Returns
        -------
        dict
            Keys: ``count``, ``mean``, ``var``, ``std``, ``min``, ``max``.
        """
        self._check_seen()
        return {
            "count": self.n_samples_seen_.copy(),
            "mean": self.mean(),
            "var": self.variance(),
            "std": self.std(),
            "min": self.minimum(),
            "max": self.maximum(),
        }

    def _check_seen(self) -> None:
        if self.n_samples_seen_ is None:
            raise RuntimeError("No data seen yet; call update_stats first.")


class StreamingHistogram:
    """Fixed-range streaming histogram with constant memory.

    Bin edges are fixed up front from ``value_range`` so counts can be
    accumulated across chunks without storing raw data.

    Parameters
    ----------
    bins : int
        Number of equal-width bins. Must be >= 1.
    value_range : tuple of (float, float)
        Lower and upper bound of the histogram range.

    Attributes
    ----------
    counts_ : np.ndarray, shape (bins,)
        Accumulated counts per bin.
    bin_edges_ : np.ndarray, shape (bins + 1,)
        Fixed bin edges.
    """

    def __init__(self, bins: int = 10, value_range: tuple[float, float] = (0.0, 1.0)) -> None:
        if bins < 1:
            raise ValueError(f"bins must be >= 1, got {bins}.")
        lo, hi = value_range
        if hi <= lo:
            raise ValueError("value_range must satisfy lo < hi.")
        self.bins = int(bins)
        self.value_range = (float(lo), float(hi))
        self.bin_edges_ = np.linspace(lo, hi, bins + 1)
        self.counts_ = np.zeros(bins, dtype=int)

    def update(self, x_chunk: np.ndarray) -> "StreamingHistogram":
        """Accumulate counts from a new chunk (NaNs ignored, values clipped).

        Parameters
        ----------
        x_chunk : array-like
            New values (flattened internally).

        Returns
        -------
        self : StreamingHistogram

        Time complexity: O(n_chunk)
        """
        x = np.asarray(x_chunk, dtype=float).ravel()
        x = x[~np.isnan(x)]
        if x.size:
            counts, _ = np.histogram(x, bins=self.bin_edges_)
            self.counts_ = self.counts_ + counts
        return self

    def result(self) -> tuple[np.ndarray, np.ndarray]:
        """Return accumulated ``(counts, bin_edges)``."""
        return self.counts_.copy(), self.bin_edges_.copy()
