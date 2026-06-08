from __future__ import annotations
import warnings
import numpy as np


class StandardScaler:
    """Per-feature standardisation: ``(X - mean) / scale`` with safe division.

    Supports both batch ``fit`` and streaming ``partial_fit``. The streaming
    path maintains a running mean and M2 (sum of squared deviations) per
    feature using Welford's algorithm, so the scaler can be updated chunk by
    chunk without storing past data.
    """

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        # Streaming state 
        self.n_samples_seen_: np.ndarray | None = None
        self._m2_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> StandardScaler:
        """Compute mean/scale from a full batch (resets any streaming state).

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        self : StandardScaler

        Raises
        ------
        ValueError
            If X is not 2-D.

        Time complexity: O(n_samples * n_features)
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2-D with shape (n_samples, n_features).")
        self.mean_ = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0)
        if np.any(std == 0):
            warnings.warn(
                "Zero standard deviation detected; scale set to 1 for those features.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.scale_ = np.where(std == 0, 1.0, std)
        # Sync streaming state 
        n = np.sum(~np.isnan(X), axis=0).astype(float)
        self.n_samples_seen_ = n
        self._m2_ = np.nanvar(X, axis=0) * n
        return self

    def partial_fit(self, X: np.ndarray) -> StandardScaler:
        """Update running mean/variance with a new chunk (Welford, vectorised).

        Uses the parallel/chunked form of Welford's algorithm to merge the
        new chunk's statistics into the running accumulator. NaNs are ignored
        per feature.

        Parameters
        ----------
        X : np.ndarray, shape (n_chunk, n_features)
            New data chunk.

        Returns
        -------
        self : StandardScaler

        Raises
        ------
        ValueError
            If X is not 2-D, or if the feature count differs from prior chunks.

        Time complexity: O(n_chunk * n_features)
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2-D with shape (n_samples, n_features).")
        n_features = X.shape[1]

        if self.n_samples_seen_ is None:
            self.n_samples_seen_ = np.zeros(n_features, dtype=float)
            self.mean_ = np.zeros(n_features, dtype=float)
            self._m2_ = np.zeros(n_features, dtype=float)
        elif self.mean_.shape[0] != n_features:
            raise ValueError(
                f"X has {n_features} features but scaler was tracking "
                f"{self.mean_.shape[0]} features."
            )

        # Per-feature chunk statistics ignoring NaN
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

   
        safe_seen = np.where(self.n_samples_seen_ == 0, 1.0, self.n_samples_seen_)
        var = self._m2_ / safe_seen
        std = np.sqrt(var)
        self.scale_ = np.where(std == 0, 1.0, std)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Standardise X using the fitted mean/scale.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples, n_features)

        Raises
        ------
        RuntimeError
            If the scaler has not been fitted.

        Time complexity: O(n_samples * n_features)
        """
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler must be fitted before transform.")
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class OneHotEncoder:
    """One-hot expansion for discrete columns (vectorised column blocks).

    Supports streaming ``partial_fit`` which expands the known category set
    incrementally as new categories appear in later chunks.
    """

    def __init__(self) -> None:
        self.categories_: list[np.ndarray] | None = None

    def fit(self, X: np.ndarray) -> OneHotEncoder:
        """Learn category sets from a full batch (resets prior state).

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        self : OneHotEncoder

        Raises
        ------
        ValueError
            If X is not 2-D.

        Time complexity: O(n_samples * n_features)
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be 2-D with shape (n_samples, n_features).")
        self.categories_ = [np.unique(X[:, j]) for j in range(X.shape[1])]
        return self

    def partial_fit(self, X: np.ndarray) -> OneHotEncoder:
        """Expand category sets with any new categories in this chunk.

        Parameters
        ----------
        X : np.ndarray, shape (n_chunk, n_features)

        Returns
        -------
        self : OneHotEncoder

        Raises
        ------
        ValueError
            If X is not 2-D, or feature count differs from prior chunks.

        Time complexity: O(n_chunk * n_features * log)
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be 2-D with shape (n_samples, n_features).")
        if self.categories_ is None:
            self.categories_ = [np.unique(X[:, j]) for j in range(X.shape[1])]
            return self
        if len(self.categories_) != X.shape[1]:
            raise ValueError(
                f"X has {X.shape[1]} features but encoder was tracking "
                f"{len(self.categories_)} features."
            )
        for j in range(X.shape[1]):
            merged = np.union1d(self.categories_[j], np.unique(X[:, j]))
            self.categories_[j] = merged
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """One-hot encode X using the known category sets.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples, total_categories)

        Raises
        ------
        RuntimeError
            If the encoder has not been fitted.

        Time complexity: O(n_samples * total_categories)
        """
        if self.categories_ is None:
            raise RuntimeError("OneHotEncoder must be fitted before transform.")
        X = np.asarray(X)
        blocks: list[np.ndarray] = []
        for j, cats in enumerate(self.categories_):
            col = X[:, j]
            blocks.append((col[:, None] == cats[None, :]).astype(float))
        return np.hstack(blocks) if blocks else np.empty((X.shape[0], 0))

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class MinMaxScaler:
    """Per-feature min-max scaling to a target range.

    Supports streaming ``partial_fit`` by tracking running per-feature
    min and max across chunks.
    """

    def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0)) -> None:
        lo, hi = feature_range
        if hi <= lo:
            raise ValueError("feature_range must satisfy min < max.")
        self.feature_range = (float(lo), float(hi))
        self.data_min_: np.ndarray | None = None
        self.data_max_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "MinMaxScaler":
        """Compute min/max from a full batch (resets prior state).

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        self : MinMaxScaler

        Raises
        ------
        ValueError
            If X is not 2-D.

        Time complexity: O(n_samples * n_features)
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2-D with shape (n_samples, n_features).")
        self.data_min_ = np.nanmin(X, axis=0)
        self.data_max_ = np.nanmax(X, axis=0)
        self._refresh_scale()
        return self

    def partial_fit(self, X: np.ndarray) -> "MinMaxScaler":
        """Update running min/max with a new chunk.

        Parameters
        ----------
        X : np.ndarray, shape (n_chunk, n_features)

        Returns
        -------
        self : MinMaxScaler

        Raises
        ------
        ValueError
            If X is not 2-D, or feature count differs from prior chunks.

        Time complexity: O(n_chunk * n_features)
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2-D with shape (n_samples, n_features).")
        chunk_min = np.nanmin(X, axis=0)
        chunk_max = np.nanmax(X, axis=0)
        if self.data_min_ is None:
            self.data_min_ = chunk_min
            self.data_max_ = chunk_max
        else:
            if self.data_min_.shape[0] != X.shape[1]:
                raise ValueError(
                    f"X has {X.shape[1]} features but scaler was tracking "
                    f"{self.data_min_.shape[0]} features."
                )
            self.data_min_ = np.minimum(self.data_min_, chunk_min)
            self.data_max_ = np.maximum(self.data_max_, chunk_max)
        self._refresh_scale()
        return self

    def _refresh_scale(self) -> None:
        span = self.data_max_ - self.data_min_
        self.scale_ = np.where(span == 0.0, 1.0, span)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale X to ``feature_range`` using the fitted min/max.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples, n_features)

        Raises
        ------
        RuntimeError
            If the scaler has not been fitted.

        Time complexity: O(n_samples * n_features)
        """
        if self.data_min_ is None or self.data_max_ is None or self.scale_ is None:
            raise RuntimeError("MinMaxScaler must be fitted before transform.")
        X = np.asarray(X, dtype=float)
        lo, hi = self.feature_range
        X_std = (X - self.data_min_) / self.scale_
        X_scaled = X_std * (hi - lo) + lo
        constant = self.data_max_ == self.data_min_
        if np.any(constant):
            X_scaled[:, constant] = 0.0
        return X_scaled

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class SimpleImputer:
    """Fill NaNs with per-feature statistics or constants.

    Supports streaming ``partial_fit``. For ``mean`` strategy a running mean
    is maintained per feature; for ``median`` a bounded reservoir of recent
    values is kept per feature to estimate the median online; ``constant``
    needs no state.
    """

    def __init__(
        self,
        strategy: str = "mean",
        fill_value: float = 0.0,
        reservoir_size: int = 10_000,
    ) -> None:
        if strategy not in ("mean", "median", "constant"):
            raise ValueError("strategy must be 'mean', 'median', or 'constant'.")
        self.strategy = strategy
        self.fill_value = float(fill_value)
        self.reservoir_size = int(reservoir_size)
        self.statistics_: np.ndarray | None = None
        # Streaming state
        self._count_: np.ndarray | None = None   
        self._sum_: np.ndarray | None = None      
        self._reservoir_: list[np.ndarray] | None = None 

    def fit(self, X: np.ndarray) -> "SimpleImputer":
        """Compute fill statistics from a full batch (resets prior state).

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        self : SimpleImputer

        Raises
        ------
        ValueError
            If X is not 2-D.

        Time complexity: O(n_samples * n_features)
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2-D with shape (n_samples, n_features).")
        if self.strategy == "mean":
            stats = np.nanmean(X, axis=0)
        elif self.strategy == "median":
            stats = np.nanmedian(X, axis=0)
        else:
            stats = np.full(X.shape[1], self.fill_value, dtype=float)
        self.statistics_ = np.where(np.isnan(stats), 0.0, stats)
        return self

    def partial_fit(self, X: np.ndarray) -> "SimpleImputer":
        """Update fill statistics with a new chunk.

        Parameters
        ----------
        X : np.ndarray, shape (n_chunk, n_features)

        Returns
        -------
        self : SimpleImputer

        Raises
        ------
        ValueError
            If X is not 2-D.

        Time complexity: O(n_chunk * n_features)
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2-D with shape (n_samples, n_features).")
        n_features = X.shape[1]

        if self.strategy == "constant":
            self.statistics_ = np.full(n_features, self.fill_value, dtype=float)
            return self

        if self.strategy == "mean":
            mask = ~np.isnan(X)
            chunk_sum = np.nansum(X, axis=0)
            chunk_count = mask.sum(axis=0).astype(float)
            if self._sum_ is None:
                self._sum_ = np.zeros(n_features, dtype=float)
                self._count_ = np.zeros(n_features, dtype=float)
            self._sum_ += chunk_sum
            self._count_ += chunk_count
            safe = np.where(self._count_ == 0, 1.0, self._count_)
            stats = self._sum_ / safe
            self.statistics_ = np.where(self._count_ == 0, 0.0, stats)
            return self


        if self._reservoir_ is None:
            self._reservoir_ = [np.array([], dtype=float) for _ in range(n_features)]
        for j in range(n_features):
            col = X[:, j]
            col = col[~np.isnan(col)]
            if col.size:
                combined = np.concatenate([self._reservoir_[j], col])
                if combined.size > self.reservoir_size:
                    combined = combined[-self.reservoir_size :]
                self._reservoir_[j] = combined
        stats = np.array(
            [
                np.median(r) if r.size else 0.0
                for r in self._reservoir_
            ],
            dtype=float,
        )
        self.statistics_ = stats
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Replace NaNs in X with the fitted fill statistics.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples, n_features)

        Raises
        ------
        RuntimeError
            If the imputer has not been fitted.

        Time complexity: O(n_samples * n_features)
        """
        if self.statistics_ is None:
            raise RuntimeError("SimpleImputer must be fitted before transform.")
        X = np.asarray(X, dtype=float).copy()
        nan_mask = np.isnan(X)
        if np.any(nan_mask):
            X[nan_mask] = np.take(self.statistics_, np.where(nan_mask)[1])
        return X

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
