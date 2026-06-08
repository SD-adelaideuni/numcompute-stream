from __future__ import annotations

import numpy as np

from .utils import check_X, check_X_y, gini_impurity, entropy


class _Node:
    """Internal node of a decision tree.

    A node is either a leaf (``feature is None``) holding a class-probability
    vector, or an internal split on ``feature <= threshold``.
    """

    __slots__ = ("feature", "threshold", "left", "right", "proba", "n_samples")

    def __init__(self) -> None:
        self.feature: int | None = None
        self.threshold: float | None = None
        self.left: _Node | None = None
        self.right: _Node | None = None
        self.proba: np.ndarray | None = None
        self.n_samples: int = 0


class DecisionTreeClassifier:
    """Depth-limited binary decision tree (Gini or entropy split criterion).

    Streaming is supported via ``partial_fit``, which buffers incoming chunks
    and rebuilds the tree on the accumulated buffer. This "mini-batch
    retrain" design is a deliberate, documented choice: a fully incremental
    tree (e.g. Hoeffding tree) is out of scope, and rebuilding on a bounded
    buffer gives stable, correct behaviour under streaming while keeping the
    splitting logic simple and vectorised.

    Parameters
    ----------
    max_depth : int
        Maximum tree depth. Must be >= 1.
    min_samples_split : int
        Minimum samples required to split an internal node. Must be >= 2.
    max_features : int, str or None
        Features considered per split:
        - None  -> all features
        - 'sqrt' -> floor(sqrt(n_features))
        - 'log2' -> floor(log2(n_features))
        - int   -> that many features
        Used to enable Random Forest feature subsampling.
    criterion : str
        'gini' or 'entropy'.
    max_buffer : int
        Max samples retained for streaming rebuilds in ``partial_fit``.
    random_state : int or None
        Seed for reproducible feature subsampling / tie-breaking.

    Attributes
    ----------
    classes_ : np.ndarray
        Sorted array of class labels seen during training.
    root_ : _Node
        Root node of the fitted tree.
    n_features_in_ : int
        Number of input features.
    """

    def __init__(
        self,
        max_depth: int = 10,
        min_samples_split: int = 2,
        max_features: int | str | None = None,
        criterion: str = "gini",
        max_buffer: int = 10_000,
        random_state: int | None = None,
    ) -> None:
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1.")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be >= 2.")
        if criterion not in ("gini", "entropy"):
            raise ValueError("criterion must be 'gini' or 'entropy'.")
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.max_features = max_features
        self.criterion = criterion
        self.max_buffer = int(max_buffer)
        self.random_state = random_state

        self.classes_: np.ndarray | None = None
        self.root_: _Node | None = None
        self.n_features_in_: int | None = None
        # Streaming buffer
        self._buf_X: np.ndarray | None = None
        self._buf_y: np.ndarray | None = None
        self._rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        """Build the tree from a full batch (resets any streaming buffer).

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)

        Returns
        -------
        self : DecisionTreeClassifier

        Raises
        ------
        ValueError
            If X/y are invalid or inconsistent.

        Time complexity: O(n_features * n_samples * log(n_samples) * 2^depth)
        in the worst case.
        """
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.root_ = self._build(X, y, depth=0)
        # Reset buffer to match the batch just fitted.
        self._buf_X = X.copy()
        self._buf_y = y.copy()
        self._trim_buffer()
        return self

    def partial_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        classes: np.ndarray | None = None,
    ) -> "DecisionTreeClassifier":
        """Update the tree with a new chunk (buffer + rebuild).

        On the first call, ``classes`` should be supplied if the first chunk
        may not contain every label; otherwise classes are inferred and
        extended as new labels appear.

        Parameters
        ----------
        X : np.ndarray, shape (n_chunk, n_features)
        y : np.ndarray, shape (n_chunk,)
        classes : np.ndarray or None
            Full set of class labels (recommended on the first call).

        Returns
        -------
        self : DecisionTreeClassifier

        Raises
        ------
        ValueError
            If X/y are invalid, inconsistent, or feature count changes.

        Time complexity: O(rebuild on buffer) per call.
        """
        X, y = check_X_y(X, y)

        if self.n_features_in_ is None:
            self.n_features_in_ = X.shape[1]
        elif X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features but tree was tracking "
                f"{self.n_features_in_} features."
            )

        # Maintain the known class set.
        seen = np.unique(y) if classes is None else np.unique(classes)
        if self.classes_ is None:
            self.classes_ = seen
        else:
            self.classes_ = np.union1d(self.classes_, seen)

        # Append to buffer.
        if self._buf_X is None:
            self._buf_X = X.copy()
            self._buf_y = y.copy()
        else:
            self._buf_X = np.vstack([self._buf_X, X])
            self._buf_y = np.concatenate([self._buf_y, y])
        self._trim_buffer()

        # Rebuild on the accumulated buffer.
        self.root_ = self._build(self._buf_X, self._buf_y, depth=0)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples, n_classes)
            Per-class probabilities aligned with ``classes_``.

        Raises
        ------
        RuntimeError
            If called before fitting.

        Time complexity: O(n_samples * depth)
        """
        if self.root_ is None:
            raise RuntimeError("Tree must be fitted before predict_proba.")
        X = check_X(X)
        out = np.empty((X.shape[0], len(self.classes_)), dtype=float)
        for i in range(X.shape[0]):
            node = self.root_
            row = X[i]
            while node.feature is not None:
                node = node.left if row[node.feature] <= node.threshold else node.right
            out[i] = node.proba
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,)
            Predicted labels from ``classes_``.

        Time complexity: O(n_samples * depth)
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    # ------------------------------------------------------------------
    # Internal building
    # ------------------------------------------------------------------

    def _trim_buffer(self) -> None:
        if self._buf_X is not None and self._buf_X.shape[0] > self.max_buffer:
            self._buf_X = self._buf_X[-self.max_buffer :]
            self._buf_y = self._buf_y[-self.max_buffer :]

    def _class_counts(self, y: np.ndarray) -> np.ndarray:
        # Counts aligned to self.classes_
        counts = np.zeros(len(self.classes_), dtype=float)
        idx = np.searchsorted(self.classes_, y)
        np.add.at(counts, idx, 1)
        return counts

    def _leaf(self, y: np.ndarray) -> _Node:
        node = _Node()
        counts = self._class_counts(y)
        total = counts.sum()
        node.proba = counts / total if total > 0 else np.full_like(counts, 1.0 / len(counts))
        node.n_samples = int(y.shape[0])
        return node

    def _impurity(self, counts: np.ndarray) -> float:
        return gini_impurity(counts) if self.criterion == "gini" else entropy(counts)

    def _n_features_to_consider(self, n_features: int) -> int:
        mf = self.max_features
        if mf is None:
            return n_features
        if mf == "sqrt":
            return max(1, int(np.floor(np.sqrt(n_features))))
        if mf == "log2":
            return max(1, int(np.floor(np.log2(n_features))))
        if isinstance(mf, (int, np.integer)):
            return max(1, min(int(mf), n_features))
        raise ValueError(f"Invalid max_features: {mf!r}")

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        """Find the best (feature, threshold) minimising child impurity.

        Returns (feature, threshold, gain) or (None, None, 0) if no split
        improves on the parent impurity.
        """
        n_samples, n_features = X.shape
        parent_counts = self._class_counts(y)
        parent_imp = self._impurity(parent_counts)

        k = self._n_features_to_consider(n_features)
        if k < n_features:
            feat_idx = self._rng.choice(n_features, size=k, replace=False)
        else:
            feat_idx = np.arange(n_features)

        best_gain = 0.0
        best_feat = None
        best_thr = None

        for f in feat_idx:
            col = X[:, f]
            order = np.argsort(col, kind="stable")
            col_sorted = col[order]
            y_sorted = y[order]

            # Candidate thresholds = midpoints between distinct consecutive values
            distinct = np.where(np.diff(col_sorted) > 0)[0]
            if distinct.size == 0:
                continue  # constant feature, no split

            y_idx = np.searchsorted(self.classes_, y_sorted)
            n_classes = len(self.classes_)
            onehot = np.zeros((n_samples, n_classes), dtype=float)
            onehot[np.arange(n_samples), y_idx] = 1.0
            cum = np.cumsum(onehot, axis=0)  # left counts at each cut
            total = cum[-1]

            for split_pos in distinct:
                left_counts = cum[split_pos]
                right_counts = total - left_counts
                n_left = left_counts.sum()
                n_right = right_counts.sum()
                if n_left == 0 or n_right == 0:
                    continue
                imp_left = self._impurity(left_counts)
                imp_right = self._impurity(right_counts)
                weighted = (n_left * imp_left + n_right * imp_right) / n_samples
                gain = parent_imp - weighted
                if gain > best_gain:
                    best_gain = gain
                    best_feat = int(f)
                    best_thr = float(
                        (col_sorted[split_pos] + col_sorted[split_pos + 1]) / 2.0
                    )
        return best_feat, best_thr, best_gain

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        # Stopping conditions
        if (
            depth >= self.max_depth
            or X.shape[0] < self.min_samples_split
            or np.unique(y).size == 1
        ):
            return self._leaf(y)

        feat, thr, gain = self._best_split(X, y)
        if feat is None or gain <= 0.0:
            return self._leaf(y)

        mask = X[:, feat] <= thr
        if not mask.any() or mask.all():
            return self._leaf(y)

        node = _Node()
        node.feature = feat
        node.threshold = thr
        node.n_samples = int(X.shape[0])
        node.left = self._build(X[mask], y[mask], depth + 1)
        node.right = self._build(X[~mask], y[~mask], depth + 1)
        return node
