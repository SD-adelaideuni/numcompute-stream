from __future__ import annotations

import numpy as np

from .tree import DecisionTreeClassifier
from .utils import check_X, check_X_y


class RandomForestClassifier:
    """Random Forest built from streaming-capable decision trees.

    Combines N :class:`DecisionTreeClassifier` estimators trained on
    bootstrap samples with per-split feature subsampling (``max_features``).
    Predictions use soft voting (averaged class probabilities).

    Streaming is supported via ``partial_fit``: each incoming chunk is
    bootstrapped independently per tree and passed to that tree's own
    ``partial_fit`` (buffer + rebuild). This keeps the trees decorrelated
    while letting the whole forest adapt as data arrives.

    Parameters
    ----------
    n_estimators : int
        Number of trees. Must be >= 1.
    max_depth : int
        Max depth per tree.
    min_samples_split : int
        Min samples to split per tree.
    max_features : int, str or None
        Per-split feature sampling. Default 'sqrt' (classic RF).
    criterion : str
        'gini' or 'entropy'.
    bootstrap : bool
        If True, sample each tree's chunk with replacement.
    max_buffer : int
        Per-tree streaming buffer cap.
    random_state : int or None
        Seed controlling bootstrap + feature sampling.

    Attributes
    ----------
    estimators_ : list of DecisionTreeClassifier
    classes_ : np.ndarray
    n_features_in_ : int
    """

    def __init__(
        self,
        n_estimators: int = 10,
        max_depth: int = 10,
        min_samples_split: int = 2,
        max_features: int | str | None = "sqrt",
        criterion: str = "gini",
        bootstrap: bool = True,
        max_buffer: int = 10_000,
        random_state: int | None = None,
    ) -> None:
        if n_estimators < 1:
            raise ValueError("n_estimators must be >= 1.")
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.max_features = max_features
        self.criterion = criterion
        self.bootstrap = bool(bootstrap)
        self.max_buffer = int(max_buffer)
        self.random_state = random_state

        self._rng = np.random.default_rng(random_state)
        self.estimators_: list[DecisionTreeClassifier] = []
        self.classes_: np.ndarray | None = None
        self.n_features_in_: int | None = None

    # ------------------------------------------------------------------

    def _make_trees(self) -> None:
        self.estimators_ = []
        for i in range(self.n_estimators):
            seed = None if self.random_state is None else self.random_state + i + 1
            self.estimators_.append(
                DecisionTreeClassifier(
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    max_features=self.max_features,
                    criterion=self.criterion,
                    max_buffer=self.max_buffer,
                    random_state=seed,
                )
            )

    def _bootstrap_idx(self, n: int) -> np.ndarray:
        if self.bootstrap:
            return self._rng.integers(0, n, size=n)
        return np.arange(n)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestClassifier":
        """Train the forest on a full batch.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)

        Returns
        -------
        self : RandomForestClassifier

        Raises
        ------
        ValueError
            If X/y are invalid or inconsistent.

        Time complexity: O(n_estimators * tree_fit_cost)
        """
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self._make_trees()
        n = X.shape[0]
        for tree in self.estimators_:
            idx = self._bootstrap_idx(n)
            tree.fit(X[idx], y[idx])
        return self

    def partial_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        classes: np.ndarray | None = None,
    ) -> "RandomForestClassifier":
        """Update the forest with a new chunk.

        Parameters
        ----------
        X : np.ndarray, shape (n_chunk, n_features)
        y : np.ndarray, shape (n_chunk,)
        classes : np.ndarray or None
            Full class set (recommended on the first call).

        Returns
        -------
        self : RandomForestClassifier

        Raises
        ------
        ValueError
            If X/y invalid, inconsistent, or feature count changes.

        Time complexity: O(n_estimators * tree_partial_fit_cost)
        """
        X, y = check_X_y(X, y)
        if self.n_features_in_ is None:
            self.n_features_in_ = X.shape[1]
        elif X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features but forest was tracking "
                f"{self.n_features_in_} features."
            )

        seen = np.unique(y) if classes is None else np.unique(classes)
        if self.classes_ is None:
            self.classes_ = seen
        else:
            self.classes_ = np.union1d(self.classes_, seen)

        if not self.estimators_:
            self._make_trees()

        n = X.shape[0]
        for tree in self.estimators_:
            idx = self._bootstrap_idx(n)
            tree.partial_fit(X[idx], y[idx], classes=self.classes_)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Soft-vote class probabilities averaged across trees.

        Handles trees whose individual ``classes_`` differ by realigning each
        tree's output onto the forest's global ``classes_``.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples, n_classes)

        Raises
        ------
        RuntimeError
            If called before fitting.

        Time complexity: O(n_estimators * n_samples * depth)
        """
        if not self.estimators_:
            raise RuntimeError("Forest must be fitted before predict_proba.")
        X = check_X(X)
        n_classes = len(self.classes_)
        acc = np.zeros((X.shape[0], n_classes), dtype=float)
        for tree in self.estimators_:
            proba = tree.predict_proba(X)
            # Realign tree.classes_ -> forest.classes_
            col_idx = np.searchsorted(self.classes_, tree.classes_)
            aligned = np.zeros((X.shape[0], n_classes), dtype=float)
            aligned[:, col_idx] = proba
            acc += aligned
        return acc / len(self.estimators_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels by soft voting.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,)

        Time complexity: O(n_estimators * n_samples * depth)
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
