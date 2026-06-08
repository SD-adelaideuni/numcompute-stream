from __future__ import annotations

import sys
import time
from typing import Any

import numpy as np

from .metrics import accuracy, precision, recall, f1
from .utils import check_X_y


def _estimate_size_mb(obj: Any) -> float:
    """Rough in-memory footprint of an object's array attributes, in MB.

    Walks one level of attributes plus list/tuple contents, summing nbytes of
    any NumPy arrays found. This is an approximation for per-chunk logging.
    """
    seen_bytes = 0
    stack = [obj]
    visited = set()
    while stack:
        item = stack.pop()
        oid = id(item)
        if oid in visited:
            continue
        visited.add(oid)
        if isinstance(item, np.ndarray):
            seen_bytes += item.nbytes
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
        elif hasattr(item, "__dict__"):
            stack.extend(vars(item).values())
        elif hasattr(item, "__slots__"):
            for slot in item.__slots__:
                if hasattr(item, slot):
                    stack.append(getattr(item, slot))
    return seen_bytes / (1024 * 1024)


class StreamTrainer:
    """Drive incremental training of a model (or pipeline) over data chunks.

    For each chunk the trainer optionally evaluates *before* learning
    (prequential / test-then-train) and logs per-chunk metrics, wall-clock
    time, model memory footprint, and cumulative accuracy.

    Parameters
    ----------
    model : object
        Anything implementing ``partial_fit(X, y, classes=...)`` and
        ``predict(X)``. May be a bare estimator or a Pipeline.
    metrics : dict[str, callable] or None
        Map of name -> function(y_true, y_pred) -> float. Defaults to
        accuracy / precision / recall / f1 (macro).
    prequential : bool
        If True, predict on each chunk before training on it
        (test-then-train). If False, train first then score the same chunk.
    verbose : bool
        If True, print a one-line summary per chunk.

    Attributes
    ----------
    history_ : dict[str, list]
        Per-chunk logs. Always includes 'chunk', 'n_samples', 'time_sec',
        'model_mb', 'cumulative_accuracy', plus one key per metric.
    """

    def __init__(
        self,
        model: Any,
        metrics: dict | None = None,
        prequential: bool = True,
        verbose: bool = False,
    ) -> None:
        if not callable(getattr(model, "partial_fit", None)):
            raise ValueError("model must implement partial_fit.")
        if not callable(getattr(model, "predict", None)):
            raise ValueError("model must implement predict.")
        self.model = model
        self.prequential = bool(prequential)
        self.verbose = bool(verbose)
        if metrics is None:
            metrics = {
                "accuracy": accuracy,
                "precision": lambda yt, yp: precision(yt, yp, average="macro"),
                "recall": lambda yt, yp: recall(yt, yp, average="macro"),
                "f1": lambda yt, yp: f1(yt, yp, average="macro"),
            }
        self.metrics = metrics
        self.history_: dict[str, list] = {
            "chunk": [],
            "n_samples": [],
            "time_sec": [],
            "model_mb": [],
            "cumulative_accuracy": [],
        }
        for name in self.metrics:
            self.history_[name] = []
        self._cum_correct = 0
        self._cum_total = 0
        self._chunk_idx = 0
        self._classes_seen: np.ndarray | None = None

    def fit_chunk(
        self,
        X: np.ndarray,
        y: np.ndarray,
        classes: np.ndarray | None = None,
    ) -> dict:
        """Train on one chunk and log metrics.

        Parameters
        ----------
        X : np.ndarray, shape (n_chunk, n_features)
        y : np.ndarray, shape (n_chunk,)
        classes : np.ndarray or None
            Full class set; recommended on the first call.

        Returns
        -------
        dict
            The log record for this chunk.

        Time complexity: O(model partial_fit + predict on chunk).
        """
        X, y = check_X_y(X, y)

        # Track the union of classes seen so far.
        seen = np.unique(y) if classes is None else np.unique(classes)
        self._classes_seen = (
            seen if self._classes_seen is None
            else np.union1d(self._classes_seen, seen)
        )

        t0 = time.perf_counter()
        already_fitted = self._chunk_idx > 0

        if self.prequential and already_fitted:
            y_pred = self.model.predict(X)
            self.model.partial_fit(X, y, classes=self._classes_seen)
        else:
            self.model.partial_fit(X, y, classes=self._classes_seen)
            y_pred = self.model.predict(X)
        elapsed = time.perf_counter() - t0

        # Update cumulative accuracy.
        self._cum_correct += int(np.sum(y_pred == y))
        self._cum_total += int(y.shape[0])
        cum_acc = self._cum_correct / self._cum_total if self._cum_total else 0.0

        record = {
            "chunk": self._chunk_idx,
            "n_samples": int(y.shape[0]),
            "time_sec": elapsed,
            "model_mb": _estimate_size_mb(self.model),
            "cumulative_accuracy": cum_acc,
        }
        for name, fn in self.metrics.items():
            try:
                record[name] = float(fn(y, y_pred))
            except Exception:
                record[name] = float("nan")

        for k, v in record.items():
            self.history_[k].append(v)

        if self.verbose:
            msg = (
                f"[chunk {self._chunk_idx:>3}] "
                f"n={record['n_samples']:>4}  "
                f"acc={record.get('accuracy', float('nan')):.3f}  "
                f"cum_acc={cum_acc:.3f}  "
                f"t={elapsed*1000:.1f}ms  "
                f"mem={record['model_mb']:.2f}MB"
            )
            print(msg, file=sys.stdout)

        self._chunk_idx += 1
        return record

    def score_chunk(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Evaluate the current model on a chunk without training.

        Parameters
        ----------
        X : np.ndarray, shape (n_chunk, n_features)
        y : np.ndarray, shape (n_chunk,)

        Returns
        -------
        dict
            Metric name -> value for this chunk.

        Raises
        ------
        RuntimeError
            If the model has not been trained on any chunk yet.

        Time complexity: O(predict on chunk).
        """
        if self._chunk_idx == 0:
            raise RuntimeError("Model has not been trained yet; call fit_chunk first.")
        X, y = check_X_y(X, y)
        y_pred = self.model.predict(X)
        scores = {}
        for name, fn in self.metrics.items():
            try:
                scores[name] = float(fn(y, y_pred))
            except Exception:
                scores[name] = float("nan")
        return scores

    def run(
        self,
        chunks,
        classes: np.ndarray | None = None,
    ) -> dict:
        """Consume an iterable of (X, y) chunks, training on each.

        Parameters
        ----------
        chunks : iterable of (np.ndarray, np.ndarray)
            Stream of (X_chunk, y_chunk).
        classes : np.ndarray or None
            Full class set; forwarded on the first chunk.

        Returns
        -------
        dict
            The full ``history_`` log.

        Time complexity: sum of per-chunk costs.
        """
        first = True
        for X_chunk, y_chunk in chunks:
            self.fit_chunk(
                X_chunk, y_chunk, classes=classes if first else None
            )
            first = False
        return self.history_
