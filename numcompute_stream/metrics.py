from __future__ import annotations

from collections import deque

import numpy as np


def _validate_inputs(y_true, y_pred):
    """Check that y_true and y_pred are 1-D arrays of equal length."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError(
            f"y_true and y_pred must be 1-D. "
            f"Got shapes {y_true.shape} and {y_pred.shape}."
        )
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            f"Length mismatch: {y_true.shape[0]} vs {y_pred.shape[0]}."
        )
    return y_true, y_pred


# ---------------------------------------------------------------------------
# Batch metrics (carried over from Assignment 1)
# ---------------------------------------------------------------------------

def accuracy(y_true, y_pred):
    """Proportion of correct predictions.

    Parameters
    ----------
    y_true : array-like, shape (n,)
    y_pred : array-like, shape (n,)

    Returns
    -------
    float
        Accuracy in [0.0, 1.0].

    Raises
    ------
    ValueError
        If inputs are not 1-D or have different lengths.

    Time complexity: O(n)
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true, y_pred, labels=None):
    """Build a (C x C) confusion matrix.

    Parameters
    ----------
    y_true : array-like, shape (n,)
    y_pred : array-like, shape (n,)
    labels : array-like or None
        Optional fixed class ordering. If None, inferred from the data.

    Returns
    -------
    np.ndarray, shape (C, C)
        ``cm[i, j]`` counts true label i predicted as j.

    Raises
    ------
    ValueError
        If inputs are not 1-D or have different lengths.

    Time complexity: O(n + C^2)
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    if labels is None:
        classes = np.unique(np.concatenate([y_true, y_pred]))
    else:
        classes = np.asarray(labels)
    n_classes = len(classes)
    label_to_idx = {label: idx for idx, label in enumerate(classes)}
    true_idx = np.array([label_to_idx[l] for l in y_true])
    pred_idx = np.array([label_to_idx[l] for l in y_pred])
    cm = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(cm, (true_idx, pred_idx), 1)
    return cm


def precision(y_true, y_pred, average="macro"):
    """Precision: TP / (TP + FP), per class then averaged.

    Parameters
    ----------
    y_true : array-like, shape (n,)
    y_pred : array-like, shape (n,)
    average : str
        'macro' for unweighted mean of per-class precision.
        'binary' for positive-class precision (requires exactly 2 classes).

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If inputs invalid or average='binary' with more than 2 classes.

    Time complexity: O(n * C)
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    if average == "binary":
        if len(classes) > 2:
            raise ValueError(
                f"average='binary' requires exactly 2 classes, found {len(classes)}."
            )
        pos = classes[-1]
        tp = float(np.sum((y_pred == pos) & (y_true == pos)))
        fp = float(np.sum((y_pred == pos) & (y_true != pos)))
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0
    per_class = np.zeros(len(classes))
    for i, cls in enumerate(classes):
        tp = float(np.sum((y_pred == cls) & (y_true == cls)))
        fp = float(np.sum((y_pred == cls) & (y_true != cls)))
        per_class[i] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    return float(np.mean(per_class))


def recall(y_true, y_pred, average="macro"):
    """Recall: TP / (TP + FN), per class then averaged.

    Parameters
    ----------
    y_true : array-like, shape (n,)
    y_pred : array-like, shape (n,)
    average : str
        'macro' or 'binary'.

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If inputs invalid or average='binary' with more than 2 classes.

    Time complexity: O(n * C)
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    if average == "binary":
        if len(classes) > 2:
            raise ValueError(
                f"average='binary' requires exactly 2 classes, found {len(classes)}."
            )
        pos = classes[-1]
        tp = float(np.sum((y_pred == pos) & (y_true == pos)))
        fn = float(np.sum((y_pred != pos) & (y_true == pos)))
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0
    per_class = np.zeros(len(classes))
    for i, cls in enumerate(classes):
        tp = float(np.sum((y_pred == cls) & (y_true == cls)))
        fn = float(np.sum((y_pred != cls) & (y_true == cls)))
        per_class[i] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return float(np.mean(per_class))


def f1(y_true, y_pred, average="macro"):
    """F1 score: harmonic mean of precision and recall.

    Parameters
    ----------
    y_true : array-like, shape (n,)
    y_pred : array-like, shape (n,)
    average : str
        'macro' or 'binary'.

    Returns
    -------
    float

    Time complexity: O(n * C)
    """
    p = precision(y_true, y_pred, average=average)
    r = recall(y_true, y_pred, average=average)
    return float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0


def mse(y_true, y_pred):
    """Mean squared error for regression.

    Parameters
    ----------
    y_true : array-like, shape (n,)
    y_pred : array-like, shape (n,)

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If inputs are not 1-D or have different lengths.

    Time complexity: O(n)
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def roc_curve(y_true_binary, y_scores):
    """ROC curve for a binary classifier.

    Parameters
    ----------
    y_true_binary : array-like, shape (n,)
        Binary ground-truth labels (0 or 1).
    y_scores : array-like, shape (n,)
        Continuous scores for the positive class.

    Returns
    -------
    fpr : np.ndarray
    tpr : np.ndarray
    thresholds : np.ndarray

    Raises
    ------
    ValueError
        If inputs are not 1-D, different lengths, or not binary.

    Time complexity: O(n log n)
    """
    y_true_binary, y_scores = _validate_inputs(y_true_binary, y_scores)
    unique_labels = np.unique(y_true_binary)
    if len(unique_labels) != 2:
        raise ValueError(
            f"roc_curve requires exactly 2 classes, found {len(unique_labels)}."
        )
    desc_idx = np.argsort(-y_scores)
    y_sorted = y_true_binary[desc_idx]
    thresholds = y_scores[desc_idx]
    total_pos = float(np.sum(y_true_binary == 1))
    total_neg = float(np.sum(y_true_binary == 0))
    tp_cumsum = np.cumsum(y_sorted == 1).astype(float)
    fp_cumsum = np.cumsum(y_sorted == 0).astype(float)
    tpr = tp_cumsum / total_pos if total_pos > 0 else np.zeros_like(tp_cumsum)
    fpr = fp_cumsum / total_neg if total_neg > 0 else np.zeros_like(fp_cumsum)
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])
    thresholds = np.concatenate([[thresholds[0] + 1], thresholds])
    return fpr, tpr, thresholds


def auc(fpr, tpr):
    """Area under the ROC curve via the trapezoidal rule.

    Parameters
    ----------
    fpr : array-like, shape (n,)
    tpr : array-like, shape (n,)

    Returns
    -------
    float
        AUC value. 1.0 = perfect, ~0.5 = random.

    Time complexity: O(n)
    """
    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)
    # np.trapezoid (NumPy >= 2.0); fall back to trapz on older versions.
    trap = getattr(np, "trapezoid", None) or np.trapz
    return float(trap(tpr, fpr))


# ---------------------------------------------------------------------------
# Streaming metrics (new for Assignment 2.2)
# ---------------------------------------------------------------------------

class StreamingConfusionMatrix:
    """Confusion matrix that accumulates over streaming chunks.

    Parameters
    ----------
    labels : array-like
        Fixed, ordered list of all possible class labels. Required so the
        matrix shape stays consistent across chunks.

    Attributes
    ----------
    matrix_ : np.ndarray, shape (C, C)
        Accumulated counts. ``matrix_[i, j]`` = true i predicted j.
    """

    def __init__(self, labels) -> None:
        self.labels = np.asarray(labels)
        self._index = {label: i for i, label in enumerate(self.labels)}
        c = len(self.labels)
        self.matrix_ = np.zeros((c, c), dtype=int)

    def update(self, y_true_chunk, y_pred_chunk) -> "StreamingConfusionMatrix":
        """Add a chunk of predictions to the matrix.

        Parameters
        ----------
        y_true_chunk : array-like, shape (n,)
        y_pred_chunk : array-like, shape (n,)

        Returns
        -------
        self : StreamingConfusionMatrix

        Raises
        ------
        ValueError
            If inputs invalid or contain unknown labels.

        Time complexity: O(n)
        """
        y_true, y_pred = _validate_inputs(y_true_chunk, y_pred_chunk)
        try:
            ti = np.array([self._index[v] for v in y_true])
            pi = np.array([self._index[v] for v in y_pred])
        except KeyError as exc:
            raise ValueError(f"Encountered label not in `labels`: {exc}") from None
        np.add.at(self.matrix_, (ti, pi), 1)
        return self

    def reset(self) -> "StreamingConfusionMatrix":
        """Zero out the accumulated matrix."""
        self.matrix_[:] = 0
        return self

    def result(self) -> np.ndarray:
        """Return a copy of the accumulated confusion matrix."""
        return self.matrix_.copy()


class StreamingClassificationMetrics:
    """Accumulate accuracy/precision/recall/F1 over streaming chunks.

    Maintains a running confusion matrix internally, so all four metrics are
    derived consistently at any point via :meth:`result`.

    Parameters
    ----------
    labels : array-like
        Fixed, ordered list of all class labels.
    average : str
        'macro' or 'binary' for precision/recall/F1. Default 'macro'.
    """

    def __init__(self, labels, average: str = "macro") -> None:
        if average not in ("macro", "binary"):
            raise ValueError("average must be 'macro' or 'binary'.")
        self.labels = np.asarray(labels)
        if average == "binary" and len(self.labels) != 2:
            raise ValueError("average='binary' requires exactly 2 labels.")
        self.average = average
        self._cm = StreamingConfusionMatrix(self.labels)

    def update(self, y_true_chunk, y_pred_chunk) -> "StreamingClassificationMetrics":
        """Update the running confusion matrix with a chunk.

        Parameters
        ----------
        y_true_chunk : array-like, shape (n,)
        y_pred_chunk : array-like, shape (n,)

        Returns
        -------
        self

        Time complexity: O(n)
        """
        self._cm.update(y_true_chunk, y_pred_chunk)
        return self

    def reset(self) -> "StreamingClassificationMetrics":
        """Reset all accumulated counts."""
        self._cm.reset()
        return self

    def result(self) -> dict:
        """Compute current metrics from the accumulated confusion matrix.

        Returns
        -------
        dict
            Keys: ``accuracy``, ``precision``, ``recall``, ``f1``.

        Time complexity: O(C^2)
        """
        cm = self._cm.matrix_
        total = cm.sum()
        acc = float(np.trace(cm) / total) if total > 0 else 0.0

        tp = np.diag(cm).astype(float)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp

        with np.errstate(invalid="ignore", divide="ignore"):
            prec_pc = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)
            rec_pc = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)

        if self.average == "binary":
            # positive class = last label (matches A1 convention)
            p = float(prec_pc[-1])
            r = float(rec_pc[-1])
        else:
            p = float(np.mean(prec_pc))
            r = float(np.mean(rec_pc))
        f = float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        return {"accuracy": acc, "precision": p, "recall": r, "f1": f}


class RollingAccuracy:
    """Accuracy over a sliding window of the most recent N samples.

    Parameters
    ----------
    window : int
        Number of most-recent samples to retain. Must be >= 1.

    Attributes
    ----------
    window : int
    """

    def __init__(self, window: int = 1000) -> None:
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}.")
        self.window = int(window)
        self._buf: deque[int] = deque(maxlen=self.window)

    def update(self, y_true_chunk, y_pred_chunk) -> "RollingAccuracy":
        """Add a chunk of correctness flags to the rolling window.

        Parameters
        ----------
        y_true_chunk : array-like, shape (n,)
        y_pred_chunk : array-like, shape (n,)

        Returns
        -------
        self

        Time complexity: O(n)
        """
        y_true, y_pred = _validate_inputs(y_true_chunk, y_pred_chunk)
        correct = (y_true == y_pred).astype(int)
        self._buf.extend(correct.tolist())
        return self

    def reset(self) -> "RollingAccuracy":
        """Empty the rolling window."""
        self._buf.clear()
        return self

    def result(self) -> float:
        """Return accuracy over the current window (0.0 if empty)."""
        if not self._buf:
            return 0.0
        return float(np.mean(self._buf))


class StreamingAUC:
    """Approximate streaming AUC for binary classification.

    Keeps a bounded reservoir of (score, label) pairs and computes AUC over
    the retained sample on demand. Exact AUC needs all pairs, so this is an
    approximation when the stream exceeds ``reservoir_size``.

    Parameters
    ----------
    reservoir_size : int
        Max (score, label) pairs retained. Default 10000.
    """

    def __init__(self, reservoir_size: int = 10_000) -> None:
        self.reservoir_size = int(reservoir_size)
        self._scores = np.array([], dtype=float)
        self._labels = np.array([], dtype=float)

    def update(self, y_true_chunk, y_score_chunk) -> "StreamingAUC":
        """Add a chunk of (label, score) pairs.

        Parameters
        ----------
        y_true_chunk : array-like, shape (n,)
            Binary labels (0/1).
        y_score_chunk : array-like, shape (n,)
            Positive-class scores.

        Returns
        -------
        self

        Time complexity: O(n)
        """
        yt, ys = _validate_inputs(y_true_chunk, y_score_chunk)
        self._labels = np.concatenate([self._labels, yt.astype(float)])
        self._scores = np.concatenate([self._scores, ys.astype(float)])
        if self._labels.size > self.reservoir_size:
            self._labels = self._labels[-self.reservoir_size :]
            self._scores = self._scores[-self.reservoir_size :]
        return self

    def reset(self) -> "StreamingAUC":
        """Empty the reservoir."""
        self._scores = np.array([], dtype=float)
        self._labels = np.array([], dtype=float)
        return self

    def result(self) -> float:
        """Compute AUC over the retained reservoir.

        Returns
        -------
        float
            AUC, or NaN if only one class is present in the reservoir.

        Time complexity: O(m log m), m = reservoir size.
        """
        if np.unique(self._labels).size != 2:
            return float("nan")
        fpr, tpr, _ = roc_curve(self._labels.astype(int), self._scores)
        return auc(fpr, tpr)
