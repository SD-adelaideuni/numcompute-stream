"""Unit tests for numcompute_stream.metrics (batch + streaming)."""
import numpy as np
import pytest

from numcompute_stream.metrics import (
    accuracy, precision, recall, f1, confusion_matrix, roc_curve, auc,
    StreamingClassificationMetrics, RollingAccuracy, StreamingAUC,
)


def test_accuracy_perfect():
    y = np.array([0, 1, 2, 1])
    assert accuracy(y, y) == 1.0


def test_confusion_matrix_shape_and_counts():
    yt = np.array([0, 1, 1, 0])
    yp = np.array([0, 1, 0, 0])
    cm = confusion_matrix(yt, yp)
    assert cm.shape == (2, 2)
    assert cm[0, 0] == 2 and cm[1, 0] == 1 and cm[1, 1] == 1


def test_precision_recall_f1_range():
    rng = np.random.default_rng(0)
    yt = rng.integers(0, 3, 100)
    yp = rng.integers(0, 3, 100)
    for fn in (precision, recall, f1):
        v = fn(yt, yp)
        assert 0.0 <= v <= 1.0


def test_roc_auc_perfect_separation():
    yt = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    fpr, tpr, _ = roc_curve(yt, scores)
    assert np.isclose(auc(fpr, tpr), 1.0)


def test_streaming_metrics_match_batch():
    rng = np.random.default_rng(1)
    yt = rng.integers(0, 3, 600)
    yp = rng.integers(0, 3, 600)
    m = StreamingClassificationMetrics(labels=np.array([0, 1, 2]), average="macro")
    for i in range(0, 600, 60):
        m.update(yt[i:i + 60], yp[i:i + 60])
    res = m.result()
    assert np.isclose(res["accuracy"], accuracy(yt, yp))
    assert np.isclose(res["precision"], precision(yt, yp, average="macro"))
    assert np.isclose(res["recall"], recall(yt, yp, average="macro"))


def test_rolling_accuracy_window():
    r = RollingAccuracy(window=4)
    r.update(np.array([0, 0]), np.array([0, 0]))   # 2 correct
    r.update(np.array([1, 1]), np.array([0, 0]))   # 2 wrong
    assert np.isclose(r.result(), 0.5)


def test_streaming_auc_perfect():
    a = StreamingAUC()
    a.update(np.array([0, 0]), np.array([0.1, 0.2]))
    a.update(np.array([1, 1]), np.array([0.8, 0.9]))
    assert np.isclose(a.result(), 1.0)
