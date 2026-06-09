"""Unit tests for numcompute_stream.tree."""
import numpy as np
import pytest

from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.utils import iter_chunks


def _two_blobs(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(0, 1, (n, 4)), rng.normal(5, 1, (n, 4))])
    y = np.array([0] * n + [1] * n)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def test_tree_fits_separable_data():
    X, y = _two_blobs()
    t = DecisionTreeClassifier(max_depth=5, random_state=0).fit(X, y)
    assert np.mean(t.predict(X) == y) > 0.95


def test_tree_predict_proba_sums_to_one():
    X, y = _two_blobs()
    t = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X, y)
    proba = t.predict_proba(X)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_tree_entropy_criterion():
    X, y = _two_blobs()
    t = DecisionTreeClassifier(max_depth=5, criterion="entropy", random_state=0).fit(X, y)
    assert np.mean(t.predict(X) == y) > 0.95


def test_tree_partial_fit_improves():
    X, y = _two_blobs(n=400)
    t = DecisionTreeClassifier(max_depth=6, random_state=0)
    for Xc, yc in iter_chunks(X, y, chunk_size=100):
        t.partial_fit(Xc, yc, classes=np.array([0, 1]))
    assert np.mean(t.predict(X) == y) > 0.9


def test_tree_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        DecisionTreeClassifier().predict(np.ones((2, 2)))
