"""Unit tests for numcompute_stream.ensemble."""
import numpy as np
import pytest

from numcompute_stream.ensemble import RandomForestClassifier
from numcompute_stream.utils import iter_chunks


def _three_blobs(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = np.vstack([
        rng.normal(0, 1, (n, 5)),
        rng.normal(6, 1, (n, 5)),
        rng.normal(-6, 1, (n, 5)),
    ])
    y = np.array([0] * n + [1] * n + [2] * n)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def test_forest_fits_multiclass():
    X, y = _three_blobs()
    rf = RandomForestClassifier(n_estimators=10, max_depth=6, random_state=0).fit(X, y)
    assert np.mean(rf.predict(X) == y) > 0.95


def test_forest_predict_proba_sums_to_one():
    X, y = _three_blobs()
    rf = RandomForestClassifier(n_estimators=8, max_depth=5, random_state=0).fit(X, y)
    proba = rf.predict_proba(X)
    assert proba.shape == (X.shape[0], 3)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_forest_partial_fit_streaming():
    X, y = _three_blobs(n=200)
    rf = RandomForestClassifier(n_estimators=8, max_depth=6, random_state=0)
    for Xc, yc in iter_chunks(X, y, chunk_size=120, shuffle=True, random_state=1):
        rf.partial_fit(Xc, yc, classes=np.array([0, 1, 2]))
    assert np.mean(rf.predict(X) == y) > 0.9


def test_forest_reproducible_with_seed():
    X, y = _three_blobs()
    p1 = RandomForestClassifier(n_estimators=8, random_state=42).fit(X, y).predict(X)
    p2 = RandomForestClassifier(n_estimators=8, random_state=42).fit(X, y).predict(X)
    assert np.array_equal(p1, p2)
