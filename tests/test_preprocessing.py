"""Unit tests for numcompute_stream.preprocessing (batch + streaming)."""
import numpy as np
import pytest

from numcompute_stream.preprocessing import (
    StandardScaler, MinMaxScaler, OneHotEncoder, SimpleImputer,
)
from numcompute_stream.utils import iter_chunks


def test_standard_scaler_batch_zero_mean_unit_std():
    rng = np.random.default_rng(0)
    X = rng.normal(5, 3, (200, 4))
    Xs = StandardScaler().fit_transform(X)
    assert np.allclose(Xs.mean(axis=0), 0, atol=1e-7)
    assert np.allclose(Xs.std(axis=0), 1, atol=1e-7)


def test_standard_scaler_partial_fit_matches_batch():
    rng = np.random.default_rng(1)
    X = rng.normal(2, 5, (1000, 3))
    s = StandardScaler()
    for Xc, _ in iter_chunks(X, chunk_size=97):
        s.partial_fit(Xc)
    assert np.allclose(s.mean_, X.mean(axis=0))
    assert np.allclose(s.scale_, X.std(axis=0))


def test_standard_scaler_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        StandardScaler().transform(np.ones((2, 2)))


def test_minmax_scaler_range():
    rng = np.random.default_rng(2)
    X = rng.normal(0, 10, (100, 3))
    Xs = MinMaxScaler(feature_range=(0, 1)).fit_transform(X)
    assert Xs.min() >= -1e-9 and Xs.max() <= 1 + 1e-9


def test_onehot_partial_fit_expands_categories():
    enc = OneHotEncoder()
    enc.partial_fit(np.array([[0], [1]]))
    enc.partial_fit(np.array([[2], [3]]))
    out = enc.transform(np.array([[3]]))
    assert out.shape == (1, 4)
    assert out[0, -1] == 1.0


def test_imputer_mean_partial_fit_matches_batch():
    rng = np.random.default_rng(4)
    X = rng.normal(0, 1, (500, 3))
    X[::10, 0] = np.nan
    batch = SimpleImputer(strategy="mean").fit(X).statistics_
    s = SimpleImputer(strategy="mean")
    for Xc, _ in iter_chunks(X, chunk_size=50):
        s.partial_fit(Xc)
    assert np.allclose(s.statistics_, batch, equal_nan=True)
