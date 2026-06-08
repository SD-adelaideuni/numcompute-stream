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


def test_standard_scaler_constant_column_no_div_zero():
    X = np.ones((10, 2))
    with pytest.warns(RuntimeWarning):
        Xs = StandardScaler().fit_transform(X)
    assert np.all(np.isfinite(Xs))


def test_standard_scaler_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        StandardScaler().transform(np.ones((2, 2)))


def test_standard_scaler_partial_fit_feature_mismatch_raises():
    s = StandardScaler()
    s.partial_fit(np.ones((5, 3)))
    with pytest.raises(ValueError):
        s.partial_fit(np.ones((5, 4)))


def test_minmax_scaler_range():
    rng = np.random.default_rng(2)
    X = rng.normal(0, 10, (100, 3))
    Xs = MinMaxScaler(feature_range=(0, 1)).fit_transform(X)
    assert Xs.min() >= -1e-9 and Xs.max() <= 1 + 1e-9


def test_minmax_partial_fit_tracks_global_min_max():
    X1 = np.array([[0.0], [5.0]])
    X2 = np.array([[-3.0], [10.0]])
    s = MinMaxScaler()
    s.partial_fit(X1)
    s.partial_fit(X2)
    assert s.data_min_[0] == -3.0
    assert s.data_max_[0] == 10.0


def test_minmax_invalid_range_raises():
    with pytest.raises(ValueError):
        MinMaxScaler(feature_range=(1, 0))


def test_onehot_basic():
    X = np.array([[0], [1], [2], [1]])
    out = OneHotEncoder().fit_transform(X)
    expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]], dtype=float)
    assert np.array_equal(out, expected)


def test_onehot_partial_fit_expands_categories():
    enc = OneHotEncoder()
    enc.partial_fit(np.array([[0], [1]]))
    enc.partial_fit(np.array([[2], [3]]))
    # 4 categories now known
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


def test_imputer_constant_fills_nan():
    X = np.array([[1.0, np.nan], [np.nan, 4.0]])
    out = SimpleImputer(strategy="constant", fill_value=-1.0).fit_transform(X)
    assert out[0, 1] == -1.0 and out[1, 0] == -1.0


def test_imputer_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        SimpleImputer().transform(np.ones((2, 2)))


def test_imputer_invalid_strategy_raises():
    with pytest.raises(ValueError):
        SimpleImputer(strategy="bogus")
