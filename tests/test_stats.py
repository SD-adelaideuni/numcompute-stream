"""Unit tests for numcompute_stream.stats (batch + streaming)."""
import numpy as np
import pytest

from numcompute_stream.stats import mean, StreamingStats, StreamingHistogram
from numcompute_stream.utils import iter_chunks


def test_mean_ignores_nan():
    arr = np.array([1.0, 2.0, np.nan, 3.0])
    assert mean(arr) == 2.0


def test_streaming_stats_mean_var_match_batch():
    rng = np.random.default_rng(1)
    X = rng.normal(3, 7, (1000, 4))
    ss = StreamingStats()
    for Xc, _ in iter_chunks(X, chunk_size=83):
        ss.update_stats(Xc)
    assert np.allclose(ss.mean(), X.mean(axis=0))
    assert np.allclose(ss.variance(), X.var(axis=0))


def test_streaming_stats_min_max():
    X = np.array([[1.0], [5.0], [-2.0], [9.0]])
    ss = StreamingStats()
    for Xc, _ in iter_chunks(X, chunk_size=1):
        ss.update_stats(Xc)
    assert ss.minimum()[0] == -2.0
    assert ss.maximum()[0] == 9.0


def test_streaming_stats_quantile_estimate():
    rng = np.random.default_rng(2)
    X = rng.uniform(0, 1, (2000, 1))
    ss = StreamingStats(track_quantiles=True)
    for Xc, _ in iter_chunks(X, chunk_size=200):
        ss.update_stats(Xc)
    med = ss.quantile(0.5)
    assert abs(float(med[0]) - 0.5) < 0.05


def test_streaming_stats_before_update_raises():
    with pytest.raises(RuntimeError):
        StreamingStats().mean()


def test_streaming_histogram_accumulates():
    h = StreamingHistogram(bins=4, value_range=(0, 4))
    h.update(np.array([0.5, 1.5]))
    h.update(np.array([2.5, 3.5]))
    counts, edges = h.result()
    assert counts.sum() == 4
    assert len(edges) == 5
