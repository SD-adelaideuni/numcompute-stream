"""Unit tests for numcompute_stream.pipeline and stream."""
import numpy as np
import pytest

from numcompute_stream.preprocessing import StandardScaler, MinMaxScaler
from numcompute_stream.ensemble import RandomForestClassifier
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.pipeline import Pipeline, FeatureUnion
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.utils import iter_chunks


def _blobs(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(0, 1, (n, 4)), rng.normal(5, 1, (n, 4))])
    y = np.array([0] * n + [1] * n)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def test_pipeline_transformer_only():
    X, _ = _blobs()
    pipe = Pipeline([("scale", StandardScaler()), ("mm", MinMaxScaler())])
    out = pipe.fit_transform(X)
    assert out.shape == X.shape


def test_pipeline_with_estimator_batch():
    X, y = _blobs()
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=8, max_depth=5, random_state=0)),
    ])
    pipe.fit(X, y)
    assert np.mean(pipe.predict(X) == y) > 0.95


def test_pipeline_partial_fit_streaming():
    X, y = _blobs(n=400)
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=8, max_depth=6, random_state=0)),
    ])
    for Xc, yc in iter_chunks(X, y, chunk_size=100):
        pipe.partial_fit(Xc, yc, classes=np.array([0, 1]))
    assert np.mean(pipe.predict(X) == y) > 0.9


def test_pipeline_predict_before_fit_raises():
    pipe = Pipeline([("scale", StandardScaler()),
                     ("tree", DecisionTreeClassifier(random_state=0))])
    with pytest.raises(RuntimeError):
        pipe.predict(np.ones((2, 4)))


def test_feature_union_concatenates():
    X, _ = _blobs()
    fu = FeatureUnion([("s", StandardScaler()), ("m", MinMaxScaler())])
    out = fu.fit_transform(X)
    assert out.shape == (X.shape[0], X.shape[1] * 2)


def test_stream_trainer_logs_history():
    X, y = _blobs(n=300)
    rf = RandomForestClassifier(n_estimators=6, max_depth=5, random_state=0)
    tr = StreamTrainer(rf, verbose=False)
    hist = tr.run(iter_chunks(X, y, chunk_size=100), classes=np.array([0, 1]))
    assert len(hist["chunk"]) == len(hist["accuracy"])
    assert all(0.0 <= a <= 1.0 for a in hist["cumulative_accuracy"])
    assert all(t >= 0 for t in hist["time_sec"])


def test_stream_trainer_requires_partial_fit():
    class NoPartial:
        def predict(self, X):
            return np.zeros(len(X))
    with pytest.raises(ValueError):
        StreamTrainer(NoPartial())


def test_stream_trainer_cumulative_accuracy_high_on_easy_data():
    X, y = _blobs(n=400)
    rf = RandomForestClassifier(n_estimators=8, max_depth=6, random_state=0)
    tr = StreamTrainer(rf, prequential=True)
    hist = tr.run(iter_chunks(X, y, chunk_size=80), classes=np.array([0, 1]))
    assert hist["cumulative_accuracy"][-1] > 0.85
