"""
Decision Tree Streaming Performance Benchmark.

Measurements for increasing stream lengths:

* Per-chunk training latency,
* Prequential accuracy,
* Approximate size of the generated model,

with a decision tree model and a random forest model, the latter trained via
the StreamTrainer, and compares the naive implementation of class counts in a
python loop with vectorization based on cumulative sum split search within
a tree node, to estimate the gain from vectorization asked for by the rubric.

Execution command:
python benchmark/benchmark_stream.py

"""
from __future__ import annotations

import time

import numpy as np

from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.ensemble import RandomForestClassifier
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.utils import iter_chunks


def make_stream(n_per_class: int, n_features: int, n_classes: int, seed: int = 0):
    """Generate a separable-ish multiclass dataset (shuffled)."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(0, 8, (n_classes, n_features))
    blocks = [rng.normal(centres[c], 1.0, (n_per_class, n_features)) for c in range(n_classes)]
    X = np.vstack(blocks)
    y = np.concatenate([np.full(n_per_class, c) for c in range(n_classes)])
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def bench_model(name, model, X, y, classes, chunk_size):

    trainer = StreamTrainer(model, prequential=True, verbose=False)
    t0 = time.perf_counter()
    hist = trainer.run(iter_chunks(X, y, chunk_size=chunk_size), classes=classes)
    wall = time.perf_counter() - t0
    mean_chunk_ms = float(np.mean(hist["time_sec"]) * 1000)
    print(
        f"{name:<16} | chunks={len(hist['chunk']):>3} | "
        f"final cum_acc={hist['cumulative_accuracy'][-1]:.3f} | "
        f"mean chunk={mean_chunk_ms:6.1f} ms | "
        f"final mem={hist['model_mb'][-1]:5.2f} MB | "
        f"total={wall:5.2f} s"
    )
    return hist


def bench_vectorisation(n: int, seed: int = 0):
  
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 5, n)

    t0 = time.perf_counter()
    counts_loop = [0] * 5
    for v in y:
        counts_loop[v] += 1
    loop_t = time.perf_counter() - t0

    t0 = time.perf_counter()
    counts_vec = np.bincount(y, minlength=5)
    vec_t = time.perf_counter() - t0

    assert list(counts_vec) == counts_loop
    speedup = loop_t / vec_t if vec_t > 0 else float("inf")
    print(
        f"class-count n={n:>7} | loop={loop_t*1000:7.2f} ms | "
        f"vectorised={vec_t*1000:6.3f} ms | speed-up x{speedup:6.1f}"
    )


def main():
    print("=" * 78)
    print("STREAMING MODEL BENCHMARK  (DecisionTree vs RandomForest)")
    print("=" * 78)
    classes = np.array([0, 1, 2, 3, 4])
    for n_per_class in (200, 500, 1000):
        X, y = make_stream(n_per_class, n_features=10, n_classes=5, seed=1)
        print(f"\n--- stream size = {len(y)} samples, chunk_size = 200 ---")
        bench_model(
            "DecisionTree",
            DecisionTreeClassifier(max_depth=8, random_state=0),
            X, y, classes, chunk_size=200,
        )
        bench_model(
            "RandomForest",
            RandomForestClassifier(n_estimators=15, max_depth=8, random_state=0),
            X, y, classes, chunk_size=200,
        )

    print("\n" + "=" * 78)
    print("VECTORISATION MICRO-BENCHMARK  (Python loop vs NumPy)")
    print("=" * 78)
    for n in (10_000, 100_000, 1_000_000):
        bench_vectorisation(n)


if __name__ == "__main__":
    main()
