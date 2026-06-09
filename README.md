# numcompute_stream

Streaming machine learning framework using decision trees based solely on
NumPy and matplotlib. Extension of the Assignment 1 library with
online learning, i.e. the ability to update all components chunk by chunk
using `partial_fit`/`update` and train on infinite streams of data that do not
fit into memory.

The framework supports streaming data pre-processing,
calculation of streaming metrics and statistics, implementation of a decision
tree classifier from scratch, Random Forest ensemble,
pipeline connecting them together, orchestrating their streaming training, and a
simple visualization library.

---

## Installation

```bash
# from the numcompute_stream/ directory
pip install -e .
```

This installs the package (with `numpy` and `matplotlib`) and makes
`import numcompute_stream` work from anywhere. To also get the test / notebook
tooling:

```bash
pip install -e ".[dev]"
```

Without installing, you can run scripts from the repo root with
`PYTHONPATH=.` (e.g. `PYTHONPATH=. python benchmark/benchmark_stream.py`).

---

## Quick start

```python
import numpy as np
from numcompute_stream.preprocessing import StandardScaler
from numcompute_stream.ensemble import RandomForestClassifier
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.utils import iter_chunks

# A scaler + forest, trained incrementally.
pipe = Pipeline([
    ("scale", StandardScaler()),
    ("rf", RandomForestClassifier(n_estimators=15, max_depth=8, random_state=0)),
])

trainer = StreamTrainer(pipe, prequential=True, verbose=True)
history = trainer.run(iter_chunks(X, y, chunk_size=250), classes=np.unique(y))

print("final cumulative accuracy:", history["cumulative_accuracy"][-1])
```

See [`demo/stream_demo.ipynb`](demo/stream_demo.ipynb) for a complete,
executed walkthrough: CSV streaming, incremental training, metric logging and
plots.

---

## Module overview

| Module | Contents | Streaming entry points |
|---|---|---|
| `io` | `load_csv`, `load_csv_chunked` (generator), `save_csv` | `load_csv_chunked` streams rows without reading the whole file |
| `utils` | input validation, `iter_chunks`, `gini_impurity`, `entropy`, `softmax` | `iter_chunks` turns arrays into a chunk stream |
| `preprocessing` | `StandardScaler`, `MinMaxScaler`, `OneHotEncoder`, `SimpleImputer` | every class has `partial_fit` |
| `stats` | A1 batch functions + `StreamingStats`, `StreamingHistogram` | `update_stats`, `update` |
| `metrics` | A1 batch metrics + `StreamingConfusionMatrix`, `StreamingClassificationMetrics`, `RollingAccuracy`, `StreamingAUC` | `update` / `reset` / `result` |
| `tree` | `DecisionTreeClassifier` (Gini/entropy, depth limit, feature subsampling) | `partial_fit` |
| `ensemble` | `RandomForestClassifier` (bootstrap + soft voting) | `partial_fit` |
| `pipeline` | `Pipeline` (transformers and/or final estimator), `FeatureUnion` | `partial_fit` |
| `stream` | `StreamTrainer` (prequential eval, per-chunk logging) | `fit_chunk`, `run` |
| `visualise` | `plot_metric_over_time`, `compare_models`, `plot_predictions_vs_ground_truth` | — |

### Design conventions (shared with Assignment 1)

- `from __future__ import annotations` and `|` type hints throughout.
- Fitted state stored in trailing-underscore attributes (`mean_`, `scale_`,
  `categories_`, `statistics_`, `classes_`, ...).
- `RuntimeError` when used before fitting; `ValueError` for bad shapes/inputs.
- NaN-safe reductions (`np.nanmean`, `np.nanstd`, ...) where appropriate.
- Vectorised hot paths (Welford merges, one-hot blocks, cumulative-sum split
  search, soft-vote averaging).

---

## How streaming trees work (design note)

A fully incremental tree (like a Hoeffding tree) is not covered in this assignment.
assignment. Instead, `partial_fit` uses a **mini-batch retrain** strategy: each
A buffer of a fixed maximum size (max_buffer) is appended by each incoming chunk of data. The entire buffer is then used to retrain the tree.
we “retrain” from chunk to chunk, rebuild the tree from the combined chunk buffer.

- **Pros:** correct, stable, easy to reason about; predictions always reflect all retained data; splitting logic stays simple and vectorised.
all retained data; splitting logic stays simple and vectorised.
- **Cons:** the cost of retraining the tree for a chunk increases with the size of the buffer up to the size of the buffer when it is full, as specified by the max_buffer size.
`max_buffer`. The benchmark below shows this clearly.

The Random Forest applies the same idea per tree, by independently bootstrapping every chunk for every tree in the ensemble.
each tree in the Random Forest gets an independent bootstrap of every chunk in that dataset, so the predictions from the trees in the ensemble remain decorrelated..

---

## Streaming metrics

`StreamTrainer` evaluates **prequentially** (test-then-train): each chunk is
predicted before it is used to train the model, thus providing an accurate estimate of performance on unseen data.
of performance on unseen data. The logged `history_` always contains `chunk`,
`n_samples`, `time_sec`, `model_mb`, `cumulative_accuracy`, plus one key per
configured metric (e.g. accuracy, precision, recall, F1 — all macro).

---

## Benchmarks

Run with:

```bash
PYTHONPATH=. python benchmark/benchmark_stream.py
```

**Streaming model comparison** (5-class, 10 features, chunk size 200, soft
voting forest of 15 trees; representative run):

| Stream size | Model | Final cum. accuracy | Mean chunk time | Final model size |
|---|---|---|---|---|
| 1,000 | DecisionTree | 1.000 | ~0.31 s | ~0.08 MB |
| 1,000 | RandomForest | 1.000 | ~0.88 s | ~1.26 MB |
| 2,500 | DecisionTree | 1.000 | ~0.68 s | ~0.21 MB |
| 2,500 | RandomForest | 1.000 | ~1.96 s | ~3.15 MB |
| 5,000 | DecisionTree | 1.000 | ~1.19 s | ~0.42 MB |
| 5,000 | RandomForest | 1.000 | ~3.72 s | ~6.30 MB |

The forest is about an order of magnitude compute and memory as the trees.
single tree (it *is* 15 trees), and per-chunk time rises as the retrain buffer
grows — exactly the documented mini-batch-retrain trade-off.

**Vectorisation micro-benchmark** (class counting, Python loop vs NumPy):

| n | Python loop | Vectorised | Speed-up |
|---|---|---|---|
| 10,000 | ~1.3 ms | ~0.12 ms | ~11x |
| 100,000 | ~12.6 ms | ~0.20 ms | ~63x |
| 1,000,000 | ~124.7 ms | ~2.14 ms | ~58x |

(Absolute numbers vary by machine; the relative gap is the point.)

---

## Testing

```bash
pip install -e ".[dev]"
pytest
```

The suite contains 40 tests organized in folders for preprocessing, stats, metrics, tree, ensemble, pipeline, stream, io and visualise.
All of the streaming components (preprocessing, stats, metrics, tree, ensemble,
pipeline, stream, io and visualise) are tested.
for *numerical agreement with their batch equivalents* (e.g. streaming `StandardScaler` mean and std values against the full-batch computation for a `StandardScaler` against a `StreamingStats` variance against a full-batch computation for ).
`StandardScaler` mean/std and `StreamingStats` variance match a full-batch
computation exactly), alongside error-path and edge-case tests.

---

## Project layout

```
numcompute_stream/
├── numcompute_stream/      # the package
│   ├── __init__.py
│   ├── io.py
│   ├── utils.py
│   ├── preprocessing.py
│   ├── stats.py
│   ├── metrics.py
│   ├── tree.py
│   ├── ensemble.py
│   ├── pipeline.py
│   ├── stream.py
│   └── visualise.py
├── tests/                  # 81 unit tests
├── demo/
│   └── stream_demo.ipynb   # executed end-to-end walkthrough
├── benchmark/
│   └── benchmark_stream.py
├── pyproject.toml
└── README.md
```
