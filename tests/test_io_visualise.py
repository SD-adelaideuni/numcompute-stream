"""Unit tests for numcompute_stream.io and visualise."""
import os
import tempfile

import numpy as np
import pytest

from numcompute_stream.io import load_csv, load_csv_chunked, save_csv
from numcompute_stream.visualise import (
    plot_metric_over_time, compare_models, plot_predictions_vs_ground_truth,
)


def _tmp_csv(content):
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    f.write(content)
    f.close()
    return f.name


def test_io_round_trip():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    path = tempfile.NamedTemporaryFile(suffix=".csv", delete=False).name
    try:
        save_csv(arr, path)
        loaded = load_csv(path)
        np.testing.assert_allclose(loaded, arr)
    finally:
        os.unlink(path)


def test_io_missing_values_nan():
    path = _tmp_csv("1.0,,3.0\n4.0,5.0,\n")
    try:
        data = load_csv(path)
        assert np.isnan(data[0, 1]) and np.isnan(data[1, 2])
    finally:
        os.unlink(path)


def test_io_bad_path_raises():
    with pytest.raises(FileNotFoundError):
        load_csv("/no/such/file.csv")


def test_io_chunked_matches_full():
    path = _tmp_csv("1,2\n3,4\n5,6\n7,8\n")
    try:
        full = load_csv(path)
        combined = np.vstack(list(load_csv_chunked(path, chunksize=2)))
        np.testing.assert_allclose(combined, full)
    finally:
        os.unlink(path)


def test_visualise_metric_over_time_saves(tmp_path):
    out = tmp_path / "metric.png"
    plot_metric_over_time([0.5, 0.7, 0.9], save_path=str(out))
    assert out.exists() and out.stat().st_size > 0


def test_visualise_compare_models_saves(tmp_path):
    out = tmp_path / "cmp.png"
    compare_models([0.5, 0.6], [0.4, 0.8], labels=("A", "B"), save_path=str(out))
    assert out.exists()


def test_visualise_confusion_saves(tmp_path):
    out = tmp_path / "cm.png"
    plot_predictions_vs_ground_truth(
        np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0]), save_path=str(out)
    )
    assert out.exists()


def test_visualise_confusion_length_mismatch_raises():
    with pytest.raises(ValueError):
        plot_predictions_vs_ground_truth(np.array([0, 1]), np.array([0]))
