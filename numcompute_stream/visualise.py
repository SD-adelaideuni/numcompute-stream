from __future__ import annotations

import numpy as np
import matplotlib

# Use a non-interactive backend by default so plots work in scripts/CI.
matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt  # noqa: E402


def _finish(fig, save_path: str | None, show: bool):
    """Save and/or show a figure, returning the Axes for further tweaking."""
    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_metric_over_time(
    metric_values,
    title: str = "Metric over time",
    ylabel: str = "Value",
    xlabel: str = "Chunk",
    save_path: str | None = None,
    show: bool = False,
):
    """Plot a single metric across streaming chunks.

    Parameters
    ----------
    metric_values : array-like, shape (n_chunks,)
        Metric value per chunk (e.g. accuracy over time).
    title : str
        Plot title.
    ylabel : str
        Y-axis label.
    xlabel : str
        X-axis label. Default 'Chunk'.
    save_path : str or None
        If given, save the figure to this path.
    show : bool
        If True, display interactively.

    Returns
    -------
    matplotlib.figure.Figure

    Time complexity: O(n_chunks)
    """
    values = np.asarray(metric_values, dtype=float)
    x = np.arange(values.shape[0])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, values, marker="o", linewidth=1.8, markersize=4)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    return _finish(fig, save_path, show)


def compare_models(
    metric1,
    metric2,
    labels=("Model 1", "Model 2"),
    title: str = "Model comparison",
    ylabel: str = "Value",
    xlabel: str = "Chunk",
    save_path: str | None = None,
    show: bool = False,
):
    """Compare two models' streaming metric curves on the same axes.

    Parameters
    ----------
    metric1 : array-like, shape (n_chunks,)
        First model's per-chunk metric.
    metric2 : array-like, shape (n_chunks,)
        Second model's per-chunk metric.
    labels : tuple of (str, str)
        Legend labels.
    title, ylabel, xlabel : str
        Plot text.
    save_path : str or None
        Save path.
    show : bool
        Display interactively.

    Returns
    -------
    matplotlib.figure.Figure

    Time complexity: O(n_chunks)
    """
    m1 = np.asarray(metric1, dtype=float)
    m2 = np.asarray(metric2, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(np.arange(m1.shape[0]), m1, marker="o", markersize=4,
            linewidth=1.8, label=labels[0])
    ax.plot(np.arange(m2.shape[0]), m2, marker="s", markersize=4,
            linewidth=1.8, label=labels[1])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _finish(fig, save_path, show)


def plot_predictions_vs_ground_truth(
    y_true,
    y_pred,
    title: str = "Predictions vs ground truth",
    save_path: str | None = None,
    show: bool = False,
):
    """Visualise predictions against actuals for the latest chunk.

    For classification this renders a confusion-matrix heatmap. The matrix is
    computed inline (no external deps) so the function is self-contained.

    Parameters
    ----------
    y_true : array-like, shape (n,)
        Ground-truth labels.
    y_pred : array-like, shape (n,)
        Predicted labels.
    title : str
        Plot title.
    save_path : str or None
        Save path.
    show : bool
        Display interactively.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If inputs are not 1-D or have different lengths.

    Time complexity: O(n + C^2)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be 1-D.")
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have equal length.")

    classes = np.unique(np.concatenate([y_true, y_pred]))
    idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    ti = np.array([idx[v] for v in y_true])
    pi = np.array([idx[v] for v in y_pred])
    np.add.at(cm, (ti, pi), 1)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    # Annotate counts.
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    return _finish(fig, save_path, show)
