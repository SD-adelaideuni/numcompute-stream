"""
numcompute_stream — streaming, decision-tree-based ML framework.

Extension to the Assignment 1 NumCompute toolkit with incremental (streaming)
learning, decision trees, a Random Forest ensemble, and a matplotlib-based
visualisation module. """

from . import (
    io,
    utils,
    preprocessing,
    stats,
    metrics,
    tree,
    ensemble,
    pipeline,
    stream,
    visualise,
)

__version__ = "0.2.0"
__all__ = [
    "io",
    "utils",
    "preprocessing",
    "stats",
    "metrics",
    "tree",
    "ensemble",
    "pipeline",
    "stream",
    "visualise",
]
