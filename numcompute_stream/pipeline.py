from __future__ import annotations
from typing import Any, Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class Transformer(Protocol):
    """Structural protocol for transformer steps in a :class:`Pipeline`."""
    def fit(self, X: np.ndarray, /) -> Any: ...
    def transform(self, X: np.ndarray, /) -> np.ndarray: ...
    def fit_transform(self, X: np.ndarray, /) -> np.ndarray: ...


def _validate_transformer(name: str, obj: object) -> None:
    for attr in ("fit", "transform", "fit_transform"):
        if not callable(getattr(obj, attr, None)):
            raise ValueError(f"Step '{name}' does not implement required method '{attr}'.")


def _is_estimator(obj: object) -> bool:
    """An estimator has predict (and may have fit / partial_fit)."""
    return callable(getattr(obj, "predict", None))


def _has_partial_fit(obj: object) -> bool:
    return callable(getattr(obj, "partial_fit", None))


class Pipeline:
    """Sequentially chain transformers, with an optional final estimator.

    Two usage modes:

    1. **Transformer-only** (Assignment 1 behaviour): every step implements
       ``fit`` / ``transform`` / ``fit_transform``. ``fit`` runs
       ``fit_transform`` on all but the last step, then ``fit`` on the last.

    2. **Estimator-terminated**: the final step implements ``predict`` (e.g.
       a tree or forest). Transformers run first, then the estimator is
       trained on the transformed features. ``predict`` / ``predict_proba``
       route the input through the transformers and into the estimator.

    Streaming is supported via ``partial_fit``: each transformer is updated
    with ``partial_fit`` (or ``fit`` as a fallback), the chunk is transformed,
    and the final estimator's ``partial_fit`` is called on the transformed
    chunk.

    Parameters
    ----------
    steps : list of (str, object)
        Named steps. Names must be unique. All but the last must be
        transformers; the last may be a transformer or an estimator.
    """

    def __init__(self, steps: list[tuple[str, Any]]) -> None:
        if not steps:
            raise ValueError("Pipeline requires at least one step.")
        names = [s[0] for s in steps]
        if len(set(names)) != len(names):
            raise ValueError("Pipeline step names must be unique.")
        # Validate: all but last must be transformers; last can be estimator.
        for name, est in steps[:-1]:
            _validate_transformer(name, est)
        last_name, last_est = steps[-1]
        if not _is_estimator(last_est):
            _validate_transformer(last_name, last_est)
        self.steps = steps
        self._fitted = False

    # ------------------------------------------------------------------

    @property
    def _final(self):
        return self.steps[-1][1]

    @property
    def _final_is_estimator(self) -> bool:
        return _is_estimator(self._final)

    # ------------------------------------------------------------------
    # Batch fit
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "Pipeline":
        """Fit all steps on a full batch.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray or None
            Required if the final step is an estimator.

        Returns
        -------
        self : Pipeline

        Raises
        ------
        ValueError
            If the final step is an estimator but y is None.

        Time complexity: sum of per-step fit costs.
        """
        Xt = np.asarray(X)
        if self._final_is_estimator:
            if y is None:
                raise ValueError("y is required when the final step is an estimator.")
            for _, est in self.steps[:-1]:
                Xt = est.fit_transform(Xt)
            self._final.fit(Xt, y)
        else:
            for _, est in self.steps[:-1]:
                Xt = est.fit_transform(Xt)
            self._final.fit(Xt)
        self._fitted = True
        return self

    def partial_fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        classes: np.ndarray | None = None,
    ) -> "Pipeline":
        """Incrementally update all steps with a chunk.

        Parameters
        ----------
        X : np.ndarray, shape (n_chunk, n_features)
        y : np.ndarray or None
            Required if the final step is an estimator.
        classes : np.ndarray or None
            Full class set, forwarded to the estimator's partial_fit.

        Returns
        -------
        self : Pipeline

        Raises
        ------
        ValueError
            If the final step is an estimator but y is None.

        Time complexity: sum of per-step partial_fit costs.
        """
        Xt = np.asarray(X)
        transformers = self.steps[:-1] if self._final_is_estimator else self.steps
        for _, est in transformers:
            if _has_partial_fit(est):
                est.partial_fit(Xt)
            else:
                est.fit(Xt)  # fallback for stateless / batch-only transformers
            Xt = est.transform(Xt)

        if self._final_is_estimator:
            if y is None:
                raise ValueError("y is required when the final step is an estimator.")
            if _has_partial_fit(self._final):
                self._final.partial_fit(Xt, y, classes=classes)
            else:
                self._final.fit(Xt, y)
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Transform / predict
    # ------------------------------------------------------------------

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply all transformer steps (errors if final step is an estimator).

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray

        Raises
        ------
        RuntimeError
            If not fitted, or if the final step is an estimator.
        """
        if not self._fitted:
            raise RuntimeError("This Pipeline instance is not fitted yet; call fit first.")
        if self._final_is_estimator:
            raise RuntimeError(
                "Final step is an estimator; use predict/predict_proba, not transform."
            )
        Xt = np.asarray(X)
        for _, est in self.steps:
            Xt = est.transform(Xt)
        return Xt

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """Fit then transform (transformer-only pipelines)."""
        self.fit(X, y)
        return self.transform(X)

    def _apply_transformers(self, X: np.ndarray) -> np.ndarray:
        Xt = np.asarray(X)
        transformers = self.steps[:-1] if self._final_is_estimator else self.steps
        for _, est in transformers:
            Xt = est.transform(Xt)
        return Xt

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Route X through transformers and the final estimator's predict.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,)

        Raises
        ------
        RuntimeError
            If not fitted or final step is not an estimator.
        """
        if not self._fitted:
            raise RuntimeError("This Pipeline instance is not fitted yet; call fit first.")
        if not self._final_is_estimator:
            raise RuntimeError("Final step is not an estimator; cannot predict.")
        Xt = self._apply_transformers(X)
        return self._final.predict(Xt)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Route X through transformers and the estimator's predict_proba.

        Raises
        ------
        RuntimeError
            If not fitted, final step is not an estimator, or the estimator
            has no predict_proba.
        """
        if not self._fitted:
            raise RuntimeError("This Pipeline instance is not fitted yet; call fit first.")
        if not self._final_is_estimator:
            raise RuntimeError("Final step is not an estimator; cannot predict_proba.")
        if not callable(getattr(self._final, "predict_proba", None)):
            raise RuntimeError("Final estimator does not implement predict_proba.")
        Xt = self._apply_transformers(X)
        return self._final.predict_proba(Xt)


class FeatureUnion:
    """Fit several transformers on the same X and concatenate outputs.

    Supports ``partial_fit`` by forwarding to each transformer's
    ``partial_fit`` (or ``fit`` as a fallback).
    """

    def __init__(self, transformers: list[tuple[str, Transformer]]) -> None:
        if not transformers:
            raise ValueError("FeatureUnion requires at least one transformer.")
        names = [t[0] for t in transformers]
        if len(set(names)) != len(names):
            raise ValueError("FeatureUnion names must be unique.")
        for name, est in transformers:
            _validate_transformer(name, est)
        self.transformers = transformers
        self._fitted = False

    def fit(self, X: np.ndarray) -> "FeatureUnion":
        """Fit every transformer on X."""
        X = np.asarray(X)
        for _, est in self.transformers:
            est.fit(X)
        self._fitted = True
        return self

    def partial_fit(self, X: np.ndarray) -> "FeatureUnion":
        """Incrementally update every transformer on a chunk."""
        X = np.asarray(X)
        for _, est in self.transformers:
            if _has_partial_fit(est):
                est.partial_fit(X)
            else:
                est.fit(X)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Concatenate transformer outputs column-wise."""
        if not self._fitted:
            raise RuntimeError("This FeatureUnion instance is not fitted yet; call fit first.")
        X = np.asarray(X)
        parts = [est.transform(X) for _, est in self.transformers]
        return np.hstack(parts)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        parts = [est.fit_transform(X) for _, est in self.transformers]
        self._fitted = True
        return np.hstack(parts)
