"""sklearn-compatible wrappers for ferrolearn clustering models."""

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data

from ferrolearn._ferrolearn_rs import _RsKMeans


def _ensure_f64(arr):
    """Ensure array is C-contiguous float64 (required by Rust bindings)."""
    return np.ascontiguousarray(arr, dtype=np.float64)


class KMeans(TransformerMixin, ClusterMixin, BaseEstimator):
    """K-Means clustering backed by Rust.

    Parameters
    ----------
    n_clusters : int, default=8
        Number of clusters to form.
    max_iter : int, default=300
        Maximum number of iterations.
    tol : float, default=1e-4
        Relative tolerance for convergence.
    n_init : int, default=10
        Number of initializations to run.
    random_state : int or None, default=None
        Random seed for reproducibility.
    """

    def __init__(
        self, *, n_clusters=8, max_iter=300, tol=1e-4, n_init=10, random_state=None
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state

    def fit(self, X, y=None):
        X = validate_data(self, X, dtype="float64")
        self._rs = _RsKMeans(
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            tol=self.tol,
            n_init=self.n_init,
            random_state=self.random_state,
        )
        self._rs.fit(_ensure_f64(X))
        self.cluster_centers_ = np.array(self._rs.cluster_centers_)
        self.labels_ = np.asarray(self._rs.labels_)
        self.inertia_ = float(self._rs.inertia_)
        self.n_iter_ = int(self._rs.n_iter_)
        self._fit_X = X.copy()
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype="float64")
        X = _ensure_f64(X)
        if not hasattr(self, "_rs"):
            self._rebuild_rs()
        return np.asarray(self._rs.predict(X))

    def transform(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype="float64")
        X = _ensure_f64(X)
        if not hasattr(self, "_rs"):
            self._rebuild_rs()
        return np.array(self._rs.transform(X))

    def _rebuild_rs(self):
        self._rs = _RsKMeans(
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            tol=self.tol,
            n_init=self.n_init,
            random_state=self.random_state,
        )
        self._rs.fit(_ensure_f64(self._fit_X))

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_rs", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if hasattr(self, "_fit_X"):
            self._rebuild_rs()
