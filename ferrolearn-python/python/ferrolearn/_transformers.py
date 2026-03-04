"""sklearn-compatible wrappers for ferrolearn transformer models."""

import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data

from ferrolearn._ferrolearn_rs import _RsPCA, _RsStandardScaler


def _ensure_f64(arr):
    """Ensure array is C-contiguous float64 (required by Rust bindings)."""
    return np.ascontiguousarray(arr, dtype=np.float64)


def _fit_rust(rs, X, y=None):
    """Call rs.fit() and translate Rust errors to sklearn-conforming messages."""
    try:
        if y is not None:
            rs.fit(X, y)
        else:
            rs.fit(X)
    except ValueError as e:
        msg = str(e)
        m = re.search(r"got (\d+)", msg)
        if m and "Insufficient" in msg:
            n = m.group(1)
            raise ValueError(
                f"n_samples={n} is not enough; this estimator needs at least "
                f"as many samples as features. {msg}"
            ) from e
        raise


class StandardScaler(TransformerMixin, BaseEstimator):
    """Standardize features by removing the mean and scaling to unit variance.

    Backed by Rust.
    """

    def __init__(self, *, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std

    def fit(self, X, y=None):
        X = validate_data(self, X, dtype="float64", accept_sparse=False)
        self._rs = _RsStandardScaler()
        _fit_rust(self._rs, _ensure_f64(X))
        self.mean_ = np.array(self._rs.mean_)
        self.scale_ = np.array(self._rs.scale_)
        self.n_samples_seen_ = X.shape[0]
        self.var_ = self.scale_ ** 2
        self._fit_X = X.copy()
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype="float64")
        X = _ensure_f64(X)
        if hasattr(self, "_rs"):
            return np.array(self._rs.transform(X))
        # Fallback using stored attributes
        result = X.copy()
        if self.with_mean:
            result = result - self.mean_
        if self.with_std:
            result = result / np.where(self.scale_ == 0, 1.0, self.scale_)
        return result

    def inverse_transform(self, X):
        check_is_fitted(self)
        X = np.asarray(X, dtype="float64")
        X = _ensure_f64(X)
        if hasattr(self, "_rs"):
            return np.array(self._rs.inverse_transform(X))
        result = X.copy()
        if self.with_std:
            result = result * self.scale_
        if self.with_mean:
            result = result + self.mean_
        return result

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_rs", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Reconstruct _rs if we have training data
        if hasattr(self, "_fit_X"):
            self._rs = _RsStandardScaler()
            self._rs.fit(_ensure_f64(self._fit_X))


class PCA(TransformerMixin, BaseEstimator):
    """Principal Component Analysis backed by Rust.

    Parameters
    ----------
    n_components : int, default=2
        Number of components to keep.
    """

    def __init__(self, *, n_components=2):
        self.n_components = n_components

    def fit(self, X, y=None):
        X = validate_data(self, X, dtype="float64")
        self._rs = _RsPCA(n_components=self.n_components)
        _fit_rust(self._rs, _ensure_f64(X))
        self.components_ = np.array(self._rs.components_)
        self.explained_variance_ = np.array(self._rs.explained_variance_)
        self.explained_variance_ratio_ = np.array(self._rs.explained_variance_ratio_)
        self.mean_ = np.array(self._rs.mean_)
        self.singular_values_ = np.array(self._rs.singular_values_)
        self._fit_X = X.copy()
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype="float64")
        X = _ensure_f64(X)
        if hasattr(self, "_rs"):
            return np.array(self._rs.transform(X))
        # Fallback: (X - mean_) @ components_.T
        return (X - self.mean_) @ self.components_.T

    def inverse_transform(self, X):
        check_is_fitted(self)
        X = np.asarray(X, dtype="float64")
        X = _ensure_f64(X)
        if hasattr(self, "_rs"):
            return np.array(self._rs.inverse_transform(X))
        return X @ self.components_ + self.mean_

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_rs", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if hasattr(self, "_fit_X"):
            self._rs = _RsPCA(n_components=self.n_components)
            self._rs.fit(_ensure_f64(self._fit_X))
