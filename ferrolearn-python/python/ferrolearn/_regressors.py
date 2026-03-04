"""sklearn-compatible wrappers for ferrolearn regression models."""

import re

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted, validate_data

from ferrolearn._ferrolearn_rs import (
    _RsElasticNet,
    _RsLasso,
    _RsLinearRegression,
    _RsRidge,
)


def _ensure_f64(arr):
    """Ensure array is C-contiguous float64 (required by Rust bindings)."""
    return np.ascontiguousarray(arr, dtype=np.float64)


def _predict_linear(X, coef, intercept):
    """Fallback linear prediction using stored coefficients."""
    return X @ coef + intercept


def _fit_rust(rs, X, y=None):
    """Call rs.fit() and translate Rust errors to sklearn-conforming messages."""
    try:
        if y is not None:
            rs.fit(X, y)
        else:
            rs.fit(X)
    except ValueError as e:
        msg = str(e)
        # Translate "Insufficient samples: need at least N, got M" to sklearn format
        m = re.search(r"got (\d+)", msg)
        if m and "Insufficient" in msg:
            n = m.group(1)
            raise ValueError(
                f"n_samples={n} is not enough; this estimator needs at least "
                f"as many samples as features. {msg}"
            ) from e
        raise


class LinearRegression(RegressorMixin, BaseEstimator):
    """Ordinary Least Squares regression backed by Rust.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.
    """

    def __init__(self, *, fit_intercept=True):
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        X, y = validate_data(self, X, y, dtype="float64", y_numeric=True)
        X, y = _ensure_f64(X), _ensure_f64(y)
        self._rs = _RsLinearRegression(fit_intercept=self.fit_intercept)
        _fit_rust(self._rs, X, y)
        self.coef_ = np.array(self._rs.coef_)
        self.intercept_ = float(self._rs.intercept_)
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype="float64")
        X = _ensure_f64(X)
        if hasattr(self, "_rs"):
            return np.array(self._rs.predict(X))
        return _predict_linear(X, self.coef_, self.intercept_)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_rs", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


class Ridge(RegressorMixin, BaseEstimator):
    """Ridge regression backed by Rust.

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength.
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.
    """

    def __init__(self, *, alpha=1.0, fit_intercept=True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        X, y = validate_data(self, X, y, dtype="float64", y_numeric=True)
        X, y = _ensure_f64(X), _ensure_f64(y)
        self._rs = _RsRidge(alpha=self.alpha, fit_intercept=self.fit_intercept)
        _fit_rust(self._rs, X, y)
        self.coef_ = np.array(self._rs.coef_)
        self.intercept_ = float(self._rs.intercept_)
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype="float64")
        X = _ensure_f64(X)
        if hasattr(self, "_rs"):
            return np.array(self._rs.predict(X))
        return _predict_linear(X, self.coef_, self.intercept_)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_rs", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


class Lasso(RegressorMixin, BaseEstimator):
    """Lasso regression backed by Rust.

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength.
    max_iter : int, default=1000
        Maximum number of iterations.
    tol : float, default=1e-4
        Tolerance for convergence.
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.
    """

    def __init__(self, *, alpha=1.0, max_iter=1000, tol=1e-4, fit_intercept=True):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        X, y = validate_data(self, X, y, dtype="float64", y_numeric=True)
        X, y = _ensure_f64(X), _ensure_f64(y)
        self._rs = _RsLasso(
            alpha=self.alpha,
            max_iter=self.max_iter,
            tol=self.tol,
            fit_intercept=self.fit_intercept,
        )
        _fit_rust(self._rs, X, y)
        self.coef_ = np.array(self._rs.coef_)
        self.intercept_ = float(self._rs.intercept_)
        self.n_iter_ = self.max_iter
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype="float64")
        X = _ensure_f64(X)
        if hasattr(self, "_rs"):
            return np.array(self._rs.predict(X))
        return _predict_linear(X, self.coef_, self.intercept_)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_rs", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


class ElasticNet(RegressorMixin, BaseEstimator):
    """ElasticNet regression backed by Rust.

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength.
    l1_ratio : float, default=0.5
        Mix of L1 vs L2 penalty (0=Ridge, 1=Lasso).
    max_iter : int, default=1000
        Maximum number of iterations.
    tol : float, default=1e-4
        Tolerance for convergence.
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.
    """

    def __init__(
        self, *, alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-4, fit_intercept=True
    ):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        X, y = validate_data(self, X, y, dtype="float64", y_numeric=True)
        X, y = _ensure_f64(X), _ensure_f64(y)
        self._rs = _RsElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            max_iter=self.max_iter,
            tol=self.tol,
            fit_intercept=self.fit_intercept,
        )
        _fit_rust(self._rs, X, y)
        self.coef_ = np.array(self._rs.coef_)
        self.intercept_ = float(self._rs.intercept_)
        self.n_iter_ = self.max_iter
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype="float64")
        X = _ensure_f64(X)
        if hasattr(self, "_rs"):
            return np.array(self._rs.predict(X))
        return _predict_linear(X, self.coef_, self.intercept_)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_rs", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
