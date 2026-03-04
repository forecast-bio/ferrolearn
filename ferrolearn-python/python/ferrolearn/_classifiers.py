"""sklearn-compatible wrappers for ferrolearn classification models."""

import re

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted, validate_data

from ferrolearn._ferrolearn_rs import (
    _RsDecisionTreeClassifier,
    _RsGaussianNB,
    _RsKNeighborsClassifier,
    _RsLogisticRegression,
    _RsRandomForestClassifier,
)


def _ensure_f64(arr):
    """Ensure array is C-contiguous float64 (required by Rust bindings)."""
    return np.ascontiguousarray(arr, dtype=np.float64)


def _encode_labels(y):
    """Encode arbitrary labels to contiguous integers and return the mapping."""
    classes = np.unique(y)
    label_map = {c: i for i, c in enumerate(classes)}
    y_encoded = np.array([label_map[v] for v in y], dtype=np.int64)
    return y_encoded, classes


def _decode_labels(y_encoded, classes):
    """Decode integer labels back to original labels."""
    return classes[y_encoded]


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


def _check_classification_target(y):
    """Raise ValueError if y is continuous (not a classification target)."""
    y_type = type_of_target(y)
    if y_type in ("continuous", "continuous-multioutput"):
        raise ValueError(
            f"Unknown label type: {y_type!r}. Maybe you are trying to fit a "
            "classifier on a regression target with continuous values."
        )


class _ClassifierPickleMixin:
    """Mixin for pickling classifiers. Stores training data for re-fit on unpickle."""

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_rs", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Reconstruct _rs by re-fitting if we have stored training data
        if hasattr(self, "_fit_X") and hasattr(self, "_fit_y_encoded"):
            self._rebuild_rs()

    def _store_training_data(self, X, y_encoded):
        """Store training data for pickle reconstruction."""
        self._fit_X = X.copy()
        self._fit_y_encoded = y_encoded.copy()


class LogisticRegression(_ClassifierPickleMixin, ClassifierMixin, BaseEstimator):
    """Logistic Regression backed by Rust.

    Parameters
    ----------
    C : float, default=1.0
        Inverse of regularization strength.
    max_iter : int, default=1000
        Maximum number of iterations.
    tol : float, default=1e-4
        Tolerance for convergence.
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.
    """

    def __init__(self, *, C=1.0, max_iter=1000, tol=1e-4, fit_intercept=True):
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        X, y = validate_data(self, X, y, dtype="float64")
        _check_classification_target(y)
        X = _ensure_f64(X)
        y_encoded, self.classes_ = _encode_labels(y)
        self._rs = _RsLogisticRegression(
            c=float(self.C),
            max_iter=self.max_iter,
            tol=self.tol,
            fit_intercept=self.fit_intercept,
        )
        _fit_rust(self._rs, X, y_encoded)
        self.coef_ = np.array(self._rs.coef_).reshape(1, -1)
        self.intercept_ = np.array([float(self._rs.intercept_)])
        self.n_iter_ = self.max_iter
        self._store_training_data(X, y_encoded)
        return self

    def _rebuild_rs(self):
        self._rs = _RsLogisticRegression(
            c=float(self.C),
            max_iter=self.max_iter,
            tol=self.tol,
            fit_intercept=self.fit_intercept,
        )
        self._rs.fit(self._fit_X, self._fit_y_encoded)

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype="float64")
        X = _ensure_f64(X)
        if not hasattr(self, "_rs"):
            self._rebuild_rs()
        y_encoded = np.asarray(self._rs.predict(X))
        return _decode_labels(y_encoded, self.classes_)

    def predict_proba(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype="float64")
        X = _ensure_f64(X)
        if not hasattr(self, "_rs"):
            self._rebuild_rs()
        return np.array(self._rs.predict_proba(X))


class DecisionTreeClassifier(_ClassifierPickleMixin, ClassifierMixin, BaseEstimator):
    """Decision Tree Classifier backed by Rust.

    Parameters
    ----------
    max_depth : int or None, default=None
        Maximum depth of the tree.
    min_samples_split : int, default=2
        Minimum number of samples required to split a node.
    min_samples_leaf : int, default=1
        Minimum number of samples required at a leaf node.
    """

    def __init__(
        self, *, max_depth=None, min_samples_split=2, min_samples_leaf=1
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        X, y = validate_data(self, X, y, dtype="float64")
        _check_classification_target(y)
        X = _ensure_f64(X)
        y_encoded, self.classes_ = _encode_labels(y)
        self._rs = _RsDecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
        )
        _fit_rust(self._rs, X, y_encoded)
        self._store_training_data(X, y_encoded)
        return self

    def _rebuild_rs(self):
        self._rs = _RsDecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
        )
        self._rs.fit(self._fit_X, self._fit_y_encoded)

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype="float64")
        X = _ensure_f64(X)
        if not hasattr(self, "_rs"):
            self._rebuild_rs()
        y_encoded = np.asarray(self._rs.predict(X))
        return _decode_labels(y_encoded, self.classes_)

    def predict_proba(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype="float64")
        X = _ensure_f64(X)
        if not hasattr(self, "_rs"):
            self._rebuild_rs()
        return np.array(self._rs.predict_proba(X))


class RandomForestClassifier(_ClassifierPickleMixin, ClassifierMixin, BaseEstimator):
    """Random Forest Classifier backed by Rust.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest.
    max_depth : int or None, default=None
        Maximum depth of the trees.
    min_samples_split : int, default=2
        Minimum number of samples required to split a node.
    min_samples_leaf : int, default=1
        Minimum number of samples required at a leaf node.
    random_state : int or None, default=None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        *,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def fit(self, X, y):
        X, y = validate_data(self, X, y, dtype="float64")
        _check_classification_target(y)
        X = _ensure_f64(X)
        y_encoded, self.classes_ = _encode_labels(y)
        self._rs = _RsRandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )
        _fit_rust(self._rs, X, y_encoded)
        self._store_training_data(X, y_encoded)
        return self

    def _rebuild_rs(self):
        self._rs = _RsRandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )
        self._rs.fit(self._fit_X, self._fit_y_encoded)

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype="float64")
        X = _ensure_f64(X)
        if not hasattr(self, "_rs"):
            self._rebuild_rs()
        y_encoded = np.asarray(self._rs.predict(X))
        return _decode_labels(y_encoded, self.classes_)


class KNeighborsClassifier(_ClassifierPickleMixin, ClassifierMixin, BaseEstimator):
    """K-Nearest Neighbors Classifier backed by Rust.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use.
    """

    def __init__(self, *, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        X, y = validate_data(self, X, y, dtype="float64")
        _check_classification_target(y)
        X = _ensure_f64(X)
        y_encoded, self.classes_ = _encode_labels(y)
        self._rs = _RsKNeighborsClassifier(n_neighbors=self.n_neighbors)
        _fit_rust(self._rs, X, y_encoded)
        self._store_training_data(X, y_encoded)
        return self

    def _rebuild_rs(self):
        self._rs = _RsKNeighborsClassifier(n_neighbors=self.n_neighbors)
        self._rs.fit(self._fit_X, self._fit_y_encoded)

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype="float64")
        X = _ensure_f64(X)
        if not hasattr(self, "_rs"):
            self._rebuild_rs()
        y_encoded = np.asarray(self._rs.predict(X))
        return _decode_labels(y_encoded, self.classes_)


class GaussianNB(_ClassifierPickleMixin, ClassifierMixin, BaseEstimator):
    """Gaussian Naive Bayes classifier backed by Rust.

    Parameters
    ----------
    var_smoothing : float, default=1e-9
        Variance smoothing parameter.
    """

    def __init__(self, *, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        X, y = validate_data(self, X, y, dtype="float64")
        _check_classification_target(y)
        X = _ensure_f64(X)
        y_encoded, self.classes_ = _encode_labels(y)
        self._rs = _RsGaussianNB(var_smoothing=self.var_smoothing)
        _fit_rust(self._rs, X, y_encoded)
        self._store_training_data(X, y_encoded)
        return self

    def _rebuild_rs(self):
        self._rs = _RsGaussianNB(var_smoothing=self.var_smoothing)
        self._rs.fit(self._fit_X, self._fit_y_encoded)

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype="float64")
        X = _ensure_f64(X)
        if not hasattr(self, "_rs"):
            self._rebuild_rs()
        y_encoded = np.asarray(self._rs.predict(X))
        return _decode_labels(y_encoded, self.classes_)

    def predict_proba(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype="float64")
        X = _ensure_f64(X)
        if not hasattr(self, "_rs"):
            self._rebuild_rs()
        return np.array(self._rs.predict_proba(X))
