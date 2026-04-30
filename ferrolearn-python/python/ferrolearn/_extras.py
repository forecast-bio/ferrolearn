"""Phase-2 binding wrappers — sklearn-compatible Python classes for the
~40 estimators added in extras.rs. Minimal API: __init__, fit, predict
or transform. Inherits sklearn mixins for `.score()` / `fit_transform()`.
"""

import numpy as np
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    ClusterMixin,
    RegressorMixin,
    TransformerMixin,
)

from ferrolearn._ferrolearn_rs import (
    _RsARDRegression,
    _RsAdaBoostClassifier,
    _RsAgglomerativeClustering,
    _RsBaggingClassifier,
    _RsBayesianRidge,
    _RsBernoulliNB,
    _RsBirch,
    _RsComplementNB,
    _RsDBSCAN,
    _RsDecisionTreeRegressor,
    _RsExtraTreeClassifier,
    _RsExtraTreesClassifier,
    _RsExtraTreesRegressor,
    _RsFactorAnalysis,
    _RsFastICA,
    _RsGaussianMixture,
    _RsGradientBoostingClassifier,
    _RsGradientBoostingRegressor,
    _RsHistGradientBoostingClassifier,
    _RsHistGradientBoostingRegressor,
    _RsHuberRegressor,
    _RsIncrementalPCA,
    _RsKNeighborsRegressor,
    _RsKernelPCA,
    _RsKernelRidge,
    _RsLinearSVC,
    _RsMaxAbsScaler,
    _RsMinMaxScaler,
    _RsMiniBatchKMeans,
    _RsMultinomialNB,
    _RsNMF,
    _RsNearestCentroid,
    _RsNystroem,
    _RsPowerTransformer,
    _RsQDA,
    _RsQuantileRegressor,
    _RsRBFSampler,
    _RsRandomForestRegressor,
    _RsRidgeClassifier,
    _RsRobustScaler,
    _RsSparsePCA,
    _RsTruncatedSVD,
)


def _f64(a):
    return np.ascontiguousarray(a, dtype=np.float64)


def _i64(a):
    return np.ascontiguousarray(a, dtype=np.int64)


def _encode(y):
    classes = np.unique(y)
    enc = np.searchsorted(classes, y).astype(np.int64)
    return enc, classes


# ---------------------------------------------------------------------------
# Regressor wrappers
# ---------------------------------------------------------------------------

class _RegressorWrapper(RegressorMixin, BaseEstimator):
    _RsClass = None

    def fit(self, X, y):
        self._rs = self._make_rs()
        self._rs.fit(_f64(X), _f64(y))
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        return np.asarray(self._rs.predict(_f64(X)))


class BayesianRidge(_RegressorWrapper):
    def __init__(self, *, max_iter=300, tol=1e-3, fit_intercept=True):
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept

    def _make_rs(self):
        return _RsBayesianRidge(max_iter=self.max_iter, tol=self.tol,
                                fit_intercept=self.fit_intercept)


class ARDRegression(_RegressorWrapper):
    def __init__(self, *, max_iter=300, tol=1e-3, fit_intercept=True):
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept

    def _make_rs(self):
        return _RsARDRegression(max_iter=self.max_iter, tol=self.tol,
                                fit_intercept=self.fit_intercept)


class HuberRegressor(_RegressorWrapper):
    def __init__(self, *, epsilon=1.35, alpha=1e-4, max_iter=100, tol=1e-5,
                 fit_intercept=True):
        self.epsilon = epsilon
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept

    def _make_rs(self):
        return _RsHuberRegressor(epsilon=self.epsilon, alpha=self.alpha,
                                 max_iter=self.max_iter, tol=self.tol,
                                 fit_intercept=self.fit_intercept)


class QuantileRegressor(_RegressorWrapper):
    def __init__(self, *, quantile=0.5, alpha=1.0, max_iter=10000,
                 tol=1e-6, fit_intercept=True):
        self.quantile = quantile
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept

    def _make_rs(self):
        return _RsQuantileRegressor(quantile=self.quantile, alpha=self.alpha,
                                    max_iter=self.max_iter, tol=self.tol,
                                    fit_intercept=self.fit_intercept)


class DecisionTreeRegressor(_RegressorWrapper):
    def __init__(self, *, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def _make_rs(self):
        return _RsDecisionTreeRegressor(max_depth=self.max_depth,
                                        min_samples_split=self.min_samples_split,
                                        min_samples_leaf=self.min_samples_leaf)


class RandomForestRegressor(_RegressorWrapper):
    def __init__(self, *, n_estimators=100, max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def _make_rs(self):
        return _RsRandomForestRegressor(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state)


class ExtraTreesRegressor(_RegressorWrapper):
    def __init__(self, *, n_estimators=100, max_depth=None, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

    def _make_rs(self):
        return _RsExtraTreesRegressor(n_estimators=self.n_estimators,
                                      max_depth=self.max_depth,
                                      random_state=self.random_state)


class GradientBoostingRegressor(_RegressorWrapper):
    def __init__(self, *, n_estimators=100, learning_rate=0.1,
                 max_depth=3, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state

    def _make_rs(self):
        return _RsGradientBoostingRegressor(
            n_estimators=self.n_estimators, learning_rate=self.learning_rate,
            max_depth=self.max_depth, random_state=self.random_state)


class HistGradientBoostingRegressor(_RegressorWrapper):
    def __init__(self, *, n_estimators=100, learning_rate=0.1,
                 max_depth=None, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state

    def _make_rs(self):
        return _RsHistGradientBoostingRegressor(
            n_estimators=self.n_estimators, learning_rate=self.learning_rate,
            max_depth=self.max_depth, random_state=self.random_state)


class KNeighborsRegressor(_RegressorWrapper):
    def __init__(self, *, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def _make_rs(self):
        return _RsKNeighborsRegressor(n_neighbors=self.n_neighbors)


class KernelRidge(_RegressorWrapper):
    def __init__(self, *, alpha=1.0):
        self.alpha = alpha

    def _make_rs(self):
        return _RsKernelRidge(alpha=self.alpha)


# ---------------------------------------------------------------------------
# Classifier wrappers
# ---------------------------------------------------------------------------

class _ClassifierWrapper(ClassifierMixin, BaseEstimator):
    _preprocess_X = staticmethod(_f64)

    def fit(self, X, y):
        Xp = self._preprocess_X(X)
        y_enc, self.classes_ = _encode(y)
        self._rs = self._make_rs()
        self._rs.fit(Xp, y_enc)
        self.n_features_in_ = Xp.shape[1]
        return self

    def predict(self, X):
        Xp = self._preprocess_X(X)
        y_enc = np.asarray(self._rs.predict(Xp))
        return self.classes_[y_enc]


class RidgeClassifier(_ClassifierWrapper):
    def __init__(self, *, alpha=1.0, fit_intercept=True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept

    def _make_rs(self):
        return _RsRidgeClassifier(alpha=self.alpha,
                                  fit_intercept=self.fit_intercept)


class LinearSVC(_ClassifierWrapper):
    def __init__(self, *, C=1.0, max_iter=1000, tol=1e-4):
        self.C = C
        self.max_iter = max_iter
        self.tol = tol

    def _make_rs(self):
        return _RsLinearSVC(c=float(self.C), max_iter=self.max_iter,
                            tol=self.tol)


class QuadraticDiscriminantAnalysis(_ClassifierWrapper):
    def __init__(self, *, reg_param=0.0):
        self.reg_param = reg_param

    def _make_rs(self):
        return _RsQDA(reg_param=self.reg_param)


class MultinomialNB(_ClassifierWrapper):
    def __init__(self, *, alpha=1.0, fit_prior=True):
        self.alpha = alpha
        self.fit_prior = fit_prior

    def _make_rs(self):
        return _RsMultinomialNB(alpha=self.alpha, fit_prior=self.fit_prior)


class BernoulliNB(_ClassifierWrapper):
    def __init__(self, *, alpha=1.0, fit_prior=True, binarize=0.0):
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.binarize = binarize

    def _make_rs(self):
        return _RsBernoulliNB(alpha=self.alpha, fit_prior=self.fit_prior,
                              binarize=self.binarize)


class ComplementNB(_ClassifierWrapper):
    def __init__(self, *, alpha=1.0, fit_prior=True, norm=False):
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.norm = norm

    def _make_rs(self):
        return _RsComplementNB(alpha=self.alpha, fit_prior=self.fit_prior,
                               norm=self.norm)


class ExtraTreeClassifier(_ClassifierWrapper):
    def __init__(self, *, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def _make_rs(self):
        return _RsExtraTreeClassifier(max_depth=self.max_depth,
                                      min_samples_split=self.min_samples_split,
                                      min_samples_leaf=self.min_samples_leaf)


class ExtraTreesClassifier(_ClassifierWrapper):
    def __init__(self, *, n_estimators=100, max_depth=None, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

    def _make_rs(self):
        return _RsExtraTreesClassifier(n_estimators=self.n_estimators,
                                       max_depth=self.max_depth,
                                       random_state=self.random_state)


class AdaBoostClassifier(_ClassifierWrapper):
    def __init__(self, *, n_estimators=50, learning_rate=1.0, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state

    def _make_rs(self):
        return _RsAdaBoostClassifier(n_estimators=self.n_estimators,
                                     learning_rate=self.learning_rate,
                                     random_state=self.random_state)


class GradientBoostingClassifier(_ClassifierWrapper):
    def __init__(self, *, n_estimators=100, learning_rate=0.1,
                 max_depth=3, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state

    def _make_rs(self):
        return _RsGradientBoostingClassifier(
            n_estimators=self.n_estimators, learning_rate=self.learning_rate,
            max_depth=self.max_depth, random_state=self.random_state)


class HistGradientBoostingClassifier(_ClassifierWrapper):
    def __init__(self, *, n_estimators=100, learning_rate=0.1,
                 max_depth=None, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state

    def _make_rs(self):
        return _RsHistGradientBoostingClassifier(
            n_estimators=self.n_estimators, learning_rate=self.learning_rate,
            max_depth=self.max_depth, random_state=self.random_state)


class BaggingClassifier(_ClassifierWrapper):
    def __init__(self, *, n_estimators=10, random_state=None):
        self.n_estimators = n_estimators
        self.random_state = random_state

    def _make_rs(self):
        return _RsBaggingClassifier(n_estimators=self.n_estimators,
                                    random_state=self.random_state)


class NearestCentroid(_ClassifierWrapper):
    def __init__(self):
        """No tunable hyperparameters; defaults match sklearn."""

    def _make_rs(self):
        return _RsNearestCentroid()


# ---------------------------------------------------------------------------
# Cluster wrappers
# ---------------------------------------------------------------------------

class _ClusterWrapper(ClusterMixin, BaseEstimator):
    def fit(self, X, y=None):
        self._rs = self._make_rs()
        self._rs.fit(_f64(X))
        self.labels_ = np.asarray(self._rs.labels_)
        self.n_features_in_ = X.shape[1]
        return self

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_


class MiniBatchKMeans(_ClusterWrapper):
    def __init__(self, *, n_clusters=8, max_iter=100, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state

    def _make_rs(self):
        return _RsMiniBatchKMeans(n_clusters=self.n_clusters,
                                  max_iter=self.max_iter,
                                  random_state=self.random_state)

    def predict(self, X):
        return np.asarray(self._rs.predict(_f64(X)))


class DBSCAN(_ClusterWrapper):
    def __init__(self, *, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples

    def _make_rs(self):
        return _RsDBSCAN(eps=self.eps, min_samples=self.min_samples)


class AgglomerativeClustering(_ClusterWrapper):
    def __init__(self, *, n_clusters=2):
        self.n_clusters = n_clusters

    def _make_rs(self):
        return _RsAgglomerativeClustering(n_clusters=self.n_clusters)


class Birch(_ClusterWrapper):
    def __init__(self, *, n_clusters=None, threshold=0.5):
        self.n_clusters = n_clusters
        self.threshold = threshold

    def _make_rs(self):
        return _RsBirch(n_clusters=self.n_clusters, threshold=self.threshold)


class GaussianMixture(BaseEstimator):
    """sklearn places GaussianMixture in `sklearn.mixture` (not cluster);
    we mirror that style — fit/predict/labels_."""

    def __init__(self, *, n_components=1, max_iter=100, random_state=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y=None):
        self._rs = _RsGaussianMixture(n_components=self.n_components,
                                      max_iter=self.max_iter,
                                      random_state=self.random_state)
        self._rs.fit(_f64(X))
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        return np.asarray(self._rs.predict(_f64(X)))

    def fit_predict(self, X, y=None):
        return self.fit(X).predict(X)


# ---------------------------------------------------------------------------
# Decomp / preprocess transformer wrappers
# ---------------------------------------------------------------------------

class _TransformerWrapper(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self._rs = self._make_rs()
        self._rs.fit(_f64(X))
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return np.asarray(self._rs.transform(_f64(X)))


class IncrementalPCA(_TransformerWrapper):
    def __init__(self, *, n_components=2):
        self.n_components = n_components

    def _make_rs(self):
        return _RsIncrementalPCA(n_components=self.n_components)


class TruncatedSVD(_TransformerWrapper):
    def __init__(self, *, n_components=2):
        self.n_components = n_components

    def _make_rs(self):
        return _RsTruncatedSVD(n_components=self.n_components)


class FastICA(_TransformerWrapper):
    def __init__(self, *, n_components=2):
        self.n_components = n_components

    def _make_rs(self):
        return _RsFastICA(n_components=self.n_components)


class NMF(_TransformerWrapper):
    def __init__(self, *, n_components=2):
        self.n_components = n_components

    def _make_rs(self):
        return _RsNMF(n_components=self.n_components)


class KernelPCA(_TransformerWrapper):
    def __init__(self, *, n_components=2):
        self.n_components = n_components

    def _make_rs(self):
        return _RsKernelPCA(n_components=self.n_components)


class SparsePCA(_TransformerWrapper):
    def __init__(self, *, n_components=2):
        self.n_components = n_components

    def _make_rs(self):
        return _RsSparsePCA(n_components=self.n_components)


class FactorAnalysis(_TransformerWrapper):
    def __init__(self, *, n_components=2):
        self.n_components = n_components

    def _make_rs(self):
        return _RsFactorAnalysis(n_components=self.n_components)


class MinMaxScaler(_TransformerWrapper):
    def __init__(self):
        """Defaults match sklearn (feature_range=(0, 1))."""

    def _make_rs(self):
        return _RsMinMaxScaler()


class MaxAbsScaler(_TransformerWrapper):
    def __init__(self):
        """No tunable hyperparameters."""

    def _make_rs(self):
        return _RsMaxAbsScaler()


class RobustScaler(_TransformerWrapper):
    def __init__(self):
        """Defaults match sklearn (quantile_range=(25.0, 75.0))."""

    def _make_rs(self):
        return _RsRobustScaler()


class PowerTransformer(_TransformerWrapper):
    def __init__(self):
        """Defaults match sklearn (method='yeo-johnson')."""

    def _make_rs(self):
        return _RsPowerTransformer()


class Nystroem(_TransformerWrapper):
    def __init__(self):
        """Defaults match sklearn (kernel='rbf', n_components=100)."""

    def _make_rs(self):
        return _RsNystroem()


class RBFSampler(_TransformerWrapper):
    def __init__(self):
        """Defaults match sklearn (gamma=1.0, n_components=100)."""

    def _make_rs(self):
        return _RsRBFSampler()
