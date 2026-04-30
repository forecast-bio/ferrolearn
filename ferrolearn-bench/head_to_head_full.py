#!/usr/bin/env python3
"""Full head-to-head bench — every bound ferrolearn estimator vs its
scikit-learn equivalent. Same dataset (sklearn `make_*`), same train/test
split, same hyperparameters, same quality metric, both libraries fit +
predict in the same Python process.

Output: JSON array of records {family, algorithm, dataset, sklearn:{...}, ferrolearn:{...}}.
Render with `render_head_to_head.py`.

Usage:
    python head_to_head_full.py > h2h_full.json
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import sklearn

# sklearn imports
import sklearn.cluster as sk_cluster
import sklearn.decomposition as sk_decomp
import sklearn.discriminant_analysis as sk_da
import sklearn.ensemble as sk_ens
import sklearn.kernel_approximation as sk_ka
import sklearn.kernel_ridge as sk_kr
import sklearn.linear_model as sk_lm
import sklearn.mixture as sk_mix
import sklearn.naive_bayes as sk_nb
import sklearn.neighbors as sk_nn
import sklearn.preprocessing as sk_pp
import sklearn.svm as sk_svm
import sklearn.tree as sk_tr
from sklearn.datasets import make_blobs, make_classification, make_regression
from sklearn.metrics import accuracy_score, adjusted_rand_score, r2_score
from sklearn.model_selection import train_test_split

import ferrolearn as fl


warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Dataset sizes
# ---------------------------------------------------------------------------

REGRESSION_SIZES = [
    ("tiny_50x5", 50, 5),
    ("small_1Kx10", 1_000, 10),
    ("medium_10Kx100", 10_000, 100),
]

CLASSIFICATION_SIZES = [
    ("tiny_50x5", 50, 5),
    ("small_1Kx10", 1_000, 10),
    ("medium_10Kx100", 10_000, 100),
]

CLUSTER_SIZES = [
    ("tiny_200x5", 200, 5),
    ("small_1Kx10", 1_000, 10),
    ("medium_5Kx20", 5_000, 20),
]


def median_us(fn, iters=7, slow=False):
    if slow:
        t = time.perf_counter()
        fn()
        return (time.perf_counter() - t) * 1e6
    times = []
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return sorted(times)[len(times) // 2] * 1e6


def regression_split(n, p, seed=42):
    X, y = make_regression(n_samples=n, n_features=p, n_informative=p,
                           noise=0.1, random_state=seed)
    return train_test_split(X, y, test_size=0.2, random_state=seed)


def classification_split(n, p, seed=42, n_classes=2):
    X, y = make_classification(n_samples=n, n_features=p,
                               n_informative=max(2, p // 2),
                               n_classes=n_classes, random_state=seed)
    return train_test_split(X, y, test_size=0.2, random_state=seed)


def cluster_data(n, p, seed=42, n_centers=8):
    return make_blobs(n_samples=n, n_features=p, centers=n_centers, random_state=seed)


# ---------------------------------------------------------------------------
# Record helpers
# ---------------------------------------------------------------------------

@dataclass
class Side:
    fit_us: Optional[float] = None
    predict_us: Optional[float] = None
    score: Optional[float] = None
    notes: Optional[str] = None

    def asdict(self):
        return {
            "fit_us": self.fit_us,
            "predict_us": self.predict_us,
            "score": float(self.score) if self.score is not None else None,
            "notes": self.notes,
        }


def record(family, algo, label, n, p, metric, sk_side, fl_side):
    return {
        "family": family,
        "algorithm": algo,
        "dataset": label,
        "n_samples": n,
        "n_features": p,
        "metric": metric,
        "sklearn": sk_side.asdict(),
        "ferrolearn": fl_side.asdict(),
    }


def safe(fn):
    try:
        return fn(), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def time_supervised(side, factory, xtr, ytr, xte, yte, score_fn, slow):
    res, err = safe(lambda: median_us(lambda: factory().fit(xtr, ytr), slow=slow))
    if err:
        side.notes = err
        return
    side.fit_us = res
    fitted, err = safe(lambda: factory().fit(xtr, ytr))
    if err:
        side.notes = err
        return
    side.predict_us, _ = safe(lambda: median_us(lambda: fitted.predict(xte)))
    yhat, err = safe(lambda: fitted.predict(xte))
    if err:
        side.notes = err
        return
    s, _ = safe(lambda: score_fn(yte, yhat))
    side.score = s


def head_to_head_supervised(family, algo, label, n, p, metric, score_fn,
                            xtr, xte, ytr, yte, sk_factory, fl_factory, slow=False):
    sk_side = Side()
    fl_side = Side()
    time_supervised(sk_side, sk_factory, xtr, ytr, xte, yte, score_fn, slow)
    time_supervised(fl_side, fl_factory, xtr, ytr, xte, yte, score_fn, slow)
    return record(family, algo, label, n, p, metric, sk_side, fl_side)


def head_to_head_cluster(algo, label, n, p, x, y_true, sk_factory, fl_factory, slow=False):
    sk_side = Side()
    fl_side = Side()

    def time_one(side, factory):
        res, err = safe(lambda: median_us(lambda: factory().fit(x), slow=slow))
        if err:
            side.notes = err
            return
        side.fit_us = res
        fitted, err = safe(lambda: factory().fit(x))
        if err:
            side.notes = err
            return
        labels = getattr(fitted, "labels_", None)
        if labels is None:
            try:
                labels = fitted.predict(x)
            except Exception as exc:
                side.notes = f"{type(exc).__name__}: {exc}"
                return
        s, _ = safe(lambda: adjusted_rand_score(y_true, labels))
        side.score = s

    time_one(sk_side, sk_factory)
    time_one(fl_side, fl_factory)
    return record("cluster", algo, label, n, p, "ari", sk_side, fl_side)


def head_to_head_decomp(algo, label, n, p, x, sk_factory, fl_factory, slow=False):
    """Reconstruction error = ||X - inverse_transform(transform(X))||_F / ||X||_F."""
    sk_side = Side()
    fl_side = Side()

    def relrec(fitted):
        z = fitted.transform(x)
        try:
            xh = fitted.inverse_transform(z)
            return float(np.linalg.norm(x - xh) / max(np.linalg.norm(x), 1e-30))
        except Exception:
            return None

    def time_one(side, factory):
        res, err = safe(lambda: median_us(lambda: factory().fit(x), slow=slow))
        if err:
            side.notes = err
            return
        side.fit_us = res
        fitted, err = safe(lambda: factory().fit(x))
        if err:
            side.notes = err
            return
        side.predict_us, _ = safe(lambda: median_us(lambda: fitted.transform(x)))
        side.score, _ = safe(lambda: relrec(fitted))

    time_one(sk_side, sk_factory)
    time_one(fl_side, fl_factory)
    return record("decomp", algo, label, n, p, "recon_rel", sk_side, fl_side)


def head_to_head_scaler(algo, label, n, p, x, sk_factory, fl_factory):
    """Numerical agreement after fit_transform: ||sk_xt - fl_xt||_F / ||sk_xt||_F."""
    sk_side = Side()
    fl_side = Side()

    def time_one(side, factory):
        res, err = safe(lambda: median_us(lambda: factory().fit(x)))
        if err:
            side.notes = err
            return None
        side.fit_us = res
        fitted, err = safe(lambda: factory().fit(x))
        if err:
            side.notes = err
            return None
        side.predict_us, _ = safe(lambda: median_us(lambda: fitted.transform(x)))
        out, _ = safe(lambda: fitted.transform(x))
        return out

    sk_xt = time_one(sk_side, sk_factory)
    fl_xt = time_one(fl_side, fl_factory)

    if sk_xt is not None and fl_xt is not None:
        denom = max(float(np.linalg.norm(sk_xt)), 1e-30)
        sk_side.score = 0.0
        fl_side.score = float(np.linalg.norm(sk_xt - fl_xt) / denom)

    return record("preprocess", algo, label, n, p, "rel_diff_vs_sklearn",
                  sk_side, fl_side)


# ---------------------------------------------------------------------------
# Hyperparameter pairs (sklearn factory, ferrolearn factory)
# Each entry takes (label, n, p) and returns (sk_factory, fl_factory).
# ---------------------------------------------------------------------------

def reg_pairs(label, n, p):
    return [
        ("LinearRegression", lambda: sk_lm.LinearRegression(),
                              lambda: fl.LinearRegression()),
        ("Ridge", lambda: sk_lm.Ridge(alpha=1.0),
                  lambda: fl.Ridge(alpha=1.0)),
        ("Lasso", lambda: sk_lm.Lasso(alpha=1.0, max_iter=1000, tol=1e-4),
                  lambda: fl.Lasso(alpha=1.0, max_iter=1000, tol=1e-4)),
        ("ElasticNet", lambda: sk_lm.ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-4),
                       lambda: fl.ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-4)),
        ("BayesianRidge", lambda: sk_lm.BayesianRidge(max_iter=300, tol=1e-3),
                          lambda: fl.BayesianRidge(max_iter=300, tol=1e-3)),
        ("ARDRegression", lambda: sk_lm.ARDRegression(max_iter=300, tol=1e-3),
                          lambda: fl.ARDRegression(max_iter=300, tol=1e-3)),
        ("HuberRegressor", lambda: sk_lm.HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=100, tol=1e-5),
                           lambda: fl.HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=100, tol=1e-5)),
        ("QuantileRegressor", lambda: sk_lm.QuantileRegressor(quantile=0.5, alpha=1.0),
                              lambda: fl.QuantileRegressor(quantile=0.5, alpha=1.0)),
        ("DecisionTreeRegressor", lambda: sk_tr.DecisionTreeRegressor(random_state=42),
                                  lambda: fl.DecisionTreeRegressor()),
        ("RandomForestRegressor",
            lambda: sk_ens.RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            lambda: fl.RandomForestRegressor(n_estimators=100, random_state=42)),
        ("ExtraTreesRegressor",
            lambda: sk_ens.ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            lambda: fl.ExtraTreesRegressor(n_estimators=100, random_state=42)),
        ("HistGradientBoostingRegressor",
            lambda: sk_ens.HistGradientBoostingRegressor(max_iter=100, learning_rate=0.1, random_state=42),
            lambda: fl.HistGradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)),
        ("KNeighborsRegressor", lambda: sk_nn.KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
                                lambda: fl.KNeighborsRegressor(n_neighbors=5)),
    ] + ([
        ("GradientBoostingRegressor",
            lambda: sk_ens.GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
            lambda: fl.GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)),
        ("KernelRidge", lambda: sk_kr.KernelRidge(alpha=1.0),
                        lambda: fl.KernelRidge(alpha=1.0)),
    ] if n <= 2_000 else [])


def cls_pairs(label, n, p):
    pairs = [
        ("LogisticRegression",
            lambda: sk_lm.LogisticRegression(C=1.0, max_iter=1000, tol=1e-4),
            lambda: fl.LogisticRegression(C=1.0, max_iter=1000, tol=1e-4)),
        ("RidgeClassifier", lambda: sk_lm.RidgeClassifier(alpha=1.0),
                            lambda: fl.RidgeClassifier(alpha=1.0)),
        ("LinearSVC", lambda: sk_svm.LinearSVC(C=1.0, max_iter=1000),
                      lambda: fl.LinearSVC(C=1.0, max_iter=1000)),
        ("QDA", lambda: sk_da.QuadraticDiscriminantAnalysis(reg_param=0.0),
                lambda: fl.QuadraticDiscriminantAnalysis(reg_param=0.0)),
        ("GaussianNB", lambda: sk_nb.GaussianNB(var_smoothing=1e-9),
                       lambda: fl.GaussianNB(var_smoothing=1e-9)),
        ("DecisionTreeClassifier",
            lambda: sk_tr.DecisionTreeClassifier(random_state=42),
            lambda: fl.DecisionTreeClassifier()),
        ("ExtraTreeClassifier",
            lambda: sk_tr.ExtraTreeClassifier(random_state=42),
            lambda: fl.ExtraTreeClassifier()),
        ("RandomForestClassifier",
            lambda: sk_ens.RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            lambda: fl.RandomForestClassifier(n_estimators=100, random_state=42)),
        ("ExtraTreesClassifier",
            lambda: sk_ens.ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            lambda: fl.ExtraTreesClassifier(n_estimators=100, random_state=42)),
        ("HistGradientBoostingClassifier",
            lambda: sk_ens.HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, random_state=42),
            lambda: fl.HistGradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)),
        ("KNeighborsClassifier",
            lambda: sk_nn.KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
            lambda: fl.KNeighborsClassifier(n_neighbors=5)),
        ("NearestCentroid", lambda: sk_nn.NearestCentroid(),
                            lambda: fl.NearestCentroid()),
    ]
    if n <= 1_000:
        pairs += [
            ("AdaBoostClassifier",
                lambda: sk_ens.AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=42),
                lambda: fl.AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=42)),
            ("BaggingClassifier",
                lambda: sk_ens.BaggingClassifier(n_estimators=10, random_state=42, n_jobs=-1),
                lambda: fl.BaggingClassifier(n_estimators=10, random_state=42)),
            ("GradientBoostingClassifier",
                lambda: sk_ens.GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
                lambda: fl.GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)),
        ]
    return pairs


def cluster_pairs(label, n, p):
    pairs = [
        ("KMeans",
            lambda: sk_cluster.KMeans(n_clusters=8, n_init=10, random_state=42),
            lambda: fl.KMeans(n_clusters=8, n_init=10, random_state=42)),
        ("MiniBatchKMeans",
            lambda: sk_cluster.MiniBatchKMeans(n_clusters=8, random_state=42, n_init=3),
            lambda: fl.MiniBatchKMeans(n_clusters=8, random_state=42)),
        ("GaussianMixture",
            lambda: sk_mix.GaussianMixture(n_components=8, max_iter=100, random_state=42),
            lambda: fl.GaussianMixture(n_components=8, max_iter=100, random_state=42)),
    ]
    if n <= 1_000:
        pairs += [
            ("AgglomerativeClustering",
                lambda: sk_cluster.AgglomerativeClustering(n_clusters=8),
                lambda: fl.AgglomerativeClustering(n_clusters=8)),
            ("DBSCAN", lambda: sk_cluster.DBSCAN(eps=1.0, min_samples=5),
                       lambda: fl.DBSCAN(eps=1.0, min_samples=5)),
            ("Birch", lambda: sk_cluster.Birch(n_clusters=8, threshold=0.5),
                      lambda: fl.Birch(n_clusters=8, threshold=0.5)),
        ]
    return pairs


def decomp_pairs(label, n, p):
    n_comp = max(1, min(p - 1, 5))
    pairs = [
        ("PCA", lambda: sk_decomp.PCA(n_components=n_comp, random_state=42),
                lambda: fl.PCA(n_components=n_comp)),
        ("IncrementalPCA",
            lambda: sk_decomp.IncrementalPCA(n_components=n_comp),
            lambda: fl.IncrementalPCA(n_components=n_comp)),
        ("TruncatedSVD",
            lambda: sk_decomp.TruncatedSVD(n_components=n_comp, random_state=42),
            lambda: fl.TruncatedSVD(n_components=n_comp)),
    ]
    if n <= 1_000:
        pairs += [
            ("FactorAnalysis",
                lambda: sk_decomp.FactorAnalysis(n_components=n_comp, random_state=42),
                lambda: fl.FactorAnalysis(n_components=n_comp)),
            ("FastICA",
                lambda: sk_decomp.FastICA(n_components=n_comp, random_state=42, max_iter=200),
                lambda: fl.FastICA(n_components=n_comp)),
            ("SparsePCA",
                lambda: sk_decomp.SparsePCA(n_components=n_comp, random_state=42),
                lambda: fl.SparsePCA(n_components=n_comp)),
        ]
    return pairs


def preprocess_pairs(label, n, p):
    return [
        ("StandardScaler", lambda: sk_pp.StandardScaler(), lambda: fl.StandardScaler()),
        ("MinMaxScaler", lambda: sk_pp.MinMaxScaler(), lambda: fl.MinMaxScaler()),
        ("MaxAbsScaler", lambda: sk_pp.MaxAbsScaler(), lambda: fl.MaxAbsScaler()),
        ("RobustScaler", lambda: sk_pp.RobustScaler(), lambda: fl.RobustScaler()),
    ] + ([
        ("PowerTransformer", lambda: sk_pp.PowerTransformer(), lambda: fl.PowerTransformer()),
    ] if n <= 1_000 else [])


# ---------------------------------------------------------------------------
# Family runners
# ---------------------------------------------------------------------------

def run_regressors(records):
    for label, n, p in REGRESSION_SIZES:
        xtr, xte, ytr, yte = regression_split(n, p)
        slow = n > 1_000
        for algo, sk_f, fl_f in reg_pairs(label, n, p):
            print(f"  reg {algo} @ {label}", file=sys.stderr)
            records.append(head_to_head_supervised(
                "regressor", algo, label, n, p, "r2", r2_score,
                xtr, xte, ytr, yte, sk_f, fl_f, slow=slow))


def run_classifiers(records):
    for label, n, p in CLASSIFICATION_SIZES:
        xtr, xte, ytr, yte = classification_split(n, p)
        slow = n > 1_000
        for algo, sk_f, fl_f in cls_pairs(label, n, p):
            print(f"  cls {algo} @ {label}", file=sys.stderr)
            records.append(head_to_head_supervised(
                "classifier", algo, label, n, p, "accuracy", accuracy_score,
                xtr, xte, ytr, yte, sk_f, fl_f, slow=slow))

        # Non-negative-feature NB variants (use |X|)
        x_pos = np.abs(np.vstack([xtr, xte]))
        n_train = xtr.shape[0]
        xtr_pos, xte_pos = x_pos[:n_train], x_pos[n_train:]
        for algo, sk_class, fl_class in [
            ("MultinomialNB", sk_nb.MultinomialNB, fl.MultinomialNB),
            ("ComplementNB", sk_nb.ComplementNB, fl.ComplementNB),
        ]:
            print(f"  cls {algo} @ {label}", file=sys.stderr)
            records.append(head_to_head_supervised(
                "classifier", algo, label, n, p, "accuracy", accuracy_score,
                xtr_pos, xte_pos, ytr, yte,
                lambda c=sk_class: c(alpha=1.0),
                lambda c=fl_class: c(alpha=1.0),
                slow=slow))

        # Bernoulli NB on binarized X
        x_bin = (np.vstack([xtr, xte]) > 0).astype(np.float64)
        xtr_bin, xte_bin = x_bin[:n_train], x_bin[n_train:]
        print(f"  cls BernoulliNB @ {label}", file=sys.stderr)
        records.append(head_to_head_supervised(
            "classifier", "BernoulliNB", label, n, p, "accuracy", accuracy_score,
            xtr_bin, xte_bin, ytr, yte,
            lambda: sk_nb.BernoulliNB(alpha=1.0),
            lambda: fl.BernoulliNB(alpha=1.0),
            slow=slow))


def run_clusterers(records):
    for label, n, p in CLUSTER_SIZES:
        x, y_true = cluster_data(n, p)
        slow = n > 1_000
        for algo, sk_f, fl_f in cluster_pairs(label, n, p):
            print(f"  cluster {algo} @ {label}", file=sys.stderr)
            records.append(head_to_head_cluster(
                algo, label, n, p, x, y_true, sk_f, fl_f, slow=slow))


def run_decomp(records):
    for label, n, p in REGRESSION_SIZES:
        x, _ = make_regression(n_samples=n, n_features=p, n_informative=p,
                               noise=0.1, random_state=42)
        slow = n > 1_000
        for algo, sk_f, fl_f in decomp_pairs(label, n, p):
            print(f"  decomp {algo} @ {label}", file=sys.stderr)
            records.append(head_to_head_decomp(
                algo, label, n, p, x, sk_f, fl_f, slow=slow))


def run_preprocess(records):
    for label, n, p in REGRESSION_SIZES:
        x, _ = make_regression(n_samples=n, n_features=p, n_informative=p,
                               noise=0.1, random_state=42)
        for algo, sk_f, fl_f in preprocess_pairs(label, n, p):
            print(f"  preprocess {algo} @ {label}", file=sys.stderr)
            records.append(head_to_head_scaler(
                algo, label, n, p, x, sk_f, fl_f))


def run_kernel_approx(records):
    for label, n, p in REGRESSION_SIZES:
        x, _ = make_regression(n_samples=n, n_features=p, n_informative=p,
                               noise=0.1, random_state=42)
        # Nystroem / RBFSampler — output dimensions don't match between libs
        # (different default n_components or random projections). Use scaler-style
        # comparison (timing only, no rel_diff).
        for algo, sk_class, fl_class in [
            ("Nystroem", sk_ka.Nystroem, fl.Nystroem),
            ("RBFSampler", sk_ka.RBFSampler, fl.RBFSampler),
        ]:
            print(f"  kernel-approx {algo} @ {label}", file=sys.stderr)

            sk_side = Side()
            fl_side = Side()

            def time_one(side, factory):
                res, err = safe(lambda: median_us(lambda: factory().fit(x)))
                if err:
                    side.notes = err
                    return
                side.fit_us = res
                fitted, err = safe(lambda: factory().fit(x))
                if err:
                    side.notes = err
                    return
                side.predict_us, _ = safe(lambda: median_us(lambda: fitted.transform(x)))

            time_one(sk_side, lambda: sk_class(random_state=42))
            time_one(fl_side, lambda: fl_class())
            records.append(record("kernel", algo, label, n, p, "timing_only",
                                  sk_side, fl_side))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"# sklearn={sklearn.__version__}, "
          f"ferrolearn=0.3.0, numpy={np.__version__}, "
          f"python={sys.version.split()[0]}", file=sys.stderr)

    records = []
    print("[h2h-full] regressors...", file=sys.stderr)
    run_regressors(records)
    print("[h2h-full] classifiers...", file=sys.stderr)
    run_classifiers(records)
    print("[h2h-full] clusterers...", file=sys.stderr)
    run_clusterers(records)
    print("[h2h-full] decomp...", file=sys.stderr)
    run_decomp(records)
    print("[h2h-full] preprocess...", file=sys.stderr)
    run_preprocess(records)
    print("[h2h-full] kernel approx...", file=sys.stderr)
    run_kernel_approx(records)

    print(json.dumps(records, indent=2))
    print(f"[h2h-full] {len(records)} records emitted", file=sys.stderr)


if __name__ == "__main__":
    main()
