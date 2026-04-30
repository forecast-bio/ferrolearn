#!/usr/bin/env python3
"""Comprehensive sklearn benchmark — matches the Rust harness binary.

Emits a JSON array with the same schema as `cargo run --release --bin harness`:

    [
      {
        "family": "regressor",
        "algorithm": "Ridge",
        "dataset": "small_1Kx10",
        "n_samples": 1000,
        "n_features": 10,
        "fit_us": 482.0,
        "predict_us": 27.0,
        "metric": "r2",
        "score": 0.99
      }
    ]
    (one entry per estimator x dataset combination)

Pair with the Rust harness output, then run `compare.py` to render the
ferrolearn-vs-sklearn report.

Usage:
    python sklearn_bench.py > sklearn_bench.json
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from typing import Callable

import numpy as np
from sklearn.cluster import (
    DBSCAN,
    AgglomerativeClustering,
    Birch,
    BisectingKMeans,
    KMeans,
    MeanShift,
    MiniBatchKMeans,
    SpectralClustering,
)
from sklearn.datasets import make_blobs, make_classification, make_regression
from sklearn.decomposition import (
    PCA,
    FactorAnalysis,
    FastICA,
    IncrementalPCA,
    KernelPCA,
    NMF,
    SparsePCA,
    TruncatedSVD,
)
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import (
    ARDRegression,
    BayesianRidge,
    ElasticNet,
    HuberRegressor,
    Lasso,
    LinearRegression,
    LogisticRegression,
    QuantileRegressor,
    Ridge,
    RidgeClassifier,
)
from sklearn.metrics import accuracy_score, adjusted_rand_score, r2_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, ComplementNB, GaussianNB, MultinomialNB
from sklearn.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
    NearestCentroid,
)
from sklearn.preprocessing import (
    KBinsDiscretizer,
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    PolynomialFeatures,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)
from sklearn.svm import LinearSVC
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    ExtraTreeClassifier,
)


warnings.filterwarnings("ignore")  # silence convergence warnings under benchmarks

SIZES = [
    ("tiny_50x5", 50, 5),
    ("small_1Kx10", 1_000, 10),
    ("medium_10Kx100", 10_000, 100),
]

KERNEL_SIZES = [
    ("tiny_50x5", 50, 5),
    ("small_500x10", 500, 10),
    ("medium_2Kx20", 2_000, 20),
]

CLUSTER_SIZES = [
    ("tiny_200x5", 200, 5),
    ("small_1Kx10", 1_000, 10),
    ("medium_5Kx20", 5_000, 20),
]


def median_time_micros(fn: Callable[[], None], iters: int = 7, slow: bool = False) -> float:
    """Median wall-clock time in microseconds."""
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


def regression_data(n: int, p: int):
    return make_regression(
        n_samples=n,
        n_features=p,
        n_informative=p,
        noise=0.1,
        random_state=42,
    )


def classification_data(n: int, p: int, n_classes: int = 2):
    return make_classification(
        n_samples=n,
        n_features=p,
        n_informative=max(2, p // 2),
        n_classes=n_classes,
        random_state=42,
    )


def cluster_data(n: int, p: int, n_centers: int = 8):
    return make_blobs(n_samples=n, n_features=p, centers=n_centers, random_state=42)


def split(x, y, regression=False):
    return train_test_split(x, y, test_size=0.2, random_state=42)


def relative_recon(x, x_hat):
    return float(np.linalg.norm(x - x_hat) / max(np.linalg.norm(x), 1e-30))


def make_record(family, algo, label, n, p, fit_us, predict_us, metric, score):
    return {
        "family": family,
        "algorithm": algo,
        "dataset": label,
        "n_samples": n,
        "n_features": p,
        "fit_us": fit_us,
        "predict_us": predict_us,
        "metric": metric,
        "score": float(score) if score is not None else None,
    }


def _safe(fn, *, default=None):
    """Run fn() and return its result, or `default` if it raises.

    sklearn occasionally fails on small/degenerate data (e.g. QDA without
    full-rank covariance). Skipping a single estimator is preferable to
    aborting the entire benchmark sweep.
    """
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        print(f"[sklearn-bench] WARN skipped: {exc}", file=sys.stderr)
        return default


def run_regressor(records, name, label, n, p, build):
    xtr, xte, ytr, yte = split(*regression_data(n, p))
    slow = n > 1_000

    fit_us = _safe(lambda: median_time_micros(lambda: build().fit(xtr, ytr), slow=slow))
    if fit_us is None:
        return
    fitted = _safe(lambda: build().fit(xtr, ytr))
    if fitted is None:
        return
    predict_us = _safe(lambda: median_time_micros(lambda: fitted.predict(xte)))
    score = _safe(lambda: r2_score(yte, fitted.predict(xte)))
    records.append(make_record("regressor", name, label, n, p, fit_us, predict_us, "r2", score))


def run_classifier(records, name, label, n, p, build, n_classes=2):
    xtr, xte, ytr, yte = split(*classification_data(n, p, n_classes=n_classes))
    slow = n > 1_000

    fit_us = _safe(lambda: median_time_micros(lambda: build().fit(xtr, ytr), slow=slow))
    if fit_us is None:
        return
    fitted = _safe(lambda: build().fit(xtr, ytr))
    if fitted is None:
        return
    predict_us = _safe(lambda: median_time_micros(lambda: fitted.predict(xte)))
    score = _safe(lambda: accuracy_score(yte, fitted.predict(xte)))
    records.append(make_record(
        "classifier", name, label, n, p, fit_us, predict_us, "accuracy", score
    ))


def run_classifier_pos(records, name, label, n, p, build):
    """Classifier on |X| (non-negative features)."""
    x, y = classification_data(n, p)
    xtr, xte, ytr, yte = train_test_split(np.abs(x), y, test_size=0.2, random_state=42)
    slow = n > 1_000

    fit_us = _safe(lambda: median_time_micros(lambda: build().fit(xtr, ytr), slow=slow))
    if fit_us is None:
        return
    fitted = _safe(lambda: build().fit(xtr, ytr))
    if fitted is None:
        return
    predict_us = _safe(lambda: median_time_micros(lambda: fitted.predict(xte)))
    score = _safe(lambda: accuracy_score(yte, fitted.predict(xte)))
    records.append(make_record(
        "classifier", name, label, n, p, fit_us, predict_us, "accuracy", score
    ))


def run_classifier_bin(records, name, label, n, p, build):
    """Classifier on binarized X."""
    x, y = classification_data(n, p)
    xb = (x > 0).astype(np.float64)
    xtr, xte, ytr, yte = train_test_split(xb, y, test_size=0.2, random_state=42)
    slow = n > 1_000

    fit_us = _safe(lambda: median_time_micros(lambda: build().fit(xtr, ytr), slow=slow))
    if fit_us is None:
        return
    fitted = _safe(lambda: build().fit(xtr, ytr))
    if fitted is None:
        return
    predict_us = _safe(lambda: median_time_micros(lambda: fitted.predict(xte)))
    score = _safe(lambda: accuracy_score(yte, fitted.predict(xte)))
    records.append(make_record(
        "classifier", name, label, n, p, fit_us, predict_us, "accuracy", score
    ))


# ---------------------------------------------------------------------------
# Family runners
# ---------------------------------------------------------------------------

def bench_regressors(records):
    for label, n, p in SIZES:
        run_regressor(records, "LinearRegression", label, n, p, lambda: LinearRegression())
        run_regressor(records, "Ridge", label, n, p, lambda: Ridge())
        run_regressor(records, "Lasso", label, n, p, lambda: Lasso())
        run_regressor(records, "ElasticNet", label, n, p, lambda: ElasticNet())
        run_regressor(records, "BayesianRidge", label, n, p, lambda: BayesianRidge())
        run_regressor(records, "ARDRegression", label, n, p, lambda: ARDRegression())
        if n <= 1_000:
            run_regressor(records, "HuberRegressor", label, n, p, lambda: HuberRegressor())
            run_regressor(records, "QuantileRegressor", label, n, p, lambda: QuantileRegressor())
        run_regressor(records, "DecisionTreeRegressor", label, n, p,
                      lambda: DecisionTreeRegressor(random_state=42))
        run_regressor(records, "RandomForestRegressor", label, n, p,
                      lambda: RandomForestRegressor(random_state=42, n_jobs=-1))
        run_regressor(records, "ExtraTreesRegressor", label, n, p,
                      lambda: ExtraTreesRegressor(random_state=42, n_jobs=-1))
        if n <= 1_000:
            run_regressor(records, "GradientBoostingRegressor", label, n, p,
                          lambda: GradientBoostingRegressor(random_state=42))
        run_regressor(records, "HistGradientBoostingRegressor", label, n, p,
                      lambda: HistGradientBoostingRegressor(random_state=42))
        run_regressor(records, "KNeighborsRegressor", label, n, p, lambda: KNeighborsRegressor())
        if n <= 2_000:
            run_regressor(records, "KernelRidge", label, n, p, lambda: KernelRidge())


def bench_classifiers(records):
    for label, n, p in SIZES:
        run_classifier(records, "LogisticRegression", label, n, p,
                       lambda: LogisticRegression(max_iter=200))
        run_classifier(records, "RidgeClassifier", label, n, p, lambda: RidgeClassifier())
        run_classifier(records, "LinearSVC", label, n, p, lambda: LinearSVC(max_iter=2000))
        run_classifier(records, "QDA", label, n, p, lambda: QuadraticDiscriminantAnalysis())
        run_classifier(records, "GaussianNB", label, n, p, lambda: GaussianNB())
        run_classifier(records, "DecisionTreeClassifier", label, n, p,
                       lambda: DecisionTreeClassifier(random_state=42))
        run_classifier(records, "ExtraTreeClassifier", label, n, p,
                       lambda: ExtraTreeClassifier(random_state=42))
        run_classifier(records, "RandomForestClassifier", label, n, p,
                       lambda: RandomForestClassifier(random_state=42, n_jobs=-1))
        run_classifier(records, "ExtraTreesClassifier", label, n, p,
                       lambda: ExtraTreesClassifier(random_state=42, n_jobs=-1))
        if n <= 1_000:
            run_classifier(records, "AdaBoostClassifier", label, n, p,
                           lambda: AdaBoostClassifier(random_state=42))
            run_classifier(records, "BaggingClassifier", label, n, p,
                           lambda: BaggingClassifier(random_state=42, n_jobs=-1))
            run_classifier(records, "GradientBoostingClassifier", label, n, p,
                           lambda: GradientBoostingClassifier(random_state=42))
        run_classifier(records, "HistGradientBoostingClassifier", label, n, p,
                       lambda: HistGradientBoostingClassifier(random_state=42))
        run_classifier(records, "KNeighborsClassifier", label, n, p,
                       lambda: KNeighborsClassifier(n_jobs=-1))
        run_classifier(records, "NearestCentroid", label, n, p, lambda: NearestCentroid())

        run_classifier_pos(records, "MultinomialNB", label, n, p, lambda: MultinomialNB())
        run_classifier_pos(records, "ComplementNB", label, n, p, lambda: ComplementNB())
        run_classifier_bin(records, "BernoulliNB", label, n, p, lambda: BernoulliNB())

    # Multi-class (5-class)
    label, n, p = "multiclass_2Kx20", 2_000, 20
    run_classifier(records, "LogisticRegression(5class)", label, n, p,
                   lambda: LogisticRegression(max_iter=200), n_classes=5)
    run_classifier(records, "RandomForestClassifier(5class)", label, n, p,
                   lambda: RandomForestClassifier(random_state=42, n_jobs=-1), n_classes=5)


def bench_clusterers(records):
    for label, n, p in CLUSTER_SIZES:
        x, y_true = cluster_data(n, p)

        def fit_predict_ari(name, build, slow=False):
            def run():
                fit_us = median_time_micros(lambda: build().fit(x), slow=slow)
                fitted = build().fit(x)
                labels = fitted.labels_
                ari = adjusted_rand_score(y_true, labels)
                records.append(make_record(
                    "cluster", name, label, n, p, fit_us, None, "ari", ari))
            _safe(run)

        slow = n > 1_000
        fit_predict_ari("KMeans",
                        lambda: KMeans(n_clusters=8, n_init=3, random_state=42),
                        slow=slow)
        fit_predict_ari("MiniBatchKMeans",
                        lambda: MiniBatchKMeans(n_clusters=8, random_state=42, n_init=3),
                        slow=slow)
        fit_predict_ari("BisectingKMeans",
                        lambda: BisectingKMeans(n_clusters=8, random_state=42),
                        slow=slow)

        # GaussianMixture (sklearn separates fit + predict; use predict for labels)
        def run_gm():
            def fit_gm():
                return GaussianMixture(n_components=8, random_state=42).fit(x)
            fit_us = median_time_micros(fit_gm, slow=slow)
            fitted = fit_gm()
            labels = fitted.predict(x)
            ari = adjusted_rand_score(y_true, labels)
            records.append(make_record(
                "cluster", "GaussianMixture", label, n, p, fit_us, None, "ari", ari))
        _safe(run_gm)

        if n <= 1_000:
            fit_predict_ari("AgglomerativeClustering",
                            lambda: AgglomerativeClustering(n_clusters=8), slow=True)
            fit_predict_ari("SpectralClustering",
                            lambda: SpectralClustering(n_clusters=8, random_state=42), slow=True)
            fit_predict_ari("DBSCAN", lambda: DBSCAN(eps=1.0), slow=True)
            fit_predict_ari("Birch", lambda: Birch(n_clusters=8), slow=True)
            fit_predict_ari("MeanShift", lambda: MeanShift(), slow=True)


def bench_decomp(records):
    for label, n, p in SIZES:
        x, _ = regression_data(n, p)
        # sklearn's TruncatedSVD requires n_components < n_features.
        n_comp = max(1, min(p - 1, 5))

        def reconstructable(name, build):
            def run():
                slow = n > 1_000
                fit_us = median_time_micros(lambda: build().fit(x), slow=slow)
                fitted = build().fit(x)
                predict_us = median_time_micros(lambda: fitted.transform(x))
                z = fitted.transform(x)
                try:
                    x_hat = fitted.inverse_transform(z)
                    err = relative_recon(x, x_hat)
                    records.append(make_record(
                        "decomp", name, label, n, p,
                        fit_us, predict_us, "recon_rel", err))
                except Exception:
                    records.append(make_record(
                        "decomp", name, label, n, p, fit_us, predict_us, None, None))
            _safe(run)

        reconstructable("PCA", lambda: PCA(n_components=n_comp))
        reconstructable("IncrementalPCA",
                        lambda: IncrementalPCA(n_components=n_comp))
        reconstructable("TruncatedSVD",
                        lambda: TruncatedSVD(n_components=n_comp, random_state=42))

        if n <= 1_000:
            reconstructable("FactorAnalysis",
                            lambda: FactorAnalysis(n_components=n_comp, random_state=42))

            def transform_only(name, build):
                def run():
                    fit_us = median_time_micros(lambda: build().fit(x))
                    fitted = build().fit(x)
                    predict_us = median_time_micros(lambda: fitted.transform(x))
                    records.append(make_record(
                        "decomp", name, label, n, p, fit_us, predict_us, None, None))
                _safe(run)
            transform_only("FastICA",
                           lambda: FastICA(n_components=n_comp, random_state=42, max_iter=200))
            transform_only("KernelPCA",
                           lambda: KernelPCA(n_components=n_comp, random_state=42))
            transform_only("SparsePCA",
                           lambda: SparsePCA(n_components=n_comp, random_state=42))

            def run_nmf():
                x_pos = np.abs(x)
                fit_us = median_time_micros(
                    lambda: NMF(n_components=n_comp, random_state=42, max_iter=200).fit(x_pos))
                fitted = NMF(n_components=n_comp, random_state=42, max_iter=200).fit(x_pos)
                predict_us = median_time_micros(lambda: fitted.transform(x_pos))
                records.append(make_record(
                    "decomp", "NMF", label, n, p, fit_us, predict_us, None, None))
            _safe(run_nmf)


def bench_preprocess(records):
    for label, n, p in SIZES:
        x, _ = regression_data(n, p)

        def state_xform(name, build):
            def run():
                fit_us = median_time_micros(lambda: build().fit(x))
                fitted = build().fit(x)
                predict_us = median_time_micros(lambda: fitted.transform(x))
                records.append(make_record(
                    "preprocess", name, label, n, p, fit_us, predict_us, None, None))
            _safe(run)

        state_xform("StandardScaler", lambda: StandardScaler())
        state_xform("MinMaxScaler", lambda: MinMaxScaler())
        state_xform("MaxAbsScaler", lambda: MaxAbsScaler())
        state_xform("RobustScaler", lambda: RobustScaler())
        if n <= 1_000:
            state_xform("PowerTransformer", lambda: PowerTransformer())
        state_xform("KBinsDiscretizer",
                    lambda: KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="uniform"))

        # Stateless
        def run_norm():
            normalizer = Normalizer(norm="l2")
            predict_us = median_time_micros(lambda: normalizer.transform(x))
            records.append(make_record(
                "preprocess", "Normalizer(L2)", label, n, p, 0.0, predict_us, None, None))
        _safe(run_norm)

        def run_poly():
            poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
            predict_us = median_time_micros(lambda: poly.fit_transform(x))
            records.append(make_record(
                "preprocess", "PolynomialFeatures(d=2)", label, n, p,
                0.0, predict_us, None, None))
        _safe(run_poly)


def bench_kernel_methods(records):
    for label, n, p in KERNEL_SIZES:
        xtr, xte, ytr, yte = split(*regression_data(n, p))
        slow = n > 500

        def run_kr():
            fit_us = median_time_micros(lambda: KernelRidge().fit(xtr, ytr), slow=slow)
            fitted = KernelRidge().fit(xtr, ytr)
            predict_us = median_time_micros(lambda: fitted.predict(xte))
            score = r2_score(yte, fitted.predict(xte))
            records.append(make_record(
                "kernel", "KernelRidge", label, n, p, fit_us, predict_us, "r2", score))
        _safe(run_kr)

        def run_nystroem():
            fit_us = median_time_micros(lambda: Nystroem(random_state=42).fit(xtr))
            fitted = Nystroem(random_state=42).fit(xtr)
            predict_us = median_time_micros(lambda: fitted.transform(xte))
            records.append(make_record(
                "kernel", "Nystroem", label, n, p, fit_us, predict_us, None, None))
        _safe(run_nystroem)

        def run_rbf():
            fit_us = median_time_micros(lambda: RBFSampler(random_state=42).fit(xtr))
            fitted = RBFSampler(random_state=42).fit(xtr)
            predict_us = median_time_micros(lambda: fitted.transform(xte))
            records.append(make_record(
                "kernel", "RBFSampler", label, n, p, fit_us, predict_us, None, None))
        _safe(run_rbf)


def bench_outlier(records):
    for label, n, p in SIZES[:2]:
        x, _ = classification_data(n, p)

        def run_if():
            fit_us = median_time_micros(lambda: IsolationForest(random_state=42).fit(x))
            fitted = IsolationForest(random_state=42).fit(x)
            predict_us = median_time_micros(lambda: fitted.predict(x))
            records.append(make_record(
                "outlier", "IsolationForest", label, n, p, fit_us, predict_us, None, None))
        _safe(run_if)


def main():
    records: list[dict] = []
    print("[sklearn-bench] regressors...", file=sys.stderr)
    bench_regressors(records)
    print("[sklearn-bench] classifiers...", file=sys.stderr)
    bench_classifiers(records)
    print("[sklearn-bench] clusterers...", file=sys.stderr)
    bench_clusterers(records)
    print("[sklearn-bench] decomp...", file=sys.stderr)
    bench_decomp(records)
    print("[sklearn-bench] preprocess...", file=sys.stderr)
    bench_preprocess(records)
    print("[sklearn-bench] kernel methods...", file=sys.stderr)
    bench_kernel_methods(records)
    print("[sklearn-bench] outlier...", file=sys.stderr)
    bench_outlier(records)

    print(json.dumps(records, indent=2))
    print(f"[sklearn-bench] {len(records)} records emitted", file=sys.stderr)


if __name__ == "__main__":
    main()
