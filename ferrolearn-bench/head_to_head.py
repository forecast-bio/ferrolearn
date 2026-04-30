#!/usr/bin/env python3
"""Head-to-head ferrolearn vs scikit-learn benchmark + accuracy harness.

A single Python process imports both `sklearn` and `ferrolearn`, generates one
canonical dataset per (algorithm, size), and runs fit + predict on both
libraries with identical hyperparameters, identical hold-out, identical
quality metric. Output: a JSON record per (algorithm, size) with sklearn and
ferrolearn measurements side by side.

Usage:
    python head_to_head.py > head_to_head.json
    python render_head_to_head.py head_to_head.json > REPORT.md
"""

from __future__ import annotations

import json
import math
import sys
import time
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
from sklearn.cluster import KMeans as SkKMeans
from sklearn.datasets import make_blobs, make_classification, make_regression
from sklearn.decomposition import PCA as SkPCA
from sklearn.ensemble import RandomForestClassifier as SkRandomForestClassifier
from sklearn.linear_model import (
    ElasticNet as SkElasticNet,
    Lasso as SkLasso,
    LinearRegression as SkLinearRegression,
    LogisticRegression as SkLogisticRegression,
    Ridge as SkRidge,
)
from sklearn.metrics import accuracy_score, adjusted_rand_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB as SkGaussianNB
from sklearn.neighbors import KNeighborsClassifier as SkKNeighborsClassifier
from sklearn.preprocessing import StandardScaler as SkStandardScaler
from sklearn.tree import DecisionTreeClassifier as SkDecisionTreeClassifier

import ferrolearn as fl


warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Dataset sizes (shared)
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


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def median_us(fn: Callable[[], Any], iters: int = 7, slow: bool = False) -> float:
    """Return median wall-clock time of fn() in microseconds."""
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


# ---------------------------------------------------------------------------
# Datasets — sklearn make_* is the canonical source
# ---------------------------------------------------------------------------

def regression_split(n: int, p: int, seed: int = 42):
    X, y = make_regression(n_samples=n, n_features=p, n_informative=p,
                           noise=0.1, random_state=seed)
    return train_test_split(X, y, test_size=0.2, random_state=seed)


def classification_split(n: int, p: int, seed: int = 42, n_classes: int = 2):
    X, y = make_classification(n_samples=n, n_features=p,
                               n_informative=max(2, p // 2),
                               n_classes=n_classes, random_state=seed)
    return train_test_split(X, y, test_size=0.2, random_state=seed)


def cluster_data(n: int, p: int, seed: int = 42, n_centers: int = 8):
    return make_blobs(n_samples=n, n_features=p, centers=n_centers,
                      random_state=seed)


# ---------------------------------------------------------------------------
# Record schema
# ---------------------------------------------------------------------------

@dataclass
class Side:
    fit_us: Optional[float] = None
    predict_us: Optional[float] = None
    score: Optional[float] = None
    notes: Optional[str] = None

    def asdict(self) -> dict:
        return {
            "fit_us": self.fit_us,
            "predict_us": self.predict_us,
            "score": float(self.score) if self.score is not None else None,
            "notes": self.notes,
        }


def record(family: str, algo: str, dataset: str, n: int, p: int,
           metric: str, sklearn: Side, ferrolearn: Side) -> dict:
    return {
        "family": family,
        "algorithm": algo,
        "dataset": dataset,
        "n_samples": n,
        "n_features": p,
        "metric": metric,
        "sklearn": sklearn.asdict(),
        "ferrolearn": ferrolearn.asdict(),
    }


def safe_run(fn: Callable[[], Any]) -> tuple[Any, Optional[str]]:
    """Return (result, None) on success, (None, error_message) on failure."""
    try:
        return fn(), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Generic head-to-head runners
# ---------------------------------------------------------------------------

def head_to_head_supervised(family: str, algo: str, label: str, n: int, p: int,
                            metric: str, score_fn: Callable,
                            xtr, xte, ytr, yte,
                            sk_factory: Callable, fl_factory: Callable,
                            slow: bool = False) -> dict:
    sk_side = Side()
    fl_side = Side()

    # sklearn
    res, err = safe_run(lambda: median_us(lambda: sk_factory().fit(xtr, ytr), slow=slow))
    if err:
        sk_side.notes = err
    else:
        sk_side.fit_us = res
        fitted, err = safe_run(lambda: sk_factory().fit(xtr, ytr))
        if err:
            sk_side.notes = err
        else:
            sk_side.predict_us, _ = safe_run(lambda: median_us(lambda: fitted.predict(xte)))
            yhat, err = safe_run(lambda: fitted.predict(xte))
            if not err:
                s, _ = safe_run(lambda: score_fn(yte, yhat))
                sk_side.score = s

    # ferrolearn
    res, err = safe_run(lambda: median_us(lambda: fl_factory().fit(xtr, ytr), slow=slow))
    if err:
        fl_side.notes = err
    else:
        fl_side.fit_us = res
        fitted, err = safe_run(lambda: fl_factory().fit(xtr, ytr))
        if err:
            fl_side.notes = err
        else:
            fl_side.predict_us, _ = safe_run(lambda: median_us(lambda: fitted.predict(xte)))
            yhat, err = safe_run(lambda: fitted.predict(xte))
            if not err:
                s, _ = safe_run(lambda: score_fn(yte, yhat))
                fl_side.score = s

    return record(family, algo, label, n, p, metric, sk_side, fl_side)


def head_to_head_cluster(algo: str, label: str, n: int, p: int,
                         x, y_true,
                         sk_factory: Callable, fl_factory: Callable,
                         slow: bool = False) -> dict:
    sk_side = Side()
    fl_side = Side()

    res, err = safe_run(lambda: median_us(lambda: sk_factory().fit(x), slow=slow))
    if err:
        sk_side.notes = err
    else:
        sk_side.fit_us = res
        fitted, err = safe_run(lambda: sk_factory().fit(x))
        if not err:
            labels = fitted.labels_
            s, _ = safe_run(lambda: adjusted_rand_score(y_true, labels))
            sk_side.score = s

    res, err = safe_run(lambda: median_us(lambda: fl_factory().fit(x), slow=slow))
    if err:
        fl_side.notes = err
    else:
        fl_side.fit_us = res
        fitted, err = safe_run(lambda: fl_factory().fit(x))
        if not err:
            labels = fitted.labels_
            s, _ = safe_run(lambda: adjusted_rand_score(y_true, labels))
            fl_side.score = s

    return record("cluster", algo, label, n, p, "ari", sk_side, fl_side)


def head_to_head_decomp(algo: str, label: str, n: int, p: int,
                        x,
                        sk_factory: Callable, fl_factory: Callable,
                        slow: bool = False) -> dict:
    """Reconstruction error: ||X - inverse_transform(transform(X))||_F / ||X||_F."""
    sk_side = Side()
    fl_side = Side()

    def relrec(fitted):
        z = fitted.transform(x)
        x_hat = fitted.inverse_transform(z)
        return float(np.linalg.norm(x - x_hat) / max(np.linalg.norm(x), 1e-30))

    res, err = safe_run(lambda: median_us(lambda: sk_factory().fit(x), slow=slow))
    if err:
        sk_side.notes = err
    else:
        sk_side.fit_us = res
        fitted, err = safe_run(lambda: sk_factory().fit(x))
        if not err:
            sk_side.predict_us, _ = safe_run(lambda: median_us(lambda: fitted.transform(x)))
            sk_side.score, _ = safe_run(lambda: relrec(fitted))

    res, err = safe_run(lambda: median_us(lambda: fl_factory().fit(x), slow=slow))
    if err:
        fl_side.notes = err
    else:
        fl_side.fit_us = res
        fitted, err = safe_run(lambda: fl_factory().fit(x))
        if not err:
            fl_side.predict_us, _ = safe_run(lambda: median_us(lambda: fitted.transform(x)))
            fl_side.score, _ = safe_run(lambda: relrec(fitted))

    return record("decomp", algo, label, n, p, "recon_rel", sk_side, fl_side)


def head_to_head_scaler(algo: str, label: str, n: int, p: int,
                        x,
                        sk_factory: Callable, fl_factory: Callable) -> dict:
    """Numerical agreement after fit_transform: ||sk - fl||_F / ||sk||_F."""
    sk_side = Side()
    fl_side = Side()

    res, err = safe_run(lambda: median_us(lambda: sk_factory().fit(x)))
    if err:
        sk_side.notes = err
    else:
        sk_side.fit_us = res
        fitted, err = safe_run(lambda: sk_factory().fit(x))
        if not err:
            sk_side.predict_us, _ = safe_run(lambda: median_us(lambda: fitted.transform(x)))
            sk_xt = fitted.transform(x)
        else:
            sk_xt = None

    res, err = safe_run(lambda: median_us(lambda: fl_factory().fit(x)))
    if err:
        fl_side.notes = err
    else:
        fl_side.fit_us = res
        fitted, err = safe_run(lambda: fl_factory().fit(x))
        if not err:
            fl_side.predict_us, _ = safe_run(lambda: median_us(lambda: fitted.transform(x)))
            fl_xt = fitted.transform(x)
        else:
            fl_xt = None

    if sk_xt is not None and fl_xt is not None:
        denom = max(float(np.linalg.norm(sk_xt)), 1e-30)
        rel = float(np.linalg.norm(sk_xt - fl_xt) / denom)
        sk_side.score = 0.0  # reference
        fl_side.score = rel  # divergence from sklearn

    return record("preprocess", algo, label, n, p, "rel_diff_vs_sklearn",
                  sk_side, fl_side)


# ---------------------------------------------------------------------------
# Family runners
# ---------------------------------------------------------------------------

def bench_regressors(records: list):
    for label, n, p in REGRESSION_SIZES:
        xtr, xte, ytr, yte = regression_split(n, p)
        slow = n > 1_000

        records.append(head_to_head_supervised(
            "regressor", "LinearRegression", label, n, p, "r2", r2_score,
            xtr, xte, ytr, yte,
            sk_factory=lambda: SkLinearRegression(),
            fl_factory=lambda: fl.LinearRegression(),
            slow=slow,
        ))
        records.append(head_to_head_supervised(
            "regressor", "Ridge", label, n, p, "r2", r2_score,
            xtr, xte, ytr, yte,
            sk_factory=lambda: SkRidge(alpha=1.0, fit_intercept=True),
            fl_factory=lambda: fl.Ridge(alpha=1.0, fit_intercept=True),
            slow=slow,
        ))
        records.append(head_to_head_supervised(
            "regressor", "Lasso", label, n, p, "r2", r2_score,
            xtr, xte, ytr, yte,
            sk_factory=lambda: SkLasso(alpha=1.0, max_iter=1000, tol=1e-4,
                                       fit_intercept=True),
            fl_factory=lambda: fl.Lasso(alpha=1.0, max_iter=1000, tol=1e-4,
                                        fit_intercept=True),
            slow=slow,
        ))
        records.append(head_to_head_supervised(
            "regressor", "ElasticNet", label, n, p, "r2", r2_score,
            xtr, xte, ytr, yte,
            sk_factory=lambda: SkElasticNet(alpha=1.0, l1_ratio=0.5,
                                            max_iter=1000, tol=1e-4,
                                            fit_intercept=True),
            fl_factory=lambda: fl.ElasticNet(alpha=1.0, l1_ratio=0.5,
                                             max_iter=1000, tol=1e-4,
                                             fit_intercept=True),
            slow=slow,
        ))


def bench_classifiers(records: list):
    for label, n, p in CLASSIFICATION_SIZES:
        xtr, xte, ytr, yte = classification_split(n, p)
        slow = n > 1_000

        records.append(head_to_head_supervised(
            "classifier", "LogisticRegression", label, n, p, "accuracy",
            accuracy_score, xtr, xte, ytr, yte,
            # match defaults: C=1.0, max_iter=100 in sklearn but our default is 1000.
            # Use 1000 on both sides for fair convergence.
            sk_factory=lambda: SkLogisticRegression(C=1.0, max_iter=1000, tol=1e-4,
                                                    fit_intercept=True),
            fl_factory=lambda: fl.LogisticRegression(C=1.0, max_iter=1000, tol=1e-4,
                                                     fit_intercept=True),
            slow=slow,
        ))
        records.append(head_to_head_supervised(
            "classifier", "DecisionTreeClassifier", label, n, p, "accuracy",
            accuracy_score, xtr, xte, ytr, yte,
            sk_factory=lambda: SkDecisionTreeClassifier(
                max_depth=None, min_samples_split=2, min_samples_leaf=1,
                random_state=42),
            fl_factory=lambda: fl.DecisionTreeClassifier(
                max_depth=None, min_samples_split=2, min_samples_leaf=1),
            slow=slow,
        ))
        records.append(head_to_head_supervised(
            "classifier", "RandomForestClassifier", label, n, p, "accuracy",
            accuracy_score, xtr, xte, ytr, yte,
            sk_factory=lambda: SkRandomForestClassifier(
                n_estimators=100, max_depth=None,
                min_samples_split=2, min_samples_leaf=1,
                random_state=42, n_jobs=-1),
            fl_factory=lambda: fl.RandomForestClassifier(
                n_estimators=100, max_depth=None,
                min_samples_split=2, min_samples_leaf=1,
                random_state=42),
            slow=slow,
        ))
        records.append(head_to_head_supervised(
            "classifier", "KNeighborsClassifier", label, n, p, "accuracy",
            accuracy_score, xtr, xte, ytr, yte,
            sk_factory=lambda: SkKNeighborsClassifier(n_neighbors=5),
            fl_factory=lambda: fl.KNeighborsClassifier(n_neighbors=5),
            slow=slow,
        ))
        records.append(head_to_head_supervised(
            "classifier", "GaussianNB", label, n, p, "accuracy",
            accuracy_score, xtr, xte, ytr, yte,
            sk_factory=lambda: SkGaussianNB(var_smoothing=1e-9),
            fl_factory=lambda: fl.GaussianNB(var_smoothing=1e-9),
            slow=slow,
        ))


def bench_clusterers(records: list):
    for label, n, p in CLUSTER_SIZES:
        x, y_true = cluster_data(n, p)
        slow = n > 1_000
        records.append(head_to_head_cluster(
            "KMeans", label, n, p, x, y_true,
            sk_factory=lambda: SkKMeans(n_clusters=8, n_init=10, max_iter=300,
                                        tol=1e-4, random_state=42),
            fl_factory=lambda: fl.KMeans(n_clusters=8, n_init=10, max_iter=300,
                                         tol=1e-4, random_state=42),
            slow=slow,
        ))


def bench_decomp(records: list):
    for label, n, p in REGRESSION_SIZES:
        x, _ = make_regression(n_samples=n, n_features=p, n_informative=p,
                               noise=0.1, random_state=42)
        # n_components < n_features (sklearn TruncatedSVD constraint, mirrored here)
        n_comp = max(1, min(p - 1, 5))
        slow = n > 1_000
        records.append(head_to_head_decomp(
            "PCA", label, n, p, x,
            sk_factory=lambda: SkPCA(n_components=n_comp, random_state=42),
            fl_factory=lambda: fl.PCA(n_components=n_comp),
            slow=slow,
        ))


def bench_preprocess(records: list):
    for label, n, p in REGRESSION_SIZES:
        x, _ = make_regression(n_samples=n, n_features=p, n_informative=p,
                               noise=0.1, random_state=42)
        records.append(head_to_head_scaler(
            "StandardScaler", label, n, p, x,
            sk_factory=lambda: SkStandardScaler(with_mean=True, with_std=True),
            fl_factory=lambda: fl.StandardScaler(with_mean=True, with_std=True),
        ))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import sklearn
    print(f"# ferrolearn vs scikit-learn — head-to-head", file=sys.stderr)
    print(f"# sklearn={sklearn.__version__}, ferrolearn=0.3.0, "
          f"numpy={np.__version__}, python={sys.version.split()[0]}",
          file=sys.stderr)

    records: list = []
    print("[h2h] regressors...", file=sys.stderr)
    bench_regressors(records)
    print("[h2h] classifiers...", file=sys.stderr)
    bench_classifiers(records)
    print("[h2h] clusterers...", file=sys.stderr)
    bench_clusterers(records)
    print("[h2h] decomp...", file=sys.stderr)
    bench_decomp(records)
    print("[h2h] preprocess...", file=sys.stderr)
    bench_preprocess(records)

    print(json.dumps(records, indent=2))
    print(f"[h2h] {len(records)} records emitted", file=sys.stderr)


if __name__ == "__main__":
    main()
