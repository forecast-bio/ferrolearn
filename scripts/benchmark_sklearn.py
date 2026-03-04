#!/usr/bin/env python3
"""Benchmark sklearn models for comparison with ferrolearn criterion results.

Outputs JSON to stdout with median times in seconds for each (model, size, operation).
Run: python3 scripts/benchmark_sklearn.py
"""

import json
import sys
import timeit

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, make_classification, make_regression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
)

SIZES = [
    ("tiny_50x5", 50, 5),
    ("small_1Kx10", 1_000, 10),
    ("medium_10Kx100", 10_000, 100),
]

METRIC_SIZES = [
    ("1K", 1_000),
    ("10K", 10_000),
    ("100K", 100_000),
]


def time_fn(fn, number=None):
    """Return median time in seconds over multiple runs."""
    if number is None:
        # Auto-calibrate: run once to estimate, then pick a count
        t0 = timeit.timeit(fn, number=1)
        if t0 > 1.0:
            number = 3
        elif t0 > 0.1:
            number = 10
        elif t0 > 0.01:
            number = 50
        else:
            number = 200

    times = []
    for _ in range(number):
        t = timeit.timeit(fn, number=1)
        times.append(t)
    times.sort()
    return times[len(times) // 2]  # median


def bench_regressors():
    results = {}
    models = {
        "LinearRegression": lambda: LinearRegression(),
        "Ridge": lambda: Ridge(),
        "Lasso": lambda: Lasso(max_iter=1000),
        "ElasticNet": lambda: ElasticNet(max_iter=1000),
    }
    for name, make_model in models.items():
        for label, n, p in SIZES:
            X, y = make_regression(n_samples=n, n_features=p, n_informative=p, noise=0.1, random_state=42)
            model = make_model()
            fit_time = time_fn(lambda: make_model().fit(X, y))
            fitted = model.fit(X, y)
            predict_time = time_fn(lambda: fitted.predict(X))
            results[f"{name}/fit/{label}"] = fit_time
            results[f"{name}/predict/{label}"] = predict_time
            print(f"  {name:30s} {label:20s} fit={fit_time*1000:10.3f}ms  predict={predict_time*1000:10.3f}ms", file=sys.stderr)
    return results


def bench_classifiers():
    results = {}
    models = {
        "LogisticRegression": lambda: LogisticRegression(max_iter=1000),
        "DecisionTreeClassifier": lambda: DecisionTreeClassifier(),
        "RandomForestClassifier": lambda: RandomForestClassifier(n_estimators=100, random_state=42),
        "KNeighborsClassifier": lambda: KNeighborsClassifier(n_neighbors=5),
        "GaussianNB": lambda: GaussianNB(),
    }
    for name, make_model in models.items():
        for label, n, p in SIZES:
            X, y = make_classification(n_samples=n, n_features=p, n_informative=max(2, p // 2), random_state=42)
            model = make_model()
            fit_time = time_fn(lambda: make_model().fit(X, y))
            fitted = model.fit(X, y)
            predict_time = time_fn(lambda: fitted.predict(X))
            results[f"{name}/fit/{label}"] = fit_time
            results[f"{name}/predict/{label}"] = predict_time
            print(f"  {name:30s} {label:20s} fit={fit_time*1000:10.3f}ms  predict={predict_time*1000:10.3f}ms", file=sys.stderr)
    return results


def bench_transformers():
    results = {}
    for label, n, p in SIZES:
        X, _ = make_regression(n_samples=n, n_features=p, random_state=42)

        # StandardScaler
        fit_time = time_fn(lambda: StandardScaler().fit(X))
        fitted = StandardScaler().fit(X)
        transform_time = time_fn(lambda: fitted.transform(X))
        results[f"StandardScaler/fit/{label}"] = fit_time
        results[f"StandardScaler/transform/{label}"] = transform_time
        print(f"  {'StandardScaler':30s} {label:20s} fit={fit_time*1000:10.3f}ms  transform={transform_time*1000:10.3f}ms", file=sys.stderr)

        # PCA
        nc = min(p, 10)
        fit_time = time_fn(lambda: PCA(n_components=nc).fit(X))
        fitted = PCA(n_components=nc).fit(X)
        transform_time = time_fn(lambda: fitted.transform(X))
        results[f"PCA/fit/{label}"] = fit_time
        results[f"PCA/transform/{label}"] = transform_time
        print(f"  {'PCA':30s} {label:20s} fit={fit_time*1000:10.3f}ms  transform={transform_time*1000:10.3f}ms", file=sys.stderr)

    return results


def bench_clusterers():
    results = {}
    for label, n, p in SIZES:
        X, _ = make_blobs(n_samples=n, n_features=p, centers=8, random_state=42)
        fit_time = time_fn(lambda: KMeans(n_clusters=8, n_init=3, random_state=42).fit(X))
        fitted = KMeans(n_clusters=8, n_init=3, random_state=42).fit(X)
        predict_time = time_fn(lambda: fitted.predict(X))
        results[f"KMeans/fit/{label}"] = fit_time
        results[f"KMeans/predict/{label}"] = predict_time
        print(f"  {'KMeans':30s} {label:20s} fit={fit_time*1000:10.3f}ms  predict={predict_time*1000:10.3f}ms", file=sys.stderr)
    return results


def bench_metrics():
    results = {}
    for label, n in METRIC_SIZES:
        y_true_cls = np.array([i % 3 for i in range(n)])
        y_pred_cls = np.array([(i + 1) % 3 for i in range(n)])
        y_true_reg = np.arange(n, dtype=np.float64) * 0.1
        y_pred_reg = y_true_reg + 0.01

        t = time_fn(lambda: accuracy_score(y_true_cls, y_pred_cls))
        results[f"accuracy_score/{label}"] = t
        print(f"  {'accuracy_score':30s} {label:20s} {t*1000:10.3f}ms", file=sys.stderr)

        t = time_fn(lambda: f1_score(y_true_cls, y_pred_cls, average="macro"))
        results[f"f1_score/{label}"] = t
        print(f"  {'f1_score':30s} {label:20s} {t*1000:10.3f}ms", file=sys.stderr)

        t = time_fn(lambda: mean_squared_error(y_true_reg, y_pred_reg))
        results[f"mean_squared_error/{label}"] = t
        print(f"  {'mean_squared_error':30s} {label:20s} {t*1000:10.3f}ms", file=sys.stderr)

        t = time_fn(lambda: r2_score(y_true_reg, y_pred_reg))
        results[f"r2_score/{label}"] = t
        print(f"  {'r2_score':30s} {label:20s} {t*1000:10.3f}ms", file=sys.stderr)

    return results


def main():
    all_results = {}
    print("Benchmarking regressors...", file=sys.stderr)
    all_results.update(bench_regressors())
    print("Benchmarking classifiers...", file=sys.stderr)
    all_results.update(bench_classifiers())
    print("Benchmarking transformers...", file=sys.stderr)
    all_results.update(bench_transformers())
    print("Benchmarking clusterers...", file=sys.stderr)
    all_results.update(bench_clusterers())
    print("Benchmarking metrics...", file=sys.stderr)
    all_results.update(bench_metrics())

    json.dump(all_results, sys.stdout, indent=2)
    print(file=sys.stderr)
    print(f"Done. {len(all_results)} benchmarks recorded.", file=sys.stderr)


if __name__ == "__main__":
    main()
