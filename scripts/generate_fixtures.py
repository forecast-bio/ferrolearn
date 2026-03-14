#!/usr/bin/env python3
"""Generate golden test fixtures from scikit-learn for ferrolearn oracle tests.

Run this script to regenerate fixtures after scikit-learn upgrades or when
adding new test cases. Output JSON files are written to the fixtures/ directory
relative to the project root.

Usage:
    python scripts/generate_fixtures.py
"""

import json
import os
import sys

import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression, ElasticNet
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    MaxAbsScaler, Normalizer, Binarizer,
    PolynomialFeatures, OneHotEncoder, LabelEncoder,
    QuantileTransformer, KBinsDiscretizer, PowerTransformer,
)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
    roc_auc_score,
    log_loss,
    adjusted_rand_score,
    adjusted_mutual_info_score,
    davies_bouldin_score,
    mean_absolute_percentage_error,
    explained_variance_score,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    AdaBoostClassifier,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB
from sklearn.cluster import (
    KMeans, DBSCAN, AgglomerativeClustering,
    MiniBatchKMeans, MeanShift, SpectralClustering,
    OPTICS, Birch,
)
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, NMF, TruncatedSVD
from sklearn.datasets import load_iris, load_diabetes

SKLEARN_VERSION = sklearn.__version__

# Output directory is fixtures/ relative to this script's parent (project root)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
FIXTURES_DIR = os.path.join(PROJECT_ROOT, "fixtures")

os.makedirs(FIXTURES_DIR, exist_ok=True)


def to_list(arr):
    """Convert a numpy array to a plain Python list (nested for 2-D)."""
    if isinstance(arr, np.ndarray):
        return arr.tolist()
    return arr


def write_fixture(name: str, data: dict) -> None:
    path = os.path.join(FIXTURES_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# 1. Linear Regression
# ---------------------------------------------------------------------------
def gen_linear_regression():
    rng = np.random.default_rng(42)
    n, p = 50, 5
    X = rng.standard_normal((n, p))
    true_coef = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    noise = rng.standard_normal(n) * 0.1
    y = X @ true_coef + noise

    model = LinearRegression(fit_intercept=True)
    model.fit(X, y)
    preds = model.predict(X)

    write_fixture(
        "linear_regression",
        {
            "description": "LinearRegression fixture from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": 42,
            "input": {"X": to_list(X), "y": to_list(y)},
            "params": {"fit_intercept": True},
            "expected": {
                "coefficients": to_list(model.coef_),
                "intercept": float(model.intercept_),
                "predictions": to_list(preds),
            },
        },
    )


# ---------------------------------------------------------------------------
# 2. Ridge Regression
# ---------------------------------------------------------------------------
def gen_ridge():
    rng = np.random.default_rng(42)
    n, p = 50, 5
    X = rng.standard_normal((n, p))
    true_coef = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    noise = rng.standard_normal(n) * 0.1
    y = X @ true_coef + noise

    model = Ridge(alpha=1.0, fit_intercept=True, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)

    write_fixture(
        "ridge",
        {
            "description": "Ridge regression fixture from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": 42,
            "input": {"X": to_list(X), "y": to_list(y)},
            "params": {"alpha": 1.0, "fit_intercept": True},
            "expected": {
                "coefficients": to_list(model.coef_),
                "intercept": float(model.intercept_),
                "predictions": to_list(preds),
            },
        },
    )


# ---------------------------------------------------------------------------
# 3. Lasso
# ---------------------------------------------------------------------------
def gen_lasso():
    rng = np.random.default_rng(42)
    n, p = 50, 5
    X = rng.standard_normal((n, p))
    true_coef = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    noise = rng.standard_normal(n) * 0.1
    y = X @ true_coef + noise

    model = Lasso(alpha=0.1, fit_intercept=True, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)

    write_fixture(
        "lasso",
        {
            "description": "Lasso regression fixture from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": 42,
            "input": {"X": to_list(X), "y": to_list(y)},
            "params": {"alpha": 0.1, "fit_intercept": True},
            "expected": {
                "coefficients": to_list(model.coef_),
                "intercept": float(model.intercept_),
                "predictions": to_list(preds),
            },
        },
    )


# ---------------------------------------------------------------------------
# 4. Logistic Regression  (binary, iris-like 100×4)
# ---------------------------------------------------------------------------
def gen_logistic_regression():
    rng = np.random.default_rng(42)
    n, p = 100, 4
    # Two separable blobs in 4-D
    X0 = rng.standard_normal((n // 2, p)) + np.array([-1.0, -1.0, -1.0, -1.0])
    X1 = rng.standard_normal((n // 2, p)) + np.array([1.0, 1.0, 1.0, 1.0])
    X = np.vstack([X0, X1])
    y = np.array([0] * (n // 2) + [1] * (n // 2))

    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200, random_state=42)
    model.fit(X, y)
    pred_classes = model.predict(X)
    pred_proba = model.predict_proba(X)

    write_fixture(
        "logistic_regression",
        {
            "description": "LogisticRegression binary fixture from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": 42,
            "input": {"X": to_list(X), "y": to_list(y)},
            "params": {"C": 1.0, "solver": "lbfgs", "fit_intercept": True},
            "expected": {
                "coefficients": to_list(model.coef_),
                "intercept": to_list(model.intercept_),
                "predicted_classes": to_list(pred_classes),
                "predicted_proba": to_list(pred_proba),
            },
        },
    )


# ---------------------------------------------------------------------------
# 5. StandardScaler
# ---------------------------------------------------------------------------
def gen_standard_scaler():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((30, 4)) * 5 + 3  # non-zero mean/std

    scaler = StandardScaler()
    X_out = scaler.fit_transform(X)

    write_fixture(
        "standard_scaler",
        {
            "description": "StandardScaler fixture from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": 42,
            "input": {"X": to_list(X)},
            "params": {"with_mean": True, "with_std": True},
            "expected": {
                "mean": to_list(scaler.mean_),
                "std": to_list(scaler.scale_),
                "transformed": to_list(X_out),
            },
        },
    )


# ---------------------------------------------------------------------------
# 6. MinMaxScaler
# ---------------------------------------------------------------------------
def gen_minmax_scaler():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((30, 4)) * 5 + 3

    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    X_out = scaler.fit_transform(X)

    write_fixture(
        "minmax_scaler",
        {
            "description": "MinMaxScaler fixture from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": 42,
            "input": {"X": to_list(X)},
            "params": {"feature_range": [0.0, 1.0]},
            "expected": {
                "data_min": to_list(scaler.data_min_),
                "data_max": to_list(scaler.data_max_),
                "transformed": to_list(X_out),
            },
        },
    )


# ---------------------------------------------------------------------------
# 7. RobustScaler
# ---------------------------------------------------------------------------
def gen_robust_scaler():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((30, 4)) * 5 + 3

    scaler = RobustScaler()
    X_out = scaler.fit_transform(X)

    write_fixture(
        "robust_scaler",
        {
            "description": "RobustScaler fixture from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": 42,
            "input": {"X": to_list(X)},
            "params": {"with_centering": True, "with_scaling": True},
            "expected": {
                "center": to_list(scaler.center_),
                "scale": to_list(scaler.scale_),
                "transformed": to_list(X_out),
            },
        },
    )


# ---------------------------------------------------------------------------
# 8. Classification metrics
# ---------------------------------------------------------------------------
def gen_classification_metrics():
    rng = np.random.default_rng(42)
    n = 100
    y_true = rng.integers(0, 2, size=n)
    # Introduce ~10% errors for non-trivial metrics
    flip_mask = rng.random(n) < 0.10
    y_pred = y_true.copy()
    y_pred[flip_mask] = 1 - y_pred[flip_mask]

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    write_fixture(
        "classification_metrics",
        {
            "description": "Binary classification metrics fixture from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": 42,
            "input": {"y_true": to_list(y_true), "y_pred": to_list(y_pred)},
            "params": {"average": "binary"},
            "expected": {
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "confusion_matrix": to_list(cm),
            },
        },
    )


# ---------------------------------------------------------------------------
# 9. Regression metrics
# ---------------------------------------------------------------------------
def gen_regression_metrics():
    rng = np.random.default_rng(42)
    n = 100
    y_true = rng.standard_normal(n) * 10
    noise = rng.standard_normal(n) * 2
    y_pred = y_true + noise

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)

    write_fixture(
        "regression_metrics",
        {
            "description": "Regression metrics fixture from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": 42,
            "input": {"y_true": to_list(y_true), "y_pred": to_list(y_pred)},
            "params": {},
            "expected": {
                "mae": float(mae),
                "mse": float(mse),
                "rmse": rmse,
                "r2": float(r2),
            },
        },
    )


# ---------------------------------------------------------------------------
# 10. KFold cross-validation indices
# ---------------------------------------------------------------------------
def gen_kfold():
    n_samples = 100
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=False)

    folds = []
    indices = np.arange(n_samples)
    for train_idx, test_idx in kf.split(indices):
        folds.append(
            {
                "train": to_list(train_idx),
                "test": to_list(test_idx),
            }
        )

    write_fixture(
        "kfold",
        {
            "description": "KFold cross-validation index fixture from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": None,
            "input": {"n_samples": n_samples},
            "params": {"n_splits": n_splits, "shuffle": False},
            "expected": {"folds": folds},
        },
    )


# ---------------------------------------------------------------------------
# 11. Decision Tree Classifier
# ---------------------------------------------------------------------------
def gen_decision_tree_classifier():
    X, y = load_iris(return_X_y=True)
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)
    importances = model.feature_importances_

    write_fixture(
        "decision_tree_classifier",
        {
            "description": "DecisionTreeClassifier on iris from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": 42,
            "input": {"X": to_list(X), "y": to_list(y)},
            "params": {"max_depth": 3, "random_state": 42},
            "expected": {
                "predictions": to_list(preds),
                "feature_importances": to_list(importances),
                "accuracy": float(accuracy_score(y, preds)),
            },
        },
    )


# ---------------------------------------------------------------------------
# 12. Decision Tree Regressor
# ---------------------------------------------------------------------------
def gen_decision_tree_regressor():
    X, y = load_diabetes(return_X_y=True)
    X_small, y_small = X[:100], y[:100]
    model = DecisionTreeRegressor(max_depth=4, random_state=42)
    model.fit(X_small, y_small)
    preds = model.predict(X_small)

    write_fixture(
        "decision_tree_regressor",
        {
            "description": "DecisionTreeRegressor on diabetes (first 100) from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": 42,
            "input": {"X": to_list(X_small), "y": to_list(y_small)},
            "params": {"max_depth": 4, "random_state": 42},
            "expected": {
                "predictions": to_list(preds),
                "feature_importances": to_list(model.feature_importances_),
                "r2": float(r2_score(y_small, preds)),
            },
        },
    )


# ---------------------------------------------------------------------------
# 13. Random Forest Classifier
# ---------------------------------------------------------------------------
def gen_random_forest_classifier():
    X, y = load_iris(return_X_y=True)
    model = RandomForestClassifier(
        n_estimators=20, max_depth=4, random_state=42, n_jobs=1
    )
    model.fit(X, y)
    preds = model.predict(X)
    importances = model.feature_importances_

    write_fixture(
        "random_forest_classifier",
        {
            "description": "RandomForestClassifier on iris from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": 42,
            "input": {"X": to_list(X), "y": to_list(y)},
            "params": {"n_estimators": 20, "max_depth": 4, "random_state": 42},
            "expected": {
                "predictions": to_list(preds),
                "feature_importances": to_list(importances),
                "accuracy": float(accuracy_score(y, preds)),
            },
        },
    )


# ---------------------------------------------------------------------------
# 14. Random Forest Regressor
# ---------------------------------------------------------------------------
def gen_random_forest_regressor():
    X, y = load_diabetes(return_X_y=True)
    X_small, y_small = X[:100], y[:100]
    model = RandomForestRegressor(
        n_estimators=20, max_depth=4, random_state=42, n_jobs=1
    )
    model.fit(X_small, y_small)
    preds = model.predict(X_small)

    write_fixture(
        "random_forest_regressor",
        {
            "description": "RandomForestRegressor on diabetes (first 100) from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": 42,
            "input": {"X": to_list(X_small), "y": to_list(y_small)},
            "params": {"n_estimators": 20, "max_depth": 4, "random_state": 42},
            "expected": {
                "predictions": to_list(preds),
                "feature_importances": to_list(model.feature_importances_),
                "r2": float(r2_score(y_small, preds)),
            },
        },
    )


# ---------------------------------------------------------------------------
# 15. Gradient Boosting Classifier
# ---------------------------------------------------------------------------
def gen_gradient_boosting_classifier():
    X, y = load_iris(return_X_y=True)
    model = GradientBoostingClassifier(
        n_estimators=50, max_depth=2, learning_rate=0.1, random_state=42
    )
    model.fit(X, y)
    preds = model.predict(X)

    write_fixture(
        "gradient_boosting_classifier",
        {
            "description": "GradientBoostingClassifier on iris from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": 42,
            "input": {"X": to_list(X), "y": to_list(y)},
            "params": {
                "n_estimators": 50,
                "max_depth": 2,
                "learning_rate": 0.1,
                "random_state": 42,
            },
            "expected": {
                "predictions": to_list(preds),
                "feature_importances": to_list(model.feature_importances_),
                "accuracy": float(accuracy_score(y, preds)),
            },
        },
    )


# ---------------------------------------------------------------------------
# 16. Gradient Boosting Regressor
# ---------------------------------------------------------------------------
def gen_gradient_boosting_regressor():
    X, y = load_diabetes(return_X_y=True)
    X_small, y_small = X[:100], y[:100]
    model = GradientBoostingRegressor(
        n_estimators=50, max_depth=2, learning_rate=0.1, random_state=42
    )
    model.fit(X_small, y_small)
    preds = model.predict(X_small)

    write_fixture(
        "gradient_boosting_regressor",
        {
            "description": "GradientBoostingRegressor on diabetes (first 100) from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": 42,
            "input": {"X": to_list(X_small), "y": to_list(y_small)},
            "params": {
                "n_estimators": 50,
                "max_depth": 2,
                "learning_rate": 0.1,
                "random_state": 42,
            },
            "expected": {
                "predictions": to_list(preds),
                "r2": float(r2_score(y_small, preds)),
            },
        },
    )


# ---------------------------------------------------------------------------
# 17. AdaBoost Classifier
# ---------------------------------------------------------------------------
def gen_adaboost_classifier():
    X, y = load_iris(return_X_y=True)
    model = AdaBoostClassifier(
        n_estimators=50, learning_rate=1.0, random_state=42, algorithm="SAMME"
    )
    model.fit(X, y)
    preds = model.predict(X)

    write_fixture(
        "adaboost_classifier",
        {
            "description": "AdaBoostClassifier (SAMME) on iris from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": 42,
            "input": {"X": to_list(X), "y": to_list(y)},
            "params": {
                "n_estimators": 50,
                "learning_rate": 1.0,
                "algorithm": "SAMME",
                "random_state": 42,
            },
            "expected": {
                "predictions": to_list(preds),
                "accuracy": float(accuracy_score(y, preds)),
            },
        },
    )


# ---------------------------------------------------------------------------
# 18. KNeighbors Classifier
# ---------------------------------------------------------------------------
def gen_kneighbors_classifier():
    X, y = load_iris(return_X_y=True)
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X, y)
    preds = model.predict(X)

    write_fixture(
        "kneighbors_classifier",
        {
            "description": "KNeighborsClassifier on iris from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": None,
            "input": {"X": to_list(X), "y": to_list(y)},
            "params": {"n_neighbors": 5},
            "expected": {
                "predictions": to_list(preds),
                "accuracy": float(accuracy_score(y, preds)),
            },
        },
    )


# ---------------------------------------------------------------------------
# 19. KNeighbors Regressor
# ---------------------------------------------------------------------------
def gen_kneighbors_regressor():
    X, y = load_diabetes(return_X_y=True)
    X_small, y_small = X[:100], y[:100]
    model = KNeighborsRegressor(n_neighbors=5)
    model.fit(X_small, y_small)
    preds = model.predict(X_small)

    write_fixture(
        "kneighbors_regressor",
        {
            "description": "KNeighborsRegressor on diabetes (first 100) from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": None,
            "input": {"X": to_list(X_small), "y": to_list(y_small)},
            "params": {"n_neighbors": 5},
            "expected": {
                "predictions": to_list(preds),
                "r2": float(r2_score(y_small, preds)),
            },
        },
    )


# ---------------------------------------------------------------------------
# 20. Gaussian Naive Bayes
# ---------------------------------------------------------------------------
def gen_gaussian_nb():
    X, y = load_iris(return_X_y=True)
    model = GaussianNB()
    model.fit(X, y)
    preds = model.predict(X)

    write_fixture(
        "gaussian_nb",
        {
            "description": "GaussianNB on iris from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": None,
            "input": {"X": to_list(X), "y": to_list(y)},
            "params": {},
            "expected": {
                "predictions": to_list(preds),
                "accuracy": float(accuracy_score(y, preds)),
                "class_prior": to_list(model.class_prior_),
                "theta": to_list(model.theta_),
                "var": to_list(model.var_),
            },
        },
    )


# ---------------------------------------------------------------------------
# 21. KMeans
# ---------------------------------------------------------------------------
def gen_kmeans():
    rng = np.random.default_rng(42)
    # 3 well-separated clusters
    c0 = rng.standard_normal((50, 2)) + np.array([-5.0, -5.0])
    c1 = rng.standard_normal((50, 2)) + np.array([5.0, -5.0])
    c2 = rng.standard_normal((50, 2)) + np.array([0.0, 5.0])
    X = np.vstack([c0, c1, c2])

    model = KMeans(n_clusters=3, random_state=42, n_init=10, max_iter=300)
    model.fit(X)
    labels = model.labels_
    centers = model.cluster_centers_
    inertia = model.inertia_

    write_fixture(
        "kmeans",
        {
            "description": "KMeans (k=3) on synthetic blobs from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": 42,
            "input": {"X": to_list(X)},
            "params": {"n_clusters": 3, "random_state": 42, "n_init": 10},
            "expected": {
                "labels": to_list(labels),
                "cluster_centers": to_list(centers),
                "inertia": float(inertia),
                "n_iter": int(model.n_iter_),
            },
        },
    )


# ---------------------------------------------------------------------------
# 22. DBSCAN
# ---------------------------------------------------------------------------
def gen_dbscan():
    rng = np.random.default_rng(42)
    c0 = rng.standard_normal((30, 2)) + np.array([-5.0, 0.0])
    c1 = rng.standard_normal((30, 2)) + np.array([5.0, 0.0])
    noise = rng.uniform(-10, 10, size=(5, 2))
    X = np.vstack([c0, c1, noise])

    model = DBSCAN(eps=1.5, min_samples=5)
    labels = model.fit_predict(X)
    core_indices = model.core_sample_indices_

    write_fixture(
        "dbscan",
        {
            "description": "DBSCAN on synthetic data from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": 42,
            "input": {"X": to_list(X)},
            "params": {"eps": 1.5, "min_samples": 5},
            "expected": {
                "labels": to_list(labels),
                "core_sample_indices": to_list(core_indices),
                "n_clusters": int(len(set(labels)) - (1 if -1 in labels else 0)),
                "n_noise": int(np.sum(labels == -1)),
            },
        },
    )


# ---------------------------------------------------------------------------
# 23. Agglomerative Clustering
# ---------------------------------------------------------------------------
def gen_agglomerative():
    rng = np.random.default_rng(42)
    c0 = rng.standard_normal((30, 2)) + np.array([-4.0, 0.0])
    c1 = rng.standard_normal((30, 2)) + np.array([4.0, 0.0])
    c2 = rng.standard_normal((30, 2)) + np.array([0.0, 6.0])
    X = np.vstack([c0, c1, c2])

    model = AgglomerativeClustering(n_clusters=3, linkage="ward")
    labels = model.fit_predict(X)

    write_fixture(
        "agglomerative_clustering",
        {
            "description": "AgglomerativeClustering (ward, k=3) from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": 42,
            "input": {"X": to_list(X)},
            "params": {"n_clusters": 3, "linkage": "ward"},
            "expected": {
                "labels": to_list(labels),
                "n_clusters": 3,
            },
        },
    )


# ---------------------------------------------------------------------------
# 24. PCA
# ---------------------------------------------------------------------------
def gen_pca():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 5))

    model = PCA(n_components=3)
    X_out = model.fit_transform(X)

    write_fixture(
        "pca",
        {
            "description": "PCA (3 components) from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": 42,
            "input": {"X": to_list(X)},
            "params": {"n_components": 3},
            "expected": {
                "components": to_list(model.components_),
                "explained_variance": to_list(model.explained_variance_),
                "explained_variance_ratio": to_list(model.explained_variance_ratio_),
                "mean": to_list(model.mean_),
                "transformed": to_list(X_out),
            },
        },
    )


# ---------------------------------------------------------------------------
# 25. NMF
# ---------------------------------------------------------------------------
def gen_nmf():
    rng = np.random.default_rng(42)
    X = np.abs(rng.standard_normal((40, 6)))  # NMF requires non-negative

    model = NMF(n_components=3, init="nndsvd", random_state=42, max_iter=500)
    W = model.fit_transform(X)
    H = model.components_

    write_fixture(
        "nmf",
        {
            "description": "NMF (3 components, nndsvd) from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": 42,
            "input": {"X": to_list(X)},
            "params": {"n_components": 3, "init": "nndsvd", "random_state": 42},
            "expected": {
                "W": to_list(W),
                "H": to_list(H),
                "reconstruction_error": float(model.reconstruction_err_),
            },
        },
    )


# ---------------------------------------------------------------------------
# 26. ElasticNet
# ---------------------------------------------------------------------------
def gen_elastic_net():
    rng = np.random.default_rng(42)
    n, p = 50, 5
    X = rng.standard_normal((n, p))
    true_coef = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    noise = rng.standard_normal(n) * 0.1
    y = X @ true_coef + noise

    model = ElasticNet(alpha=0.1, l1_ratio=0.5, fit_intercept=True, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)

    write_fixture(
        "elastic_net",
        {
            "description": "ElasticNet fixture from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": 42,
            "input": {"X": to_list(X), "y": to_list(y)},
            "params": {"alpha": 0.1, "l1_ratio": 0.5, "fit_intercept": True},
            "expected": {
                "coefficients": to_list(model.coef_),
                "intercept": float(model.intercept_),
                "predictions": to_list(preds),
            },
        },
    )


# ===========================================================================
# NEW FIXTURES — Naive Bayes
# ===========================================================================


# ---------------------------------------------------------------------------
# 27. Multinomial Naive Bayes
# ---------------------------------------------------------------------------
def gen_multinomial_nb():
    rng = np.random.default_rng(42)
    n, p = 100, 6
    X = rng.integers(0, 10, size=(n, p)).astype(float)
    y = rng.integers(0, 3, size=n)

    model = MultinomialNB(alpha=1.0)
    model.fit(X, y)
    preds = model.predict(X)

    write_fixture(
        "multinomial_nb",
        {
            "description": "MultinomialNB on synthetic count data from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": 42,
            "input": {"X": to_list(X), "y": to_list(y)},
            "params": {"alpha": 1.0},
            "expected": {
                "predictions": to_list(preds),
                "accuracy": float(accuracy_score(y, preds)),
                "class_log_prior": to_list(model.class_log_prior_),
                "feature_log_prob": to_list(model.feature_log_prob_),
            },
        },
    )


# ---------------------------------------------------------------------------
# 28. Bernoulli Naive Bayes
# ---------------------------------------------------------------------------
def gen_bernoulli_nb():
    rng = np.random.default_rng(42)
    n, p = 100, 8
    X = (rng.random((n, p)) > 0.5).astype(float)
    y = rng.integers(0, 2, size=n)

    model = BernoulliNB(alpha=1.0)
    model.fit(X, y)
    preds = model.predict(X)

    write_fixture(
        "bernoulli_nb",
        {
            "description": "BernoulliNB on synthetic binary data from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": 42,
            "input": {"X": to_list(X), "y": to_list(y)},
            "params": {"alpha": 1.0},
            "expected": {
                "predictions": to_list(preds),
                "accuracy": float(accuracy_score(y, preds)),
                "class_log_prior": to_list(model.class_log_prior_),
                "feature_log_prob": to_list(model.feature_log_prob_),
            },
        },
    )


# ---------------------------------------------------------------------------
# 29. Complement Naive Bayes
# ---------------------------------------------------------------------------
def gen_complement_nb():
    rng = np.random.default_rng(42)
    n, p = 100, 6
    X = rng.integers(0, 10, size=(n, p)).astype(float)
    y = rng.integers(0, 3, size=n)

    model = ComplementNB(alpha=1.0)
    model.fit(X, y)
    preds = model.predict(X)

    write_fixture(
        "complement_nb",
        {
            "description": "ComplementNB on synthetic count data from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": 42,
            "input": {"X": to_list(X), "y": to_list(y)},
            "params": {"alpha": 1.0},
            "expected": {
                "predictions": to_list(preds),
                "accuracy": float(accuracy_score(y, preds)),
                "class_log_prior": to_list(model.class_log_prior_),
                "feature_log_prob": to_list(model.feature_log_prob_),
            },
        },
    )


# ===========================================================================
# NEW FIXTURES — Clustering
# ===========================================================================


# ---------------------------------------------------------------------------
# 30. MiniBatchKMeans
# ---------------------------------------------------------------------------
def gen_mini_batch_kmeans():
    rng = np.random.default_rng(42)
    c0 = rng.standard_normal((50, 2)) + np.array([-5.0, -5.0])
    c1 = rng.standard_normal((50, 2)) + np.array([5.0, -5.0])
    c2 = rng.standard_normal((50, 2)) + np.array([0.0, 5.0])
    X = np.vstack([c0, c1, c2])

    model = MiniBatchKMeans(n_clusters=3, random_state=42, batch_size=50, n_init=10)
    model.fit(X)

    write_fixture(
        "mini_batch_kmeans",
        {
            "description": "MiniBatchKMeans (k=3) on synthetic blobs from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": 42,
            "input": {"X": to_list(X)},
            "params": {"n_clusters": 3, "random_state": 42, "batch_size": 50},
            "expected": {
                "labels": to_list(model.labels_),
                "cluster_centers": to_list(model.cluster_centers_),
                "inertia": float(model.inertia_),
            },
        },
    )


# ---------------------------------------------------------------------------
# 31. MeanShift
# ---------------------------------------------------------------------------
def gen_mean_shift():
    rng = np.random.default_rng(42)
    c0 = rng.standard_normal((50, 2)) + np.array([-5.0, -5.0])
    c1 = rng.standard_normal((50, 2)) + np.array([5.0, -5.0])
    c2 = rng.standard_normal((50, 2)) + np.array([0.0, 5.0])
    X = np.vstack([c0, c1, c2])

    model = MeanShift(bandwidth=2.0)
    model.fit(X)

    write_fixture(
        "mean_shift",
        {
            "description": "MeanShift on synthetic blobs from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": 42,
            "input": {"X": to_list(X)},
            "params": {"bandwidth": 2.0},
            "expected": {
                "labels": to_list(model.labels_),
                "cluster_centers": to_list(model.cluster_centers_),
                "n_clusters": int(len(np.unique(model.labels_))),
            },
        },
    )


# ---------------------------------------------------------------------------
# 32. GaussianMixture
# ---------------------------------------------------------------------------
def gen_gaussian_mixture():
    rng = np.random.default_rng(42)
    c0 = rng.standard_normal((50, 2)) + np.array([-5.0, -5.0])
    c1 = rng.standard_normal((50, 2)) + np.array([5.0, -5.0])
    c2 = rng.standard_normal((50, 2)) + np.array([0.0, 5.0])
    X = np.vstack([c0, c1, c2])

    model = GaussianMixture(n_components=3, random_state=42, max_iter=200)
    model.fit(X)
    preds = model.predict(X)

    write_fixture(
        "gaussian_mixture",
        {
            "description": "GaussianMixture (k=3) on synthetic blobs from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": 42,
            "input": {"X": to_list(X)},
            "params": {"n_components": 3, "random_state": 42},
            "expected": {
                "labels": to_list(preds),
                "means": to_list(model.means_),
                "weights": to_list(model.weights_),
                "covariances": to_list(model.covariances_),
                "n_clusters": int(len(np.unique(preds))),
            },
        },
    )


# ---------------------------------------------------------------------------
# 33. OPTICS
# ---------------------------------------------------------------------------
def gen_optics():
    rng = np.random.default_rng(42)
    c0 = rng.standard_normal((50, 2)) + np.array([-5.0, -5.0])
    c1 = rng.standard_normal((50, 2)) + np.array([5.0, -5.0])
    c2 = rng.standard_normal((50, 2)) + np.array([0.0, 5.0])
    X = np.vstack([c0, c1, c2])

    model = OPTICS(min_samples=5, max_eps=10.0)
    model.fit(X)

    write_fixture(
        "optics",
        {
            "description": "OPTICS on synthetic blobs from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": 42,
            "input": {"X": to_list(X)},
            "params": {"min_samples": 5, "max_eps": 10.0},
            "expected": {
                "labels": to_list(model.labels_),
                "ordering": to_list(model.ordering_),
                "reachability": to_list(
                    np.where(np.isinf(model.reachability_), -1.0, model.reachability_)
                ),
                "n_clusters": int(len(set(model.labels_)) - (1 if -1 in model.labels_ else 0)),
                "n_noise": int(np.sum(model.labels_ == -1)),
            },
        },
    )


# ---------------------------------------------------------------------------
# 34. Birch
# ---------------------------------------------------------------------------
def gen_birch():
    rng = np.random.default_rng(42)
    c0 = rng.standard_normal((50, 2)) + np.array([-5.0, -5.0])
    c1 = rng.standard_normal((50, 2)) + np.array([5.0, -5.0])
    c2 = rng.standard_normal((50, 2)) + np.array([0.0, 5.0])
    X = np.vstack([c0, c1, c2])

    model = Birch(n_clusters=3, threshold=0.5)
    model.fit(X)
    labels = model.predict(X)

    write_fixture(
        "birch",
        {
            "description": "Birch (k=3) on synthetic blobs from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": 42,
            "input": {"X": to_list(X)},
            "params": {"n_clusters": 3, "threshold": 0.5},
            "expected": {
                "labels": to_list(labels),
                "subcluster_centers": to_list(model.subcluster_centers_),
                "n_clusters": int(len(np.unique(labels))),
            },
        },
    )


# ---------------------------------------------------------------------------
# 35. SpectralClustering
# ---------------------------------------------------------------------------
def gen_spectral_clustering():
    rng = np.random.default_rng(42)
    c0 = rng.standard_normal((50, 2)) + np.array([-5.0, -5.0])
    c1 = rng.standard_normal((50, 2)) + np.array([5.0, -5.0])
    c2 = rng.standard_normal((50, 2)) + np.array([0.0, 5.0])
    X = np.vstack([c0, c1, c2])

    model = SpectralClustering(n_clusters=3, random_state=42, affinity="rbf")
    labels = model.fit_predict(X)

    write_fixture(
        "spectral_clustering",
        {
            "description": "SpectralClustering (k=3, rbf) on synthetic blobs from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": 42,
            "input": {"X": to_list(X)},
            "params": {"n_clusters": 3, "random_state": 42, "affinity": "rbf"},
            "expected": {
                "labels": to_list(labels),
                "n_clusters": int(len(np.unique(labels))),
            },
        },
    )


# ===========================================================================
# NEW FIXTURES — Preprocessing
# ===========================================================================


# ---------------------------------------------------------------------------
# 36. MaxAbsScaler
# ---------------------------------------------------------------------------
def gen_max_abs_scaler():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((30, 4)) * 5 + 3

    scaler = MaxAbsScaler()
    X_out = scaler.fit_transform(X)

    write_fixture(
        "max_abs_scaler",
        {
            "description": "MaxAbsScaler fixture from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": 42,
            "input": {"X": to_list(X)},
            "params": {},
            "expected": {
                "max_abs": to_list(scaler.max_abs_),
                "transformed": to_list(X_out),
            },
        },
    )


# ---------------------------------------------------------------------------
# 37. Normalizer
# ---------------------------------------------------------------------------
def gen_normalizer():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((30, 4)) * 5 + 3

    scaler = Normalizer(norm="l2")
    X_out = scaler.fit_transform(X)

    write_fixture(
        "normalizer",
        {
            "description": "Normalizer (L2) fixture from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": 42,
            "input": {"X": to_list(X)},
            "params": {"norm": "l2"},
            "expected": {
                "transformed": to_list(X_out),
            },
        },
    )


# ---------------------------------------------------------------------------
# 38. Binarizer
# ---------------------------------------------------------------------------
def gen_binarizer():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((30, 4))

    scaler = Binarizer(threshold=0.0)
    X_out = scaler.fit_transform(X)

    write_fixture(
        "binarizer",
        {
            "description": "Binarizer (threshold=0.0) fixture from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": 42,
            "input": {"X": to_list(X)},
            "params": {"threshold": 0.0},
            "expected": {
                "transformed": to_list(X_out),
            },
        },
    )


# ---------------------------------------------------------------------------
# 39. PolynomialFeatures
# ---------------------------------------------------------------------------
def gen_polynomial_features():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((20, 3))

    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
    X_out = poly.fit_transform(X)

    write_fixture(
        "polynomial_features",
        {
            "description": "PolynomialFeatures (degree=2) fixture from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": 42,
            "input": {"X": to_list(X)},
            "params": {"degree": 2, "include_bias": False, "interaction_only": False},
            "expected": {
                "n_output_features": int(poly.n_output_features_),
                "transformed": to_list(X_out),
            },
        },
    )


# ---------------------------------------------------------------------------
# 40. OneHotEncoder
# ---------------------------------------------------------------------------
def gen_one_hot_encoder():
    rng = np.random.default_rng(42)
    X = rng.integers(0, 4, size=(30, 3))

    enc = OneHotEncoder(sparse_output=False)
    X_out = enc.fit_transform(X)
    categories = [to_list(c) for c in enc.categories_]

    write_fixture(
        "one_hot_encoder",
        {
            "description": "OneHotEncoder fixture from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": 42,
            "input": {"X": to_list(X)},
            "params": {"sparse_output": False},
            "expected": {
                "categories": categories,
                "transformed": to_list(X_out),
            },
        },
    )


# ---------------------------------------------------------------------------
# 41. LabelEncoder
# ---------------------------------------------------------------------------
def gen_label_encoder():
    labels = ["cat", "dog", "bird", "cat", "bird", "dog", "dog", "cat",
              "bird", "cat", "dog", "bird", "cat", "dog", "bird",
              "cat", "dog", "bird", "cat", "dog"]

    enc = LabelEncoder()
    y_out = enc.fit_transform(labels)

    write_fixture(
        "label_encoder",
        {
            "description": "LabelEncoder fixture from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": None,
            "input": {"labels": labels},
            "params": {},
            "expected": {
                "classes": to_list(enc.classes_),
                "transformed": to_list(y_out),
            },
        },
    )


# ---------------------------------------------------------------------------
# 42. QuantileTransformer
# ---------------------------------------------------------------------------
def gen_quantile_transformer():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 4)) * 5 + 3

    qt = QuantileTransformer(
        n_quantiles=50, output_distribution="uniform", random_state=42
    )
    X_out = qt.fit_transform(X)

    write_fixture(
        "quantile_transformer",
        {
            "description": "QuantileTransformer (uniform) fixture from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": 42,
            "input": {"X": to_list(X)},
            "params": {
                "n_quantiles": 50,
                "output_distribution": "uniform",
                "random_state": 42,
            },
            "expected": {
                "quantiles": to_list(qt.quantiles_),
                "references": to_list(qt.references_),
                "transformed": to_list(X_out),
            },
        },
    )


# ---------------------------------------------------------------------------
# 43. KBinsDiscretizer
# ---------------------------------------------------------------------------
def gen_kbins_discretizer():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((30, 4)) * 5 + 3

    kbd = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="uniform")
    X_out = kbd.fit_transform(X)
    bin_edges = [to_list(e) for e in kbd.bin_edges_]

    write_fixture(
        "kbins_discretizer",
        {
            "description": "KBinsDiscretizer (ordinal, uniform) fixture from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": 42,
            "input": {"X": to_list(X)},
            "params": {"n_bins": 5, "encode": "ordinal", "strategy": "uniform"},
            "expected": {
                "bin_edges": bin_edges,
                "transformed": to_list(X_out),
            },
        },
    )


# ---------------------------------------------------------------------------
# 44. SimpleImputer
# ---------------------------------------------------------------------------
def gen_simple_imputer():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((30, 4))
    # Inject ~10% NaNs
    mask = rng.random((30, 4)) < 0.10
    X[mask] = np.nan

    imp = SimpleImputer(strategy="mean")
    X_out = imp.fit_transform(X)

    write_fixture(
        "simple_imputer",
        {
            "description": "SimpleImputer (mean) fixture from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": 42,
            "input": {"X": to_list(np.where(np.isnan(X), None, X))},
            "params": {"strategy": "mean"},
            "expected": {
                "statistics": to_list(imp.statistics_),
                "transformed": to_list(X_out),
            },
        },
    )


# ---------------------------------------------------------------------------
# 45. PowerTransformer
# ---------------------------------------------------------------------------
def gen_power_transformer():
    rng = np.random.default_rng(42)
    X = np.abs(rng.standard_normal((50, 4))) + 0.1  # positive for yeo-johnson

    pt = PowerTransformer(method="yeo-johnson", standardize=True)
    X_out = pt.fit_transform(X)

    write_fixture(
        "power_transformer",
        {
            "description": "PowerTransformer (yeo-johnson) fixture from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": 42,
            "input": {"X": to_list(X)},
            "params": {"method": "yeo-johnson", "standardize": True},
            "expected": {
                "lambdas": to_list(pt.lambdas_),
                "transformed": to_list(X_out),
            },
        },
    )


# ===========================================================================
# NEW FIXTURES — Model Selection
# ===========================================================================


# ---------------------------------------------------------------------------
# 46. StratifiedKFold
# ---------------------------------------------------------------------------
def gen_stratified_kfold():
    n_samples = 100
    n_splits = 5
    # 3 classes, slightly unbalanced (34/33/33)
    y = np.array([0] * 34 + [1] * 33 + [2] * 33)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=False)
    folds = []
    for train_idx, test_idx in skf.split(np.zeros(n_samples), y):
        folds.append({"train": to_list(train_idx), "test": to_list(test_idx)})

    write_fixture(
        "stratified_kfold",
        {
            "description": "StratifiedKFold (5 folds, unbalanced 34/33/33) from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": None,
            "input": {"n_samples": n_samples, "y": to_list(y)},
            "params": {"n_splits": n_splits, "shuffle": False},
            "expected": {"folds": folds},
        },
    )


# ---------------------------------------------------------------------------
# 47. TimeSeriesSplit
# ---------------------------------------------------------------------------
def gen_time_series_split():
    n_samples = 100
    n_splits = 5

    tscv = TimeSeriesSplit(n_splits=n_splits)
    folds = []
    indices = np.arange(n_samples)
    for train_idx, test_idx in tscv.split(indices):
        folds.append({"train": to_list(train_idx), "test": to_list(test_idx)})

    write_fixture(
        "time_series_split",
        {
            "description": "TimeSeriesSplit (5 folds) from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": None,
            "input": {"n_samples": n_samples},
            "params": {"n_splits": n_splits},
            "expected": {"folds": folds},
        },
    )


# ===========================================================================
# NEW FIXTURES — Metrics
# ===========================================================================


# ---------------------------------------------------------------------------
# 48. ROC AUC
# ---------------------------------------------------------------------------
def gen_roc_auc():
    rng = np.random.default_rng(42)
    n = 100
    y_true = rng.integers(0, 2, size=n)
    # Continuous scores with some signal
    y_score = y_true * 0.6 + rng.standard_normal(n) * 0.3

    auc = roc_auc_score(y_true, y_score)

    write_fixture(
        "roc_auc",
        {
            "description": "ROC AUC score fixture from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": 42,
            "input": {"y_true": to_list(y_true), "y_score": to_list(y_score)},
            "params": {},
            "expected": {"auc": float(auc)},
        },
    )


# ---------------------------------------------------------------------------
# 49. Log Loss
# ---------------------------------------------------------------------------
def gen_log_loss():
    rng = np.random.default_rng(42)
    n = 100
    y_true = rng.integers(0, 2, size=n)
    # Generate probabilities: sigmoid of noisy linear signal
    raw = y_true * 2.0 - 1.0 + rng.standard_normal(n) * 0.5
    proba_1 = 1.0 / (1.0 + np.exp(-raw))
    proba_0 = 1.0 - proba_1
    y_prob = np.column_stack([proba_0, proba_1])

    loss = log_loss(y_true, y_prob)

    write_fixture(
        "log_loss",
        {
            "description": "Log loss fixture from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": 42,
            "input": {"y_true": to_list(y_true), "y_prob": to_list(y_prob)},
            "params": {},
            "expected": {"loss": float(loss)},
        },
    )


# ---------------------------------------------------------------------------
# 50. Clustering Metrics
# ---------------------------------------------------------------------------
def gen_clustering_metrics():
    rng = np.random.default_rng(42)
    c0 = rng.standard_normal((50, 2)) + np.array([-5.0, -5.0])
    c1 = rng.standard_normal((50, 2)) + np.array([5.0, -5.0])
    c2 = rng.standard_normal((50, 2)) + np.array([0.0, 5.0])
    X = np.vstack([c0, c1, c2])
    labels_true = np.array([0] * 50 + [1] * 50 + [2] * 50)

    # Predicted labels from KMeans
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels_pred = km.fit_predict(X)

    sil = silhouette_score(X, labels_pred)
    ari = adjusted_rand_score(labels_true, labels_pred)
    ami = adjusted_mutual_info_score(labels_true, labels_pred)
    dbi = davies_bouldin_score(X, labels_pred)

    write_fixture(
        "clustering_metrics",
        {
            "description": "Clustering metrics (silhouette, ARI, AMI, DBI) from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": 42,
            "input": {
                "X": to_list(X),
                "labels_true": to_list(labels_true),
                "labels_pred": to_list(labels_pred),
            },
            "params": {},
            "expected": {
                "silhouette": float(sil),
                "adjusted_rand": float(ari),
                "adjusted_mutual_info": float(ami),
                "davies_bouldin": float(dbi),
            },
        },
    )


# ---------------------------------------------------------------------------
# 51. Extended Regression Metrics
# ---------------------------------------------------------------------------
def gen_regression_metrics_extended():
    rng = np.random.default_rng(42)
    n = 100
    y_true = rng.standard_normal(n) * 10 + 5  # offset to avoid zero values for MAPE
    noise = rng.standard_normal(n) * 2
    y_pred = y_true + noise

    mape = mean_absolute_percentage_error(y_true, y_pred)
    evs = explained_variance_score(y_true, y_pred)

    write_fixture(
        "regression_metrics_extended",
        {
            "description": "Extended regression metrics (MAPE, explained variance) from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": 42,
            "input": {"y_true": to_list(y_true), "y_pred": to_list(y_pred)},
            "params": {},
            "expected": {
                "mape": float(mape),
                "explained_variance": float(evs),
            },
        },
    )


# ===========================================================================
# NEW FIXTURES — Numerical (scipy)
# ===========================================================================

try:
    import scipy.interpolate
    import scipy.sparse
    import scipy.sparse.linalg
    import scipy.stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ---------------------------------------------------------------------------
# 52. Cubic Spline
# ---------------------------------------------------------------------------
def gen_cubic_spline():
    if not HAS_SCIPY:
        print("  SKIPPED cubic_spline (scipy not installed)")
        return

    x_knots = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    y_knots = np.array([0.0, 0.8, 0.9, 0.1, -0.8, -1.0])

    cs = scipy.interpolate.CubicSpline(x_knots, y_knots, bc_type="not-a-knot")
    eval_points = np.linspace(0.0, 5.0, 21)
    eval_values = cs(eval_points)

    write_fixture(
        "cubic_spline",
        {
            "description": "CubicSpline (not-a-knot) fixture from scipy",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": None,
            "input": {
                "x_knots": to_list(x_knots),
                "y_knots": to_list(y_knots),
            },
            "params": {"bc_type": "not-a-knot"},
            "expected": {
                "eval_points": to_list(eval_points),
                "eval_values": to_list(eval_values),
            },
        },
    )


# ---------------------------------------------------------------------------
# 53. Distributions (PDF/CDF/SF reference values)
# ---------------------------------------------------------------------------
def gen_distributions():
    if not HAS_SCIPY:
        print("  SKIPPED distributions (scipy not installed)")
        return

    test_points = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]
    positive_points = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]

    # Normal(0, 1)
    norm = scipy.stats.norm(0, 1)
    norm_data = {
        "points": test_points,
        "pdf": [float(norm.pdf(x)) for x in test_points],
        "cdf": [float(norm.cdf(x)) for x in test_points],
        "sf": [float(norm.sf(x)) for x in test_points],
    }

    # Chi-squared(df=5)
    chi2 = scipy.stats.chi2(5)
    chi2_data = {
        "df": 5,
        "points": positive_points,
        "pdf": [float(chi2.pdf(x)) for x in positive_points],
        "cdf": [float(chi2.cdf(x)) for x in positive_points],
        "sf": [float(chi2.sf(x)) for x in positive_points],
    }

    # F(df1=5, df2=10)
    f_dist = scipy.stats.f(5, 10)
    f_data = {
        "df1": 5,
        "df2": 10,
        "points": positive_points,
        "pdf": [float(f_dist.pdf(x)) for x in positive_points],
        "cdf": [float(f_dist.cdf(x)) for x in positive_points],
        "sf": [float(f_dist.sf(x)) for x in positive_points],
    }

    # Student's t(df=10)
    t_dist = scipy.stats.t(10)
    t_data = {
        "df": 10,
        "points": test_points,
        "pdf": [float(t_dist.pdf(x)) for x in test_points],
        "cdf": [float(t_dist.cdf(x)) for x in test_points],
        "sf": [float(t_dist.sf(x)) for x in test_points],
    }

    # Beta(a=2, b=5)
    beta = scipy.stats.beta(2, 5)
    unit_points = [0.01, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.99]
    beta_data = {
        "a": 2,
        "b": 5,
        "points": unit_points,
        "pdf": [float(beta.pdf(x)) for x in unit_points],
        "cdf": [float(beta.cdf(x)) for x in unit_points],
        "sf": [float(beta.sf(x)) for x in unit_points],
    }

    # Gamma(a=3, scale=2)
    gamma = scipy.stats.gamma(3, scale=2)
    gamma_data = {
        "shape": 3,
        "scale": 2,
        "points": positive_points,
        "pdf": [float(gamma.pdf(x)) for x in positive_points],
        "cdf": [float(gamma.cdf(x)) for x in positive_points],
        "sf": [float(gamma.sf(x)) for x in positive_points],
    }

    write_fixture(
        "distributions",
        {
            "description": "Statistical distribution reference values from scipy.stats",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": None,
            "input": {},
            "params": {},
            "expected": {
                "normal": norm_data,
                "chi_squared": chi2_data,
                "f_distribution": f_data,
                "students_t": t_data,
                "beta": beta_data,
                "gamma": gamma_data,
            },
        },
    )


# ---------------------------------------------------------------------------
# 54. Sparse Eigenvalues (eigsh)
# ---------------------------------------------------------------------------
def gen_sparse_eigsh():
    if not HAS_SCIPY:
        print("  SKIPPED sparse_eigsh (scipy not installed)")
        return

    # Symmetric positive definite sparse matrix
    n = 20
    rng = np.random.default_rng(42)
    A_dense = rng.standard_normal((n, n))
    A_dense = A_dense @ A_dense.T  # make symmetric positive semi-definite
    A_dense += np.eye(n) * 5  # ensure positive definite

    A_sparse = scipy.sparse.csr_matrix(A_dense)
    k = 3
    eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(A_sparse, k=k, which="LM")

    # Sort by descending eigenvalue
    idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    write_fixture(
        "sparse_eigsh",
        {
            "description": "Sparse eigenvalue decomposition (eigsh, largest 3) from scipy",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": 42,
            "input": {"matrix": to_list(A_dense), "n": n},
            "params": {"k": k, "which": "LM"},
            "expected": {
                "eigenvalues": to_list(eigenvalues),
                "eigenvectors": to_list(eigenvectors),
            },
        },
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"scikit-learn version: {SKLEARN_VERSION}")
    print(f"Writing fixtures to: {FIXTURES_DIR}")
    print()

    generators = [
        # --- Original (1-26) ---
        ("linear_regression", gen_linear_regression),
        ("ridge", gen_ridge),
        ("lasso", gen_lasso),
        ("logistic_regression", gen_logistic_regression),
        ("standard_scaler", gen_standard_scaler),
        ("minmax_scaler", gen_minmax_scaler),
        ("robust_scaler", gen_robust_scaler),
        ("classification_metrics", gen_classification_metrics),
        ("regression_metrics", gen_regression_metrics),
        ("kfold", gen_kfold),
        ("decision_tree_classifier", gen_decision_tree_classifier),
        ("decision_tree_regressor", gen_decision_tree_regressor),
        ("random_forest_classifier", gen_random_forest_classifier),
        ("random_forest_regressor", gen_random_forest_regressor),
        ("gradient_boosting_classifier", gen_gradient_boosting_classifier),
        ("gradient_boosting_regressor", gen_gradient_boosting_regressor),
        ("adaboost_classifier", gen_adaboost_classifier),
        ("kneighbors_classifier", gen_kneighbors_classifier),
        ("kneighbors_regressor", gen_kneighbors_regressor),
        ("gaussian_nb", gen_gaussian_nb),
        ("kmeans", gen_kmeans),
        ("dbscan", gen_dbscan),
        ("agglomerative_clustering", gen_agglomerative),
        ("pca", gen_pca),
        ("nmf", gen_nmf),
        ("elastic_net", gen_elastic_net),
        # --- Naive Bayes (27-29) ---
        ("multinomial_nb", gen_multinomial_nb),
        ("bernoulli_nb", gen_bernoulli_nb),
        ("complement_nb", gen_complement_nb),
        # --- Clustering (30-35) ---
        ("mini_batch_kmeans", gen_mini_batch_kmeans),
        ("mean_shift", gen_mean_shift),
        ("gaussian_mixture", gen_gaussian_mixture),
        ("optics", gen_optics),
        ("birch", gen_birch),
        ("spectral_clustering", gen_spectral_clustering),
        # --- Preprocessing (36-45) ---
        ("max_abs_scaler", gen_max_abs_scaler),
        ("normalizer", gen_normalizer),
        ("binarizer", gen_binarizer),
        ("polynomial_features", gen_polynomial_features),
        ("one_hot_encoder", gen_one_hot_encoder),
        ("label_encoder", gen_label_encoder),
        ("quantile_transformer", gen_quantile_transformer),
        ("kbins_discretizer", gen_kbins_discretizer),
        ("simple_imputer", gen_simple_imputer),
        ("power_transformer", gen_power_transformer),
        # --- Model Selection (46-47) ---
        ("stratified_kfold", gen_stratified_kfold),
        ("time_series_split", gen_time_series_split),
        # --- Metrics (48-51) ---
        ("roc_auc", gen_roc_auc),
        ("log_loss", gen_log_loss),
        ("clustering_metrics", gen_clustering_metrics),
        ("regression_metrics_extended", gen_regression_metrics_extended),
        # --- Numerical / scipy (52-54) ---
        ("cubic_spline", gen_cubic_spline),
        ("distributions", gen_distributions),
        ("sparse_eigsh", gen_sparse_eigsh),
    ]

    errors = []
    for name, fn in generators:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR generating {name}: {exc}", file=sys.stderr)
            errors.append(name)

    print()
    if errors:
        print(f"FAILED: {', '.join(errors)}", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"All {len(generators)} fixtures generated successfully.")


if __name__ == "__main__":
    main()
