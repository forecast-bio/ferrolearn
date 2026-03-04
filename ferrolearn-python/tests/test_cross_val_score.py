"""Test that ferrolearn estimators work with sklearn's cross_val_score."""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import cross_val_score

from ferrolearn import (
    DecisionTreeClassifier,
    ElasticNet,
    GaussianNB,
    KMeans,
    KNeighborsClassifier,
    Lasso,
    LinearRegression,
    LogisticRegression,
    PCA,
    RandomForestClassifier,
    Ridge,
    StandardScaler,
)


@pytest.fixture
def regression_data():
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    return X, y


@pytest.fixture
def classification_data():
    X, y = make_classification(
        n_samples=100, n_features=5, n_informative=3, random_state=42
    )
    return X, y


# --- Regressors ---


@pytest.mark.parametrize(
    "estimator",
    [
        LinearRegression(),
        Ridge(),
        Lasso(),
        ElasticNet(),
    ],
    ids=["LinearRegression", "Ridge", "Lasso", "ElasticNet"],
)
def test_regressor_cross_val_score(estimator, regression_data):
    X, y = regression_data
    scores = cross_val_score(estimator, X, y, cv=3, scoring="r2")
    assert len(scores) == 3
    assert all(isinstance(s, float) for s in scores)
    # R^2 should be positive for reasonable data
    assert np.mean(scores) > 0.0


# --- Classifiers ---


@pytest.mark.parametrize(
    "estimator",
    [
        LogisticRegression(),
        DecisionTreeClassifier(),
        RandomForestClassifier(n_estimators=5, random_state=42),
        KNeighborsClassifier(n_neighbors=3),
        GaussianNB(),
    ],
    ids=[
        "LogisticRegression",
        "DecisionTreeClassifier",
        "RandomForestClassifier",
        "KNeighborsClassifier",
        "GaussianNB",
    ],
)
def test_classifier_cross_val_score(estimator, classification_data):
    X, y = classification_data
    scores = cross_val_score(estimator, X, y, cv=3, scoring="accuracy")
    assert len(scores) == 3
    assert all(isinstance(s, float) for s in scores)
    # Accuracy should be above random (50%)
    assert np.mean(scores) > 0.5
