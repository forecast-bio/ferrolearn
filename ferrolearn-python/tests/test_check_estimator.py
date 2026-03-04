"""Test that all ferrolearn estimators pass sklearn's check_estimator."""

import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

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

ALL_ESTIMATORS = [
    LinearRegression(),
    Ridge(),
    Lasso(),
    ElasticNet(),
    LogisticRegression(),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=5, random_state=42),
    KNeighborsClassifier(n_neighbors=3),
    GaussianNB(),
    StandardScaler(),
    PCA(n_components=2),
    KMeans(n_clusters=3, random_state=42, n_init=2),
]


@parametrize_with_checks(ALL_ESTIMATORS)
def test_estimator(estimator, check):
    check(estimator)
