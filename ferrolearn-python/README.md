# ferrolearn-python

Python bindings for the [ferrolearn](https://crates.io/crates/ferrolearn)
machine learning framework, built with [PyO3](https://pyo3.rs). Provides a
**scikit-learn compatible API** backed by Rust for performance.

Validated against scikit-learn 1.8.0 head-to-head with **144 paired
measurements** across all 54 exposed estimators — see the
[workspace BENCHMARKS.md](../BENCHMARKS.md) for the full report.

## Installation

```bash
pip install ferrolearn
```

Or for development from this repo:

```bash
cd ferrolearn-python
maturin develop --release
```

## Available estimators (54 total)

### Regressors (17)

`LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`, `BayesianRidge`,
`ARDRegression`, `HuberRegressor`, `QuantileRegressor`,
`DecisionTreeRegressor`, `RandomForestRegressor`, `ExtraTreesRegressor`,
`GradientBoostingRegressor`, `HistGradientBoostingRegressor`,
`KNeighborsRegressor`, `KernelRidge`.

### Classifiers (19)

`LogisticRegression`, `RidgeClassifier`, `LinearSVC`,
`QuadraticDiscriminantAnalysis`, `GaussianNB`, `MultinomialNB`,
`BernoulliNB`, `ComplementNB`, `DecisionTreeClassifier`,
`ExtraTreeClassifier`, `RandomForestClassifier`, `ExtraTreesClassifier`,
`AdaBoostClassifier`, `BaggingClassifier`, `GradientBoostingClassifier`,
`HistGradientBoostingClassifier`, `KNeighborsClassifier`,
`NearestCentroid`.

### Clusterers (6)

`KMeans`, `MiniBatchKMeans`, `DBSCAN`, `AgglomerativeClustering`, `Birch`,
`GaussianMixture`.

### Decomposition / dimensionality reduction (8)

`PCA`, `IncrementalPCA`, `TruncatedSVD`, `FastICA`, `NMF`, `KernelPCA`,
`SparsePCA`, `FactorAnalysis`.

### Preprocessing & kernel approximation (6)

`StandardScaler`, `MinMaxScaler`, `MaxAbsScaler`, `RobustScaler`,
`PowerTransformer`, `Nystroem`, `RBFSampler`.

## Example

```python
from ferrolearn import Ridge, RandomForestClassifier, KMeans, GaussianMixture
import numpy as np

# Regression
X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
y = np.array([1.0, 2.0, 3.0])
model = Ridge(alpha=1.0)
model.fit(X, y)
print(model.predict(X))

# Classification
X = np.random.randn(200, 5)
y = (X.sum(axis=1) > 0).astype(int)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)
print(clf.score(X, y))

# Clustering
gmm = GaussianMixture(n_components=3, random_state=42)
labels = gmm.fit(X).predict(X)
```

All wrappers inherit from scikit-learn's `BaseEstimator` and the appropriate
mixin (`RegressorMixin`, `ClassifierMixin`, `ClusterMixin`,
`TransformerMixin`), so `score()`, `fit_transform()`, pipeline composition,
and `cross_val_score` all work out of the box.

## Performance & parity

Geomean speedups vs scikit-learn 1.8.0 across the 144-row head-to-head bench:

| Family | n compared | fit geomean | predict geomean | mean Δ score |
|---|---:|---:|---:|---:|
| regressor  | 43 | **8.21×** | **4.39×** | -0.0006 R²       |
| classifier | 51 | **6.75×** | **8.88×** | +0.0035 accuracy |
| cluster    | 15 | 1.35×     | —         | +0.0000 ARI (perfect parity) |
| decomp     | 15 | **5.16×** | **4.56×** | —                |
| preprocess | 14 | **9.82×** | **2.74×** | —                |

Reproduce with:

```bash
maturin develop --release
python ferrolearn-bench/head_to_head_full.py > h2h.json
python ferrolearn-bench/render_head_to_head.py h2h.json > REPORT.md
```

## License

Licensed under either of [Apache License, Version 2.0](../LICENSE-APACHE) or
[MIT License](../LICENSE-MIT) at your option.
