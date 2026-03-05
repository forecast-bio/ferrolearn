# ferrolearn

A scikit-learn equivalent for Rust. Type-safe, modular machine learning built on [ndarray](https://github.com/rust-ndarray/ndarray).

```rust
use ferrolearn::prelude::*;
use ferrolearn::{linear, preprocess, decomp, datasets};

// Load data
let (x, y) = datasets::load_iris::<f64>().unwrap();

// Build a pipeline: scale -> PCA -> logistic regression
let pipeline = Pipeline::new()
    .transform_step("scaler", Box::new(preprocess::StandardScaler::<f64>::new()))
    .transform_step("pca", Box::new(decomp::PCA::<f64>::new(2)))
    .estimator_step("clf", Box::new(linear::LogisticRegression::<f64>::new()));
```

## Features

**Supervised Learning**
- Linear models: `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`, `LogisticRegression`, `BayesianRidge`, `HuberRegressor`, `SGDClassifier`, `SGDRegressor`, `LDA`
- Tree models: `DecisionTreeClassifier`, `DecisionTreeRegressor`
- Ensembles: `RandomForestClassifier`, `RandomForestRegressor`, `GradientBoostingClassifier`, `GradientBoostingRegressor`, `AdaBoostClassifier`
- Neighbors: `KNeighborsClassifier`, `KNeighborsRegressor` (with KD-tree acceleration)
- Naive Bayes: `GaussianNB`, `MultinomialNB`, `BernoulliNB`, `ComplementNB`

**Unsupervised Learning**
- Clustering: `KMeans`, `MiniBatchKMeans`, `DBSCAN`, `AgglomerativeClustering`, `GaussianMixture`, `MeanShift`, `SpectralClustering`, `OPTICS`
- Decomposition: `PCA`, `IncrementalPCA`, `TruncatedSVD`, `NMF`, `KernelPCA`, `FactorAnalysis`, `FastICA`
- Manifold learning: `Isomap`, `MDS`, `SpectralEmbedding`, `LLE`

**Preprocessing**
- Scalers: `StandardScaler`, `MinMaxScaler`, `RobustScaler`, `MaxAbsScaler`, `Normalizer`
- Encoders: `OneHotEncoder`, `OrdinalEncoder`, `LabelEncoder`
- Feature engineering: `PolynomialFeatures`, `Binarizer`, `PowerTransformer`
- Missing data: `SimpleImputer`
- Composition: `ColumnTransformer`, `FunctionTransformer`
- Feature selection: `VarianceThreshold`, `SelectKBest`

**Model Selection**
- Cross-validation: `KFold`, `StratifiedKFold`, `TimeSeriesSplit`, `cross_val_score`
- Hyperparameter search: `GridSearchCV`, `RandomizedSearchCV`, `HalvingGridSearch`
- Data splitting: `train_test_split`
- Calibration: `CalibratedClassifierCV`
- Semi-supervised: `SelfTrainingClassifier`

**Metrics**
- Classification: `accuracy`, `precision`, `recall`, `f1`, `confusion_matrix`, `roc_auc`, `log_loss`
- Regression: `mae`, `mse`, `rmse`, `r2`, `mape`
- Clustering: `silhouette_score`, `adjusted_rand_index`, `normalized_mutual_info`

**Infrastructure**
- Datasets: `load_iris`, `load_diabetes`, `load_wine`, `make_blobs`, `make_classification`, `make_regression`
- Serialization: MessagePack and JSON via `ferrolearn-io`
- Sparse matrices: CSR, CSC, COO formats via `ferrolearn-sparse`
- Pipelines: type-safe `Pipeline` with compile-time guarantees (unfitted models can't predict)
- Backend trait: pluggable linear algebra with `NdarrayFaerBackend` (gemm, svd, qr, cholesky, eigh)

## Architecture

ferrolearn is a workspace of 14 crates. Use the umbrella crate for convenience, or depend on individual crates for smaller binaries:

| Crate | Description |
|-------|-------------|
| `ferrolearn` | Umbrella re-export crate |
| `ferrolearn-core` | Traits (`Fit`, `Predict`, `Transform`), error types, pipeline, backend |
| `ferrolearn-linear` | Linear and generalized linear models |
| `ferrolearn-tree` | Decision trees and ensemble methods |
| `ferrolearn-neighbors` | k-Nearest Neighbors with KD-tree |
| `ferrolearn-bayes` | Naive Bayes classifiers |
| `ferrolearn-cluster` | Clustering algorithms |
| `ferrolearn-decomp` | Dimensionality reduction and decomposition |
| `ferrolearn-preprocess` | Scalers, encoders, imputers, feature engineering |
| `ferrolearn-metrics` | Evaluation metrics |
| `ferrolearn-model-sel` | Cross-validation, hyperparameter search, calibration |
| `ferrolearn-datasets` | Toy datasets and synthetic data generators |
| `ferrolearn-io` | Model serialization (MessagePack, JSON) |
| `ferrolearn-sparse` | Sparse matrix formats (CSR, CSC, COO) |

## Core traits

All models follow a consistent type-state pattern:

```rust
// Unfitted model — can configure, cannot predict
let model = Ridge::<f64>::new().with_alpha(1.0);

// Fit returns a new FittedRidge type
let fitted = model.fit(&x, &y)?;

// Only the fitted type can predict
let predictions = fitted.predict(&x_test)?;
```

The key traits from `ferrolearn-core`:

- **`Fit<X, Y>`** — Train a model, producing a fitted type
- **`Predict<X>`** — Generate predictions from a fitted model
- **`Transform<X>`** — Transform data (scalers, PCA, etc.)
- **`PartialFit<X, Y>`** — Incremental/online learning
- **`FitTransform<X>`** — Fit and transform in one step

## Requirements

- Rust edition 2024
- MSRV: 1.85

## Testing

ferrolearn is validated against scikit-learn with 26 numerical oracle tests that compare predictions, coefficients, and metrics against sklearn reference values:

```bash
# Run the full test suite (1,932 tests)
cargo test --workspace

# Run only oracle tests
cargo test --workspace --test oracle_tests

# Regenerate sklearn fixtures (requires Python + scikit-learn)
python scripts/generate_fixtures.py
```

## License

Licensed under MIT OR Apache-2.0 at your option.
