# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [ferrolearn-bayes 0.2.1] - 2026-04-28

### Added
- Add conjugate priors module with closed-form posterior updates (`ferrolearn_bayes::conjugate`) (#235)
  - `posterior_normal_normal` — Normal-Normal conjugate update for the latent mean of a Gaussian likelihood with known per-observation variance, given a Normal prior on the mean.
  - `NormalNormalPosterior { mean, var }` — typed posterior summary.

## [Unreleased] - 2026-03-14

### Added
- Expand oracle test coverage to 59 tests across 11 crates (28 new fixtures, 28 new tests)
- Add `brent_bounded` 1-D minimizer to ferrolearn-numerical (Brent's method with bounded interval)
- Add oracle tests for MultinomialNB, BernoulliNB, ComplementNB
- Add oracle tests for MiniBatchKMeans, MeanShift, GaussianMixture, OPTICS, Birch, SpectralClustering
- Add oracle tests for MaxAbsScaler, Normalizer, Binarizer, PolynomialFeatures, OneHotEncoder, LabelEncoder, QuantileTransformer, KBinsDiscretizer, SimpleImputer, PowerTransformer
- Add oracle tests for StratifiedKFold, TimeSeriesSplit
- Add oracle tests for ROC AUC, log loss, clustering metrics, extended regression metrics
- Add oracle tests for CubicSpline, statistical distributions, sparse eigendecomposition

### Fixed
- Fix OPTICS Xi cluster extraction: rewrite to use steep-down areas with MIB tracking, region extension, and predecessor correction (matching sklearn's Figure 19 algorithm)
- Fix Birch final clustering: replace KMeans (naive init) with AgglomerativeClustering Ward linkage, eliminating initialization-dependent convergence failures
- Fix PowerTransformer lambda optimization: replace 201-point grid search (0.03 step) with Brent's method for continuous-precision optimization matching sklearn
- Fix StratifiedKFold remainder distribution: use round-robin fold offset across classes for balanced fold sizes (was front-loading extras to first folds)

## [0.1.0] - 2026-03-04

### Added
- Add missing scipy-equivalent numerical foundations (#19)
- Resolve open questions in kernel regression design document (#18)
- Add kernel regression design document for ferrolearn-kernel crate (#17)
- Add Pipeline support for f32 data (generic over float type) (#14)

Initial release with full scikit-learn-equivalent coverage across 14 crates.

### Phase 1: Foundation

- **ferrolearn-core**: `Fit`, `Predict`, `Transform`, `FitTransform` traits; `Dataset` type; `FerroError` error hierarchy; `Pipeline` with type-safe unfitted/fitted state; introspection traits (`HasCoefficients`, `HasFeatureImportances`, `HasClasses`)
- **ferrolearn-linear**: `LinearRegression`, `Ridge`, `Lasso`, `LogisticRegression` with L-BFGS optimizer
- **ferrolearn-preprocess**: `StandardScaler`, `MinMaxScaler`, `RobustScaler`
- **ferrolearn-metrics**: Classification metrics (accuracy, precision, recall, F1, confusion matrix, ROC AUC, log loss); regression metrics (MAE, MSE, RMSE, R², MAPE)
- **ferrolearn-model-sel**: `KFold`, `StratifiedKFold`, `train_test_split`, `cross_val_score`
- **ferrolearn-sparse**: CSR, CSC, COO sparse matrix formats with conversions and arithmetic
- **ferrolearn-datasets**: `load_iris`, `load_diabetes`, `load_wine`; synthetic generators (`make_blobs`, `make_classification`, `make_regression`, `make_moons`, `make_circles`)
- Compile-fail tests ensuring unfitted models cannot call `predict()`
- Oracle test infrastructure with 10 sklearn fixture generators

### Phase 2: Classical ML

- **ferrolearn-tree**: `DecisionTreeClassifier`, `DecisionTreeRegressor`, `RandomForestClassifier`, `RandomForestRegressor` with feature importances and configurable criteria
- **ferrolearn-neighbors**: `KNeighborsClassifier`, `KNeighborsRegressor` with KD-tree acceleration (auto-selected for dims <= 20) and distance weighting
- **ferrolearn-cluster**: `KMeans`, `DBSCAN`, `AgglomerativeClustering` (Ward/Complete/Average/Single linkage), `GaussianMixture`
- **ferrolearn-decomp**: `PCA`, `TruncatedSVD`, `NMF`, `KernelPCA` (RBF/polynomial/linear/sigmoid kernels)
- **ferrolearn-preprocess**: `SimpleImputer`, `VarianceThreshold`, `SelectKBest`, `OneHotEncoder`, `OrdinalEncoder`, `LabelEncoder`, `PolynomialFeatures`
- **ferrolearn-io**: MessagePack and JSON model serialization with CRC32 integrity checks
- **ferrolearn-model-sel**: `GridSearchCV`, `RandomizedSearchCV`, `HalvingGridSearch`, `param_grid!` macro, `TimeSeriesSplit`
- **ferrolearn-metrics**: Clustering metrics (`silhouette_score`, `adjusted_rand_index`, `normalized_mutual_info`, `calinski_harabasz`, `davies_bouldin`)

### Phase 3: Completeness

- **ferrolearn-tree**: `GradientBoostingClassifier`, `GradientBoostingRegressor` (least squares, LAD, Huber loss), `AdaBoostClassifier` (SAMME/SAMME.R)
- **ferrolearn-preprocess**: `MaxAbsScaler`, `Normalizer`, `Binarizer`, `PowerTransformer` (Yeo-Johnson/Box-Cox), `FunctionTransformer`
- **ferrolearn-core**: Compile-time type-safe `TypedPipeline`; pluggable `Backend` trait with `NdarrayFaerBackend` (gemm, svd, qr, cholesky, solve, eigh, det, inv)
- **ferrolearn-bayes**: `GaussianNB`, `MultinomialNB`, `BernoulliNB`, `ComplementNB`

### Phase 4: Beyond sklearn Baseline

- **ferrolearn-core**: `PartialFit` trait for online/incremental learning
- **ferrolearn-linear**: `ElasticNet`, `BayesianRidge`, `HuberRegressor`, `SGDClassifier`, `SGDRegressor`, `LDA` (Linear Discriminant Analysis)
- **ferrolearn-preprocess**: `ColumnTransformer`
- **ferrolearn-decomp**: `IncrementalPCA`, `FactorAnalysis`, `FastICA`, `Isomap`, `MDS`, `SpectralEmbedding`, `LLE`
- **ferrolearn-cluster**: `MiniBatchKMeans`, `MeanShift`, `SpectralClustering`, `OPTICS`
- **ferrolearn-model-sel**: `CalibratedClassifierCV`, `SelfTrainingClassifier`
- **ferrolearn-datasets**: `make_sparse_uncorrelated`

### Testing & Validation

- 1,468 tests across 14 crates, 0 failures
- 26 sklearn oracle tests comparing numerical output (predictions, coefficients, metrics) against scikit-learn 1.7.2 reference fixtures
- 7 end-to-end integration tests (classification pipeline, regression pipeline, clustering, cross-validation, serialization roundtrip, tree ensemble, preprocessing chain)
- Compile-fail tests for type-safety guarantees
- Fixture generation script (`scripts/generate_fixtures.py`) for reproducible sklearn baselines
