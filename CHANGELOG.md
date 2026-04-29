# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.2.2] - 2026-04-29

Coordinated workspace bump for all crates from `0.2.0` (and `ferrolearn-bayes 0.2.1`) to `0.2.2`. Includes the conjugate-priors module previously released as `ferrolearn-bayes 0.2.1`, GP-classifier feature completion, and a workspace-wide maintenance pass.

### Added
- **ferrolearn-kernel**: `GaussianProcessClassifier::log_marginal_likelihood()` — Laplace-approximation log marginal likelihood (Rasmussen & Williams eq. 3.32 / Algorithm 5.1), summed across one-vs-rest binary models for multiclass. Standard objective for kernel hyperparameter selection and model comparison (#237)
- **ferrolearn-kernel**: `FittedGaussianProcessClassifier::classes()` accessor returning sorted class labels (#237)
- **ferrolearn-kernel**: Expose `KernelRidge`/`FittedKernelRidge` (dual-form kernel ridge regression with RBF/Polynomial/Linear/Sigmoid/Laplacian/Cosine kernels), `Nystroem`/`FittedNystroem`/`KernelType` (low-rank Nyström kernel approximation), and `RBFSampler`/`FittedRBFSampler` (random Fourier features per Rahimi & Recht 2007) — these implementations were already in the source tree but the parent modules were not declared in `lib.rs`, so they were unreachable from outside the crate. Activates 52 previously-dormant tests (#4)
- **ferrolearn-bayes**: Conjugate priors module with closed-form posterior updates (`ferrolearn_bayes::conjugate`) (#235, originally released as ferrolearn-bayes 0.2.1)
  - `posterior_normal_normal` — Normal-Normal conjugate update for the latent mean of a Gaussian likelihood with known per-observation variance, given a Normal prior on the mean.
  - `NormalNormalPosterior { mean, var }` — typed posterior summary.

### Changed
- **ferrolearn-kernel**: GP-classifier prediction now uses Rasmussen & Williams Algorithm 3.2 — predictive variance via `K(x*, x*) − ‖L⁻¹√W K(x*, X)ᵀ‖²` and MacKay probit approximation `π̄* = σ(f̄*/√(1+πv*/8))` — replacing the prior shortcut that ignored predictive variance. Probability values are now better-calibrated for points far from training data (#237)
- **ferrolearn-numerical**: Replaced manual `(a + b) / 2.0` with `f64::midpoint(a, b)` in adaptive Simpson, Gauss-Kronrod, and cubic-spline routines for overflow-safe averaging (#239)

### Fixed
- **ferrolearn-decomp**: `LLE::test_lle_different_n_neighbors` now asserts a real difference (`diff_sum > 1e-10`) instead of the no-op `diff_sum > 1e-10 || true` that always passed (#237)
- **ferrolearn-neighbors**: `test_all_algorithms_agree_kneighbors` now compares per-row sorted index sets across BruteForce/KdTree/BallTree, restoring an invariant that was previously dropped (the `reference_idxs` variable was assigned but never read) (#237)
- **ferrolearn-decomp** (`FittedPLSCanonical`, `FittedCCA`): removed stale `#[allow(dead_code)]` on `y_std_` field — it is in fact read by `transform_y` (#237)

### Maintenance
- Bumped 48 transitive dependency versions via `cargo update` (all patch-level, no breaking changes) (#237)
- Cleared 72 default-clippy warnings introduced by the rust 1.95 / clippy update (#238); remaining 67 auto-fixed via `cargo clippy --fix`
- Pedantic+nursery clippy: ~830 fixes across two passes — `redundant_closure`, `manual_let_else`, `single_match_else`, `uninlined_format_args`, `items_after_statements`, `explicit_iter_loop`, `cast_lossless`, `manual_midpoint`, `map_unwrap_or`, `option_if_let_else`, `semicolon_if_nothing_returned`, `ignored_unit_patterns`, `redundant_else`, `used_underscore_binding`, plus ~197 `or_fun_call` rewrites (`or_insert(F::zero())` → `or_insert_with(F::zero)`, `unwrap_or(F::epsilon())` → `unwrap_or_else(F::epsilon)`, etc.) (#239)
- 4 new GP classifier tests covering log-marginal-likelihood structural properties (finiteness, separability monotonicity, multiclass summation) and the new `classes()` accessor (#237)

### Added (post-0.1.0 features rolled into 0.2.2)
- Add RegressorChain for chained multi-target regression (#211)
- Add r_regression Pearson correlation for regression (#101)
- Add LassoLarsCV cross-validated LassoLars (#16)
- Add LeaveOneGroupOut and LeavePGroupsOut splitters (#159)
- Add AdditiveChi2Sampler for additive chi-squared kernel (#193)
- Add GraphicalLasso and GraphicalLassoCV sparse precision matrix (#202)
- Add StratifiedGroupKFold combined stratified+group split (#158)
- Add GroupShuffleSplit group-aware shuffle split (#157)
- Add PolynomialCountSketch for polynomial kernel (#195)
- Add LassoLarsIC Lasso with AIC/BIC selection (#17)
- Add PredefinedSplit for custom fold indices (#161)
- Add ClassifierChain for chained multi-label classification (#210)
- Add mutual_info_classif mutual information for classification (#99)
- Add OutputCodeClassifier error-correcting output codes (#206)
- Add mutual_info_regression mutual information for regression (#100)
- Add SkewedChi2Sampler for skewed chi-squared kernel (#194)
- Add LeavePOut exhaustive P-out cross-validation (#160)
- Expand oracle test coverage to 59 tests across 11 crates (28 new fixtures, 28 new tests)
- Add `brent_bounded` 1-D minimizer to ferrolearn-numerical (Brent's method with bounded interval)
- Add oracle tests for MultinomialNB, BernoulliNB, ComplementNB
- Add oracle tests for MiniBatchKMeans, MeanShift, GaussianMixture, OPTICS, Birch, SpectralClustering
- Add oracle tests for MaxAbsScaler, Normalizer, Binarizer, PolynomialFeatures, OneHotEncoder, LabelEncoder, QuantileTransformer, KBinsDiscretizer, SimpleImputer, PowerTransformer
- Add oracle tests for StratifiedKFold, TimeSeriesSplit
- Add oracle tests for ROC AUC, log loss, clustering metrics, extended regression metrics
- Add oracle tests for CubicSpline, statistical distributions, sparse eigendecomposition

### Fixed (post-0.1.0 fixes rolled into 0.2.2)
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
