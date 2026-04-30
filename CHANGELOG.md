# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.3.0] - 2026-04-29

Workspace-wide parity audit against scikit-learn 1.8.0, accompanied by a 4×
expansion of the Python bindings (12 → 54 estimators) and a new dual-library
benchmark harness that runs ferrolearn and scikit-learn head-to-head in one
process across 144 paired measurements.

### Added
- **ferrolearn-bench**: Head-to-head benchmark harness — `head_to_head_full.py` runs all 54 bound estimators against their scikit-learn equivalents in a single Python process with identical datasets, hyperparameters, train/test splits, and quality metrics. Companion `render_head_to_head.py` produces Markdown reports. Per-bench JSON snapshots preserved under `ferrolearn-bench/reports/`. (#330)
- **ferrolearn-python**: 42 new PyO3 bindings — Python now exposes 54 sklearn-compatible estimators (was 12). New: `ARDRegression`, `BayesianRidge`, `HuberRegressor`, `QuantileRegressor`, `RidgeClassifier`, `LinearSVC`, `QuadraticDiscriminantAnalysis`, `MultinomialNB`, `BernoulliNB`, `ComplementNB`, `DecisionTreeRegressor`, `ExtraTreeClassifier`, `ExtraTreesClassifier`, `ExtraTreesRegressor`, `RandomForestRegressor`, `AdaBoostClassifier`, `BaggingClassifier`, `GradientBoostingClassifier`, `GradientBoostingRegressor`, `HistGradientBoostingClassifier`, `HistGradientBoostingRegressor`, `KNeighborsRegressor`, `NearestCentroid`, `MiniBatchKMeans`, `DBSCAN`, `AgglomerativeClustering`, `Birch`, `GaussianMixture`, `IncrementalPCA`, `TruncatedSVD`, `FastICA`, `NMF`, `KernelPCA`, `SparsePCA`, `FactorAnalysis`, `MinMaxScaler`, `MaxAbsScaler`, `RobustScaler`, `PowerTransformer`, `KernelRidge`, `Nystroem`, `RBFSampler`. (#330)
- **BENCHMARKS.md**: Comprehensive head-to-head report across all 144 paired measurements, with per-family geomean speedups and per-row timings + accuracy/R²/ARI deltas. (#331)

### Changed (sklearn parity fixes — measured before/after)
- **ferrolearn-tree**: `RandomForestClassifier` and `RandomForestRegressor` now sample features **per-split** (Breiman 2001 / sklearn behaviour) rather than picking a single fixed feature subset per tree. Closed a -16.05pp accuracy gap at medium_10Kx100. New helper `build_classification_tree_per_split_features` / `build_regression_tree_per_split_features`. (#330)
- **ferrolearn-linear**: `LinearSVC` rewritten with **coordinate-Newton** updates replacing fixed-step (LR=0.01) gradient descent — closed a -21.05pp accuracy gap at medium_10Kx100 while running 2× faster on fit. (#330)
- **ferrolearn-kernel**: `KernelRidge` default kernel changed from `Rbf` to `Linear` to match scikit-learn's `KernelRidge(kernel='linear')` default. Closed a -0.20 R² gap at tiny scale (now exact parity). (#330)
- **ferrolearn-tree**: `AdaBoostClassifier` default algorithm changed from `SAMME.R` to `SAMME` to match scikit-learn ≥ 1.4 (which removed `SAMME.R` in 1.6). Closed a -19.00pp accuracy gap at small scale. (#330)
- **ferrolearn-cluster**: `GaussianMixture` initialisation upgraded from random-row sampling to **Greedy KMeans++** (Arthur & Vassilvitskii 2007 with `2 + log(k)` trial selection, matching sklearn's `_kmeans_plusplus`). M-step now adds `reg_covar = 1e-6` to component covariance diagonals. Closed -0.27 ARI / -0.17 / -0.16 gaps at tiny / small / medium scales (now all exact parity). (#330)
- **ferrolearn-cluster**: `MiniBatchKMeans` defaults switched to scikit-learn 1.4+ values: `batch_size 100 → 1024`, `max_iter 300 → 100`, `tol 1e-4 → 0.0`. Closed a -0.16 ARI gap at medium_5Kx20 (now exact parity). (#330)
- **ferrolearn-cluster**: `KMeans`, `MiniBatchKMeans` initialisations upgraded to **Greedy KMeans++** for robustness at scale. (#330)
- **ferrolearn-linear**: `QuantileRegressor` IRLS L1 penalty now scaled by `n_samples` so the user-facing `alpha` parameter is numerically equivalent to scikit-learn's. Previously `alpha=1.0` in ferrolearn was effectively `~1/n` of sklearn's `alpha=1.0`. (#332)

### Workspace
- All workspace crates bumped from 0.2.2 → 0.3.0. (#329)
- Workspace test count: **3,662 tests passing**, 0 failing.

### Bench results — geomean speedups vs scikit-learn 1.8.0 (n=144 paired runs)

| Family | n | fit geomean | predict geomean | mean Δ score |
|---|---:|---:|---:|---:|
| regressor | 43 | **8.21×** | **4.39×** | -0.0006 R² |
| classifier | 51 | **6.75×** | **8.88×** | +0.0035 accuracy |
| cluster | 15 | 1.35× | — | +0.0000 ARI (exact parity, 15/15) |
| decomp | 15 | **5.16×** | **4.56×** | — |
| preprocess | 14 | **9.82×** | **2.74×** | — (numerical agreement to 1e-16) |
| kernel approx | 6 | **6.78×** | 1.26× | — |

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
- datasets: add network fetch_* loaders + cache management (fetch_california_housing, get_data_home, clear_data_home, fetch_openml) (#321)
- numerical: scipy parity audit — special functions (gamma, beta, erf, etc.) + linalg (decompositions live in core::backend) (#322)
- model-sel: add make_pipeline, make_union helpers + threshold classifiers (FixedThresholdClassifier, TunedThresholdClassifierCV) (#316)
- model-sel: add inspection module (partial_dependence, permutation_importance) (#315)
- datasets: add file I/O loaders (load_svmlight_file, dump_svmlight_file, load_files) (#320)
- metrics: add scorer registry (get_scorer, get_scorer_names, check_scoring) + DistanceMetric trait (#308)
- model-sel: add ClassifierChain, RegressorChain, OutputCodeClassifier (#313)
- model-sel: add group-aware CV splitters (GroupKFold, GroupShuffleSplit, LeaveOneGroupOut, LeavePGroupsOut, StratifiedGroupKFold) (#312)
- metrics: add ranking edge cases (coverage_error, label_ranking_average_precision_score, label_ranking_loss) (#307)
- metrics: add d2_* family (d2_absolute_error_score, d2_pinball_score, d2_tweedie_score, d2_brier_score, d2_log_loss_score) (#306)
- Add new crates for uncovered sklearn modules: covariance, neural_network (#252)
- covariance + neural: write api_proof.rs for both new crates (#328)
- neural: implement BernoulliRBM (Bernoulli-Bernoulli RBM with CD-1 training) (#327)
- neural: create ferrolearn-neural crate; move mlp.rs from linear; add to workspace + umbrella (#326)
- covariance: add GraphicalLasso + GraphicalLassoCV + function-style exports (empirical_covariance, ledoit_wolf, oas, shrunk_covariance, log_likelihood, fast_mcd) (#325)
- covariance: create ferrolearn-covariance crate; move covariance.rs from decomp; add to workspace + umbrella (#324)
- Audit utility crates (ferrolearn-core, ferrolearn-datasets, ferrolearn-sparse, ferrolearn-numerical, ferrolearn-io) vs sklearn equivalents (#251)
- utility crates: write tests/api_proof.rs for core, datasets, sparse, numerical, io (#323)
- sparse: add stack/eye/diags helpers (hstack, vstack, eye, diags, sparse_random) (#319)
- datasets: add 7 missing generators (make_friedman1/2/3, make_low_rank_matrix, make_spd_matrix, make_sparse_spd_matrix, make_gaussian_quantiles, make_hastie_10_2, make_multilabel_classification) (#318)
- Audit ferrolearn-model-sel vs sklearn (model_selection + pipeline + compose + multiclass + multioutput + dummy + frozen + inspection + calibration): close gaps, add API proof tests (#249)
- model-sel: write tests/api_proof.rs covering every public API (#317)
- model-sel: add dummy.rs (DummyClassifier, DummyRegressor) (#314)
- model-sel: add basic CV splitters (LeaveOneOut, LeavePOut, ShuffleSplit, StratifiedShuffleSplit, RepeatedKFold, RepeatedStratifiedKFold, PredefinedSplit) (#311)
- model-sel: wire 4 orphaned modules (feature_union, multiclass, multioutput, transformed_target) in lib.rs (#310)
- Audit ferrolearn-metrics vs sklearn: close gaps, add API proof tests (#248)
- metrics: write tests/api_proof.rs covering every public API in ferrolearn-metrics (#309)
- metrics: add missing pairwise (pairwise_distances_argmin, argmin_min, pairwise_kernels) (#305)
- metrics: add missing clustering metrics (mutual_info_score, pair_confusion_matrix, homogeneity_completeness_v_measure, contingency_matrix) (#304)
- metrics: add 13 missing classification metrics (hamming, zero_one, balanced_accuracy, matthews_corrcoef, cohen_kappa, jaccard, fbeta, brier_score, hinge, multilabel_confusion_matrix, precision_recall_fscore_support, classification_report, det_curve) (#303)
- metrics: wire orphaned scorer module + 11 regression/clustering/pairwise re-exports in lib.rs (#302)
- Audit ferrolearn-preprocess vs sklearn (preprocessing + impute + feature_extraction + feature_selection): close gaps, add API proof tests (#247)
- Add proof-of-API integration test for ferrolearn-preprocess (#301)
- Wire orphaned preprocess estimators into lib.rs (LabelBinarizer, MultiLabelBinarizer, SelectFpr/Fdr/Fwe, SequentialFeatureSelector, feature scoring fns) (#299)
- Add GaussianRandomProjection / SparseRandomProjection / johnson_lindenstrauss_min_dim (#296)
- Audit ferrolearn-decomp vs sklearn (decomposition + cross_decomposition + manifold + random_projection): close gaps, add API proof tests (#246)
- Add proof-of-API integration test for ferrolearn-decomp (#298)
- Add inverse_transform to KernelPCA / IncrementalPCA / NMF / TruncatedSVD / FactorAnalysis (#295)
- Wire orphaned MiniBatchNMF and SparsePCA into ferrolearn-decomp lib.rs (#294)
- Audit ferrolearn-kernel vs sklearn (kernel_approximation + kernel_ridge): close gaps, add API proof tests (#250)
- Add proof-of-API integration test for ferrolearn-kernel (#292)
- Add sample_y() to GaussianProcessRegressor for posterior sampling (#291)
- Add predict_log_proba to GaussianProcessClassifier (#290)
- Add score() to KernelRidge / GaussianProcessRegressor / GaussianProcessClassifier (#289)
- Audit ferrolearn-linear vs sklearn (linear_model + svm + isotonic + discriminant_analysis): close gaps, add API proof tests (#245)
- Add proof-of-API integration test for ferrolearn-linear (#288)
- Add decision_function to LDA / QDA / RidgeClassifier / LogisticRegression / LogisticRegressionCV / LinearSVC / SGDClassifier (#287)
- Add predict_proba and predict_log_proba to classifiers missing them (LDA, QDA, RidgeClassifier, LogRegCV, SGDClassifier, LinearSVC) (#286)
- Add score() to every fitted linear / SVM / isotonic / discriminant_analysis estimator (#285)
- Wire 14 orphaned linear estimators into lib.rs (ARD, GLM family, Lars+LassoLars, LinearSVC/R, LogRegCV, OMP, QDA, QuantileRegressor, RidgeClassifier, MLP) (#284)
- Audit ferrolearn-cluster vs sklearn (cluster + mixture + semi_supervised): close gaps, add API proof tests (#244)
- Add proof-of-API integration test for ferrolearn-cluster (#282)
- Add predict_proba + score to LabelPropagation and LabelSpreading (#281)
- Add transform() to KMeans / MiniBatchKMeans / BisectingKMeans (#280)
- Fix GMM bic()/aic() signatures and add to BayesianGaussianMixture (#279)
- Add predict_proba, score, score_samples to GaussianMixture and BayesianGaussianMixture (#278)
- Add fit_predict and labels() accessor to all clustering estimators (#277)
- Audit ferrolearn-tree vs sklearn (tree + ensemble): close gaps, add API proof tests (#243)
- Add proof-of-API integration test for ferrolearn-tree (#275)
- Add decision_function to GradientBoosting / HistGradientBoosting / AdaBoost classifiers (#272)
- Add predict_log_proba to all classifiers (#270)
- Add predict_proba to remaining classifiers (RF, GB, HGB, AdaBoost, Bagging, Voting) (#269)
- Add feature_importances_ accessor to every tree-based estimator (#271)
- Add score() method to every fitted tree / ensemble estimator (#268)
- Wire orphaned modules into lib.rs: BaggingClassifier, BaggingRegressor, AdaBoostRegressor (#267)
- Audit ferrolearn-neighbors vs sklearn: close gaps, add API proof tests (#242)
- Add proof-of-API integration test for ferrolearn-neighbors (#266)
- Complete LocalOutlierFactor sklearn API: decision_function, fit_predict, score_samples, novelty mode (#265)
- Add kneighbors_graph and radius_neighbors_graph (free fns + methods) plus sort_graph_by_row_values (#264)
- Add kneighbors() and radius_neighbors() methods to supervised neighbors estimators (#263)
- Add score() method to all neighbors estimators (#262)
- Add predict_proba to KNeighborsClassifier and RadiusNeighborsClassifier (#261)
- Audit ferrolearn-bayes vs sklearn (naive_bayes + gaussian_process): close gaps, add API proof tests (#241)
- Add proof-of-API integration test exercising every public ferrolearn-bayes estimator end-to-end (#260)
- Add partial_fit method to CategoricalNB (#259)
- Add min_categories parameter to CategoricalNB (#258)
- Add norm parameter to ComplementNB (#257)
- Add force_alpha parameter to discrete Naive Bayes estimators (#256)
- Add fit_prior parameter to discrete Naive Bayes estimators (Multinomial, Bernoulli, Complement, Categorical) (#255)
- Add score() convenience method (mean accuracy) to all Naive Bayes fitted estimators (#254)
- Add predict_log_proba and predict_joint_log_proba methods to all Naive Bayes fitted estimators (#253)
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
