# sklearn vs ferrolearn — Gap Report

> Generated 2026-03-25 by parallel analysis of scikit-learn source (HEAD) against ferrolearn workspace (15 crates, 697 tests).

## Executive Summary

| Category | sklearn items | ferrolearn items | Coverage |
|----------|:---:|:---:|:---:|
| Linear Models | 38 | 14 | 37% |
| Tree + Ensemble | 20 | 9 | 45% |
| Neighbors | 13 | 6 | 46% |
| Naive Bayes | 5 | 4 | 80% |
| Clustering + Mixture | 17 | 12 | 71% |
| Decomposition + Cross-Decomp | 30 | 15 | 50% |
| Preprocessing + Features | 65 | 26 | 40% |
| Metrics | 72 | 17 | 24% |
| Model Selection + Pipeline | 40 | 18 | 45% |
| Uncovered Modules (SVM, GP, NN, etc.) | ~60 | 4 | 7% |
| **Total** | **~360** | **~125** | **~35%** |

### Strongest coverage
- **Naive Bayes** (80%) — 4 of 5 estimators, missing only CategoricalNB
- **Clustering** (71%) — 12 estimators including HDBSCAN, OPTICS, Spectral, GMM, LabelPropagation/Spreading
- **Decomposition** (50%) — 15 modules including PCA, ICA, NMF, LDA, t-SNE, UMAP, manifold methods

### Largest gaps
- **Metrics** (24%) — missing ranking metrics, pairwise distances, many classification/regression metrics
- **Uncovered modules** (7%) — entire sklearn domains with no ferrolearn crate (GP, NN, covariance, calibration, multiclass/multioutput strategies)
- **Linear Models** (37%) — missing all CV variants, LARS, GLMs, multi-task estimators

---

## 1. Linear Models (`sklearn.linear_model` → `ferrolearn-linear`)

### Present (14 estimators)
| Estimator | Notes |
|-----------|-------|
| LinearRegression | OLS via QR decomposition |
| Ridge | L2-regularized, Cholesky |
| Lasso | L1-regularized, coordinate descent |
| ElasticNet | L1+L2, coordinate descent |
| BayesianRidge | Evidence maximization |
| LogisticRegression | Binary/multiclass, L-BFGS |
| HuberRegressor | Robust, IRLS |
| SGDClassifier | Multiple loss functions |
| SGDRegressor | Multiple loss functions |
| RANSACRegressor | Robust meta-estimator |
| IsotonicRegression | PAV algorithm |
| LDA | Linear Discriminant Analysis |
| SVC | SMO with kernels (Linear/RBF/Poly/Sigmoid) |
| SVR | SMO with kernels |

### Missing (24 estimators)

**High priority:**
| Estimator | Description |
|-----------|-------------|
| RidgeCV | Cross-validated Ridge |
| LassoCV | Cross-validated Lasso |
| ElasticNetCV | Cross-validated ElasticNet |
| ARDRegression | Bayesian automatic feature selection |
| RidgeClassifier | Ridge for classification |
| RidgeClassifierCV | Cross-validated RidgeClassifier |

**Medium priority:**
| Estimator | Description |
|-----------|-------------|
| Lars | Least Angle Regression |
| LarsCV | Cross-validated LARS |
| LassoLars | Lasso via LARS |
| LassoLarsCV | Cross-validated LassoLars |
| LassoLarsIC | Lasso LARS with AIC/BIC |
| OrthogonalMatchingPursuit | Greedy sparse approximation |
| OrthogonalMatchingPursuitCV | Cross-validated OMP |
| MultiTaskLasso | L1 across multiple targets |
| MultiTaskElasticNet | L1+L2 across multiple targets |
| MultiTaskLassoCV | Cross-validated multi-task Lasso |
| MultiTaskElasticNetCV | Cross-validated multi-task ElasticNet |
| QuantileRegressor | Conditional quantile regression |
| TheilSenRegressor | Nonparametric robust estimator |

**Low-medium priority:**
| Estimator | Description |
|-----------|-------------|
| GammaRegressor | GLM, Gamma family |
| PoissonRegressor | GLM, Poisson family |
| TweedieRegressor | GLM, Tweedie family |
| PassiveAggressiveClassifier | Online hinge loss (deprecated sklearn 1.8) |
| PassiveAggressiveRegressor | Online regression (deprecated sklearn 1.8) |

### Partial implementations
- **LogisticRegression** — missing: LogisticRegressionCV, L1/elasticnet penalties, liblinear/sag/saga solvers, sample_weight, class_weight
- **SGDClassifier/SGDRegressor** — missing: momentum/Nesterov, L1/elasticnet penalties, warm_start, class_weight, sample_weight

---

## 2. Tree + Ensemble (`sklearn.tree` + `sklearn.ensemble` → `ferrolearn-tree`)

### Present (9 estimators)
| Estimator | Notes |
|-----------|-------|
| DecisionTreeClassifier | CART, Gini/Entropy |
| DecisionTreeRegressor | CART, MSE |
| RandomForestClassifier | Rayon parallel, max_features strategies |
| RandomForestRegressor | Bootstrap aggregation |
| GradientBoostingClassifier | Log loss, exponential |
| GradientBoostingRegressor | MSE, MAE, Huber, Quantile |
| HistGradientBoostingClassifier | Binning, subtraction trick, native NaN |
| HistGradientBoostingRegressor | Histogram-based |
| AdaBoostClassifier | SAMME/SAMME.R |

### Missing (11 estimators)

**High priority:**
| Estimator | Description |
|-----------|-------------|
| ExtraTreeClassifier | Extremely randomized tree |
| ExtraTreeRegressor | Extremely randomized tree (regression) |
| ExtraTreesClassifier | Ensemble of extra trees |
| ExtraTreesRegressor | Ensemble of extra trees (regression) |
| BaggingClassifier | Generic bagging meta-estimator |
| BaggingRegressor | Generic bagging (regression) |
| AdaBoostRegressor | AdaBoost for regression |

**Medium priority:**
| Estimator | Description |
|-----------|-------------|
| StackingClassifier | Meta-learner stacking |
| StackingRegressor | Meta-learner stacking (regression) |
| VotingClassifier | Hard/soft voting ensemble |
| VotingRegressor | Voting ensemble (regression) |
| IsolationForest | Anomaly detection |

### Partial implementations (existing estimators)
- **DecisionTree***: missing `splitter` (best vs random), `max_leaf_nodes`, `ccp_alpha` (cost-complexity pruning), `class_weight`, `monotonic_cst`, sample_weight
- **RandomForest***: missing `warm_start`, `oob_score`, `max_samples`, sample_weight
- **All tree models**: missing `export_graphviz`, `export_text` for visualization

---

## 3. Neighbors (`sklearn.neighbors` → `ferrolearn-neighbors`)

### Present (6 items)
| Item | Notes |
|------|-------|
| KNeighborsClassifier | Uniform/Distance/Custom weights, Auto/KdTree/BallTree/BruteForce |
| KNeighborsRegressor | Same weight/algorithm options |
| KdTree | Spatial index for low-dim |
| BallTree | Spatial index for high-dim |
| Algorithm enum | Auto/KdTree/BallTree/BruteForce |
| Weights enum | Uniform/Distance/Custom |

### Missing (7 estimators)

**High priority:**
| Estimator | Description |
|-----------|-------------|
| RadiusNeighborsClassifier | All neighbors within radius |
| RadiusNeighborsRegressor | Radius-based regression |
| NearestNeighbors | Unsupervised neighbor search |

**Medium priority:**
| Estimator | Description |
|-----------|-------------|
| LocalOutlierFactor | Local density outlier detection |
| NearestCentroid | Centroid-based classifier |

**Lower priority:**
| Estimator | Description |
|-----------|-------------|
| NeighborhoodComponentsAnalysis | Metric learning |
| KernelDensity | Kernel density estimation |

### Partial implementations
- **Distance metrics** — hardcoded Euclidean; sklearn supports minkowski, manhattan, and many scipy metrics
- **Sparse data** — not supported; sklearn handles CSR/CSC input
- **Precomputed distances** — not supported; sklearn accepts `metric='precomputed'`

---

## 4. Naive Bayes (`sklearn.naive_bayes` → `ferrolearn-bayes`)

### Present (4 estimators)
| Estimator | Notes |
|-----------|-------|
| GaussianNB | var_smoothing, predict_proba |
| MultinomialNB | alpha smoothing |
| BernoulliNB | alpha, binarize threshold |
| ComplementNB | alpha, imbalanced-data variant |

### Missing (1 estimator)
| Estimator | Priority | Description |
|-----------|----------|-------------|
| CategoricalNB | HIGH | Naive Bayes for categorical features |

### Cross-cutting gaps (all 4 estimators)
- `partial_fit` — incremental/online learning (MEDIUM)
- `class_prior` parameter — user-specified priors (MEDIUM)
- `sample_weight` support (LOW)
- Fitted attribute introspection traits (LOW)

---

## 5. Clustering + Mixture (`sklearn.cluster` + `sklearn.mixture` → `ferrolearn-cluster`)

### Present (12 estimators)
| Estimator | Notes |
|-----------|-------|
| KMeans | k-Means++ init, Rayon parallel, multi-start |
| MiniBatchKMeans | Online variant, learning rate |
| DBSCAN | Density-based, epsilon/min_samples |
| HDBSCAN | Hierarchical DBSCAN, auto clusters |
| OPTICS | Reachability ordering |
| AgglomerativeClustering | Ward/Complete/Average/Single linkage |
| Birch | Balanced iterative reducing |
| SpectralClustering | Graph Laplacian eigenmap |
| MeanShift | Mode-seeking |
| GaussianMixture | EM, 4 covariance types |
| LabelPropagation | Semi-supervised (unique to ferrolearn-cluster) |
| LabelSpreading | Semi-supervised (unique to ferrolearn-cluster) |

### Missing (6 estimators)

**High priority:**
| Estimator | Description |
|-----------|-------------|
| AffinityPropagation | Message-passing, auto cluster count |
| BisectingKMeans | Divisive hierarchical k-means |

**Medium priority:**
| Estimator | Description |
|-----------|-------------|
| BayesianGaussianMixture | Dirichlet process GMM |
| FeatureAgglomeration | Feature-space hierarchical clustering |
| SpectralBiclustering | Row+column biclustering |
| SpectralCoclustering | Log-diagonal biclustering |

### Missing utilities
- `estimate_bandwidth`, `cluster_optics_dbscan`, `cluster_optics_xi` (MEDIUM)

---

## 6. Decomposition (`sklearn.decomposition` + `sklearn.cross_decomposition` → `ferrolearn-decomp`)

### Present (15 modules)
| Estimator | Notes |
|-----------|-------|
| PCA | Full fit/transform, explained variance |
| IncrementalPCA | Online/mini-batch |
| TruncatedSVD | Randomized algorithm |
| FastICA | logcosh/exp/cube nonlinearities |
| NMF | MU + CD solvers, NNDSVD init |
| KernelPCA | Non-linear dimensionality reduction |
| FactorAnalysis | Probabilistic latent factors |
| LatentDirichletAllocation | Topic modeling |
| DictionaryLearning | Sparse coding with learned dictionary |
| MDS | Classical multidimensional scaling |
| Isomap | Geodesic distances on kNN graphs |
| LLE | Locally Linear Embedding |
| SpectralEmbedding | Laplacian Eigenmaps |
| t-SNE | Barnes-Hut approximation |
| UMAP | Topological manifold learning (beyond sklearn) |

### Missing (15 items)

**High priority (cross-decomposition — entirely absent):**
| Estimator | Description |
|-----------|-------------|
| PLSRegression | Partial Least Squares regression |
| PLSCanonical | PLS canonical correlation |
| CCA | Canonical Correlation Analysis |
| PLSSVD | SVD of cross-covariance |

**Medium priority:**
| Estimator | Description |
|-----------|-------------|
| SparsePCA | L1-penalized PCA |
| MiniBatchSparsePCA | Online SparsePCA |
| MiniBatchNMF | Online NMF |
| MiniBatchDictionaryLearning | Online dictionary learning |
| SparseCoder | Transform-only sparse encoder |

**Low priority (utility functions):**
- `dict_learning`, `dict_learning_online`, `fastica`, `non_negative_factorization`, `randomized_svd`, `sparse_encode`

---

## 7. Preprocessing + Feature Engineering (`sklearn.preprocessing` + `feature_extraction` + `feature_selection` + `impute` → `ferrolearn-preprocess`)

### Present (26 items)
| Category | Implemented |
|----------|-------------|
| **Scalers** (6/6) | StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer, PowerTransformer |
| **Encoders** (4/4) | OneHotEncoder, OrdinalEncoder, LabelEncoder, TargetEncoder |
| **Discretizers** (2/2) | KBinsDiscretizer, SplineTransformer |
| **Feature Engineering** (3/3) | PolynomialFeatures, Binarizer, FunctionTransformer |
| **Imputers** (3/3) | SimpleImputer, KNNImputer, IterativeImputer |
| **Feature Selection** (5/11) | VarianceThreshold, SelectKBest, SelectPercentile, RFE, RFECV |
| **Composition** (1) | ColumnTransformer |

### Missing (39 items)

**High priority:**
| Item | Description |
|------|-------------|
| LabelBinarizer | Multilabel one-vs-rest binarization |
| MultiLabelBinarizer | Multi-hot label encoding |
| SelectFromModel | Feature selection from fitted model importances |
| SequentialFeatureSelector | Forward/backward greedy selection |

**Medium-high priority (feature selection score functions):**
| Function | Description |
|----------|-------------|
| f_classif | ANOVA F-statistic for classification |
| f_regression | Univariate F-statistic for regression |
| chi2 | Chi-squared test for non-negative features |
| mutual_info_classif | Mutual information for classification |
| mutual_info_regression | Mutual information for regression |
| SelectFdr / SelectFpr / SelectFwe | Statistical threshold selectors |
| GenericUnivariateSelect | Configurable univariate filter |

**Medium priority (feature extraction — entire submodule absent):**
| Item | Description |
|------|-------------|
| CountVectorizer | Bag-of-words term frequency |
| TfidfVectorizer | TF-IDF from raw text |
| TfidfTransformer | TF-IDF on pre-tokenized counts |
| HashingVectorizer | Hashing trick for text |
| DictVectorizer | Dict to sparse matrix |
| FeatureHasher | Generic feature hashing |

**Low priority:**
- Stateless preprocessing functions (`scale`, `robust_scale`, `minmax_scale`, etc.)
- Image feature extraction (`PatchExtractor`, `extract_patches_2d`, etc.)
- `MissingIndicator`
- `KernelCenterer`

### Partial implementations
- **RFE/RFECV** — accepts pre-computed importance vectors rather than wrapping estimators (architectural choice to avoid circular deps)
- **SelectFromModel** — exists as pass-through; needs meta-estimator wrapping

---

## 8. Metrics (`sklearn.metrics` → `ferrolearn-metrics`)

### Present (17 functions)
| Category | Implemented |
|----------|-------------|
| **Classification** (7) | accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, log_loss |
| **Regression** (6) | mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score, mean_absolute_percentage_error, explained_variance_score |
| **Clustering** (4) | silhouette_score, adjusted_rand_score, adjusted_mutual_info, davies_bouldin_score |

### Missing (55 items)

**High priority — classification:**
| Metric | Description |
|--------|-------------|
| balanced_accuracy_score | Class-balanced accuracy |
| matthews_corrcoef | MCC for binary classification |
| cohen_kappa_score | Inter-rater agreement |
| fbeta_score | Weighted harmonic mean |
| classification_report | Text summary report |
| brier_score_loss | Probabilistic prediction error |
| hamming_loss | Fraction of incorrect labels |
| jaccard_score | Intersection / union |

**High priority — regression:**
| Metric | Description |
|--------|-------------|
| median_absolute_error | Median of absolute residuals |
| max_error | Maximum absolute error |
| mean_squared_log_error | MSE on log-transformed targets |
| root_mean_squared_log_error | RMSE on log-transformed targets |

**High priority — clustering:**
| Metric | Description |
|--------|-------------|
| silhouette_samples | Per-sample silhouette |
| calinski_harabasz_score | Variance ratio criterion |
| homogeneity_score / completeness_score / v_measure_score | Homogeneity/completeness/V-measure |

**High priority — ranking (entire category absent):**
| Metric | Description |
|--------|-------------|
| auc | Area under arbitrary curve |
| average_precision_score | Average precision for ranking |
| roc_curve | FPR/TPR at thresholds |
| precision_recall_curve | Precision/recall at thresholds |
| ndcg_score | Normalized DCG |
| top_k_accuracy_score | Top-k accuracy |

**Medium-high priority — pairwise (entire category absent):**
| Item | Description |
|------|-------------|
| pairwise_distances | Pairwise distance computation |
| euclidean_distances | Pairwise Euclidean |
| pairwise_kernels | Pairwise kernel matrices |
| DistanceMetric | Distance metric class |

**Medium priority:**
- Scorer utilities (`make_scorer`, `get_scorer`, `check_scoring`)
- Threshold curves (`det_curve`, `confusion_matrix_at_thresholds`)
- Additional clustering metrics (`rand_score`, `normalized_mutual_info_score`, `fowlkes_mallows_score`)
- Deviance metrics (`mean_poisson_deviance`, `mean_gamma_deviance`, `mean_tweedie_deviance`)

---

## 9. Model Selection + Pipeline (`sklearn.model_selection` + `pipeline` + `compose` → `ferrolearn-model-sel` + `ferrolearn-core`)

### Present (18 items)
| Category | Implemented |
|----------|-------------|
| **CV Splitters** (3) | KFold, StratifiedKFold, TimeSeriesSplit |
| **Search** (3) | GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV |
| **Distributions** (5) | Uniform, LogUniform, IntUniform, Choice, Distribution trait |
| **Validation** (2) | cross_val_score, train_test_split |
| **Meta-estimators** (2) | CalibratedClassifierCV, SelfTrainingClassifier |
| **Pipeline** (2) | Pipeline (dynamic-dispatch), TypedPipeline (compile-time) |
| **Macro** (1) | param_grid! |

### Missing (22 items)

**High priority — CV splitters:**
| Splitter | Description |
|----------|-------------|
| LeaveOneOut | LOO cross-validation |
| RepeatedKFold | Multiple independent k-fold runs |
| RepeatedStratifiedKFold | Stratified repeated k-fold |
| GroupKFold | Group-aware k-fold |
| ShuffleSplit | Random train/test splits |
| StratifiedShuffleSplit | Stratified random splits |

**High priority — validation functions:**
| Function | Description |
|----------|-------------|
| cross_validate | Multi-metric CV with timing |
| cross_val_predict | Aggregate predictions across folds |
| learning_curve | Score vs training set size |
| validation_curve | Score vs hyperparameter value |
| permutation_test_score | Statistical significance |

**High priority — search:**
| Estimator | Description |
|-----------|-------------|
| HalvingRandomSearchCV | Successive halving + random sampling |

**Medium priority — pipeline:**
| Item | Description |
|------|-------------|
| FeatureUnion | Parallel transformer composition |
| TransformedTargetRegressor | Target-space transformation |
| make_pipeline | Convenience constructor |

**Medium priority — other:**
| Item | Description |
|------|-------------|
| LeavePOut | Leave-P-out CV |
| PredefinedSplit | Custom fold indices |
| GroupShuffleSplit | Group-aware shuffle split |
| FixedThresholdClassifier | Custom probability threshold |
| TunedThresholdClassifierCV | Threshold tuning via CV |

---

## 10. Uncovered sklearn Modules (no dedicated ferrolearn crate)

These represent entire sklearn domains with minimal or no coverage.

### SVM (`sklearn.svm`) — PARTIALLY COVERED via ferrolearn-linear

| Estimator | Status | Priority |
|-----------|--------|----------|
| SVC | Implemented (ferrolearn-linear) | -- |
| SVR | Implemented (ferrolearn-linear) | -- |
| LinearSVC | **Missing** | HIGH |
| LinearSVR | **Missing** | HIGH |
| NuSVC | **Missing** | MEDIUM |
| NuSVR | **Missing** | MEDIUM |
| OneClassSVM | **Missing** | MEDIUM |

### Gaussian Processes — COMPLETELY MISSING

| Estimator | Priority |
|-----------|----------|
| GaussianProcessClassifier | HIGH |
| GaussianProcessRegressor | HIGH |
| GP Kernels (RBF, Matern, ConstantKernel, WhiteKernel, DotProduct, RationalQuadratic, ExpSineSquared) | HIGH |
| Kernel operators (Sum, Product, Exponentiation) | MEDIUM |

### Neural Networks — COMPLETELY MISSING

| Estimator | Priority |
|-----------|----------|
| MLPClassifier | HIGH |
| MLPRegressor | HIGH |
| BernoulliRBM | MEDIUM |

### Discriminant Analysis — PARTIALLY COVERED

| Estimator | Status | Priority |
|-----------|--------|----------|
| LinearDiscriminantAnalysis | Implemented as LDA (ferrolearn-linear) | -- |
| QuadraticDiscriminantAnalysis | **Missing** | HIGH |

### Manifold Learning — MOSTLY COVERED via ferrolearn-decomp

| Estimator | Status |
|-----------|--------|
| MDS, t-SNE, Isomap, LLE, SpectralEmbedding | Implemented |
| trustworthiness() metric | **Missing** (LOW) |

### Semi-Supervised — PARTIALLY COVERED

| Estimator | Status | Priority |
|-----------|--------|----------|
| LabelPropagation | Implemented (ferrolearn-cluster) | -- |
| LabelSpreading | Implemented (ferrolearn-cluster) | -- |
| SelfTrainingClassifier | Implemented (ferrolearn-model-sel) | -- |

### Kernel Approximation — COMPLETELY MISSING

| Estimator | Priority |
|-----------|----------|
| RBFSampler | HIGH |
| Nystroem | HIGH |
| AdditiveChi2Sampler | MEDIUM |
| SkewedChi2Sampler | MEDIUM |
| PolynomialCountSketch | LOW |

### Kernel Ridge — MISSING

| Estimator | Priority |
|-----------|----------|
| KernelRidge | HIGH |

### Covariance Estimation — COMPLETELY MISSING

| Estimator | Priority |
|-----------|----------|
| EmpiricalCovariance | MEDIUM |
| ShrunkCovariance | MEDIUM |
| LedoitWolf | MEDIUM |
| OAS | MEDIUM |
| MinCovDet | MEDIUM |
| EllipticEnvelope | MEDIUM |
| GraphicalLasso / GraphicalLassoCV | LOW |

### Multi-class / Multi-output — COMPLETELY MISSING

| Estimator | Priority |
|-----------|----------|
| OneVsRestClassifier | HIGH |
| OneVsOneClassifier | HIGH |
| MultiOutputClassifier | HIGH |
| MultiOutputRegressor | HIGH |
| ClassifierChain | MEDIUM |
| RegressorChain | MEDIUM |
| OutputCodeClassifier | MEDIUM |

### Random Projection — COMPLETELY MISSING

| Estimator | Priority |
|-----------|----------|
| GaussianRandomProjection | MEDIUM |
| SparseRandomProjection | MEDIUM |

### Calibration — PARTIALLY COVERED

| Estimator | Status | Priority |
|-----------|--------|----------|
| CalibratedClassifierCV | Implemented (ferrolearn-model-sel) | -- |
| calibration_curve() | **Missing** | MEDIUM |

### Isotonic Regression — COVERED

| Estimator | Status |
|-----------|--------|
| IsotonicRegression | Implemented (ferrolearn-linear) |

### Dummy Estimators — MISSING

| Estimator | Priority |
|-----------|----------|
| DummyClassifier | LOW |
| DummyRegressor | LOW |

### Inspection — MISSING

| Function | Priority |
|----------|----------|
| partial_dependence | MEDIUM |
| permutation_importance | MEDIUM |

---

## Priority Tiers for Future Work

### Tier 1 — High Impact, Commonly Used

These gaps affect the most common ML workflows:

1. **Cross-validated model variants** — RidgeCV, LassoCV, ElasticNetCV (users expect auto-tuning)
2. **Metrics expansion** — balanced_accuracy, MCC, classification_report, roc_curve, precision_recall_curve, auc, pairwise_distances
3. **Extra Trees + Bagging** — ExtraTrees*, Bagging*, AdaBoostRegressor (common ensemble baselines)
4. **CV splitters** — RepeatedKFold, ShuffleSplit, GroupKFold (essential for proper evaluation)
5. **Multi-class/multi-output strategies** — OvR, OvO, MultiOutput* (needed for real-world multiclass)
6. **Validation functions** — cross_validate, learning_curve, validation_curve
7. **Gaussian Processes** — GP classifier/regressor + kernel library
8. **Neural Networks** — MLP classifier/regressor
9. **Kernel methods** — KernelRidge, RBFSampler, Nystroem

### Tier 2 — Medium Impact, Specialized

10. **Cross-decomposition** — PLS, CCA (common in chemometrics, bioinformatics)
11. **Feature selection functions** — f_classif, chi2, mutual_info_*, SelectFromModel
12. **Text feature extraction** — CountVectorizer, TfidfVectorizer (large NLP use case)
13. **Covariance estimation** — LedoitWolf, EllipticEnvelope
14. **Stacking/Voting ensembles** — StackingClassifier/Regressor, VotingClassifier/Regressor
15. **SVM variants** — LinearSVC, LinearSVR, NuSVC, NuSVR
16. **QDA** — QuadraticDiscriminantAnalysis
17. **Decision tree enhancements** — ccp_alpha pruning, random splitter, max_leaf_nodes, oob_score

### Tier 3 — Lower Impact, Niche

18. **Sparse PCA variants** — SparsePCA, MiniBatchSparsePCA
19. **LARS family** — Lars, LassoLars and CV variants
20. **GLM family** — PoissonRegressor, GammaRegressor, TweedieRegressor
21. **Random projection** — GaussianRandomProjection, SparseRandomProjection
22. **Semi-supervised** (partially covered already)
23. **Dummy estimators** — DummyClassifier, DummyRegressor
24. **Pipeline enhancements** — FeatureUnion, make_pipeline, TransformedTargetRegressor
25. **Inspection tools** — partial_dependence, permutation_importance
26. **Deprecated estimators** — Perceptron, PassiveAggressive* (deprecated in sklearn 1.8)

---

## What ferrolearn Has That sklearn Doesn't

| Feature | Crate | Notes |
|---------|-------|-------|
| UMAP | ferrolearn-decomp | sklearn requires separate umap-learn package |
| HDBSCAN | ferrolearn-cluster | Only added to sklearn recently |
| TypedPipeline | ferrolearn-core | Compile-time type-safe pipeline (Rust-specific) |
| Kernel regression (Nadaraya-Watson, Local Polynomial) | ferrolearn-kernel | Not in sklearn |
| Wild bootstrap CI, heteroscedasticity tests | ferrolearn-kernel | Statistical diagnostics |
| Scipy-equivalent numerical foundations | ferrolearn-numerical | Optimization, interpolation, integration |
| DynKernel runtime kernel selection | ferrolearn-kernel | Dynamic dispatch kernel selection |
| Silverman/Scott/CV bandwidth selection | ferrolearn-kernel | Automatic bandwidth for kernel methods |
