# ferrolearn

A scikit-learn equivalent for Rust. Type-safe, modular machine learning built on [ndarray][1].

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
- Linear models: [`LinearRegression`][2], [`Ridge`][3], [`Lasso`][4], [`ElasticNet`][5], [`LogisticRegression`][6], [`BayesianRidge`][7], [`HuberRegressor`][8], [`SGDClassifier`][9], [`SGDRegressor`][10], [`LDA`][11]
- Tree models: [`DecisionTreeClassifier`][12], [`DecisionTreeRegressor`][13]
- Ensembles: [`RandomForestClassifier`][14], [`RandomForestRegressor`][15], [`GradientBoostingClassifier`][16], [`GradientBoostingRegressor`][17], [`AdaBoostClassifier`][18]
- Neighbors: [`KNeighborsClassifier`][19], [`KNeighborsRegressor`][20] (with KD-tree acceleration)
- Naive Bayes: [`GaussianNB`][21], [`MultinomialNB`][22], [`BernoulliNB`][23], [`ComplementNB`][24]

**Unsupervised Learning**
- Clustering: [`KMeans`][25], [`MiniBatchKMeans`][26], [`DBSCAN`][27], [`AgglomerativeClustering`][28], [`GaussianMixture`][29], [`MeanShift`][30], [`SpectralClustering`][31], [`OPTICS`][32]
- Decomposition: [`PCA`][33], [`IncrementalPCA`][34], [`TruncatedSVD`][35], [`NMF`][36], [`KernelPCA`][37], [`FactorAnalysis`][38], [`FastICA`][39]
- Manifold learning: [`Isomap`][40], [`MDS`][41], [`SpectralEmbedding`][42], [`LLE`][43]

**Preprocessing**
- Scalers: [`StandardScaler`][44], [`MinMaxScaler`][45], [`RobustScaler`][46], [`MaxAbsScaler`][47], [`Normalizer`][48]
- Encoders: [`OneHotEncoder`][49], [`LabelEncoder`][50]
- Feature engineering: [`PolynomialFeatures`][51], [`Binarizer`][52], [`PowerTransformer`][53]
- Missing data: [`SimpleImputer`][54]
- Feature selection: [`VarianceThreshold`][55], [`SelectKBest`][56]

**Model Selection**
- Cross-validation: [`KFold`][57], [`StratifiedKFold`][58], [`TimeSeriesSplit`][59], [`cross_val_score`][60]
- Hyperparameter search: [`GridSearchCV`][61], [`RandomizedSearchCV`][62], [`HalvingGridSearchCV`][63]
- Data splitting: [`train_test_split`][64]
- Calibration: [`CalibratedClassifierCV`][65]
- Semi-supervised: [`SelfTrainingClassifier`][66]

**Metrics**
- Classification: [`accuracy_score`][67], [`precision_score`][68], [`recall_score`][69], [`f1_score`][70], [`confusion_matrix`][71], [`roc_auc_score`][72], [`log_loss`][73]
- Regression: [`mean_absolute_error`][74], [`mean_squared_error`][75], [`root_mean_squared_error`][76], [`r2_score`][77], [`mean_absolute_percentage_error`][78]
- Clustering: [`silhouette_score`][79], [`adjusted_rand_score`][80], [`adjusted_mutual_info`][81]

**Infrastructure**
- Datasets: [`load_iris`][82], [`load_diabetes`][83], [`load_wine`][84], [`make_blobs`][85], [`make_classification`][86], [`make_regression`][87]
- Serialization: MessagePack and JSON via [`ferrolearn-io`][88]
- Sparse matrices: CSR, CSC, COO formats via [`ferrolearn-sparse`][89]
- Pipelines: type-safe [`Pipeline`][90] with compile-time guarantees (unfitted models can't predict)
- Backend trait: pluggable linear algebra with [`NdarrayFaerBackend`][91] (gemm, svd, qr, cholesky, eigh)

## Architecture

ferrolearn is a workspace of 14 crates. Use the umbrella crate for convenience, or depend on individual crates for smaller binaries:

| Crate | Description |
|-------|-------------|
| [`ferrolearn`][92] | Umbrella re-export crate |
| [`ferrolearn-core`][93] | Traits ([`Fit`][106], [`Predict`][107], [`Transform`][108]), error types, pipeline, backend |
| [`ferrolearn-linear`][94] | Linear and generalized linear models |
| [`ferrolearn-tree`][95] | Decision trees and ensemble methods |
| [`ferrolearn-neighbors`][96] | k-Nearest Neighbors with KD-tree |
| [`ferrolearn-bayes`][97] | Naive Bayes classifiers |
| [`ferrolearn-cluster`][98] | Clustering algorithms |
| [`ferrolearn-decomp`][99] | Dimensionality reduction and decomposition |
| [`ferrolearn-preprocess`][100] | Scalers, encoders, imputers, feature engineering |
| [`ferrolearn-metrics`][101] | Evaluation metrics |
| [`ferrolearn-model-sel`][102] | Cross-validation, hyperparameter search, calibration |
| [`ferrolearn-datasets`][103] | Toy datasets and synthetic data generators |
| [`ferrolearn-io`][104] | Model serialization (MessagePack, JSON) |
| [`ferrolearn-sparse`][105] | Sparse matrix formats (CSR, CSC, COO) |

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

The key traits from [`ferrolearn-core`][93]:

- **[`Fit<X, Y>`][106]** — Train a model, producing a fitted type
- **[`Predict<X>`][107]** — Generate predictions from a fitted model
- **[`Transform<X>`][108]** — Transform data (scalers, PCA, etc.)
- **[`PartialFit<X, Y>`][109]** — Incremental/online learning
- **[`FitTransform<X>`][110]** — Fit and transform in one step

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

[1]: https://github.com/rust-ndarray/ndarray
[2]: https://docs.rs/ferrolearn-linear/latest/ferrolearn_linear/linear_regression/struct.LinearRegression.html
[3]: https://docs.rs/ferrolearn-linear/latest/ferrolearn_linear/ridge/struct.Ridge.html
[4]: https://docs.rs/ferrolearn-linear/latest/ferrolearn_linear/lasso/struct.Lasso.html
[5]: https://docs.rs/ferrolearn-linear/latest/ferrolearn_linear/elastic_net/struct.ElasticNet.html
[6]: https://docs.rs/ferrolearn-linear/latest/ferrolearn_linear/logistic_regression/struct.LogisticRegression.html
[7]: https://docs.rs/ferrolearn-linear/latest/ferrolearn_linear/bayesian_ridge/struct.BayesianRidge.html
[8]: https://docs.rs/ferrolearn-linear/latest/ferrolearn_linear/huber_regressor/struct.HuberRegressor.html
[9]: https://docs.rs/ferrolearn-linear/latest/ferrolearn_linear/sgd/struct.SGDClassifier.html
[10]: https://docs.rs/ferrolearn-linear/latest/ferrolearn_linear/sgd/struct.SGDRegressor.html
[11]: https://docs.rs/ferrolearn-linear/latest/ferrolearn_linear/lda/struct.LDA.html
[12]: https://docs.rs/ferrolearn-tree/latest/ferrolearn_tree/decision_tree/struct.DecisionTreeClassifier.html
[13]: https://docs.rs/ferrolearn-tree/latest/ferrolearn_tree/decision_tree/struct.DecisionTreeRegressor.html
[14]: https://docs.rs/ferrolearn-tree/latest/ferrolearn_tree/random_forest/struct.RandomForestClassifier.html
[15]: https://docs.rs/ferrolearn-tree/latest/ferrolearn_tree/random_forest/struct.RandomForestRegressor.html
[16]: https://docs.rs/ferrolearn-tree/latest/ferrolearn_tree/gradient_boosting/struct.GradientBoostingClassifier.html
[17]: https://docs.rs/ferrolearn-tree/latest/ferrolearn_tree/gradient_boosting/struct.GradientBoostingRegressor.html
[18]: https://docs.rs/ferrolearn-tree/latest/ferrolearn_tree/adaboost/struct.AdaBoostClassifier.html
[19]: https://docs.rs/ferrolearn-neighbors/latest/ferrolearn_neighbors/knn/struct.KNeighborsClassifier.html
[20]: https://docs.rs/ferrolearn-neighbors/latest/ferrolearn_neighbors/knn/struct.KNeighborsRegressor.html
[21]: https://docs.rs/ferrolearn-bayes/latest/ferrolearn_bayes/gaussian/struct.GaussianNB.html
[22]: https://docs.rs/ferrolearn-bayes/latest/ferrolearn_bayes/multinomial/struct.MultinomialNB.html
[23]: https://docs.rs/ferrolearn-bayes/latest/ferrolearn_bayes/bernoulli/struct.BernoulliNB.html
[24]: https://docs.rs/ferrolearn-bayes/latest/ferrolearn_bayes/complement/struct.ComplementNB.html
[25]: https://docs.rs/ferrolearn-cluster/latest/ferrolearn_cluster/kmeans/struct.KMeans.html
[26]: https://docs.rs/ferrolearn-cluster/latest/ferrolearn_cluster/mini_batch_kmeans/struct.MiniBatchKMeans.html
[27]: https://docs.rs/ferrolearn-cluster/latest/ferrolearn_cluster/dbscan/struct.DBSCAN.html
[28]: https://docs.rs/ferrolearn-cluster/latest/ferrolearn_cluster/agglomerative_clustering/struct.AgglomerativeClustering.html
[29]: https://docs.rs/ferrolearn-cluster/latest/ferrolearn_cluster/gaussian_mixture/struct.GaussianMixture.html
[30]: https://docs.rs/ferrolearn-cluster/latest/ferrolearn_cluster/mean_shift/struct.MeanShift.html
[31]: https://docs.rs/ferrolearn-cluster/latest/ferrolearn_cluster/spectral_clustering/struct.SpectralClustering.html
[32]: https://docs.rs/ferrolearn-cluster/latest/ferrolearn_cluster/optics/struct.OPTICS.html
[33]: https://docs.rs/ferrolearn-decomp/latest/ferrolearn_decomp/pca/struct.PCA.html
[34]: https://docs.rs/ferrolearn-decomp/latest/ferrolearn_decomp/incremental_pca/struct.IncrementalPCA.html
[35]: https://docs.rs/ferrolearn-decomp/latest/ferrolearn_decomp/truncated_svd/struct.TruncatedSVD.html
[36]: https://docs.rs/ferrolearn-decomp/latest/ferrolearn_decomp/nmf/struct.NMF.html
[37]: https://docs.rs/ferrolearn-decomp/latest/ferrolearn_decomp/kernel_pca/struct.KernelPCA.html
[38]: https://docs.rs/ferrolearn-decomp/latest/ferrolearn_decomp/factor_analysis/struct.FactorAnalysis.html
[39]: https://docs.rs/ferrolearn-decomp/latest/ferrolearn_decomp/fast_ica/struct.FastICA.html
[40]: https://docs.rs/ferrolearn-decomp/latest/ferrolearn_decomp/isomap/struct.Isomap.html
[41]: https://docs.rs/ferrolearn-decomp/latest/ferrolearn_decomp/mds/struct.MDS.html
[42]: https://docs.rs/ferrolearn-decomp/latest/ferrolearn_decomp/spectral_embedding/struct.SpectralEmbedding.html
[43]: https://docs.rs/ferrolearn-decomp/latest/ferrolearn_decomp/lle/struct.LLE.html
[44]: https://docs.rs/ferrolearn-preprocess/latest/ferrolearn_preprocess/standard_scaler/struct.StandardScaler.html
[45]: https://docs.rs/ferrolearn-preprocess/latest/ferrolearn_preprocess/min_max_scaler/struct.MinMaxScaler.html
[46]: https://docs.rs/ferrolearn-preprocess/latest/ferrolearn_preprocess/robust_scaler/struct.RobustScaler.html
[47]: https://docs.rs/ferrolearn-preprocess/latest/ferrolearn_preprocess/max_abs_scaler/struct.MaxAbsScaler.html
[48]: https://docs.rs/ferrolearn-preprocess/latest/ferrolearn_preprocess/normalizer/struct.Normalizer.html
[49]: https://docs.rs/ferrolearn-preprocess/latest/ferrolearn_preprocess/one_hot_encoder/struct.OneHotEncoder.html
[50]: https://docs.rs/ferrolearn-preprocess/latest/ferrolearn_preprocess/label_encoder/struct.LabelEncoder.html
[51]: https://docs.rs/ferrolearn-preprocess/latest/ferrolearn_preprocess/polynomial_features/struct.PolynomialFeatures.html
[52]: https://docs.rs/ferrolearn-preprocess/latest/ferrolearn_preprocess/binarizer/struct.Binarizer.html
[53]: https://docs.rs/ferrolearn-preprocess/latest/ferrolearn_preprocess/power_transformer/struct.PowerTransformer.html
[54]: https://docs.rs/ferrolearn-preprocess/latest/ferrolearn_preprocess/imputer/struct.SimpleImputer.html
[55]: https://docs.rs/ferrolearn-preprocess/latest/ferrolearn_preprocess/feature_selection/struct.VarianceThreshold.html
[56]: https://docs.rs/ferrolearn-preprocess/latest/ferrolearn_preprocess/feature_selection/struct.SelectKBest.html
[57]: https://docs.rs/ferrolearn-model-sel/latest/ferrolearn_model_sel/cross_validation/struct.KFold.html
[58]: https://docs.rs/ferrolearn-model-sel/latest/ferrolearn_model_sel/cross_validation/struct.StratifiedKFold.html
[59]: https://docs.rs/ferrolearn-model-sel/latest/ferrolearn_model_sel/time_series_split/struct.TimeSeriesSplit.html
[60]: https://docs.rs/ferrolearn-model-sel/latest/ferrolearn_model_sel/cross_validation/fn.cross_val_score.html
[61]: https://docs.rs/ferrolearn-model-sel/latest/ferrolearn_model_sel/grid_search/struct.GridSearchCV.html
[62]: https://docs.rs/ferrolearn-model-sel/latest/ferrolearn_model_sel/random_search/struct.RandomizedSearchCV.html
[63]: https://docs.rs/ferrolearn-model-sel/latest/ferrolearn_model_sel/halving_grid_search/struct.HalvingGridSearchCV.html
[64]: https://docs.rs/ferrolearn-model-sel/latest/ferrolearn_model_sel/split/fn.train_test_split.html
[65]: https://docs.rs/ferrolearn-model-sel/latest/ferrolearn_model_sel/calibration/struct.CalibratedClassifierCV.html
[66]: https://docs.rs/ferrolearn-model-sel/latest/ferrolearn_model_sel/self_training/struct.SelfTrainingClassifier.html
[67]: https://docs.rs/ferrolearn-metrics/latest/ferrolearn_metrics/classification/fn.accuracy_score.html
[68]: https://docs.rs/ferrolearn-metrics/latest/ferrolearn_metrics/classification/fn.precision_score.html
[69]: https://docs.rs/ferrolearn-metrics/latest/ferrolearn_metrics/classification/fn.recall_score.html
[70]: https://docs.rs/ferrolearn-metrics/latest/ferrolearn_metrics/classification/fn.f1_score.html
[71]: https://docs.rs/ferrolearn-metrics/latest/ferrolearn_metrics/classification/fn.confusion_matrix.html
[72]: https://docs.rs/ferrolearn-metrics/latest/ferrolearn_metrics/classification/fn.roc_auc_score.html
[73]: https://docs.rs/ferrolearn-metrics/latest/ferrolearn_metrics/classification/fn.log_loss.html
[74]: https://docs.rs/ferrolearn-metrics/latest/ferrolearn_metrics/regression/fn.mean_absolute_error.html
[75]: https://docs.rs/ferrolearn-metrics/latest/ferrolearn_metrics/regression/fn.mean_squared_error.html
[76]: https://docs.rs/ferrolearn-metrics/latest/ferrolearn_metrics/regression/fn.root_mean_squared_error.html
[77]: https://docs.rs/ferrolearn-metrics/latest/ferrolearn_metrics/regression/fn.r2_score.html
[78]: https://docs.rs/ferrolearn-metrics/latest/ferrolearn_metrics/regression/fn.mean_absolute_percentage_error.html
[79]: https://docs.rs/ferrolearn-metrics/latest/ferrolearn_metrics/clustering/fn.silhouette_score.html
[80]: https://docs.rs/ferrolearn-metrics/latest/ferrolearn_metrics/clustering/fn.adjusted_rand_score.html
[81]: https://docs.rs/ferrolearn-metrics/latest/ferrolearn_metrics/clustering/fn.adjusted_mutual_info.html
[82]: https://docs.rs/ferrolearn-datasets/latest/ferrolearn_datasets/toy/fn.load_iris.html
[83]: https://docs.rs/ferrolearn-datasets/latest/ferrolearn_datasets/toy/fn.load_diabetes.html
[84]: https://docs.rs/ferrolearn-datasets/latest/ferrolearn_datasets/toy/fn.load_wine.html
[85]: https://docs.rs/ferrolearn-datasets/latest/ferrolearn_datasets/generators/fn.make_blobs.html
[86]: https://docs.rs/ferrolearn-datasets/latest/ferrolearn_datasets/generators/fn.make_classification.html
[87]: https://docs.rs/ferrolearn-datasets/latest/ferrolearn_datasets/generators/fn.make_regression.html
[88]: https://docs.rs/ferrolearn-io/latest/ferrolearn_io/
[89]: https://docs.rs/ferrolearn-sparse/latest/ferrolearn_sparse/
[90]: https://docs.rs/ferrolearn-core/latest/ferrolearn_core/pipeline/struct.Pipeline.html
[91]: https://docs.rs/ferrolearn-core/latest/ferrolearn_core/backend_faer/struct.NdarrayFaerBackend.html
[92]: https://docs.rs/ferrolearn/latest/ferrolearn/
[93]: https://docs.rs/ferrolearn-core/latest/ferrolearn_core/
[94]: https://docs.rs/ferrolearn-linear/latest/ferrolearn_linear/
[95]: https://docs.rs/ferrolearn-tree/latest/ferrolearn_tree/
[96]: https://docs.rs/ferrolearn-neighbors/latest/ferrolearn_neighbors/
[97]: https://docs.rs/ferrolearn-bayes/latest/ferrolearn_bayes/
[98]: https://docs.rs/ferrolearn-cluster/latest/ferrolearn_cluster/
[99]: https://docs.rs/ferrolearn-decomp/latest/ferrolearn_decomp/
[100]: https://docs.rs/ferrolearn-preprocess/latest/ferrolearn_preprocess/
[101]: https://docs.rs/ferrolearn-metrics/latest/ferrolearn_metrics/
[102]: https://docs.rs/ferrolearn-model-sel/latest/ferrolearn_model_sel/
[103]: https://docs.rs/ferrolearn-datasets/latest/ferrolearn_datasets/
[104]: https://docs.rs/ferrolearn-io/latest/ferrolearn_io/
[105]: https://docs.rs/ferrolearn-sparse/latest/ferrolearn_sparse/
[106]: https://docs.rs/ferrolearn-core/latest/ferrolearn_core/traits/trait.Fit.html
[107]: https://docs.rs/ferrolearn-core/latest/ferrolearn_core/traits/trait.Predict.html
[108]: https://docs.rs/ferrolearn-core/latest/ferrolearn_core/traits/trait.Transform.html
[109]: https://docs.rs/ferrolearn-core/latest/ferrolearn_core/traits/trait.PartialFit.html
[110]: https://docs.rs/ferrolearn-core/latest/ferrolearn_core/traits/trait.FitTransform.html
