# ferrolearn-bench

Benchmarks for the [ferrolearn](https://crates.io/crates/ferrolearn) machine
learning framework. Not published to crates.io.

The crate ships two complementary harnesses:

- **Criterion benchmarks** — statistical timings (with HTML reports) for the
  headline algorithms in each family. Best for tracking regressions on a
  single algorithm across commits.
- **JSON harness binary** — a comprehensive sweep across 60+ estimators that
  measures fit/predict latency *and* fitted-model quality (R², accuracy, ARI,
  reconstruction error). Pairs with a Python `sklearn_bench.py` script that
  emits the same JSON schema, plus a `compare.py` script that renders a
  side-by-side ferrolearn-vs-scikit-learn Markdown report.

## Criterion suites

| Suite              | What it benchmarks                                                              |
| ------------------ | ------------------------------------------------------------------------------- |
| `regressors`       | LinearRegression, Ridge, Lasso, ElasticNet                                      |
| `classifiers`      | LogisticRegression, DecisionTree, RandomForest, KNN, GaussianNB                 |
| `bayes`            | GaussianNB, MultinomialNB, ComplementNB, BernoulliNB                            |
| `neighbors`        | KNN classifier/regressor, RadiusNeighbors, NearestCentroid, LocalOutlierFactor  |
| `transformers`     | StandardScaler, PCA                                                             |
| `decomposition`    | PCA, IncrementalPCA, TruncatedSVD, FactorAnalysis, FastICA, NMF, KernelPCA, SparsePCA |
| `kernel_methods`   | KernelRidge, GaussianProcessRegressor, Nystroem, RBFSampler                     |
| `kernel_regression`| Nadaraya-Watson, LocalPolynomialRegression, kernel weights, hat matrix          |
| `clusterers`       | KMeans                                                                          |
| `metrics`          | accuracy, f1, MSE, R²                                                           |
| `numerical`        | Sparse eig (Lanczos), sparse graph, distributions, optimization, interpolation, quadrature |

```bash
# Run every Criterion suite
cargo bench -p ferrolearn-bench

# One suite
cargo bench -p ferrolearn-bench --bench regressors
```

Results are written to `target/criterion/` with HTML reports.

## End-to-end ferrolearn vs scikit-learn report

```bash
# 1. Run the Rust harness — emits JSON of fit/predict timings + quality
#    metrics for every supported estimator.
cargo run --release -p ferrolearn-bench --bin harness > ferrolearn_bench.json

# 2. Run the matching sklearn benchmark — emits the same JSON schema for the
#    sklearn equivalents.
python3 ferrolearn-bench/sklearn_bench.py > sklearn_bench.json

# 3. Render the side-by-side Markdown comparison report.
python3 ferrolearn-bench/compare.py ferrolearn_bench.json sklearn_bench.json > REPORT.md
```

The harness exercises all seven families:

| Family       | Algorithms                                                                                                                                                            | Quality metric                  |
| ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------- |
| `regressor`  | LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, ARDRegression, HuberRegressor, QuantileRegressor, DT/RF/ExtraTrees regressor, GBR, HistGBR, KNN-R, KernelRidge | R² on hold-out                  |
| `classifier` | LogisticRegression, RidgeClassifier, LinearSVC, QDA, GaussianNB/MultinomialNB/BernoulliNB/ComplementNB, DT/ExtraTree/RF/ExtraTrees/AdaBoost/Bagging/GB/HistGB classifier, KNN, NearestCentroid | accuracy on hold-out            |
| `cluster`    | KMeans, MiniBatchKMeans, BisectingKMeans, GaussianMixture, AgglomerativeClustering, SpectralClustering, DBSCAN, Birch, MeanShift                                     | adjusted Rand index             |
| `decomp`     | PCA, IncrementalPCA, TruncatedSVD, FactorAnalysis, FastICA, KernelPCA, SparsePCA, NMF                                                                                 | relative Frobenius reconstruction error (where supported) |
| `preprocess` | StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, PowerTransformer, KBinsDiscretizer, Normalizer, PolynomialFeatures                                          | timing only                     |
| `kernel`     | KernelRidge, Nystroem, RBFSampler                                                                                                                                     | R² where applicable             |
| `outlier`    | IsolationForest                                                                                                                                                       | timing only                     |

### Caveat: cross-library accuracy

Both harnesses use their **own** library's `make_*` data generators with the
same seed. These generators are not bit-identical (sklearn's
`make_classification` mixes informative/redundant/repeated features with class
flipping; ferrolearn's is simpler Gaussian clusters). So **timings are
directly comparable** (same shape, same dtype, same workload), but absolute
quality numbers reflect each library's data, not a shared benchmark. To
compare quality on identical data, persist a single dataset to disk and load
it from both harnesses.

## License

Licensed under either of [Apache License, Version
2.0](../LICENSE-APACHE) or [MIT License](../LICENSE-MIT) at your option.
