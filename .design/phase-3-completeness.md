# Feature: Phase 3 — Completeness

## Summary
Fill remaining algorithm gaps to reach full scikit-learn classical ML parity, add advanced infrastructure (BLAS backend abstraction, ONNX export, Polars/Arrow integration, compile-time pipeline), and complete the verification stack through statistical equivalence benchmarks. After this phase, ferrolearn is feature-complete for a 1.0 release candidate.

## Requirements

### Algorithms

- REQ-1: Gradient Boosting classifier and regressor in `ferrolearn-tree` with histogram-based variant (HistGradientBoosting), configurable learning rate, max depth, n_estimators, subsample ratio
- REQ-2: AdaBoost classifier in `ferrolearn-tree` with SAMME and SAMME.R algorithms
- REQ-3: Gaussian Mixture Models (GMM) in `ferrolearn-cluster` with EM algorithm, supporting full/tied/diag/spherical covariance types, BIC/AIC model selection
- REQ-4: HDBSCAN in `ferrolearn-cluster` with minimum cluster size parameter and soft clustering (probability scores)
- REQ-5: Agglomerative Clustering in `ferrolearn-cluster` with Ward, complete, average, single linkage
- REQ-6: t-SNE in `ferrolearn-decomp` with Barnes-Hut approximation for O(n log n) on large datasets
- REQ-7: NMF (Non-negative Matrix Factorization) in `ferrolearn-decomp` with multiplicative update and coordinate descent solvers
- REQ-8: Kernel PCA in `ferrolearn-decomp` with RBF, polynomial, sigmoid kernels via pluggable `Kernel` trait
- REQ-9: Kernel SVM in `ferrolearn-linear` using the `Kernel` trait from REQ-8
- REQ-10: All imputers (`SimpleImputer`, `KNNImputer`, `IterativeImputer`) in `ferrolearn-preprocess`
- REQ-11: Full feature selection suite (`VarianceThreshold`, `SelectKBest`, `SelectPercentile`, `RFE`, `RFECV`, `SelectFromModel`) in `ferrolearn-preprocess`
- REQ-12: Remaining scalers (`MaxAbsScaler`, `Normalizer`, `PowerTransformer`, `QuantileTransformer`) in `ferrolearn-preprocess`
- REQ-13: Remaining encoders (`OrdinalEncoder`, `TargetEncoder`, `BinaryEncoder`) in `ferrolearn-preprocess`
- REQ-14: Feature engineering transformers (`PolynomialFeatures`, `SplineTransformer`, `KBinsDiscretizer`, `Binarizer`, `FunctionTransformer`) in `ferrolearn-preprocess`
- REQ-15: `TimeSeriesSplit` in `ferrolearn-model-sel`
- REQ-16: `HalvingGridSearchCV` (successive halving) in `ferrolearn-model-sel`
- REQ-17: Remaining synthetic generators (`make_swiss_roll`, `make_s_curve`, `make_sparse_uncorrelated`) in `ferrolearn-datasets`
- REQ-18: Remaining toy datasets (Digits, Linnerud, Olivetti Faces) in `ferrolearn-datasets`

### Infrastructure

- REQ-19: Pluggable `Backend` trait in `ferrolearn-core` abstracting matrix operations (gemm, svd, qr, cholesky, eigendecomposition), with `NdarrayFaerBackend` as default implementation
- REQ-20: Optional `BLASBackend` (behind `blas` feature flag) linking system BLAS/LAPACK via `ndarray-linalg` or direct FFI
- REQ-21: ONNX model export in `ferrolearn-io` (behind `onnx` feature flag) for linear models, decision trees, random forests, and gradient boosting — producing valid ONNX protobuf files loadable by `tract` or `onnxruntime`
- REQ-22: `polars::DataFrame` as `Dataset` implementation (behind `polars` feature flag) with zero-copy where possible
- REQ-23: `arrow::RecordBatch` as `Dataset` implementation (behind `arrow` feature flag)
- REQ-24: Compile-time type-safe pipeline variant using const generics or type-level lists, available alongside the dynamic-dispatch pipeline
- REQ-25: `no_std` support for `ferrolearn-core` traits and `ferrolearn-metrics` (behind default `std` feature flag; disabling `std` removes I/O and thread-dependent code)

### Verification

- REQ-26: Statistical equivalence benchmark suite (`benchmarks/statistical_equivalence.py`) covering all P0 algorithms across minimum 10 datasets, with Welch's t-test (α=0.05, one-sided)
- REQ-27: Oracle fixtures for all edge cases (minimum input, zero-variance features, extreme scales, sparse inputs) for every algorithm
- REQ-28: Property-based tests complete for all algorithms (minimum 8 invariants each)
- REQ-29: All fuzz targets run for cumulative 24 CPU-hours against the full corpus before 1.0 RC
- REQ-30: Algorithm equivalence documents for all P0 algorithms

## Acceptance Criteria

- [ ] AC-1: HistGradientBoosting on the adult dataset achieves accuracy within 0.5% of scikit-learn's `HistGradientBoostingClassifier`
- [ ] AC-2: GMM with `n_components=3` on Iris produces log-likelihood within 1% of scikit-learn; BIC selection picks the same component count
- [ ] AC-3: HDBSCAN on moons dataset with `min_cluster_size=15` produces identical cluster assignments to the `hdbscan` Python package
- [ ] AC-4: t-SNE on Digits dataset produces a 2D embedding where k-NN accuracy on the embedding exceeds 90% (quality check, not exact match — t-SNE is stochastic)
- [ ] AC-5: NMF reconstruction error on a sparse TF-IDF matrix is within 1% of scikit-learn
- [ ] AC-6: `save_model` + ONNX export of a fitted LogisticRegression, loaded in `tract`, produces identical predictions
- [ ] AC-7: `Pipeline::new().step("scaler", StandardScaler::new()).step("clf", LogisticRegression::new())` works with both `Array2<f64>` and `polars::DataFrame` inputs (when `polars` feature enabled)
- [ ] AC-8: `BLASBackend` with OpenBLAS achieves >= 2x speedup on 100k x 1000 matrix multiplication vs. `NdarrayFaerBackend`
- [ ] AC-9: Compile-time pipeline compiles and runs, with the compiler rejecting pipelines where step output types don't match next step input types
- [ ] AC-10: `cargo build --no-default-features` compiles `ferrolearn-core` and `ferrolearn-metrics` successfully in `no_std` context
- [ ] AC-11: Statistical equivalence benchmarks pass for all P0 algorithms (no statistically significant worse performance than scikit-learn)
- [ ] AC-12: All algorithm equivalence documents reviewed and committed under `docs/algorithm_equivalence/`
- [ ] AC-13: `SimpleImputer(strategy=mean)` on a matrix with NaN values matches scikit-learn output within 4 ULPs
- [ ] AC-14: `RFE` with `LogisticRegression` on Breast Cancer dataset selects the same top-10 features as scikit-learn

## Architecture

### Backend Trait

File: `ferrolearn-core/src/backend.rs`

```rust
pub trait Backend: Send + Sync + 'static {
    fn gemm<F: Float>(a: &Array2<F>, b: &Array2<F>) -> Array2<F>;
    fn svd<F: Float>(a: &Array2<F>) -> Result<(Array2<F>, Array1<F>, Array2<F>), FerroError>;
    fn qr<F: Float>(a: &Array2<F>) -> Result<(Array2<F>, Array2<F>), FerroError>;
    fn cholesky<F: Float>(a: &Array2<F>) -> Result<Array2<F>, FerroError>;
    fn solve<F: Float>(a: &Array2<F>, b: &Array1<F>) -> Result<Array1<F>, FerroError>;
    fn eigh<F: Float>(a: &Array2<F>) -> Result<(Array1<F>, Array2<F>), FerroError>;
}
```

The default `NdarrayFaerBackend` delegates to `faer` 0.24 (pure Rust, no system dependencies). The `BLASBackend` delegates to system BLAS/LAPACK via FFI. Algorithms receive the backend as a type parameter with a default: `struct PCA<B: Backend = NdarrayFaerBackend> { ... }`.

### Gradient Boosting

File: `ferrolearn-tree/src/gradient_boosting.rs`

Standard variant: sequential boosting of decision trees. Each tree fits the negative gradient of the loss function. Supports `least_squares`, `lad`, `huber` losses for regression; `log_loss` for binary/multiclass classification.

Histogram-based variant (`HistGradientBoosting`): bins continuous features into 256 discrete bins, builds histograms at each node for O(n) split finding instead of O(n log n). This is the scikit-learn default as of 1.0+. Uses the "subtraction trick" (child histogram = parent - sibling) to halve histogram computation.

### ONNX Export

File: `ferrolearn-io/src/onnx.rs`

Uses the ONNX protobuf schema (generated via `prost`) to create model graphs. Supported models:
- Linear models → ONNX `LinearRegressor` / `LinearClassifier` ops
- Decision trees → ONNX `TreeEnsembleRegressor` / `TreeEnsembleClassifier` ops
- Random Forest / Gradient Boosting → same tree ensemble ops with aggregation

Validation: after export, load in `tract` (Rust ONNX runtime) and verify predictions match.

### Polars Integration

File: `ferrolearn-core/src/dataset_polars.rs` (behind `polars` feature flag)

```rust
impl Dataset for polars::DataFrame {
    fn n_samples(&self) -> usize { self.height() }
    fn n_features(&self) -> usize { self.width() }
}
```

Conversion to `Array2<f64>` uses Polars' `to_ndarray()` method. For algorithms that can operate on columnar data directly (tree-based models), a `ColumnarDataset` trait provides column-wise access without materializing the full dense matrix.

### Compile-Time Pipeline

File: `ferrolearn-core/src/typed_pipeline.rs`

Uses a recursive type list pattern:

```rust
struct TypedPipeline<Steps> { steps: Steps }
// Steps: (Scaler, (PCA, (LogisticRegression, ())))
// Compiled to a chain of .fit().transform() calls at compile time
```

The compiler verifies that each step's output type matches the next step's input type. This is zero-cost at runtime but requires all types to be known at compile time (no dynamic step addition).

### Statistical Equivalence Infrastructure

```
benchmarks/
├── statistical_equivalence.py   # Python harness
├── ferrolearn_bench/            # Rust binary that outputs JSON scores
│   └── src/main.rs
├── datasets/                    # Cached OpenML datasets
└── results/                     # Machine-readable result artifacts
    └── YYYY-MM-DD.json
```

The Python harness runs both sklearn and ferrolearn_bench on each (algorithm, dataset) pair with 10-fold CV, then applies Welch's t-test. Results are committed as artifacts for regression detection.

## Resolved Questions

### Q6: prost from official ONNX `.proto` schema
**Decision:** Use `prost` to auto-generate Rust types from the official ONNX `.proto` schema files. This guarantees spec correctness by construction and is the standard Rust approach to protobuf. The build cost is one `build.rs` in `ferrolearn-io` that invokes `prost-build`. Hand-written structs would inevitably drift from the ONNX spec, and coupling to tract's types would tie releases to tract's cycle. The generated types are internal — the public API exposes `export_onnx(&model, path)` without leaking protobuf details.

### Q7: Recursive tuple type + `pipeline!()` proc macro
**Decision:** The compile-time pipeline uses a recursive tuple type `(A, (B, (C, ())))` internally for zero-cost type-level composition. A `pipeline!()` proc macro provides ergonomic syntax that expands to the nested type, hiding the nesting from users. Example: `pipeline!(StandardScaler, PCA, LogisticRegression)` expands to the nested tuple type. This avoids const generic limitations (still insufficient for this use case in Rust 1.85) while achieving zero runtime overhead. Error messages from the compiler will reference the expanded types, but the macro's span information helps locate the source.

## Out of Scope
- GPU backend (Phase 4)
- Online/streaming learning (Phase 4)
- Semi-supervised learning (Phase 4)
- Calibration (`CalibratedClassifierCV`) (Phase 4)
- `ColumnTransformer` (Phase 4)
- UMAP (Phase 4 — requires separate licensing consideration)
- PMML export (Phase 4)
- LDA topic model (Phase 4)
- Manifold learning (Isomap, LLE, MDS) — Phase 4, P2 priority
