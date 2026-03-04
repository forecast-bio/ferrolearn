# Feature: Phase 1 — Foundation

## Summary
Bootstrap the ferrolearn workspace with core traits, error types, dense/sparse matrix integration, basic preprocessing, linear models, evaluation metrics, cross-validation, and a dynamic-dispatch pipeline. This phase produces a compilable, testable library that proves the architecture works end-to-end: a user can load data, preprocess it, fit a linear model in a pipeline, evaluate it with cross-validation, and get metric scores — all with type-safe compile-time unfitted/fitted separation.

## Requirements

- REQ-1: Cargo workspace with 13 crates (original 11 plus `ferrolearn-neighbors` and `ferrolearn-bayes`), all compiling with `cargo build` on Rust 1.85+ (2024 edition)
- REQ-2: Core traits `Fit<X, Y>`, `Predict<X>`, `Transform<X>`, `FitTransform<X>` in `ferrolearn-core` with compile-time enforcement that `predict()` cannot be called on an unfitted model
- REQ-3: `FerroError` enum with `ShapeMismatch`, `InsufficientSamples`, `ConvergenceFailure`, `InvalidParameter`, `NumericalInstability`, `IoError`, `SerdeError` variants, each carrying diagnostic context
- REQ-4: `Dataset` trait with implementations for `ndarray::Array2<f32>` and `ndarray::Array2<f64>`, supporting `n_samples()`, `n_features()`, `is_sparse()`
- REQ-5: Sparse matrix types `CsrMatrix<T>`, `CscMatrix<T>`, `CooMatrix<T>` in `ferrolearn-sparse` with format conversion, slicing, and arithmetic operations
- REQ-6: `StandardScaler`, `MinMaxScaler`, `RobustScaler` implementing `FitTransform<Array2<F>>` in `ferrolearn-preprocess`, generic over `F: Float`
- REQ-7: `OneHotEncoder` and `LabelEncoder` in `ferrolearn-preprocess`
- REQ-8: `LinearRegression` (OLS via QR), `Ridge` (L2), `Lasso` (coordinate descent), `LogisticRegression` (L-BFGS) in `ferrolearn-linear`, all implementing `Fit` and producing fitted types implementing `Predict`
- REQ-9: All classification metrics (accuracy, precision, recall, F1, ROC AUC, confusion matrix, log loss) and all regression metrics (MAE, MSE, RMSE, R², MAPE, explained variance) in `ferrolearn-metrics`
- REQ-10: `train_test_split`, `KFold`, `StratifiedKFold`, and `cross_val_score` in `ferrolearn-model-sel`
- REQ-11: Dynamic-dispatch `Pipeline` in `ferrolearn-core` that composes transformers and a final estimator, implementing `Fit` and producing a fitted pipeline implementing `Predict`
- REQ-12: Oracle fixture tests (Layer 1) for all four linear models against scikit-learn golden fixtures, with ULP-based tolerance
- REQ-13: Property-based tests (Layer 2) with minimum 8 invariant tests per algorithm using `proptest`
- REQ-14: Fuzz targets (Layer 5) for all public `fit`, `transform`, `predict` entry points using `cargo-fuzz`
- REQ-15: All algorithms generic over `F: num_traits::Float + Send + Sync` supporting both `f32` and `f64`

## Acceptance Criteria

- [ ] AC-1: `cargo build --workspace` succeeds on Rust 1.85 stable with zero warnings
- [ ] AC-2: `cargo test --workspace` passes with all oracle fixtures, property tests, and unit tests green
- [ ] AC-3: Calling `predict()` on an unfitted model produces a compile error, not a runtime error — verified by `trybuild` compile-fail tests
- [ ] AC-4: `Pipeline::new().step("scaler", StandardScaler::new()).step("clf", LogisticRegression::new()).fit(&x, &y)?.predict(&x_test)?` compiles and runs correctly
- [ ] AC-5: `cross_val_score(&LogisticRegression::new(), &x, &y, &KFold::new(5), accuracy_score)` returns 5 scores matching scikit-learn within 4 ULPs
- [ ] AC-6: `StandardScaler` fit-then-transform produces zero-mean, unit-variance columns (verified by proptest across random matrices)
- [ ] AC-7: `CsrMatrix::from_coo(&coo).to_dense()` round-trips correctly for all valid inputs
- [ ] AC-8: `LinearRegression` coefficients match scikit-learn OLS on the fixture suite within 4 ULPs
- [ ] AC-9: `LogisticRegression` predictions (class labels) are exact matches with scikit-learn fixtures; probabilities within 4 ULPs
- [ ] AC-10: All fuzz targets run for a minimum of 1 hour without panics on seed corpus
- [ ] AC-11: `cargo clippy --workspace -- -D warnings` passes
- [ ] AC-12: `cargo doc --workspace --no-deps` builds without warnings; every public item has a doc comment

## Architecture

### Workspace Layout

```
ferrolearn/
├── Cargo.toml                    # Workspace root
├── ferrolearn/Cargo.toml         # Main crate — re-exports
├── ferrolearn-core/              # Traits, Dataset, Pipeline, FerroError
├── ferrolearn-linear/            # Linear/Ridge/Lasso/LogisticRegression
├── ferrolearn-tree/              # (empty stub — Phase 2)
├── ferrolearn-neighbors/         # (empty stub — Phase 2, kNN)
├── ferrolearn-bayes/             # (empty stub — Phase 2, Naive Bayes)
├── ferrolearn-cluster/           # (empty stub — Phase 2)
├── ferrolearn-decomp/            # (empty stub — Phase 2)
├── ferrolearn-preprocess/        # Scalers, Encoders
├── ferrolearn-metrics/           # Classification + Regression metrics
├── ferrolearn-model-sel/         # KFold, StratifiedKFold, cross_val_score
├── ferrolearn-sparse/            # CsrMatrix, CscMatrix, CooMatrix
├── ferrolearn-datasets/          # (empty stub — Phase 2)
├── ferrolearn-io/                # (empty stub — Phase 2)
├── fixtures/                     # Golden JSON fixtures from scikit-learn
├── scripts/generate_fixtures.py  # Fixture generation script
└── fuzz/                         # Fuzz targets
```

### Dependency Versions (pinned)

| Crate | Version | Purpose |
|-------|---------|---------|
| ndarray | 0.17 | Dense array type (Array2, ArrayView2, new ArrayRef) |
| faer | 0.24 | QR decomposition, SVD, linear solvers |
| sprs | 0.11 | CSR/CSC sparse matrix storage |
| rayon | 1.11 | Parallel iteration (default feature) |
| serde | 1.0 | Serialization derives (default feature) |
| num-traits | 0.2 | `Float`, `Zero`, `One` trait bounds |
| thiserror | 2.0 | `#[derive(Error)]` for FerroError |
| approx | 0.5 | `assert_abs_diff_eq!` in tests |
| float-cmp | 0.10 | ULP-based comparison in oracle tests |
| proptest | 1.9 | Property-based testing |
| criterion | 0.8 | Benchmarks (dev-dependency) |
| libfuzzer-sys | 0.4 | Fuzz targets |

### Core Trait Design

The `Fit` trait takes `&self` (configuration/hyperparameters) and returns a *new fitted type* — the unfitted config struct never implements `Predict`. This is the central type-safety guarantee.

```
[StandardScaler]  --fit(&x)--> [FittedStandardScaler] --transform(&x)--> Array2<F>
[LogisticRegression] --fit(&x, &y)--> [FittedLogisticRegression] --predict(&x)--> Array1<usize>
```

The `Pipeline` stores steps as `Box<dyn PipelineStep>` for heterogeneous composition. A `PipelineStep` trait unifies transformers and the final estimator behind a common interface with runtime method dispatch.

### Numeric Generics Strategy

All algorithms are generic over `F: Float + Send + Sync + 'static` from `num-traits`. Internally, algorithms that need linear algebra call into `faer` which natively supports both `f32` and `f64`. The `ndarray` `Array2<F>` is the primary input/output type.

### Error Handling

Every public function returns `Result<T, FerroError>`. The `#[non_exhaustive]` attribute on `FerroError` allows adding variants without breaking semver. `thiserror` 2.0 derives `Display` and `Error` implementations.

### Testing Infrastructure

- **Fixture generation**: `scripts/generate_fixtures.py` runs scikit-learn with fixed seeds, outputs JSON to `fixtures/`. Each fixture includes input data, hyperparameters, fitted parameters, and predictions.
- **Oracle tests**: Load fixtures in Rust tests, fit the ferrolearn model with identical parameters, compare outputs using `float_cmp::assert_approx_eq!` with ULP tolerances.
- **Property tests**: `proptest` strategies generate random valid matrices and label vectors. Invariants are checked (e.g., scaler produces zero mean, classifier probabilities sum to 1.0).
- **Fuzz targets**: One target per public `fit`/`transform`/`predict` function. Contract: never panic on any input — must return `Ok` or `Err`.
- **Compile-fail tests**: `trybuild` crate verifies that calling `predict()` on unfitted types fails at compile time.

### Linear Algebra Dispatch

- `LinearRegression`: QR decomposition via `faer::linalg::qr`
- `Ridge`: Closed-form solution `(X^T X + αI)^{-1} X^T y` via `faer` Cholesky
- `Lasso`: Coordinate descent (custom implementation, no external solver)
- `LogisticRegression`: L-BFGS optimization (custom implementation following scikit-learn's `_logistic_loss_and_grad`)

### Pipeline Internals

```rust
pub struct Pipeline { steps: Vec<(String, Box<dyn PipelineStep>)> }
pub struct FittedPipeline { steps: Vec<(String, Box<dyn FittedPipelineStep>)> }
```

Each step is either a transformer (implements `FitTransform`) or an estimator (implements `Fit`). The final step must be an estimator. `FittedPipeline` implements `Predict` by chaining transforms through the fitted transformers and calling predict on the fitted estimator.

## Resolved Questions

### Q1: L-BFGS implementation — custom
**Decision:** Custom implementation matching scikit-learn's exact behavior (Wolfe line search, m=10 history). This ensures numerical equivalence with scikit-learn's `_logistic_loss_and_grad` within the 4-ULP budget and avoids external dependencies (C FFI from `lbfgs` crate) or heavy dep trees (`argmin`). The implementation lives in `ferrolearn-linear/src/optim/lbfgs.rs` as an internal module, not a public API.

### Q2: Newtype wrappers over sprs
**Decision:** Own types backed by sprs — `CsrMatrix<T>`, `CscMatrix<T>`, `CooMatrix<T>` are newtype wrappers around `sprs::CsMat`/`sprs::TriMat` with ferrolearn's own API surface. This decouples the public API from sprs internals, allows swapping to faer-sparse or a custom backend later without breaking changes, while leveraging sprs's proven correctness for the underlying storage. `faer-sparse` 0.17.1 is used for factorization (LU, QR, Cholesky) when needed by algorithms.

## Out of Scope
- GPU acceleration (Phase 4)
- Model serialization/persistence (Phase 2)
- Tree-based models, kNN, Naive Bayes, SVM (Phase 2)
- Clustering, PCA, dimensionality reduction (Phase 2)
- Toy datasets and generators (Phase 2)
- `GridSearchCV`, `RandomizedSearchCV` (Phase 2)
- Polars/Arrow integration (Phase 3)
- ONNX export (Phase 3)
- `no_std` support (Phase 3)
- BLAS backend abstraction (Phase 3)
- Compile-time pipeline variant (Phase 3)
- Formal verification via Prusti (Phase 4 stretch goal)
