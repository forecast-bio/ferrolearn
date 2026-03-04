# Feature: Phase 4 — Beyond Scikit-Learn

## Summary
Extend ferrolearn past scikit-learn parity with GPU acceleration, online/streaming learning, model calibration, semi-supervised learning, advanced column transformations, and formal verification of core primitives. This phase differentiates ferrolearn from a "sklearn clone" into a library that leverages Rust's unique strengths: zero-cost GPU abstraction, compile-time correctness guarantees, and machine-verified numerical properties.

## Requirements

### GPU Backend

- REQ-1: `CudaBackend` implementation of the `Backend` trait (behind `cuda` feature flag) using `cudarc` for CUDA FFI, delegating GEMM to cuBLAS and SVD/eigendecomposition to cuSOLVER
- REQ-2: `WgpuBackend` implementation of the `Backend` trait (behind `wgpu` feature flag) using `wgpu` compute shaders for cross-platform GPU support (Vulkan/Metal/DX12/WebGPU)
- REQ-3: Automatic host-device memory transfer management: data stays on GPU between pipeline steps when using a GPU backend, only transferring back on explicit `.to_host()` or when results are consumed
- REQ-4: GPU-accelerated k-Means with data-parallel assignment step and reduction-based centroid update
- REQ-5: GPU-accelerated Random Forest inference (batch prediction on GPU)
- REQ-6: GPU-accelerated PCA via cuSOLVER's truncated SVD (CUDA) or custom WGSL compute shader (WGPU)

### Online/Streaming Learning

- REQ-7: `PartialFit` trait for incremental learning on data batches without refitting from scratch
- REQ-8: `SGDClassifier` and `SGDRegressor` implementing `PartialFit` with configurable loss functions (hinge, log_loss, squared_error, huber) and learning rate schedules (constant, optimal, invscaling, adaptive)
- REQ-9: `MiniBatchKMeans` implementing `PartialFit` for streaming clustering
- REQ-10: `IncrementalPCA` implementing `PartialFit` for streaming dimensionality reduction
- REQ-11: Streaming data adapter that reads from an iterator of batches and feeds them to `PartialFit` models

### Calibration and Semi-Supervised

- REQ-12: `CalibratedClassifierCV` that wraps any probabilistic classifier and calibrates its `predict_proba` output using Platt scaling (sigmoid) or isotonic regression, with cross-validation
- REQ-13: `LabelPropagation` and `LabelSpreading` semi-supervised classifiers that propagate labels through a similarity graph
- REQ-14: `SelfTrainingClassifier` meta-estimator that iteratively assigns pseudo-labels to unlabeled data and retrains

### Column Transformer

- REQ-15: `ColumnTransformer` that applies different transformers to different column subsets, analogous to scikit-learn's `ColumnTransformer`, with ergonomic column selection (by index, by name when using Polars, by dtype)
- REQ-16: `make_column_transformer()` convenience constructor
- REQ-17: `ColumnTransformer` integration with the typed pipeline for compile-time column type checking when input schema is known

### Remaining Algorithms

- REQ-18: UMAP in `ferrolearn-decomp` (implementation following the original McInnes et al. algorithm, checking license compatibility — BSD-3 for the algorithm, but the reference implementation has specific terms)
- REQ-19: LDA (Latent Dirichlet Allocation) topic model in `ferrolearn-decomp` with online variational Bayes solver
- REQ-20: Manifold learning: Isomap, Locally Linear Embedding, MDS, Spectral Embedding in `ferrolearn-decomp`
- REQ-21: PMML export in `ferrolearn-io` (behind `pmml` feature flag) for enterprise interoperability
- REQ-22: Isotonic Regression, RANSAC, Huber Regressor, Bayesian Ridge in `ferrolearn-linear`
- REQ-23: Mean Shift, Spectral Clustering, OPTICS, Birch in `ferrolearn-cluster`
- REQ-24: Factor Analysis, ICA, Dictionary Learning in `ferrolearn-decomp`

### Formal Verification

- REQ-25: Prusti pre/postcondition annotations on all metric functions in `ferrolearn-metrics` verifying output range guarantees (e.g., `accuracy_score` returns `[0.0, 1.0]`)
- REQ-26: Prusti structural invariant verification on `CsrMatrix<T>` proving `indptr.len() == n_rows + 1`, monotonicity of `indptr`, and valid column indices
- REQ-27: Document the compile-time guarantee (unfitted model cannot call `predict`) as a formal proof carried by Rust's type system

### Performance

- REQ-28: `criterion` benchmarks for all P0 algorithms across small (100x10), medium (10kx100), and large (100kx1000) datasets in both `f32` and `f64`
- REQ-29: Performance targets: match or exceed scikit-learn+NumPy throughput on CPU for all P0 algorithms; achieve >= 5x speedup on GPU for algorithms in REQ-4/5/6 on datasets > 10k samples

## Acceptance Criteria

- [ ] AC-1: `CudaBackend::gemm` on 10000x1000 matrices achieves >= 5x speedup over `NdarrayFaerBackend` (NVIDIA GPU with CUDA 12+)
- [ ] AC-2: `WgpuBackend::gemm` on 10000x1000 matrices achieves >= 3x speedup over `NdarrayFaerBackend` on a discrete GPU
- [ ] AC-3: k-Means with `CudaBackend` on 100k samples x 100 features trains in < 50% of CPU time
- [ ] AC-4: A pipeline using `CudaBackend` keeps data on GPU between scaler and PCA steps — verified by memory transfer profiling (no redundant host<->device copies)
- [ ] AC-5: `SGDClassifier` with `partial_fit` on 10 batches of 1000 samples produces a model with accuracy within 2% of the equivalent full `fit` on all 10000 samples
- [ ] AC-6: `CalibratedClassifierCV` improves Brier score of an uncalibrated SVC on a held-out set
- [ ] AC-7: `LabelPropagation` on a dataset with 10% labeled samples achieves accuracy within 5% of fully supervised training
- [ ] AC-8: `ColumnTransformer` with `StandardScaler` on numeric columns and `OneHotEncoder` on categorical columns produces correct output verified against scikit-learn
- [ ] AC-9: Prusti verifies all metric function contracts without errors (on a compatible nightly toolchain)
- [ ] AC-10: Prusti verifies `CsrMatrix` structural invariants without errors
- [ ] AC-11: ONNX export of a Gradient Boosting model loads and runs correctly in `onnxruntime`
- [ ] AC-12: PMML export of LinearRegression validates against a PMML schema checker
- [ ] AC-13: `criterion` benchmark suite runs and produces HTML reports via `cargo bench`
- [ ] AC-14: All P0 algorithms meet or exceed scikit-learn throughput in CPU benchmarks

## Architecture

### GPU Backend — CUDA

File: `ferrolearn-core/src/backend_cuda.rs` (behind `cuda` feature flag)

Dependencies: `cudarc` (safe CUDA bindings), providing access to cuBLAS (GEMM, TRSM), cuSOLVER (SVD, eigendecomposition, Cholesky), and cuSPARSE (sparse matrix operations).

```rust
pub struct CudaBackend {
    device: Arc<cudarc::driver::CudaDevice>,
}
```

GPU memory is managed via `CudaSlice<F>` from cudarc. A `GpuArray2<F>` wrapper provides shape information alongside the device memory. The `Backend` trait methods accept and return `GpuArray2` when the backend is `CudaBackend`.

### GPU Backend — WGPU

File: `ferrolearn-core/src/backend_wgpu.rs` (behind `wgpu` feature flag)

Uses WGSL compute shaders for matrix operations. Less optimized than CUDA (no cuBLAS-level tuning) but works on all GPU vendors and in the browser via WebGPU.

Matrix multiplication shader uses tiled shared-memory approach (tile size = 16x16) for reasonable performance without vendor-specific optimization.

### Online Learning Trait

File: `ferrolearn-core/src/traits.rs`

```rust
pub trait PartialFit<X, Y>: Sized {
    type FitResult: Predict<X> + PartialFit<X, Y>;
    type Error: std::error::Error;

    fn partial_fit(self, x: &X, y: &Y) -> Result<Self::FitResult, Self::Error>;
}
```

Key difference from `Fit`: `partial_fit` is callable on both unfitted and fitted models (the fitted model also implements `PartialFit`). This allows chaining: `model.partial_fit(&batch1)?.partial_fit(&batch2)?.predict(&x)?`.

### SGD Implementation

File: `ferrolearn-linear/src/sgd.rs`

Supports multiple loss functions via a `Loss` trait:

```rust
pub trait Loss: Send + Sync {
    fn loss(&self, y_true: f64, y_pred: f64) -> f64;
    fn gradient(&self, y_true: f64, y_pred: f64) -> f64;
}
```

Built-in losses: `Hinge`, `LogLoss`, `SquaredError`, `Huber`, `ModifiedHuber`, `EpsilonInsensitive`.

Learning rate schedules: `Constant`, `Optimal` (1 / (alpha * t)), `InvScaling` (eta0 / t^power_t), `Adaptive` (halves when loss doesn't improve for `n_iter_no_change` epochs).

### ColumnTransformer

File: `ferrolearn-preprocess/src/column_transformer.rs`

```rust
pub struct ColumnTransformer {
    transformers: Vec<(String, Box<dyn PipelineStep>, ColumnSelector)>,
    remainder: Remainder, // Drop, Passthrough, or a transformer
}

pub enum ColumnSelector {
    Indices(Vec<usize>),
    Names(Vec<String>),  // Requires named columns (Polars/Arrow)
    DType(DTypeFilter),  // Numeric, Categorical, etc.
}
```

During `fit`, each transformer is fit on its selected columns. During `transform`, the outputs are horizontally concatenated. Column ordering follows the transformer declaration order, with remainder columns appended.

### Formal Verification Strategy

Prusti verification is attempted on the latest compatible nightly. Per Q10, CI is staged: informational during development, blocking on release tags only. Since Prusti typically lags behind stable Rust by 6-12 months, this prevents toolchain issues from blocking daily work while ensuring verified properties hold at release time.

Verified components:
1. **Metric functions**: Pure functions with clear mathematical contracts
2. **Sparse matrix invariants**: Structural properties that must hold after construction/mutation
3. **Type-system proofs**: Documented in API docs, verified by `trybuild` compile-fail tests

For components where Prusti is insufficient, Lean 4 reference specifications serve as the mathematical ground truth, with correspondence verified by fixture tests.

## Resolved Questions

### Q8: Clean-room UMAP under Apache-2.0/MIT
**Decision:** Implement UMAP as a clean-room implementation from the McInnes et al. (2018) paper "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction" (arXiv:1802.03426). The algorithm is described in a peer-reviewed paper with full mathematical specification — this is sufficient for an independent implementation. The reference Python implementation is BSD-3-Clause, which is compatible, but we do not port code from it. The ferrolearn implementation is written from the paper's mathematical description only, licensed Apache-2.0/MIT consistent with the rest of the library.

### Q9: Associated type on Backend trait for GPU arrays
**Decision:** The `Backend` trait defines an associated type `type Array2<F>` that each backend provides. `NdarrayFaerBackend::Array2<F>` is `ndarray::Array2<F>`. `CudaBackend::Array2<F>` is `CudaArray2<F>` (wrapping `cudarc::CudaSlice<F>` + shape). `WgpuBackend::Array2<F>` is `WgpuArray2<F>` (wrapping a GPU buffer + shape). Algorithms are generic over `B: Backend` and use `B::Array2<F>`, enabling data to stay on GPU across pipeline steps without redundant transfers. A `ToHost` trait provides explicit device-to-host conversion when results are consumed. This is more complex to implement but is the entire point of the Backend abstraction — without it, GPU backends provide no pipeline-level benefit.

### Q10: Staged Prusti CI
**Decision:** Prusti CI is informational during development and blocking on release tags only. During normal development, the Prusti verification job runs in CI and reports results as annotations but does not block PR merges. On release tags (e.g., `v1.0.0`), the Prusti job becomes a required check. This balances verification rigor with practical development velocity — Prusti's nightly pinning and occasional regressions would otherwise block unrelated work. The staging boundary is enforced by CI configuration (separate workflows for `push` vs `tag`).

## Out of Scope
- Deep learning (explicitly a non-goal per design doc Section 3)
- Python bindings via pyo3 (separate crate/project)
- Data visualization (delegate to `plotters`)
- Replacing ndarray or faer as primitives
