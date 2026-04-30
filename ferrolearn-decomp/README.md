# ferrolearn-decomp

Dimensionality reduction and matrix decomposition for the
[ferrolearn](https://crates.io/crates/ferrolearn) machine learning framework.
Validated against scikit-learn 1.8.0 head-to-head — see the
[workspace BENCHMARKS.md](../BENCHMARKS.md).

## Algorithms

### Linear methods

| Model | Description |
|-------|-------------|
| `PCA` | Principal Component Analysis via faer self-adjoint eigensolver |
| `IncrementalPCA` | Out-of-core PCA for large datasets |
| `TruncatedSVD` | Randomized SVD (Halko algorithm) — works on uncentered/sparse data |
| `NMF` / `MiniBatchNMF` | Non-negative Matrix Factorization (coordinate descent and multiplicative update solvers) |
| `SparsePCA` | Sparse PCA via online dictionary learning |
| `FactorAnalysis` | Factor Analysis via EM algorithm |
| `FastICA` | Independent Component Analysis (parallel + deflation) |
| `DictionaryLearning` | Dictionary learning for sparse coding |
| `LDATopic` | Latent Dirichlet Allocation for topic modelling |
| `CCA` (cross_decomposition) | Canonical Correlation Analysis & PLS variants |

### Manifold learning

| Model | Description |
|-------|-------------|
| `KernelPCA` | Non-linear PCA via RBF, polynomial, or sigmoid kernels |
| `Isomap` | Isometric mapping via geodesic distances on a kNN graph |
| `MDS` | Classical and metric Multidimensional Scaling |
| `SpectralEmbedding` | Laplacian Eigenmaps |
| `LLE` | Locally Linear Embedding (standard + modified) |
| `Tsne` | t-distributed Stochastic Neighbor Embedding |
| `Umap` | Uniform Manifold Approximation and Projection |

## Example

```rust
use ferrolearn_decomp::PCA;
use ferrolearn_core::{Fit, Transform};
use ndarray::array;

let x = array![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];

let pca = PCA::<f64>::new(2);
let fitted = pca.fit(&x, &()).unwrap();
let projected = fitted.transform(&x).unwrap();
assert_eq!(projected.ncols(), 2);

// Inspect explained variance
let variance_ratio = fitted.explained_variance_ratio();
```

## sklearn parity note

`PCA`, `IncrementalPCA`, `TruncatedSVD`, and `FactorAnalysis` produce
reconstruction errors numerically identical to scikit-learn (`recon_rel`
ratio = 1.000× across all measured dataset sizes).

## License

Licensed under either of [Apache License, Version 2.0](../LICENSE-APACHE) or
[MIT License](../LICENSE-MIT) at your option.
