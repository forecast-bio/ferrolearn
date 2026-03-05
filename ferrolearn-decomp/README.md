# ferrolearn-decomp

Dimensionality reduction and matrix decomposition for the [ferrolearn](https://crates.io/crates/ferrolearn) machine learning framework.

## Algorithms

### Linear methods

| Model | Description |
|-------|-------------|
| `PCA` | Principal Component Analysis — project onto directions of maximum variance |
| `IncrementalPCA` | Incremental PCA for large datasets that don't fit in memory |
| `TruncatedSVD` | Randomized SVD (Halko algorithm) — works on uncentered/sparse data |
| `NMF` | Non-negative Matrix Factorization (coordinate descent and multiplicative update solvers) |
| `FactorAnalysis` | Factor Analysis via EM algorithm |
| `FastICA` | Independent Component Analysis |

### Manifold learning

| Model | Description |
|-------|-------------|
| `KernelPCA` | Non-linear PCA via RBF, polynomial, or sigmoid kernels |
| `Isomap` | Isometric mapping via geodesic distances on a kNN graph |
| `MDS` | Classical Multidimensional Scaling |
| `SpectralEmbedding` | Laplacian Eigenmaps |
| `LLE` | Locally Linear Embedding |

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

## License

Licensed under either of [Apache License, Version 2.0](../LICENSE-APACHE) or [MIT License](../LICENSE-MIT) at your option.
