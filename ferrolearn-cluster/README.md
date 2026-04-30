# ferrolearn-cluster

Clustering algorithms for the [ferrolearn](https://crates.io/crates/ferrolearn)
machine learning framework. Validated against scikit-learn 1.8.0 with
**exact ARI parity on every measured estimator** — see the
[workspace BENCHMARKS.md](../BENCHMARKS.md).

## Algorithms

| Model | Description |
|-------|-------------|
| `KMeans` | K-Means with **Greedy KMeans++** initialisation (matches sklearn's `_kmeans_plusplus`) |
| `MiniBatchKMeans` | Mini-batch K-Means; sklearn 1.4+ defaults (`batch_size=1024`, `max_iter=100`, `tol=0`) |
| `BisectingKMeans` | Hierarchical bisecting K-Means |
| `DBSCAN` | Density-based clustering — discovers clusters of arbitrary shape |
| `OPTICS` | Ordering Points To Identify the Clustering Structure |
| `HDBSCAN` | Hierarchical density-based clustering |
| `AgglomerativeClustering` | Ward / complete / average / single linkage |
| `Birch` | Memory-efficient hierarchical clustering with CF-Tree |
| `MeanShift` | Non-parametric mode-seeking clustering |
| `SpectralClustering` | Graph-Laplacian eigenmap clustering |
| `AffinityPropagation` | Message-passing clustering |
| `FeatureAgglomeration` | Hierarchical clustering of features (transformer) |
| `GaussianMixture` | Gaussian Mixture Model via EM (full / tied / diag / spherical covariance) with **Greedy KMeans++** init + `reg_covar=1e-6` M-step regularisation |
| `BayesianGaussianMixture` | Variational-Bayes GMM |
| `LabelPropagation` / `LabelSpreading` | Semi-supervised graph propagation |

## Example

```rust
use ferrolearn_cluster::{KMeans, FittedKMeans};
use ferrolearn_core::{Fit, Predict};
use ndarray::array;

let x = array![
    [1.0_f64, 2.0], [1.5, 1.8], [1.2, 2.2],
    [5.0, 6.0], [5.5, 5.8], [5.2, 6.2],
];

let model = KMeans::<f64>::new(2).with_max_iter(100);
let fitted = model.fit(&x, &()).unwrap();

let labels = fitted.predict(&x).unwrap();
let distances = fitted.transform(&x).unwrap();
```

## sklearn parity highlights (0.3.0)

- `KMeans`, `MiniBatchKMeans`, `GaussianMixture` all upgraded to
  **Greedy KMeans++** initialisation (Arthur & Vassilvitskii 2007 with
  `2 + log(k)` trial selection — matches sklearn's `_kmeans_plusplus`).
- `MiniBatchKMeans` defaults switched to sklearn 1.4+ values
  (`batch_size 100 → 1024`, `max_iter 300 → 100`, `tol 1e-4 → 0.0`).
- `GaussianMixture` M-step now adds `reg_covar = 1e-6` to component
  covariance diagonals, matching scikit-learn.
- Result: **mean Δ ARI = 0.0000 across all 15 paired bench runs**.

## License

Licensed under either of [Apache License, Version 2.0](../LICENSE-APACHE) or
[MIT License](../LICENSE-MIT) at your option.
