# ferrolearn-cluster

Clustering algorithms for the [ferrolearn](https://crates.io/crates/ferrolearn) machine learning framework.

## Algorithms

| Model | Description |
|-------|-------------|
| `KMeans` | K-Means with k-Means++ initialization and parallel assignment via Rayon |
| `MiniBatchKMeans` | Mini-batch variant of K-Means for large datasets |
| `DBSCAN` | Density-based clustering — discovers clusters of arbitrary shape |
| `AgglomerativeClustering` | Hierarchical clustering (Ward, complete, average, single linkage) |
| `GaussianMixture` | Gaussian Mixture Models via EM (full, tied, diagonal, spherical covariance) |
| `MeanShift` | Non-parametric mode-seeking clustering |
| `SpectralClustering` | Graph Laplacian eigenmap clustering |
| `OPTICS` | Ordering Points To Identify the Clustering Structure |

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

// Assign new points to clusters
let labels = fitted.predict(&x).unwrap();

// Get distances to each centroid
let distances = fitted.transform(&x).unwrap();
```

## License

Licensed under either of [Apache License, Version 2.0](../LICENSE-APACHE) or [MIT License](../LICENSE-MIT) at your option.
