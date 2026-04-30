# ferrolearn-neighbors

Nearest-neighbor models for the
[ferrolearn](https://crates.io/crates/ferrolearn) machine learning framework.
Validated against scikit-learn 1.8.0 head-to-head — see the
[workspace BENCHMARKS.md](../BENCHMARKS.md).

## Algorithms

| Model | Description |
|-------|-------------|
| `KNeighborsClassifier` | Majority-vote classification by k nearest neighbors |
| `KNeighborsRegressor` | (Weighted) mean of k nearest neighbors |
| `RadiusNeighborsClassifier` / `RadiusNeighborsRegressor` | All neighbors within a fixed radius |
| `NearestCentroid` | Classify by nearest class centroid |
| `NearestNeighbors` | Unsupervised nearest-neighbors lookup / graph |
| `LocalOutlierFactor` | Density-based outlier / anomaly detection |

Plus graph utilities: `kneighbors_graph`, `radius_neighbors_graph`,
`sort_graph_by_row_values`.

## Spatial indexing

- **KD-Tree** — efficient nearest-neighbor search for low-dimensional data (d ≲ 20)
- **Ball Tree** — metric tree for higher-dimensional or non-Euclidean data
- **Brute force** — exhaustive search fallback

The algorithm is selected automatically based on data dimensionality, or can
be set explicitly via `with_algorithm`.

## Example

```rust
use ferrolearn_neighbors::{KNeighborsClassifier, Weights};
use ferrolearn_core::{Fit, Predict};
use ndarray::{array, Array2};

let x = Array2::from_shape_vec((6, 2), vec![
    1.0, 2.0, 1.5, 1.8, 1.2, 2.2,
    5.0, 6.0, 5.5, 5.8, 5.2, 6.2,
]).unwrap();
let y = array![0usize, 0, 0, 1, 1, 1];

let model = KNeighborsClassifier::<f64>::new()
    .with_n_neighbors(3)
    .with_weights(Weights::Distance);
let fitted = model.fit(&x, &y).unwrap();
let predictions = fitted.predict(&x).unwrap();
```

## sklearn parity note

ferrolearn's KNN models build the spatial index **eagerly** during `fit()`,
while scikit-learn defers it to first `predict()`. This is a deliberate
trade-off: `fit()` is slower, but repeated predict calls amortise the index
construction across many queries. Accuracy parity with scikit-learn is exact.

## License

Licensed under either of [Apache License, Version 2.0](../LICENSE-APACHE) or
[MIT License](../LICENSE-MIT) at your option.
