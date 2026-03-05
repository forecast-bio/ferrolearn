# ferrolearn-neighbors

Nearest neighbor models for the [ferrolearn](https://crates.io/crates/ferrolearn) machine learning framework.

## Algorithms

| Model | Description |
|-------|-------------|
| `KNeighborsClassifier` | Classify by majority vote of k nearest neighbors |
| `KNeighborsRegressor` | Predict as (weighted) mean of k nearest neighbors |

## Spatial indexing

- **KD-Tree** — efficient nearest neighbor search for low-dimensional data (d <= 20)
- **Ball Tree** — metric tree for higher-dimensional or non-Euclidean data
- **Brute force** — exhaustive search fallback

The algorithm is selected automatically based on data dimensionality, or can be set explicitly.

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
    .with_k(3)
    .with_weights(Weights::Distance);
let fitted = model.fit(&x, &y).unwrap();
let predictions = fitted.predict(&x).unwrap();
```

## License

Licensed under either of [Apache License, Version 2.0](../LICENSE-APACHE) or [MIT License](../LICENSE-MIT) at your option.
