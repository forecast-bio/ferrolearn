# ferrolearn-tree

Decision tree and ensemble tree models for the [ferrolearn](https://crates.io/crates/ferrolearn) machine learning framework.

## Algorithms

| Model | Description |
|-------|-------------|
| `DecisionTreeClassifier` | CART classification tree with Gini impurity or entropy splitting |
| `DecisionTreeRegressor` | CART regression tree with MSE or MAE splitting |
| `RandomForestClassifier` | Bootstrap-aggregated ensemble with parallel tree building via Rayon |
| `RandomForestRegressor` | Random forest for regression tasks |
| `GradientBoostingClassifier` | Sequential gradient boosting with configurable loss functions |
| `GradientBoostingRegressor` | Gradient boosting for regression (least squares, LAD, Huber) |
| `AdaBoostClassifier` | Adaptive Boosting with SAMME and SAMME.R algorithms |

## Example

```rust
use ferrolearn_tree::{RandomForestClassifier, MaxFeatures};
use ferrolearn_core::{Fit, Predict};
use ndarray::{array, Array2};

let x = Array2::from_shape_vec((6, 2), vec![
    1.0, 2.0, 1.5, 1.8, 1.2, 2.2,
    5.0, 6.0, 5.5, 5.8, 5.2, 6.2,
]).unwrap();
let y = array![0usize, 0, 0, 1, 1, 1];

let model = RandomForestClassifier::<f64>::new()
    .with_n_estimators(100)
    .with_max_features(MaxFeatures::Sqrt);
let fitted = model.fit(&x, &y).unwrap();
let predictions = fitted.predict(&x).unwrap();
```

All tree hyperparameters (max depth, min samples split, min samples leaf, etc.) are configurable via builder methods.

## License

Licensed under either of [Apache License, Version 2.0](../LICENSE-APACHE) or [MIT License](../LICENSE-MIT) at your option.
