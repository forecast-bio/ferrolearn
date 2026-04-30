# ferrolearn-tree

Decision tree and ensemble tree models for the
[ferrolearn](https://crates.io/crates/ferrolearn) machine learning framework.
Validated against scikit-learn 1.8.0 head-to-head — see the
[workspace BENCHMARKS.md](../BENCHMARKS.md) for the full report.

## Algorithms

| Model | Description |
|-------|-------------|
| `DecisionTreeClassifier` / `DecisionTreeRegressor` | CART trees with Gini / entropy / MSE / MAE splitting |
| `ExtraTreeClassifier` / `ExtraTreeRegressor` | Extremely Randomized single trees |
| `RandomForestClassifier` / `RandomForestRegressor` | Bagging ensemble with **per-split** random feature sampling (Breiman 2001), parallel via Rayon |
| `ExtraTreesClassifier` / `ExtraTreesRegressor` | Bagging ensemble of Extra Trees |
| `GradientBoostingClassifier` / `GradientBoostingRegressor` | Sequential gradient boosting |
| `HistGradientBoostingClassifier` / `HistGradientBoostingRegressor` | Histogram-based gradient boosting (256-bin) |
| `AdaBoostClassifier` / `AdaBoostRegressor` | Adaptive Boosting (default `algorithm = SAMME` to match sklearn ≥ 1.4) |
| `BaggingClassifier` / `BaggingRegressor` | Generic bagging meta-estimator |
| `VotingClassifier` / `VotingRegressor` | Hard / soft voting ensembles |
| `IsolationForest` | Outlier / anomaly detection |
| `RandomTreesEmbedding` | Tree-based feature transformation |

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

All tree hyperparameters (`max_depth`, `min_samples_split`,
`min_samples_leaf`, `max_features`, `criterion`, `random_state`, …) are
configurable via builder methods.

## sklearn parity highlights (0.3.0)

- `RandomForest{Classifier,Regressor}` were fixed to do **per-split** feature
  sampling (Breiman 2001) instead of a fixed per-tree subset — closed a
  -16pp accuracy gap at medium scale.
- `AdaBoostClassifier` default changed from `SAMME.R` to `SAMME` to match
  scikit-learn ≥ 1.4 (which deprecated `SAMME.R` in 1.6) — closed a -19pp
  gap at small scale.

## License

Licensed under either of [Apache License, Version 2.0](../LICENSE-APACHE) or
[MIT License](../LICENSE-MIT) at your option.
