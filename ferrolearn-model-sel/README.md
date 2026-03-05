# ferrolearn-model-sel

Model selection utilities for the [ferrolearn](https://crates.io/crates/ferrolearn) machine learning framework.

## Data splitting

| Function / Type | Description |
|-----------------|-------------|
| `train_test_split` | Shuffle and split data into train/test sets |
| `KFold` | K-fold cross-validation splitter |
| `StratifiedKFold` | Stratified K-fold preserving class balance |
| `TimeSeriesSplit` | Time-series aware cross-validation |

## Cross-validation

| Function / Type | Description |
|-----------------|-------------|
| `cross_val_score` | Evaluate a pipeline using cross-validation |

## Hyperparameter search

| Type | Description |
|------|-------------|
| `GridSearchCV` | Exhaustive search over a parameter grid |
| `RandomizedSearchCV` | Randomized search over parameter distributions |
| `HalvingGridSearchCV` | Successive-halving search for efficient tuning |
| `param_grid!` | Macro for building Cartesian-product parameter grids |

## Meta-estimators

| Type | Description |
|------|-------------|
| `CalibratedClassifierCV` | Probability calibration via cross-validation |
| `SelfTrainingClassifier` | Semi-supervised self-training |

## Example

```rust
use ferrolearn_model_sel::{train_test_split, KFold};
use ndarray::{Array1, Array2};

let x = Array2::<f64>::zeros((100, 5));
let y = Array1::<f64>::zeros(100);

// Split 80/20
let (x_train, x_test, y_train, y_test) =
    train_test_split(&x, &y, 0.2, Some(42)).unwrap();

// 5-fold cross-validation
let kf = KFold::new(5);
let folds = kf.split(100);
```

## License

Licensed under either of [Apache License, Version 2.0](../LICENSE-APACHE) or [MIT License](../LICENSE-MIT) at your option.
