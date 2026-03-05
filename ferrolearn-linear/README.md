# ferrolearn-linear

Linear models for the [ferrolearn](https://crates.io/crates/ferrolearn) machine learning framework.

## Algorithms

### Regression

| Model | Description |
|-------|-------------|
| `LinearRegression` | Ordinary Least Squares via QR decomposition |
| `Ridge` | L2-regularized regression via Cholesky decomposition |
| `Lasso` | L1-regularized regression via coordinate descent |
| `ElasticNet` | Combined L1/L2 regularization via coordinate descent |
| `BayesianRidge` | Bayesian ridge with automatic regularization tuning |
| `HuberRegressor` | Robust regression via IRLS with Huber loss |
| `SGDRegressor` | Stochastic gradient descent regressor |

### Classification

| Model | Description |
|-------|-------------|
| `LogisticRegression` | Binary and multiclass classification via L-BFGS |
| `LDA` | Linear Discriminant Analysis |
| `SGDClassifier` | Stochastic gradient descent classifier |

## Example

```rust
use ferrolearn_linear::{Ridge, FittedRidge};
use ferrolearn_core::{Fit, Predict};
use ndarray::array;

let x = array![[1.0_f64, 2.0], [3.0, 4.0], [5.0, 6.0]];
let y = array![1.0, 2.0, 3.0];

let model = Ridge::<f64>::new().with_alpha(1.0);
let fitted = model.fit(&x, &y).unwrap();
let predictions = fitted.predict(&x).unwrap();
```

All models follow the compile-time safety pattern: unfitted structs implement `Fit`, fitted structs implement `Predict`. Calling `predict()` on an unfitted model is a compile error.

## Float generics

All models are generic over `F: Float + Send + Sync + 'static`, supporting both `f32` and `f64`.

## License

Licensed under either of [Apache License, Version 2.0](../LICENSE-APACHE) or [MIT License](../LICENSE-MIT) at your option.
