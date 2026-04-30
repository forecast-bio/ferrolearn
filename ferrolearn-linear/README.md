# ferrolearn-linear

Linear models for the [ferrolearn](https://crates.io/crates/ferrolearn) machine
learning framework. Validated against scikit-learn 1.8.0 head-to-head — see
the [workspace BENCHMARKS.md](../BENCHMARKS.md) for the full report.

## Algorithms

### Regression

| Model | Description |
|-------|-------------|
| `LinearRegression` | Ordinary Least Squares via Cholesky / QR fallback |
| `Ridge` | L2-regularized regression via Cholesky decomposition |
| `RidgeCV` | Ridge with built-in alpha cross-validation |
| `Lasso` | L1-regularized regression via coordinate descent |
| `LassoCV` | Lasso with alpha CV |
| `LassoLars` | Lasso via Least-Angle Regression |
| `Lars` | Least-Angle Regression |
| `OrthogonalMatchingPursuit` | Sparse coding via OMP |
| `ElasticNet` / `ElasticNetCV` | Combined L1/L2 regularization |
| `BayesianRidge` | Bayesian ridge with automatic regularization tuning |
| `ARDRegression` | Automatic Relevance Determination |
| `HuberRegressor` | Robust regression via IRLS with Huber loss |
| `QuantileRegressor` | L1-regularized quantile regression (sklearn-equivalent α scale) |
| `RANSACRegressor` | Robust regression with outlier rejection |
| `SGDRegressor` | Stochastic gradient descent regressor |
| `LinearSVR` / `NuSVR` | Linear / Nu Support Vector Regression |
| `IsotonicRegression` | Non-decreasing 1D regression |

### Classification

| Model | Description |
|-------|-------------|
| `LogisticRegression` | Binary and multiclass classification via L-BFGS |
| `LogisticRegressionCV` | Logistic regression with C cross-validation |
| `RidgeClassifier` | Ridge regression cast as a classifier |
| `LinearSVC` | Linear Support Vector Classifier (coordinate-Newton primal solver) |
| `NuSVC` / `SVC` | Kernel SVMs |
| `OneClassSVM` | Novelty / outlier detection |
| `LDA` | Linear Discriminant Analysis |
| `QDA` | Quadratic Discriminant Analysis |
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

All models follow the compile-time safety pattern: unfitted structs implement
`Fit`, fitted structs implement `Predict`. Calling `predict()` on an unfitted
model is a compile error.

## sklearn parity highlights (0.3.0)

- `LinearSVC` was rewritten with coordinate-Newton steps replacing fixed-step
  gradient descent — closed a -21pp accuracy gap at medium scale.
- `QuantileRegressor`'s `alpha` was rescaled by `n_samples` so it's
  numerically equivalent to scikit-learn's `alpha`.
- All regressor and classifier defaults now match scikit-learn ≥ 1.4.

## Float generics

All models are generic over `F: Float + Send + Sync + 'static`, supporting
both `f32` and `f64`.

## License

Licensed under either of [Apache License, Version 2.0](../LICENSE-APACHE) or
[MIT License](../LICENSE-MIT) at your option.
