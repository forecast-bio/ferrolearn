# ferrolearn-kernel

Kernel methods for the [ferrolearn](https://crates.io/crates/ferrolearn) machine learning framework: nonparametric regression, Gaussian processes, kernel ridge, and kernel approximations.

## Algorithms

### Estimators

| Model | Description |
|-------|-------------|
| `NadarayaWatson` | Local constant (order 0) kernel regression |
| `LocalPolynomialRegression` | Local polynomial regression (orders 0–3+) with Tikhonov regularization |
| `KernelRidge` | Dual-form kernel ridge regression `(K + αI) c = y` |
| `GaussianProcessRegressor` | Bayesian nonparametric regression with predictive mean & variance |
| `GaussianProcessClassifier` | Probabilistic classification via Laplace approximation (R&W Algorithm 3.2), with `log_marginal_likelihood()` for hyperparameter selection |

### Kernel Approximations

| Method | Description |
|--------|-------------|
| `Nystroem` | Low-rank kernel approximation by sampling basis points |
| `RBFSampler` | Random Fourier features for the RBF kernel (Rahimi & Recht 2007) |

### Kernel Functions

| Kernel | Support |
|--------|---------|
| `GaussianKernel` | Infinite |
| `EpanechnikovKernel` | Compact |
| `TricubeKernel` | Compact |
| `BiweightKernel` | Compact |
| `TriweightKernel` | Compact |
| `UniformKernel` | Compact |
| `CosineKernel` | Compact |
| `DynKernel` | Runtime-selected (any of the above) |

### Bandwidth Selection

| Strategy | Description |
|----------|-------------|
| `Fixed` / `PerDimension` | User-specified scalar or per-dimension bandwidth |
| `CrossValidated` | LOO or k-fold CV with O(n) hat matrix shortcut |
| `Silverman` | Rule of thumb: `1.06 × σ_robust × n^(-1/5)` |
| `Scott` | Rule of thumb: `σ × n^(-1/(d+4))` |

### Diagnostics

| Tool | Description |
|------|-------------|
| `GoodnessOfFit` | R², adjusted R², AIC, BIC, effective degrees of freedom |
| `heteroscedasticity_test` | White, Breusch-Pagan, Goldfeld-Quandt, Dette-Munk-Wagner |
| `residual_diagnostics` | Jarque-Bera normality, skewness, kurtosis |

### Confidence Intervals

| Tool | Description |
|------|-------------|
| `wild_bootstrap_confidence_intervals` | Wild bootstrap CI with 5 bias correction methods |
| `fan_yao_variance_estimation` | Nonparametric variance function σ²(x) |
| `conformal_calibrate_ci` | Split conformal prediction intervals |

### GP Kernels

`RBFKernel`, `MaternKernel` (ν ∈ {0.5, 1.5, 2.5}), `ConstantKernel`, `WhiteKernel`, `DotProductKernel`, plus `SumKernel` and `ProductKernel` for composition via the [`GPKernel`] trait.

## Example

```rust
use ferrolearn_kernel::{NadarayaWatson, GaussianKernel};
use ferrolearn_core::{Fit, Predict};
use ndarray::{array, Array2};

let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
let y = array![0.0_f64, 0.84, 0.91, 0.14, -0.76]; // sin(x)

let model = NadarayaWatson::new();
let fitted = model.fit(&x, &y).unwrap();
let predictions = fitted.predict(&x).unwrap();
```

All estimators follow the compile-time safety pattern: unfitted structs implement `Fit`, fitted structs implement `Predict`. Calling `predict()` on an unfitted model is a compile error.

## Performance

2–27x faster than the Python original with bit-identical numerical accuracy (≤42 ULP). See [BENCHMARKS.md](BENCHMARKS.md) for details.

## Float Generics

All models are generic over `F: Float + Send + Sync + 'static`, supporting both `f32` and `f64`.

## License

Licensed under either of [Apache License, Version 2.0](../LICENSE-APACHE) or [MIT License](../LICENSE-MIT) at your option.
