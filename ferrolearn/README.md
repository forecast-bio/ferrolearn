# ferrolearn

A scikit-learn equivalent for Rust. Type-safe, modular machine learning built on [ndarray](https://github.com/rust-ndarray/ndarray).

This is the **umbrella crate** that re-exports all ferrolearn sub-crates under a single dependency. If you want everything, depend on this crate. If you only need specific algorithms, depend on the individual sub-crates for smaller compile times and binary sizes.

## Quick start

```toml
[dependencies]
ferrolearn = "0.1"
```

```rust
use ferrolearn::prelude::*;
use ferrolearn::{linear, preprocess, datasets, metrics};

// Load the iris dataset
let (x, y) = datasets::load_iris::<f64>().unwrap();

// Scale features to zero mean and unit variance
let scaler = preprocess::StandardScaler::<f64>::new();
let fitted_scaler = scaler.fit(&x, &()).unwrap();
let x_scaled = fitted_scaler.transform(&x).unwrap();

// Train a logistic regression classifier
let model = linear::LogisticRegression::<f64>::new();
let fitted = model.fit(&x_scaled, &y).unwrap();
let predictions = fitted.predict(&x_scaled).unwrap();
```

## Sub-crates

| Crate | Description |
|-------|-------------|
| [`ferrolearn-core`](https://crates.io/crates/ferrolearn-core) | Traits (`Fit`, `Predict`, `Transform`), error types, pipeline, backend |
| [`ferrolearn-linear`](https://crates.io/crates/ferrolearn-linear) | Linear and generalized linear models |
| [`ferrolearn-tree`](https://crates.io/crates/ferrolearn-tree) | Decision trees and ensemble methods |
| [`ferrolearn-neighbors`](https://crates.io/crates/ferrolearn-neighbors) | k-Nearest Neighbors with KD-tree |
| [`ferrolearn-bayes`](https://crates.io/crates/ferrolearn-bayes) | Naive Bayes classifiers |
| [`ferrolearn-cluster`](https://crates.io/crates/ferrolearn-cluster) | Clustering algorithms |
| [`ferrolearn-decomp`](https://crates.io/crates/ferrolearn-decomp) | Dimensionality reduction and decomposition |
| [`ferrolearn-preprocess`](https://crates.io/crates/ferrolearn-preprocess) | Scalers, encoders, imputers, feature engineering |
| [`ferrolearn-metrics`](https://crates.io/crates/ferrolearn-metrics) | Evaluation metrics |
| [`ferrolearn-model-sel`](https://crates.io/crates/ferrolearn-model-sel) | Cross-validation, hyperparameter search, calibration |
| [`ferrolearn-datasets`](https://crates.io/crates/ferrolearn-datasets) | Toy datasets and synthetic data generators |
| [`ferrolearn-io`](https://crates.io/crates/ferrolearn-io) | Model serialization (MessagePack, JSON) |
| [`ferrolearn-sparse`](https://crates.io/crates/ferrolearn-sparse) | Sparse matrix formats (CSR, CSC, COO) |

## Requirements

- Rust 1.85+ (edition 2024)

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT License](LICENSE-MIT) at your option.
