# ferrolearn-core

Core traits, error types, dataset abstractions, pipeline infrastructure, and pluggable linear algebra backend for the [ferrolearn](https://crates.io/crates/ferrolearn) machine learning framework.

Every other ferrolearn crate depends on this one.

## Key types

### ML traits

- **`Fit<X, Y>`** — train a model, producing a fitted type
- **`Predict<X>`** — generate predictions from a fitted model
- **`Transform<X>`** — transform data (scalers, PCA, etc.)
- **`PartialFit<X, Y>`** — incremental / online learning
- **`FitTransform<X>`** — fit and transform in one step

### Compile-time safety

Unfitted models implement `Fit` but *not* `Predict`. Calling `fit()` returns a new fitted type that implements `Predict`. Attempting to call `predict()` on an unfitted model is a **compile error**.

```rust
use ferrolearn_core::{Fit, Predict};

// model.predict(&x);  // compile error — Ridge does not implement Predict
let fitted = model.fit(&x, &y)?;
let y_hat = fitted.predict(&x_test)?;  // OK — FittedRidge implements Predict
```

### Error handling

All public functions return `Result<T, FerroError>`. Library code never panics.

### Pipeline

`Pipeline` composes transformers and a final estimator using dynamic dispatch:

```rust
use ferrolearn_core::pipeline::Pipeline;

let pipeline = Pipeline::new()
    .add_transformer(scaler)
    .add_estimator(model);
```

### Backend trait

The `Backend` trait abstracts linear algebra operations (SVD, QR, Cholesky, eigendecomposition). The default `NdarrayFaerBackend` delegates to the [faer](https://crates.io/crates/faer) crate.

### Introspection

- `HasCoefficients` — access fitted model coefficients
- `HasFeatureImportances` — access feature importance scores
- `HasClasses` — access class labels from classifiers

## License

Licensed under either of [Apache License, Version 2.0](../LICENSE-APACHE) or [MIT License](../LICENSE-MIT) at your option.
