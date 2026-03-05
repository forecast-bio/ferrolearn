# ferrolearn-bayes

Naive Bayes classifiers for the [ferrolearn](https://crates.io/crates/ferrolearn) machine learning framework.

## Algorithms

| Model | Best for |
|-------|----------|
| `GaussianNB` | Continuous features with Gaussian distributions |
| `MultinomialNB` | Discrete count data (e.g., word counts in text classification) |
| `BernoulliNB` | Binary/boolean features with optional binarization threshold |
| `ComplementNB` | Imbalanced datasets (complement-class variant of Multinomial NB) |

All classifiers support `predict_proba` for class probability estimates.

## Example

```rust
use ferrolearn_bayes::GaussianNB;
use ferrolearn_core::{Fit, Predict};
use ndarray::{array, Array2};

let x = Array2::from_shape_vec((6, 2), vec![
    1.0, 2.0, 1.5, 2.5, 1.2, 1.8,
    6.0, 7.0, 5.8, 6.5, 6.2, 7.2,
]).unwrap();
let y = array![0usize, 0, 0, 1, 1, 1];

let model = GaussianNB::<f64>::new();
let fitted = model.fit(&x, &y).unwrap();
let predictions = fitted.predict(&x).unwrap();

// Get class probabilities
let probas = fitted.predict_proba(&x).unwrap();
```

## License

Licensed under either of [Apache License, Version 2.0](../LICENSE-APACHE) or [MIT License](../LICENSE-MIT) at your option.
