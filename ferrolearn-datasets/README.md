# ferrolearn-datasets

Built-in datasets and synthetic data generators for the [ferrolearn](https://crates.io/crates/ferrolearn) machine learning framework.

## Toy datasets

Classic datasets embedded at compile time for testing and examples:

| Function | Samples | Features | Task |
|----------|---------|----------|------|
| `load_iris` | 150 | 4 | 3-class classification |
| `load_wine` | 178 | 13 | 3-class classification |
| `load_breast_cancer` | 569 | 30 | Binary classification |
| `load_diabetes` | 442 | 10 | Regression |
| `load_digits` | 1797 | 64 | 10-class classification |

## Synthetic generators

| Function | Description |
|----------|-------------|
| `make_classification` | Random n-class classification with configurable informative/redundant features |
| `make_regression` | Random regression with configurable noise |
| `make_blobs` | Isotropic Gaussian blobs for clustering |
| `make_moons` | Two interleaving half-circles |
| `make_circles` | Concentric circles |
| `make_swiss_roll` | Swiss roll manifold in 3D |
| `make_s_curve` | S-curve manifold in 3D |
| `make_sparse_uncorrelated` | Sparse uncorrelated regression |

## Example

```rust
use ferrolearn_datasets::{load_iris, make_blobs};

// Load a classic dataset
let (x, y) = load_iris::<f64>().unwrap();
assert_eq!(x.nrows(), 150);

// Generate synthetic clustering data
let (x, labels) = make_blobs::<f64>(300, 3, 5, Some(42)).unwrap();
assert_eq!(x.nrows(), 300);
```

## License

Licensed under either of [Apache License, Version 2.0](../LICENSE-APACHE) or [MIT License](../LICENSE-MIT) at your option.
