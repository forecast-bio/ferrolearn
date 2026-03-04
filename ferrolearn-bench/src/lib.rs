//! Shared data generation helpers for ferrolearn benchmarks.

use ferrolearn_datasets::generators::{make_blobs, make_classification, make_regression};
use ndarray::{Array1, Array2};

/// Dataset sizes for benchmarking.
pub const SIZES: &[(&str, usize, usize)] = &[
    ("tiny_50x5", 50, 5),
    ("small_1Kx10", 1_000, 10),
    ("medium_10Kx100", 10_000, 100),
];

/// Generate regression data (X, y) for a given size.
pub fn regression_data(n_samples: usize, n_features: usize) -> (Array2<f64>, Array1<f64>) {
    let n_informative = n_features.min(n_features);
    make_regression::<f64>(n_samples, n_features, n_informative, 0.1, Some(42)).unwrap()
}

/// Generate classification data (X, y) for a given size.
pub fn classification_data(n_samples: usize, n_features: usize) -> (Array2<f64>, Array1<usize>) {
    let n_classes = 2;
    make_classification::<f64>(n_samples, n_features, n_classes, Some(42)).unwrap()
}

/// Generate clustering data (X, y) for a given size.
pub fn clustering_data(n_samples: usize, n_features: usize) -> (Array2<f64>, Array1<usize>) {
    make_blobs::<f64>(n_samples, n_features, 8, 1.0, Some(42)).unwrap()
}
