//! Shared data generation, timing, and reporting helpers for ferrolearn benchmarks.

use ferrolearn_datasets::generators::{make_blobs, make_classification, make_regression};
use ndarray::{Array1, Array2};
use serde::Serialize;
use std::time::{Duration, Instant};

/// Dataset sizes for benchmarking the bulk of estimator families.
pub const SIZES: &[(&str, usize, usize)] = &[
    ("tiny_50x5", 50, 5),
    ("small_1Kx10", 1_000, 10),
    ("medium_10Kx100", 10_000, 100),
];

/// Generate regression data (X, y) for a given size.
pub fn regression_data(n_samples: usize, n_features: usize) -> (Array2<f64>, Array1<f64>) {
    make_regression::<f64>(n_samples, n_features, n_features, 0.1, Some(42)).unwrap()
}

/// Generate binary classification data (X, y) for a given size.
pub fn classification_data(n_samples: usize, n_features: usize) -> (Array2<f64>, Array1<usize>) {
    make_classification::<f64>(n_samples, n_features, 2, Some(42)).unwrap()
}

/// Generate multi-class classification data (X, y).
pub fn multiclass_data(
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
) -> (Array2<f64>, Array1<usize>) {
    make_classification::<f64>(n_samples, n_features, n_classes, Some(42)).unwrap()
}

/// Generate clustering data (X, y) — fixed at 8 isotropic blobs.
pub fn clustering_data(n_samples: usize, n_features: usize) -> (Array2<f64>, Array1<usize>) {
    make_blobs::<f64>(n_samples, n_features, 8, 1.0, Some(42)).unwrap()
}

/// Hold-out split for regression: returns `(x_train, x_test, y_train, y_test)`
/// using a deterministic 80/20 partition with no shuffling. The seeded
/// generator already produces randomized rows, so a fixed prefix split is
/// reproducible without an extra RNG hop.
pub fn split_regression(
    x: &Array2<f64>,
    y: &Array1<f64>,
) -> (Array2<f64>, Array2<f64>, Array1<f64>, Array1<f64>) {
    let n = x.nrows();
    let n_train = (n * 4) / 5;
    let x_train = x.slice(ndarray::s![..n_train, ..]).to_owned();
    let x_test = x.slice(ndarray::s![n_train.., ..]).to_owned();
    let y_train = y.slice(ndarray::s![..n_train]).to_owned();
    let y_test = y.slice(ndarray::s![n_train..]).to_owned();
    (x_train, x_test, y_train, y_test)
}

/// Hold-out split for classification with `usize` labels.
pub fn split_classification(
    x: &Array2<f64>,
    y: &Array1<usize>,
) -> (Array2<f64>, Array2<f64>, Array1<usize>, Array1<usize>) {
    let n = x.nrows();
    let n_train = (n * 4) / 5;
    let x_train = x.slice(ndarray::s![..n_train, ..]).to_owned();
    let x_test = x.slice(ndarray::s![n_train.., ..]).to_owned();
    let y_train = y.slice(ndarray::s![..n_train]).to_owned();
    let y_test = y.slice(ndarray::s![n_train..]).to_owned();
    (x_train, x_test, y_train, y_test)
}

/// Run `f` repeatedly and return the median elapsed time in microseconds.
///
/// At least `min_iters` iterations are performed; if total elapsed time is
/// below `min_total`, more iterations are run up to `max_iters`. Useful for
/// algorithms where a single run is too noisy.
pub fn median_micros<F>(mut f: F, min_iters: usize, max_iters: usize, min_total: Duration) -> f64
where
    F: FnMut(),
{
    let mut samples: Vec<Duration> = Vec::with_capacity(max_iters);
    let start = Instant::now();
    for _ in 0..min_iters {
        let t = Instant::now();
        f();
        samples.push(t.elapsed());
    }
    while samples.len() < max_iters && start.elapsed() < min_total {
        let t = Instant::now();
        f();
        samples.push(t.elapsed());
    }
    samples.sort();
    let mid = samples.len() / 2;
    samples[mid].as_secs_f64() * 1e6
}

/// One-shot timing: run once and return microseconds. Use for fits that take
/// seconds.
pub fn time_once_micros<F>(mut f: F) -> f64
where
    F: FnMut(),
{
    let t = Instant::now();
    f();
    t.elapsed().as_secs_f64() * 1e6
}

/// One row of the harness output: identifies an estimator + dataset, the
/// timings, and the quality metric.
#[derive(Debug, Clone, Serialize)]
pub struct BenchRecord {
    pub family: String,
    pub algorithm: String,
    pub dataset: String,
    pub n_samples: usize,
    pub n_features: usize,
    /// Median fit() time in microseconds.
    pub fit_us: f64,
    /// Median predict()/transform() time in microseconds. `None` if the
    /// estimator does not support a separate predict path.
    pub predict_us: Option<f64>,
    /// Name of the quality metric (e.g. "r2", "accuracy", "ari", "recon_l2").
    /// `None` for transformers without a clear scalar quality metric.
    pub metric: Option<String>,
    /// The metric value. `NaN` if the fit failed.
    pub score: Option<f64>,
}

impl BenchRecord {
    pub fn new(family: &str, algorithm: &str, dataset: &str, n: usize, p: usize) -> Self {
        BenchRecord {
            family: family.to_string(),
            algorithm: algorithm.to_string(),
            dataset: dataset.to_string(),
            n_samples: n,
            n_features: p,
            fit_us: 0.0,
            predict_us: None,
            metric: None,
            score: None,
        }
    }
}
