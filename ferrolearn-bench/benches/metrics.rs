use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use ferrolearn_metrics::{classification, regression};
use ndarray::Array1;

const METRIC_SIZES: &[(&str, usize)] = &[("1K", 1_000), ("10K", 10_000), ("100K", 100_000)];

fn make_regression_predictions(n: usize) -> (Array1<f64>, Array1<f64>) {
    let y_true = Array1::from_iter((0..n).map(|i| i as f64 * 0.1));
    let y_pred = Array1::from_iter((0..n).map(|i| i as f64 * 0.1 + 0.01));
    (y_true, y_pred)
}

fn make_classification_predictions(n: usize) -> (Array1<usize>, Array1<usize>) {
    let y_true = Array1::from_iter((0..n).map(|i| i % 3));
    let y_pred = Array1::from_iter((0..n).map(|i| (i + 1) % 3));
    (y_true, y_pred)
}

fn bench_accuracy(c: &mut Criterion) {
    let mut group = c.benchmark_group("accuracy_score");
    for &(label, n) in METRIC_SIZES {
        let (y_true, y_pred) = make_classification_predictions(n);
        group.bench_with_input(BenchmarkId::from_parameter(label), &(), |b, ()| {
            b.iter(|| classification::accuracy_score(&y_true, &y_pred).unwrap());
        });
    }
    group.finish();
}

fn bench_f1(c: &mut Criterion) {
    let mut group = c.benchmark_group("f1_score");
    for &(label, n) in METRIC_SIZES {
        let (y_true, y_pred) = make_classification_predictions(n);
        group.bench_with_input(BenchmarkId::from_parameter(label), &(), |b, ()| {
            b.iter(|| {
                classification::f1_score(&y_true, &y_pred, classification::Average::Macro).unwrap()
            });
        });
    }
    group.finish();
}

fn bench_mse(c: &mut Criterion) {
    let mut group = c.benchmark_group("mean_squared_error");
    for &(label, n) in METRIC_SIZES {
        let (y_true, y_pred) = make_regression_predictions(n);
        group.bench_with_input(BenchmarkId::from_parameter(label), &(), |b, ()| {
            b.iter(|| regression::mean_squared_error(&y_true, &y_pred).unwrap());
        });
    }
    group.finish();
}

fn bench_r2(c: &mut Criterion) {
    let mut group = c.benchmark_group("r2_score");
    for &(label, n) in METRIC_SIZES {
        let (y_true, y_pred) = make_regression_predictions(n);
        group.bench_with_input(BenchmarkId::from_parameter(label), &(), |b, ()| {
            b.iter(|| regression::r2_score(&y_true, &y_pred).unwrap());
        });
    }
    group.finish();
}

criterion_group!(benches, bench_accuracy, bench_f1, bench_mse, bench_r2);
criterion_main!(benches);
