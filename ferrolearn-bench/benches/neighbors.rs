//! Neighbors benchmarks: KNN classifier/regressor, RadiusNeighbors,
//! NearestCentroid, LocalOutlierFactor.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use ferrolearn_bench::{SIZES, classification_data, regression_data};
use ferrolearn_core::{Fit, Predict};
use ferrolearn_neighbors::{
    KNeighborsClassifier, KNeighborsRegressor, LocalOutlierFactor, NearestCentroid,
    RadiusNeighborsClassifier,
};

fn bench_knn_classifier(c: &mut Criterion) {
    let mut group = c.benchmark_group("KNeighborsClassifier");
    for &(label, n, p) in SIZES {
        let (x, y) = classification_data(n, p);
        group.bench_with_input(BenchmarkId::new("fit", label), &(), |b, ()| {
            b.iter(|| KNeighborsClassifier::<f64>::new().fit(&x, &y).unwrap());
        });
        let fitted = KNeighborsClassifier::<f64>::new().fit(&x, &y).unwrap();
        group.bench_with_input(BenchmarkId::new("predict", label), &(), |b, ()| {
            b.iter(|| fitted.predict(&x).unwrap());
        });
    }
    group.finish();
}

fn bench_knn_regressor(c: &mut Criterion) {
    let mut group = c.benchmark_group("KNeighborsRegressor");
    for &(label, n, p) in SIZES {
        let (x, y) = regression_data(n, p);
        group.bench_with_input(BenchmarkId::new("fit", label), &(), |b, ()| {
            b.iter(|| KNeighborsRegressor::<f64>::new().fit(&x, &y).unwrap());
        });
        let fitted = KNeighborsRegressor::<f64>::new().fit(&x, &y).unwrap();
        group.bench_with_input(BenchmarkId::new("predict", label), &(), |b, ()| {
            b.iter(|| fitted.predict(&x).unwrap());
        });
    }
    group.finish();
}

fn bench_nearest_centroid(c: &mut Criterion) {
    let mut group = c.benchmark_group("NearestCentroid");
    for &(label, n, p) in SIZES {
        let (x, y) = classification_data(n, p);
        group.bench_with_input(BenchmarkId::new("fit", label), &(), |b, ()| {
            b.iter(|| NearestCentroid::<f64>::new().fit(&x, &y).unwrap());
        });
        let fitted = NearestCentroid::<f64>::new().fit(&x, &y).unwrap();
        group.bench_with_input(BenchmarkId::new("predict", label), &(), |b, ()| {
            b.iter(|| fitted.predict(&x).unwrap());
        });
    }
    group.finish();
}

fn bench_radius_neighbors(c: &mut Criterion) {
    // RadiusNeighbors at 10K x 100 is extremely slow; restrict to the smaller
    // sizes so the bench finishes in reasonable wall-clock time.
    let mut group = c.benchmark_group("RadiusNeighborsClassifier");
    group.sample_size(10);
    for &(label, n, p) in &SIZES[..2] {
        let (x, y) = classification_data(n, p);
        group.bench_with_input(BenchmarkId::new("fit", label), &(), |b, ()| {
            b.iter(|| {
                RadiusNeighborsClassifier::<f64>::new()
                    .with_radius(5.0)
                    .fit(&x, &y)
                    .unwrap()
            });
        });
        let fitted = RadiusNeighborsClassifier::<f64>::new()
            .with_radius(5.0)
            .fit(&x, &y)
            .unwrap();
        group.bench_with_input(BenchmarkId::new("predict", label), &(), |b, ()| {
            b.iter(|| fitted.predict(&x).unwrap());
        });
    }
    group.finish();
}

fn bench_lof(c: &mut Criterion) {
    let mut group = c.benchmark_group("LocalOutlierFactor");
    group.sample_size(10);
    for &(label, n, p) in &SIZES[..2] {
        let (x, _) = classification_data(n, p);
        group.bench_with_input(BenchmarkId::new("fit", label), &(), |b, ()| {
            b.iter(|| LocalOutlierFactor::<f64>::new().fit(&x, &()).unwrap());
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_knn_classifier,
    bench_knn_regressor,
    bench_nearest_centroid,
    bench_radius_neighbors,
    bench_lof,
);
criterion_main!(benches);
