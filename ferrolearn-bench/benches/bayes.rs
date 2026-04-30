//! Naive Bayes family benchmarks.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use ferrolearn_bayes::{BernoulliNB, ComplementNB, GaussianNB, MultinomialNB};
use ferrolearn_bench::{SIZES, classification_data};
use ferrolearn_core::{Fit, Predict};
use ndarray::Array2;

fn nonneg_features(x: &Array2<f64>) -> Array2<f64> {
    x.mapv(|v| v.abs())
}

fn binary_features(x: &Array2<f64>) -> Array2<f64> {
    x.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 })
}

fn bench_gaussian_nb(c: &mut Criterion) {
    let mut group = c.benchmark_group("GaussianNB");
    for &(label, n, p) in SIZES {
        let (x, y) = classification_data(n, p);
        group.bench_with_input(BenchmarkId::new("fit", label), &(), |b, ()| {
            b.iter(|| GaussianNB::<f64>::new().fit(&x, &y).unwrap());
        });
        let fitted = GaussianNB::<f64>::new().fit(&x, &y).unwrap();
        group.bench_with_input(BenchmarkId::new("predict", label), &(), |b, ()| {
            b.iter(|| fitted.predict(&x).unwrap());
        });
    }
    group.finish();
}

fn bench_multinomial_nb(c: &mut Criterion) {
    let mut group = c.benchmark_group("MultinomialNB");
    for &(label, n, p) in SIZES {
        let (x_raw, y) = classification_data(n, p);
        let x = nonneg_features(&x_raw);
        group.bench_with_input(BenchmarkId::new("fit", label), &(), |b, ()| {
            b.iter(|| MultinomialNB::<f64>::new().fit(&x, &y).unwrap());
        });
        let fitted = MultinomialNB::<f64>::new().fit(&x, &y).unwrap();
        group.bench_with_input(BenchmarkId::new("predict", label), &(), |b, ()| {
            b.iter(|| fitted.predict(&x).unwrap());
        });
    }
    group.finish();
}

fn bench_complement_nb(c: &mut Criterion) {
    let mut group = c.benchmark_group("ComplementNB");
    for &(label, n, p) in SIZES {
        let (x_raw, y) = classification_data(n, p);
        let x = nonneg_features(&x_raw);
        group.bench_with_input(BenchmarkId::new("fit", label), &(), |b, ()| {
            b.iter(|| ComplementNB::<f64>::new().fit(&x, &y).unwrap());
        });
        let fitted = ComplementNB::<f64>::new().fit(&x, &y).unwrap();
        group.bench_with_input(BenchmarkId::new("predict", label), &(), |b, ()| {
            b.iter(|| fitted.predict(&x).unwrap());
        });
    }
    group.finish();
}

fn bench_bernoulli_nb(c: &mut Criterion) {
    let mut group = c.benchmark_group("BernoulliNB");
    for &(label, n, p) in SIZES {
        let (x_raw, y) = classification_data(n, p);
        let x = binary_features(&x_raw);
        group.bench_with_input(BenchmarkId::new("fit", label), &(), |b, ()| {
            b.iter(|| BernoulliNB::<f64>::new().fit(&x, &y).unwrap());
        });
        let fitted = BernoulliNB::<f64>::new().fit(&x, &y).unwrap();
        group.bench_with_input(BenchmarkId::new("predict", label), &(), |b, ()| {
            b.iter(|| fitted.predict(&x).unwrap());
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_gaussian_nb,
    bench_multinomial_nb,
    bench_complement_nb,
    bench_bernoulli_nb,
);
criterion_main!(benches);
