use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use ferrolearn_bench::{SIZES, regression_data};
use ferrolearn_core::{Fit, Predict};
use ferrolearn_linear::{
    ElasticNet, Lasso, LinearRegression, Ridge,
};

fn bench_linear_regression(c: &mut Criterion) {
    let mut group = c.benchmark_group("LinearRegression");
    for &(label, n, p) in SIZES {
        let (x, y) = regression_data(n, p);
        group.bench_with_input(BenchmarkId::new("fit", label), &(), |b, _| {
            b.iter(|| LinearRegression::<f64>::new().fit(&x, &y).unwrap());
        });
        let fitted = LinearRegression::<f64>::new().fit(&x, &y).unwrap();
        group.bench_with_input(BenchmarkId::new("predict", label), &(), |b, _| {
            b.iter(|| fitted.predict(&x).unwrap());
        });
    }
    group.finish();
}

fn bench_ridge(c: &mut Criterion) {
    let mut group = c.benchmark_group("Ridge");
    for &(label, n, p) in SIZES {
        let (x, y) = regression_data(n, p);
        group.bench_with_input(BenchmarkId::new("fit", label), &(), |b, _| {
            b.iter(|| Ridge::<f64>::new().fit(&x, &y).unwrap());
        });
        let fitted = Ridge::<f64>::new().fit(&x, &y).unwrap();
        group.bench_with_input(BenchmarkId::new("predict", label), &(), |b, _| {
            b.iter(|| fitted.predict(&x).unwrap());
        });
    }
    group.finish();
}

fn bench_lasso(c: &mut Criterion) {
    let mut group = c.benchmark_group("Lasso");
    group.sample_size(10);
    for &(label, n, p) in SIZES {
        let (x, y) = regression_data(n, p);
        group.bench_with_input(BenchmarkId::new("fit", label), &(), |b, _| {
            b.iter(|| Lasso::<f64>::new().fit(&x, &y).unwrap());
        });
        let fitted = Lasso::<f64>::new().fit(&x, &y).unwrap();
        group.bench_with_input(BenchmarkId::new("predict", label), &(), |b, _| {
            b.iter(|| fitted.predict(&x).unwrap());
        });
    }
    group.finish();
}

fn bench_elastic_net(c: &mut Criterion) {
    let mut group = c.benchmark_group("ElasticNet");
    group.sample_size(10);
    for &(label, n, p) in SIZES {
        let (x, y) = regression_data(n, p);
        group.bench_with_input(BenchmarkId::new("fit", label), &(), |b, _| {
            b.iter(|| ElasticNet::<f64>::new().fit(&x, &y).unwrap());
        });
        let fitted = ElasticNet::<f64>::new().fit(&x, &y).unwrap();
        group.bench_with_input(BenchmarkId::new("predict", label), &(), |b, _| {
            b.iter(|| fitted.predict(&x).unwrap());
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_linear_regression,
    bench_ridge,
    bench_lasso,
    bench_elastic_net,
);
criterion_main!(benches);
