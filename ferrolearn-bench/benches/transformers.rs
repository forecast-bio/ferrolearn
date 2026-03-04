use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use ferrolearn_bench::{SIZES, regression_data};
use ferrolearn_core::{Fit, Transform};
use ferrolearn_decomp::PCA;
use ferrolearn_preprocess::StandardScaler;

fn bench_standard_scaler(c: &mut Criterion) {
    let mut group = c.benchmark_group("StandardScaler");
    for &(label, n, p) in SIZES {
        let (x, _) = regression_data(n, p);
        group.bench_with_input(BenchmarkId::new("fit", label), &(), |b, _| {
            b.iter(|| StandardScaler::<f64>::new().fit(&x, &()).unwrap());
        });
        let fitted = StandardScaler::<f64>::new().fit(&x, &()).unwrap();
        group.bench_with_input(BenchmarkId::new("transform", label), &(), |b, _| {
            b.iter(|| fitted.transform(&x).unwrap());
        });
    }
    group.finish();
}

fn bench_pca(c: &mut Criterion) {
    let mut group = c.benchmark_group("PCA");
    group.sample_size(10);
    for &(label, n, p) in SIZES {
        let (x, _) = regression_data(n, p);
        let n_components = p.min(10);
        group.bench_with_input(BenchmarkId::new("fit", label), &(), |b, _| {
            b.iter(|| PCA::<f64>::new(n_components).fit(&x, &()).unwrap());
        });
        let fitted = PCA::<f64>::new(n_components).fit(&x, &()).unwrap();
        group.bench_with_input(BenchmarkId::new("transform", label), &(), |b, _| {
            b.iter(|| fitted.transform(&x).unwrap());
        });
    }
    group.finish();
}

criterion_group!(benches, bench_standard_scaler, bench_pca);
criterion_main!(benches);
