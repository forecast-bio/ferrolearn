//! Kernel methods benchmarks: KernelRidge, GaussianProcess, Nystroem, RBFSampler.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use ferrolearn_bench::{regression_data};
use ferrolearn_core::{Fit, Predict, Transform};
use ferrolearn_kernel::{
    GaussianProcessRegressor, KernelRidge, Nystroem, RBFSampler,
    gp_kernels::RBFKernel,
};

const KERNEL_SIZES: &[(&str, usize, usize)] = &[
    ("tiny_50x5", 50, 5),
    ("small_500x10", 500, 10),
    ("medium_2Kx20", 2_000, 20),
];

fn bench_kernel_ridge(c: &mut Criterion) {
    let mut group = c.benchmark_group("KernelRidge");
    group.sample_size(10);
    for &(label, n, p) in KERNEL_SIZES {
        let (x, y) = regression_data(n, p);
        group.bench_with_input(BenchmarkId::new("fit", label), &(), |b, ()| {
            b.iter(|| KernelRidge::<f64>::new().fit(&x, &y).unwrap());
        });
        let fitted = KernelRidge::<f64>::new().fit(&x, &y).unwrap();
        group.bench_with_input(BenchmarkId::new("predict", label), &(), |b, ()| {
            b.iter(|| fitted.predict(&x).unwrap());
        });
    }
    group.finish();
}

fn bench_gaussian_process(c: &mut Criterion) {
    let mut group = c.benchmark_group("GaussianProcessRegressor");
    group.sample_size(10);
    // Cubic in n; cap the size sweep.
    for &(label, n, p) in &KERNEL_SIZES[..2] {
        let (x, y) = regression_data(n, p);
        group.bench_with_input(BenchmarkId::new("fit", label), &(), |b, ()| {
            b.iter(|| {
                GaussianProcessRegressor::<f64>::new(Box::new(RBFKernel::new(1.0)))
                    .fit(&x, &y)
                    .unwrap()
            });
        });
        let fitted = GaussianProcessRegressor::<f64>::new(Box::new(RBFKernel::new(1.0)))
            .fit(&x, &y)
            .unwrap();
        group.bench_with_input(BenchmarkId::new("predict", label), &(), |b, ()| {
            b.iter(|| fitted.predict(&x).unwrap());
        });
    }
    group.finish();
}

fn bench_nystroem(c: &mut Criterion) {
    let mut group = c.benchmark_group("Nystroem");
    group.sample_size(10);
    for &(label, n, p) in KERNEL_SIZES {
        let (x, _) = regression_data(n, p);
        group.bench_with_input(BenchmarkId::new("fit", label), &(), |b, ()| {
            b.iter(|| Nystroem::<f64>::new().fit(&x, &()).unwrap());
        });
        let fitted = Nystroem::<f64>::new().fit(&x, &()).unwrap();
        group.bench_with_input(BenchmarkId::new("transform", label), &(), |b, ()| {
            b.iter(|| fitted.transform(&x).unwrap());
        });
    }
    group.finish();
}

fn bench_rbf_sampler(c: &mut Criterion) {
    let mut group = c.benchmark_group("RBFSampler");
    group.sample_size(10);
    for &(label, n, p) in KERNEL_SIZES {
        let (x, _) = regression_data(n, p);
        group.bench_with_input(BenchmarkId::new("fit", label), &(), |b, ()| {
            b.iter(|| RBFSampler::<f64>::new().fit(&x, &()).unwrap());
        });
        let fitted = RBFSampler::<f64>::new().fit(&x, &()).unwrap();
        group.bench_with_input(BenchmarkId::new("transform", label), &(), |b, ()| {
            b.iter(|| fitted.transform(&x).unwrap());
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_kernel_ridge,
    bench_gaussian_process,
    bench_nystroem,
    bench_rbf_sampler,
);
criterion_main!(benches);
