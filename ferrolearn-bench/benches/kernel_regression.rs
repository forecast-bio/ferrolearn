use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use ferrolearn_bench::regression_data;
use ferrolearn_core::{Fit, Predict};
use ferrolearn_kernel::bandwidth::BandwidthStrategy;
use ferrolearn_kernel::kernels::{EpanechnikovKernel, GaussianKernel};
use ferrolearn_kernel::local_polynomial::LocalPolynomialRegression;
use ferrolearn_kernel::nadaraya_watson::NadarayaWatson;
use ferrolearn_kernel::weights;
use ndarray::Array1;

const KERNEL_SIZES: &[(&str, usize)] = &[
    ("100x1", 100),
    ("500x1", 500),
    ("1Kx1", 1_000),
    ("5Kx1", 5_000),
];

fn bench_nw_fit_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("NadarayaWatson");
    group.sample_size(10);

    for &(label, n) in KERNEL_SIZES {
        let (x, y) = regression_data(n, 1);

        group.bench_with_input(BenchmarkId::new("fit_gaussian", label), &(), |b, _| {
            let nw = NadarayaWatson::with_kernel(GaussianKernel, BandwidthStrategy::Silverman);
            b.iter(|| nw.fit(&x, &y).unwrap());
        });

        let nw = NadarayaWatson::with_kernel(GaussianKernel, BandwidthStrategy::Silverman);
        let fitted = nw.fit(&x, &y).unwrap();
        group.bench_with_input(BenchmarkId::new("predict_gaussian", label), &(), |b, _| {
            b.iter(|| fitted.predict(&x).unwrap());
        });

        group.bench_with_input(BenchmarkId::new("fit_epanechnikov", label), &(), |b, _| {
            let nw = NadarayaWatson::with_kernel(EpanechnikovKernel, BandwidthStrategy::Silverman);
            b.iter(|| nw.fit(&x, &y).unwrap());
        });
    }
    group.finish();
}

fn bench_lpr_fit_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("LocalPolynomial");
    group.sample_size(10);

    for &(label, n) in &KERNEL_SIZES[..3] {
        let (x, y) = regression_data(n, 1);

        group.bench_with_input(BenchmarkId::new("fit_order1", label), &(), |b, _| {
            let lpr = LocalPolynomialRegression::with_kernel(
                GaussianKernel,
                BandwidthStrategy::Silverman,
                1,
            );
            b.iter(|| lpr.fit(&x, &y).unwrap());
        });

        let lpr =
            LocalPolynomialRegression::with_kernel(GaussianKernel, BandwidthStrategy::Silverman, 1);
        let fitted = lpr.fit(&x, &y).unwrap();
        group.bench_with_input(BenchmarkId::new("predict_order1", label), &(), |b, _| {
            b.iter(|| fitted.predict(&x).unwrap());
        });
    }
    group.finish();
}

fn bench_kernel_weights(c: &mut Criterion) {
    let mut group = c.benchmark_group("KernelWeights");
    group.sample_size(10);

    for &(label, n) in KERNEL_SIZES {
        let (x, _) = regression_data(n, 1);
        let bw = ndarray::array![0.5f64];

        group.bench_with_input(BenchmarkId::new("gaussian", label), &(), |b, _| {
            b.iter(|| weights::compute_kernel_weights(&x, &x, &bw, &GaussianKernel));
        });

        group.bench_with_input(BenchmarkId::new("epanechnikov", label), &(), |b, _| {
            b.iter(|| weights::compute_kernel_weights(&x, &x, &bw, &EpanechnikovKernel));
        });
    }
    group.finish();
}

fn bench_hat_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("HatMatrix");
    group.sample_size(10);

    for &(label, n) in &KERNEL_SIZES[..3] {
        let (x, y) = regression_data(n, 1);
        let bw = ndarray::array![0.5f64];

        group.bench_with_input(BenchmarkId::new("loocv_shortcut", label), &(), |b, _| {
            b.iter(|| {
                ferrolearn_kernel::hat_matrix::loocv_hat_matrix_shortcut(
                    &x,
                    &y,
                    &bw,
                    &GaussianKernel,
                )
            });
        });

        group.bench_with_input(BenchmarkId::new("effective_df", label), &(), |b, _| {
            b.iter(|| ferrolearn_kernel::hat_matrix::effective_df(&x, &bw, &GaussianKernel));
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_nw_fit_predict,
    bench_lpr_fit_predict,
    bench_kernel_weights,
    bench_hat_matrix,
);
criterion_main!(benches);
