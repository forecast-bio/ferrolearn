//! Decomposition / dimensionality reduction benchmarks.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use ferrolearn_bench::{SIZES, regression_data};
use ferrolearn_core::{Fit, Transform};
use ferrolearn_decomp::{
    FactorAnalysis, FastICA, IncrementalPCA, KernelPCA, NMF, PCA, SparsePCA, TruncatedSVD,
};

fn bench_pca(c: &mut Criterion) {
    let mut group = c.benchmark_group("PCA");
    group.sample_size(10);
    for &(label, n, p) in SIZES {
        let (x, _) = regression_data(n, p);
        let n_comp = p.min(10);
        group.bench_with_input(BenchmarkId::new("fit", label), &(), |b, ()| {
            b.iter(|| PCA::<f64>::new(n_comp).fit(&x, &()).unwrap());
        });
        let fitted = PCA::<f64>::new(n_comp).fit(&x, &()).unwrap();
        group.bench_with_input(BenchmarkId::new("transform", label), &(), |b, ()| {
            b.iter(|| fitted.transform(&x).unwrap());
        });
    }
    group.finish();
}

fn bench_incremental_pca(c: &mut Criterion) {
    let mut group = c.benchmark_group("IncrementalPCA");
    group.sample_size(10);
    for &(label, n, p) in SIZES {
        let (x, _) = regression_data(n, p);
        let n_comp = p.min(10);
        group.bench_with_input(BenchmarkId::new("fit", label), &(), |b, ()| {
            b.iter(|| IncrementalPCA::<f64>::new(n_comp).fit(&x, &()).unwrap());
        });
    }
    group.finish();
}

fn bench_truncated_svd(c: &mut Criterion) {
    let mut group = c.benchmark_group("TruncatedSVD");
    group.sample_size(10);
    for &(label, n, p) in SIZES {
        let (x, _) = regression_data(n, p);
        let n_comp = p.min(5);
        group.bench_with_input(BenchmarkId::new("fit", label), &(), |b, ()| {
            b.iter(|| TruncatedSVD::<f64>::new(n_comp).fit(&x, &()).unwrap());
        });
    }
    group.finish();
}

fn bench_factor_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("FactorAnalysis");
    group.sample_size(10);
    // FA is iterative; restrict to small/medium.
    for &(label, n, p) in &SIZES[..2] {
        let (x, _) = regression_data(n, p);
        let n_comp = p.min(5);
        group.bench_with_input(BenchmarkId::new("fit", label), &(), |b, ()| {
            b.iter(|| FactorAnalysis::<f64>::new(n_comp).fit(&x, &()).unwrap());
        });
    }
    group.finish();
}

fn bench_fast_ica(c: &mut Criterion) {
    let mut group = c.benchmark_group("FastICA");
    group.sample_size(10);
    for &(label, n, p) in &SIZES[..2] {
        let (x, _) = regression_data(n, p);
        let n_comp = p.min(5);
        group.bench_with_input(BenchmarkId::new("fit", label), &(), |b, ()| {
            b.iter(|| FastICA::<f64>::new(n_comp).fit(&x, &()).unwrap());
        });
    }
    group.finish();
}

fn bench_nmf(c: &mut Criterion) {
    let mut group = c.benchmark_group("NMF");
    group.sample_size(10);
    for &(label, n, p) in &SIZES[..2] {
        let (x, _) = regression_data(n, p);
        let x_pos = x.mapv(f64::abs);
        let n_comp = p.min(5);
        group.bench_with_input(BenchmarkId::new("fit", label), &(), |b, ()| {
            b.iter(|| NMF::<f64>::new(n_comp).fit(&x_pos, &()).unwrap());
        });
    }
    group.finish();
}

fn bench_kernel_pca(c: &mut Criterion) {
    let mut group = c.benchmark_group("KernelPCA");
    group.sample_size(10);
    for &(label, n, p) in &SIZES[..2] {
        let (x, _) = regression_data(n, p);
        let n_comp = p.min(5);
        group.bench_with_input(BenchmarkId::new("fit", label), &(), |b, ()| {
            b.iter(|| KernelPCA::<f64>::new(n_comp).fit(&x, &()).unwrap());
        });
    }
    group.finish();
}

fn bench_sparse_pca(c: &mut Criterion) {
    let mut group = c.benchmark_group("SparsePCA");
    group.sample_size(10);
    for &(label, n, p) in &SIZES[..2] {
        let (x, _) = regression_data(n, p);
        let n_comp = p.min(5);
        group.bench_with_input(BenchmarkId::new("fit", label), &(), |b, ()| {
            b.iter(|| SparsePCA::<f64>::new(n_comp).fit(&x, &()).unwrap());
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_pca,
    bench_incremental_pca,
    bench_truncated_svd,
    bench_factor_analysis,
    bench_fast_ica,
    bench_nmf,
    bench_kernel_pca,
    bench_sparse_pca,
);
criterion_main!(benches);
