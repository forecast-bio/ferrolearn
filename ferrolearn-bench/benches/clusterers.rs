use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use ferrolearn_bench::{SIZES, clustering_data};
use ferrolearn_cluster::KMeans;
use ferrolearn_core::{Fit, Predict};

fn bench_kmeans(c: &mut Criterion) {
    let mut group = c.benchmark_group("KMeans");
    group.sample_size(10);
    for &(label, n, p) in SIZES {
        let (x, _) = clustering_data(n, p);
        group.bench_with_input(BenchmarkId::new("fit", label), &(), |b, _| {
            b.iter(|| {
                KMeans::<f64>::new(8)
                    .with_random_state(42)
                    .with_n_init(3)
                    .fit(&x, &())
                    .unwrap()
            });
        });
        let fitted = KMeans::<f64>::new(8)
            .with_random_state(42)
            .with_n_init(3)
            .fit(&x, &())
            .unwrap();
        group.bench_with_input(BenchmarkId::new("predict", label), &(), |b, _| {
            b.iter(|| fitted.predict(&x).unwrap());
        });
    }
    group.finish();
}

criterion_group!(benches, bench_kmeans);
criterion_main!(benches);
