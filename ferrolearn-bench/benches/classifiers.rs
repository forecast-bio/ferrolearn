use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use ferrolearn_bayes::GaussianNB;
use ferrolearn_bench::{SIZES, classification_data};
use ferrolearn_core::{Fit, Predict};
use ferrolearn_linear::LogisticRegression;
use ferrolearn_neighbors::KNeighborsClassifier;
use ferrolearn_tree::{DecisionTreeClassifier, RandomForestClassifier};

fn bench_logistic_regression(c: &mut Criterion) {
    let mut group = c.benchmark_group("LogisticRegression");
    group.sample_size(10);
    for &(label, n, p) in SIZES {
        let (x, y) = classification_data(n, p);
        group.bench_with_input(BenchmarkId::new("fit", label), &(), |b, ()| {
            b.iter(|| LogisticRegression::<f64>::new().fit(&x, &y).unwrap());
        });
        let fitted = LogisticRegression::<f64>::new().fit(&x, &y).unwrap();
        group.bench_with_input(BenchmarkId::new("predict", label), &(), |b, ()| {
            b.iter(|| fitted.predict(&x).unwrap());
        });
    }
    group.finish();
}

fn bench_decision_tree(c: &mut Criterion) {
    let mut group = c.benchmark_group("DecisionTreeClassifier");
    for &(label, n, p) in SIZES {
        let (x, y) = classification_data(n, p);
        group.bench_with_input(BenchmarkId::new("fit", label), &(), |b, ()| {
            b.iter(|| DecisionTreeClassifier::<f64>::new().fit(&x, &y).unwrap());
        });
        let fitted = DecisionTreeClassifier::<f64>::new().fit(&x, &y).unwrap();
        group.bench_with_input(BenchmarkId::new("predict", label), &(), |b, ()| {
            b.iter(|| fitted.predict(&x).unwrap());
        });
    }
    group.finish();
}

fn bench_random_forest(c: &mut Criterion) {
    let mut group = c.benchmark_group("RandomForestClassifier");
    group.sample_size(10);
    for &(label, n, p) in SIZES {
        let (x, y) = classification_data(n, p);
        group.bench_with_input(BenchmarkId::new("fit", label), &(), |b, ()| {
            b.iter(|| {
                RandomForestClassifier::<f64>::new()
                    .with_random_state(42)
                    .fit(&x, &y)
                    .unwrap()
            });
        });
        let fitted = RandomForestClassifier::<f64>::new()
            .with_random_state(42)
            .fit(&x, &y)
            .unwrap();
        group.bench_with_input(BenchmarkId::new("predict", label), &(), |b, ()| {
            b.iter(|| fitted.predict(&x).unwrap());
        });
    }
    group.finish();
}

fn bench_knn(c: &mut Criterion) {
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

criterion_group!(
    benches,
    bench_logistic_regression,
    bench_decision_tree,
    bench_random_forest,
    bench_knn,
    bench_gaussian_nb,
);
criterion_main!(benches);
