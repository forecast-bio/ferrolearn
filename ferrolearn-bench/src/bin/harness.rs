//! Comprehensive speed + accuracy harness for ferrolearn estimators.
//!
//! Runs every supported estimator at three dataset sizes, measuring:
//!   * median fit() wall-clock time
//!   * median predict() / transform() wall-clock time (where applicable)
//!   * a quality metric (R², accuracy, ARI, reconstruction L2, etc.)
//!
//! Output is a JSON array of [`BenchRecord`]. Pair with `sklearn_bench.py`
//! (which emits the same shape) and feed both into `compare.py` to render
//! the comparison report.
//!
//! ## Note on cross-library accuracy comparison
//!
//! Both sides use seeded synthetic data. ferrolearn calls
//! `ferrolearn_datasets::generators::make_*` and sklearn calls
//! `sklearn.datasets.make_*` — these are **different generators** with
//! different statistical structure (sklearn's `make_classification` mixes
//! informative/redundant/repeated features and flips class labels; ferrolearn's
//! is simple Gaussian clusters). So speed numbers are directly comparable
//! (same shapes, same dtypes) but absolute accuracy/R² values reflect each
//! library's data, not a shared benchmark. To compare quality on identical
//! data, persist a dataset to disk and load it from both harnesses.
//!
//! Usage: `cargo run --release --bin harness > ferrolearn_bench.json`

use std::time::Duration;

use ferrolearn_bench::{
    BenchRecord, classification_data, clustering_data, median_micros, multiclass_data,
    regression_data, split_classification, split_regression, time_once_micros,
};
use ferrolearn_core::{Fit, Predict, Transform};
use ferrolearn_metrics::{classification, clustering, regression};

use ferrolearn_bayes::{BernoulliNB, ComplementNB, GaussianNB, MultinomialNB};
use ferrolearn_cluster::{
    AgglomerativeClustering, Birch, BisectingKMeans, DBSCAN, GaussianMixture, KMeans, MeanShift,
    MiniBatchKMeans, SpectralClustering,
};
use ferrolearn_decomp::{
    FactorAnalysis, FastICA, IncrementalPCA, KernelPCA, NMF, PCA, SparsePCA, TruncatedSVD,
};
use ferrolearn_kernel::{KernelRidge, Nystroem, RBFSampler};
use ferrolearn_linear::{
    ARDRegression, BayesianRidge, ElasticNet, HuberRegressor, Lasso, LinearRegression,
    LinearSVC, LogisticRegression, QDA, QuantileRegressor, Ridge, RidgeClassifier,
};
use ferrolearn_neighbors::{KNeighborsClassifier, KNeighborsRegressor, NearestCentroid};
use ferrolearn_preprocess::{
    BinEncoding, BinStrategy, KBinsDiscretizer, MaxAbsScaler, MinMaxScaler, Normalizer,
    PolynomialFeatures, PowerTransformer, RobustScaler, StandardScaler,
    normalizer::NormType,
};
use ferrolearn_tree::{
    AdaBoostClassifier, BaggingClassifier, DecisionTreeClassifier, DecisionTreeRegressor,
    ExtraTreeClassifier, ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier,
    GradientBoostingRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    IsolationForest, RandomForestClassifier, RandomForestRegressor,
};

use ndarray::{Array1, Array2};

const SIZES: &[(&str, usize, usize)] = &[
    ("tiny_50x5", 50, 5),
    ("small_1Kx10", 1_000, 10),
    ("medium_10Kx100", 10_000, 100),
];

const KERNEL_SIZES: &[(&str, usize, usize)] = &[
    ("tiny_50x5", 50, 5),
    ("small_500x10", 500, 10),
    ("medium_2Kx20", 2_000, 20),
];

const CLUSTER_SIZES: &[(&str, usize, usize)] = &[
    ("tiny_200x5", 200, 5),
    ("small_1Kx10", 1_000, 10),
    ("medium_5Kx20", 5_000, 20),
];

fn fast_median<F: FnMut()>(f: F) -> f64 {
    median_micros(f, 3, 25, Duration::from_millis(1500))
}

fn slow_once<F: FnMut()>(f: F) -> f64 {
    time_once_micros(f)
}

fn frobenius_norm(a: &Array2<f64>) -> f64 {
    a.iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn relative_recon(x: &Array2<f64>, x_hat: &Array2<f64>) -> f64 {
    let resid = x - x_hat;
    frobenius_norm(&resid) / frobenius_norm(x).max(1e-30)
}

fn ari_usize(y_true: &Array1<usize>, labels: &Array1<usize>) -> Option<f64> {
    let truth: Array1<isize> = y_true.mapv(|v| v as isize);
    let pred: Array1<isize> = labels.mapv(|v| v as isize);
    clustering::adjusted_rand_score(&truth, &pred).ok()
}

fn ari_isize(y_true: &Array1<usize>, labels: &Array1<isize>) -> Option<f64> {
    let truth: Array1<isize> = y_true.mapv(|v| v as isize);
    clustering::adjusted_rand_score(&truth, labels).ok()
}

// ---------------------------------------------------------------------------
// Macros for fit+predict timing on concrete types.
// ---------------------------------------------------------------------------

macro_rules! reg_bench {
    ($records:ident, $name:expr, $label:expr, $n:expr, $p:expr,
     $xtr:expr, $xte:expr, $ytr:expr, $yte:expr, $build:expr) => {{
        let mut rec = BenchRecord::new("regressor", $name, $label, $n, $p);
        rec.fit_us = if $n <= 1_000 {
            fast_median(|| {
                let _ = $build.fit($xtr, $ytr).unwrap();
            })
        } else {
            slow_once(|| {
                let _ = $build.fit($xtr, $ytr).unwrap();
            })
        };
        let fitted = $build.fit($xtr, $ytr).unwrap();
        rec.predict_us = Some(fast_median(|| {
            let _ = fitted.predict($xte).unwrap();
        }));
        let yhat = fitted.predict($xte).unwrap();
        rec.metric = Some("r2".into());
        rec.score = regression::r2_score($yte, &yhat).ok();
        $records.push(rec);
    }};
}

macro_rules! cls_bench {
    ($records:ident, $name:expr, $label:expr, $n:expr, $p:expr,
     $xtr:expr, $xte:expr, $ytr:expr, $yte:expr, $build:expr) => {{
        let mut rec = BenchRecord::new("classifier", $name, $label, $n, $p);
        rec.fit_us = if $n <= 1_000 {
            fast_median(|| {
                let _ = $build.fit($xtr, $ytr).unwrap();
            })
        } else {
            slow_once(|| {
                let _ = $build.fit($xtr, $ytr).unwrap();
            })
        };
        let fitted = $build.fit($xtr, $ytr).unwrap();
        rec.predict_us = Some(fast_median(|| {
            let _ = fitted.predict($xte).unwrap();
        }));
        let yhat = fitted.predict($xte).unwrap();
        rec.metric = Some("accuracy".into());
        rec.score = classification::accuracy_score($yte, &yhat).ok();
        $records.push(rec);
    }};
}

// ---------------------------------------------------------------------------
// Regressors
// ---------------------------------------------------------------------------

fn bench_regressors(records: &mut Vec<BenchRecord>) {
    for &(label, n, p) in SIZES {
        let (x, y) = regression_data(n, p);
        let (xtr, xte, ytr, yte) = split_regression(&x, &y);

        reg_bench!(records, "LinearRegression", label, n, p,
            &xtr, &xte, &ytr, &yte, LinearRegression::<f64>::new());
        reg_bench!(records, "Ridge", label, n, p,
            &xtr, &xte, &ytr, &yte, Ridge::<f64>::new());
        reg_bench!(records, "Lasso", label, n, p,
            &xtr, &xte, &ytr, &yte, Lasso::<f64>::new());
        reg_bench!(records, "ElasticNet", label, n, p,
            &xtr, &xte, &ytr, &yte, ElasticNet::<f64>::new());
        reg_bench!(records, "BayesianRidge", label, n, p,
            &xtr, &xte, &ytr, &yte, BayesianRidge::<f64>::new());
        reg_bench!(records, "ARDRegression", label, n, p,
            &xtr, &xte, &ytr, &yte, ARDRegression::<f64>::new());
        if n <= 1_000 {
            reg_bench!(records, "HuberRegressor", label, n, p,
                &xtr, &xte, &ytr, &yte, HuberRegressor::<f64>::new());
            reg_bench!(records, "QuantileRegressor", label, n, p,
                &xtr, &xte, &ytr, &yte, QuantileRegressor::<f64>::new());
        }
        reg_bench!(records, "DecisionTreeRegressor", label, n, p,
            &xtr, &xte, &ytr, &yte, DecisionTreeRegressor::<f64>::new());
        reg_bench!(records, "RandomForestRegressor", label, n, p,
            &xtr, &xte, &ytr, &yte,
            RandomForestRegressor::<f64>::new().with_random_state(42));
        reg_bench!(records, "ExtraTreesRegressor", label, n, p,
            &xtr, &xte, &ytr, &yte,
            ExtraTreesRegressor::<f64>::new().with_random_state(42));
        if n <= 1_000 {
            reg_bench!(records, "GradientBoostingRegressor", label, n, p,
                &xtr, &xte, &ytr, &yte,
                GradientBoostingRegressor::<f64>::new().with_random_state(42));
        }
        reg_bench!(records, "HistGradientBoostingRegressor", label, n, p,
            &xtr, &xte, &ytr, &yte,
            HistGradientBoostingRegressor::<f64>::new().with_random_state(42));
        reg_bench!(records, "KNeighborsRegressor", label, n, p,
            &xtr, &xte, &ytr, &yte, KNeighborsRegressor::<f64>::new());
        if n <= 2_000 {
            reg_bench!(records, "KernelRidge", label, n, p,
                &xtr, &xte, &ytr, &yte, KernelRidge::<f64>::new());
        }
    }
}

// ---------------------------------------------------------------------------
// Classifiers
// ---------------------------------------------------------------------------

fn bench_classifiers(records: &mut Vec<BenchRecord>) {
    for &(label, n, p) in SIZES {
        let (x, y) = classification_data(n, p);
        let (xtr, xte, ytr, yte) = split_classification(&x, &y);

        cls_bench!(records, "LogisticRegression", label, n, p,
            &xtr, &xte, &ytr, &yte, LogisticRegression::<f64>::new());
        cls_bench!(records, "RidgeClassifier", label, n, p,
            &xtr, &xte, &ytr, &yte, RidgeClassifier::<f64>::new());
        cls_bench!(records, "LinearSVC", label, n, p,
            &xtr, &xte, &ytr, &yte, LinearSVC::<f64>::new());
        cls_bench!(records, "QDA", label, n, p,
            &xtr, &xte, &ytr, &yte, QDA::<f64>::new());
        cls_bench!(records, "GaussianNB", label, n, p,
            &xtr, &xte, &ytr, &yte, GaussianNB::<f64>::new());
        cls_bench!(records, "DecisionTreeClassifier", label, n, p,
            &xtr, &xte, &ytr, &yte, DecisionTreeClassifier::<f64>::new());
        cls_bench!(records, "ExtraTreeClassifier", label, n, p,
            &xtr, &xte, &ytr, &yte, ExtraTreeClassifier::<f64>::new());
        cls_bench!(records, "RandomForestClassifier", label, n, p,
            &xtr, &xte, &ytr, &yte,
            RandomForestClassifier::<f64>::new().with_random_state(42));
        cls_bench!(records, "ExtraTreesClassifier", label, n, p,
            &xtr, &xte, &ytr, &yte,
            ExtraTreesClassifier::<f64>::new().with_random_state(42));
        if n <= 1_000 {
            cls_bench!(records, "AdaBoostClassifier", label, n, p,
                &xtr, &xte, &ytr, &yte,
                AdaBoostClassifier::<f64>::new().with_random_state(42));
            cls_bench!(records, "BaggingClassifier", label, n, p,
                &xtr, &xte, &ytr, &yte,
                BaggingClassifier::<f64>::new().with_random_state(42));
            cls_bench!(records, "GradientBoostingClassifier", label, n, p,
                &xtr, &xte, &ytr, &yte,
                GradientBoostingClassifier::<f64>::new().with_random_state(42));
        }
        cls_bench!(records, "HistGradientBoostingClassifier", label, n, p,
            &xtr, &xte, &ytr, &yte,
            HistGradientBoostingClassifier::<f64>::new().with_random_state(42));
        cls_bench!(records, "KNeighborsClassifier", label, n, p,
            &xtr, &xte, &ytr, &yte, KNeighborsClassifier::<f64>::new());
        cls_bench!(records, "NearestCentroid", label, n, p,
            &xtr, &xte, &ytr, &yte, NearestCentroid::<f64>::new());

        // Non-negative-feature NB variants.
        let xtr_pos = xtr.mapv(f64::abs);
        let xte_pos = xte.mapv(f64::abs);
        cls_bench!(records, "MultinomialNB", label, n, p,
            &xtr_pos, &xte_pos, &ytr, &yte, MultinomialNB::<f64>::new());
        cls_bench!(records, "ComplementNB", label, n, p,
            &xtr_pos, &xte_pos, &ytr, &yte, ComplementNB::<f64>::new());

        // Bernoulli NB needs binary features.
        let xtr_bin = xtr.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });
        let xte_bin = xte.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });
        cls_bench!(records, "BernoulliNB", label, n, p,
            &xtr_bin, &xte_bin, &ytr, &yte, BernoulliNB::<f64>::new());
    }

    // Multi-class addition (one-vs-rest path).
    let (label, n, p) = ("multiclass_2Kx20", 2_000, 20);
    let (x, y) = multiclass_data(n, p, 5);
    let (xtr, xte, ytr, yte) = split_classification(&x, &y);
    cls_bench!(records, "LogisticRegression(5class)", label, n, p,
        &xtr, &xte, &ytr, &yte, LogisticRegression::<f64>::new());
    cls_bench!(records, "RandomForestClassifier(5class)", label, n, p,
        &xtr, &xte, &ytr, &yte,
        RandomForestClassifier::<f64>::new().with_random_state(42));
}

// ---------------------------------------------------------------------------
// Clusterers
// ---------------------------------------------------------------------------

fn bench_clusterers(records: &mut Vec<BenchRecord>) {
    for &(label, n, p) in CLUSTER_SIZES {
        let (x, y_true) = clustering_data(n, p);

        // KMeans
        {
            let mut rec = BenchRecord::new("cluster", "KMeans", label, n, p);
            rec.fit_us = if n <= 1_000 {
                fast_median(|| {
                    let _ = KMeans::<f64>::new(8)
                        .with_random_state(42)
                        .with_n_init(3)
                        .fit(&x, &())
                        .unwrap();
                })
            } else {
                slow_once(|| {
                    let _ = KMeans::<f64>::new(8)
                        .with_random_state(42)
                        .with_n_init(3)
                        .fit(&x, &())
                        .unwrap();
                })
            };
            let fitted = KMeans::<f64>::new(8)
                .with_random_state(42)
                .with_n_init(3)
                .fit(&x, &())
                .unwrap();
            let labels: Array1<usize> = fitted.predict(&x).unwrap();
            rec.metric = Some("ari".into());
            rec.score = ari_usize(&y_true, &labels);
            records.push(rec);
        }

        // MiniBatchKMeans
        {
            let mut rec = BenchRecord::new("cluster", "MiniBatchKMeans", label, n, p);
            rec.fit_us = if n <= 1_000 {
                fast_median(|| {
                    let _ = MiniBatchKMeans::<f64>::new(8)
                        .with_random_state(42)
                        .fit(&x, &())
                        .unwrap();
                })
            } else {
                slow_once(|| {
                    let _ = MiniBatchKMeans::<f64>::new(8)
                        .with_random_state(42)
                        .fit(&x, &())
                        .unwrap();
                })
            };
            let fitted = MiniBatchKMeans::<f64>::new(8)
                .with_random_state(42)
                .fit(&x, &())
                .unwrap();
            let labels: Array1<usize> = fitted.predict(&x).unwrap();
            rec.metric = Some("ari".into());
            rec.score = ari_usize(&y_true, &labels);
            records.push(rec);
        }

        // BisectingKMeans (predict returns isize)
        {
            let mut rec = BenchRecord::new("cluster", "BisectingKMeans", label, n, p);
            rec.fit_us = if n <= 1_000 {
                fast_median(|| {
                    let _ = BisectingKMeans::<f64>::new(8)
                        .with_random_state(42)
                        .fit(&x, &())
                        .unwrap();
                })
            } else {
                slow_once(|| {
                    let _ = BisectingKMeans::<f64>::new(8)
                        .with_random_state(42)
                        .fit(&x, &())
                        .unwrap();
                })
            };
            let fitted = BisectingKMeans::<f64>::new(8)
                .with_random_state(42)
                .fit(&x, &())
                .unwrap();
            let labels: Array1<isize> = fitted.predict(&x).unwrap();
            rec.metric = Some("ari".into());
            rec.score = ari_isize(&y_true, &labels);
            records.push(rec);
        }

        // GaussianMixture
        {
            let mut rec = BenchRecord::new("cluster", "GaussianMixture", label, n, p);
            rec.fit_us = if n <= 1_000 {
                fast_median(|| {
                    let _ = GaussianMixture::<f64>::new(8)
                        .with_random_state(42)
                        .fit(&x, &())
                        .unwrap();
                })
            } else {
                slow_once(|| {
                    let _ = GaussianMixture::<f64>::new(8)
                        .with_random_state(42)
                        .fit(&x, &())
                        .unwrap();
                })
            };
            let fitted = GaussianMixture::<f64>::new(8)
                .with_random_state(42)
                .fit(&x, &())
                .unwrap();
            let labels: Array1<usize> = fitted.predict(&x).unwrap();
            rec.metric = Some("ari".into());
            rec.score = ari_usize(&y_true, &labels);
            records.push(rec);
        }

        // O(n²) algorithms — restrict to small sizes.
        if n <= 1_000 {
            {
                let mut rec =
                    BenchRecord::new("cluster", "AgglomerativeClustering", label, n, p);
                rec.fit_us = slow_once(|| {
                    let _ = AgglomerativeClustering::<f64>::new(8).fit(&x, &()).unwrap();
                });
                let fitted = AgglomerativeClustering::<f64>::new(8).fit(&x, &()).unwrap();
                let labels = fitted.labels().clone();
                rec.metric = Some("ari".into());
                rec.score = ari_usize(&y_true, &labels);
                records.push(rec);
            }
            {
                let mut rec = BenchRecord::new("cluster", "SpectralClustering", label, n, p);
                rec.fit_us = slow_once(|| {
                    let _ = SpectralClustering::<f64>::new(8).fit(&x, &()).unwrap();
                });
                let fitted = SpectralClustering::<f64>::new(8).fit(&x, &()).unwrap();
                let labels = fitted.labels().clone();
                rec.metric = Some("ari".into());
                rec.score = ari_usize(&y_true, &labels);
                records.push(rec);
            }
            {
                let mut rec = BenchRecord::new("cluster", "DBSCAN", label, n, p);
                rec.fit_us = slow_once(|| {
                    let _ = DBSCAN::<f64>::new(1.0).fit(&x, &()).unwrap();
                });
                let fitted = DBSCAN::<f64>::new(1.0).fit(&x, &()).unwrap();
                let labels = fitted.labels().clone();
                rec.metric = Some("ari".into());
                rec.score = ari_isize(&y_true, &labels);
                records.push(rec);
            }
            {
                let mut rec = BenchRecord::new("cluster", "Birch", label, n, p);
                rec.fit_us = slow_once(|| {
                    let _ = Birch::<f64>::new().with_n_clusters(8).fit(&x, &()).unwrap();
                });
                let fitted = Birch::<f64>::new().with_n_clusters(8).fit(&x, &()).unwrap();
                let labels = fitted.labels().clone();
                rec.metric = Some("ari".into());
                rec.score = ari_usize(&y_true, &labels);
                records.push(rec);
            }
            {
                let mut rec = BenchRecord::new("cluster", "MeanShift", label, n, p);
                rec.fit_us = slow_once(|| {
                    let _ = MeanShift::<f64>::new().fit(&x, &()).unwrap();
                });
                let fitted = MeanShift::<f64>::new().fit(&x, &()).unwrap();
                let labels = fitted.labels().clone();
                rec.metric = Some("ari".into());
                rec.score = ari_usize(&y_true, &labels);
                records.push(rec);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Decomposition — quality = relative reconstruction error where supported.
// ---------------------------------------------------------------------------

fn bench_decomp(records: &mut Vec<BenchRecord>) {
    for &(label, n, p) in SIZES {
        let (x, _) = regression_data(n, p);
        // sklearn's TruncatedSVD requires n_components < n_features (strict);
        // clamp to p-1 so the same value works for every decomposer.
        let n_comp = p.saturating_sub(1).clamp(1, 5);

        // PCA (has inverse_transform)
        {
            let mut rec = BenchRecord::new("decomp", "PCA", label, n, p);
            rec.fit_us = if n <= 1_000 {
                fast_median(|| {
                    let _ = PCA::<f64>::new(n_comp).fit(&x, &()).unwrap();
                })
            } else {
                slow_once(|| {
                    let _ = PCA::<f64>::new(n_comp).fit(&x, &()).unwrap();
                })
            };
            let fitted = PCA::<f64>::new(n_comp).fit(&x, &()).unwrap();
            rec.predict_us = Some(fast_median(|| {
                let _ = fitted.transform(&x).unwrap();
            }));
            let z = fitted.transform(&x).unwrap();
            let x_hat = fitted.inverse_transform(&z).unwrap();
            rec.metric = Some("recon_rel".into());
            rec.score = Some(relative_recon(&x, &x_hat));
            records.push(rec);
        }
        {
            let mut rec = BenchRecord::new("decomp", "IncrementalPCA", label, n, p);
            rec.fit_us = if n <= 1_000 {
                fast_median(|| {
                    let _ = IncrementalPCA::<f64>::new(n_comp).fit(&x, &()).unwrap();
                })
            } else {
                slow_once(|| {
                    let _ = IncrementalPCA::<f64>::new(n_comp).fit(&x, &()).unwrap();
                })
            };
            let fitted = IncrementalPCA::<f64>::new(n_comp).fit(&x, &()).unwrap();
            rec.predict_us = Some(fast_median(|| {
                let _ = fitted.transform(&x).unwrap();
            }));
            let z = fitted.transform(&x).unwrap();
            let x_hat = fitted.inverse_transform(&z).unwrap();
            rec.metric = Some("recon_rel".into());
            rec.score = Some(relative_recon(&x, &x_hat));
            records.push(rec);
        }
        {
            let mut rec = BenchRecord::new("decomp", "TruncatedSVD", label, n, p);
            rec.fit_us = if n <= 1_000 {
                fast_median(|| {
                    let _ = TruncatedSVD::<f64>::new(n_comp).fit(&x, &()).unwrap();
                })
            } else {
                slow_once(|| {
                    let _ = TruncatedSVD::<f64>::new(n_comp).fit(&x, &()).unwrap();
                })
            };
            let fitted = TruncatedSVD::<f64>::new(n_comp).fit(&x, &()).unwrap();
            rec.predict_us = Some(fast_median(|| {
                let _ = fitted.transform(&x).unwrap();
            }));
            let z = fitted.transform(&x).unwrap();
            let x_hat = fitted.inverse_transform(&z).unwrap();
            rec.metric = Some("recon_rel".into());
            rec.score = Some(relative_recon(&x, &x_hat));
            records.push(rec);
        }
        if n <= 1_000 {
            {
                let mut rec = BenchRecord::new("decomp", "FactorAnalysis", label, n, p);
                rec.fit_us = fast_median(|| {
                    let _ = FactorAnalysis::<f64>::new(n_comp).fit(&x, &()).unwrap();
                });
                let fitted = FactorAnalysis::<f64>::new(n_comp).fit(&x, &()).unwrap();
                rec.predict_us = Some(fast_median(|| {
                    let _ = fitted.transform(&x).unwrap();
                }));
                let z = fitted.transform(&x).unwrap();
                let x_hat = fitted.inverse_transform(&z).unwrap();
                rec.metric = Some("recon_rel".into());
                rec.score = Some(relative_recon(&x, &x_hat));
                records.push(rec);
            }

            macro_rules! transform_only {
                ($name:expr, $build:expr) => {{
                    let mut rec = BenchRecord::new("decomp", $name, label, n, p);
                    rec.fit_us = fast_median(|| {
                        let _ = $build.fit(&x, &()).unwrap();
                    });
                    let fitted = $build.fit(&x, &()).unwrap();
                    rec.predict_us = Some(fast_median(|| {
                        let _ = fitted.transform(&x).unwrap();
                    }));
                    records.push(rec);
                }};
            }
            transform_only!("FastICA", FastICA::<f64>::new(n_comp));
            transform_only!("KernelPCA", KernelPCA::<f64>::new(n_comp));
            transform_only!("SparsePCA", SparsePCA::<f64>::new(n_comp));

            // NMF on |X|
            let x_pos = x.mapv(f64::abs);
            let mut rec = BenchRecord::new("decomp", "NMF", label, n, p);
            rec.fit_us = fast_median(|| {
                let _ = NMF::<f64>::new(n_comp).fit(&x_pos, &()).unwrap();
            });
            let fitted = NMF::<f64>::new(n_comp).fit(&x_pos, &()).unwrap();
            rec.predict_us = Some(fast_median(|| {
                let _ = fitted.transform(&x_pos).unwrap();
            }));
            records.push(rec);
        }
    }
}

// ---------------------------------------------------------------------------
// Preprocess — timings only.
// ---------------------------------------------------------------------------

fn bench_preprocess(records: &mut Vec<BenchRecord>) {
    for &(label, n, p) in SIZES {
        let (x, _) = regression_data(n, p);

        macro_rules! state_xform {
            ($name:expr, $build:expr) => {{
                let mut rec = BenchRecord::new("preprocess", $name, label, n, p);
                rec.fit_us = fast_median(|| {
                    let _ = $build.fit(&x, &()).unwrap();
                });
                let fitted = $build.fit(&x, &()).unwrap();
                rec.predict_us = Some(fast_median(|| {
                    let _ = fitted.transform(&x).unwrap();
                }));
                records.push(rec);
            }};
        }

        state_xform!("StandardScaler", StandardScaler::<f64>::new());
        state_xform!("MinMaxScaler", MinMaxScaler::<f64>::new());
        state_xform!("MaxAbsScaler", MaxAbsScaler::<f64>::new());
        state_xform!("RobustScaler", RobustScaler::<f64>::new());
        if n <= 1_000 {
            state_xform!("PowerTransformer", PowerTransformer::<f64>::new());
        }
        state_xform!(
            "KBinsDiscretizer",
            KBinsDiscretizer::<f64>::new(5, BinEncoding::Ordinal, BinStrategy::Uniform)
        );

        // Stateless transformers
        let normalizer = Normalizer::<f64>::new(NormType::L2);
        let mut rec = BenchRecord::new("preprocess", "Normalizer(L2)", label, n, p);
        rec.fit_us = 0.0;
        rec.predict_us = Some(fast_median(|| {
            let _ = normalizer.transform(&x).unwrap();
        }));
        records.push(rec);

        let poly = PolynomialFeatures::<f64>::new(2, false, false).unwrap();
        let mut rec = BenchRecord::new("preprocess", "PolynomialFeatures(d=2)", label, n, p);
        rec.fit_us = 0.0;
        rec.predict_us = Some(fast_median(|| {
            let _ = poly.transform(&x).unwrap();
        }));
        records.push(rec);
    }
}

// ---------------------------------------------------------------------------
// Kernel methods
// ---------------------------------------------------------------------------

fn bench_kernel_methods(records: &mut Vec<BenchRecord>) {
    for &(label, n, p) in KERNEL_SIZES {
        let (x, y) = regression_data(n, p);
        let (xtr, xte, ytr, yte) = split_regression(&x, &y);

        {
            let mut rec = BenchRecord::new("kernel", "KernelRidge", label, n, p);
            rec.fit_us = if n <= 500 {
                fast_median(|| {
                    let _ = KernelRidge::<f64>::new().fit(&xtr, &ytr).unwrap();
                })
            } else {
                slow_once(|| {
                    let _ = KernelRidge::<f64>::new().fit(&xtr, &ytr).unwrap();
                })
            };
            let fitted = KernelRidge::<f64>::new().fit(&xtr, &ytr).unwrap();
            rec.predict_us = Some(fast_median(|| {
                let _ = fitted.predict(&xte).unwrap();
            }));
            let yhat = fitted.predict(&xte).unwrap();
            rec.metric = Some("r2".into());
            rec.score = regression::r2_score(&yte, &yhat).ok();
            records.push(rec);
        }
        {
            let mut rec = BenchRecord::new("kernel", "Nystroem", label, n, p);
            rec.fit_us = fast_median(|| {
                let _ = Nystroem::<f64>::new().fit(&xtr, &()).unwrap();
            });
            let fitted = Nystroem::<f64>::new().fit(&xtr, &()).unwrap();
            rec.predict_us = Some(fast_median(|| {
                let _ = fitted.transform(&xte).unwrap();
            }));
            records.push(rec);
        }
        {
            let mut rec = BenchRecord::new("kernel", "RBFSampler", label, n, p);
            rec.fit_us = fast_median(|| {
                let _ = RBFSampler::<f64>::new().fit(&xtr, &()).unwrap();
            });
            let fitted = RBFSampler::<f64>::new().fit(&xtr, &()).unwrap();
            rec.predict_us = Some(fast_median(|| {
                let _ = fitted.transform(&xte).unwrap();
            }));
            records.push(rec);
        }
    }
}

// ---------------------------------------------------------------------------
// Outlier detection
// ---------------------------------------------------------------------------

fn bench_outlier(records: &mut Vec<BenchRecord>) {
    for &(label, n, p) in &SIZES[..2] {
        let (x, _) = classification_data(n, p);
        let mut rec = BenchRecord::new("outlier", "IsolationForest", label, n, p);
        rec.fit_us = fast_median(|| {
            let _ = IsolationForest::<f64>::new()
                .with_random_state(42)
                .fit(&x, &())
                .unwrap();
        });
        let fitted = IsolationForest::<f64>::new()
            .with_random_state(42)
            .fit(&x, &())
            .unwrap();
        rec.predict_us = Some(fast_median(|| {
            let _ = fitted.predict(&x).unwrap();
        }));
        records.push(rec);
    }
}

fn main() {
    let mut records: Vec<BenchRecord> = Vec::new();

    eprintln!("[harness] regressors...");
    bench_regressors(&mut records);
    eprintln!("[harness] classifiers...");
    bench_classifiers(&mut records);
    eprintln!("[harness] clusterers...");
    bench_clusterers(&mut records);
    eprintln!("[harness] decomp...");
    bench_decomp(&mut records);
    eprintln!("[harness] preprocess...");
    bench_preprocess(&mut records);
    eprintln!("[harness] kernel methods...");
    bench_kernel_methods(&mut records);
    eprintln!("[harness] outlier...");
    bench_outlier(&mut records);

    let json = serde_json::to_string_pretty(&records).expect("JSON serialization");
    println!("{json}");
    eprintln!("[harness] {} records emitted", records.len());
}
