//! Proof-of-API integration test for ferrolearn-preprocess.
//!
//! Audit deliverable for crosslink #301 (under #247). Exercises every
//! public estimator end-to-end after the orphan wiring in #299. Every
//! call uses verified-from-source signatures.

use ferrolearn_core::traits::{Fit, FitTransform, Transform};
use ferrolearn_preprocess::feature_selection::SelectFromModel as FeatureSelectionSelectFromModel;
use ferrolearn_preprocess::imputer::ImputeStrategy;
use ferrolearn_preprocess::normalizer::NormType;
use ferrolearn_preprocess::{
    Binarizer, BinaryEncoder, BinEncoding, BinStrategy, CountVectorizer, Direction,
    FunctionTransformer, GaussianRandomProjection, InitialStrategy, IterativeImputer,
    KBinsDiscretizer, KNNImputer, KNNWeights, KnotStrategy, LabelBinarizer, LabelEncoder,
    MaxAbsScaler, MinMaxScaler, MultiLabelBinarizer, Normalizer, OneHotEncoder, OrdinalEncoder,
    OutputDistribution, PolynomialFeatures, PowerTransformer, QuantileTransformer, RobustScaler,
    ScoreFunc, SelectFdr, SelectFpr, SelectFwe, SelectKBest, SelectPercentile,
    SequentialFeatureSelector, SimpleImputer, SparseRandomProjection, SplineTransformer,
    StandardScaler, TargetEncoder, TfidfTransformer, VarianceThreshold, chi2, f_classif,
    f_regression,
};
use ndarray::{Array1, Array2, array};

fn small_data() -> Array2<f64> {
    Array2::from_shape_vec(
        (8, 3),
        vec![
            1.0, 10.0, 100.0, 2.0, 20.0, 200.0, 3.0, 30.0, 300.0, 4.0, 40.0, 400.0, 5.0, 50.0,
            500.0, 6.0, 60.0, 600.0, 7.0, 70.0, 700.0, 8.0, 80.0, 800.0,
        ],
    )
    .unwrap()
}

fn binary_labels_usize() -> Array1<usize> {
    array![0usize, 0, 0, 0, 1, 1, 1, 1]
}

// =============================================================================
// Scalers
// =============================================================================
#[test]
fn api_proof_scalers() {
    let x = small_data();
    let _ = StandardScaler::<f64>::new().fit_transform(&x).unwrap();
    let _ = MinMaxScaler::<f64>::new().fit_transform(&x).unwrap();
    let _ = MinMaxScaler::<f64>::with_feature_range(-1.0, 1.0)
        .unwrap()
        .fit_transform(&x)
        .unwrap();
    let _ = MaxAbsScaler::<f64>::new().fit_transform(&x).unwrap();
    let _ = RobustScaler::<f64>::new().fit_transform(&x).unwrap();

    // Normalizer is stateless: just .transform(&x).
    for norm in [NormType::L1, NormType::L2, NormType::Max] {
        let _ = Normalizer::<f64>::new(norm).transform(&x).unwrap();
    }
}

// =============================================================================
// Power / Quantile transforms
// =============================================================================
#[test]
fn api_proof_power_quantile() {
    let x = small_data();
    let _ = PowerTransformer::<f64>::new().fit_transform(&x).unwrap();
    for dist in [OutputDistribution::Uniform, OutputDistribution::Normal] {
        let _ = QuantileTransformer::<f64>::new(8, dist, 0)
            .fit_transform(&x)
            .unwrap();
    }
}

// =============================================================================
// PolynomialFeatures, Binarizer, FunctionTransformer (all stateless)
// =============================================================================
#[test]
fn api_proof_feature_engineering() {
    let x = small_data();

    let _ = PolynomialFeatures::<f64>::new(2, true, false).unwrap().transform(&x).unwrap();
    let _ = Binarizer::<f64>::new(50.0).transform(&x).unwrap();
    // FunctionTransformer takes an element-wise Fn(F) -> F.
    let _ = FunctionTransformer::<f64>::new(|v: f64| v * 2.0).transform(&x).unwrap();
}

// =============================================================================
// KBinsDiscretizer + SplineTransformer
// =============================================================================
#[test]
fn api_proof_kbins_and_splines() {
    let x = small_data();
    for strategy in [BinStrategy::Uniform, BinStrategy::Quantile, BinStrategy::KMeans] {
        for encode in [BinEncoding::Ordinal, BinEncoding::OneHot] {
            let _ = KBinsDiscretizer::<f64>::new(3, encode, strategy)
                .fit_transform(&x)
                .unwrap();
        }
    }
    for knots in [KnotStrategy::Uniform, KnotStrategy::Quantile] {
        let _ = SplineTransformer::<f64>::new(4, 3, knots).fit_transform(&x).unwrap();
    }
}

// =============================================================================
// Encoders
// =============================================================================
#[test]
fn api_proof_encoders() {
    let x_cat = Array2::from_shape_vec((4, 2), vec![0usize, 1, 1, 0, 0, 2, 2, 1]).unwrap();
    let f = OneHotEncoder::<f64>::new().fit(&x_cat, &()).unwrap();
    let _ = f.transform(&x_cat).unwrap();

    // OrdinalEncoder smoke (constructor only; per-column string fit varies).
    let _ = OrdinalEncoder::new();

    // LabelEncoder fits Array1<String>.
    let labels: Array1<String> = Array1::from(vec![
        "a".to_string(),
        "b".to_string(),
        "a".to_string(),
        "c".to_string(),
        "b".to_string(),
    ]);
    let f = LabelEncoder.fit(&labels, &()).unwrap();
    let _ = f.transform(&labels).unwrap();

    // LabelBinarizer fits Array1<usize>.
    let y = binary_labels_usize();
    let f = LabelBinarizer.fit(&y, &()).unwrap();
    let _ = f.transform(&y).unwrap();

    // MultiLabelBinarizer fits Vec<Vec<usize>>.
    let y_multi: Vec<Vec<usize>> = vec![vec![0, 1], vec![1, 2], vec![0]];
    let f = MultiLabelBinarizer.fit(&y_multi, &()).unwrap();
    let _ = f.transform(&y_multi).unwrap();

    // BinaryEncoder
    let _ = BinaryEncoder::<f64>::new().fit(&x_cat, &()).unwrap();

    // TargetEncoder
    let y_cont: Array1<f64> = array![0.0, 1.0, 0.0, 1.0];
    let f = TargetEncoder::<f64>::new(1.0).fit(&x_cat, &y_cont).unwrap();
    let _ = f.transform(&x_cat).unwrap();
}

// =============================================================================
// Imputers
// =============================================================================
#[test]
fn api_proof_imputers() {
    let x_with_nan = Array2::from_shape_vec(
        (5, 3),
        vec![
            1.0, f64::NAN, 3.0, 2.0, 2.0, f64::NAN, f64::NAN, 3.0, 1.0, 4.0, 4.0, 2.0, 5.0, 5.0,
            3.0,
        ],
    )
    .unwrap();

    for strat in [
        ImputeStrategy::Mean,
        ImputeStrategy::Median,
        ImputeStrategy::Constant(0.0),
    ] {
        let _ = SimpleImputer::<f64>::new(strat).fit_transform(&x_with_nan).unwrap();
    }

    for w in [KNNWeights::Uniform, KNNWeights::Distance] {
        let _ = KNNImputer::<f64>::new(2, w).fit_transform(&x_with_nan).unwrap();
    }

    for init in [InitialStrategy::Mean, InitialStrategy::Median] {
        let _ = IterativeImputer::<f64>::new(5, 1e-3, init)
            .fit_transform(&x_with_nan)
            .unwrap();
    }
}

// =============================================================================
// Feature selection
// =============================================================================
#[test]
fn api_proof_feature_selection() {
    let x = small_data();
    let y = binary_labels_usize();
    let y_f64 = array![0.0f64, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

    // VarianceThreshold (unsupervised)
    let f = VarianceThreshold::<f64>::new(0.0).fit(&x, &()).unwrap();
    let _ = f.transform(&x).unwrap();

    // SelectKBest / SelectPercentile (supervised, FClassif)
    let f = SelectKBest::<f64>::new(2, ScoreFunc::FClassif).fit(&x, &y).unwrap();
    let _ = f.transform(&x).unwrap();
    let f = SelectPercentile::<f64>::new(50, ScoreFunc::FClassif).fit(&x, &y).unwrap();
    let _ = f.transform(&x).unwrap();

    // SelectFpr/Fdr/Fwe take p-values (Array1<F>) — get them from f_classif first.
    let (_f_stats, p_values) = f_classif::<f64>(&x, &y).unwrap();
    let f = SelectFpr::<f64>::new(0.5).fit(&p_values, &()).unwrap();
    let _ = f.transform(&x).unwrap();
    let f = SelectFdr::<f64>::new(0.5).fit(&p_values, &()).unwrap();
    let _ = f.transform(&x).unwrap();
    let f = SelectFwe::<f64>::new(0.5).fit(&p_values, &()).unwrap();
    let _ = f.transform(&x).unwrap();

    // SelectFromModel — new_from_importances(importances, threshold: Option<F>).
    let importances = Array1::from(vec![0.1f64, 0.5, 0.9]);
    let f = FeatureSelectionSelectFromModel::<f64>::new_from_importances(
        &importances,
        Some(0.3),
    )
    .unwrap();
    let _ = f.transform(&x).unwrap();

    // SequentialFeatureSelector::fit(x, y, score_fn) — takes a scoring closure.
    let score_fn = |_x: &Array2<f64>, _y: &Array1<f64>| -> Result<f64, ferrolearn_core::error::FerroError> {
        Ok(0.0)
    };
    for dir in [Direction::Forward, Direction::Backward] {
        let _ = SequentialFeatureSelector::new(2, dir).fit(&x, &y_f64, score_fn).unwrap();
    }

    // Module-level scoring helpers.
    let (chi2_stats, _p) = chi2::<f64>(&x, &y).unwrap();
    assert_eq!(chi2_stats.len(), 3);
    let (r_stats, _p) = f_regression::<f64>(&x, &y_f64).unwrap();
    assert_eq!(r_stats.len(), 3);
}

// =============================================================================
// Text processing
// =============================================================================
#[test]
fn api_proof_text() {
    let docs: Vec<String> = vec![
        "the quick brown fox".to_string(),
        "the lazy dog".to_string(),
        "the brown dog jumps".to_string(),
    ];
    let f = CountVectorizer::new().fit(&docs).unwrap();
    let counts = f.transform(&docs).unwrap();
    assert_eq!(counts.nrows(), 3);
    let counts_f64 = counts.mapv(|v| v as f64);
    let f = TfidfTransformer::<f64>::new().fit(&counts_f64).unwrap();
    let _ = f.transform(&counts_f64).unwrap();
}

// =============================================================================
// Random projection
// =============================================================================
#[test]
fn api_proof_random_projection() {
    let x = Array2::<f64>::from_shape_vec((8, 50), (0..400).map(|i| i as f64).collect()).unwrap();
    let _ = GaussianRandomProjection::<f64>::new(10).fit_transform(&x).unwrap();
    let _ = SparseRandomProjection::<f64>::new(10).fit_transform(&x).unwrap();
}
