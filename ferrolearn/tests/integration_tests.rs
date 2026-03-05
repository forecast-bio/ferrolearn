//! End-to-end integration tests for the ferrolearn workspace.
//!
//! These tests verify that the full ML workflow works across crates,
//! exercising datasets, preprocessing, models, metrics, model selection,
//! and I/O in realistic pipelines.

use ndarray::Array2;

use ferrolearn::datasets;
use ferrolearn::preprocess::{self, StandardScaler};
use ferrolearn::{Fit, FitTransform, Pipeline, Predict, Transform};

// =========================================================================
// 1. Classification Pipeline E2E
//    load_iris → StandardScaler → PCA(2) → LogisticRegression → accuracy
// =========================================================================

#[test]
fn test_classification_pipeline_e2e() {
    // Load dataset.
    let (x, y) = datasets::load_iris::<f64>().expect("failed to load iris");
    assert_eq!(x.nrows(), 150);
    assert_eq!(x.ncols(), 4);
    assert_eq!(y.len(), 150);

    // Step 1: StandardScaler.
    let scaler = StandardScaler::<f64>::new();
    let fitted_scaler = scaler.fit(&x, &()).expect("scaler fit failed");
    let x_scaled = fitted_scaler
        .transform(&x)
        .expect("scaler transform failed");

    // Step 2: PCA(2 components).
    let pca = ferrolearn::decomp::PCA::<f64>::new(2);
    let fitted_pca = pca.fit(&x_scaled, &()).expect("PCA fit failed");
    let x_pca = fitted_pca
        .transform(&x_scaled)
        .expect("PCA transform failed");
    assert_eq!(x_pca.ncols(), 2);
    assert_eq!(x_pca.nrows(), 150);

    // Step 3: LogisticRegression.
    let lr = ferrolearn::linear::LogisticRegression::<f64>::new()
        .with_max_iter(2000)
        .with_c(10.0);
    let fitted_lr = lr.fit(&x_pca, &y).expect("LogisticRegression fit failed");
    let y_pred = fitted_lr.predict(&x_pca).expect("predict failed");

    // Evaluate accuracy.
    let accuracy = ferrolearn::metrics::accuracy_score(&y, &y_pred).expect("accuracy_score failed");

    assert!(
        accuracy > 0.8,
        "Expected accuracy > 0.8 on iris, got {accuracy}",
    );
}

// =========================================================================
// 2. Regression Pipeline E2E
//    load_diabetes → StandardScaler → Ridge → R²
// =========================================================================

#[test]
fn test_regression_pipeline_e2e() {
    // Load dataset.
    let (x, y) = datasets::load_diabetes::<f64>().expect("failed to load diabetes");
    assert_eq!(x.nrows(), 442);
    assert_eq!(x.ncols(), 10);

    // Step 1: StandardScaler.
    let scaler = StandardScaler::<f64>::new();
    let fitted_scaler = scaler.fit(&x, &()).expect("scaler fit failed");
    let x_scaled = fitted_scaler
        .transform(&x)
        .expect("scaler transform failed");

    // Step 2: Ridge regression.
    let ridge = ferrolearn::linear::Ridge::<f64>::new().with_alpha(1.0);
    let fitted_ridge = ridge.fit(&x_scaled, &y).expect("Ridge fit failed");
    let y_pred = fitted_ridge.predict(&x_scaled).expect("predict failed");

    // Evaluate R².
    let r2 = ferrolearn::metrics::r2_score(&y, &y_pred).expect("r2_score failed");

    assert!(r2 > 0.3, "Expected R² > 0.3 on diabetes, got {r2}",);
}

// =========================================================================
// 3. Clustering E2E
//    make_blobs(300, centers=3) → StandardScaler → KMeans(3) → silhouette
// =========================================================================

#[test]
fn test_clustering_e2e() {
    // Generate synthetic blobs.
    let (x, _y) = datasets::make_blobs::<f64>(300, 2, 3, 1.0, Some(42)).expect("make_blobs failed");
    assert_eq!(x.nrows(), 300);
    assert_eq!(x.ncols(), 2);

    // Step 1: StandardScaler.
    let scaler = StandardScaler::<f64>::new();
    let x_scaled = scaler
        .fit_transform(&x)
        .expect("scaler fit_transform failed");

    // Step 2: KMeans.
    let kmeans = ferrolearn::cluster::KMeans::<f64>::new(3)
        .with_random_state(42)
        .with_n_init(5);
    let fitted_kmeans = kmeans.fit(&x_scaled, &()).expect("KMeans fit failed");
    let labels = fitted_kmeans
        .predict(&x_scaled)
        .expect("KMeans predict failed");
    assert_eq!(labels.len(), 300);

    // Compute silhouette score. The silhouette_score function expects
    // Array1<isize> labels, so convert from usize.
    let labels_isize = labels.mapv(|v| v as isize);
    let silhouette = ferrolearn::metrics::silhouette_score(&x_scaled, &labels_isize)
        .expect("silhouette_score failed");

    assert!(
        silhouette > 0.5,
        "Expected silhouette > 0.5 on well-separated blobs, got {silhouette}",
    );
}

// =========================================================================
// 4. Cross-Validation E2E
//    load_wine → Pipeline(StandardScaler, LogisticRegression) → 5-fold CV
// =========================================================================

#[test]
fn test_cross_validation_e2e() {
    use ferrolearn::Pipeline;
    use ferrolearn::model_selection::{KFold, cross_val_score};

    // Load dataset.
    let (x, y) = datasets::load_wine::<f64>().expect("failed to load wine");

    // Convert integer labels to f64 for the pipeline interface.
    let y_f64 = y.mapv(|v| v as f64);

    // Build pipeline: StandardScaler → LogisticRegression.
    let pipeline = Pipeline::new()
        .transform_step("scaler", Box::new(StandardScaler::<f64>::new()))
        .estimator_step(
            "clf",
            Box::new(
                ferrolearn::linear::LogisticRegression::<f64>::new()
                    .with_max_iter(3000)
                    .with_c(10.0),
            ),
        );

    // 5-fold cross-validation with an accuracy-like scoring function.
    // cross_val_score expects fn(&Array1<f64>, &Array1<f64>) -> Result<f64, FerroError>
    // where predictions are f64. We compare rounded predictions.
    fn accuracy_scoring(
        y_true: &ndarray::Array1<f64>,
        y_pred: &ndarray::Array1<f64>,
    ) -> Result<f64, ferrolearn::FerroError> {
        let n = y_true.len();
        if n == 0 {
            return Err(ferrolearn::FerroError::InsufficientSamples {
                required: 1,
                actual: 0,
                context: "accuracy_scoring".into(),
            });
        }
        let correct = y_true
            .iter()
            .zip(y_pred.iter())
            .filter(|(a, b)| (*a - *b).abs() < 0.5)
            .count();
        Ok(correct as f64 / n as f64)
    }

    let kfold = KFold::new(5).shuffle(true).random_state(42);
    let scores = cross_val_score(&pipeline, &x, &y_f64, &kfold, accuracy_scoring)
        .expect("cross_val_score failed");

    // Verify we got 5 scores.
    assert_eq!(
        scores.len(),
        5,
        "Expected 5 fold scores, got {}",
        scores.len()
    );

    // Each score should be > 0.7.
    for (i, &score) in scores.iter().enumerate() {
        assert!(
            score > 0.7,
            "Fold {i}: expected accuracy > 0.7, got {score}",
        );
    }
}

// =========================================================================
// 5. Serialization Roundtrip
//    Fit LinearRegression → serialize predictions → deserialize → verify
// =========================================================================

#[test]
fn test_serialization_roundtrip() {
    use serde::{Deserialize, Serialize};

    // Build a simple model and store its predictions for comparison.
    let x = Array2::from_shape_vec(
        (5, 2),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    )
    .unwrap();
    let y = ndarray::array![5.0, 11.0, 17.0, 23.0, 29.0];

    let model = ferrolearn::linear::LinearRegression::<f64>::new();
    let fitted = model.fit(&x, &y).expect("LinearRegression fit failed");
    let preds_before = fitted.predict(&x).expect("predict failed");

    // Since FittedLinearRegression does not implement Serialize/Deserialize,
    // we serialize a model-like struct that captures the essential state and
    // verify the IO crate's roundtrip mechanism works correctly.
    #[derive(Serialize, Deserialize, PartialEq, Debug)]
    struct ModelSnapshot {
        predictions: Vec<f64>,
        n_samples: usize,
        n_features: usize,
    }

    let snapshot = ModelSnapshot {
        predictions: preds_before.to_vec(),
        n_samples: x.nrows(),
        n_features: x.ncols(),
    };

    // Serialize to bytes using ferrolearn-io.
    let bytes = ferrolearn::io::save_model_bytes(&snapshot).expect("save_model_bytes failed");

    // Deserialize from bytes.
    let loaded: ModelSnapshot =
        ferrolearn::io::load_model_bytes(&bytes).expect("load_model_bytes failed");

    assert_eq!(snapshot, loaded, "Serialization roundtrip mismatch");

    // Also test file-based roundtrip.
    let dir = std::env::temp_dir().join(format!(
        "ferrolearn_integ_test_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .subsec_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model_snapshot.flrn");

    ferrolearn::io::save_model(&snapshot, &path).expect("save_model failed");
    let loaded_file: ModelSnapshot = ferrolearn::io::load_model(&path).expect("load_model failed");

    assert_eq!(snapshot, loaded_file, "File-based roundtrip mismatch");

    // Verify predictions match the original model output.
    for (i, (orig, loaded_val)) in preds_before
        .iter()
        .zip(loaded_file.predictions.iter())
        .enumerate()
    {
        assert!(
            (orig - loaded_val).abs() < 1e-10,
            "Prediction mismatch at index {i}: original={orig}, loaded={loaded_val}",
        );
    }

    // Clean up.
    let _ = std::fs::remove_dir_all(&dir);
}

// =========================================================================
// 6. Tree Ensemble E2E
//    load_iris → train_test_split(0.2) → RandomForestClassifier → accuracy
// =========================================================================

#[test]
fn test_tree_ensemble_e2e() {
    // Load dataset.
    let (x, y) = datasets::load_iris::<f64>().expect("failed to load iris");

    // Convert labels to f64 for train_test_split (which requires Array1<F>).
    let y_f64 = y.mapv(|v| v as f64);

    // Split into train/test.
    let (x_train, x_test, y_train_f64, y_test_f64) =
        ferrolearn::model_selection::train_test_split(&x, &y_f64, 0.2, Some(42))
            .expect("train_test_split failed");

    // Convert labels back to usize for the classifier.
    let y_train = y_train_f64.mapv(|v| v as usize);
    let y_test = y_test_f64.mapv(|v| v as usize);

    assert_eq!(x_train.nrows() + x_test.nrows(), 150);

    // Build RandomForestClassifier.
    let rf = ferrolearn::tree::RandomForestClassifier::<f64>::new()
        .with_n_estimators(50)
        .with_random_state(42);
    let fitted_rf = rf.fit(&x_train, &y_train).expect("RF fit failed");
    let y_pred = fitted_rf.predict(&x_test).expect("RF predict failed");

    // Evaluate accuracy on the test set.
    let accuracy =
        ferrolearn::metrics::accuracy_score(&y_test, &y_pred).expect("accuracy_score failed");

    assert!(
        accuracy > 0.85,
        "Expected accuracy > 0.85 on iris test set, got {accuracy}",
    );
}

// =========================================================================
// 7. Preprocessing Chain E2E
//    NaN data → SimpleImputer → StandardScaler → PolynomialFeatures → shape
// =========================================================================

#[test]
fn test_preprocessing_chain_e2e() {
    // Create data with NaN values.
    let x = Array2::from_shape_vec(
        (6, 3),
        vec![
            1.0,
            f64::NAN,
            3.0,
            4.0,
            5.0,
            f64::NAN,
            7.0,
            8.0,
            9.0,
            10.0,
            11.0,
            12.0,
            13.0,
            14.0,
            15.0,
            16.0,
            17.0,
            18.0,
        ],
    )
    .unwrap();

    let n_samples = x.nrows();
    let n_features = x.ncols();
    assert_eq!(n_samples, 6);
    assert_eq!(n_features, 3);

    // Step 1: SimpleImputer (fill NaN with mean).
    let imputer = preprocess::SimpleImputer::<f64>::new(preprocess::ImputeStrategy::Mean);
    let fitted_imputer = imputer.fit(&x, &()).expect("imputer fit failed");
    let x_imputed = fitted_imputer
        .transform(&x)
        .expect("imputer transform failed");

    // Verify no NaN values remain.
    for val in x_imputed.iter() {
        assert!(
            !val.is_nan(),
            "Found NaN after imputation; imputer did not fill all missing values"
        );
    }
    assert_eq!(x_imputed.dim(), (n_samples, n_features));

    // Step 2: StandardScaler.
    let scaler = StandardScaler::<f64>::new();
    let fitted_scaler = scaler.fit(&x_imputed, &()).expect("scaler fit failed");
    let x_scaled = fitted_scaler
        .transform(&x_imputed)
        .expect("scaler transform failed");
    assert_eq!(x_scaled.dim(), (n_samples, n_features));

    // Verify columns have approximately zero mean.
    for col_idx in 0..n_features {
        let col_mean: f64 = x_scaled.column(col_idx).iter().sum::<f64>() / n_samples as f64;
        assert!(
            col_mean.abs() < 1e-10,
            "Column {col_idx} mean is {col_mean}, expected ~0 after scaling",
        );
    }

    // Step 3: PolynomialFeatures (degree=2, no interaction_only, with bias).
    let poly = preprocess::PolynomialFeatures::<f64>::new(2, false, true)
        .expect("PolynomialFeatures::new failed");
    let x_poly = poly.transform(&x_scaled).expect("poly transform failed");

    // For 3 input features, degree=2, include_bias=true:
    // output features = C(3 + 2, 2) = 10: [1, a, b, c, a², ab, ac, b², bc, c²]
    assert_eq!(x_poly.nrows(), n_samples);
    assert_eq!(
        x_poly.ncols(),
        10,
        "Expected 10 polynomial features for 3 input features at degree 2 with bias, got {}",
        x_poly.ncols()
    );

    // Verify the bias column is all ones.
    for &val in x_poly.column(0).iter() {
        assert!(
            (val - 1.0).abs() < 1e-10,
            "Bias column should be 1.0, got {val}",
        );
    }
}

// =========================================================================
// 8. Classification Pipeline FP32 E2E
//    load_iris → StandardScaler → PCA(2) → LogisticRegression → accuracy
// =========================================================================

#[test]
fn test_classification_pipeline_f32_e2e() {
    // Load dataset.
    let (x, y) = datasets::load_iris::<f32>().expect("failed to load iris");
    assert_eq!(x.nrows(), 150);
    assert_eq!(x.ncols(), 4);
    assert_eq!(y.len(), 150);

    let pipeline = Pipeline::new()
        .transform_step("scale", Box::new(preprocess::StandardScaler::<f32>::new()))
        .transform_step("pca", Box::new(ferrolearn::decomp::PCA::<f32>::new(1)))
        .estimator_step(
            "logistic regression",
            Box::new(
                ferrolearn::linear::LogisticRegression::<f32>::new()
                    .with_max_iter(2000)
                    .with_c(10.0),
            ),
        );
    let y_f32 = y.mapv(|v| v as f32);
    let fitted = pipeline.fit(&x, &y_f32).expect("fit failed");
    let y_pred = fitted.predict(&x).expect("predict failed");
    let y_pred_usize = y_pred.mapv(|v| v as usize);

    // Evaluate accuracy.
    let accuracy =
        ferrolearn::metrics::accuracy_score(&y, &y_pred_usize).expect("accuracy_score failed");

    assert!(
        accuracy > 0.8,
        "Expected accuracy > 0.8 on iris, got {accuracy}",
    );
}
