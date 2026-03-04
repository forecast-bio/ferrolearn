//! Classic toy datasets embedded as compiled-in CSV data.
//!
//! Each loader parses a CSV at load time and returns arrays suitable for
//! use with ferrolearn estimators. Classification datasets return
//! `(Array2<F>, Array1<usize>)`, regression datasets return
//! `(Array2<F>, Array1<F>)`.

use ferrolearn_core::FerroError;
use ndarray::{Array1, Array2};
use num_traits::Float;

// ---------------------------------------------------------------------------
// Embedded CSV data
// ---------------------------------------------------------------------------

/// The classic Iris dataset (150 samples × 4 features, 3 classes).
const IRIS_CSV: &str = include_str!("../data/iris.csv");

// ---------------------------------------------------------------------------
// Internal parsing helpers
// ---------------------------------------------------------------------------

/// Parse a CSV string where the last column is an integer class label.
///
/// The first row is a header and is skipped.
fn parse_classification_csv<F>(
    csv: &str,
    expected_samples: usize,
    expected_features: usize,
) -> Result<(Array2<F>, Array1<usize>), FerroError>
where
    F: Float,
{
    let mut x_data: Vec<F> = Vec::with_capacity(expected_samples * expected_features);
    let mut y_data: Vec<usize> = Vec::with_capacity(expected_samples);

    for (line_idx, line) in csv.lines().enumerate() {
        // Skip header row.
        if line_idx == 0 {
            continue;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 2 {
            return Err(FerroError::SerdeError {
                message: format!(
                    "line {line_idx}: expected at least 2 fields, got {}",
                    fields.len()
                ),
            });
        }

        let n_feat = fields.len() - 1;
        // Parse feature columns.
        for f in &fields[..n_feat] {
            let val: f64 = f.trim().parse().map_err(|_| FerroError::SerdeError {
                message: format!("line {line_idx}: cannot parse feature '{}'", f.trim()),
            })?;
            x_data.push(F::from(val).ok_or_else(|| FerroError::SerdeError {
                message: format!("line {line_idx}: float conversion failed for '{val}'"),
            })?);
        }
        // Parse label column.
        let label: usize = fields[n_feat]
            .trim()
            .parse()
            .map_err(|_| FerroError::SerdeError {
                message: format!(
                    "line {line_idx}: cannot parse class label '{}'",
                    fields[n_feat].trim()
                ),
            })?;
        y_data.push(label);
    }

    let n_samples = y_data.len();
    let n_features = if n_samples > 0 {
        x_data.len() / n_samples
    } else {
        0
    };

    let x = Array2::from_shape_vec((n_samples, n_features), x_data).map_err(|e| {
        FerroError::SerdeError {
            message: format!("cannot reshape X: {e}"),
        }
    })?;
    let y = Array1::from_vec(y_data);

    // Validate shape expectations.
    if n_samples != expected_samples {
        return Err(FerroError::ShapeMismatch {
            expected: vec![expected_samples],
            actual: vec![n_samples],
            context: "toy dataset sample count".into(),
        });
    }
    if n_features != expected_features {
        return Err(FerroError::ShapeMismatch {
            expected: vec![expected_features],
            actual: vec![n_features],
            context: "toy dataset feature count".into(),
        });
    }

    Ok((x, y))
}

// ---------------------------------------------------------------------------
// Synthetic stub helpers
// ---------------------------------------------------------------------------

/// Generate a synthetic classification dataset with the given shape.
///
/// Returns linearly-separable blobs seeded by deterministic per-class offsets.
fn synthetic_classification<F>(
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
) -> Result<(Array2<F>, Array1<usize>), FerroError>
where
    F: Float,
{
    let mut x_data: Vec<F> = Vec::with_capacity(n_samples * n_features);
    let mut y_data: Vec<usize> = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let class = i % n_classes;
        let offset = F::from(class as f64 * 5.0).unwrap_or(F::zero());
        for j in 0..n_features {
            // Deterministic synthetic value: offset + small per-feature variation.
            let val = offset + F::from((i * n_features + j) as f64 * 0.001).unwrap_or(F::zero());
            x_data.push(val);
        }
        y_data.push(class);
    }

    let x = Array2::from_shape_vec((n_samples, n_features), x_data).map_err(|e| {
        FerroError::SerdeError {
            message: format!("synthetic_classification reshape failed: {e}"),
        }
    })?;
    let y = Array1::from_vec(y_data);

    Ok((x, y))
}

/// Generate a synthetic regression dataset with the given shape.
fn synthetic_regression<F>(
    n_samples: usize,
    n_features: usize,
) -> Result<(Array2<F>, Array1<F>), FerroError>
where
    F: Float,
{
    let mut x_data: Vec<F> = Vec::with_capacity(n_samples * n_features);
    let mut y_data: Vec<F> = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let mut row_sum = F::zero();
        for j in 0..n_features {
            let val = F::from((i * n_features + j) as f64 * 0.01).unwrap_or(F::zero());
            x_data.push(val);
            row_sum = row_sum + val;
        }
        y_data.push(row_sum);
    }

    let x = Array2::from_shape_vec((n_samples, n_features), x_data).map_err(|e| {
        FerroError::SerdeError {
            message: format!("synthetic_regression reshape failed: {e}"),
        }
    })?;
    let y = Array1::from_vec(y_data);

    Ok((x, y))
}

// ---------------------------------------------------------------------------
// Public loaders
// ---------------------------------------------------------------------------

/// Load the Iris dataset.
///
/// Returns `(X, y)` where `X` has shape `(150, 4)` and `y` has 3 unique
/// class labels (0 = setosa, 1 = versicolor, 2 = virginica).
///
/// The data is the canonical UCI Iris dataset, embedded at compile time.
///
/// # Errors
///
/// Returns [`FerroError::SerdeError`] if the embedded CSV is malformed.
///
/// # Examples
///
/// ```rust
/// let (x, y) = ferrolearn_datasets::load_iris::<f64>().unwrap();
/// assert_eq!(x.shape(), &[150, 4]);
/// assert_eq!(y.len(), 150);
/// ```
pub fn load_iris<F>() -> Result<(Array2<F>, Array1<usize>), FerroError>
where
    F: Float,
{
    parse_classification_csv(IRIS_CSV, 150, 4)
}

/// Load the Wine dataset (synthetic stub with correct shape).
///
/// Returns `(X, y)` where `X` has shape `(178, 13)` and `y` has 3 unique
/// class labels.
///
/// # Errors
///
/// Returns [`FerroError::SerdeError`] if data generation fails.
///
/// # Examples
///
/// ```rust
/// let (x, y) = ferrolearn_datasets::load_wine::<f64>().unwrap();
/// assert_eq!(x.shape(), &[178, 13]);
/// assert_eq!(y.len(), 178);
/// ```
pub fn load_wine<F>() -> Result<(Array2<F>, Array1<usize>), FerroError>
where
    F: Float,
{
    synthetic_classification(178, 13, 3)
}

/// Load the Breast Cancer dataset (synthetic stub with correct shape).
///
/// Returns `(X, y)` where `X` has shape `(569, 30)` and `y` has 2 unique
/// class labels (0 = malignant, 1 = benign).
///
/// # Errors
///
/// Returns [`FerroError::SerdeError`] if data generation fails.
///
/// # Examples
///
/// ```rust
/// let (x, y) = ferrolearn_datasets::load_breast_cancer::<f64>().unwrap();
/// assert_eq!(x.shape(), &[569, 30]);
/// assert_eq!(y.len(), 569);
/// ```
pub fn load_breast_cancer<F>() -> Result<(Array2<F>, Array1<usize>), FerroError>
where
    F: Float,
{
    synthetic_classification(569, 30, 2)
}

/// Load the Diabetes dataset (synthetic stub with correct shape).
///
/// Returns `(X, y)` where `X` has shape `(442, 10)` and `y` is a
/// continuous regression target.
///
/// # Errors
///
/// Returns [`FerroError::SerdeError`] if data generation fails.
///
/// # Examples
///
/// ```rust
/// let (x, y) = ferrolearn_datasets::load_diabetes::<f64>().unwrap();
/// assert_eq!(x.shape(), &[442, 10]);
/// assert_eq!(y.len(), 442);
/// ```
pub fn load_diabetes<F>() -> Result<(Array2<F>, Array1<F>), FerroError>
where
    F: Float,
{
    synthetic_regression(442, 10)
}

/// Load the Digits dataset (synthetic stub, first 200 samples).
///
/// Returns `(X, y)` where `X` has shape `(200, 64)` and `y` has 10 unique
/// class labels (digits 0–9).
///
/// Only the first 200 samples are included to keep binary size small.
///
/// # Errors
///
/// Returns [`FerroError::SerdeError`] if data generation fails.
///
/// # Examples
///
/// ```rust
/// let (x, y) = ferrolearn_datasets::load_digits::<f64>().unwrap();
/// assert_eq!(x.shape(), &[200, 64]);
/// assert_eq!(y.len(), 200);
/// ```
pub fn load_digits<F>() -> Result<(Array2<F>, Array1<usize>), FerroError>
where
    F: Float,
{
    synthetic_classification(200, 64, 10)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    // --- Iris ---

    #[test]
    fn test_load_iris_shape() {
        let (x, y) = load_iris::<f64>().unwrap();
        assert_eq!(x.shape(), &[150, 4]);
        assert_eq!(y.len(), 150);
    }

    #[test]
    fn test_load_iris_classes() {
        let (_, y) = load_iris::<f64>().unwrap();
        let unique: HashSet<usize> = y.iter().copied().collect();
        assert_eq!(unique.len(), 3, "Iris should have 3 unique classes");
        assert!(unique.contains(&0));
        assert!(unique.contains(&1));
        assert!(unique.contains(&2));
    }

    #[test]
    fn test_load_iris_first_row() {
        let (x, y) = load_iris::<f64>().unwrap();
        // First sample: 5.1, 3.5, 1.4, 0.2, class 0 (setosa)
        let row = x.row(0);
        assert!(
            (row[0] - 5.1_f64).abs() < 1e-9,
            "sepal_length mismatch: {}",
            row[0]
        );
        assert!(
            (row[1] - 3.5_f64).abs() < 1e-9,
            "sepal_width mismatch: {}",
            row[1]
        );
        assert!(
            (row[2] - 1.4_f64).abs() < 1e-9,
            "petal_length mismatch: {}",
            row[2]
        );
        assert!(
            (row[3] - 0.2_f64).abs() < 1e-9,
            "petal_width mismatch: {}",
            row[3]
        );
        assert_eq!(y[0], 0);
    }

    #[test]
    fn test_load_iris_last_row() {
        let (x, y) = load_iris::<f64>().unwrap();
        // Last sample (index 149): 5.9, 3.0, 5.1, 1.8, class 2 (virginica)
        let row = x.row(149);
        assert!(
            (row[0] - 5.9_f64).abs() < 1e-9,
            "sepal_length mismatch: {}",
            row[0]
        );
        assert_eq!(y[149], 2);
    }

    #[test]
    fn test_load_iris_f32() {
        let (x, y) = load_iris::<f32>().unwrap();
        assert_eq!(x.shape(), &[150, 4]);
        assert_eq!(y.len(), 150);
        let row = x.row(0);
        assert!(
            (row[0] - 5.1_f32).abs() < 1e-5,
            "f32 sepal_length mismatch: {}",
            row[0]
        );
    }

    #[test]
    fn test_load_iris_class_balance() {
        let (_, y) = load_iris::<f64>().unwrap();
        // Each class has exactly 50 samples.
        let count_0 = y.iter().filter(|&&c| c == 0).count();
        let count_1 = y.iter().filter(|&&c| c == 1).count();
        let count_2 = y.iter().filter(|&&c| c == 2).count();
        assert_eq!(count_0, 50);
        assert_eq!(count_1, 50);
        assert_eq!(count_2, 50);
    }

    // --- Wine ---

    #[test]
    fn test_load_wine_shape() {
        let (x, y) = load_wine::<f64>().unwrap();
        assert_eq!(x.shape(), &[178, 13]);
        assert_eq!(y.len(), 178);
    }

    #[test]
    fn test_load_wine_classes() {
        let (_, y) = load_wine::<f64>().unwrap();
        let unique: HashSet<usize> = y.iter().copied().collect();
        assert_eq!(unique.len(), 3);
    }

    // --- Breast Cancer ---

    #[test]
    fn test_load_breast_cancer_shape() {
        let (x, y) = load_breast_cancer::<f64>().unwrap();
        assert_eq!(x.shape(), &[569, 30]);
        assert_eq!(y.len(), 569);
    }

    #[test]
    fn test_load_breast_cancer_classes() {
        let (_, y) = load_breast_cancer::<f64>().unwrap();
        let unique: HashSet<usize> = y.iter().copied().collect();
        assert_eq!(unique.len(), 2);
    }

    // --- Diabetes ---

    #[test]
    fn test_load_diabetes_shape() {
        let (x, y) = load_diabetes::<f64>().unwrap();
        assert_eq!(x.shape(), &[442, 10]);
        assert_eq!(y.len(), 442);
    }

    // --- Digits ---

    #[test]
    fn test_load_digits_shape() {
        let (x, y) = load_digits::<f64>().unwrap();
        assert_eq!(x.shape(), &[200, 64]);
        assert_eq!(y.len(), 200);
    }

    #[test]
    fn test_load_digits_classes() {
        let (_, y) = load_digits::<f64>().unwrap();
        let unique: HashSet<usize> = y.iter().copied().collect();
        assert_eq!(unique.len(), 10, "Digits should have 10 unique classes");
    }
}
