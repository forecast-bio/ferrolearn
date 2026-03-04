//! Train/test splitting utilities.
//!
//! This module provides [`train_test_split`], which shuffles and partitions
//! a dataset into training and test subsets.

use ferrolearn_core::FerroError;
use ndarray::{Array1, Array2};
use num_traits::Float;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;

/// Result type for [`train_test_split`]: `(x_train, x_test, y_train, y_test)`.
pub type TrainTestSplit<F> = (Array2<F>, Array2<F>, Array1<F>, Array1<F>);

/// Split arrays into random train and test subsets.
///
/// The dataset rows are shuffled (using `random_state` as an optional seed)
/// and then split so that approximately `test_size` fraction of the samples
/// are held out as the test set.
///
/// # Parameters
///
/// - `x` — Feature matrix with shape `(n_samples, n_features)`.
/// - `y` — Target array with length `n_samples`.
/// - `test_size` — Fraction of samples to use for the test set (e.g. `0.2`).
///   Must be in the open interval `(0.0, 1.0)`.
/// - `random_state` — Optional RNG seed for reproducibility.
///
/// # Returns
///
/// A tuple `(x_train, x_test, y_train, y_test)`.
///
/// # Errors
///
/// - [`FerroError::InvalidParameter`] if `test_size` is not in `(0, 1)`.
/// - [`FerroError::InsufficientSamples`] if there are fewer than 2 samples.
/// - [`FerroError::ShapeMismatch`] if `x` and `y` have different numbers of
///   rows.
///
/// # Examples
///
/// ```rust
/// use ferrolearn_model_sel::train_test_split;
/// use ndarray::{Array1, Array2};
///
/// let x = Array2::<f64>::zeros((10, 3));
/// let y = Array1::<f64>::zeros(10);
/// let (x_train, x_test, y_train, y_test) =
///     train_test_split(&x, &y, 0.3, Some(0)).unwrap();
/// assert_eq!(x_train.nrows() + x_test.nrows(), 10);
/// ```
pub fn train_test_split<F>(
    x: &Array2<F>,
    y: &Array1<F>,
    test_size: f64,
    random_state: Option<u64>,
) -> Result<TrainTestSplit<F>, FerroError>
where
    F: Float + Send + Sync + 'static,
{
    let n_samples = x.nrows();

    // Validate test_size.
    if test_size <= 0.0 || test_size >= 1.0 {
        return Err(FerroError::InvalidParameter {
            name: "test_size".into(),
            reason: format!("must be in (0, 1), got {test_size}"),
        });
    }

    // Validate that x and y agree on the number of samples.
    if y.len() != n_samples {
        return Err(FerroError::ShapeMismatch {
            expected: vec![n_samples],
            actual: vec![y.len()],
            context: "train_test_split: y length must equal x number of rows".into(),
        });
    }

    // Need at least 2 samples to form non-empty splits.
    if n_samples < 2 {
        return Err(FerroError::InsufficientSamples {
            required: 2,
            actual: n_samples,
            context: "train_test_split".into(),
        });
    }

    // Build shuffled index list.
    let mut indices: Vec<usize> = (0..n_samples).collect();
    match random_state {
        Some(seed) => {
            let mut rng = SmallRng::seed_from_u64(seed);
            indices.shuffle(&mut rng);
        }
        None => {
            let mut rng = SmallRng::from_os_rng();
            indices.shuffle(&mut rng);
        }
    }

    // Compute the number of test samples (at least 1).
    let n_test = ((n_samples as f64) * test_size).round() as usize;
    let n_test = n_test.max(1).min(n_samples - 1);
    let n_train = n_samples - n_test;

    let train_idx = &indices[..n_train];
    let test_idx = &indices[n_train..];

    // Gather rows for x.
    let n_features = x.ncols();
    let mut x_train_data = Vec::with_capacity(n_train * n_features);
    for &i in train_idx {
        x_train_data.extend(x.row(i).iter().copied());
    }
    let mut x_test_data = Vec::with_capacity(n_test * n_features);
    for &i in test_idx {
        x_test_data.extend(x.row(i).iter().copied());
    }

    let x_train = Array2::from_shape_vec((n_train, n_features), x_train_data).map_err(|e| {
        FerroError::InvalidParameter {
            name: "x_train".into(),
            reason: e.to_string(),
        }
    })?;
    let x_test = Array2::from_shape_vec((n_test, n_features), x_test_data).map_err(|e| {
        FerroError::InvalidParameter {
            name: "x_test".into(),
            reason: e.to_string(),
        }
    })?;

    // Gather y values.
    let y_train: Array1<F> = train_idx.iter().map(|&i| y[i]).collect();
    let y_test: Array1<F> = test_idx.iter().map(|&i| y[i]).collect();

    Ok((x_train, x_test, y_train, y_test))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    fn make_data(n: usize) -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_fn((n, 3), |(i, j)| (i * 3 + j) as f64);
        let y = Array1::from_iter((0..n).map(|i| i as f64));
        (x, y)
    }

    #[test]
    fn test_split_sizes() {
        let (x, y) = make_data(10);
        let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.3, Some(42)).unwrap();
        assert_eq!(x_train.nrows() + x_test.nrows(), 10);
        assert_eq!(y_train.len() + y_test.len(), 10);
    }

    #[test]
    fn test_split_is_deterministic_with_seed() {
        let (x, y) = make_data(20);
        let (x_train1, x_test1, _, _) = train_test_split(&x, &y, 0.2, Some(7)).unwrap();
        let (x_train2, x_test2, _, _) = train_test_split(&x, &y, 0.2, Some(7)).unwrap();
        assert_eq!(x_train1, x_train2);
        assert_eq!(x_test1, x_test2);
    }

    #[test]
    fn test_split_different_seeds_differ() {
        let (x, y) = make_data(20);
        let (x_test1, _, _, _) = train_test_split(&x, &y, 0.2, Some(1)).unwrap();
        let (x_test2, _, _, _) = train_test_split(&x, &y, 0.2, Some(99)).unwrap();
        // Very unlikely to be identical with different seeds on 20 samples.
        assert_ne!(x_test1, x_test2);
    }

    #[test]
    fn test_no_data_overlap() {
        let (x, y) = make_data(10);
        let (_, _, y_train, y_test) = train_test_split(&x, &y, 0.3, Some(0)).unwrap();
        // Check all indices from 0..10 appear exactly once.
        let mut all: Vec<u64> = y_train
            .iter()
            .chain(y_test.iter())
            .map(|&v| v as u64)
            .collect();
        all.sort_unstable();
        let expected: Vec<u64> = (0..10).collect();
        assert_eq!(all, expected);
    }

    #[test]
    fn test_invalid_test_size_zero() {
        let (x, y) = make_data(10);
        assert!(train_test_split(&x, &y, 0.0, None).is_err());
    }

    #[test]
    fn test_invalid_test_size_one() {
        let (x, y) = make_data(10);
        assert!(train_test_split(&x, &y, 1.0, None).is_err());
    }

    #[test]
    fn test_insufficient_samples() {
        let x = Array2::<f64>::zeros((1, 3));
        let y = Array1::<f64>::zeros(1);
        assert!(train_test_split(&x, &y, 0.2, None).is_err());
    }

    #[test]
    fn test_shape_mismatch() {
        let x = Array2::<f64>::zeros((10, 3));
        let y = Array1::<f64>::zeros(8);
        assert!(train_test_split(&x, &y, 0.2, None).is_err());
    }

    #[test]
    fn test_split_no_seed() {
        let (x, y) = make_data(20);
        let result = train_test_split(&x, &y, 0.2, None);
        assert!(result.is_ok());
        let (x_train, x_test, _, _) = result.unwrap();
        assert_eq!(x_train.nrows() + x_test.nrows(), 20);
    }
}
