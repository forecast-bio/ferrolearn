//! Synthetic dataset generators for testing and prototyping.
//!
//! All generators accept an optional `random_state` seed for full
//! reproducibility. When `random_state` is `None` a random seed is drawn
//! from the OS.

use ferrolearn_core::FerroError;
use ndarray::{Array1, Array2};
use num_traits::Float;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand_distr::{Distribution, Normal, Uniform};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// RNG helper
// ---------------------------------------------------------------------------

fn make_rng(random_state: Option<u64>) -> SmallRng {
    match random_state {
        Some(seed) => SmallRng::seed_from_u64(seed),
        None => SmallRng::from_os_rng(),
    }
}

// ---------------------------------------------------------------------------
// make_classification
// ---------------------------------------------------------------------------

/// Generate a random n-class classification dataset.
///
/// Samples are drawn from Gaussian blobs, one blob per class, placed at
/// random centres. Each class has an equal number of samples.
///
/// # Parameters
///
/// - `n_samples` — total number of samples (must be ≥ `n_classes`).
/// - `n_features` — number of features.
/// - `n_classes` — number of distinct class labels.
/// - `random_state` — optional RNG seed for reproducibility.
///
/// # Returns
///
/// `(X, y)` where `X` has shape `(n_samples, n_features)` and `y` contains
/// integer labels in `0..n_classes`.
///
/// # Errors
///
/// - [`FerroError::InvalidParameter`] if `n_classes == 0`, `n_features == 0`,
///   or `n_samples < n_classes`.
///
/// # Examples
///
/// ```rust
/// use ferrolearn_datasets::make_classification;
///
/// let (x, y) = make_classification::<f64>(100, 4, 3, Some(0)).unwrap();
/// assert_eq!(x.shape(), &[100, 4]);
/// assert_eq!(y.len(), 100);
/// ```
pub fn make_classification<F>(
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
    random_state: Option<u64>,
) -> Result<(Array2<F>, Array1<usize>), FerroError>
where
    F: Float + Send + Sync + 'static,
{
    if n_classes == 0 {
        return Err(FerroError::InvalidParameter {
            name: "n_classes".into(),
            reason: "must be at least 1".into(),
        });
    }
    if n_features == 0 {
        return Err(FerroError::InvalidParameter {
            name: "n_features".into(),
            reason: "must be at least 1".into(),
        });
    }
    if n_samples < n_classes {
        return Err(FerroError::InvalidParameter {
            name: "n_samples".into(),
            reason: format!("must be at least n_classes ({n_classes}), got {n_samples}"),
        });
    }

    let mut rng = make_rng(random_state);

    // Draw random cluster centres: each centre is a vector in n_features dims.
    let centre_dist = Normal::new(0.0_f64, 3.0).map_err(|e| FerroError::InvalidParameter {
        name: "centre_distribution".into(),
        reason: e.to_string(),
    })?;
    let noise_dist = Normal::new(0.0_f64, 1.0).map_err(|e| FerroError::InvalidParameter {
        name: "noise_distribution".into(),
        reason: e.to_string(),
    })?;

    let mut centres: Vec<Vec<f64>> = Vec::with_capacity(n_classes);
    for _ in 0..n_classes {
        let c: Vec<f64> = (0..n_features)
            .map(|_| centre_dist.sample(&mut rng))
            .collect();
        centres.push(c);
    }

    let mut x_data: Vec<F> = Vec::with_capacity(n_samples * n_features);
    let mut y_data: Vec<usize> = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let class = i % n_classes;
        let centre = &centres[class];
        for &c in centre {
            let val = c + noise_dist.sample(&mut rng);
            x_data.push(F::from(val).unwrap_or(F::zero()));
        }
        y_data.push(class);
    }

    let x = Array2::from_shape_vec((n_samples, n_features), x_data).map_err(|e| {
        FerroError::SerdeError {
            message: format!("make_classification reshape failed: {e}"),
        }
    })?;
    let y = Array1::from_vec(y_data);

    Ok((x, y))
}

// ---------------------------------------------------------------------------
// make_regression
// ---------------------------------------------------------------------------

/// Generate a random regression dataset.
///
/// A random linear combination of `n_informative` features determines the
/// target. Gaussian noise with standard deviation `noise` is added to `y`.
///
/// # Parameters
///
/// - `n_samples` — total number of samples.
/// - `n_features` — total number of features (must be ≥ `n_informative`).
/// - `n_informative` — number of features that have a non-zero coefficient.
/// - `noise` — standard deviation of Gaussian noise added to the target.
/// - `random_state` — optional RNG seed for reproducibility.
///
/// # Returns
///
/// `(X, y)` where `X` has shape `(n_samples, n_features)` and `y` is a
/// continuous target array of length `n_samples`.
///
/// # Errors
///
/// - [`FerroError::InvalidParameter`] if `n_informative > n_features` or
///   either dimension is zero.
///
/// # Examples
///
/// ```rust
/// use ferrolearn_datasets::make_regression;
///
/// let (x, y) = make_regression::<f64>(100, 5, 3, 0.0, Some(0)).unwrap();
/// assert_eq!(x.shape(), &[100, 5]);
/// assert_eq!(y.len(), 100);
/// ```
pub fn make_regression<F>(
    n_samples: usize,
    n_features: usize,
    n_informative: usize,
    noise: F,
    random_state: Option<u64>,
) -> Result<(Array2<F>, Array1<F>), FerroError>
where
    F: Float + Send + Sync + 'static,
{
    if n_features == 0 {
        return Err(FerroError::InvalidParameter {
            name: "n_features".into(),
            reason: "must be at least 1".into(),
        });
    }
    if n_informative > n_features {
        return Err(FerroError::InvalidParameter {
            name: "n_informative".into(),
            reason: format!("must be ≤ n_features ({n_features}), got {n_informative}"),
        });
    }
    if n_samples == 0 {
        return Err(FerroError::InvalidParameter {
            name: "n_samples".into(),
            reason: "must be at least 1".into(),
        });
    }

    let mut rng = make_rng(random_state);

    let feature_dist = Normal::new(0.0_f64, 1.0).map_err(|e| FerroError::InvalidParameter {
        name: "feature_distribution".into(),
        reason: e.to_string(),
    })?;

    // Random coefficients for informative features.
    let coef_dist =
        Uniform::new(-10.0_f64, 10.0_f64).map_err(|e| FerroError::InvalidParameter {
            name: "coef_distribution".into(),
            reason: e.to_string(),
        })?;
    let coeffs: Vec<f64> = (0..n_informative)
        .map(|_| coef_dist.sample(&mut rng))
        .collect();

    let noise_f64 = noise.to_f64().unwrap_or(0.0);
    let noise_dist = if noise_f64 > 0.0 {
        Some(
            Normal::new(0.0_f64, noise_f64).map_err(|e| FerroError::InvalidParameter {
                name: "noise_distribution".into(),
                reason: e.to_string(),
            })?,
        )
    } else {
        None
    };

    let mut x_data: Vec<F> = Vec::with_capacity(n_samples * n_features);
    let mut y_data: Vec<F> = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let row: Vec<f64> = (0..n_features)
            .map(|_| feature_dist.sample(&mut rng))
            .collect();

        // Compute target as dot product with coefficients (informative features).
        let mut target = 0.0_f64;
        for (j, &c) in coeffs.iter().enumerate() {
            target += c * row[j];
        }
        if let Some(ref nd) = noise_dist {
            target += nd.sample(&mut rng);
        }

        for &v in &row {
            x_data.push(F::from(v).unwrap_or(F::zero()));
        }
        y_data.push(F::from(target).unwrap_or(F::zero()));
    }

    let x = Array2::from_shape_vec((n_samples, n_features), x_data).map_err(|e| {
        FerroError::SerdeError {
            message: format!("make_regression reshape failed: {e}"),
        }
    })?;
    let y = Array1::from_vec(y_data);

    Ok((x, y))
}

// ---------------------------------------------------------------------------
// make_blobs
// ---------------------------------------------------------------------------

/// Generate isotropic Gaussian blobs for clustering.
///
/// # Parameters
///
/// - `n_samples` — total number of samples.
/// - `n_features` — number of features.
/// - `centers` — number of cluster centres.
/// - `cluster_std` — standard deviation of each cluster.
/// - `random_state` — optional RNG seed for reproducibility.
///
/// # Returns
///
/// `(X, y)` where `X` has shape `(n_samples, n_features)` and `y` contains
/// integer cluster labels in `0..centers`.
///
/// # Errors
///
/// - [`FerroError::InvalidParameter`] if `centers == 0` or `n_features == 0`.
///
/// # Examples
///
/// ```rust
/// use ferrolearn_datasets::make_blobs;
///
/// let (x, y) = make_blobs::<f64>(150, 2, 3, 1.0, Some(42)).unwrap();
/// assert_eq!(x.shape(), &[150, 2]);
/// assert_eq!(y.len(), 150);
/// ```
pub fn make_blobs<F>(
    n_samples: usize,
    n_features: usize,
    centers: usize,
    cluster_std: F,
    random_state: Option<u64>,
) -> Result<(Array2<F>, Array1<usize>), FerroError>
where
    F: Float + Send + Sync + 'static,
{
    if centers == 0 {
        return Err(FerroError::InvalidParameter {
            name: "centers".into(),
            reason: "must be at least 1".into(),
        });
    }
    if n_features == 0 {
        return Err(FerroError::InvalidParameter {
            name: "n_features".into(),
            reason: "must be at least 1".into(),
        });
    }
    if n_samples == 0 {
        return Err(FerroError::InvalidParameter {
            name: "n_samples".into(),
            reason: "must be at least 1".into(),
        });
    }

    let std_f64 = cluster_std.to_f64().unwrap_or(1.0);
    if std_f64 < 0.0 {
        return Err(FerroError::InvalidParameter {
            name: "cluster_std".into(),
            reason: "must be non-negative".into(),
        });
    }

    let mut rng = make_rng(random_state);

    let centre_dist =
        Uniform::new(-10.0_f64, 10.0_f64).map_err(|e| FerroError::InvalidParameter {
            name: "centre_distribution".into(),
            reason: e.to_string(),
        })?;
    let noise_dist = Normal::new(0.0_f64, std_f64).map_err(|e| FerroError::InvalidParameter {
        name: "noise_distribution".into(),
        reason: e.to_string(),
    })?;

    // Draw random cluster centres.
    let blob_centres: Vec<Vec<f64>> = (0..centers)
        .map(|_| {
            (0..n_features)
                .map(|_| centre_dist.sample(&mut rng))
                .collect()
        })
        .collect();

    let mut x_data: Vec<F> = Vec::with_capacity(n_samples * n_features);
    let mut y_data: Vec<usize> = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let cluster = i % centers;
        let centre = &blob_centres[cluster];
        for &c in centre {
            let val = c + noise_dist.sample(&mut rng);
            x_data.push(F::from(val).unwrap_or(F::zero()));
        }
        y_data.push(cluster);
    }

    let x = Array2::from_shape_vec((n_samples, n_features), x_data).map_err(|e| {
        FerroError::SerdeError {
            message: format!("make_blobs reshape failed: {e}"),
        }
    })?;
    let y = Array1::from_vec(y_data);

    Ok((x, y))
}

// ---------------------------------------------------------------------------
// make_moons
// ---------------------------------------------------------------------------

/// Generate two interleaving half-circles ("moons").
///
/// # Parameters
///
/// - `n_samples` — total number of samples (split evenly between the two moons).
/// - `noise` — standard deviation of Gaussian noise added to each point.
/// - `random_state` — optional RNG seed for reproducibility.
///
/// # Returns
///
/// `(X, y)` where `X` has shape `(n_samples, 2)` and `y` ∈ {0, 1}.
///
/// # Errors
///
/// - [`FerroError::InvalidParameter`] if `n_samples == 0` or `noise < 0`.
///
/// # Examples
///
/// ```rust
/// use ferrolearn_datasets::make_moons;
///
/// let (x, y) = make_moons::<f64>(100, 0.1, Some(0)).unwrap();
/// assert_eq!(x.shape(), &[100, 2]);
/// assert_eq!(y.len(), 100);
/// ```
pub fn make_moons<F>(
    n_samples: usize,
    noise: F,
    random_state: Option<u64>,
) -> Result<(Array2<F>, Array1<usize>), FerroError>
where
    F: Float + Send + Sync + 'static,
{
    if n_samples == 0 {
        return Err(FerroError::InvalidParameter {
            name: "n_samples".into(),
            reason: "must be at least 1".into(),
        });
    }
    let noise_f64 = noise.to_f64().unwrap_or(0.0);
    if noise_f64 < 0.0 {
        return Err(FerroError::InvalidParameter {
            name: "noise".into(),
            reason: "must be non-negative".into(),
        });
    }

    let mut rng = make_rng(random_state);

    let n_upper = n_samples / 2;
    let n_lower = n_samples - n_upper;

    let noise_dist = if noise_f64 > 0.0 {
        Some(
            Normal::new(0.0_f64, noise_f64).map_err(|e| FerroError::InvalidParameter {
                name: "noise_distribution".into(),
                reason: e.to_string(),
            })?,
        )
    } else {
        None
    };

    let mut x_data: Vec<F> = Vec::with_capacity(n_samples * 2);
    let mut y_data: Vec<usize> = Vec::with_capacity(n_samples);

    // Upper moon (class 0): angles in [0, π].
    for i in 0..n_upper {
        let theta = PI * (i as f64) / (n_upper.max(1) as f64);
        let mut px = theta.cos();
        let mut py = theta.sin();
        if let Some(ref nd) = noise_dist {
            px += nd.sample(&mut rng);
            py += nd.sample(&mut rng);
        }
        x_data.push(F::from(px).unwrap_or(F::zero()));
        x_data.push(F::from(py).unwrap_or(F::zero()));
        y_data.push(0);
    }

    // Lower moon (class 1): angles in [0, π], offset by (1, -0.5).
    for i in 0..n_lower {
        let theta = PI * (i as f64) / (n_lower.max(1) as f64);
        let mut px = 1.0 - theta.cos();
        let mut py = 1.0 - theta.sin() - 0.5;
        if let Some(ref nd) = noise_dist {
            px += nd.sample(&mut rng);
            py += nd.sample(&mut rng);
        }
        x_data.push(F::from(px).unwrap_or(F::zero()));
        x_data.push(F::from(py).unwrap_or(F::zero()));
        y_data.push(1);
    }

    let x = Array2::from_shape_vec((n_samples, 2), x_data).map_err(|e| FerroError::SerdeError {
        message: format!("make_moons reshape failed: {e}"),
    })?;
    let y = Array1::from_vec(y_data);

    Ok((x, y))
}

// ---------------------------------------------------------------------------
// make_circles
// ---------------------------------------------------------------------------

/// Generate a large circle containing a smaller circle ("circles").
///
/// # Parameters
///
/// - `n_samples` — total number of samples (split evenly between the two circles).
/// - `noise` — standard deviation of Gaussian noise added to each point.
/// - `factor` — scale factor between inner and outer circle (0 < `factor` < 1).
/// - `random_state` — optional RNG seed for reproducibility.
///
/// # Returns
///
/// `(X, y)` where `X` has shape `(n_samples, 2)` and `y` ∈ {0, 1}.
///
/// # Errors
///
/// - [`FerroError::InvalidParameter`] if `n_samples == 0`, `noise < 0`, or
///   `factor` is not in `(0, 1)`.
///
/// # Examples
///
/// ```rust
/// use ferrolearn_datasets::make_circles;
///
/// let (x, y) = make_circles::<f64>(100, 0.05, 0.5, Some(0)).unwrap();
/// assert_eq!(x.shape(), &[100, 2]);
/// assert_eq!(y.len(), 100);
/// ```
pub fn make_circles<F>(
    n_samples: usize,
    noise: F,
    factor: F,
    random_state: Option<u64>,
) -> Result<(Array2<F>, Array1<usize>), FerroError>
where
    F: Float + Send + Sync + 'static,
{
    if n_samples == 0 {
        return Err(FerroError::InvalidParameter {
            name: "n_samples".into(),
            reason: "must be at least 1".into(),
        });
    }
    let noise_f64 = noise.to_f64().unwrap_or(0.0);
    if noise_f64 < 0.0 {
        return Err(FerroError::InvalidParameter {
            name: "noise".into(),
            reason: "must be non-negative".into(),
        });
    }
    let factor_f64 = factor.to_f64().unwrap_or(0.5);
    if factor_f64 <= 0.0 || factor_f64 >= 1.0 {
        return Err(FerroError::InvalidParameter {
            name: "factor".into(),
            reason: format!("must be in (0, 1), got {factor_f64}"),
        });
    }

    let mut rng = make_rng(random_state);

    let n_outer = n_samples / 2;
    let n_inner = n_samples - n_outer;

    let noise_dist = if noise_f64 > 0.0 {
        Some(
            Normal::new(0.0_f64, noise_f64).map_err(|e| FerroError::InvalidParameter {
                name: "noise_distribution".into(),
                reason: e.to_string(),
            })?,
        )
    } else {
        None
    };

    let mut x_data: Vec<F> = Vec::with_capacity(n_samples * 2);
    let mut y_data: Vec<usize> = Vec::with_capacity(n_samples);

    // Outer circle (class 0), radius 1.
    for i in 0..n_outer {
        let theta = 2.0 * PI * (i as f64) / (n_outer.max(1) as f64);
        let mut px = theta.cos();
        let mut py = theta.sin();
        if let Some(ref nd) = noise_dist {
            px += nd.sample(&mut rng);
            py += nd.sample(&mut rng);
        }
        x_data.push(F::from(px).unwrap_or(F::zero()));
        x_data.push(F::from(py).unwrap_or(F::zero()));
        y_data.push(0);
    }

    // Inner circle (class 1), radius `factor`.
    for i in 0..n_inner {
        let theta = 2.0 * PI * (i as f64) / (n_inner.max(1) as f64);
        let mut px = factor_f64 * theta.cos();
        let mut py = factor_f64 * theta.sin();
        if let Some(ref nd) = noise_dist {
            px += nd.sample(&mut rng);
            py += nd.sample(&mut rng);
        }
        x_data.push(F::from(px).unwrap_or(F::zero()));
        x_data.push(F::from(py).unwrap_or(F::zero()));
        y_data.push(1);
    }

    let x = Array2::from_shape_vec((n_samples, 2), x_data).map_err(|e| FerroError::SerdeError {
        message: format!("make_circles reshape failed: {e}"),
    })?;
    let y = Array1::from_vec(y_data);

    Ok((x, y))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    // --- make_classification ---

    #[test]
    fn test_make_classification_shape() {
        let (x, y) = make_classification::<f64>(100, 4, 3, Some(0)).unwrap();
        assert_eq!(x.shape(), &[100, 4]);
        assert_eq!(y.len(), 100);
    }

    #[test]
    fn test_make_classification_classes() {
        let (_, y) = make_classification::<f64>(90, 4, 3, Some(1)).unwrap();
        let unique: HashSet<usize> = y.iter().copied().collect();
        assert_eq!(
            unique.len(),
            3,
            "expected 3 unique classes, got {:?}",
            unique
        );
    }

    #[test]
    fn test_make_classification_reproducible() {
        let (x1, y1) = make_classification::<f64>(50, 3, 2, Some(42)).unwrap();
        let (x2, y2) = make_classification::<f64>(50, 3, 2, Some(42)).unwrap();
        assert_eq!(x1, x2);
        assert_eq!(y1, y2);
    }

    #[test]
    fn test_make_classification_different_seeds() {
        let (x1, _) = make_classification::<f64>(50, 3, 2, Some(1)).unwrap();
        let (x2, _) = make_classification::<f64>(50, 3, 2, Some(999)).unwrap();
        assert_ne!(x1, x2, "different seeds should produce different data");
    }

    #[test]
    fn test_make_classification_invalid_n_classes_zero() {
        assert!(make_classification::<f64>(10, 4, 0, None).is_err());
    }

    #[test]
    fn test_make_classification_invalid_n_features_zero() {
        assert!(make_classification::<f64>(10, 0, 2, None).is_err());
    }

    #[test]
    fn test_make_classification_invalid_too_few_samples() {
        assert!(make_classification::<f64>(1, 4, 3, None).is_err());
    }

    #[test]
    fn test_make_classification_f32() {
        let (x, y) = make_classification::<f32>(30, 2, 2, Some(7)).unwrap();
        assert_eq!(x.shape(), &[30, 2]);
        assert_eq!(y.len(), 30);
    }

    // --- make_regression ---

    #[test]
    fn test_make_regression_shape() {
        let (x, y) = make_regression::<f64>(100, 5, 3, 0.0, Some(0)).unwrap();
        assert_eq!(x.shape(), &[100, 5]);
        assert_eq!(y.len(), 100);
    }

    #[test]
    fn test_make_regression_zero_noise_deterministic() {
        // With noise=0 and same seed, two calls must be identical.
        let (x1, y1) = make_regression::<f64>(50, 4, 4, 0.0, Some(123)).unwrap();
        let (x2, y2) = make_regression::<f64>(50, 4, 4, 0.0, Some(123)).unwrap();
        assert_eq!(x1, x2);
        assert_eq!(y1, y2);
    }

    #[test]
    fn test_make_regression_reproducible() {
        let (x1, y1) = make_regression::<f64>(50, 3, 2, 0.5, Some(7)).unwrap();
        let (x2, y2) = make_regression::<f64>(50, 3, 2, 0.5, Some(7)).unwrap();
        assert_eq!(x1, x2);
        assert_eq!(y1, y2);
    }

    #[test]
    fn test_make_regression_invalid_n_informative() {
        assert!(make_regression::<f64>(10, 3, 5, 0.0, None).is_err());
    }

    #[test]
    fn test_make_regression_invalid_n_features_zero() {
        assert!(make_regression::<f64>(10, 0, 0, 0.0, None).is_err());
    }

    // --- make_blobs ---

    #[test]
    fn test_make_blobs_shape() {
        let (x, y) = make_blobs::<f64>(150, 2, 3, 1.0, Some(42)).unwrap();
        assert_eq!(x.shape(), &[150, 2]);
        assert_eq!(y.len(), 150);
    }

    #[test]
    fn test_make_blobs_clusters() {
        let (_, y) = make_blobs::<f64>(90, 2, 3, 0.5, Some(0)).unwrap();
        let unique: HashSet<usize> = y.iter().copied().collect();
        assert_eq!(unique.len(), 3, "expected 3 clusters");
    }

    #[test]
    fn test_make_blobs_distinct_centers() {
        // With small noise and many samples, class means should differ
        // substantially from each other.
        let (x, y) = make_blobs::<f64>(300, 2, 3, 0.1, Some(5)).unwrap();
        let mut means: Vec<[f64; 2]> = vec![[0.0; 2]; 3];
        let mut counts = vec![0_usize; 3];
        for (i, &cls) in y.iter().enumerate() {
            means[cls][0] += x[[i, 0]];
            means[cls][1] += x[[i, 1]];
            counts[cls] += 1;
        }
        for (cls, cnt) in counts.iter().enumerate() {
            if *cnt > 0 {
                means[cls][0] /= *cnt as f64;
                means[cls][1] /= *cnt as f64;
            }
        }
        // Check that not all means are the same (clusters are distinct).
        let all_same = means
            .windows(2)
            .all(|w| (w[0][0] - w[1][0]).abs() < 0.5 && (w[0][1] - w[1][1]).abs() < 0.5);
        assert!(!all_same, "cluster centers should be distinct");
    }

    #[test]
    fn test_make_blobs_reproducible() {
        let (x1, y1) = make_blobs::<f64>(60, 2, 3, 1.0, Some(99)).unwrap();
        let (x2, y2) = make_blobs::<f64>(60, 2, 3, 1.0, Some(99)).unwrap();
        assert_eq!(x1, x2);
        assert_eq!(y1, y2);
    }

    #[test]
    fn test_make_blobs_invalid_centers_zero() {
        assert!(make_blobs::<f64>(10, 2, 0, 1.0, None).is_err());
    }

    // --- make_moons ---

    #[test]
    fn test_make_moons_shape() {
        let (x, y) = make_moons::<f64>(100, 0.0, Some(0)).unwrap();
        assert_eq!(x.shape(), &[100, 2]);
        assert_eq!(y.len(), 100);
    }

    #[test]
    fn test_make_moons_two_classes() {
        let (_, y) = make_moons::<f64>(100, 0.0, Some(0)).unwrap();
        let unique: HashSet<usize> = y.iter().copied().collect();
        assert_eq!(unique.len(), 2, "make_moons should produce 2 classes");
        assert!(unique.contains(&0));
        assert!(unique.contains(&1));
    }

    #[test]
    fn test_make_moons_interleaving() {
        // Without noise the upper moon (y=0) has positive y-coordinate on avg,
        // while the lower moon (y=1) has negative average y-coordinate.
        let (x, y) = make_moons::<f64>(200, 0.0, Some(0)).unwrap();
        let (mut sum0, mut sum1) = (0.0_f64, 0.0_f64);
        let (mut cnt0, mut cnt1) = (0_usize, 0_usize);
        for (i, &cls) in y.iter().enumerate() {
            if cls == 0 {
                sum0 += x[[i, 1]];
                cnt0 += 1;
            } else {
                sum1 += x[[i, 1]];
                cnt1 += 1;
            }
        }
        let mean0 = sum0 / cnt0 as f64;
        let mean1 = sum1 / cnt1 as f64;
        // Upper moon centred around y ≈ 0.64, lower around y ≈ -0.14.
        assert!(
            mean0 > mean1,
            "upper moon should have higher mean y than lower moon"
        );
    }

    #[test]
    fn test_make_moons_reproducible() {
        let (x1, y1) = make_moons::<f64>(60, 0.1, Some(11)).unwrap();
        let (x2, y2) = make_moons::<f64>(60, 0.1, Some(11)).unwrap();
        assert_eq!(x1, x2);
        assert_eq!(y1, y2);
    }

    #[test]
    fn test_make_moons_invalid_n_samples_zero() {
        assert!(make_moons::<f64>(0, 0.0, None).is_err());
    }

    // --- make_circles ---

    #[test]
    fn test_make_circles_shape() {
        let (x, y) = make_circles::<f64>(100, 0.0, 0.5, Some(0)).unwrap();
        assert_eq!(x.shape(), &[100, 2]);
        assert_eq!(y.len(), 100);
    }

    #[test]
    fn test_make_circles_two_classes() {
        let (_, y) = make_circles::<f64>(100, 0.0, 0.5, Some(0)).unwrap();
        let unique: HashSet<usize> = y.iter().copied().collect();
        assert_eq!(unique.len(), 2);
        assert!(unique.contains(&0));
        assert!(unique.contains(&1));
    }

    #[test]
    fn test_make_circles_radii() {
        // Without noise: outer circle (y=0) has radius ≈ 1, inner (y=1) ≈ factor.
        let factor = 0.4_f64;
        let (x, y) = make_circles::<f64>(200, 0.0, factor, Some(0)).unwrap();
        let mut sum_r0 = 0.0_f64;
        let mut sum_r1 = 0.0_f64;
        let mut cnt0 = 0_usize;
        let mut cnt1 = 0_usize;
        for (i, &cls) in y.iter().enumerate() {
            let r = (x[[i, 0]].powi(2) + x[[i, 1]].powi(2)).sqrt();
            if cls == 0 {
                sum_r0 += r;
                cnt0 += 1;
            } else {
                sum_r1 += r;
                cnt1 += 1;
            }
        }
        let mean_r0 = sum_r0 / cnt0 as f64;
        let mean_r1 = sum_r1 / cnt1 as f64;
        assert!(
            (mean_r0 - 1.0).abs() < 0.05,
            "outer circle mean radius should be ≈ 1, got {mean_r0}"
        );
        assert!(
            (mean_r1 - factor).abs() < 0.05,
            "inner circle mean radius should be ≈ {factor}, got {mean_r1}"
        );
    }

    #[test]
    fn test_make_circles_reproducible() {
        let (x1, y1) = make_circles::<f64>(80, 0.05, 0.5, Some(33)).unwrap();
        let (x2, y2) = make_circles::<f64>(80, 0.05, 0.5, Some(33)).unwrap();
        assert_eq!(x1, x2);
        assert_eq!(y1, y2);
    }

    #[test]
    fn test_make_circles_invalid_factor_zero() {
        assert!(make_circles::<f64>(100, 0.0, 0.0, None).is_err());
    }

    #[test]
    fn test_make_circles_invalid_factor_one() {
        assert!(make_circles::<f64>(100, 0.0, 1.0, None).is_err());
    }
}
