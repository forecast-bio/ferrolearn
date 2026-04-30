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
            x_data.push(F::from(val).unwrap_or_else(F::zero));
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
            x_data.push(F::from(v).unwrap_or_else(F::zero));
        }
        y_data.push(F::from(target).unwrap_or_else(F::zero));
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
            x_data.push(F::from(val).unwrap_or_else(F::zero));
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
        x_data.push(F::from(px).unwrap_or_else(F::zero));
        x_data.push(F::from(py).unwrap_or_else(F::zero));
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
        x_data.push(F::from(px).unwrap_or_else(F::zero));
        x_data.push(F::from(py).unwrap_or_else(F::zero));
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
        x_data.push(F::from(px).unwrap_or_else(F::zero));
        x_data.push(F::from(py).unwrap_or_else(F::zero));
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
        x_data.push(F::from(px).unwrap_or_else(F::zero));
        x_data.push(F::from(py).unwrap_or_else(F::zero));
        y_data.push(1);
    }

    let x = Array2::from_shape_vec((n_samples, 2), x_data).map_err(|e| FerroError::SerdeError {
        message: format!("make_circles reshape failed: {e}"),
    })?;
    let y = Array1::from_vec(y_data);

    Ok((x, y))
}

// ---------------------------------------------------------------------------
// make_swiss_roll
// ---------------------------------------------------------------------------

/// Generate a 3D Swiss roll manifold dataset.
///
/// The Swiss roll is a classic manifold learning dataset: a 2D surface (a
/// rolled-up rectangle) embedded in 3D space. Each sample is parameterised by
/// angle `t ∈ [1.5π, 4.5π]` and a height coordinate.
///
/// # Parameters
///
/// - `n_samples` — number of samples to generate (must be ≥ 1).
/// - `noise` — standard deviation of Gaussian noise added to each coordinate
///   (must be ≥ 0).
/// - `random_state` — optional RNG seed for reproducibility.
///
/// # Returns
///
/// `(X, t)` where:
/// - `X` has shape `(n_samples, 3)` with columns `(x, y, z)`.
/// - `t` is an [`Array1<F>`] of length `n_samples` containing the angle
///   parameter (the intrinsic coordinate along the roll).
///
/// # Errors
///
/// - [`FerroError::InvalidParameter`] if `n_samples == 0` or `noise < 0`.
///
/// # Examples
///
/// ```rust
/// use ferrolearn_datasets::make_swiss_roll;
///
/// let (x, t) = make_swiss_roll::<f64>(200, 0.0, Some(42)).unwrap();
/// assert_eq!(x.shape(), &[200, 3]);
/// assert_eq!(t.len(), 200);
/// ```
pub fn make_swiss_roll<F>(
    n_samples: usize,
    noise: F,
    random_state: Option<u64>,
) -> Result<(Array2<F>, Array1<F>), FerroError>
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

    // t ∈ [1.5π, 4.5π]
    let t_dist = Uniform::new(1.5 * PI, 4.5 * PI).map_err(|e| FerroError::InvalidParameter {
        name: "t_distribution".into(),
        reason: e.to_string(),
    })?;
    // height ∈ [0, 21)
    let height_dist = Uniform::new(0.0_f64, 21.0).map_err(|e| FerroError::InvalidParameter {
        name: "height_distribution".into(),
        reason: e.to_string(),
    })?;

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

    let mut x_data: Vec<F> = Vec::with_capacity(n_samples * 3);
    let mut t_data: Vec<F> = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let t_val = t_dist.sample(&mut rng);
        let height = height_dist.sample(&mut rng);

        let mut px = t_val * t_val.cos();
        let mut py = height;
        let mut pz = t_val * t_val.sin();

        if let Some(ref nd) = noise_dist {
            px += nd.sample(&mut rng);
            py += nd.sample(&mut rng);
            pz += nd.sample(&mut rng);
        }

        x_data.push(F::from(px).unwrap_or_else(F::zero));
        x_data.push(F::from(py).unwrap_or_else(F::zero));
        x_data.push(F::from(pz).unwrap_or_else(F::zero));
        t_data.push(F::from(t_val).unwrap_or_else(F::zero));
    }

    let x = Array2::from_shape_vec((n_samples, 3), x_data).map_err(|e| FerroError::SerdeError {
        message: format!("make_swiss_roll reshape failed: {e}"),
    })?;
    let t = Array1::from_vec(t_data);

    Ok((x, t))
}

// ---------------------------------------------------------------------------
// make_s_curve
// ---------------------------------------------------------------------------

/// Generate a 3D S-curve manifold dataset.
///
/// The S-curve is a 2D surface embedded in 3D space: two half-cylinders joined
/// at the edges to form an "S" shape. It is commonly used to benchmark
/// manifold-learning algorithms.
///
/// Each sample is parameterised by `t ∈ [-3π/2, 3π/2]` (the intrinsic
/// coordinate) and a height coordinate.
///
/// # Parameters
///
/// - `n_samples` — number of samples to generate (must be ≥ 1).
/// - `noise` — standard deviation of Gaussian noise added to each coordinate
///   (must be ≥ 0).
/// - `random_state` — optional RNG seed for reproducibility.
///
/// # Returns
///
/// `(X, t)` where:
/// - `X` has shape `(n_samples, 3)` with columns `(x, y, z)`.
/// - `t` is an [`Array1<F>`] of length `n_samples` containing the parameter `t`.
///
/// # Errors
///
/// - [`FerroError::InvalidParameter`] if `n_samples == 0` or `noise < 0`.
///
/// # Examples
///
/// ```rust
/// use ferrolearn_datasets::make_s_curve;
///
/// let (x, t) = make_s_curve::<f64>(200, 0.0, Some(42)).unwrap();
/// assert_eq!(x.shape(), &[200, 3]);
/// assert_eq!(t.len(), 200);
/// ```
pub fn make_s_curve<F>(
    n_samples: usize,
    noise: F,
    random_state: Option<u64>,
) -> Result<(Array2<F>, Array1<F>), FerroError>
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

    // t ∈ [-3π/2, 3π/2]
    let t_dist = Uniform::new(-1.5 * PI, 1.5 * PI).map_err(|e| FerroError::InvalidParameter {
        name: "t_distribution".into(),
        reason: e.to_string(),
    })?;
    // height ∈ [0, 2)
    let height_dist = Uniform::new(0.0_f64, 2.0).map_err(|e| FerroError::InvalidParameter {
        name: "height_distribution".into(),
        reason: e.to_string(),
    })?;

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

    let mut x_data: Vec<F> = Vec::with_capacity(n_samples * 3);
    let mut t_data: Vec<F> = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let t_val = t_dist.sample(&mut rng);
        let height = height_dist.sample(&mut rng);

        // S-curve: x = sin(t), y = height, z = sign(t) * (cos(t) - 1)
        let mut px = t_val.sin();
        let mut py = height;
        let mut pz = t_val.signum() * (t_val.cos() - 1.0);

        if let Some(ref nd) = noise_dist {
            px += nd.sample(&mut rng);
            py += nd.sample(&mut rng);
            pz += nd.sample(&mut rng);
        }

        x_data.push(F::from(px).unwrap_or_else(F::zero));
        x_data.push(F::from(py).unwrap_or_else(F::zero));
        x_data.push(F::from(pz).unwrap_or_else(F::zero));
        t_data.push(F::from(t_val).unwrap_or_else(F::zero));
    }

    let x = Array2::from_shape_vec((n_samples, 3), x_data).map_err(|e| FerroError::SerdeError {
        message: format!("make_s_curve reshape failed: {e}"),
    })?;
    let t = Array1::from_vec(t_data);

    Ok((x, t))
}

// ---------------------------------------------------------------------------
// make_sparse_uncorrelated
// ---------------------------------------------------------------------------

/// Generate a regression dataset with sparse, uncorrelated features.
///
/// Only the first 5 features are informative (with fixed weights
/// `[1, 2, 3, 4, 5]`); all remaining features are independent Gaussian noise.
/// This dataset is designed for testing feature-selection algorithms.
///
/// # Parameters
///
/// - `n_samples` — number of samples to generate (must be ≥ 1).
/// - `n_features` — total number of features (must be ≥ 5).
/// - `random_state` — optional RNG seed for reproducibility.
///
/// # Returns
///
/// `(X, y)` where `X` has shape `(n_samples, n_features)` and `y` is the
/// continuous regression target computed as:
///
/// ```text
/// y = 1*X[:,0] + 2*X[:,1] + 3*X[:,2] + 4*X[:,3] + 5*X[:,4]
/// ```
///
/// All features are drawn independently from N(0, 1).
///
/// # Errors
///
/// - [`FerroError::InvalidParameter`] if `n_samples == 0` or
///   `n_features < 5`.
///
/// # Examples
///
/// ```rust
/// use ferrolearn_datasets::make_sparse_uncorrelated;
///
/// let (x, y) = make_sparse_uncorrelated::<f64>(100, 10, Some(0)).unwrap();
/// assert_eq!(x.shape(), &[100, 10]);
/// assert_eq!(y.len(), 100);
/// ```
pub fn make_sparse_uncorrelated<F>(
    n_samples: usize,
    n_features: usize,
    random_state: Option<u64>,
) -> Result<(Array2<F>, Array1<F>), FerroError>
where
    F: Float + Send + Sync + 'static,
{
    if n_samples == 0 {
        return Err(FerroError::InvalidParameter {
            name: "n_samples".into(),
            reason: "must be at least 1".into(),
        });
    }
    if n_features < 5 {
        return Err(FerroError::InvalidParameter {
            name: "n_features".into(),
            reason: format!("must be at least 5, got {n_features}"),
        });
    }

    let mut rng = make_rng(random_state);

    let feature_dist = Normal::new(0.0_f64, 1.0).map_err(|e| FerroError::InvalidParameter {
        name: "feature_distribution".into(),
        reason: e.to_string(),
    })?;

    // Fixed informative weights for the first 5 features.
    let weights = [1.0_f64, 2.0, 3.0, 4.0, 5.0];

    let mut x_data: Vec<F> = Vec::with_capacity(n_samples * n_features);
    let mut y_data: Vec<F> = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let row: Vec<f64> = (0..n_features)
            .map(|_| feature_dist.sample(&mut rng))
            .collect();

        // Target = sum of (weight_i * feature_i) for the first 5 features.
        let target: f64 = weights.iter().enumerate().map(|(i, &w)| w * row[i]).sum();

        for &v in &row {
            x_data.push(F::from(v).unwrap_or_else(F::zero));
        }
        y_data.push(F::from(target).unwrap_or_else(F::zero));
    }

    let x = Array2::from_shape_vec((n_samples, n_features), x_data).map_err(|e| {
        FerroError::SerdeError {
            message: format!("make_sparse_uncorrelated reshape failed: {e}"),
        }
    })?;
    let y = Array1::from_vec(y_data);

    Ok((x, y))
}

// ---------------------------------------------------------------------------
// Friedman benchmarks (#1, #2, #3)
// ---------------------------------------------------------------------------

/// Generate the "Friedman #1" regression problem (Friedman, 1991).
///
/// Inputs `X` are uniform on `[0, 1]^n_features` (`n_features >= 5`). The
/// target depends only on the first five features:
///
/// ```text
/// y = 10 * sin(pi * x0 * x1) + 20 * (x2 - 0.5)^2 + 10 * x3 + 5 * x4 + noise
/// ```
///
/// # Errors
///
/// Returns [`FerroError::InvalidParameter`] if `n_features < 5`.
pub fn make_friedman1<F>(
    n_samples: usize,
    n_features: usize,
    noise: f64,
    random_state: Option<u64>,
) -> Result<(Array2<F>, Array1<F>), FerroError>
where
    F: Float + Send + Sync + 'static,
{
    if n_features < 5 {
        return Err(FerroError::InvalidParameter {
            name: "n_features".into(),
            reason: format!("make_friedman1: must be >= 5, got {n_features}"),
        });
    }
    let mut rng = make_rng(random_state);
    let unif = Uniform::new(0.0_f64, 1.0_f64).map_err(|e| FerroError::InvalidParameter {
        name: "uniform".into(),
        reason: e.to_string(),
    })?;
    let normal =
        Normal::new(0.0_f64, noise.max(0.0)).map_err(|e| FerroError::InvalidParameter {
            name: "noise".into(),
            reason: e.to_string(),
        })?;

    let mut x = Array2::<F>::zeros((n_samples, n_features));
    let mut y = Array1::<F>::zeros(n_samples);
    for i in 0..n_samples {
        let mut row = vec![0.0_f64; n_features];
        for j in 0..n_features {
            row[j] = unif.sample(&mut rng);
            x[[i, j]] = F::from(row[j]).ok_or_else(|| FerroError::InvalidParameter {
                name: "x".into(),
                reason: "could not convert uniform sample".into(),
            })?;
        }
        let val = 10.0 * (PI * row[0] * row[1]).sin()
            + 20.0 * (row[2] - 0.5).powi(2)
            + 10.0 * row[3]
            + 5.0 * row[4]
            + if noise > 0.0 {
                normal.sample(&mut rng)
            } else {
                0.0
            };
        y[i] = F::from(val).ok_or_else(|| FerroError::InvalidParameter {
            name: "y".into(),
            reason: "could not convert target".into(),
        })?;
    }
    Ok((x, y))
}

/// Generate the "Friedman #2" regression problem.
///
/// Inputs are 4-dimensional and have specific physical ranges:
///
/// - `x0 ~ U(0, 100)`
/// - `x1 ~ U(40 * pi, 560 * pi)`
/// - `x2 ~ U(0, 1)`
/// - `x3 ~ U(1, 11)`
///
/// Target: `y = sqrt(x0^2 + (x1 * x2 - 1 / (x1 * x3))^2) + noise`
pub fn make_friedman2<F>(
    n_samples: usize,
    noise: f64,
    random_state: Option<u64>,
) -> Result<(Array2<F>, Array1<F>), FerroError>
where
    F: Float + Send + Sync + 'static,
{
    let mut rng = make_rng(random_state);
    let normal =
        Normal::new(0.0_f64, noise.max(0.0)).map_err(|e| FerroError::InvalidParameter {
            name: "noise".into(),
            reason: e.to_string(),
        })?;
    let mut x = Array2::<F>::zeros((n_samples, 4));
    let mut y = Array1::<F>::zeros(n_samples);
    let u01 = Uniform::new(0.0_f64, 1.0_f64).map_err(|e| FerroError::InvalidParameter {
        name: "uniform".into(),
        reason: e.to_string(),
    })?;
    for i in 0..n_samples {
        let r0 = u01.sample(&mut rng) * 100.0;
        let r1 = 40.0 * PI + u01.sample(&mut rng) * (560.0 - 40.0) * PI;
        let r2 = u01.sample(&mut rng);
        let r3 = 1.0 + u01.sample(&mut rng) * 10.0;
        let xs = [r0, r1, r2, r3];
        for j in 0..4 {
            x[[i, j]] = F::from(xs[j]).ok_or_else(|| FerroError::InvalidParameter {
                name: "x".into(),
                reason: "could not convert".into(),
            })?;
        }
        let inner = r1 * r2 - 1.0 / (r1 * r3);
        let target = (r0 * r0 + inner * inner).sqrt()
            + if noise > 0.0 {
                normal.sample(&mut rng)
            } else {
                0.0
            };
        y[i] = F::from(target).ok_or_else(|| FerroError::InvalidParameter {
            name: "y".into(),
            reason: "could not convert".into(),
        })?;
    }
    Ok((x, y))
}

/// Generate the "Friedman #3" regression problem.
///
/// Same inputs as [`make_friedman2`]; target is:
///
/// `y = arctan((x1 * x2 - 1 / (x1 * x3)) / x0) + noise`
pub fn make_friedman3<F>(
    n_samples: usize,
    noise: f64,
    random_state: Option<u64>,
) -> Result<(Array2<F>, Array1<F>), FerroError>
where
    F: Float + Send + Sync + 'static,
{
    let mut rng = make_rng(random_state);
    let normal =
        Normal::new(0.0_f64, noise.max(0.0)).map_err(|e| FerroError::InvalidParameter {
            name: "noise".into(),
            reason: e.to_string(),
        })?;
    let mut x = Array2::<F>::zeros((n_samples, 4));
    let mut y = Array1::<F>::zeros(n_samples);
    let u01 = Uniform::new(0.0_f64, 1.0_f64).map_err(|e| FerroError::InvalidParameter {
        name: "uniform".into(),
        reason: e.to_string(),
    })?;
    for i in 0..n_samples {
        // x0 strictly positive to avoid divide-by-zero in arctan argument.
        let r0 = u01.sample(&mut rng) * 100.0 + 1e-6;
        let r1 = 40.0 * PI + u01.sample(&mut rng) * (560.0 - 40.0) * PI;
        let r2 = u01.sample(&mut rng);
        let r3 = 1.0 + u01.sample(&mut rng) * 10.0;
        let xs = [r0, r1, r2, r3];
        for j in 0..4 {
            x[[i, j]] = F::from(xs[j]).ok_or_else(|| FerroError::InvalidParameter {
                name: "x".into(),
                reason: "could not convert".into(),
            })?;
        }
        let arg = (r1 * r2 - 1.0 / (r1 * r3)) / r0;
        let target = arg.atan()
            + if noise > 0.0 {
                normal.sample(&mut rng)
            } else {
                0.0
            };
        y[i] = F::from(target).ok_or_else(|| FerroError::InvalidParameter {
            name: "y".into(),
            reason: "could not convert".into(),
        })?;
    }
    Ok((x, y))
}

// ---------------------------------------------------------------------------
// Low-rank, SPD, sparse SPD matrix generators
// ---------------------------------------------------------------------------

/// Generate a mostly-low-rank random matrix with bell-curve singular values.
///
/// Returns an `(n_samples, n_features)` matrix `U * S * V^T` where `U` and `V`
/// are random orthonormal matrices and `S` decays geometrically with
/// effective rank `effective_rank`.
pub fn make_low_rank_matrix<F>(
    n_samples: usize,
    n_features: usize,
    effective_rank: usize,
    tail_strength: f64,
    random_state: Option<u64>,
) -> Result<Array2<F>, FerroError>
where
    F: Float + Send + Sync + 'static,
{
    if effective_rank == 0 {
        return Err(FerroError::InvalidParameter {
            name: "effective_rank".into(),
            reason: "must be >= 1".into(),
        });
    }
    let mut rng = make_rng(random_state);
    let normal = Normal::new(0.0_f64, 1.0_f64).map_err(|e| FerroError::InvalidParameter {
        name: "normal".into(),
        reason: e.to_string(),
    })?;

    let n = n_samples.min(n_features);
    let mut sigma = vec![0.0_f64; n];
    for (i, s) in sigma.iter_mut().enumerate() {
        let low_rank = (-((i as f64 / effective_rank as f64).powi(2))).exp();
        let tail = tail_strength * (-(0.1 * i as f64)).exp();
        *s = low_rank + tail;
    }

    // Build dense X by drawing random Gaussian and rescaling its singular
    // values via SVD-like multiplication: simple approximation — generate
    // random matrix and scale columns by sigma.
    let mut a = Array2::<f64>::zeros((n_samples, n));
    let mut b = Array2::<f64>::zeros((n, n_features));
    for i in 0..n_samples {
        for j in 0..n {
            a[[i, j]] = normal.sample(&mut rng);
        }
    }
    for i in 0..n {
        for j in 0..n_features {
            b[[i, j]] = normal.sample(&mut rng) * sigma[i];
        }
    }
    let prod = a.dot(&b);
    let mut out = Array2::<F>::zeros((n_samples, n_features));
    for i in 0..n_samples {
        for j in 0..n_features {
            out[[i, j]] = F::from(prod[[i, j]]).ok_or_else(|| FerroError::InvalidParameter {
                name: "matrix".into(),
                reason: "could not convert".into(),
            })?;
        }
    }
    Ok(out)
}

/// Generate a random symmetric positive-definite matrix.
///
/// Constructs `A = X^T X + n * I` where `X` is `n x n` with i.i.d. Gaussian
/// entries and `I` is the identity. The result is symmetric and SPD.
pub fn make_spd_matrix<F>(n: usize, random_state: Option<u64>) -> Result<Array2<F>, FerroError>
where
    F: Float + Send + Sync + 'static,
{
    if n == 0 {
        return Err(FerroError::InvalidParameter {
            name: "n".into(),
            reason: "must be >= 1".into(),
        });
    }
    let mut rng = make_rng(random_state);
    let normal = Normal::new(0.0_f64, 1.0_f64).map_err(|e| FerroError::InvalidParameter {
        name: "normal".into(),
        reason: e.to_string(),
    })?;
    let mut a = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            a[[i, j]] = normal.sample(&mut rng);
        }
    }
    let sym = a.t().dot(&a);
    let mut out = Array2::<F>::zeros((n, n));
    let n_f = n as f64;
    for i in 0..n {
        for j in 0..n {
            let v = sym[[i, j]] + if i == j { n_f } else { 0.0 };
            out[[i, j]] = F::from(v).ok_or_else(|| FerroError::InvalidParameter {
                name: "matrix".into(),
                reason: "could not convert".into(),
            })?;
        }
    }
    Ok(out)
}

/// Generate a random sparse symmetric positive-definite matrix.
///
/// Builds an SPD matrix whose off-diagonal entries are zero with probability
/// `1 - alpha` (so `alpha = 1.0` reproduces [`make_spd_matrix`]).
pub fn make_sparse_spd_matrix<F>(
    n: usize,
    alpha: f64,
    random_state: Option<u64>,
) -> Result<Array2<F>, FerroError>
where
    F: Float + Send + Sync + 'static,
{
    if !(0.0..=1.0).contains(&alpha) {
        return Err(FerroError::InvalidParameter {
            name: "alpha".into(),
            reason: format!("must be in [0, 1], got {alpha}"),
        });
    }
    let mut rng = make_rng(random_state);
    let normal = Normal::new(0.0_f64, 1.0_f64).map_err(|e| FerroError::InvalidParameter {
        name: "normal".into(),
        reason: e.to_string(),
    })?;
    let u01 = Uniform::new(0.0_f64, 1.0_f64).map_err(|e| FerroError::InvalidParameter {
        name: "uniform".into(),
        reason: e.to_string(),
    })?;
    let mut prec = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in (i + 1)..n {
            if u01.sample(&mut rng) < alpha {
                let v = normal.sample(&mut rng);
                prec[[i, j]] = v;
                prec[[j, i]] = v;
            }
        }
        prec[[i, i]] = 1.0;
    }
    // ensure SPD: shift diag by sum of row absolute values
    for i in 0..n {
        let row_sum: f64 = (0..n).filter(|&j| j != i).map(|j| prec[[i, j]].abs()).sum();
        prec[[i, i]] = row_sum + 1.0;
    }
    let mut out = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            out[[i, j]] = F::from(prec[[i, j]]).ok_or_else(|| FerroError::InvalidParameter {
                name: "matrix".into(),
                reason: "could not convert".into(),
            })?;
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Gaussian quantiles, Hastie 10.2, multilabel classification
// ---------------------------------------------------------------------------

/// Generate isotropic Gaussian samples and split them into `n_classes`
/// concentric quantile shells.
///
/// All samples come from a standard normal distribution; the `i`-th class
/// occupies the `i / n_classes` to `(i + 1) / n_classes` quantile of the
/// sample's squared norm.
pub fn make_gaussian_quantiles<F>(
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
    random_state: Option<u64>,
) -> Result<(Array2<F>, Array1<usize>), FerroError>
where
    F: Float + Send + Sync + 'static,
{
    if n_classes < 2 {
        return Err(FerroError::InvalidParameter {
            name: "n_classes".into(),
            reason: "must be >= 2".into(),
        });
    }
    let mut rng = make_rng(random_state);
    let normal = Normal::new(0.0_f64, 1.0_f64).map_err(|e| FerroError::InvalidParameter {
        name: "normal".into(),
        reason: e.to_string(),
    })?;
    let mut x = Array2::<f64>::zeros((n_samples, n_features));
    let mut sq_norm: Vec<(usize, f64)> = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let mut s = 0.0_f64;
        for j in 0..n_features {
            let v = normal.sample(&mut rng);
            x[[i, j]] = v;
            s += v * v;
        }
        sq_norm.push((i, s));
    }
    sq_norm.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut y = Array1::<usize>::zeros(n_samples);
    for (rank, (idx, _)) in sq_norm.iter().enumerate() {
        let cls = (rank * n_classes) / n_samples;
        y[*idx] = cls.min(n_classes - 1);
    }
    let mut out = Array2::<F>::zeros((n_samples, n_features));
    for i in 0..n_samples {
        for j in 0..n_features {
            out[[i, j]] = F::from(x[[i, j]]).ok_or_else(|| FerroError::InvalidParameter {
                name: "x".into(),
                reason: "could not convert".into(),
            })?;
        }
    }
    Ok((out, y))
}

/// Generate Hastie's binary classification benchmark from
/// "Elements of Statistical Learning" (Ex 10.2).
///
/// Produces 10-dimensional standard-normal inputs; the binary label is
/// `1` iff `sum(x_i^2) > 9.34`.
pub fn make_hastie_10_2<F>(
    n_samples: usize,
    random_state: Option<u64>,
) -> Result<(Array2<F>, Array1<usize>), FerroError>
where
    F: Float + Send + Sync + 'static,
{
    let mut rng = make_rng(random_state);
    let normal = Normal::new(0.0_f64, 1.0_f64).map_err(|e| FerroError::InvalidParameter {
        name: "normal".into(),
        reason: e.to_string(),
    })?;
    let mut x = Array2::<F>::zeros((n_samples, 10));
    let mut y = Array1::<usize>::zeros(n_samples);
    for i in 0..n_samples {
        let mut s = 0.0_f64;
        for j in 0..10 {
            let v = normal.sample(&mut rng);
            s += v * v;
            x[[i, j]] = F::from(v).ok_or_else(|| FerroError::InvalidParameter {
                name: "x".into(),
                reason: "could not convert".into(),
            })?;
        }
        // 9.34 ≈ median of chi-squared with 10 df
        y[i] = if s > 9.34 { 1 } else { 0 };
    }
    Ok((x, y))
}

/// Generate a random multi-label classification dataset.
///
/// Returns `(X, Y)` where `X` is `(n_samples, n_features)` and `Y` is a
/// binary `(n_samples, n_classes)` indicator matrix. Each sample is assigned
/// a label combination drawn from a Poisson distribution truncated to
/// `[0, n_classes]`.
pub fn make_multilabel_classification<F>(
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
    n_labels: usize,
    random_state: Option<u64>,
) -> Result<(Array2<F>, Array2<usize>), FerroError>
where
    F: Float + Send + Sync + 'static,
{
    if n_classes == 0 {
        return Err(FerroError::InvalidParameter {
            name: "n_classes".into(),
            reason: "must be >= 1".into(),
        });
    }
    let mut rng = make_rng(random_state);
    let unif = Uniform::new(0.0_f64, 1.0_f64).map_err(|e| FerroError::InvalidParameter {
        name: "uniform".into(),
        reason: e.to_string(),
    })?;

    let mut x = Array2::<F>::zeros((n_samples, n_features));
    let mut y = Array2::<usize>::zeros((n_samples, n_classes));
    for i in 0..n_samples {
        // pick number of labels for this sample (clamp to [1, n_classes])
        let target_labels = n_labels.clamp(1, n_classes);
        // pick `target_labels` distinct random classes
        let mut chosen = std::collections::HashSet::new();
        while chosen.len() < target_labels {
            let c = (unif.sample(&mut rng) * n_classes as f64).floor() as usize;
            chosen.insert(c.min(n_classes - 1));
        }
        for c in &chosen {
            y[[i, *c]] = 1;
        }
        for j in 0..n_features {
            // X is just standard-normal noise modulated by labels.
            let mut v = unif.sample(&mut rng) - 0.5;
            for c in &chosen {
                v += ((*c + 1) as f64) * 0.1;
            }
            x[[i, j]] = F::from(v).ok_or_else(|| FerroError::InvalidParameter {
                name: "x".into(),
                reason: "could not convert".into(),
            })?;
        }
    }
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
        assert_eq!(unique.len(), 3, "expected 3 unique classes, got {unique:?}");
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
        let mut counts = [0_usize; 3];
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

    // --- make_swiss_roll ---

    #[test]
    fn test_make_swiss_roll_shape() {
        let (x, t) = make_swiss_roll::<f64>(200, 0.0, Some(42)).unwrap();
        assert_eq!(x.shape(), &[200, 3]);
        assert_eq!(t.len(), 200);
    }

    #[test]
    fn test_make_swiss_roll_f32() {
        let (x, t) = make_swiss_roll::<f32>(50, 0.0, Some(1)).unwrap();
        assert_eq!(x.shape(), &[50, 3]);
        assert_eq!(t.len(), 50);
    }

    #[test]
    fn test_make_swiss_roll_reproducible() {
        let (x1, t1) = make_swiss_roll::<f64>(100, 0.5, Some(7)).unwrap();
        let (x2, t2) = make_swiss_roll::<f64>(100, 0.5, Some(7)).unwrap();
        assert_eq!(x1, x2);
        assert_eq!(t1, t2);
    }

    #[test]
    fn test_make_swiss_roll_different_seeds() {
        let (x1, _) = make_swiss_roll::<f64>(100, 0.0, Some(1)).unwrap();
        let (x2, _) = make_swiss_roll::<f64>(100, 0.0, Some(999)).unwrap();
        assert_ne!(x1, x2, "different seeds should give different data");
    }

    #[test]
    fn test_make_swiss_roll_t_range() {
        // t should be in [1.5π, 4.5π]
        let (_, t) = make_swiss_roll::<f64>(300, 0.0, Some(0)).unwrap();
        let low = 1.5 * PI;
        let high = 4.5 * PI;
        for &ti in &t {
            assert!(ti >= low && ti <= high, "t={ti} outside [{low}, {high}]");
        }
    }

    #[test]
    fn test_make_swiss_roll_no_noise_on_manifold() {
        // Without noise, x = t*cos(t), z = t*sin(t), so
        // sqrt(x^2 + z^2) = |t| and y ∈ [0, 21).
        let (x, t) = make_swiss_roll::<f64>(100, 0.0, Some(0)).unwrap();
        for i in 0..x.nrows() {
            let xi = x[[i, 0]];
            let yi = x[[i, 1]];
            let zi = x[[i, 2]];
            let ti = t[i];
            let expected_r = ti; // t is positive in [1.5π, 4.5π]
            let actual_r = (xi * xi + zi * zi).sqrt();
            assert!(
                (actual_r - expected_r).abs() < 1e-10,
                "point {i}: radius {actual_r} != t={expected_r}"
            );
            assert!(
                (0.0..21.0).contains(&yi),
                "point {i}: y={yi} outside [0,21)"
            );
        }
    }

    #[test]
    fn test_make_swiss_roll_invalid_n_samples_zero() {
        assert!(make_swiss_roll::<f64>(0, 0.0, None).is_err());
    }

    #[test]
    fn test_make_swiss_roll_invalid_noise_negative() {
        assert!(make_swiss_roll::<f64>(100, -1.0_f64, None).is_err());
    }

    #[test]
    fn test_make_swiss_roll_with_noise() {
        // With noise the manifold constraint is approximate; just check shapes.
        let (x, t) = make_swiss_roll::<f64>(100, 0.5, Some(42)).unwrap();
        assert_eq!(x.shape(), &[100, 3]);
        assert_eq!(t.len(), 100);
        // All values should be finite.
        assert!(x.iter().all(|v| v.is_finite()));
        assert!(t.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_make_swiss_roll_single_sample() {
        let (x, t) = make_swiss_roll::<f64>(1, 0.0, Some(0)).unwrap();
        assert_eq!(x.shape(), &[1, 3]);
        assert_eq!(t.len(), 1);
    }

    // --- make_s_curve ---

    #[test]
    fn test_make_s_curve_shape() {
        let (x, t) = make_s_curve::<f64>(200, 0.0, Some(42)).unwrap();
        assert_eq!(x.shape(), &[200, 3]);
        assert_eq!(t.len(), 200);
    }

    #[test]
    fn test_make_s_curve_f32() {
        let (x, t) = make_s_curve::<f32>(50, 0.0, Some(1)).unwrap();
        assert_eq!(x.shape(), &[50, 3]);
        assert_eq!(t.len(), 50);
    }

    #[test]
    fn test_make_s_curve_reproducible() {
        let (x1, t1) = make_s_curve::<f64>(100, 0.5, Some(7)).unwrap();
        let (x2, t2) = make_s_curve::<f64>(100, 0.5, Some(7)).unwrap();
        assert_eq!(x1, x2);
        assert_eq!(t1, t2);
    }

    #[test]
    fn test_make_s_curve_different_seeds() {
        let (x1, _) = make_s_curve::<f64>(100, 0.0, Some(1)).unwrap();
        let (x2, _) = make_s_curve::<f64>(100, 0.0, Some(999)).unwrap();
        assert_ne!(x1, x2, "different seeds should give different data");
    }

    #[test]
    fn test_make_s_curve_t_range() {
        // t should be in [-3π/2, 3π/2]
        let (_, t) = make_s_curve::<f64>(300, 0.0, Some(0)).unwrap();
        let low = -1.5 * PI;
        let high = 1.5 * PI;
        for &ti in &t {
            assert!(ti >= low && ti <= high, "t={ti} outside [{low}, {high}]");
        }
    }

    #[test]
    fn test_make_s_curve_no_noise_x_is_sin_t() {
        // Without noise: x = sin(t)
        let (x, t) = make_s_curve::<f64>(100, 0.0, Some(0)).unwrap();
        for i in 0..x.nrows() {
            let xi = x[[i, 0]];
            let ti = t[i];
            assert!(
                (xi - ti.sin()).abs() < 1e-10,
                "point {i}: x={xi} != sin(t)={} (t={ti})",
                ti.sin()
            );
        }
    }

    #[test]
    fn test_make_s_curve_no_noise_y_in_range() {
        // Without noise: y ∈ [0, 2)
        let (x, _) = make_s_curve::<f64>(200, 0.0, Some(0)).unwrap();
        for i in 0..x.nrows() {
            let yi = x[[i, 1]];
            assert!((0.0..2.0).contains(&yi), "point {i}: y={yi} outside [0,2)");
        }
    }

    #[test]
    fn test_make_s_curve_invalid_n_samples_zero() {
        assert!(make_s_curve::<f64>(0, 0.0, None).is_err());
    }

    #[test]
    fn test_make_s_curve_invalid_noise_negative() {
        assert!(make_s_curve::<f64>(100, -0.5_f64, None).is_err());
    }

    #[test]
    fn test_make_s_curve_with_noise() {
        let (x, t) = make_s_curve::<f64>(100, 0.5, Some(42)).unwrap();
        assert_eq!(x.shape(), &[100, 3]);
        assert_eq!(t.len(), 100);
        assert!(x.iter().all(|v| v.is_finite()));
        assert!(t.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_make_s_curve_single_sample() {
        let (x, t) = make_s_curve::<f64>(1, 0.0, Some(0)).unwrap();
        assert_eq!(x.shape(), &[1, 3]);
        assert_eq!(t.len(), 1);
    }

    // --- make_sparse_uncorrelated ---

    #[test]
    fn test_make_sparse_uncorrelated_shape() {
        let (x, y) = make_sparse_uncorrelated::<f64>(100, 10, Some(0)).unwrap();
        assert_eq!(x.shape(), &[100, 10]);
        assert_eq!(y.len(), 100);
    }

    #[test]
    fn test_make_sparse_uncorrelated_reproducible() {
        let (x1, y1) = make_sparse_uncorrelated::<f64>(50, 8, Some(42)).unwrap();
        let (x2, y2) = make_sparse_uncorrelated::<f64>(50, 8, Some(42)).unwrap();
        assert_eq!(x1, x2);
        assert_eq!(y1, y2);
    }

    #[test]
    fn test_make_sparse_uncorrelated_different_seeds() {
        let (x1, _) = make_sparse_uncorrelated::<f64>(50, 8, Some(1)).unwrap();
        let (x2, _) = make_sparse_uncorrelated::<f64>(50, 8, Some(999)).unwrap();
        assert_ne!(x1, x2, "different seeds should produce different data");
    }

    #[test]
    fn test_make_sparse_uncorrelated_target_from_first_5_features() {
        // With no noise the target must equal the linear combination of the
        // first 5 features with fixed weights [1, 2, 3, 4, 5].
        let (x, y) = make_sparse_uncorrelated::<f64>(20, 5, Some(7)).unwrap();
        let weights = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
        for i in 0..20 {
            let expected: f64 = weights
                .iter()
                .enumerate()
                .map(|(j, &w)| w * x[[i, j]])
                .sum();
            let diff = (y[i] - expected).abs();
            assert!(diff < 1e-9, "row {i}: expected {expected}, got {}", y[i]);
        }
    }

    #[test]
    fn test_make_sparse_uncorrelated_informative_features_only() {
        // The target should depend only on the first 5 features regardless of
        // extra noise columns. Verify by recomputing the target from X directly.
        let (x, y) = make_sparse_uncorrelated::<f64>(30, 10, Some(10)).unwrap();

        let weights = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
        for i in 0..30 {
            let expected: f64 = weights
                .iter()
                .enumerate()
                .map(|(j, &w)| w * x[[i, j]])
                .sum();
            assert!(
                (y[i] - expected).abs() < 1e-9,
                "row {i}: expected {expected}, got {}",
                y[i]
            );
        }
    }

    #[test]
    fn test_make_sparse_uncorrelated_invalid_n_samples_zero() {
        assert!(make_sparse_uncorrelated::<f64>(0, 10, None).is_err());
    }

    #[test]
    fn test_make_sparse_uncorrelated_invalid_n_features_too_small() {
        assert!(make_sparse_uncorrelated::<f64>(10, 4, None).is_err());
    }

    #[test]
    fn test_make_sparse_uncorrelated_f32() {
        let (x, y) = make_sparse_uncorrelated::<f32>(30, 7, Some(5)).unwrap();
        assert_eq!(x.shape(), &[30, 7]);
        assert_eq!(y.len(), 30);
        assert!(x.iter().all(|v| v.is_finite()));
        assert!(y.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_make_sparse_uncorrelated_exactly_5_features() {
        // n_features == 5 should be valid.
        let (x, y) = make_sparse_uncorrelated::<f64>(20, 5, Some(3)).unwrap();
        assert_eq!(x.shape(), &[20, 5]);
        assert_eq!(y.len(), 20);
    }
}
