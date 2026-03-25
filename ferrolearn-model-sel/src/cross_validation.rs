//! Cross-validation utilities.
//!
//! This module provides:
//!
//! - [`KFold`] — k-fold cross-validation index splitter.
//! - [`StratifiedKFold`] — stratified k-fold that preserves class proportions.
//! - [`LeaveOneOut`] — leave-one-out cross-validation (n folds, 1 test sample each).
//! - [`LeavePOut`] — leave-p-out cross-validation (all C(n, p) combinations).
//! - [`RepeatedKFold`] — repeated k-fold with different shuffles per repeat.
//! - [`RepeatedStratifiedKFold`] — repeated stratified k-fold.
//! - [`ShuffleSplit`] — random train/test splits.
//! - [`StratifiedShuffleSplit`] — stratified random train/test splits.
//! - [`GroupKFold`] — k-fold where groups are kept together.
//! - [`CrossValidator`] — a trait abstracting over fold-index generators.
//! - [`cross_val_score`] — run a [`ferrolearn_core::pipeline::Pipeline`]
//!   through each fold and collect scores.

use std::collections::HashMap;

use ferrolearn_core::pipeline::Pipeline;
use ferrolearn_core::{FerroError, Fit, Predict};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;

/// A single cross-validation split: `(train_indices, test_indices)`.
pub type FoldSplit = (Vec<usize>, Vec<usize>);

/// The result type returned by [`CrossValidator::fold_indices`].
pub type FoldSplits = Vec<FoldSplit>;

// ---------------------------------------------------------------------------
// CrossValidator trait
// ---------------------------------------------------------------------------

/// A trait for objects that can generate fold indices.
///
/// Implement this trait to provide custom splitting strategies that can be
/// passed to [`cross_val_score`].
pub trait CrossValidator {
    /// Return `(train_indices, test_indices)` pairs for each fold.
    ///
    /// The number of pairs equals the number of folds/splits.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError`] if the split cannot be produced (e.g. not
    /// enough samples for the requested number of splits).
    fn fold_indices(&self, n_samples: usize) -> Result<FoldSplits, FerroError>;
}

// ---------------------------------------------------------------------------
// KFold
// ---------------------------------------------------------------------------

/// K-fold cross-validation splitter.
///
/// Splits data into `n_splits` consecutive folds. Each fold is used once as a
/// test set while the remaining `k−1` folds form the training set.
///
/// # Examples
///
/// ```rust
/// use ferrolearn_model_sel::KFold;
///
/// let kf = KFold::new(5);
/// let folds = kf.split(20);
/// assert_eq!(folds.len(), 5);
/// for (train, test) in &folds {
///     assert_eq!(train.len() + test.len(), 20);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct KFold {
    /// Number of folds.
    n_splits: usize,
    /// Whether to shuffle samples before splitting.
    shuffle: bool,
    /// Optional RNG seed used when `shuffle` is `true`.
    random_state: Option<u64>,
}

impl KFold {
    /// Create a new [`KFold`] with the given number of splits.
    ///
    /// By default shuffling is disabled.
    ///
    /// # Panics
    ///
    /// Does not panic; invalid `n_splits` is caught at [`split`](KFold::split)
    /// time.
    pub fn new(n_splits: usize) -> Self {
        Self {
            n_splits,
            shuffle: false,
            random_state: None,
        }
    }

    /// Enable or disable shuffling of samples before splitting.
    #[must_use]
    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Set the RNG seed used when shuffling is enabled.
    #[must_use]
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Produce `(train_indices, test_indices)` pairs for each fold.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] if `n_splits < 2`.
    /// Returns [`FerroError::InsufficientSamples`] if `n_samples < n_splits`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ferrolearn_model_sel::KFold;
    ///
    /// let folds = KFold::new(3).split(9);
    /// // Each fold has 3 test samples.
    /// for (train, test) in folds {
    ///     assert_eq!(test.len(), 3);
    /// }
    /// ```
    pub fn split(&self, n_samples: usize) -> FoldSplits {
        // Errors are returned as an empty vec here; the `CrossValidator` impl
        // returns proper `Result`s.
        self.split_result(n_samples).unwrap_or_default()
    }

    /// Internal splitting logic that returns a `Result`.
    fn split_result(&self, n_samples: usize) -> Result<FoldSplits, FerroError> {
        if self.n_splits < 2 {
            return Err(FerroError::InvalidParameter {
                name: "n_splits".into(),
                reason: format!("must be >= 2, got {}", self.n_splits),
            });
        }
        if n_samples < self.n_splits {
            return Err(FerroError::InsufficientSamples {
                required: self.n_splits,
                actual: n_samples,
                context: format!("KFold with n_splits={}", self.n_splits),
            });
        }

        // Build (possibly shuffled) index list.
        let mut indices: Vec<usize> = (0..n_samples).collect();
        if self.shuffle {
            match self.random_state {
                Some(seed) => {
                    let mut rng = SmallRng::seed_from_u64(seed);
                    indices.shuffle(&mut rng);
                }
                None => {
                    let mut rng = SmallRng::from_os_rng();
                    indices.shuffle(&mut rng);
                }
            }
        }

        // Compute fold boundary positions.
        // Distribute the remainder across the first `n_samples % n_splits`
        // folds so that fold sizes differ by at most 1.
        let base_size = n_samples / self.n_splits;
        let remainder = n_samples % self.n_splits;

        // fold_starts[i] is the start index of fold i in `indices`.
        let mut fold_starts = Vec::with_capacity(self.n_splits + 1);
        let mut pos = 0usize;
        for fold in 0..self.n_splits {
            fold_starts.push(pos);
            pos += base_size + if fold < remainder { 1 } else { 0 };
        }
        fold_starts.push(n_samples); // sentinel

        let mut folds = Vec::with_capacity(self.n_splits);
        for fold in 0..self.n_splits {
            let test_start = fold_starts[fold];
            let test_end = fold_starts[fold + 1];
            let test: Vec<usize> = indices[test_start..test_end].to_vec();
            let train: Vec<usize> = indices[..test_start]
                .iter()
                .chain(indices[test_end..].iter())
                .copied()
                .collect();
            folds.push((train, test));
        }
        Ok(folds)
    }
}

impl CrossValidator for KFold {
    fn fold_indices(&self, n_samples: usize) -> Result<FoldSplits, FerroError> {
        self.split_result(n_samples)
    }
}

// ---------------------------------------------------------------------------
// StratifiedKFold
// ---------------------------------------------------------------------------

/// Stratified k-fold cross-validation splitter.
///
/// Like [`KFold`] but preserves the percentage of samples for each class in
/// every fold. This is useful when the target has imbalanced class counts.
///
/// # Examples
///
/// ```rust
/// use ferrolearn_model_sel::StratifiedKFold;
/// use ndarray::array;
///
/// let y = array![0usize, 0, 0, 1, 1, 1, 2, 2, 2];
/// let skf = StratifiedKFold::new(3);
/// let folds = skf.split(&y).unwrap();
/// assert_eq!(folds.len(), 3);
/// ```
#[derive(Debug, Clone)]
pub struct StratifiedKFold {
    /// Number of folds.
    n_splits: usize,
    /// Whether to shuffle within each stratum before assigning to folds.
    shuffle: bool,
    /// Optional RNG seed used when `shuffle` is `true`.
    random_state: Option<u64>,
}

impl StratifiedKFold {
    /// Create a new [`StratifiedKFold`] with the given number of splits.
    pub fn new(n_splits: usize) -> Self {
        Self {
            n_splits,
            shuffle: false,
            random_state: None,
        }
    }

    /// Enable or disable shuffling within each stratum before splitting.
    #[must_use]
    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Set the RNG seed used when shuffling is enabled.
    #[must_use]
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Generate `(train_indices, test_indices)` pairs for each fold,
    /// preserving class distribution.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InvalidParameter`] if `n_splits < 2`.
    /// - [`FerroError::InsufficientSamples`] if any class has fewer samples
    ///   than `n_splits`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ferrolearn_model_sel::StratifiedKFold;
    /// use ndarray::Array1;
    ///
    /// let y: Array1<usize> = Array1::from_iter(
    ///     (0..12).map(|i| i % 3)
    /// );
    /// let skf = StratifiedKFold::new(3);
    /// let folds = skf.split(&y).unwrap();
    /// assert_eq!(folds.len(), 3);
    /// for (train, test) in &folds {
    ///     assert_eq!(train.len() + test.len(), 12);
    /// }
    /// ```
    pub fn split(&self, y: &Array1<usize>) -> Result<FoldSplits, FerroError> {
        let n_samples = y.len();

        if self.n_splits < 2 {
            return Err(FerroError::InvalidParameter {
                name: "n_splits".into(),
                reason: format!("must be >= 2, got {}", self.n_splits),
            });
        }
        if n_samples < self.n_splits {
            return Err(FerroError::InsufficientSamples {
                required: self.n_splits,
                actual: n_samples,
                context: format!("StratifiedKFold with n_splits={}", self.n_splits),
            });
        }

        // Group sample indices by class label.
        let mut class_indices: HashMap<usize, Vec<usize>> = HashMap::new();
        for (i, &label) in y.iter().enumerate() {
            class_indices.entry(label).or_default().push(i);
        }

        // Sort classes for deterministic behaviour.
        let mut classes: Vec<usize> = class_indices.keys().copied().collect();
        classes.sort_unstable();

        // Optionally shuffle within each class stratum.
        if self.shuffle {
            let mut rng: SmallRng = match self.random_state {
                Some(seed) => SmallRng::seed_from_u64(seed),
                None => SmallRng::from_os_rng(),
            };
            for class in &classes {
                class_indices.get_mut(class).unwrap().shuffle(&mut rng);
            }
        }

        // Validate that every class has enough samples.
        for &class in &classes {
            let count = class_indices[&class].len();
            if count < self.n_splits {
                return Err(FerroError::InsufficientSamples {
                    required: self.n_splits,
                    actual: count,
                    context: format!("StratifiedKFold: class {class} has too few samples"),
                });
            }
        }

        // For each class, assign its samples to folds in round-robin fashion.
        // fold_test_indices[fold] accumulates the test indices for that fold.
        //
        // A global `fold_offset` rotates which folds receive the extra samples
        // when `class_count % n_splits != 0`. This matches sklearn's behaviour
        // and avoids front-loading the first few folds.
        let mut fold_test_indices: Vec<Vec<usize>> = vec![Vec::new(); self.n_splits];
        let mut fold_offset: usize = 0;
        for class in &classes {
            let idx = &class_indices[class];
            let base = idx.len() / self.n_splits;
            let extra = idx.len() % self.n_splits;
            let mut pos = 0;
            for (fold_idx, bucket) in fold_test_indices.iter_mut().enumerate() {
                // This fold gets an extra sample if it falls within the `extra`
                // slots starting at `fold_offset` (wrapping around).
                let gets_extra = if extra > 0 {
                    // Distance from fold_offset, modulo n_splits
                    let d = (fold_idx + self.n_splits - fold_offset) % self.n_splits;
                    d < extra
                } else {
                    false
                };
                let size = base + if gets_extra { 1 } else { 0 };
                bucket.extend_from_slice(&idx[pos..pos + size]);
                pos += size;
            }
            fold_offset = (fold_offset + extra) % self.n_splits;
        }

        // Build (train, test) pairs.
        let all_indices: Vec<usize> = (0..n_samples).collect();
        let mut folds = Vec::with_capacity(self.n_splits);
        for test in fold_test_indices {
            let test_set: std::collections::HashSet<usize> = test.iter().copied().collect();
            let train: Vec<usize> = all_indices
                .iter()
                .copied()
                .filter(|i| !test_set.contains(i))
                .collect();
            folds.push((train, test));
        }
        Ok(folds)
    }
}

// ---------------------------------------------------------------------------
// LeaveOneOut
// ---------------------------------------------------------------------------

/// Leave-one-out cross-validation splitter.
///
/// Each sample is used once as the test set, with all remaining samples
/// forming the training set. This produces `n_samples` folds.
///
/// # Examples
///
/// ```rust
/// use ferrolearn_model_sel::LeaveOneOut;
/// use ferrolearn_model_sel::CrossValidator;
///
/// let loo = LeaveOneOut;
/// let folds = loo.fold_indices(5).unwrap();
/// assert_eq!(folds.len(), 5);
/// for (train, test) in &folds {
///     assert_eq!(test.len(), 1);
///     assert_eq!(train.len(), 4);
/// }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct LeaveOneOut;

impl CrossValidator for LeaveOneOut {
    fn fold_indices(&self, n_samples: usize) -> Result<FoldSplits, FerroError> {
        if n_samples < 2 {
            return Err(FerroError::InsufficientSamples {
                required: 2,
                actual: n_samples,
                context: "LeaveOneOut requires at least 2 samples".into(),
            });
        }
        let mut folds = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            let test = vec![i];
            let train: Vec<usize> = (0..n_samples).filter(|&j| j != i).collect();
            folds.push((train, test));
        }
        Ok(folds)
    }
}

// ---------------------------------------------------------------------------
// LeavePOut
// ---------------------------------------------------------------------------

/// Leave-p-out cross-validation splitter.
///
/// Generates all C(n, p) combinations of `p` samples as the test set,
/// with the remaining `n - p` samples forming the training set.
///
/// # Warning
///
/// The number of folds grows combinatorially: C(n, p). For large `n` and
/// moderate `p` this can be very expensive. Use with caution.
///
/// # Examples
///
/// ```rust
/// use ferrolearn_model_sel::LeavePOut;
/// use ferrolearn_model_sel::CrossValidator;
///
/// let lpo = LeavePOut::new(2);
/// let folds = lpo.fold_indices(4).unwrap();
/// // C(4, 2) = 6 folds
/// assert_eq!(folds.len(), 6);
/// for (train, test) in &folds {
///     assert_eq!(test.len(), 2);
///     assert_eq!(train.len(), 2);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct LeavePOut {
    /// Number of samples to leave out as the test set.
    p: usize,
}

impl LeavePOut {
    /// Create a new [`LeavePOut`] splitter that leaves out `p` samples per fold.
    pub fn new(p: usize) -> Self {
        Self { p }
    }
}

impl CrossValidator for LeavePOut {
    fn fold_indices(&self, n_samples: usize) -> Result<FoldSplits, FerroError> {
        if self.p == 0 {
            return Err(FerroError::InvalidParameter {
                name: "p".into(),
                reason: "must be >= 1".into(),
            });
        }
        if self.p >= n_samples {
            return Err(FerroError::InsufficientSamples {
                required: self.p + 1,
                actual: n_samples,
                context: format!("LeavePOut with p={}", self.p),
            });
        }

        let indices: Vec<usize> = (0..n_samples).collect();
        let combinations = combinations_of(&indices, self.p);
        let mut folds = Vec::with_capacity(combinations.len());
        for test in combinations {
            let test_set: std::collections::HashSet<usize> = test.iter().copied().collect();
            let train: Vec<usize> = indices
                .iter()
                .copied()
                .filter(|i| !test_set.contains(i))
                .collect();
            folds.push((train, test));
        }
        Ok(folds)
    }
}

/// Generate all combinations of `k` elements from `items`.
fn combinations_of(items: &[usize], k: usize) -> Vec<Vec<usize>> {
    let n = items.len();
    if k == 0 || k > n {
        return Vec::new();
    }
    let mut result = Vec::new();
    let mut combo = vec![0usize; k];
    combinations_recurse(items, k, 0, 0, &mut combo, &mut result);
    result
}

fn combinations_recurse(
    items: &[usize],
    k: usize,
    start: usize,
    depth: usize,
    combo: &mut Vec<usize>,
    result: &mut Vec<Vec<usize>>,
) {
    if depth == k {
        result.push(combo.clone());
        return;
    }
    let n = items.len();
    for i in start..=(n - k + depth) {
        combo[depth] = items[i];
        combinations_recurse(items, k, i + 1, depth + 1, combo, result);
    }
}

// ---------------------------------------------------------------------------
// RepeatedKFold
// ---------------------------------------------------------------------------

/// Repeated k-fold cross-validation splitter.
///
/// Runs [`KFold`] `n_repeats` times with different random shuffles,
/// producing `n_splits * n_repeats` total folds.
///
/// # Examples
///
/// ```rust
/// use ferrolearn_model_sel::RepeatedKFold;
/// use ferrolearn_model_sel::CrossValidator;
///
/// let rkf = RepeatedKFold::new(5, 3).random_state(42);
/// let folds = rkf.fold_indices(20).unwrap();
/// assert_eq!(folds.len(), 15); // 5 splits * 3 repeats
/// ```
#[derive(Debug, Clone)]
pub struct RepeatedKFold {
    /// Number of folds per repetition.
    n_splits: usize,
    /// Number of times to repeat the k-fold procedure.
    n_repeats: usize,
    /// Optional RNG seed for reproducibility.
    random_state: Option<u64>,
}

impl RepeatedKFold {
    /// Create a new [`RepeatedKFold`] with the given number of splits and repeats.
    pub fn new(n_splits: usize, n_repeats: usize) -> Self {
        Self {
            n_splits,
            n_repeats,
            random_state: None,
        }
    }

    /// Set the RNG seed for reproducible shuffles.
    #[must_use]
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

impl CrossValidator for RepeatedKFold {
    fn fold_indices(&self, n_samples: usize) -> Result<FoldSplits, FerroError> {
        if self.n_repeats == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_repeats".into(),
                reason: "must be >= 1".into(),
            });
        }

        let mut all_folds = Vec::with_capacity(self.n_splits * self.n_repeats);

        // Use a master RNG to derive per-repeat seeds.
        let mut master_rng: SmallRng = match self.random_state {
            Some(seed) => SmallRng::seed_from_u64(seed),
            None => SmallRng::from_os_rng(),
        };

        for _ in 0..self.n_repeats {
            // Derive a seed for this repeat from the master RNG.
            let repeat_seed = rand::Rng::random::<u64>(&mut master_rng);
            let kf = KFold::new(self.n_splits)
                .shuffle(true)
                .random_state(repeat_seed);
            let folds = kf.split_result(n_samples)?;
            all_folds.extend(folds);
        }

        Ok(all_folds)
    }
}

// ---------------------------------------------------------------------------
// RepeatedStratifiedKFold
// ---------------------------------------------------------------------------

/// Repeated stratified k-fold cross-validation splitter.
///
/// Runs [`StratifiedKFold`] `n_repeats` times with different random shuffles,
/// producing `n_splits * n_repeats` total folds while preserving class
/// proportions in each fold.
///
/// Since stratified splitting requires class labels, call [`split`](RepeatedStratifiedKFold::split)
/// directly rather than the label-free [`CrossValidator::fold_indices`] trait method.
/// The trait method returns an error advising to use `split` instead.
///
/// # Examples
///
/// ```rust
/// use ferrolearn_model_sel::RepeatedStratifiedKFold;
/// use ndarray::Array1;
///
/// let y: Array1<usize> = Array1::from_iter((0..12).map(|i| i % 3));
/// let rskf = RepeatedStratifiedKFold::new(3, 2).random_state(42);
/// let folds = rskf.split(&y).unwrap();
/// assert_eq!(folds.len(), 6); // 3 splits * 2 repeats
/// ```
#[derive(Debug, Clone)]
pub struct RepeatedStratifiedKFold {
    /// Number of folds per repetition.
    n_splits: usize,
    /// Number of times to repeat the stratified k-fold procedure.
    n_repeats: usize,
    /// Optional RNG seed for reproducibility.
    random_state: Option<u64>,
}

impl RepeatedStratifiedKFold {
    /// Create a new [`RepeatedStratifiedKFold`] with the given number of splits and repeats.
    pub fn new(n_splits: usize, n_repeats: usize) -> Self {
        Self {
            n_splits,
            n_repeats,
            random_state: None,
        }
    }

    /// Set the RNG seed for reproducible shuffles.
    #[must_use]
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Generate `(train_indices, test_indices)` pairs for each fold across
    /// all repeats, preserving class distribution.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] if `n_splits < 2` or `n_repeats == 0`.
    /// Returns [`FerroError::InsufficientSamples`] if any class has fewer samples than `n_splits`.
    pub fn split(&self, y: &Array1<usize>) -> Result<FoldSplits, FerroError> {
        if self.n_repeats == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_repeats".into(),
                reason: "must be >= 1".into(),
            });
        }

        let mut all_folds = Vec::with_capacity(self.n_splits * self.n_repeats);

        let mut master_rng: SmallRng = match self.random_state {
            Some(seed) => SmallRng::seed_from_u64(seed),
            None => SmallRng::from_os_rng(),
        };

        for _ in 0..self.n_repeats {
            let repeat_seed = rand::Rng::random::<u64>(&mut master_rng);
            let skf = StratifiedKFold::new(self.n_splits)
                .shuffle(true)
                .random_state(repeat_seed);
            let folds = skf.split(y)?;
            all_folds.extend(folds);
        }

        Ok(all_folds)
    }
}

impl CrossValidator for RepeatedStratifiedKFold {
    /// Returns an error advising to use [`split`](RepeatedStratifiedKFold::split) with labels.
    fn fold_indices(&self, _n_samples: usize) -> Result<FoldSplits, FerroError> {
        Err(FerroError::InvalidParameter {
            name: "labels".into(),
            reason: "RepeatedStratifiedKFold requires class labels; \
                     use the `split(y)` method instead of `fold_indices`"
                .into(),
        })
    }
}

// ---------------------------------------------------------------------------
// ShuffleSplit
// ---------------------------------------------------------------------------

/// Random permutation cross-validator (Monte Carlo cross-validation).
///
/// Generates `n_splits` random train/test splits. Unlike [`KFold`], splits
/// may overlap between iterations (the same sample may appear in the test
/// set of multiple splits).
///
/// # Examples
///
/// ```rust
/// use ferrolearn_model_sel::ShuffleSplit;
/// use ferrolearn_model_sel::CrossValidator;
///
/// let ss = ShuffleSplit::new(5).test_size(0.25).random_state(42);
/// let folds = ss.fold_indices(100).unwrap();
/// assert_eq!(folds.len(), 5);
/// for (train, test) in &folds {
///     assert_eq!(test.len(), 25);
///     assert_eq!(train.len(), 75);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct ShuffleSplit {
    /// Number of re-shuffling & splitting iterations.
    n_splits: usize,
    /// Fraction of samples to include in the test set.
    test_size: f64,
    /// Optional RNG seed for reproducibility.
    random_state: Option<u64>,
}

impl ShuffleSplit {
    /// Create a new [`ShuffleSplit`] with the given number of splits.
    ///
    /// Default test size is 0.1 (10%).
    pub fn new(n_splits: usize) -> Self {
        Self {
            n_splits,
            test_size: 0.1,
            random_state: None,
        }
    }

    /// Set the fraction of samples to include in the test set.
    ///
    /// Must be in `(0.0, 1.0)`.
    #[must_use]
    pub fn test_size(mut self, test_size: f64) -> Self {
        self.test_size = test_size;
        self
    }

    /// Set the RNG seed for reproducible splits.
    #[must_use]
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

impl CrossValidator for ShuffleSplit {
    fn fold_indices(&self, n_samples: usize) -> Result<FoldSplits, FerroError> {
        if self.n_splits == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_splits".into(),
                reason: "must be >= 1".into(),
            });
        }
        if self.test_size <= 0.0 || self.test_size >= 1.0 {
            return Err(FerroError::InvalidParameter {
                name: "test_size".into(),
                reason: format!("must be in (0.0, 1.0), got {}", self.test_size),
            });
        }

        let n_test = (n_samples as f64 * self.test_size).ceil() as usize;
        let n_test = n_test.max(1).min(n_samples - 1);
        let n_train = n_samples - n_test;

        if n_train == 0 || n_test == 0 {
            return Err(FerroError::InsufficientSamples {
                required: 2,
                actual: n_samples,
                context: "ShuffleSplit requires at least 2 samples".into(),
            });
        }

        let mut rng: SmallRng = match self.random_state {
            Some(seed) => SmallRng::seed_from_u64(seed),
            None => SmallRng::from_os_rng(),
        };

        let mut folds = Vec::with_capacity(self.n_splits);
        for _ in 0..self.n_splits {
            let mut indices: Vec<usize> = (0..n_samples).collect();
            indices.shuffle(&mut rng);
            let test: Vec<usize> = indices[..n_test].to_vec();
            let train: Vec<usize> = indices[n_test..].to_vec();
            folds.push((train, test));
        }

        Ok(folds)
    }
}

// ---------------------------------------------------------------------------
// StratifiedShuffleSplit
// ---------------------------------------------------------------------------

/// Stratified random permutation cross-validator.
///
/// Like [`ShuffleSplit`] but preserves the percentage of samples for each
/// class in every split.
///
/// Since stratified splitting requires class labels, call
/// [`split`](StratifiedShuffleSplit::split) directly rather than the label-free
/// [`CrossValidator::fold_indices`] trait method. The trait method returns an
/// error advising to use `split` instead.
///
/// # Examples
///
/// ```rust
/// use ferrolearn_model_sel::StratifiedShuffleSplit;
/// use ndarray::Array1;
///
/// let y: Array1<usize> = Array1::from_iter((0..100).map(|i| i % 2));
/// let sss = StratifiedShuffleSplit::new(3).test_size(0.2).random_state(42);
/// let folds = sss.split(&y).unwrap();
/// assert_eq!(folds.len(), 3);
/// ```
#[derive(Debug, Clone)]
pub struct StratifiedShuffleSplit {
    /// Number of re-shuffling & splitting iterations.
    n_splits: usize,
    /// Fraction of samples to include in the test set.
    test_size: f64,
    /// Optional RNG seed for reproducibility.
    random_state: Option<u64>,
}

impl StratifiedShuffleSplit {
    /// Create a new [`StratifiedShuffleSplit`] with the given number of splits.
    ///
    /// Default test size is 0.1 (10%).
    pub fn new(n_splits: usize) -> Self {
        Self {
            n_splits,
            test_size: 0.1,
            random_state: None,
        }
    }

    /// Set the fraction of samples to include in the test set.
    ///
    /// Must be in `(0.0, 1.0)`.
    #[must_use]
    pub fn test_size(mut self, test_size: f64) -> Self {
        self.test_size = test_size;
        self
    }

    /// Set the RNG seed for reproducible splits.
    #[must_use]
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Generate `(train_indices, test_indices)` pairs for each split,
    /// preserving class proportions.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] if `test_size` is out of range or `n_splits == 0`.
    /// Returns [`FerroError::InsufficientSamples`] if any class has too few samples.
    pub fn split(&self, y: &Array1<usize>) -> Result<FoldSplits, FerroError> {
        if self.n_splits == 0 {
            return Err(FerroError::InvalidParameter {
                name: "n_splits".into(),
                reason: "must be >= 1".into(),
            });
        }
        if self.test_size <= 0.0 || self.test_size >= 1.0 {
            return Err(FerroError::InvalidParameter {
                name: "test_size".into(),
                reason: format!("must be in (0.0, 1.0), got {}", self.test_size),
            });
        }

        // Group sample indices by class label.
        let mut class_indices: HashMap<usize, Vec<usize>> = HashMap::new();
        for (i, &label) in y.iter().enumerate() {
            class_indices.entry(label).or_default().push(i);
        }

        // Sort classes for determinism.
        let mut classes: Vec<usize> = class_indices.keys().copied().collect();
        classes.sort_unstable();

        // Validate that each class has at least 2 samples (need at least 1 train + 1 test).
        for &class in &classes {
            if class_indices[&class].len() < 2 {
                return Err(FerroError::InsufficientSamples {
                    required: 2,
                    actual: class_indices[&class].len(),
                    context: format!(
                        "StratifiedShuffleSplit: class {class} needs at least 2 samples"
                    ),
                });
            }
        }

        let mut rng: SmallRng = match self.random_state {
            Some(seed) => SmallRng::seed_from_u64(seed),
            None => SmallRng::from_os_rng(),
        };

        let mut folds = Vec::with_capacity(self.n_splits);

        for _ in 0..self.n_splits {
            let mut test_indices = Vec::new();
            let mut train_indices = Vec::new();

            for &class in &classes {
                let indices = class_indices.get_mut(&class).unwrap();
                indices.shuffle(&mut rng);

                let n_class = indices.len();
                let n_test_class = (n_class as f64 * self.test_size).ceil() as usize;
                let n_test_class = n_test_class.max(1).min(n_class - 1);

                test_indices.extend_from_slice(&indices[..n_test_class]);
                train_indices.extend_from_slice(&indices[n_test_class..]);
            }

            folds.push((train_indices, test_indices));
        }

        Ok(folds)
    }
}

impl CrossValidator for StratifiedShuffleSplit {
    /// Returns an error advising to use [`split`](StratifiedShuffleSplit::split) with labels.
    fn fold_indices(&self, _n_samples: usize) -> Result<FoldSplits, FerroError> {
        Err(FerroError::InvalidParameter {
            name: "labels".into(),
            reason: "StratifiedShuffleSplit requires class labels; \
                     use the `split(y)` method instead of `fold_indices`"
                .into(),
        })
    }
}

// ---------------------------------------------------------------------------
// GroupKFold
// ---------------------------------------------------------------------------

/// K-fold cross-validation splitter that keeps groups together.
///
/// Each group appears entirely in either the training or test set for a given
/// fold. The number of distinct groups must be at least `n_splits`.
///
/// This splitter is deterministic (no shuffling) — the assignment of groups
/// to folds is based on the sorted order of unique group labels.
///
/// # Examples
///
/// ```rust
/// use ferrolearn_model_sel::GroupKFold;
/// use ndarray::array;
///
/// let groups = array![0usize, 0, 1, 1, 2, 2, 3, 3, 4, 4];
/// let gkf = GroupKFold::new(5);
/// let folds = gkf.split(&groups).unwrap();
/// assert_eq!(folds.len(), 5);
/// // Each fold's test set contains exactly one group (2 samples each).
/// for (train, test) in &folds {
///     assert_eq!(test.len(), 2);
///     assert_eq!(train.len(), 8);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct GroupKFold {
    /// Number of folds.
    n_splits: usize,
}

impl GroupKFold {
    /// Create a new [`GroupKFold`] with the given number of splits.
    pub fn new(n_splits: usize) -> Self {
        Self { n_splits }
    }

    /// Generate `(train_indices, test_indices)` pairs for each fold,
    /// ensuring that all samples from a group stay in the same fold.
    ///
    /// # Parameters
    ///
    /// - `groups` — An array of group labels of length `n_samples`. Samples
    ///   with the same label belong to the same group.
    ///
    /// # Errors
    ///
    /// - [`FerroError::InvalidParameter`] if `n_splits < 2`.
    /// - [`FerroError::InsufficientSamples`] if the number of unique groups
    ///   is less than `n_splits`.
    pub fn split(&self, groups: &Array1<usize>) -> Result<FoldSplits, FerroError> {
        if self.n_splits < 2 {
            return Err(FerroError::InvalidParameter {
                name: "n_splits".into(),
                reason: format!("must be >= 2, got {}", self.n_splits),
            });
        }

        let n_samples = groups.len();

        // Map each unique group to its sample indices.
        let mut group_indices: HashMap<usize, Vec<usize>> = HashMap::new();
        for (i, &g) in groups.iter().enumerate() {
            group_indices.entry(g).or_default().push(i);
        }

        let mut unique_groups: Vec<usize> = group_indices.keys().copied().collect();
        unique_groups.sort_unstable();
        let n_groups = unique_groups.len();

        if n_groups < self.n_splits {
            return Err(FerroError::InsufficientSamples {
                required: self.n_splits,
                actual: n_groups,
                context: format!(
                    "GroupKFold with n_splits={}: only {n_groups} unique groups",
                    self.n_splits
                ),
            });
        }

        // Assign groups to folds in round-robin fashion (sorted by group label).
        let mut fold_test_indices: Vec<Vec<usize>> = vec![Vec::new(); self.n_splits];
        for (i, &group) in unique_groups.iter().enumerate() {
            let fold_idx = i % self.n_splits;
            fold_test_indices[fold_idx].extend_from_slice(&group_indices[&group]);
        }

        // Build (train, test) pairs.
        let all_indices: Vec<usize> = (0..n_samples).collect();
        let mut folds = Vec::with_capacity(self.n_splits);
        for test in fold_test_indices {
            let test_set: std::collections::HashSet<usize> = test.iter().copied().collect();
            let train: Vec<usize> = all_indices
                .iter()
                .copied()
                .filter(|i| !test_set.contains(i))
                .collect();
            folds.push((train, test));
        }

        Ok(folds)
    }
}

impl CrossValidator for GroupKFold {
    /// Returns an error advising to use [`split`](GroupKFold::split) with group labels.
    fn fold_indices(&self, _n_samples: usize) -> Result<FoldSplits, FerroError> {
        Err(FerroError::InvalidParameter {
            name: "groups".into(),
            reason: "GroupKFold requires group labels; \
                     use the `split(groups)` method instead of `fold_indices`"
                .into(),
        })
    }
}

// ---------------------------------------------------------------------------
// cross_val_score
// ---------------------------------------------------------------------------

/// Evaluate a [`Pipeline`] using cross-validation and return per-fold scores.
///
/// For each fold produced by `cv`:
///
/// 1. Extract training and test subsets of `x` and `y`.
/// 2. Fit the pipeline on the training subset.
/// 3. Predict on the test subset.
/// 4. Compute the score using the `scoring` function.
///
/// The pipeline is cloned conceptually by fitting a fresh copy for each fold;
/// the original `pipeline` is not mutated.
///
/// # Parameters
///
/// - `pipeline` — An unfitted [`Pipeline`] to evaluate.
/// - `x` — Feature matrix with shape `(n_samples, n_features)`.
/// - `y` — Target array of length `n_samples`.
/// - `cv` — A [`CrossValidator`] (e.g. [`KFold`]) that produces fold indices.
/// - `scoring` — A function `(y_true, y_pred) -> Result<f64, FerroError>`
///   used to score each fold.
///
/// # Returns
///
/// An [`Array1<f64>`] of length `n_folds` containing the score for each fold.
///
/// # Errors
///
/// Propagates any error from fold splitting, model fitting, predicting, or
/// scoring.
///
/// # Examples
///
/// ```rust,no_run
/// use ferrolearn_model_sel::{KFold, cross_val_score};
/// use ferrolearn_core::pipeline::Pipeline;
/// use ferrolearn_core::FerroError;
/// use ndarray::{Array1, Array2};
///
/// fn mse(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64, FerroError> {
///     let diff = y_true - y_pred;
///     Ok(diff.mapv(|v| v * v).mean().unwrap_or(0.0))
/// }
///
/// // pipeline must have an estimator step set before calling cross_val_score.
/// // let scores = cross_val_score(&pipeline, &x, &y, &KFold::new(5), mse).unwrap();
/// ```
pub fn cross_val_score(
    pipeline: &Pipeline,
    x: &Array2<f64>,
    y: &Array1<f64>,
    cv: &dyn CrossValidator,
    scoring: fn(&Array1<f64>, &Array1<f64>) -> Result<f64, FerroError>,
) -> Result<Array1<f64>, FerroError> {
    let n_samples = x.nrows();

    if y.len() != n_samples {
        return Err(FerroError::ShapeMismatch {
            expected: vec![n_samples],
            actual: vec![y.len()],
            context: "cross_val_score: y length must equal x number of rows".into(),
        });
    }

    let folds = cv.fold_indices(n_samples)?;
    let mut scores = Vec::with_capacity(folds.len());

    for (train_idx, test_idx) in &folds {
        // Build training subset.
        let n_train = train_idx.len();
        let n_test = test_idx.len();
        let n_features = x.ncols();

        let mut x_train_data = Vec::with_capacity(n_train * n_features);
        for &i in train_idx {
            x_train_data.extend(x.row(i).iter().copied());
        }
        let x_train = Array2::from_shape_vec((n_train, n_features), x_train_data).map_err(|e| {
            FerroError::InvalidParameter {
                name: "x_train".into(),
                reason: e.to_string(),
            }
        })?;
        let y_train: Array1<f64> = train_idx.iter().map(|&i| y[i]).collect();

        // Build test subset.
        let mut x_test_data = Vec::with_capacity(n_test * n_features);
        for &i in test_idx {
            x_test_data.extend(x.row(i).iter().copied());
        }
        let x_test = Array2::from_shape_vec((n_test, n_features), x_test_data).map_err(|e| {
            FerroError::InvalidParameter {
                name: "x_test".into(),
                reason: e.to_string(),
            }
        })?;
        let y_test: Array1<f64> = test_idx.iter().map(|&i| y[i]).collect();

        // Fit pipeline on training data.
        let fitted = pipeline.fit(&x_train, &y_train)?;

        // Predict on test data.
        let y_pred = fitted.predict(&x_test)?;

        // Score.
        let score = scoring(&y_test, &y_pred)?;
        scores.push(score);
    }

    Ok(Array1::from_vec(scores))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ferrolearn_core::pipeline::{
        FittedPipelineEstimator as FittedEstTrait, FittedPipelineTransformer, Pipeline,
        PipelineEstimator, PipelineTransformer,
    };
    use ndarray::{Array1, Array2, array};

    // -- KFold tests ---------------------------------------------------------

    #[test]
    fn test_kfold_basic() {
        let kf = KFold::new(5);
        let folds = kf.split(20);
        assert_eq!(folds.len(), 5);
        for (train, test) in &folds {
            assert_eq!(train.len() + test.len(), 20);
        }
    }

    #[test]
    fn test_kfold_fold_sizes_equal() {
        let folds = KFold::new(4).split(20);
        for (_, test) in &folds {
            assert_eq!(test.len(), 5);
        }
    }

    #[test]
    fn test_kfold_fold_sizes_unequal() {
        // 10 samples, 3 folds → sizes 4, 3, 3.
        let folds = KFold::new(3).split(10);
        let test_sizes: Vec<usize> = folds.iter().map(|(_, t)| t.len()).collect();
        assert_eq!(test_sizes.iter().sum::<usize>(), 10);
        // Max and min sizes differ by at most 1.
        let max = *test_sizes.iter().max().unwrap();
        let min = *test_sizes.iter().min().unwrap();
        assert!(max - min <= 1);
    }

    #[test]
    fn test_kfold_no_overlap_full_coverage() {
        let folds = KFold::new(5).split(10);
        let mut all_test: Vec<usize> = folds.iter().flat_map(|(_, t)| t.iter().copied()).collect();
        all_test.sort_unstable();
        let expected: Vec<usize> = (0..10).collect();
        assert_eq!(all_test, expected);
    }

    #[test]
    fn test_kfold_shuffle_deterministic() {
        let kf = KFold::new(5).shuffle(true).random_state(42);
        let folds1 = kf.split(20);
        let folds2 = kf.split(20);
        assert_eq!(folds1, folds2);
    }

    #[test]
    fn test_kfold_shuffle_differs_from_no_shuffle() {
        let folds_no = KFold::new(5).split(20);
        let folds_sh = KFold::new(5).shuffle(true).random_state(1).split(20);
        // At least one fold should differ (extremely unlikely not to).
        let different = folds_no.iter().zip(folds_sh.iter()).any(|(a, b)| a != b);
        assert!(different);
    }

    #[test]
    fn test_kfold_invalid_n_splits() {
        let kf = KFold::new(1);
        let result = kf.fold_indices(10);
        assert!(result.is_err());
    }

    #[test]
    fn test_kfold_insufficient_samples() {
        let kf = KFold::new(5);
        let result = kf.fold_indices(3);
        assert!(result.is_err());
    }

    // -- StratifiedKFold tests -----------------------------------------------

    #[test]
    fn test_skfold_basic() {
        // 3 classes, 3 samples each, 3 folds → each fold test has 1 of each.
        let y: Array1<usize> = Array1::from_iter((0..9).map(|i| i % 3));
        let skf = StratifiedKFold::new(3);
        let folds = skf.split(&y).unwrap();
        assert_eq!(folds.len(), 3);
        for (train, test) in &folds {
            assert_eq!(train.len() + test.len(), 9);
        }
    }

    #[test]
    fn test_skfold_coverage() {
        let y: Array1<usize> = Array1::from_iter((0..12).map(|i| i % 3));
        let skf = StratifiedKFold::new(3);
        let folds = skf.split(&y).unwrap();

        let mut all_test: Vec<usize> = folds.iter().flat_map(|(_, t)| t.iter().copied()).collect();
        all_test.sort_unstable();
        let expected: Vec<usize> = (0..12).collect();
        assert_eq!(all_test, expected);
    }

    #[test]
    fn test_skfold_class_balance() {
        // 6 samples: 3 class-0 and 3 class-1.
        let y = array![0usize, 0, 0, 1, 1, 1];
        let skf = StratifiedKFold::new(3);
        let folds = skf.split(&y).unwrap();
        // Each fold's test should have 1 from each class.
        for (_, test) in &folds {
            let class0 = test.iter().filter(|&&i| y[i] == 0).count();
            let class1 = test.iter().filter(|&&i| y[i] == 1).count();
            assert_eq!(class0, 1, "expected 1 class-0 sample per test fold");
            assert_eq!(class1, 1, "expected 1 class-1 sample per test fold");
        }
    }

    #[test]
    fn test_skfold_shuffle_deterministic() {
        let y: Array1<usize> = Array1::from_iter((0..12).map(|i| i % 3));
        let skf = StratifiedKFold::new(3).shuffle(true).random_state(99);
        let folds1 = skf.split(&y).unwrap();
        let folds2 = skf.split(&y).unwrap();
        assert_eq!(folds1, folds2);
    }

    #[test]
    fn test_skfold_invalid_n_splits() {
        let y = array![0usize, 1, 2];
        let skf = StratifiedKFold::new(1);
        assert!(skf.split(&y).is_err());
    }

    #[test]
    fn test_skfold_class_too_small() {
        // Class 2 has only 1 sample but n_splits=3.
        let y = array![0usize, 0, 0, 1, 1, 1, 2];
        let skf = StratifiedKFold::new(3);
        assert!(skf.split(&y).is_err());
    }

    // -- cross_val_score tests ------------------------------------------------

    /// A trivial pipeline estimator that always predicts the mean of y_train.
    struct MeanEstimator;

    impl PipelineEstimator<f64> for MeanEstimator {
        fn fit_pipeline(
            &self,
            _x: &Array2<f64>,
            y: &Array1<f64>,
        ) -> Result<Box<dyn FittedEstTrait<f64>>, FerroError> {
            let mean = y.mean().unwrap_or(0.0);
            Ok(Box::new(FittedMean { mean }))
        }
    }

    struct FittedMean {
        mean: f64,
    }

    impl FittedEstTrait<f64> for FittedMean {
        fn predict_pipeline(&self, x: &Array2<f64>) -> Result<Array1<f64>, FerroError> {
            Ok(Array1::from_elem(x.nrows(), self.mean))
        }
    }

    /// Identity transformer (pass-through).
    struct IdentityTransformer;

    impl PipelineTransformer<f64> for IdentityTransformer {
        fn fit_pipeline(
            &self,
            _x: &Array2<f64>,
            _y: &Array1<f64>,
        ) -> Result<Box<dyn FittedPipelineTransformer<f64>>, FerroError> {
            Ok(Box::new(FittedIdentity))
        }
    }

    struct FittedIdentity;

    impl FittedPipelineTransformer<f64> for FittedIdentity {
        fn transform_pipeline(&self, x: &Array2<f64>) -> Result<Array2<f64>, FerroError> {
            Ok(x.clone())
        }
    }

    fn mse(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64, FerroError> {
        let diff = y_true - y_pred;
        Ok(diff.mapv(|v| v * v).mean().unwrap_or(0.0))
    }

    #[test]
    fn test_cross_val_score_returns_correct_number_of_scores() {
        let pipeline = Pipeline::new().estimator_step("mean", Box::new(MeanEstimator));
        let x = Array2::<f64>::zeros((20, 3));
        let y = Array1::<f64>::zeros(20);
        let kf = KFold::new(5);
        let scores = cross_val_score(&pipeline, &x, &y, &kf, mse).unwrap();
        assert_eq!(scores.len(), 5);
    }

    #[test]
    fn test_cross_val_score_perfect_constant_target() {
        // When y is constant and the estimator predicts the mean, MSE = 0.
        let pipeline = Pipeline::new().estimator_step("mean", Box::new(MeanEstimator));
        let x = Array2::<f64>::zeros((20, 3));
        let y = Array1::<f64>::from_elem(20, 5.0);
        let kf = KFold::new(5);
        let scores = cross_val_score(&pipeline, &x, &y, &kf, mse).unwrap();
        for &s in scores.iter() {
            assert!(s.abs() < 1e-10, "expected 0 MSE, got {s}");
        }
    }

    #[test]
    fn test_cross_val_score_with_transformer() {
        let pipeline = Pipeline::new()
            .transform_step("identity", Box::new(IdentityTransformer))
            .estimator_step("mean", Box::new(MeanEstimator));
        let x = Array2::<f64>::zeros((15, 2));
        let y = Array1::<f64>::from_elem(15, 3.0);
        let kf = KFold::new(3);
        let scores = cross_val_score(&pipeline, &x, &y, &kf, mse).unwrap();
        assert_eq!(scores.len(), 3);
        for &s in scores.iter() {
            assert!(s.abs() < 1e-10);
        }
    }

    #[test]
    fn test_cross_val_score_shape_mismatch() {
        let pipeline = Pipeline::new().estimator_step("mean", Box::new(MeanEstimator));
        let x = Array2::<f64>::zeros((20, 3));
        let y = Array1::<f64>::zeros(18); // wrong length
        let kf = KFold::new(5);
        assert!(cross_val_score(&pipeline, &x, &y, &kf, mse).is_err());
    }

    // -- LeaveOneOut tests ----------------------------------------------------

    #[test]
    fn test_loo_basic() {
        let loo = LeaveOneOut;
        let folds = loo.fold_indices(5).unwrap();
        assert_eq!(folds.len(), 5);
        for (train, test) in &folds {
            assert_eq!(test.len(), 1);
            assert_eq!(train.len(), 4);
        }
    }

    #[test]
    fn test_loo_full_coverage() {
        let folds = LeaveOneOut.fold_indices(8).unwrap();
        let mut all_test: Vec<usize> = folds.iter().flat_map(|(_, t)| t.iter().copied()).collect();
        all_test.sort_unstable();
        let expected: Vec<usize> = (0..8).collect();
        assert_eq!(all_test, expected);
    }

    #[test]
    fn test_loo_no_overlap_in_train_test() {
        let folds = LeaveOneOut.fold_indices(6).unwrap();
        for (train, test) in &folds {
            assert!(!train.contains(&test[0]));
        }
    }

    #[test]
    fn test_loo_insufficient_samples() {
        assert!(LeaveOneOut.fold_indices(1).is_err());
        assert!(LeaveOneOut.fold_indices(0).is_err());
    }

    // -- LeavePOut tests ------------------------------------------------------

    #[test]
    fn test_lpo_basic() {
        let lpo = LeavePOut::new(2);
        let folds = lpo.fold_indices(4).unwrap();
        // C(4, 2) = 6
        assert_eq!(folds.len(), 6);
        for (train, test) in &folds {
            assert_eq!(test.len(), 2);
            assert_eq!(train.len(), 2);
        }
    }

    #[test]
    fn test_lpo_p_equals_1_matches_loo() {
        let lpo_folds = LeavePOut::new(1).fold_indices(5).unwrap();
        let loo_folds = LeaveOneOut.fold_indices(5).unwrap();
        assert_eq!(lpo_folds.len(), loo_folds.len());
        for (lpo, loo) in lpo_folds.iter().zip(loo_folds.iter()) {
            assert_eq!(lpo.0, loo.0);
            assert_eq!(lpo.1, loo.1);
        }
    }

    #[test]
    fn test_lpo_combinations_count() {
        // C(5, 3) = 10
        let folds = LeavePOut::new(3).fold_indices(5).unwrap();
        assert_eq!(folds.len(), 10);
    }

    #[test]
    fn test_lpo_no_duplicate_combinations() {
        let folds = LeavePOut::new(2).fold_indices(5).unwrap();
        let test_sets: Vec<Vec<usize>> = folds
            .iter()
            .map(|(_, t)| {
                let mut s = t.clone();
                s.sort_unstable();
                s
            })
            .collect();
        // Check all pairs are distinct.
        for i in 0..test_sets.len() {
            for j in (i + 1)..test_sets.len() {
                assert_ne!(test_sets[i], test_sets[j]);
            }
        }
    }

    #[test]
    fn test_lpo_invalid_p_zero() {
        assert!(LeavePOut::new(0).fold_indices(5).is_err());
    }

    #[test]
    fn test_lpo_p_too_large() {
        assert!(LeavePOut::new(5).fold_indices(5).is_err());
        assert!(LeavePOut::new(6).fold_indices(5).is_err());
    }

    // -- RepeatedKFold tests --------------------------------------------------

    #[test]
    fn test_repeated_kfold_basic() {
        let rkf = RepeatedKFold::new(5, 3).random_state(42);
        let folds = rkf.fold_indices(20).unwrap();
        assert_eq!(folds.len(), 15); // 5 * 3
        for (train, test) in &folds {
            assert_eq!(train.len() + test.len(), 20);
        }
    }

    #[test]
    fn test_repeated_kfold_deterministic() {
        let rkf = RepeatedKFold::new(3, 2).random_state(99);
        let folds1 = rkf.fold_indices(12).unwrap();
        let folds2 = rkf.fold_indices(12).unwrap();
        assert_eq!(folds1, folds2);
    }

    #[test]
    fn test_repeated_kfold_repeats_differ() {
        let rkf = RepeatedKFold::new(3, 2).random_state(42);
        let folds = rkf.fold_indices(12).unwrap();
        // First repeat folds (0..3) should differ from second repeat folds (3..6).
        let first_repeat: Vec<_> = folds[0..3].to_vec();
        let second_repeat: Vec<_> = folds[3..6].to_vec();
        assert_ne!(first_repeat, second_repeat);
    }

    #[test]
    fn test_repeated_kfold_each_repeat_covers_all() {
        let rkf = RepeatedKFold::new(3, 2).random_state(42);
        let folds = rkf.fold_indices(9).unwrap();
        // Check that each repeat's test sets cover all indices.
        for repeat_start in (0..6).step_by(3) {
            let mut all_test: Vec<usize> = folds[repeat_start..repeat_start + 3]
                .iter()
                .flat_map(|(_, t)| t.iter().copied())
                .collect();
            all_test.sort_unstable();
            let expected: Vec<usize> = (0..9).collect();
            assert_eq!(all_test, expected);
        }
    }

    #[test]
    fn test_repeated_kfold_zero_repeats() {
        let rkf = RepeatedKFold::new(3, 0).random_state(42);
        assert!(rkf.fold_indices(9).is_err());
    }

    #[test]
    fn test_repeated_kfold_invalid_splits() {
        let rkf = RepeatedKFold::new(1, 3).random_state(42);
        assert!(rkf.fold_indices(9).is_err());
    }

    // -- RepeatedStratifiedKFold tests ----------------------------------------

    #[test]
    fn test_repeated_skfold_basic() {
        let y: Array1<usize> = Array1::from_iter((0..12).map(|i| i % 3));
        let rskf = RepeatedStratifiedKFold::new(3, 2).random_state(42);
        let folds = rskf.split(&y).unwrap();
        assert_eq!(folds.len(), 6); // 3 * 2
        for (train, test) in &folds {
            assert_eq!(train.len() + test.len(), 12);
        }
    }

    #[test]
    fn test_repeated_skfold_deterministic() {
        let y: Array1<usize> = Array1::from_iter((0..12).map(|i| i % 3));
        let rskf = RepeatedStratifiedKFold::new(3, 2).random_state(42);
        let folds1 = rskf.split(&y).unwrap();
        let folds2 = rskf.split(&y).unwrap();
        assert_eq!(folds1, folds2);
    }

    #[test]
    fn test_repeated_skfold_class_balance() {
        let y = array![0usize, 0, 0, 0, 1, 1, 1, 1];
        let rskf = RepeatedStratifiedKFold::new(2, 3).random_state(42);
        let folds = rskf.split(&y).unwrap();
        // Each fold should have balanced classes in test set.
        for (_, test) in &folds {
            let class0 = test.iter().filter(|&&i| y[i] == 0).count();
            let class1 = test.iter().filter(|&&i| y[i] == 1).count();
            assert_eq!(class0, class1, "test fold should have equal class counts");
        }
    }

    #[test]
    fn test_repeated_skfold_fold_indices_returns_error() {
        let rskf = RepeatedStratifiedKFold::new(3, 2);
        assert!(rskf.fold_indices(12).is_err());
    }

    #[test]
    fn test_repeated_skfold_zero_repeats() {
        let y: Array1<usize> = Array1::from_iter((0..12).map(|i| i % 3));
        let rskf = RepeatedStratifiedKFold::new(3, 0).random_state(42);
        assert!(rskf.split(&y).is_err());
    }

    // -- ShuffleSplit tests ---------------------------------------------------

    #[test]
    fn test_shuffle_split_basic() {
        let ss = ShuffleSplit::new(5).test_size(0.25).random_state(42);
        let folds = ss.fold_indices(100).unwrap();
        assert_eq!(folds.len(), 5);
        for (train, test) in &folds {
            assert_eq!(test.len(), 25);
            assert_eq!(train.len(), 75);
        }
    }

    #[test]
    fn test_shuffle_split_deterministic() {
        let ss = ShuffleSplit::new(3).test_size(0.2).random_state(42);
        let folds1 = ss.fold_indices(50).unwrap();
        let folds2 = ss.fold_indices(50).unwrap();
        assert_eq!(folds1, folds2);
    }

    #[test]
    fn test_shuffle_split_no_overlap_within_fold() {
        let ss = ShuffleSplit::new(3).test_size(0.3).random_state(42);
        let folds = ss.fold_indices(20).unwrap();
        for (train, test) in &folds {
            let train_set: std::collections::HashSet<usize> = train.iter().copied().collect();
            let test_set: std::collections::HashSet<usize> = test.iter().copied().collect();
            // No overlap within a fold.
            assert_eq!(train_set.intersection(&test_set).count(), 0);
            // Union covers all indices.
            assert_eq!(train_set.len() + test_set.len(), 20);
        }
    }

    #[test]
    fn test_shuffle_split_default_test_size() {
        // Default is 0.1.
        let ss = ShuffleSplit::new(2).random_state(42);
        let folds = ss.fold_indices(100).unwrap();
        for (_, test) in &folds {
            assert_eq!(test.len(), 10);
        }
    }

    #[test]
    fn test_shuffle_split_invalid_test_size() {
        let ss = ShuffleSplit::new(2).test_size(0.0);
        assert!(ss.fold_indices(10).is_err());
        let ss = ShuffleSplit::new(2).test_size(1.0);
        assert!(ss.fold_indices(10).is_err());
        let ss = ShuffleSplit::new(2).test_size(-0.1);
        assert!(ss.fold_indices(10).is_err());
    }

    #[test]
    fn test_shuffle_split_zero_splits() {
        let ss = ShuffleSplit::new(0).test_size(0.2);
        assert!(ss.fold_indices(10).is_err());
    }

    // -- StratifiedShuffleSplit tests -----------------------------------------

    #[test]
    fn test_stratified_shuffle_split_basic() {
        let y: Array1<usize> = Array1::from_iter((0..100).map(|i| i % 2));
        let sss = StratifiedShuffleSplit::new(3)
            .test_size(0.2)
            .random_state(42);
        let folds = sss.split(&y).unwrap();
        assert_eq!(folds.len(), 3);
        for (train, test) in &folds {
            // Total should be 100.
            assert_eq!(train.len() + test.len(), 100);
        }
    }

    #[test]
    fn test_stratified_shuffle_split_preserves_proportion() {
        // 70 class-0, 30 class-1.
        let mut labels = vec![0usize; 70];
        labels.extend(vec![1usize; 30]);
        let y = Array1::from_vec(labels);
        let sss = StratifiedShuffleSplit::new(5)
            .test_size(0.2)
            .random_state(42);
        let folds = sss.split(&y).unwrap();
        for (_, test) in &folds {
            let class0 = test.iter().filter(|&&i| y[i] == 0).count();
            let class1 = test.iter().filter(|&&i| y[i] == 1).count();
            // Expect roughly 70:30 ratio in test set.
            // With 20% test size: ~14 class-0, ~6 class-1.
            assert!(
                class0 > 0 && class1 > 0,
                "both classes should appear in test set"
            );
            let ratio = class0 as f64 / (class0 + class1) as f64;
            assert!(
                (ratio - 0.7).abs() < 0.15,
                "class proportion should be roughly preserved, got {ratio}"
            );
        }
    }

    #[test]
    fn test_stratified_shuffle_split_deterministic() {
        let y: Array1<usize> = Array1::from_iter((0..50).map(|i| i % 3));
        let sss = StratifiedShuffleSplit::new(3)
            .test_size(0.3)
            .random_state(42);
        let folds1 = sss.split(&y).unwrap();
        let folds2 = sss.split(&y).unwrap();
        assert_eq!(folds1, folds2);
    }

    #[test]
    fn test_stratified_shuffle_split_no_overlap_within_fold() {
        let y: Array1<usize> = Array1::from_iter((0..40).map(|i| i % 2));
        let sss = StratifiedShuffleSplit::new(3)
            .test_size(0.25)
            .random_state(42);
        let folds = sss.split(&y).unwrap();
        for (train, test) in &folds {
            let train_set: std::collections::HashSet<usize> = train.iter().copied().collect();
            let test_set: std::collections::HashSet<usize> = test.iter().copied().collect();
            assert_eq!(train_set.intersection(&test_set).count(), 0);
            assert_eq!(train_set.len() + test_set.len(), 40);
        }
    }

    #[test]
    fn test_stratified_shuffle_split_fold_indices_returns_error() {
        let sss = StratifiedShuffleSplit::new(3).test_size(0.2);
        assert!(sss.fold_indices(100).is_err());
    }

    #[test]
    fn test_stratified_shuffle_split_invalid_test_size() {
        let y: Array1<usize> = Array1::from_iter((0..20).map(|i| i % 2));
        let sss = StratifiedShuffleSplit::new(3).test_size(0.0);
        assert!(sss.split(&y).is_err());
        let sss = StratifiedShuffleSplit::new(3).test_size(1.0);
        assert!(sss.split(&y).is_err());
    }

    // -- GroupKFold tests -----------------------------------------------------

    #[test]
    fn test_group_kfold_basic() {
        let groups = array![0usize, 0, 1, 1, 2, 2, 3, 3, 4, 4];
        let gkf = GroupKFold::new(5);
        let folds = gkf.split(&groups).unwrap();
        assert_eq!(folds.len(), 5);
        for (train, test) in &folds {
            assert_eq!(train.len() + test.len(), 10);
        }
    }

    #[test]
    fn test_group_kfold_groups_not_split() {
        let groups = array![0usize, 0, 0, 1, 1, 2, 2, 2, 3, 3];
        let gkf = GroupKFold::new(2);
        let folds = gkf.split(&groups).unwrap();
        for (_, test) in &folds {
            // All samples in test should have the same set of groups.
            let test_groups: std::collections::HashSet<usize> =
                test.iter().map(|&i| groups[i]).collect();
            // For each group in the test set, all samples of that group should be
            // in the test set.
            for &g in &test_groups {
                let group_samples: Vec<usize> =
                    (0..groups.len()).filter(|&i| groups[i] == g).collect();
                for s in &group_samples {
                    assert!(
                        test.contains(s),
                        "sample {s} of group {g} should be in test set"
                    );
                }
            }
        }
    }

    #[test]
    fn test_group_kfold_deterministic() {
        let groups = array![0usize, 0, 1, 1, 2, 2, 3, 3];
        let gkf = GroupKFold::new(2);
        let folds1 = gkf.split(&groups).unwrap();
        let folds2 = gkf.split(&groups).unwrap();
        assert_eq!(folds1, folds2);
    }

    #[test]
    fn test_group_kfold_full_coverage() {
        let groups = array![0usize, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5];
        let gkf = GroupKFold::new(3);
        let folds = gkf.split(&groups).unwrap();
        let mut all_test: Vec<usize> = folds.iter().flat_map(|(_, t)| t.iter().copied()).collect();
        all_test.sort_unstable();
        let expected: Vec<usize> = (0..12).collect();
        assert_eq!(all_test, expected);
    }

    #[test]
    fn test_group_kfold_invalid_n_splits() {
        let groups = array![0usize, 1, 2, 3];
        let gkf = GroupKFold::new(1);
        assert!(gkf.split(&groups).is_err());
    }

    #[test]
    fn test_group_kfold_too_few_groups() {
        let groups = array![0usize, 0, 1, 1];
        let gkf = GroupKFold::new(3);
        assert!(gkf.split(&groups).is_err());
    }

    #[test]
    fn test_group_kfold_fold_indices_returns_error() {
        let gkf = GroupKFold::new(3);
        assert!(gkf.fold_indices(10).is_err());
    }

    #[test]
    fn test_group_kfold_unequal_group_sizes() {
        // Group 0: 5 samples, Group 1: 1 sample, Group 2: 3 samples, Group 3: 1 sample.
        let groups = array![0usize, 0, 0, 0, 0, 1, 2, 2, 2, 3];
        let gkf = GroupKFold::new(2);
        let folds = gkf.split(&groups).unwrap();
        assert_eq!(folds.len(), 2);
        for (train, test) in &folds {
            assert_eq!(train.len() + test.len(), 10);
        }
    }
}
