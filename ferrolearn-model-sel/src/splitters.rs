//! Additional cross-validation splitters that complement [`KFold`] and
//! [`StratifiedKFold`].
//!
//! Every splitter implements the [`CrossValidator`] trait so they can be
//! plugged into [`cross_val_score`] / [`cross_validate`] / [`cross_val_predict`].
//!
//! - [`LeaveOneOut`] — leave a single sample out as the test fold (`n` folds total).
//! - [`LeavePOut`] — leave every `p`-sized subset out (`C(n, p)` folds; small `p` only).
//! - [`ShuffleSplit`] — random train/test splits, `n_splits` repetitions.
//! - [`StratifiedShuffleSplit`] — class-balanced random train/test splits.
//! - [`RepeatedKFold`] — `KFold` repeated `n_repeats` times with different shuffles.
//! - [`RepeatedStratifiedKFold`] — `StratifiedKFold` repeated `n_repeats` times.
//! - [`PredefinedSplit`] — splits driven by a user-supplied `test_fold` array.
//!
//! [`KFold`]: crate::cross_validation::KFold
//! [`StratifiedKFold`]: crate::cross_validation::StratifiedKFold
//! [`cross_val_score`]: crate::cross_validation::cross_val_score
//! [`cross_validate`]: crate::cross_validation::cross_validate
//! [`cross_val_predict`]: crate::cross_validation::cross_val_predict

use std::collections::HashMap;

use crate::cross_validation::{CrossValidator, FoldSplit, FoldSplits};
use ferrolearn_core::FerroError;
use ndarray::Array1;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;

/// Leave-One-Out cross-validator.
///
/// Each sample becomes its own test fold once. There are `n_samples` folds
/// in total.
#[derive(Debug, Clone, Default)]
pub struct LeaveOneOut;

impl LeaveOneOut {
    /// Construct a new [`LeaveOneOut`] splitter.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Number of folds for `n_samples` (equals `n_samples`).
    #[must_use]
    pub fn get_n_splits(&self, n_samples: usize) -> usize {
        n_samples
    }
}

impl CrossValidator for LeaveOneOut {
    fn fold_indices(&self, n_samples: usize) -> Result<FoldSplits, FerroError> {
        if n_samples < 2 {
            return Err(FerroError::InsufficientSamples {
                required: 2,
                actual: n_samples,
                context: "LeaveOneOut".into(),
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

/// Leave-P-Out cross-validator.
///
/// Generates one fold for every size-`p` subset of samples used as the test
/// set. The total number of folds is `C(n_samples, p)`, so this is only
/// practical for small `p`.
#[derive(Debug, Clone)]
pub struct LeavePOut {
    p: usize,
}

impl LeavePOut {
    /// Construct a new [`LeavePOut`] splitter with the given `p`.
    #[must_use]
    pub fn new(p: usize) -> Self {
        Self { p }
    }
}

impl CrossValidator for LeavePOut {
    fn fold_indices(&self, n_samples: usize) -> Result<FoldSplits, FerroError> {
        if self.p == 0 {
            return Err(FerroError::InvalidParameter {
                name: "p".into(),
                reason: "LeavePOut: p must be >= 1".into(),
            });
        }
        if n_samples <= self.p {
            return Err(FerroError::InsufficientSamples {
                required: self.p + 1,
                actual: n_samples,
                context: format!("LeavePOut p={}", self.p),
            });
        }
        let mut folds = Vec::new();
        let mut combo = (0..self.p).collect::<Vec<usize>>();
        loop {
            let test: Vec<usize> = combo.clone();
            let test_set: std::collections::HashSet<usize> = test.iter().copied().collect();
            let train: Vec<usize> = (0..n_samples).filter(|i| !test_set.contains(i)).collect();
            folds.push((train, test));

            // Advance to next combination in lexicographic order.
            let mut i = self.p;
            while i > 0 {
                i -= 1;
                if combo[i] < n_samples - self.p + i {
                    combo[i] += 1;
                    for j in (i + 1)..self.p {
                        combo[j] = combo[j - 1] + 1;
                    }
                    break;
                }
                if i == 0 {
                    return Ok(folds);
                }
            }
        }
    }
}

/// Random train/test shuffle splitter.
///
/// Generates `n_splits` independent random train/test splits. Unlike
/// [`KFold`](crate::cross_validation::KFold), test sets may overlap.
#[derive(Debug, Clone)]
pub struct ShuffleSplit {
    n_splits: usize,
    test_size: f64,
    random_state: Option<u64>,
}

impl ShuffleSplit {
    /// Construct a new [`ShuffleSplit`] with the given number of splits and
    /// test fraction in `(0, 1)`.
    #[must_use]
    pub fn new(n_splits: usize, test_size: f64) -> Self {
        Self {
            n_splits,
            test_size,
            random_state: None,
        }
    }

    /// Set the RNG seed for reproducible shuffling.
    #[must_use]
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

impl CrossValidator for ShuffleSplit {
    fn fold_indices(&self, n_samples: usize) -> Result<FoldSplits, FerroError> {
        if !(0.0 < self.test_size && self.test_size < 1.0) {
            return Err(FerroError::InvalidParameter {
                name: "test_size".into(),
                reason: format!(
                    "ShuffleSplit: test_size must be in (0, 1), got {}",
                    self.test_size
                ),
            });
        }
        if n_samples < 2 {
            return Err(FerroError::InsufficientSamples {
                required: 2,
                actual: n_samples,
                context: "ShuffleSplit".into(),
            });
        }
        let n_test = ((n_samples as f64) * self.test_size).round().max(1.0) as usize;
        let n_test = n_test.min(n_samples - 1);
        let mut folds = Vec::with_capacity(self.n_splits);
        for split in 0..self.n_splits {
            let mut indices: Vec<usize> = (0..n_samples).collect();
            let mut rng = match self.random_state {
                Some(seed) => SmallRng::seed_from_u64(seed.wrapping_add(split as u64)),
                None => SmallRng::from_os_rng(),
            };
            indices.shuffle(&mut rng);
            let test: Vec<usize> = indices[..n_test].to_vec();
            let train: Vec<usize> = indices[n_test..].to_vec();
            folds.push((train, test));
        }
        Ok(folds)
    }
}

/// Stratified random train/test shuffle splitter.
///
/// Like [`ShuffleSplit`] but preserves class proportions in each split.
/// Use [`StratifiedShuffleSplit::split`] which takes class labels.
#[derive(Debug, Clone)]
pub struct StratifiedShuffleSplit {
    n_splits: usize,
    test_size: f64,
    random_state: Option<u64>,
}

impl StratifiedShuffleSplit {
    /// Construct a new [`StratifiedShuffleSplit`].
    #[must_use]
    pub fn new(n_splits: usize, test_size: f64) -> Self {
        Self {
            n_splits,
            test_size,
            random_state: None,
        }
    }

    /// Set the RNG seed for reproducible shuffling.
    #[must_use]
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Generate the train/test splits for class labels `y`.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] if `test_size` is not in
    /// `(0, 1)`.
    /// Returns [`FerroError::InsufficientSamples`] if any class has fewer
    /// than 2 samples (cannot stratify).
    pub fn split(&self, y: &Array1<usize>) -> Result<FoldSplits, FerroError> {
        let n = y.len();
        if !(0.0 < self.test_size && self.test_size < 1.0) {
            return Err(FerroError::InvalidParameter {
                name: "test_size".into(),
                reason: format!(
                    "StratifiedShuffleSplit: test_size must be in (0, 1), got {}",
                    self.test_size
                ),
            });
        }
        if n < 2 {
            return Err(FerroError::InsufficientSamples {
                required: 2,
                actual: n,
                context: "StratifiedShuffleSplit".into(),
            });
        }
        // Group indices by class.
        let mut by_class: HashMap<usize, Vec<usize>> = HashMap::new();
        for (i, &c) in y.iter().enumerate() {
            by_class.entry(c).or_default().push(i);
        }
        for (cls, samples) in &by_class {
            if samples.len() < 2 {
                return Err(FerroError::InsufficientSamples {
                    required: 2,
                    actual: samples.len(),
                    context: format!("StratifiedShuffleSplit: class {cls} has too few samples"),
                });
            }
        }

        let mut folds = Vec::with_capacity(self.n_splits);
        for split in 0..self.n_splits {
            let mut rng = match self.random_state {
                Some(seed) => SmallRng::seed_from_u64(seed.wrapping_add(split as u64)),
                None => SmallRng::from_os_rng(),
            };
            let mut test = Vec::new();
            let mut train = Vec::new();
            for samples in by_class.values() {
                let mut idx = samples.clone();
                idx.shuffle(&mut rng);
                let n_class_test = ((idx.len() as f64) * self.test_size).round().max(1.0) as usize;
                let n_class_test = n_class_test.min(idx.len() - 1);
                test.extend_from_slice(&idx[..n_class_test]);
                train.extend_from_slice(&idx[n_class_test..]);
            }
            train.sort_unstable();
            test.sort_unstable();
            folds.push((train, test));
        }
        Ok(folds)
    }
}

/// Repeated K-Fold cross-validator.
///
/// Repeats [`KFold`](crate::cross_validation::KFold) `n_repeats` times with a
/// different shuffle each time, yielding `n_splits * n_repeats` folds.
#[derive(Debug, Clone)]
pub struct RepeatedKFold {
    n_splits: usize,
    n_repeats: usize,
    random_state: Option<u64>,
}

impl RepeatedKFold {
    /// Construct a new [`RepeatedKFold`].
    #[must_use]
    pub fn new(n_splits: usize, n_repeats: usize) -> Self {
        Self {
            n_splits,
            n_repeats,
            random_state: None,
        }
    }

    /// Set the RNG seed for reproducible shuffling.
    #[must_use]
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

impl CrossValidator for RepeatedKFold {
    fn fold_indices(&self, n_samples: usize) -> Result<FoldSplits, FerroError> {
        let mut all = Vec::with_capacity(self.n_splits * self.n_repeats);
        for repeat in 0..self.n_repeats {
            let mut kf = crate::cross_validation::KFold::new(self.n_splits).shuffle(true);
            if let Some(seed) = self.random_state {
                kf = kf.random_state(seed.wrapping_add(repeat as u64));
            }
            let folds = kf.fold_indices(n_samples)?;
            all.extend(folds);
        }
        Ok(all)
    }
}

/// Repeated Stratified K-Fold cross-validator.
///
/// Like [`RepeatedKFold`] but preserves class proportions in each fold.
#[derive(Debug, Clone)]
pub struct RepeatedStratifiedKFold {
    n_splits: usize,
    n_repeats: usize,
    random_state: Option<u64>,
}

impl RepeatedStratifiedKFold {
    /// Construct a new [`RepeatedStratifiedKFold`].
    #[must_use]
    pub fn new(n_splits: usize, n_repeats: usize) -> Self {
        Self {
            n_splits,
            n_repeats,
            random_state: None,
        }
    }

    /// Set the RNG seed for reproducible shuffling.
    #[must_use]
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Generate the train/test splits for class labels `y`.
    pub fn split(&self, y: &Array1<usize>) -> Result<FoldSplits, FerroError> {
        let mut all = Vec::with_capacity(self.n_splits * self.n_repeats);
        for repeat in 0..self.n_repeats {
            let mut skf =
                crate::cross_validation::StratifiedKFold::new(self.n_splits).shuffle(true);
            if let Some(seed) = self.random_state {
                skf = skf.random_state(seed.wrapping_add(repeat as u64));
            }
            let folds = skf.split(y)?;
            all.extend(folds);
        }
        Ok(all)
    }
}

/// Splitter driven by a user-supplied `test_fold` index array.
///
/// Sample `i` belongs to test fold `test_fold[i]`. Samples with
/// `test_fold[i] == -1` are always in the training set.
#[derive(Debug, Clone)]
pub struct PredefinedSplit {
    test_fold: Array1<isize>,
}

impl PredefinedSplit {
    /// Construct a new [`PredefinedSplit`] from a `test_fold` array.
    ///
    /// `test_fold[i]` is the fold index that sample `i` is part of (or `-1`
    /// to keep it always in the training set).
    #[must_use]
    pub fn new(test_fold: Array1<isize>) -> Self {
        Self { test_fold }
    }
}

impl CrossValidator for PredefinedSplit {
    fn fold_indices(&self, n_samples: usize) -> Result<FoldSplits, FerroError> {
        if self.test_fold.len() != n_samples {
            return Err(FerroError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![self.test_fold.len()],
                context: "PredefinedSplit: test_fold length must equal n_samples".into(),
            });
        }
        let mut folds: HashMap<isize, Vec<usize>> = HashMap::new();
        for (i, &f) in self.test_fold.iter().enumerate() {
            if f >= 0 {
                folds.entry(f).or_default().push(i);
            }
        }
        let mut keys: Vec<isize> = folds.keys().copied().collect();
        keys.sort_unstable();
        let mut out: FoldSplits = Vec::with_capacity(keys.len());
        for k in keys {
            let test = folds.remove(&k).unwrap_or_default();
            let test_set: std::collections::HashSet<usize> = test.iter().copied().collect();
            let train: Vec<usize> = (0..n_samples).filter(|i| !test_set.contains(i)).collect();
            let pair: FoldSplit = (train, test);
            out.push(pair);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_loo_basic() {
        let folds = LeaveOneOut::new().fold_indices(5).unwrap();
        assert_eq!(folds.len(), 5);
        for (train, test) in &folds {
            assert_eq!(test.len(), 1);
            assert_eq!(train.len(), 4);
        }
    }

    #[test]
    fn test_lpo_p2() {
        let folds = LeavePOut::new(2).fold_indices(4).unwrap();
        // C(4, 2) = 6
        assert_eq!(folds.len(), 6);
    }

    #[test]
    fn test_shuffle_split_basic() {
        let folds = ShuffleSplit::new(3, 0.25)
            .random_state(42)
            .fold_indices(8)
            .unwrap();
        assert_eq!(folds.len(), 3);
        for (train, test) in &folds {
            assert_eq!(train.len() + test.len(), 8);
            assert!(test.len() >= 1);
        }
    }

    #[test]
    fn test_stratified_shuffle_split() {
        let y = array![0usize, 0, 0, 1, 1, 1, 2, 2];
        let folds = StratifiedShuffleSplit::new(2, 0.25)
            .random_state(7)
            .split(&y)
            .unwrap();
        assert_eq!(folds.len(), 2);
    }

    #[test]
    fn test_repeated_kfold() {
        let folds = RepeatedKFold::new(3, 2)
            .random_state(5)
            .fold_indices(9)
            .unwrap();
        assert_eq!(folds.len(), 6);
    }

    #[test]
    fn test_repeated_stratified() {
        let y = array![0usize, 0, 0, 1, 1, 1];
        let folds = RepeatedStratifiedKFold::new(3, 2)
            .random_state(11)
            .split(&y)
            .unwrap();
        assert_eq!(folds.len(), 6);
    }

    #[test]
    fn test_predefined_split() {
        let test_fold = array![0_isize, 1, -1, 1, 0];
        let folds = PredefinedSplit::new(test_fold).fold_indices(5).unwrap();
        assert_eq!(folds.len(), 2);
        // Index 2 has -1 so should always be in training.
        for (train, _test) in &folds {
            assert!(train.contains(&2));
        }
    }
}
