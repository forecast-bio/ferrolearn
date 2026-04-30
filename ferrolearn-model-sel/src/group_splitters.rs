//! Group-aware cross-validation splitters.
//!
//! Each splitter takes an `Array1<usize>` of group labels (one per sample)
//! and produces folds where samples sharing a group never appear in both the
//! train and test split — useful when samples are not independent.
//!
//! - [`GroupKFold`] — partition `n_groups` distinct groups into `n_splits`
//!   roughly-equal folds.
//! - [`GroupShuffleSplit`] — random group-wise train/test splits.
//! - [`LeaveOneGroupOut`] — one fold per unique group.
//! - [`LeavePGroupsOut`] — one fold per `p`-sized subset of groups.
//! - [`StratifiedGroupKFold`] — group-aware folding that also tries to
//!   preserve class balance per fold.

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::cross_validation::FoldSplits;
use ferrolearn_core::FerroError;
use ndarray::Array1;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;

fn unique_groups(groups: &Array1<usize>) -> Vec<usize> {
    let mut g: Vec<usize> = groups.iter().copied().collect();
    g.sort_unstable();
    g.dedup();
    g
}

fn check_non_empty(groups: &Array1<usize>, context: &str) -> Result<(), FerroError> {
    if groups.is_empty() {
        return Err(FerroError::InsufficientSamples {
            required: 1,
            actual: 0,
            context: context.into(),
        });
    }
    Ok(())
}

/// Partition unique groups into `n_splits` folds.
#[derive(Debug, Clone)]
pub struct GroupKFold {
    n_splits: usize,
}

impl GroupKFold {
    /// Construct a new [`GroupKFold`] with the given number of folds.
    #[must_use]
    pub fn new(n_splits: usize) -> Self {
        Self { n_splits }
    }

    /// Generate the splits for the given group labels.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::InvalidParameter`] if `n_splits < 2` or if
    /// `n_splits > n_unique_groups`.
    pub fn split(&self, groups: &Array1<usize>) -> Result<FoldSplits, FerroError> {
        check_non_empty(groups, "GroupKFold")?;
        if self.n_splits < 2 {
            return Err(FerroError::InvalidParameter {
                name: "n_splits".into(),
                reason: format!("must be >= 2, got {}", self.n_splits),
            });
        }
        let unique = unique_groups(groups);
        if unique.len() < self.n_splits {
            return Err(FerroError::InvalidParameter {
                name: "n_splits".into(),
                reason: format!(
                    "GroupKFold needs n_splits ({}) <= unique groups ({})",
                    self.n_splits,
                    unique.len()
                ),
            });
        }

        // Sort groups by descending size, assign each to the smallest-load fold.
        let mut sizes: HashMap<usize, usize> = HashMap::new();
        for &g in groups.iter() {
            *sizes.entry(g).or_insert(0) += 1;
        }
        let mut ordered: Vec<(usize, usize)> = sizes.into_iter().collect();
        ordered.sort_by_key(|t| std::cmp::Reverse(t.1));

        let mut fold_size = vec![0usize; self.n_splits];
        let mut group_to_fold: HashMap<usize, usize> = HashMap::new();
        for (group, count) in ordered {
            // pick fold with smallest current size
            let mut min_idx = 0usize;
            let mut min_val = fold_size[0];
            for (i, &v) in fold_size.iter().enumerate().skip(1) {
                if v < min_val {
                    min_val = v;
                    min_idx = i;
                }
            }
            group_to_fold.insert(group, min_idx);
            fold_size[min_idx] += count;
        }

        let mut folds: Vec<(Vec<usize>, Vec<usize>)> = (0..self.n_splits)
            .map(|_| (Vec::new(), Vec::new()))
            .collect();
        for (i, &g) in groups.iter().enumerate() {
            let fold_idx = *group_to_fold.get(&g).unwrap();
            for (k, (train, test)) in folds.iter_mut().enumerate() {
                if k == fold_idx {
                    test.push(i);
                } else {
                    train.push(i);
                }
            }
        }
        Ok(folds)
    }
}

/// Random group-wise train/test splits.
#[derive(Debug, Clone)]
pub struct GroupShuffleSplit {
    n_splits: usize,
    test_size: f64,
    random_state: Option<u64>,
}

impl GroupShuffleSplit {
    /// Construct a new [`GroupShuffleSplit`].
    #[must_use]
    pub fn new(n_splits: usize, test_size: f64) -> Self {
        Self {
            n_splits,
            test_size,
            random_state: None,
        }
    }

    /// Set the RNG seed for reproducibility.
    #[must_use]
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Generate the splits for the given group labels.
    pub fn split(&self, groups: &Array1<usize>) -> Result<FoldSplits, FerroError> {
        check_non_empty(groups, "GroupShuffleSplit")?;
        if !(0.0 < self.test_size && self.test_size < 1.0) {
            return Err(FerroError::InvalidParameter {
                name: "test_size".into(),
                reason: format!(
                    "GroupShuffleSplit: test_size must be in (0, 1), got {}",
                    self.test_size
                ),
            });
        }
        let unique = unique_groups(groups);
        let n_groups = unique.len();
        if n_groups < 2 {
            return Err(FerroError::InvalidParameter {
                name: "n_groups".into(),
                reason: "GroupShuffleSplit needs at least 2 distinct groups".into(),
            });
        }
        let n_test = ((n_groups as f64) * self.test_size).round().max(1.0) as usize;
        let n_test = n_test.min(n_groups - 1);

        let mut folds = Vec::with_capacity(self.n_splits);
        for split in 0..self.n_splits {
            let mut rng = match self.random_state {
                Some(seed) => SmallRng::seed_from_u64(seed.wrapping_add(split as u64)),
                None => SmallRng::from_os_rng(),
            };
            let mut shuffled = unique.clone();
            shuffled.shuffle(&mut rng);
            let test_groups: HashSet<usize> = shuffled[..n_test].iter().copied().collect();
            let mut train = Vec::new();
            let mut test = Vec::new();
            for (i, &g) in groups.iter().enumerate() {
                if test_groups.contains(&g) {
                    test.push(i);
                } else {
                    train.push(i);
                }
            }
            folds.push((train, test));
        }
        Ok(folds)
    }
}

/// One fold per unique group: the test set for fold `i` is exactly the
/// samples in the `i`-th group.
#[derive(Debug, Clone, Default)]
pub struct LeaveOneGroupOut;

impl LeaveOneGroupOut {
    /// Construct a new [`LeaveOneGroupOut`].
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Generate the splits for the given group labels.
    pub fn split(&self, groups: &Array1<usize>) -> Result<FoldSplits, FerroError> {
        check_non_empty(groups, "LeaveOneGroupOut")?;
        let unique = unique_groups(groups);
        if unique.len() < 2 {
            return Err(FerroError::InvalidParameter {
                name: "n_groups".into(),
                reason: "LeaveOneGroupOut needs at least 2 distinct groups".into(),
            });
        }
        let mut folds = Vec::with_capacity(unique.len());
        for &target in &unique {
            let mut train = Vec::new();
            let mut test = Vec::new();
            for (i, &g) in groups.iter().enumerate() {
                if g == target {
                    test.push(i);
                } else {
                    train.push(i);
                }
            }
            folds.push((train, test));
        }
        Ok(folds)
    }
}

/// One fold per `p`-sized subset of groups (`C(n_groups, p)` folds total).
#[derive(Debug, Clone)]
pub struct LeavePGroupsOut {
    p: usize,
}

impl LeavePGroupsOut {
    /// Construct a new [`LeavePGroupsOut`] with the given `p`.
    #[must_use]
    pub fn new(p: usize) -> Self {
        Self { p }
    }

    /// Generate the splits for the given group labels.
    pub fn split(&self, groups: &Array1<usize>) -> Result<FoldSplits, FerroError> {
        check_non_empty(groups, "LeavePGroupsOut")?;
        if self.p == 0 {
            return Err(FerroError::InvalidParameter {
                name: "p".into(),
                reason: "LeavePGroupsOut: p must be >= 1".into(),
            });
        }
        let unique = unique_groups(groups);
        if unique.len() <= self.p {
            return Err(FerroError::InvalidParameter {
                name: "p".into(),
                reason: format!(
                    "LeavePGroupsOut needs n_unique_groups ({}) > p ({})",
                    unique.len(),
                    self.p
                ),
            });
        }
        let mut folds = Vec::new();
        let n_g = unique.len();
        let mut combo: Vec<usize> = (0..self.p).collect();
        loop {
            let test_set: HashSet<usize> = combo.iter().map(|&k| unique[k]).collect();
            let mut train = Vec::new();
            let mut test = Vec::new();
            for (i, &g) in groups.iter().enumerate() {
                if test_set.contains(&g) {
                    test.push(i);
                } else {
                    train.push(i);
                }
            }
            folds.push((train, test));

            // Advance combo lexicographically.
            let mut i = self.p;
            while i > 0 {
                i -= 1;
                if combo[i] < n_g - self.p + i {
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

/// Group-aware k-fold that also tries to preserve class balance per fold.
#[derive(Debug, Clone)]
pub struct StratifiedGroupKFold {
    n_splits: usize,
}

impl StratifiedGroupKFold {
    /// Construct a new [`StratifiedGroupKFold`].
    #[must_use]
    pub fn new(n_splits: usize) -> Self {
        Self { n_splits }
    }

    /// Generate the splits for the given class labels and group labels.
    ///
    /// # Errors
    ///
    /// Returns [`FerroError::ShapeMismatch`] if `y` and `groups` have
    /// different lengths.
    /// Returns [`FerroError::InvalidParameter`] if `n_splits < 2` or
    /// `n_splits > n_unique_groups`.
    pub fn split(
        &self,
        y: &Array1<usize>,
        groups: &Array1<usize>,
    ) -> Result<FoldSplits, FerroError> {
        if y.len() != groups.len() {
            return Err(FerroError::ShapeMismatch {
                expected: vec![y.len()],
                actual: vec![groups.len()],
                context: "StratifiedGroupKFold: y and groups must have the same length".into(),
            });
        }
        check_non_empty(groups, "StratifiedGroupKFold")?;
        if self.n_splits < 2 {
            return Err(FerroError::InvalidParameter {
                name: "n_splits".into(),
                reason: format!("must be >= 2, got {}", self.n_splits),
            });
        }
        let unique_g = unique_groups(groups);
        if unique_g.len() < self.n_splits {
            return Err(FerroError::InvalidParameter {
                name: "n_splits".into(),
                reason: format!(
                    "StratifiedGroupKFold needs n_splits ({}) <= unique groups ({})",
                    self.n_splits,
                    unique_g.len()
                ),
            });
        }

        // Compute per-group class counts and total class counts.
        let unique_y = {
            let mut v: Vec<usize> = y.iter().copied().collect();
            v.sort_unstable();
            v.dedup();
            v
        };
        let n_classes = unique_y.len();
        let class_idx: HashMap<usize, usize> =
            unique_y.iter().enumerate().map(|(i, &c)| (c, i)).collect();
        let mut group_counts: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (&g, &c) in groups.iter().zip(y.iter()) {
            let entry = group_counts
                .entry(g)
                .or_insert_with(|| vec![0usize; n_classes]);
            entry[class_idx[&c]] += 1;
        }

        // Total count per class
        let mut total_per_class = vec![0usize; n_classes];
        for counts in group_counts.values() {
            for (i, &v) in counts.iter().enumerate() {
                total_per_class[i] += v;
            }
        }

        // Greedy assignment: process groups by descending size, place each
        // into the fold where it minimises the per-class deviation from the
        // target proportions.
        let target_per_fold: Vec<f64> = total_per_class
            .iter()
            .map(|&t| t as f64 / self.n_splits as f64)
            .collect();
        let mut fold_class_counts = vec![vec![0usize; n_classes]; self.n_splits];
        let mut group_to_fold: HashMap<usize, usize> = HashMap::new();

        let mut ordered: Vec<(usize, Vec<usize>)> = group_counts.into_iter().collect();
        ordered.sort_by(|a, b| {
            let sa: usize = a.1.iter().sum();
            let sb: usize = b.1.iter().sum();
            sb.cmp(&sa)
        });

        for (group, counts) in ordered {
            let mut best_fold = 0usize;
            let mut best_score = f64::INFINITY;
            for (k, fold_counts) in fold_class_counts.iter().enumerate() {
                let mut score = 0.0_f64;
                for c in 0..n_classes {
                    let new_count = (fold_counts[c] + counts[c]) as f64;
                    let dev = new_count - target_per_fold[c];
                    score += dev * dev;
                }
                if score < best_score {
                    best_score = score;
                    best_fold = k;
                }
            }
            for c in 0..n_classes {
                fold_class_counts[best_fold][c] += counts[c];
            }
            group_to_fold.insert(group, best_fold);
        }

        let mut folds: Vec<(Vec<usize>, Vec<usize>)> = (0..self.n_splits)
            .map(|_| (Vec::new(), Vec::new()))
            .collect();
        for (i, &g) in groups.iter().enumerate() {
            let fold_idx = *group_to_fold.get(&g).unwrap();
            for (k, (train, test)) in folds.iter_mut().enumerate() {
                if k == fold_idx {
                    test.push(i);
                } else {
                    train.push(i);
                }
            }
        }
        Ok(folds)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn group_kfold_partitions_groups() {
        let groups = array![0usize, 0, 1, 1, 2, 2, 3, 3];
        let folds = GroupKFold::new(2).split(&groups).unwrap();
        assert_eq!(folds.len(), 2);
        // Each fold's test indices should belong to a disjoint set of groups
        // from its train indices.
        for (train, test) in &folds {
            let test_groups: HashSet<usize> = test.iter().map(|&i| groups[i]).collect();
            let train_groups: HashSet<usize> = train.iter().map(|&i| groups[i]).collect();
            assert!(test_groups.is_disjoint(&train_groups));
        }
    }

    #[test]
    fn group_shuffle_split_deterministic() {
        let groups = array![0usize, 0, 1, 1, 2, 2, 3, 3];
        let a = GroupShuffleSplit::new(2, 0.5)
            .random_state(7)
            .split(&groups)
            .unwrap();
        let b = GroupShuffleSplit::new(2, 0.5)
            .random_state(7)
            .split(&groups)
            .unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn leave_one_group_out_one_fold_per_group() {
        let groups = array![0usize, 0, 1, 1, 2];
        let folds = LeaveOneGroupOut::new().split(&groups).unwrap();
        assert_eq!(folds.len(), 3);
    }

    #[test]
    fn leave_p_groups_out_combinations() {
        let groups = array![0usize, 1, 2, 3];
        let folds = LeavePGroupsOut::new(2).split(&groups).unwrap();
        // C(4, 2) = 6
        assert_eq!(folds.len(), 6);
    }

    #[test]
    fn stratified_group_kfold_balances() {
        let y = array![0usize, 0, 0, 0, 1, 1, 1, 1];
        let groups = array![0usize, 0, 1, 1, 2, 2, 3, 3];
        let folds = StratifiedGroupKFold::new(2).split(&y, &groups).unwrap();
        assert_eq!(folds.len(), 2);
        for (train, test) in &folds {
            let test_groups: HashSet<usize> = test.iter().map(|&i| groups[i]).collect();
            let train_groups: HashSet<usize> = train.iter().map(|&i| groups[i]).collect();
            assert!(test_groups.is_disjoint(&train_groups));
        }
    }

    #[test]
    fn stratified_group_kfold_shape_mismatch() {
        let y = array![0usize, 1];
        let groups = array![0usize];
        assert!(StratifiedGroupKFold::new(2).split(&y, &groups).is_err());
    }
}
