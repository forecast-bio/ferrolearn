//! Property-based tests for PCA mathematical invariants.

use ferrolearn_core::traits::{Fit, Transform};
use ferrolearn_decomp::PCA;
use ndarray::Array2;
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Shared strategies
// ---------------------------------------------------------------------------

/// Strategy for PCA data: at least 3 rows, 2-4 features, n_components <= n_features.
fn pca_data_strategy() -> impl Strategy<Value = (Array2<f64>, usize, usize, usize)> {
    (2usize..=4usize).prop_flat_map(|n_feat| {
        // n_components between 1 and n_feat
        (1usize..=n_feat, Just(n_feat))
    }).prop_flat_map(|(n_comp, n_feat)| {
        // n_samples between max(3, n_feat+1) and n_feat+8
        let min_samples = 3usize.max(n_feat + 1);
        let max_samples = n_feat + 8;
        (min_samples..=max_samples, Just(n_feat), Just(n_comp))
    }).prop_flat_map(|(n_samples, n_feat, n_comp)| {
        proptest::collection::vec(-5.0..5.0f64, n_samples * n_feat)
            .prop_map(move |data| {
                (
                    Array2::from_shape_vec((n_samples, n_feat), data).unwrap(),
                    n_samples,
                    n_feat,
                    n_comp,
                )
            })
    })
}

// ---------------------------------------------------------------------------
// PCA invariants
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn pca_components_orthonormal(
        (x, _n_samples, _n_feat, n_comp) in pca_data_strategy()
    ) {
        let pca = PCA::<f64>::new(n_comp);
        let result = pca.fit(&x, &());
        prop_assume!(result.is_ok());
        let fitted = result.unwrap();
        let c = fitted.components();

        // Check each component is unit length
        for i in 0..c.nrows() {
            let norm: f64 = c.row(i).iter().map(|v| v * v).sum::<f64>().sqrt();
            prop_assert!((norm - 1.0).abs() < 1e-6,
                "Component {} norm {} != 1.0", i, norm);
        }

        // Check mutual orthogonality
        for i in 0..c.nrows() {
            for j in (i + 1)..c.nrows() {
                let dot: f64 = c.row(i).iter()
                    .zip(c.row(j).iter())
                    .map(|(a, b)| a * b)
                    .sum();
                prop_assert!(dot.abs() < 1e-6,
                    "Components {} and {} not orthogonal, dot product = {}", i, j, dot);
            }
        }
    }

    #[test]
    fn pca_explained_variance_ratio_sums_le_1(
        (x, _n_samples, _n_feat, n_comp) in pca_data_strategy()
    ) {
        let pca = PCA::<f64>::new(n_comp);
        let result = pca.fit(&x, &());
        prop_assume!(result.is_ok());
        let fitted = result.unwrap();
        let ratio_sum: f64 = fitted.explained_variance_ratio().iter().sum();

        prop_assert!(ratio_sum <= 1.0 + 1e-8,
            "explained_variance_ratio_ sum {} > 1.0", ratio_sum);
    }

    #[test]
    fn pca_transform_output_has_n_components_columns(
        (x, _n_samples, _n_feat, n_comp) in pca_data_strategy()
    ) {
        let pca = PCA::<f64>::new(n_comp);
        let result = pca.fit(&x, &());
        prop_assume!(result.is_ok());
        let fitted = result.unwrap();
        let projected = fitted.transform(&x).unwrap();

        prop_assert_eq!(projected.ncols(), n_comp,
            "transform output has {} columns, expected {}", projected.ncols(), n_comp);
        prop_assert_eq!(projected.nrows(), x.nrows(),
            "transform output has {} rows, expected {}", projected.nrows(), x.nrows());
    }

    #[test]
    fn pca_inverse_transform_approx_recovers_x_full_components(
        (x, _n_samples, n_feat, _n_comp) in pca_data_strategy()
    ) {
        // Test with n_components == n_features for exact recovery
        let pca = PCA::<f64>::new(n_feat);
        let result = pca.fit(&x, &());
        prop_assume!(result.is_ok());
        let fitted = result.unwrap();
        let projected = fitted.transform(&x).unwrap();
        let recovered = fitted.inverse_transform(&projected).unwrap();

        for (a, b) in x.iter().zip(recovered.iter()) {
            prop_assert!((a - b).abs() < 1e-6,
                "Full-component inverse_transform roundtrip failed: {} != {}", a, b);
        }
    }

    #[test]
    fn pca_explained_variance_non_negative(
        (x, _n_samples, _n_feat, n_comp) in pca_data_strategy()
    ) {
        let pca = PCA::<f64>::new(n_comp);
        let result = pca.fit(&x, &());
        prop_assume!(result.is_ok());
        let fitted = result.unwrap();

        for &v in fitted.explained_variance().iter() {
            prop_assert!(v >= -1e-10,
                "explained_variance {} is negative", v);
        }
    }

    #[test]
    fn pca_explained_variance_ratio_non_negative(
        (x, _n_samples, _n_feat, n_comp) in pca_data_strategy()
    ) {
        let pca = PCA::<f64>::new(n_comp);
        let result = pca.fit(&x, &());
        prop_assume!(result.is_ok());
        let fitted = result.unwrap();

        for &v in fitted.explained_variance_ratio().iter() {
            prop_assert!(v >= -1e-10,
                "explained_variance_ratio {} is negative", v);
        }
    }
}
