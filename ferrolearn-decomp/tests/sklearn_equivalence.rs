//! Statistical equivalence tests comparing ferrolearn decomposition models against
//! scikit-learn reference values stored in `tests/fixtures/sklearn_reference/`.

use approx::assert_relative_eq;
use ferrolearn_core::{Fit, Transform};
use ndarray::Array2;

// ---------------------------------------------------------------------------
// JSON helpers
// ---------------------------------------------------------------------------

fn json_to_array2(value: &serde_json::Value) -> Array2<f64> {
    let rows: Vec<Vec<f64>> = value
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect()
        })
        .collect();
    let n_rows = rows.len();
    let n_cols = rows[0].len();
    let flat: Vec<f64> = rows.into_iter().flatten().collect();
    Array2::from_shape_vec((n_rows, n_cols), flat).unwrap()
}

fn json_to_vec_f64(value: &serde_json::Value) -> Vec<f64> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect()
}

// ---------------------------------------------------------------------------
// PCA
// ---------------------------------------------------------------------------

#[test]
fn sklearn_equiv_pca() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../tests/fixtures/sklearn_reference/pca.json"
    ))
    .unwrap();

    let x = json_to_array2(&fixture["X_train"]);
    let sklearn_explained_var = json_to_vec_f64(&fixture["explained_variance"]);
    let sklearn_explained_ratio = json_to_vec_f64(&fixture["explained_variance_ratio"]);
    let sklearn_mean = json_to_vec_f64(&fixture["mean"]);
    let sklearn_components = json_to_array2(&fixture["components"]);
    let sklearn_transformed = json_to_array2(&fixture["transformed"]);

    let pca = ferrolearn_decomp::PCA::<f64>::new(3);
    let fitted = pca.fit(&x, &()).unwrap();
    let transformed = fitted.transform(&x).unwrap();

    // Shape checks.
    assert_eq!(transformed.nrows(), x.nrows());
    assert_eq!(transformed.ncols(), 3);

    let components = fitted.components();
    assert_eq!(components.nrows(), 3);
    assert_eq!(components.ncols(), x.ncols());

    // Mean should match sklearn's closely.
    let mean = fitted.mean();
    for (i, (&actual, &expected)) in mean.iter().zip(sklearn_mean.iter()).enumerate() {
        assert_relative_eq!(actual, expected, epsilon = 1e-6);
        let _ = i;
    }

    // Explained variance should match closely.
    let ev = fitted.explained_variance();
    for (i, (&actual, &expected)) in ev.iter().zip(sklearn_explained_var.iter()).enumerate() {
        let diff = (actual - expected).abs();
        assert!(
            diff < 0.1,
            "PCA explained_variance[{i}]: actual={actual:.6}, expected={expected:.6}"
        );
    }

    // Explained variance ratio should match closely.
    let evr = fitted.explained_variance_ratio();
    for (i, (&actual, &expected)) in evr.iter().zip(sklearn_explained_ratio.iter()).enumerate() {
        let diff = (actual - expected).abs();
        assert!(
            diff < 0.05,
            "PCA explained_variance_ratio[{i}]: actual={actual:.6}, expected={expected:.6}"
        );
    }

    // Components may have sign flips. Compare abs values.
    for comp_idx in 0..3 {
        let actual_row = components.row(comp_idx);
        let sklearn_row = sklearn_components.row(comp_idx);

        for j in 0..actual_row.len() {
            let diff = (actual_row[j].abs() - sklearn_row[j].abs()).abs();
            assert!(
                diff < 0.15,
                "PCA component[{comp_idx}][{j}]: |actual|={:.6}, |sklearn|={:.6}",
                actual_row[j].abs(),
                sklearn_row[j].abs()
            );
        }
    }

    // Transformed values: compare after accounting for sign flips.
    for col in 0..3 {
        let actual_first = transformed[[0, col]];
        let sklearn_first = sklearn_transformed[[0, col]];
        let sign = if actual_first * sklearn_first >= 0.0 {
            1.0
        } else {
            -1.0
        };

        for row in 0..transformed.nrows() {
            let actual = transformed[[row, col]] * sign;
            let expected = sklearn_transformed[[row, col]];
            let diff = (actual - expected).abs();
            assert!(
                diff < 0.5,
                "PCA transformed[{row}][{col}]: actual={actual:.4}, expected={expected:.4}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// NMF
// ---------------------------------------------------------------------------

#[test]
fn sklearn_equiv_nmf() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../tests/fixtures/sklearn_reference/nmf.json"
    ))
    .unwrap();

    let x = json_to_array2(&fixture["X_train"]);
    let sklearn_reconstruction_err = fixture["reconstruction_error"].as_f64().unwrap();
    let sklearn_h = json_to_array2(&fixture["H"]);
    let sklearn_w = json_to_array2(&fixture["W"]);

    let nmf = ferrolearn_decomp::NMF::<f64>::new(3)
        .with_init(ferrolearn_decomp::NMFInit::Nndsvd)
        .with_random_state(42)
        .with_max_iter(500);
    let fitted = nmf.fit(&x, &()).unwrap();

    // H matrix shape.
    let h = fitted.components();
    assert_eq!(h.nrows(), 3);
    assert_eq!(h.ncols(), x.ncols());

    // H should be non-negative.
    for &val in h {
        assert!(val >= 0.0, "NMF H matrix contains negative value: {val}");
    }

    // Shape checks.
    assert_eq!(h.nrows(), sklearn_h.nrows());
    assert_eq!(h.ncols(), sklearn_h.ncols());

    // Each component (row of H) should have similar L2 norm.
    for i in 0..h.nrows() {
        let our_norm: f64 = h.row(i).iter().map(|v| v * v).sum::<f64>().sqrt();
        let sk_norm: f64 = sklearn_h.row(i).iter().map(|v| v * v).sum::<f64>().sqrt();
        let ratio = our_norm / sk_norm.max(1e-10);
        assert!(
            (0.1..10.0).contains(&ratio),
            "NMF H row {i} norm ratio {ratio:.2} out of range"
        );
    }

    // Reconstruction error should be in the same ballpark.
    let recon_err = fitted.reconstruction_err();
    let ratio = recon_err / sklearn_reconstruction_err;
    assert!(
        (0.5..2.0).contains(&ratio),
        "NMF reconstruction error {recon_err:.4} too far from sklearn {sklearn_reconstruction_err:.4}"
    );

    // W = transform(X) should be non-negative.
    let w = fitted.transform(&x).unwrap();
    assert_eq!(w.nrows(), x.nrows());
    assert_eq!(w.ncols(), 3);
    for &val in &w {
        assert!(val >= 0.0, "NMF W matrix contains negative value: {val}");
    }

    assert_eq!(w.nrows(), sklearn_w.nrows());
    assert_eq!(w.ncols(), sklearn_w.ncols());

    // Each column of W should have similar norm.
    for j in 0..w.ncols() {
        let our_norm: f64 = w.column(j).iter().map(|v| v * v).sum::<f64>().sqrt();
        let sk_norm: f64 = sklearn_w
            .column(j)
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        let ratio = our_norm / sk_norm.max(1e-10);
        assert!(
            (0.1..10.0).contains(&ratio),
            "NMF W col {j} norm ratio {ratio:.2} out of range"
        );
    }

    // Approximate reconstruction: X ~= W * H.
    let reconstructed = w.dot(h);
    let mut recon_frobenius = 0.0_f64;
    for (&orig, &rec) in x.iter().zip(reconstructed.iter()) {
        recon_frobenius += (orig - rec).powi(2);
    }
    recon_frobenius = recon_frobenius.sqrt();

    assert!(
        recon_frobenius < sklearn_reconstruction_err * 3.0,
        "NMF Frobenius reconstruction error {recon_frobenius:.4} too large"
    );
}
