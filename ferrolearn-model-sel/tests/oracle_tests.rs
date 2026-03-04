//! Oracle tests that compare ferrolearn model selection utilities against
//! scikit-learn reference outputs stored in `fixtures/*.json`.

// ---------------------------------------------------------------------------
// KFold oracle test
// ---------------------------------------------------------------------------

#[test]
fn test_kfold_oracle() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../fixtures/kfold.json")).unwrap();

    let n_samples = fixture["input"]["n_samples"].as_u64().unwrap() as usize;
    let n_splits = fixture["params"]["n_splits"].as_u64().unwrap() as usize;
    let shuffle = fixture["params"]["shuffle"].as_bool().unwrap();

    assert!(!shuffle, "fixture uses shuffle=false");

    let kf = ferrolearn_model_sel::KFold::new(n_splits);
    let folds = kf.split(n_samples);

    let expected_folds = fixture["expected"]["folds"].as_array().unwrap();

    assert_eq!(
        folds.len(),
        expected_folds.len(),
        "number of folds mismatch"
    );

    for (fold_idx, (actual, expected)) in folds.iter().zip(expected_folds.iter()).enumerate() {
        let (actual_train, actual_test) = actual;

        let expected_train: Vec<usize> = expected["train"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let expected_test: Vec<usize> = expected["test"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();

        assert_eq!(
            actual_train, &expected_train,
            "fold {fold_idx}: train indices mismatch"
        );
        assert_eq!(
            actual_test, &expected_test,
            "fold {fold_idx}: test indices mismatch"
        );
    }
}
