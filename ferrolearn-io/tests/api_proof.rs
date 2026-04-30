//! Proof-of-API integration test for ferrolearn-io.
//!
//! Audit deliverable for crosslink #323 (under #251). Exercises the full
//! public API surface: binary envelope round-trip (file + bytes) and JSON
//! round-trip (file). Optional ONNX/PMML modules are gated by features and
//! are out of scope for this proof.

use ferrolearn_io::{
    load_model, load_model_bytes, load_model_json, save_model, save_model_bytes, save_model_json,
};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, PartialEq, Debug)]
struct ToyModel {
    weights: Vec<f64>,
    bias: f64,
    name: String,
}

fn example() -> ToyModel {
    ToyModel {
        weights: vec![1.0, -2.5, 3.5],
        bias: 0.5,
        name: "api_proof".to_owned(),
    }
}

fn tempdir() -> std::path::PathBuf {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    let dir = std::env::temp_dir().join(format!("ferrolearn_io_apiproof_{nanos}"));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn api_proof_envelope_round_trip_file() {
    let dir = tempdir();
    let path = dir.join("model.flrn");
    let model = example();

    save_model(&model, &path).unwrap();
    let loaded: ToyModel = load_model(&path).unwrap();
    assert_eq!(model, loaded);
}

#[test]
fn api_proof_envelope_round_trip_bytes() {
    let model = example();
    let bytes = save_model_bytes(&model).unwrap();
    let loaded: ToyModel = load_model_bytes(&bytes).unwrap();
    assert_eq!(model, loaded);
}

#[test]
fn api_proof_json_round_trip_file() {
    let dir = tempdir();
    let path = dir.join("model.json");
    let model = example();

    save_model_json(&model, &path).unwrap();
    let loaded: ToyModel = load_model_json(&path).unwrap();
    assert_eq!(model, loaded);
}

#[test]
fn api_proof_envelope_corruption_rejected() {
    let model = example();
    let mut bytes = save_model_bytes(&model).unwrap();
    // Mangle a byte in the middle of the envelope.
    let mid = bytes.len() / 2;
    bytes[mid] = bytes[mid].wrapping_add(1);
    let result: Result<ToyModel, _> = load_model_bytes(&bytes);
    assert!(result.is_err());
}

#[test]
fn api_proof_load_missing_file() {
    let result: Result<ToyModel, _> = load_model("/nonexistent/api_proof/missing.flrn");
    assert!(result.is_err());
    let result: Result<ToyModel, _> = load_model_json("/nonexistent/api_proof/missing.json");
    assert!(result.is_err());
}
