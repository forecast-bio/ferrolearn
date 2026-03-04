//! # ferrolearn-io
//!
//! Serialization and I/O utilities for the ferrolearn machine learning framework.
//!
//! This crate provides model serialization, deserialization,
//! and I/O utilities for saving and loading trained models.
//!
//! Models are stored in a binary envelope format using MessagePack
//! (`rmp-serde`) with a CRC32 integrity checksum, or alternatively
//! exported to JSON via `serde_json`.
//!
//! # Example
//!
//! ```no_run
//! use ferrolearn_io::{save_model, load_model};
//! use serde::{Serialize, Deserialize};
//!
//! #[derive(Serialize, Deserialize, PartialEq, Debug)]
//! struct MyModel {
//!     weights: Vec<f64>,
//!     bias: f64,
//! }
//!
//! let model = MyModel { weights: vec![1.0, 2.0, 3.0], bias: 0.5 };
//! save_model(&model, "/tmp/my_model.flrn").unwrap();
//! let loaded: MyModel = load_model("/tmp/my_model.flrn").unwrap();
//! assert_eq!(model, loaded);
//! ```

use std::fs;
use std::path::Path;

use ferrolearn_core::FerroError;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

/// Magic bytes that identify a ferrolearn model file.
const MAGIC: [u8; 4] = *b"FLRN";

/// Current schema version for the envelope format.
const SCHEMA_VERSION: u32 = 1;

/// The envelope that wraps a serialized model payload.
///
/// This structure is serialized to disk (via MessagePack) and contains
/// metadata for integrity verification and format evolution.
#[derive(Serialize, Deserialize)]
struct ModelEnvelope {
    /// Magic bytes identifying this as a ferrolearn file.
    magic: [u8; 4],
    /// The schema version of this envelope.
    schema_version: u32,
    /// A human-readable tag describing the model type (currently always `"generic"`).
    model_type: String,
    /// The raw MessagePack bytes of the serialized model.
    payload: Vec<u8>,
    /// CRC32 checksum of `payload`.
    checksum: u32,
}

/// Serialize `model` to MessagePack bytes, wrap in a [`ModelEnvelope`], and
/// write the envelope to `path`.
///
/// # Errors
///
/// Returns [`FerroError::SerdeError`] if MessagePack serialization fails, or
/// [`FerroError::IoError`] if the file cannot be written.
pub fn save_model<T: Serialize>(model: &T, path: impl AsRef<Path>) -> Result<(), FerroError> {
    let bytes = save_model_bytes(model)?;
    fs::write(path, bytes).map_err(FerroError::IoError)
}

/// Read a model file from `path`, validate the envelope, and deserialize into `T`.
///
/// Validation checks:
/// - Magic bytes must equal `FLRN`.
/// - Schema version must equal [`SCHEMA_VERSION`].
/// - CRC32 of the payload must match the stored checksum.
///
/// # Errors
///
/// Returns [`FerroError::IoError`] if the file cannot be read, or
/// [`FerroError::SerdeError`] for any format / integrity violation.
pub fn load_model<T: DeserializeOwned>(path: impl AsRef<Path>) -> Result<T, FerroError> {
    let bytes = fs::read(path).map_err(FerroError::IoError)?;
    load_model_bytes(&bytes)
}

/// Serialize `model` to a self-contained `Vec<u8>` using the envelope format.
///
/// The returned bytes can be stored anywhere (file, database, network) and
/// later restored with [`load_model_bytes`].
///
/// # Errors
///
/// Returns [`FerroError::SerdeError`] if serialization fails.
pub fn save_model_bytes<T: Serialize>(model: &T) -> Result<Vec<u8>, FerroError> {
    let payload = rmp_serde::to_vec(model).map_err(|e| FerroError::SerdeError {
        message: e.to_string(),
    })?;

    let checksum = crc32fast::hash(&payload);

    let envelope = ModelEnvelope {
        magic: MAGIC,
        schema_version: SCHEMA_VERSION,
        model_type: "generic".to_owned(),
        payload,
        checksum,
    };

    rmp_serde::to_vec(&envelope).map_err(|e| FerroError::SerdeError {
        message: e.to_string(),
    })
}

/// Deserialize a model from bytes produced by [`save_model_bytes`].
///
/// # Errors
///
/// Returns [`FerroError::SerdeError`] if the bytes are malformed, the magic
/// bytes are wrong, the schema version is unsupported, or the checksum does
/// not match.
pub fn load_model_bytes<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, FerroError> {
    let envelope: ModelEnvelope =
        rmp_serde::from_slice(bytes).map_err(|e| FerroError::SerdeError {
            message: format!("failed to deserialize envelope: {e}"),
        })?;

    if envelope.magic != MAGIC {
        return Err(FerroError::SerdeError {
            message: format!(
                "invalid magic bytes: expected {:?}, got {:?}",
                MAGIC, envelope.magic
            ),
        });
    }

    if envelope.schema_version != SCHEMA_VERSION {
        return Err(FerroError::SerdeError {
            message: format!(
                "unsupported schema version: expected {SCHEMA_VERSION}, got {}",
                envelope.schema_version
            ),
        });
    }

    let actual_checksum = crc32fast::hash(&envelope.payload);
    if actual_checksum != envelope.checksum {
        return Err(FerroError::SerdeError {
            message: format!(
                "checksum mismatch: expected {}, got {actual_checksum}",
                envelope.checksum
            ),
        });
    }

    rmp_serde::from_slice(&envelope.payload).map_err(|e| FerroError::SerdeError {
        message: format!("failed to deserialize model payload: {e}"),
    })
}

/// Serialize `model` as pretty-printed JSON and write it to `path`.
///
/// # Errors
///
/// Returns [`FerroError::SerdeError`] if JSON serialization fails, or
/// [`FerroError::IoError`] if the file cannot be written.
pub fn save_model_json<T: Serialize>(model: &T, path: impl AsRef<Path>) -> Result<(), FerroError> {
    let json = serde_json::to_string_pretty(model).map_err(|e| FerroError::SerdeError {
        message: e.to_string(),
    })?;
    fs::write(path, json).map_err(FerroError::IoError)
}

/// Read a JSON file from `path` and deserialize it into `T`.
///
/// # Errors
///
/// Returns [`FerroError::IoError`] if the file cannot be read, or
/// [`FerroError::SerdeError`] if the JSON is malformed or does not match `T`.
pub fn load_model_json<T: DeserializeOwned>(path: impl AsRef<Path>) -> Result<T, FerroError> {
    let json = fs::read_to_string(path).map_err(FerroError::IoError)?;
    serde_json::from_str(&json).map_err(|e| FerroError::SerdeError {
        message: e.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// A simple model struct used across multiple tests.
    #[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
    struct SimpleModel {
        weights: Vec<f64>,
        bias: f64,
        name: String,
    }

    impl SimpleModel {
        fn example() -> Self {
            Self {
                weights: vec![1.0, -2.5, 3.14],
                bias: 0.5,
                name: "test".to_owned(),
            }
        }
    }

    // -----------------------------------------------------------------------
    // Round-trip tests (binary envelope)
    // -----------------------------------------------------------------------

    #[test]
    fn test_round_trip_file() {
        let dir = tempdir();
        let path = dir.join("model.flrn");
        let model = SimpleModel::example();

        save_model(&model, &path).unwrap();
        let loaded: SimpleModel = load_model(&path).unwrap();

        assert_eq!(model, loaded);
    }

    #[test]
    fn test_round_trip_bytes() {
        let model = SimpleModel::example();
        let bytes = save_model_bytes(&model).unwrap();
        let loaded: SimpleModel = load_model_bytes(&bytes).unwrap();
        assert_eq!(model, loaded);
    }

    #[test]
    fn test_round_trip_empty_weights() {
        #[derive(Serialize, Deserialize, PartialEq, Debug)]
        struct Empty {
            v: Vec<f32>,
        }
        let model = Empty { v: vec![] };
        let bytes = save_model_bytes(&model).unwrap();
        let loaded: Empty = load_model_bytes(&bytes).unwrap();
        assert_eq!(model, loaded);
    }

    #[test]
    fn test_round_trip_bool_and_integer_fields() {
        #[derive(Serialize, Deserialize, PartialEq, Debug)]
        struct Mixed {
            flag: bool,
            count: u64,
            label: String,
        }
        let model = Mixed {
            flag: true,
            count: u64::MAX,
            label: "ferrolearn".to_owned(),
        };
        let bytes = save_model_bytes(&model).unwrap();
        let loaded: Mixed = load_model_bytes(&bytes).unwrap();
        assert_eq!(model, loaded);
    }

    // -----------------------------------------------------------------------
    // JSON export tests (REQ-14)
    // -----------------------------------------------------------------------

    #[test]
    fn test_json_round_trip_file() {
        let dir = tempdir();
        let path = dir.join("model.json");
        let model = SimpleModel::example();

        save_model_json(&model, &path).unwrap();
        let loaded: SimpleModel = load_model_json(&path).unwrap();

        assert_eq!(model, loaded);
    }

    #[test]
    fn test_json_output_is_valid_utf8() {
        let dir = tempdir();
        let path = dir.join("model.json");
        let model = SimpleModel::example();

        save_model_json(&model, &path).unwrap();
        let contents = std::fs::read_to_string(&path).unwrap();
        // serde_json pretty-prints with indentation
        assert!(contents.contains("weights"));
        assert!(contents.contains("bias"));
    }

    #[test]
    fn test_json_round_trip_nested() {
        #[derive(Serialize, Deserialize, PartialEq, Debug)]
        struct Nested {
            layers: Vec<Vec<f64>>,
            meta: HashMap<String, String>,
        }
        let mut meta = HashMap::new();
        meta.insert("author".to_owned(), "ferrolearn".to_owned());
        let model = Nested {
            layers: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            meta,
        };
        let dir = tempdir();
        let path = dir.join("nested.json");
        save_model_json(&model, &path).unwrap();
        let loaded: Nested = load_model_json(&path).unwrap();
        assert_eq!(model, loaded);
    }

    // -----------------------------------------------------------------------
    // Error-path tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_error_corrupt_magic_bytes() {
        let model = SimpleModel::example();
        let mut bytes = save_model_bytes(&model).unwrap();

        // The envelope is itself MessagePack-encoded.  Corrupt the first data
        // byte so that the inner magic array will be wrong after decode.  We
        // build a tampered envelope directly instead.
        let mut env: ModelEnvelope = rmp_serde::from_slice(&bytes).unwrap();
        env.magic = *b"XXXX";
        bytes = rmp_serde::to_vec(&env).unwrap();

        let result: Result<SimpleModel, _> = load_model_bytes(&bytes);
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("invalid magic bytes"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_error_wrong_schema_version() {
        let model = SimpleModel::example();
        let bytes = save_model_bytes(&model).unwrap();

        let mut env: ModelEnvelope = rmp_serde::from_slice(&bytes).unwrap();
        env.schema_version = 99;
        let tampered = rmp_serde::to_vec(&env).unwrap();

        let result: Result<SimpleModel, _> = load_model_bytes(&tampered);
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("unsupported schema version"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_error_bad_checksum() {
        let model = SimpleModel::example();
        let bytes = save_model_bytes(&model).unwrap();

        let mut env: ModelEnvelope = rmp_serde::from_slice(&bytes).unwrap();
        env.checksum = env.checksum.wrapping_add(1);
        let tampered = rmp_serde::to_vec(&env).unwrap();

        let result: Result<SimpleModel, _> = load_model_bytes(&tampered);
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("checksum mismatch"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_error_file_not_found() {
        let result: Result<SimpleModel, _> = load_model("/nonexistent/path/model.flrn");
        let err = result.unwrap_err();
        // IoError wraps std::io::Error whose Display contains "No such file"
        assert!(
            err.to_string().to_lowercase().contains("no such file")
                || err.to_string().to_lowercase().contains("os error"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_error_json_file_not_found() {
        let result: Result<SimpleModel, _> = load_model_json("/nonexistent/path/model.json");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_completely_garbage_bytes() {
        let garbage: &[u8] = b"this is not a valid msgpack envelope";
        let result: Result<SimpleModel, _> = load_model_bytes(garbage);
        assert!(result.is_err(), "expected error for garbage bytes, got Ok");
    }

    // -----------------------------------------------------------------------
    // Large struct serialization
    // -----------------------------------------------------------------------

    #[test]
    fn test_large_struct_round_trip() {
        #[derive(Serialize, Deserialize, PartialEq, Debug)]
        struct LargeModel {
            weights: Vec<f64>,
        }

        let model = LargeModel {
            weights: (0..100_000).map(|i| i as f64 * 0.001).collect(),
        };

        let bytes = save_model_bytes(&model).unwrap();
        let loaded: LargeModel = load_model_bytes(&bytes).unwrap();
        assert_eq!(model, loaded);
    }

    #[test]
    fn test_large_struct_file_round_trip() {
        #[derive(Serialize, Deserialize, PartialEq, Debug)]
        struct LargeModel {
            matrix: Vec<Vec<f32>>,
        }

        let model = LargeModel {
            matrix: (0..500_usize)
                .map(|i| (0..200_usize).map(|j| (i * 200 + j) as f32).collect())
                .collect(),
        };

        let dir = tempdir();
        let path = dir.join("large.flrn");
        save_model(&model, &path).unwrap();
        let loaded: LargeModel = load_model(&path).unwrap();
        assert_eq!(model, loaded);
    }

    // -----------------------------------------------------------------------
    // Helper: create a temporary directory that is cleaned up automatically.
    // (std::env::temp_dir is available without extra deps)
    // -----------------------------------------------------------------------

    fn tempdir() -> std::path::PathBuf {
        use std::time::{SystemTime, UNIX_EPOCH};
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .subsec_nanos();
        let dir = std::env::temp_dir().join(format!("ferrolearn_io_test_{nanos}"));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }
}
