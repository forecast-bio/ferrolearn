//! HTTP-fetch primitives shared by the dataset-specific loaders.
//!
//! Sklearn's `RemoteFileMetadata` translates to [`RemoteFile`] here. Fetches
//! are cached on disk and SHA-256 verified against the bundled checksum.

use ferrolearn_core::FerroError;
use sha2::{Digest, Sha256};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

/// Metadata for a single network-fetched file.
#[derive(Debug, Clone)]
pub struct RemoteFile {
    /// Final filename inside the cache directory.
    pub filename: &'static str,
    /// Source URL.
    pub url: &'static str,
    /// Hex-encoded SHA-256 of the file at that URL (lower-case).
    pub sha256: &'static str,
}

/// Generic single-URL fetcher (sklearn's `fetch_file`).
///
/// Downloads `url` into `<dataset_dir>/<filename>` (creating the directory
/// if needed) and returns the absolute path. If the file already exists and
/// its SHA-256 matches `sha256`, the download is skipped.
///
/// # Errors
///
/// Returns [`FerroError::IoError`] for filesystem failures.
/// Returns [`FerroError::SerdeError`] (re-using its diagnostic slot) for
/// network or checksum failures.
pub fn fetch_file(
    url: &str,
    filename: &str,
    sha256: Option<&str>,
    dataset_dir: &Path,
) -> Result<PathBuf, FerroError> {
    fs::create_dir_all(dataset_dir).map_err(FerroError::IoError)?;
    let path = dataset_dir.join(filename);
    if path.is_file() {
        if let Some(expected) = sha256 {
            if verify_sha256(&path, expected)? {
                return Ok(path);
            }
            // Cached file is corrupt — fall through and re-download.
        } else {
            return Ok(path);
        }
    }
    download(url, &path)?;
    if let Some(expected) = sha256
        && !verify_sha256(&path, expected)?
    {
        // Wipe the bad file before reporting.
        let _ = fs::remove_file(&path);
        return Err(FerroError::SerdeError {
            message: format!(
                "ferrolearn-fetch: SHA-256 mismatch after downloading {url} (expected {expected})"
            ),
        });
    }
    Ok(path)
}

/// Download `url` to `dest` using a streaming HTTP GET.
fn download(url: &str, dest: &Path) -> Result<(), FerroError> {
    let response = ureq::get(url).call().map_err(|e| FerroError::SerdeError {
        message: format!("ferrolearn-fetch: HTTP GET {url} failed: {e}"),
    })?;
    let mut body = response.into_body();
    let mut reader = body.as_reader();
    let mut bytes = Vec::new();
    reader
        .read_to_end(&mut bytes)
        .map_err(|e| FerroError::SerdeError {
            message: format!("ferrolearn-fetch: reading body of {url} failed: {e}"),
        })?;
    fs::write(dest, &bytes).map_err(FerroError::IoError)?;
    Ok(())
}

fn verify_sha256(path: &Path, expected: &str) -> Result<bool, FerroError> {
    let bytes = fs::read(path).map_err(FerroError::IoError)?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    let digest = hasher.finalize();
    let hex = digest
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    Ok(hex.eq_ignore_ascii_case(expected))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn tmp_dir() -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .subsec_nanos();
        let p = std::env::temp_dir().join(format!("ferrolearn_fetch_inner_{nanos}"));
        fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn sha256_verifies_known_content() {
        let dir = tmp_dir();
        let path = dir.join("hello.txt");
        fs::write(&path, b"hello world").unwrap();
        // sha256("hello world") = b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9
        assert!(
            verify_sha256(
                &path,
                "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
            )
            .unwrap()
        );
        assert!(!verify_sha256(&path, &"0".repeat(64)).unwrap());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn sha256_case_insensitive() {
        let dir = tmp_dir();
        let path = dir.join("hello.txt");
        fs::write(&path, b"hello world").unwrap();
        assert!(
            verify_sha256(
                &path,
                "B94D27B9934D3E08A52E52D7DA7DABFAC484EFE37A5380EE9088F7ACE2EFCDE9"
            )
            .unwrap()
        );
        let _ = fs::remove_dir_all(&dir);
    }
}
