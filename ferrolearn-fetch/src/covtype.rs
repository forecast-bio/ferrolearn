//! `fetch_covtype` — Forest Covertype dataset.
//!
//! 581012 samples × 54 features (10 quantitative + 44 binary indicators),
//! 7-class target. Upstream is a single gzipped CSV.

use std::fs;
use std::io::Read;
use std::path::Path;

use ferrolearn_core::FerroError;
use flate2::read::GzDecoder;
use ndarray::{Array1, Array2};

use crate::cache::dataset_dir;
use crate::fetch::{RemoteFile, fetch_file};

/// URL/checksum copied from sklearn `_covtype.py`.
pub const ARCHIVE: RemoteFile = RemoteFile {
    filename: "covtype.data.gz",
    url: "https://ndownloader.figshare.com/files/5976039",
    sha256: "614360d0257557dd1792834a85a1cdebfadc3c4f30b011d56afee7ffb5b15771",
};

/// Returned dataset.
#[derive(Debug, Clone)]
pub struct Covtype {
    /// Feature matrix `(n_samples, 54)`.
    pub data: Array2<f64>,
    /// Class labels in `1..=7` (sklearn convention).
    pub target: Array1<usize>,
}

/// Fetch + parse the covertype dataset.
///
/// # Errors
///
/// Propagates cache, download, gunzip, or parse failures.
pub fn fetch_covtype(data_home: Option<&Path>) -> Result<Covtype, FerroError> {
    let dir = dataset_dir("covtype", data_home)?;
    let gz = fetch_file(ARCHIVE.url, ARCHIVE.filename, Some(ARCHIVE.sha256), &dir)?;
    let bytes = fs::read(&gz).map_err(FerroError::IoError)?;
    let mut text = String::new();
    GzDecoder::new(&bytes[..])
        .read_to_string(&mut text)
        .map_err(FerroError::IoError)?;
    parse_covtype_csv(&text)
}

fn parse_covtype_csv(raw: &str) -> Result<Covtype, FerroError> {
    // 55 columns: 54 features + 1 label
    let mut rows: Vec<Vec<f64>> = Vec::new();
    for (lineno, line) in raw.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let parts: Vec<&str> = trimmed.split(',').collect();
        if parts.len() != 55 {
            return Err(FerroError::SerdeError {
                message: format!(
                    "covtype line {} has {} columns (expected 55)",
                    lineno + 1,
                    parts.len()
                ),
            });
        }
        let mut row = Vec::with_capacity(55);
        for (i, p) in parts.iter().enumerate() {
            row.push(p.parse::<f64>().map_err(|e| FerroError::SerdeError {
                message: format!("covtype line {} col {}: '{}' ({e})", lineno + 1, i, p),
            })?);
        }
        rows.push(row);
    }
    let n = rows.len();
    let mut data = Array2::<f64>::zeros((n, 54));
    let mut target = Array1::<usize>::zeros(n);
    for (i, row) in rows.iter().enumerate() {
        for j in 0..54 {
            data[[i, j]] = row[j];
        }
        target[i] = row[54] as usize;
    }
    Ok(Covtype { data, target })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parser_two_rows() {
        let mut row = vec!["0.0"; 54].join(",");
        row.push_str(",3");
        let raw = format!("{row}\n{row}\n");
        let ds = parse_covtype_csv(&raw).unwrap();
        assert_eq!(ds.data.dim(), (2, 54));
        assert_eq!(ds.target.len(), 2);
        assert_eq!(ds.target[0], 3);
    }

    #[test]
    fn parser_rejects_wrong_column_count() {
        let raw = "1,2,3\n";
        assert!(parse_covtype_csv(raw).is_err());
    }

    #[test]
    fn metadata_matches_sklearn() {
        assert_eq!(ARCHIVE.filename, "covtype.data.gz");
        assert_eq!(ARCHIVE.sha256.len(), 64);
    }
}
