//! `fetch_kddcup99` — KDD Cup 1999 intrusion-detection dataset.
//!
//! Two variants on disk:
//! - Full (~743k samples)
//! - 10% subset (~494k samples)
//!
//! Both are CSVs with mixed numeric + categorical columns and a string label
//! at the end (e.g. `"normal."`, `"smurf."`). To stay numerical we expose the
//! parser as `parse_kddcup99_csv` returning string labels alongside numeric
//! columns; the high-level [`fetch_kddcup99`] returns numeric data + string
//! label vector.

use std::fs;
use std::io::Read;
use std::path::Path;

use ferrolearn_core::FerroError;
use flate2::read::GzDecoder;
use ndarray::Array2;

use crate::cache::dataset_dir;
use crate::fetch::{RemoteFile, fetch_file};

/// Full KDD Cup 99 archive.
pub const ARCHIVE_FULL: RemoteFile = RemoteFile {
    filename: "kddcup99_data",
    url: "https://ndownloader.figshare.com/files/5976045",
    sha256: "3b6c942aa0356c0ca35b7b595a26c89d343652c9db428893e7494f837b274292",
};

/// 10% subset.
pub const ARCHIVE_10PCT: RemoteFile = RemoteFile {
    filename: "kddcup99_10_data",
    url: "https://ndownloader.figshare.com/files/5976042",
    sha256: "8045aca0d84e70e622d1148d7df782496f6333bf6eb35a805a2cb4ddb1ec1422",
};

/// Returned dataset.
#[derive(Debug, Clone)]
pub struct KddCup99 {
    /// Numeric columns of the dataset (categorical columns are kept as
    /// f64-encoded indices into [`Self::categorical_levels`]).
    pub data: Array2<f64>,
    /// String label per row (e.g. "normal.", "smurf.").
    pub target: Vec<String>,
    /// Per-categorical-column distinct levels (in order of first appearance).
    pub categorical_levels: Vec<Vec<String>>,
    /// Index (within `data` columns) of each categorical column.
    pub categorical_columns: Vec<usize>,
}

/// Variant selector.
#[derive(Debug, Clone, Copy)]
pub enum KddSubset {
    /// Full dataset.
    Full,
    /// 10% subset.
    Percent10,
}

/// Fetch + parse KDD Cup 1999 (gzipped).
pub fn fetch_kddcup99(data_home: Option<&Path>, subset: KddSubset) -> Result<KddCup99, FerroError> {
    let archive = match subset {
        KddSubset::Full => ARCHIVE_FULL,
        KddSubset::Percent10 => ARCHIVE_10PCT,
    };
    let dir = dataset_dir("kddcup99", data_home)?;
    let path = fetch_file(archive.url, archive.filename, Some(archive.sha256), &dir)?;
    let bytes = fs::read(&path).map_err(FerroError::IoError)?;
    let mut text = String::new();
    GzDecoder::new(&bytes[..])
        .read_to_string(&mut text)
        .map_err(FerroError::IoError)?;
    parse_kddcup99_csv(&text)
}

fn parse_kddcup99_csv(raw: &str) -> Result<KddCup99, FerroError> {
    // Each line: 41 features + 1 label. Column 1 (protocol_type), 2
    // (service), 3 (flag) are categorical strings; everything else parses
    // as f64.
    const N_COLS: usize = 42;
    const CATEGORICAL: [usize; 3] = [1, 2, 3];

    let mut numeric_rows: Vec<Vec<f64>> = Vec::new();
    let mut labels: Vec<String> = Vec::new();
    let mut levels: Vec<Vec<String>> = vec![Vec::new(); CATEGORICAL.len()];

    for (lineno, line) in raw.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let parts: Vec<&str> = trimmed.split(',').collect();
        if parts.len() != N_COLS {
            return Err(FerroError::SerdeError {
                message: format!(
                    "kddcup99 line {} has {} columns (expected {N_COLS})",
                    lineno + 1,
                    parts.len()
                ),
            });
        }
        let mut row = Vec::with_capacity(N_COLS - 1);
        for (j, raw) in parts.iter().take(N_COLS - 1).enumerate() {
            if let Some(cat_idx) = CATEGORICAL.iter().position(|&c| c == j) {
                let val = (*raw).to_string();
                let pos = match levels[cat_idx].iter().position(|v| v == &val) {
                    Some(p) => p,
                    None => {
                        levels[cat_idx].push(val);
                        levels[cat_idx].len() - 1
                    }
                };
                row.push(pos as f64);
            } else {
                row.push(raw.parse::<f64>().map_err(|e| FerroError::SerdeError {
                    message: format!("kddcup99 line {} col {}: '{}' ({e})", lineno + 1, j, raw),
                })?);
            }
        }
        numeric_rows.push(row);
        labels.push(parts[N_COLS - 1].to_string());
    }
    let n = numeric_rows.len();
    let mut data = Array2::<f64>::zeros((n, N_COLS - 1));
    for (i, row) in numeric_rows.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            data[[i, j]] = v;
        }
    }
    Ok(KddCup99 {
        data,
        target: labels,
        categorical_levels: levels,
        categorical_columns: CATEGORICAL.to_vec(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_row(label: &str) -> String {
        // 41 features then label. Use 0 for numeric, "tcp"/"http"/"SF" for
        // categorical to match real-world values.
        let mut parts: Vec<String> = (0..41)
            .map(|i| match i {
                1 => "tcp".to_string(),
                2 => "http".to_string(),
                3 => "SF".to_string(),
                _ => "0".to_string(),
            })
            .collect();
        parts.push(label.to_string());
        parts.join(",")
    }

    #[test]
    fn parser_two_rows() {
        let raw = format!("{}\n{}\n", synth_row("normal."), synth_row("smurf."));
        let ds = parse_kddcup99_csv(&raw).unwrap();
        assert_eq!(ds.data.nrows(), 2);
        assert_eq!(ds.data.ncols(), 41);
        assert_eq!(ds.target, vec!["normal.".to_string(), "smurf.".to_string()]);
        assert_eq!(ds.categorical_columns, vec![1, 2, 3]);
        // All three categorical columns have one level after two identical rows.
        assert_eq!(ds.categorical_levels[0].len(), 1);
    }

    #[test]
    fn parser_rejects_wrong_column_count() {
        assert!(parse_kddcup99_csv("1,2,3,smurf.\n").is_err());
    }

    #[test]
    fn metadata_matches_sklearn() {
        assert_eq!(ARCHIVE_FULL.filename, "kddcup99_data");
        assert_eq!(ARCHIVE_10PCT.filename, "kddcup99_10_data");
    }
}
