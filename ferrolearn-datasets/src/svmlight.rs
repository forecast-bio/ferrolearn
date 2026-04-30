//! SVMlight / LIBSVM sparse text format I/O.
//!
//! The SVMlight format is one line per sample:
//!
//! ```text
//! <label> <feat>:<value> <feat>:<value> ...
//! ```
//!
//! - Feature indices are 1-based on disk (we expose them 0-based in Rust).
//! - Lines may include a trailing comment after `#`.
//! - Blank lines and lines starting with `#` are ignored.
//!
//! For simplicity this module returns dense [`ndarray::Array2<F>`] feature
//! matrices, which is adequate for small/medium datasets. For very large
//! sparse data, prefer streaming directly into a [`ferrolearn_sparse`]
//! matrix (out of scope here).

use ferrolearn_core::FerroError;
use ndarray::{Array1, Array2};
use std::fs;
use std::io::Write;
use std::path::Path;

/// Result of [`load_svmlight_file`] / [`load_svmlight_str`]: dense feature
/// matrix paired with a label vector.
pub type SvmlightDataset = (Array2<f64>, Array1<f64>);

/// Result of [`load_files`]: documents, labels, target names.
pub type LoadFilesResult = (Vec<String>, Array1<usize>, Vec<String>);

/// Parse one SVMlight line into `(label, [(feature, value), ...])`.
fn parse_line(line: &str) -> Result<(f64, Vec<(usize, f64)>), FerroError> {
    // Strip optional comment.
    let body = match line.find('#') {
        Some(i) => &line[..i],
        None => line,
    };
    let trimmed = body.trim();
    if trimmed.is_empty() {
        return Err(FerroError::InvalidParameter {
            name: "line".into(),
            reason: "svmlight: empty line".into(),
        });
    }
    let mut parts = trimmed.split_whitespace();
    let label_tok = parts.next().ok_or_else(|| FerroError::InvalidParameter {
        name: "label".into(),
        reason: "svmlight: missing label".into(),
    })?;
    let label: f64 =
        label_tok.parse().map_err(
            |e: std::num::ParseFloatError| FerroError::InvalidParameter {
                name: "label".into(),
                reason: format!("svmlight: bad label '{label_tok}': {e}"),
            },
        )?;
    let mut features = Vec::new();
    for tok in parts {
        let mut split = tok.splitn(2, ':');
        let idx_tok = split.next().ok_or_else(|| FerroError::InvalidParameter {
            name: "feature".into(),
            reason: format!("svmlight: malformed token '{tok}'"),
        })?;
        let val_tok = split.next().ok_or_else(|| FerroError::InvalidParameter {
            name: "feature".into(),
            reason: format!("svmlight: malformed token '{tok}' (no value)"),
        })?;
        let idx_one_based: usize =
            idx_tok
                .parse()
                .map_err(|e: std::num::ParseIntError| FerroError::InvalidParameter {
                    name: "feature index".into(),
                    reason: format!("svmlight: bad index '{idx_tok}': {e}"),
                })?;
        if idx_one_based == 0 {
            return Err(FerroError::InvalidParameter {
                name: "feature index".into(),
                reason: "svmlight: feature indices are 1-based, got 0".into(),
            });
        }
        let val: f64 = val_tok.parse().map_err(|e: std::num::ParseFloatError| {
            FerroError::InvalidParameter {
                name: "feature value".into(),
                reason: format!("svmlight: bad value '{val_tok}': {e}"),
            }
        })?;
        features.push((idx_one_based - 1, val));
    }
    Ok((label, features))
}

/// Load an SVMlight-format file into a dense `(X, y)` pair.
///
/// `n_features` may be `None`, in which case the number of features is
/// inferred from the maximum index seen.
///
/// # Errors
///
/// Returns [`FerroError::IoError`] if the file cannot be read.
/// Returns [`FerroError::InvalidParameter`] for malformed lines.
pub fn load_svmlight_file<P: AsRef<Path>>(
    path: P,
    n_features: Option<usize>,
) -> Result<SvmlightDataset, FerroError> {
    let text = fs::read_to_string(path).map_err(FerroError::IoError)?;
    load_svmlight_str(&text, n_features)
}

/// Like [`load_svmlight_file`] but takes the file contents as a string.
pub fn load_svmlight_str(
    contents: &str,
    n_features: Option<usize>,
) -> Result<SvmlightDataset, FerroError> {
    let mut rows: Vec<(f64, Vec<(usize, f64)>)> = Vec::new();
    let mut max_feat = 0usize;
    for (lineno, raw) in contents.lines().enumerate() {
        let stripped = raw.trim();
        if stripped.is_empty() || stripped.starts_with('#') {
            continue;
        }
        let parsed = parse_line(raw).map_err(|e| FerroError::InvalidParameter {
            name: "svmlight".into(),
            reason: format!("line {}: {e}", lineno + 1),
        })?;
        for &(i, _) in &parsed.1 {
            if i + 1 > max_feat {
                max_feat = i + 1;
            }
        }
        rows.push(parsed);
    }
    let n_samples = rows.len();
    let n_feat = match n_features {
        Some(n) => {
            if n < max_feat {
                return Err(FerroError::InvalidParameter {
                    name: "n_features".into(),
                    reason: format!(
                        "svmlight: declared n_features={n} but file has indices up to {max_feat}"
                    ),
                });
            }
            n
        }
        None => max_feat,
    };
    let mut x = Array2::<f64>::zeros((n_samples, n_feat));
    let mut y = Array1::<f64>::zeros(n_samples);
    for (i, (label, feats)) in rows.into_iter().enumerate() {
        y[i] = label;
        for (j, v) in feats {
            x[[i, j]] = v;
        }
    }
    Ok((x, y))
}

/// Load multiple SVMlight files at once. Returns one `(X, y)` per path,
/// inferring `n_features` per file unless `n_features` is provided.
pub fn load_svmlight_files<P: AsRef<Path>>(
    paths: &[P],
    n_features: Option<usize>,
) -> Result<Vec<SvmlightDataset>, FerroError> {
    paths
        .iter()
        .map(|p| load_svmlight_file(p.as_ref(), n_features))
        .collect()
}

/// Write a feature matrix `x` and labels `y` to `path` in SVMlight format.
///
/// Sparse representation on disk: only entries `|v| > 0` are written.
///
/// # Errors
///
/// Returns [`FerroError::IoError`] if the file cannot be opened or written.
/// Returns [`FerroError::ShapeMismatch`] if `x.nrows() != y.len()`.
pub fn dump_svmlight_file<P: AsRef<Path>>(
    x: &Array2<f64>,
    y: &Array1<f64>,
    path: P,
) -> Result<(), FerroError> {
    let n_samples = x.nrows();
    if y.len() != n_samples {
        return Err(FerroError::ShapeMismatch {
            expected: vec![n_samples],
            actual: vec![y.len()],
            context: "dump_svmlight_file: y length must equal x rows".into(),
        });
    }
    let mut buf = String::new();
    for i in 0..n_samples {
        buf.push_str(&format!("{}", y[i]));
        for j in 0..x.ncols() {
            let v = x[[i, j]];
            if v != 0.0 {
                buf.push(' ');
                buf.push_str(&format!("{}:{}", j + 1, v));
            }
        }
        buf.push('\n');
    }
    let mut f = fs::File::create(path).map_err(FerroError::IoError)?;
    f.write_all(buf.as_bytes()).map_err(FerroError::IoError)?;
    Ok(())
}

/// Recursively load a directory tree of text files: each subdirectory becomes
/// a class label and every regular file inside is treated as one document.
///
/// Returns `(documents, labels, target_names)` where `documents[i]` is the
/// raw text of file `i`, `labels[i]` is the (0-based) index of its parent
/// directory in `target_names`, and `target_names[k]` is the directory name
/// for class `k`.
///
/// # Errors
///
/// Returns [`FerroError::IoError`] if the root directory or any file inside
/// cannot be read.
pub fn load_files<P: AsRef<Path>>(container_path: P) -> Result<LoadFilesResult, FerroError> {
    let root = container_path.as_ref();
    let mut subdirs: Vec<std::path::PathBuf> = fs::read_dir(root)
        .map_err(FerroError::IoError)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    subdirs.sort();
    let target_names: Vec<String> = subdirs
        .iter()
        .map(|p| {
            p.file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_owned()
        })
        .collect();
    let mut docs: Vec<String> = Vec::new();
    let mut labels: Vec<usize> = Vec::new();
    for (cls_idx, dir) in subdirs.iter().enumerate() {
        let mut files: Vec<std::path::PathBuf> = fs::read_dir(dir)
            .map_err(FerroError::IoError)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.is_file())
            .collect();
        files.sort();
        for f in files {
            let text = fs::read_to_string(&f).map_err(FerroError::IoError)?;
            docs.push(text);
            labels.push(cls_idx);
        }
    }
    Ok((docs, Array1::from(labels), target_names))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn tmpdir() -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .subsec_nanos();
        let dir = std::env::temp_dir().join(format!("ferrolearn_svmlight_test_{nanos}"));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn parse_basic_line() {
        let (label, feats) = parse_line("1.0 1:0.5 3:1.5").unwrap();
        assert!((label - 1.0).abs() < 1e-12);
        assert_eq!(feats, vec![(0, 0.5), (2, 1.5)]);
    }

    #[test]
    fn parse_with_comment() {
        let (label, feats) = parse_line("0 1:1.0 2:2.0 # this is a comment").unwrap();
        assert!((label - 0.0).abs() < 1e-12);
        assert_eq!(feats, vec![(0, 1.0), (1, 2.0)]);
    }

    #[test]
    fn round_trip_dense() {
        let dir = tmpdir();
        let path = dir.join("a.svmlight");
        let x = ndarray::array![[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 5.0, 6.0]];
        let y = ndarray::array![1.0, 0.0, 1.0];
        dump_svmlight_file(&x, &y, &path).unwrap();
        let (x2, y2) = load_svmlight_file(&path, None).unwrap();
        assert_eq!(x2.dim(), (3, 3));
        assert_eq!(y2.len(), 3);
        for i in 0..3 {
            for j in 0..3 {
                assert!((x[[i, j]] - x2[[i, j]]).abs() < 1e-12);
            }
            assert!((y[i] - y2[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn declared_n_features_pads() {
        let text = "1 1:1.0 2:2.0\n0 1:3.0\n";
        let (x, _) = load_svmlight_str(text, Some(5)).unwrap();
        assert_eq!(x.ncols(), 5);
    }

    #[test]
    fn declared_n_features_too_small() {
        let text = "1 1:1.0 5:9.0\n";
        assert!(load_svmlight_str(text, Some(2)).is_err());
    }

    #[test]
    fn shape_mismatch_dump() {
        let x = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
        let y = ndarray::array![1.0];
        let dir = tmpdir();
        assert!(dump_svmlight_file(&x, &y, dir.join("bad.svmlight")).is_err());
    }

    #[test]
    fn load_files_simple_tree() {
        let dir = tmpdir();
        for cls in &["alpha", "beta"] {
            let sub = dir.join(cls);
            fs::create_dir_all(&sub).unwrap();
            for i in 0..2 {
                let f = sub.join(format!("doc{i}.txt"));
                fs::write(&f, format!("{cls}-doc{i}")).unwrap();
            }
        }
        let (docs, labels, target_names) = load_files(&dir).unwrap();
        assert_eq!(docs.len(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(target_names, vec!["alpha".to_string(), "beta".to_string()]);
    }
}
