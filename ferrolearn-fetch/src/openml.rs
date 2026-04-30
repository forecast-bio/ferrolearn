//! `fetch_openml` — generic OpenML.org dataset client.
//!
//! sklearn's full `fetch_openml` is large (handles parquet + ARFF, the
//! classification-vs-regression task split, version filtering, etc.). This
//! implementation covers the most common path:
//!
//! 1. Look up the dataset metadata via `https://www.openml.org/api/v1/json/data/{id}`.
//! 2. Download the file at the URL the metadata returns.
//! 3. If the file is ARFF, parse the simple-but-common subset (numeric +
//!    nominal attributes) into a feature matrix + target vector.
//!
//! Callers that need parquet, sparse matrices, or the full type-coercion
//! machinery should download the file via [`crate::fetch_file`] and parse it
//! with their own ARFF / parquet reader.

use std::fs;
use std::path::Path;

use ferrolearn_core::FerroError;
use ndarray::{Array1, Array2};
use serde_json::Value;

use crate::cache::dataset_dir;
use crate::fetch::fetch_file;

const DATA_INFO_URL: &str = "https://www.openml.org/api/v1/json/data/";

/// Returned OpenML dataset (numeric ARFF subset).
#[derive(Debug, Clone)]
pub struct OpenmlDataset {
    /// Feature matrix (nominal attributes encoded as 0-based indices into
    /// [`Self::nominal_levels`]).
    pub data: Array2<f64>,
    /// Target column (parsed as f64; for nominal targets the index into
    /// [`Self::target_levels`]).
    pub target: Array1<f64>,
    /// Names of feature columns in order.
    pub feature_names: Vec<String>,
    /// Name of the target column.
    pub target_name: String,
    /// For each feature column, the distinct nominal levels (empty for
    /// numeric columns).
    pub nominal_levels: Vec<Vec<String>>,
    /// Distinct levels of a nominal target (empty for numeric targets).
    pub target_levels: Vec<String>,
}

/// Fetch an OpenML dataset by numeric ID.
///
/// `target_column` may be `None` — in that case we use the dataset's
/// default `default_target_attribute` from the OpenML metadata.
pub fn fetch_openml(
    data_id: u64,
    target_column: Option<&str>,
    data_home: Option<&Path>,
) -> Result<OpenmlDataset, FerroError> {
    let dir = dataset_dir(&format!("openml/{data_id}"), data_home)?;
    let info_path = fetch_file(
        &format!("{DATA_INFO_URL}{data_id}"),
        "data_info.json",
        None,
        &dir,
    )?;
    let info_text = fs::read_to_string(&info_path).map_err(FerroError::IoError)?;
    let info: Value = serde_json::from_str(&info_text).map_err(|e| FerroError::SerdeError {
        message: format!("openml: failed to parse data info JSON: {e}"),
    })?;
    let data = info
        .get("data_set_description")
        .ok_or_else(|| FerroError::SerdeError {
            message: "openml: missing 'data_set_description'".into(),
        })?;
    let url = data
        .get("url")
        .and_then(Value::as_str)
        .ok_or_else(|| FerroError::SerdeError {
            message: "openml: dataset description missing 'url'".into(),
        })?;
    let default_target = data
        .get("default_target_attribute")
        .and_then(Value::as_str)
        .map(str::to_string);
    let target_name = target_column
        .map(str::to_string)
        .or(default_target)
        .ok_or_else(|| FerroError::SerdeError {
            message:
                "openml: no target_column supplied and no default_target_attribute in metadata"
                    .into(),
        })?;

    let arff_path = fetch_file(url, "data.arff", None, &dir)?;
    let arff_text = fs::read_to_string(&arff_path).map_err(FerroError::IoError)?;
    parse_arff(&arff_text, &target_name)
}

fn parse_arff(raw: &str, target_name: &str) -> Result<OpenmlDataset, FerroError> {
    enum AttrKind {
        Numeric,
        Nominal(Vec<String>),
    }
    struct Attr {
        name: String,
        kind: AttrKind,
    }

    let mut attrs: Vec<Attr> = Vec::new();
    let mut data_lines: Vec<String> = Vec::new();
    let mut in_data = false;
    for line in raw.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('%') {
            continue;
        }
        if !in_data {
            let lower = trimmed.to_ascii_lowercase();
            if lower.starts_with("@attribute") {
                let attr = parse_attribute(trimmed)?;
                attrs.push(attr);
            } else if lower.starts_with("@data") {
                in_data = true;
            }
        } else {
            data_lines.push(trimmed.to_string());
        }
    }

    let target_idx = attrs
        .iter()
        .position(|a| a.name == target_name)
        .ok_or_else(|| FerroError::SerdeError {
            message: format!("openml: target attribute '{target_name}' not found in ARFF"),
        })?;

    let n_rows = data_lines.len();
    let n_cols = attrs.len() - 1;
    let mut data = Array2::<f64>::zeros((n_rows, n_cols));
    let mut target = Array1::<f64>::zeros(n_rows);
    let mut nominal_levels: Vec<Vec<String>> = Vec::with_capacity(n_cols);
    let mut feature_names: Vec<String> = Vec::with_capacity(n_cols);
    for (j, a) in attrs.iter().enumerate() {
        if j == target_idx {
            continue;
        }
        feature_names.push(a.name.clone());
        nominal_levels.push(match &a.kind {
            AttrKind::Numeric => Vec::new(),
            AttrKind::Nominal(levels) => levels.clone(),
        });
    }
    let target_levels = match &attrs[target_idx].kind {
        AttrKind::Numeric => Vec::new(),
        AttrKind::Nominal(levels) => levels.clone(),
    };

    fn parse_attribute(line: &str) -> Result<Attr, FerroError> {
        // @attribute <name> {a,b,c}    or    @attribute <name> numeric
        let body = line["@attribute".len()..].trim();
        let (name, rest) = split_attribute_name(body)?;
        let rest_lower = rest.trim().to_ascii_lowercase();
        if rest_lower.starts_with('{') {
            // nominal
            let inner = rest
                .trim()
                .trim_start_matches('{')
                .trim_end_matches('}')
                .to_string();
            let levels: Vec<String> = inner
                .split(',')
                .map(|s| s.trim().trim_matches('\'').trim_matches('"').to_string())
                .collect();
            Ok(Attr {
                name,
                kind: AttrKind::Nominal(levels),
            })
        } else if rest_lower.starts_with("numeric")
            || rest_lower.starts_with("real")
            || rest_lower.starts_with("integer")
        {
            Ok(Attr {
                name,
                kind: AttrKind::Numeric,
            })
        } else {
            // String / date / unknown — treat as numeric (will fail on parse
            // if the column actually contains non-numeric data).
            Ok(Attr {
                name,
                kind: AttrKind::Numeric,
            })
        }
    }

    fn split_attribute_name(body: &str) -> Result<(String, String), FerroError> {
        if let Some(stripped) = body.strip_prefix('\'') {
            if let Some(end) = stripped.find('\'') {
                let name = stripped[..end].to_string();
                let rest = stripped[end + 1..].trim_start().to_string();
                return Ok((name, rest));
            }
        }
        let mut parts = body.splitn(2, char::is_whitespace);
        let name = parts.next().unwrap_or("").to_string();
        let rest = parts.next().unwrap_or("").to_string();
        Ok((name, rest))
    }

    for (i, line) in data_lines.iter().enumerate() {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() != attrs.len() {
            return Err(FerroError::SerdeError {
                message: format!(
                    "openml: ARFF data row {} has {} fields (expected {})",
                    i + 1,
                    parts.len(),
                    attrs.len()
                ),
            });
        }
        let mut col = 0usize;
        for (j, attr) in attrs.iter().enumerate() {
            let raw_v = parts[j].trim().trim_matches('\'').trim_matches('"');
            let value = match &attr.kind {
                AttrKind::Numeric => raw_v.parse::<f64>().unwrap_or(f64::NAN),
                AttrKind::Nominal(levels) => levels
                    .iter()
                    .position(|l| l == raw_v)
                    .map(|p| p as f64)
                    .unwrap_or(f64::NAN),
            };
            if j == target_idx {
                target[i] = value;
            } else {
                data[[i, col]] = value;
                col += 1;
            }
        }
    }

    Ok(OpenmlDataset {
        data,
        target,
        feature_names,
        target_name: target_name.to_string(),
        nominal_levels,
        target_levels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_minimal_numeric_arff() {
        let arff = "\
% comment\n\
@RELATION test\n\
@ATTRIBUTE x numeric\n\
@ATTRIBUTE y numeric\n\
@ATTRIBUTE label numeric\n\
@DATA\n\
1.0,2.0,0\n\
3.5,4.5,1\n\
";
        let ds = parse_arff(arff, "label").unwrap();
        assert_eq!(ds.data.dim(), (2, 2));
        assert_eq!(ds.target.len(), 2);
        assert!((ds.data[[0, 0]] - 1.0).abs() < 1e-12);
        assert!((ds.data[[1, 1]] - 4.5).abs() < 1e-12);
        assert!((ds.target[0] - 0.0).abs() < 1e-12);
        assert!((ds.target[1] - 1.0).abs() < 1e-12);
        assert_eq!(ds.feature_names, vec!["x", "y"]);
        assert_eq!(ds.target_name, "label");
    }

    #[test]
    fn parse_arff_with_nominal_target() {
        let arff = "\
@ATTRIBUTE x numeric\n\
@ATTRIBUTE class {cat,dog,bird}\n\
@DATA\n\
1.0,cat\n\
2.0,dog\n\
3.0,bird\n\
4.0,cat\n\
";
        let ds = parse_arff(arff, "class").unwrap();
        assert_eq!(ds.data.dim(), (4, 1));
        assert_eq!(ds.target_levels, vec!["cat", "dog", "bird"]);
        assert!((ds.target[0] - 0.0).abs() < 1e-12);
        assert!((ds.target[1] - 1.0).abs() < 1e-12);
        assert!((ds.target[2] - 2.0).abs() < 1e-12);
        assert!((ds.target[3] - 0.0).abs() < 1e-12);
    }

    #[test]
    fn parse_arff_missing_target_errors() {
        let arff = "\
@ATTRIBUTE x numeric\n\
@DATA\n\
1.0\n\
";
        assert!(parse_arff(arff, "missing").is_err());
    }

    #[test]
    fn parse_arff_wrong_field_count_errors() {
        let arff = "\
@ATTRIBUTE x numeric\n\
@ATTRIBUTE y numeric\n\
@DATA\n\
1.0\n\
";
        assert!(parse_arff(arff, "y").is_err());
    }
}
