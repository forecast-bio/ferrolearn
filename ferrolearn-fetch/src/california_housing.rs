//! `fetch_california_housing` — 20640×8 regression benchmark.
//!
//! The original sklearn dataset is hosted by figshare as a tarball. We
//! download it, extract `cal_housing.data`, then parse the comma-separated
//! values into a feature matrix + target vector.
//!
//! Feature columns (sklearn order, after re-arranging the raw file):
//! `MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude,
//! Longitude`. Target: `MedHouseVal` (median house value, 100k USD units).

use std::fs;
use std::path::{Path, PathBuf};

use ferrolearn_core::FerroError;
use flate2::read::GzDecoder;
use ndarray::{Array1, Array2};
use tar::Archive;

use crate::cache::dataset_dir;
use crate::fetch::{RemoteFile, fetch_file};

/// URL/checksum copied from sklearn `_california_housing.py`.
pub const ARCHIVE: RemoteFile = RemoteFile {
    filename: "cal_housing.tgz",
    url: "https://ndownloader.figshare.com/files/5976036",
    sha256: "aaa5c9a6afe2225cc2aed2723682ae403280c4a3695a2ddda4ffb5d8215ea681",
};

/// Returned dataset.
#[derive(Debug, Clone)]
pub struct CaliforniaHousing {
    /// Feature matrix `(20640, 8)`.
    pub data: Array2<f64>,
    /// Median house value targets.
    pub target: Array1<f64>,
    /// Column names in order.
    pub feature_names: Vec<&'static str>,
    /// Target variable name (single).
    pub target_names: Vec<&'static str>,
}

/// Download (or load from cache) the California housing dataset and return
/// the parsed `(data, target)` arrays.
///
/// Sklearn's column order:
///
/// `MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude`
///
/// derived from raw columns of the upstream tarball:
///
/// `longitude, latitude, housingMedianAge, totalRooms, totalBedrooms,
/// population, households, medianIncome, medianHouseValue`.
///
/// # Errors
///
/// Returns [`FerroError`] for any cache, download, archive, parsing, or
/// SHA mismatch failure.
pub fn fetch_california_housing(data_home: Option<&Path>) -> Result<CaliforniaHousing, FerroError> {
    let dir = dataset_dir("california_housing", data_home)?;
    let archive_path = fetch_file(ARCHIVE.url, ARCHIVE.filename, Some(ARCHIVE.sha256), &dir)?;

    let csv_path = extract_data_file(&archive_path, &dir)?;
    let raw = fs::read_to_string(&csv_path).map_err(FerroError::IoError)?;
    parse_california_csv(&raw)
}

fn extract_data_file(archive_path: &Path, dir: &Path) -> Result<PathBuf, FerroError> {
    // The tarball contains CaliforniaHousing/cal_housing.data
    let target = dir.join("cal_housing.data");
    if target.is_file() {
        return Ok(target);
    }
    let f = fs::File::open(archive_path).map_err(FerroError::IoError)?;
    let mut archive = Archive::new(GzDecoder::new(f));
    for entry in archive.entries().map_err(FerroError::IoError)? {
        let mut entry = entry.map_err(FerroError::IoError)?;
        let entry_path = entry.path().map_err(FerroError::IoError)?.into_owned();
        if entry_path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n == "cal_housing.data")
        {
            entry.unpack(&target).map_err(FerroError::IoError)?;
            return Ok(target);
        }
    }
    Err(FerroError::SerdeError {
        message: "ferrolearn-fetch: cal_housing.data not found in archive".into(),
    })
}

fn parse_california_csv(raw: &str) -> Result<CaliforniaHousing, FerroError> {
    // Raw column order:
    // 0:longitude 1:latitude 2:housingMedianAge 3:totalRooms 4:totalBedrooms
    // 5:population 6:households 7:medianIncome 8:medianHouseValue
    let mut samples: Vec<[f64; 9]> = Vec::with_capacity(20640);
    for (lineno, line) in raw.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let parts: Vec<&str> = trimmed.split(',').collect();
        if parts.len() != 9 {
            return Err(FerroError::SerdeError {
                message: format!(
                    "california_housing line {} has {} columns (expected 9)",
                    lineno + 1,
                    parts.len()
                ),
            });
        }
        let mut row = [0.0_f64; 9];
        for (i, p) in parts.iter().enumerate() {
            row[i] = p.parse::<f64>().map_err(|e| FerroError::SerdeError {
                message: format!(
                    "california_housing line {} col {}: '{}' not a number ({e})",
                    lineno + 1,
                    i,
                    p
                ),
            })?;
        }
        samples.push(row);
    }
    let n = samples.len();
    let mut data = Array2::<f64>::zeros((n, 8));
    let mut target = Array1::<f64>::zeros(n);
    for (i, row) in samples.iter().enumerate() {
        let households = row[6].max(1.0);
        let pop = row[5];
        let total_rooms = row[3];
        let total_bedrooms = row[4];
        // sklearn-derived columns:
        let med_inc = row[7];
        let house_age = row[2];
        let ave_rooms = total_rooms / households;
        let ave_bedrms = total_bedrooms / households;
        let ave_occup = pop / households;
        let latitude = row[1];
        let longitude = row[0];
        let med_house_val = row[8];
        data[[i, 0]] = med_inc;
        data[[i, 1]] = house_age;
        data[[i, 2]] = ave_rooms;
        data[[i, 3]] = ave_bedrms;
        data[[i, 4]] = pop;
        data[[i, 5]] = ave_occup;
        data[[i, 6]] = latitude;
        data[[i, 7]] = longitude;
        // Target is in dollars; sklearn rescales to 100k USD.
        target[i] = med_house_val / 100_000.0;
    }
    Ok(CaliforniaHousing {
        data,
        target,
        feature_names: vec![
            "MedInc",
            "HouseAge",
            "AveRooms",
            "AveBedrms",
            "Population",
            "AveOccup",
            "Latitude",
            "Longitude",
        ],
        target_names: vec!["MedHouseVal"],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parser_handles_synthetic_two_rows() {
        // Two synthetic rows in raw column order.
        let raw = "\
            -122.0,37.0,30.0,1000.0,200.0,400.0,100.0,5.0,250000.0\n\
            -121.0,38.0,20.0,2000.0,400.0,800.0,200.0,7.0,500000.0\n\
        ";
        let ds = parse_california_csv(raw).unwrap();
        assert_eq!(ds.data.dim(), (2, 8));
        assert_eq!(ds.target.len(), 2);
        // Row 0: MedInc=5.0, HouseAge=30.0, AveRooms=1000/100=10, AveBedrms=2,
        // Population=400, AveOccup=4, Latitude=37, Longitude=-122
        assert!((ds.data[[0, 0]] - 5.0).abs() < 1e-12);
        assert!((ds.data[[0, 1]] - 30.0).abs() < 1e-12);
        assert!((ds.data[[0, 2]] - 10.0).abs() < 1e-12);
        assert!((ds.data[[0, 3]] - 2.0).abs() < 1e-12);
        assert!((ds.data[[0, 4]] - 400.0).abs() < 1e-12);
        assert!((ds.data[[0, 5]] - 4.0).abs() < 1e-12);
        assert!((ds.data[[0, 6]] - 37.0).abs() < 1e-12);
        assert!((ds.data[[0, 7]] - (-122.0)).abs() < 1e-12);
        // target: 250000 / 100000 = 2.5
        assert!((ds.target[0] - 2.5).abs() < 1e-12);
        assert!((ds.target[1] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn parser_rejects_short_row() {
        let raw = "1,2,3,4\n";
        assert!(parse_california_csv(raw).is_err());
    }

    #[test]
    fn parser_rejects_non_numeric() {
        let raw = "1,2,3,4,5,6,7,abc,9\n";
        assert!(parse_california_csv(raw).is_err());
    }

    #[test]
    fn metadata_constants_match_sklearn() {
        assert_eq!(ARCHIVE.filename, "cal_housing.tgz");
        assert!(ARCHIVE.url.contains("figshare.com"));
        assert_eq!(ARCHIVE.sha256.len(), 64);
    }
}
