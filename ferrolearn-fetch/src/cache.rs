//! On-disk cache management.
//!
//! Mirrors `sklearn.datasets.get_data_home` / `clear_data_home`. The default
//! cache lives in `<dirs::data_local_dir>/ferrolearn_data/` (e.g.
//! `~/.local/share/ferrolearn_data/` on Linux); override via the
//! `FERROLEARN_DATA` environment variable or by passing `data_home` to the
//! fetcher.

use ferrolearn_core::FerroError;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

/// Return the configured data-home directory, creating it if it doesn't
/// already exist.
///
/// Resolution order:
/// 1. `data_home` argument (if `Some`).
/// 2. `FERROLEARN_DATA` environment variable.
/// 3. `dirs::data_local_dir()/ferrolearn_data/`.
/// 4. Fallback `./ferrolearn_data/` if no platform dir is available.
///
/// # Errors
///
/// Returns [`FerroError::IoError`] if the directory cannot be created.
pub fn get_data_home(data_home: Option<&Path>) -> Result<PathBuf, FerroError> {
    let path = if let Some(p) = data_home {
        p.to_path_buf()
    } else if let Ok(env) = env::var("FERROLEARN_DATA") {
        PathBuf::from(env)
    } else if let Some(base) = dirs::data_local_dir() {
        base.join("ferrolearn_data")
    } else {
        PathBuf::from("./ferrolearn_data")
    };
    fs::create_dir_all(&path).map_err(FerroError::IoError)?;
    Ok(path)
}

/// Recursively delete the data-home directory.
///
/// # Errors
///
/// Returns [`FerroError::IoError`] if the directory cannot be removed.
pub fn clear_data_home(data_home: Option<&Path>) -> Result<(), FerroError> {
    let path = get_data_home(data_home)?;
    if path.exists() {
        fs::remove_dir_all(&path).map_err(FerroError::IoError)?;
    }
    Ok(())
}

/// Return the dataset-specific subdirectory under `data_home`, creating it.
pub fn dataset_dir(name: &str, data_home: Option<&Path>) -> Result<PathBuf, FerroError> {
    let base = get_data_home(data_home)?;
    let dir = base.join(name);
    fs::create_dir_all(&dir).map_err(FerroError::IoError)?;
    Ok(dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_tmp() -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .subsec_nanos();
        std::env::temp_dir().join(format!("ferrolearn_fetch_test_{nanos}"))
    }

    #[test]
    fn explicit_path_used_and_created() {
        let p = unique_tmp();
        let resolved = get_data_home(Some(&p)).unwrap();
        assert_eq!(resolved, p);
        assert!(p.is_dir());
        let _ = fs::remove_dir_all(&p);
    }

    #[test]
    fn clear_data_home_removes_explicit_dir() {
        let p = unique_tmp();
        get_data_home(Some(&p)).unwrap();
        clear_data_home(Some(&p)).unwrap();
        assert!(!p.exists());
    }

    #[test]
    fn dataset_dir_creates_subdir() {
        let p = unique_tmp();
        let sub = dataset_dir("california_housing", Some(&p)).unwrap();
        assert_eq!(sub, p.join("california_housing"));
        assert!(sub.is_dir());
        let _ = fs::remove_dir_all(&p);
    }
}
