//! `fetch_20newsgroups` — text classification benchmark, 20 categories.
//!
//! The upstream tarball expands into:
//!
//! ```text
//! 20news-bydate-train/<category>/<doc>
//! 20news-bydate-test/<category>/<doc>
//! ```
//!
//! We extract the chosen `subset` ("train", "test", or "all") and reuse
//! [`ferrolearn_datasets::load_files`] to build the `(documents, labels,
//! target_names)` triple.

use std::fs;
use std::path::Path;

use ferrolearn_core::FerroError;
use ferrolearn_datasets::svmlight::{LoadFilesResult, load_files};
use flate2::read::GzDecoder;
use tar::Archive;

use crate::cache::dataset_dir;
use crate::fetch::{RemoteFile, fetch_file};

/// URL/checksum copied from sklearn `_twenty_newsgroups.py`.
pub const ARCHIVE: RemoteFile = RemoteFile {
    filename: "20news-bydate.tar.gz",
    url: "https://ndownloader.figshare.com/files/5975967",
    sha256: "8f1b2514ca22a5ade8fbb9cfa5727df95fa587f4c87b786e15c759fa66d95610",
};

/// Subset selector.
#[derive(Debug, Clone, Copy)]
pub enum NewsgroupsSubset {
    /// Training partition.
    Train,
    /// Held-out test partition.
    Test,
    /// Train + test concatenated.
    All,
}

/// Fetch + extract the 20-newsgroups dataset and return
/// `(documents, labels, target_names)` for the chosen subset.
pub fn fetch_20newsgroups(
    data_home: Option<&Path>,
    subset: NewsgroupsSubset,
) -> Result<LoadFilesResult, FerroError> {
    let dir = dataset_dir("20newsgroups", data_home)?;
    let archive_path = fetch_file(ARCHIVE.url, ARCHIVE.filename, Some(ARCHIVE.sha256), &dir)?;
    let extract_root = dir.join("extracted");
    if !extract_root.exists() {
        extract_archive(&archive_path, &extract_root)?;
    }

    match subset {
        NewsgroupsSubset::Train => load_files(extract_root.join("20news-bydate-train")),
        NewsgroupsSubset::Test => load_files(extract_root.join("20news-bydate-test")),
        NewsgroupsSubset::All => {
            let (mut docs_a, mut labels_a, names_a) =
                load_files(extract_root.join("20news-bydate-train"))?;
            let (docs_b, labels_b, names_b) = load_files(extract_root.join("20news-bydate-test"))?;
            // Class names should match across train/test; if they don't,
            // we report it loudly rather than silently merging mismatched
            // label spaces.
            if names_a != names_b {
                return Err(FerroError::SerdeError {
                    message: "20newsgroups: train and test subsets have different class names"
                        .into(),
                });
            }
            docs_a.extend(docs_b);
            // Append labels — labels_b indices already match the same name
            // ordering since names match.
            let mut out_labels = labels_a.to_vec();
            out_labels.extend(labels_b.iter().copied());
            labels_a = ndarray::Array1::from(out_labels);
            Ok((docs_a, labels_a, names_a))
        }
    }
}

fn extract_archive(archive_path: &Path, dest: &Path) -> Result<(), FerroError> {
    fs::create_dir_all(dest).map_err(FerroError::IoError)?;
    let f = fs::File::open(archive_path).map_err(FerroError::IoError)?;
    let mut archive = Archive::new(GzDecoder::new(f));
    archive.unpack(dest).map_err(FerroError::IoError)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_tmp() -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .subsec_nanos();
        std::env::temp_dir().join(format!("ferrolearn_news_test_{nanos}"))
    }

    #[test]
    fn metadata_matches_sklearn() {
        assert_eq!(ARCHIVE.filename, "20news-bydate.tar.gz");
        assert_eq!(ARCHIVE.sha256.len(), 64);
    }

    #[test]
    fn extract_archive_round_trip_synthetic() {
        // Build a tiny tar.gz, extract it, then assert load_files works
        // on the resulting tree.
        use std::io::Write;
        let dir = unique_tmp();
        fs::create_dir_all(&dir).unwrap();
        let archive_path = dir.join("synthetic.tar.gz");
        {
            let f = fs::File::create(&archive_path).unwrap();
            let gz = flate2::write::GzEncoder::new(f, flate2::Compression::default());
            let mut tar_b = tar::Builder::new(gz);
            // Add file alpha/doc0.txt
            let alpha_dir = dir.join("staging").join("alpha");
            fs::create_dir_all(&alpha_dir).unwrap();
            let doc = alpha_dir.join("doc0.txt");
            let mut fh = fs::File::create(&doc).unwrap();
            fh.write_all(b"alpha content").unwrap();
            tar_b.append_path_with_name(&doc, "alpha/doc0.txt").unwrap();
            tar_b.finish().unwrap();
        }
        let dest = dir.join("extracted");
        extract_archive(&archive_path, &dest).unwrap();
        let (docs, labels, names) = load_files(&dest).unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(labels.len(), 1);
        assert_eq!(names, vec!["alpha".to_string()]);
        let _ = fs::remove_dir_all(&dir);
    }
}
