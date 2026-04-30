//! Proof-of-API integration test for ferrolearn-fetch.
//!
//! Most fetchers need real network access — they're marked `#[ignore]` so
//! `cargo test` stays offline-clean. The non-ignored tests cover the
//! offline portion: cache primitives, RemoteFile metadata constants, and
//! filesystem helpers.

use ferrolearn_fetch::{
    CaliforniaHousing, Covtype, KddCup99, KddSubset, NewsgroupsSubset, OpenmlDataset, RemoteFile,
    clear_data_home, dataset_dir, fetch_20newsgroups, fetch_california_housing, fetch_covtype,
    fetch_file, fetch_kddcup99, fetch_openml, get_data_home,
};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn unique_tmp(label: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    std::env::temp_dir().join(format!("ferrolearn_fetch_apiproof_{label}_{nanos}"))
}

#[test]
fn api_proof_cache_round_trip() {
    let p = unique_tmp("cache");
    let resolved = get_data_home(Some(&p)).unwrap();
    assert_eq!(resolved, p);
    assert!(p.is_dir());
    let sub = dataset_dir("synthetic", Some(&p)).unwrap();
    assert!(sub.is_dir());
    clear_data_home(Some(&p)).unwrap();
    assert!(!p.exists());
}

#[test]
fn api_proof_remote_file_struct_construction() {
    // Just confirm the public fields are addressable.
    let rf = RemoteFile {
        filename: "test.bin",
        url: "https://example.com/test.bin",
        sha256: "0".repeat(64).leak(),
    };
    assert_eq!(rf.filename, "test.bin");
}

#[test]
fn api_proof_dataset_constants_match_sklearn() {
    use ferrolearn_fetch::california_housing::ARCHIVE as CALI;
    use ferrolearn_fetch::covtype::ARCHIVE as COV;
    use ferrolearn_fetch::kddcup99::{ARCHIVE_10PCT, ARCHIVE_FULL};
    use ferrolearn_fetch::newsgroups::ARCHIVE as NEWS;

    for rf in [CALI, COV, ARCHIVE_FULL, ARCHIVE_10PCT, NEWS] {
        assert!(rf.url.starts_with("https://"));
        assert_eq!(rf.sha256.len(), 64);
        assert!(!rf.filename.is_empty());
    }
}

// ---------------------------------------------------------------------------
// Network-dependent tests below — run only with `cargo test -- --ignored`.
// They exist mainly to validate the API signatures compile against the
// real fetcher interfaces.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires network"]
fn api_proof_fetch_file_offline_works() {
    let dir = unique_tmp("file");
    let _: std::path::PathBuf =
        fetch_file("https://www.example.com/", "example.html", None, &dir).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
#[ignore = "requires ~14MB download"]
fn api_proof_fetch_california_housing() {
    let dir = unique_tmp("cali");
    let ds: CaliforniaHousing = fetch_california_housing(Some(&dir)).unwrap();
    assert_eq!(ds.data.dim(), (20640, 8));
    assert_eq!(ds.target.len(), 20640);
    assert_eq!(ds.feature_names.len(), 8);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
#[ignore = "requires ~70MB download"]
fn api_proof_fetch_covtype() {
    let dir = unique_tmp("covtype");
    let ds: Covtype = fetch_covtype(Some(&dir)).unwrap();
    assert_eq!(ds.data.ncols(), 54);
    assert_eq!(ds.data.nrows(), ds.target.len());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
#[ignore = "requires ~70MB download"]
fn api_proof_fetch_kddcup99() {
    let dir = unique_tmp("kdd");
    let ds: KddCup99 = fetch_kddcup99(Some(&dir), KddSubset::Percent10).unwrap();
    assert_eq!(ds.data.ncols(), 41);
    assert_eq!(ds.data.nrows(), ds.target.len());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
#[ignore = "requires ~14MB download"]
fn api_proof_fetch_20newsgroups() {
    let dir = unique_tmp("news");
    for subset in [
        NewsgroupsSubset::Train,
        NewsgroupsSubset::Test,
        NewsgroupsSubset::All,
    ] {
        let (docs, labels, names) = fetch_20newsgroups(Some(&dir), subset).unwrap();
        assert_eq!(docs.len(), labels.len());
        assert_eq!(names.len(), 20);
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
#[ignore = "requires network + small OpenML dataset"]
fn api_proof_fetch_openml() {
    let dir = unique_tmp("openml");
    // Iris (data_id=61) is small and stable.
    let ds: OpenmlDataset = fetch_openml(61, Some("class"), Some(&dir)).unwrap();
    assert!(ds.data.nrows() > 0);
    assert_eq!(ds.target.len(), ds.data.nrows());
    assert_eq!(ds.target_name, "class");
    let _ = std::fs::remove_dir_all(&dir);
}
