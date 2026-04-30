//! # ferrolearn-fetch
//!
//! Network fetchers for sklearn's `sklearn.datasets.fetch_*` family. Each
//! fetcher downloads its dataset on first use, verifies the upstream
//! SHA-256, and caches the result on disk.
//!
//! Default cache: `<dirs::data_local_dir>/ferrolearn_data/` (override via
//! the `FERROLEARN_DATA` environment variable, or pass `data_home` to
//! the fetcher).
//!
//! # Available fetchers
//!
//! - [`fetch_california_housing`] — 20640×8 regression benchmark.
//! - [`fetch_covtype`] — 581012×54 multiclass classification.
//! - [`fetch_kddcup99`] — KDD Cup 1999 intrusion-detection dataset.
//! - [`fetch_20newsgroups`] — text classification (20 categories).
//! - [`fetch_openml`] — generic OpenML.org client (any dataset by ID).
//!
//! # Cache primitives
//!
//! - [`get_data_home`] — return (and create) the cache directory.
//! - [`clear_data_home`] — wipe the cache directory.
//! - [`fetch_file`] — generic single-URL fetcher with optional checksum.

pub mod cache;
pub mod california_housing;
pub mod covtype;
pub mod fetch;
pub mod kddcup99;
pub mod newsgroups;
pub mod openml;

pub use cache::{clear_data_home, dataset_dir, get_data_home};
pub use california_housing::{CaliforniaHousing, fetch_california_housing};
pub use covtype::{Covtype, fetch_covtype};
pub use fetch::{RemoteFile, fetch_file};
pub use kddcup99::{KddCup99, KddSubset, fetch_kddcup99};
pub use newsgroups::{NewsgroupsSubset, fetch_20newsgroups};
pub use openml::{OpenmlDataset, fetch_openml};
