# ferrolearn-fetch

Network fetchers for the [ferrolearn](https://github.com/dollspace-gay/ferrolearn) machine learning framework, mirroring scikit-learn's `sklearn.datasets.fetch_*` family.

A scikit-learn equivalent for Rust, providing:

- **Cache primitives**: `get_data_home`, `clear_data_home`, `fetch_file`.
- **Specific fetchers** (each downloads on first use, then caches):
  - `fetch_california_housing` — 20640×8 regression benchmark.
  - `fetch_covtype` — Forest Covertype, 581012×54 multiclass.
  - `fetch_kddcup99` — KDD Cup 99 intrusion detection.
  - `fetch_20newsgroups` — text classification (20 categories).
  - `fetch_openml` — generic OpenML.org client (any dataset by name + version).

## Cache location

Default `<dirs::data_local_dir>/ferrolearn_data/` (e.g. `~/.local/share/ferrolearn_data/` on Linux). Override via the `FERROLEARN_DATA` environment variable or the `data_home` parameter.

## Why a separate crate?

Network access pulls in TLS + gzip + tar + a sync HTTP client. The core `ferrolearn-datasets` crate stays purely numerical; opt into network behaviour by adding `ferrolearn-fetch` to your dependencies.

## License

Dual-licensed under MIT or Apache-2.0.
