# Feature: Phase 2 — Classical ML Core

## Summary
Build the classical machine learning algorithm suite on top of Phase 1's foundation: decision trees, random forests, kNN, Naive Bayes, linear SVM, k-Means, DBSCAN, PCA, TruncatedSVD, hyperparameter search (GridSearchCV, RandomizedSearchCV), model serialization via MessagePack, and built-in toy datasets with synthetic generators. After this phase, ferrolearn covers the most commonly used ML algorithms and can serve as a drop-in replacement for basic scikit-learn workflows.

## Requirements

- REQ-1: Decision Tree classifier and regressor in `ferrolearn-tree` with Gini and entropy criteria, configurable max depth, min samples split/leaf, and feature importance extraction
- REQ-2: Random Forest classifier and regressor in `ferrolearn-tree` with parallelized tree fitting via Rayon, `n_estimators` and `n_jobs` controls, and `feature_importances_` via mean decrease in impurity
- REQ-3: k-Nearest Neighbors classifier and regressor in `ferrolearn-neighbors` with ball tree and KD-tree spatial indexing backends
- REQ-4: Gaussian, Multinomial, Bernoulli, and Complement Naive Bayes variants in `ferrolearn-bayes`
- REQ-5: Linear SVM (SVC) with SMO solver in `ferrolearn-linear`, supporting L1 and L2 penalties
- REQ-6: k-Means clustering in `ferrolearn-cluster` with k-Means++ initialization, parallelized via Rayon, exposing `cluster_centers_`, `labels_`, `inertia_`, `n_iter_`
- REQ-7: DBSCAN clustering in `ferrolearn-cluster` with `eps` and `min_samples` parameters, exposing `labels_` with noise labeled as -1
- REQ-8: PCA (full and randomized/truncated SVD variant) in `ferrolearn-decomp`, exposing `components_`, `explained_variance_`, `explained_variance_ratio_`
- REQ-9: TruncatedSVD in `ferrolearn-decomp` operating natively on sparse matrices from `ferrolearn-sparse`
- REQ-10: `GridSearchCV` and `RandomizedSearchCV` in `ferrolearn-model-sel` with parallel evaluation via Rayon, `best_params()`, `best_score()`, `cv_results()` accessors
- REQ-11: `param_grid!` macro for ergonomic hyperparameter grid construction
- REQ-12: `Distribution` trait and implementations (`Uniform`, `LogUniform`, `IntUniform`, `Choice`) for `RandomizedSearchCV` sampling
- REQ-13: Model serialization via `rmp-serde` (MessagePack) in `ferrolearn-io` with version-tagged envelope, `save_model()` and `load_model()` functions, and clear error on version mismatch
- REQ-14: JSON export via `serde_json` for debugging in `ferrolearn-io`
- REQ-15: Toy datasets (Iris, Digits, Wine, Breast Cancer, Diabetes) embedded in `ferrolearn-datasets` as compiled-in data
- REQ-16: Synthetic generators `make_classification`, `make_regression`, `make_blobs`, `make_moons`, `make_circles` in `ferrolearn-datasets`
- REQ-17: Oracle fixture tests (Layer 1) for all new algorithms — standard case plus minimum 3 non-default hyperparameter configurations
- REQ-18: Property-based tests (Layer 2) with minimum 8 invariants per algorithm
- REQ-19: Fuzz targets (Layer 5) for all new public APIs
- REQ-20: Algorithm equivalence documents (Layer 4) for Decision Tree, Random Forest, k-Means, PCA under `docs/algorithm_equivalence/`
- REQ-21: Clustering metrics (silhouette score, adjusted Rand, adjusted mutual info, Davies-Bouldin) in `ferrolearn-metrics`
- REQ-22: `HasFeatureImportances` and `HasClasses<L>` trait implementations for all applicable fitted models

## Acceptance Criteria

- [ ] AC-1: Decision Tree classifier on Iris fixture matches scikit-learn predictions exactly; Gini impurity at each split within 4 ULPs
- [ ] AC-2: Random Forest with `n_estimators=100, random_state=42` on Breast Cancer fixture achieves accuracy within 0.5% of scikit-learn (due to parallelism ordering)
- [ ] AC-3: kNN with `k=5` on Digits fixture matches scikit-learn predictions exactly
- [ ] AC-4: GaussianNB on Iris fixture matches scikit-learn `predict_proba` within 4 ULPs
- [ ] AC-5: k-Means with `k=3, random_state=42` on Iris produces inertia within 4 ULPs of scikit-learn; every sample is assigned to its nearest centroid (property invariant)
- [ ] AC-6: DBSCAN on the moons dataset with `eps=0.3, min_samples=10` matches scikit-learn label assignments exactly
- [ ] AC-7: PCA `explained_variance_ratio_` sums to <= 1.0 and components are orthonormal (`C @ C.T ≈ I` within 4 ULPs)
- [ ] AC-8: TruncatedSVD accepts `CsrMatrix` input and produces correct decomposition verified against scikit-learn fixtures
- [ ] AC-9: `GridSearchCV` with 3x3 grid on LogisticRegression finds the same best params as scikit-learn
- [ ] AC-10: `save_model(&fitted_rf, "model.fl")?; let loaded: FittedRandomForest<f64> = load_model("model.fl")?;` round-trips and produces identical predictions
- [ ] AC-11: `datasets::load_iris()` returns 150 samples x 4 features with correct target values
- [ ] AC-12: `make_classification(n_samples=1000, n_features=20, random_state=42)` produces reproducible output matching scikit-learn
- [ ] AC-13: All fuzz targets run 1 hour without panics
- [ ] AC-14: `silhouette_score` on k-Means output matches scikit-learn within 4 ULPs
- [ ] AC-15: Algorithm equivalence documents exist for Decision Tree, Random Forest, k-Means, PCA

## Architecture

### New Crate Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| rmp-serde | 1.3 | MessagePack serialization for model persistence |
| serde_json | 1.0 | JSON export for debugging |
| rand | 0.9 | Random number generation for k-Means init, Random Forest bagging |
| rand_xoshiro | 0.7 | Deterministic PRNG (Xoshiro256++) for reproducibility |

### Decision Tree Implementation

File: `ferrolearn-tree/src/decision_tree.rs`

The tree is built recursively using the CART algorithm. Each node stores either a split (feature index, threshold, left/right child indices) or a leaf (predicted value/class distribution). The tree is stored as a flat `Vec<Node>` for cache efficiency, matching scikit-learn's `Tree` structure.

Split selection iterates over features and candidate thresholds (sorted unique values), computing impurity decrease. For classification: Gini or entropy. For regression: MSE reduction.

The fitted tree exposes `feature_importances()` computed as weighted impurity decrease summed across all splits using that feature.

### Random Forest Implementation

File: `ferrolearn-tree/src/random_forest.rs`

Fits `n_estimators` Decision Trees on bootstrap samples with `max_features` random feature subsets per split. Tree fitting is parallelized with `rayon::par_iter()`. Predictions aggregate via majority vote (classification) or mean (regression).

Deterministic reproducibility with `random_state`: a master RNG seeds per-tree child RNGs sequentially before parallel dispatch, ensuring identical trees regardless of thread scheduling.

### kNN Implementation

File: `ferrolearn-tree/src/knn.rs` (or `ferrolearn-neighbors/`)

Two spatial indexing backends:
- **KD-Tree**: Recursive binary space partitioning. O(n log n) build, O(log n) query for low dimensions. Falls back to brute force when d > 20.
- **Ball Tree**: Hierarchical nesting of hyperspheres. Better than KD-Tree in high dimensions.

The fitted model stores the training data and the spatial index. `predict()` finds k nearest neighbors and returns majority vote (classification) or mean (regression).

### k-Means Implementation

File: `ferrolearn-cluster/src/kmeans.rs`

Lloyd's algorithm with k-Means++ initialization. Each iteration: (1) assign samples to nearest centroid, (2) recompute centroids as cluster means. Convergence when centroid movement < `tol` or `max_iter` reached.

Parallelized via Rayon: assignment step splits samples across threads, centroid update uses `par_iter().fold()` with thread-local accumulators.

### PCA Implementation

File: `ferrolearn-decomp/src/pca.rs`

1. Center data (subtract column means)
2. Compute SVD via `faer::linalg::svd`
3. Extract top `n_components` right singular vectors as `components_`
4. Compute `explained_variance_` from singular values: `s² / (n_samples - 1)`

Randomized PCA variant uses the Halko-Martinsson-Tropp algorithm for truncated SVD, which is O(n_samples * n_components²) instead of O(n_samples * n_features²).

### Model Serialization Format

File: `ferrolearn-io/src/lib.rs`

```rust
struct ModelEnvelope {
    magic: [u8; 4],           // b"FLRN"
    schema_version: u32,      // Incremented on breaking changes
    model_type: String,       // e.g. "FittedRandomForest<f64>"
    payload: Vec<u8>,         // MessagePack-encoded model
    checksum: u32,            // CRC32 of payload
}
```

`save_model()` wraps the serde-serialized model in this envelope. `load_model()` validates magic bytes, checks schema version compatibility, verifies checksum, then deserializes. Version mismatch returns `FerroError::SerdeError` with a clear message.

### Hyperparameter Search

File: `ferrolearn-model-sel/src/grid_search.rs`

`GridSearchCV` iterates all parameter combinations, running `cross_val_score` for each. Results are stored in a `CvResults` table (parameter values, mean score, std score, per-fold scores).

`RandomizedSearchCV` samples `n_iter` parameter combinations from `Distribution` objects instead of exhaustive enumeration.

Both use `rayon::par_iter()` across parameter combinations (outer parallelism). The `n_jobs` parameter controls the Rayon thread pool size.

### Dataset Embedding

File: `ferrolearn-datasets/src/lib.rs`

Toy datasets are stored as compressed CSV bytes via `include_bytes!()` and parsed at load time into `Array2<f64>` + `Array1<usize>`. This avoids runtime file I/O and works in all environments including WASM.

Synthetic generators use `rand_xoshiro::Xoshiro256PlusPlus` with a user-provided `random_state` seed for exact reproducibility.

## Resolved Questions

### Q3: Separate `ferrolearn-neighbors` crate
**Decision:** kNN gets its own `ferrolearn-neighbors` crate. It shares zero code with tree-based methods, and sklearn itself separates them (`sklearn.neighbors` vs `sklearn.tree`). The workspace overhead is one `Cargo.toml` — trivial. This also leaves room for future neighbor-based algorithms (RadiusNeighbors, NearestCentroid) without polluting the tree crate.

### Q4: Separate `ferrolearn-bayes` crate
**Decision:** Naive Bayes gets its own `ferrolearn-bayes` crate. The four variants (Gaussian, Multinomial, Bernoulli, Complement) form a natural module with shared base logic (prior computation, log-likelihood). They share no implementation code with linear models despite being technically linear in log-space. Matches sklearn's `sklearn.naive_bayes` organization.

### Q5: Match sklearn's RF reproducibility behavior
**Decision:** Deterministic with fixed `random_state` AND fixed `n_jobs`, NOT deterministic across different thread counts. This matches scikit-learn's documented behavior. Guaranteeing cross-thread-count determinism would require sequential RNG dispatch that kills parallelism. The contract is documented clearly: "Results are reproducible when both `random_state` and `n_jobs` are fixed. Changing `n_jobs` may produce different results even with the same `random_state`."

## Out of Scope
- Kernel SVM (Phase 3 stretch)
- AdaBoost, Gradient Boosting (Phase 3)
- HDBSCAN, GMM, Agglomerative Clustering (Phase 3)
- t-SNE, UMAP, NMF (Phase 3)
- ONNX export (Phase 3)
- Online/streaming learning (Phase 4)
- GPU acceleration (Phase 4)
- Statistical equivalence benchmarks (Layer 3 — required before stable 1.0, not per-phase)
