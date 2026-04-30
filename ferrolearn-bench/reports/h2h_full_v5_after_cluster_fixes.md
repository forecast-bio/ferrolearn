# ferrolearn vs scikit-learn — head-to-head report

Each row is a single (algorithm, dataset) head-to-head: same canonical
dataset (sklearn `make_*`), same train/test split, same hyperparameters,
same quality metric, both libraries fit + predict in the same Python
process. Δ is `ferrolearn − sklearn` for the quality metric (positive
means ferrolearn is more accurate; for `recon_rel`, lower is better and
the cell shows `ferrolearn / sklearn` ratio).

### Classifier — 51 comparisons

| Algorithm | Dataset | sklearn fit | ferrolearn fit | fit speedup | sklearn predict | ferrolearn predict | predict speedup | metric | sklearn | ferrolearn | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AdaBoostClassifier | small_1Kx10 | 51.8 ms | 11.1 ms | **4.7x** | 1.6 ms | 44.7 us | **34.8x** | accuracy | 91.50% | 92.00% | +0.50pp |
| AdaBoostClassifier | tiny_50x5 | 25.4 ms | 209.5 us | **121.2x** | 1.3 ms | 2.9 us | **465.7x** | accuracy | 90.00% | 100.00% | +10.00pp |
| BaggingClassifier | small_1Kx10 | 11.3 ms | 2.5 ms | **4.5x** | 10.8 ms | 66.2 us | **163.1x** | accuracy | 95.00% | 96.00% | +1.00pp |
| BaggingClassifier | tiny_50x5 | 11.4 ms | 420.8 us | **27.2x** | 11.0 ms | 2.3 us | **4807.7x** | accuracy | 80.00% | 90.00% | +10.00pp |
| BernoulliNB | medium_10Kx100 | 7.3 ms | 1.8 ms | **4.1x** | 1.7 ms | 345.2 us | **4.8x** | accuracy | 75.00% | 75.00% | +0.00pp |
| BernoulliNB | small_1Kx10 | 539.6 us | 24.7 us | **21.8x** | 99.5 us | 3.6 us | **27.6x** | accuracy | 77.50% | 77.50% | +0.00pp |
| BernoulliNB | tiny_50x5 | 397.2 us | 5.1 us | **77.9x** | 83.5 us | 746 ns | **111.9x** | accuracy | 100.00% | 100.00% | +0.00pp |
| ComplementNB | medium_10Kx100 | 1.8 ms | 1.0 ms | 1.76x | 140.2 us | 146.2 us | 0.96x | accuracy | 61.20% | 61.20% | +0.00pp |
| ComplementNB | small_1Kx10 | 447.5 us | 25.2 us | **17.8x** | 22.3 us | 2.6 us | **8.4x** | accuracy | 71.00% | 71.00% | +0.00pp |
| ComplementNB | tiny_50x5 | 371.5 us | 4.2 us | **89.1x** | 19.7 us | 689 ns | **28.7x** | accuracy | 30.00% | 30.00% | +0.00pp |
| DecisionTreeClassifier | medium_10Kx100 | 597.6 ms | 291.9 ms | **2.0x** | 274.0 us | 336.1 us | 0.82x | accuracy | 73.65% | 71.95% | -1.70pp |
| DecisionTreeClassifier | small_1Kx10 | 3.0 ms | 1.8 ms | 1.67x | 41.6 us | 29.5 us | 1.41x | accuracy | 89.50% | 90.00% | +0.50pp |
| DecisionTreeClassifier | tiny_50x5 | 275.4 us | 81.0 us | **3.4x** | 33.1 us | 16.8 us | 1.97x | accuracy | 90.00% | 80.00% | -10.00pp |
| ExtraTreeClassifier | medium_10Kx100 | 8.6 ms | 10.5 ms | 0.82x | 301.0 us | 243.1 us | 1.24x | accuracy | 64.20% | 64.20% | +0.00pp |
| ExtraTreeClassifier | small_1Kx10 | 862.3 us | 273.7 us | **3.2x** | 33.2 us | 8.5 us | **3.9x** | accuracy | 81.00% | 82.00% | +1.00pp |
| ExtraTreeClassifier | tiny_50x5 | 270.4 us | 11.6 us | **23.3x** | 33.4 us | 693 ns | **48.2x** | accuracy | 90.00% | 100.00% | +10.00pp |
| ExtraTreesClassifier | medium_10Kx100 | 128.0 ms | 63.5 ms | **2.0x** | 24.8 ms | 51.4 ms | 0.48x | accuracy | 93.90% | 93.75% | -0.15pp |
| ExtraTreesClassifier | small_1Kx10 | 91.6 ms | 3.4 ms | **27.2x** | 25.5 ms | 1.5 ms | **17.1x** | accuracy | 97.00% | 96.00% | -1.00pp |
| ExtraTreesClassifier | tiny_50x5 | 60.0 ms | 1.9 ms | **31.0x** | 14.1 ms | 12.2 us | **1149.7x** | accuracy | 90.00% | 90.00% | +0.00pp |
| GaussianNB | medium_10Kx100 | 4.3 ms | 2.4 ms | 1.77x | 653.9 us | 1.6 ms | 0.42x | accuracy | 81.55% | 81.55% | +0.00pp |
| GaussianNB | small_1Kx10 | 441.6 us | 218.4 us | **2.0x** | 68.9 us | 46.5 us | 1.48x | accuracy | 89.00% | 89.00% | +0.00pp |
| GaussianNB | tiny_50x5 | 250.6 us | 75.2 us | **3.3x** | 37.6 us | 17.3 us | **2.2x** | accuracy | 100.00% | 100.00% | +0.00pp |
| GradientBoostingClassifier | small_1Kx10 | 135.8 ms | 51.2 ms | **2.7x** | 249.1 us | 78.8 us | **3.2x** | accuracy | 96.00% | 94.00% | -2.00pp |
| GradientBoostingClassifier | tiny_50x5 | 22.1 ms | 489.7 us | **45.1x** | 75.7 us | 3.4 us | **22.2x** | accuracy | 80.00% | 80.00% | +0.00pp |
| HistGradientBoostingClassifier | medium_10Kx100 | 280.7 ms | 940.8 ms | 0.30x | 1.5 ms | 13.9 ms | 0.11x | accuracy | 95.80% | 95.80% | +0.00pp |
| HistGradientBoostingClassifier | small_1Kx10 | 114.5 ms | 40.6 ms | **2.8x** | 1.1 ms | 945.2 us | 1.21x | accuracy | 96.00% | 94.00% | -2.00pp |
| HistGradientBoostingClassifier | tiny_50x5 | 24.5 ms | 385.1 us | **63.7x** | 1.1 ms | 3.3 us | **325.0x** | accuracy | 100.00% | 100.00% | +0.00pp |
| KNeighborsClassifier | medium_10Kx100 | 828.1 us | 15.1 ms | 0.05x | 17.2 ms | 44.9 ms | 0.38x | accuracy | 96.60% | 96.60% | +0.00pp |
| KNeighborsClassifier | small_1Kx10 | 399.2 us | 585.7 us | 0.68x | 15.4 ms | 6.5 ms | **2.4x** | accuracy | 91.50% | 91.50% | +0.00pp |
| KNeighborsClassifier | tiny_50x5 | 167.2 us | 100.9 us | 1.66x | 15.1 ms | 28.8 us | **524.7x** | accuracy | 90.00% | 90.00% | +0.00pp |
| LinearSVC | medium_10Kx100 | 44.3 ms | 21.4 ms | **2.1x** | 84.6 us | 75.0 us | 1.13x | accuracy | 83.60% | 83.70% | +0.10pp |
| LinearSVC | small_1Kx10 | 685.7 us | 332.8 us | **2.1x** | 35.0 us | 2.9 us | **12.0x** | accuracy | 83.50% | 83.50% | +0.00pp |
| LinearSVC | tiny_50x5 | 233.2 us | 6.8 us | **34.5x** | 25.4 us | 785 ns | **32.3x** | accuracy | 90.00% | 90.00% | +0.00pp |
| LogisticRegression | medium_10Kx100 | 798.1 ms | 47.0 ms | **17.0x** | 158.2 us | 149.6 us | 1.06x | accuracy | 83.50% | 83.45% | -0.05pp |
| LogisticRegression | small_1Kx10 | 881.3 us | 376.3 us | **2.3x** | 35.5 us | 51.6 us | 0.69x | accuracy | 83.50% | 82.00% | -1.50pp |
| LogisticRegression | tiny_50x5 | 567.7 us | 91.6 us | **6.2x** | 27.3 us | 18.5 us | 1.48x | accuracy | 100.00% | 100.00% | +0.00pp |
| MultinomialNB | medium_10Kx100 | 2.1 ms | 1.2 ms | 1.67x | 171.1 us | 151.3 us | 1.13x | accuracy | 61.20% | 61.20% | +0.00pp |
| MultinomialNB | small_1Kx10 | 461.6 us | 23.1 us | **20.0x** | 25.7 us | 2.8 us | **9.1x** | accuracy | 70.50% | 70.50% | +0.00pp |
| MultinomialNB | tiny_50x5 | 356.4 us | 5.0 us | **71.8x** | 26.0 us | 726 ns | **35.8x** | accuracy | 30.00% | 30.00% | +0.00pp |
| NearestCentroid | medium_10Kx100 | 3.8 ms | 1.7 ms | **2.2x** | 862.7 us | 189.5 us | **4.6x** | accuracy | 69.15% | 69.15% | +0.00pp |
| NearestCentroid | small_1Kx10 | 322.9 us | 27.6 us | **11.7x** | 238.4 us | 3.6 us | **65.4x** | accuracy | 77.50% | 77.50% | +0.00pp |
| NearestCentroid | tiny_50x5 | 219.6 us | 13.6 us | **16.1x** | 181.7 us | 1.8 us | **100.2x** | accuracy | 100.00% | 100.00% | +0.00pp |
| QDA | medium_10Kx100 | — | — | — | — | — | — | accuracy | — | — | — |
| QDA | small_1Kx10 | — | — | — | — | — | — | accuracy | — | — | — |
| QDA | tiny_50x5 | — | — | — | — | — | — | accuracy | — | — | — |
| RandomForestClassifier | medium_10Kx100 | 277.9 ms | 192.7 ms | 1.44x | 24.8 ms | 30.2 ms | 0.82x | accuracy | 94.25% | 93.05% | -1.20pp |
| RandomForestClassifier | small_1Kx10 | 124.4 ms | 5.1 ms | **24.2x** | 25.9 ms | 874.5 us | **29.6x** | accuracy | 94.50% | 96.00% | +1.50pp |
| RandomForestClassifier | tiny_50x5 | 79.3 ms | 1.8 ms | **43.4x** | 14.4 ms | 24.9 us | **578.4x** | accuracy | 80.00% | 90.00% | +10.00pp |
| RidgeClassifier | medium_10Kx100 | 6.3 ms | 4.6 ms | 1.37x | 100.1 us | 121.4 us | 0.82x | accuracy | 83.35% | 83.35% | +0.00pp |
| RidgeClassifier | small_1Kx10 | 954.1 us | 55.3 us | **17.3x** | 35.9 us | 3.3 us | **10.8x** | accuracy | 83.00% | 83.00% | +0.00pp |
| RidgeClassifier | tiny_50x5 | 688.7 us | 6.0 us | **114.6x** | 28.6 us | 834 ns | **34.3x** | accuracy | 90.00% | 90.00% | +0.00pp |

### Cluster — 15 comparisons

| Algorithm | Dataset | sklearn fit | ferrolearn fit | fit speedup | sklearn predict | ferrolearn predict | predict speedup | metric | sklearn | ferrolearn | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AgglomerativeClustering | small_1Kx10 | 6.5 ms | 132.0 ms | 0.05x | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |
| AgglomerativeClustering | tiny_200x5 | 501.6 us | 1.1 ms | 0.46x | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |
| Birch | small_1Kx10 | 23.6 ms | 5.2 ms | **4.6x** | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |
| Birch | tiny_200x5 | 3.4 ms | 825.6 us | **4.1x** | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |
| DBSCAN | small_1Kx10 | 3.1 ms | 2.2 ms | 1.37x | — | — | — | ari | 0.0000 | 0.0000 | +0.0000 |
| DBSCAN | tiny_200x5 | 531.4 us | 71.7 us | **7.4x** | — | — | — | ari | 0.0000 | 0.0000 | +0.0000 |
| GaussianMixture | medium_5Kx20 | 66.1 ms | 180.6 ms | 0.37x | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |
| GaussianMixture | small_1Kx10 | 6.3 ms | 6.4 ms | 0.99x | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |
| GaussianMixture | tiny_200x5 | 1.9 ms | 482.9 us | **3.9x** | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |
| KMeans | medium_5Kx20 | 45.6 ms | 32.5 ms | 1.40x | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |
| KMeans | small_1Kx10 | 9.4 ms | 2.3 ms | **4.1x** | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |
| KMeans | tiny_200x5 | 4.6 ms | 385.1 us | **11.9x** | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |
| MiniBatchKMeans | medium_5Kx20 | 12.7 ms | 34.2 ms | 0.37x | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |
| MiniBatchKMeans | small_1Kx10 | 16.0 ms | 19.2 ms | 0.84x | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |
| MiniBatchKMeans | tiny_200x5 | 2.6 ms | 6.1 ms | 0.42x | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |

### Decomp — 15 comparisons

| Algorithm | Dataset | sklearn fit | ferrolearn fit | fit speedup | sklearn predict | ferrolearn predict | predict speedup | metric | sklearn | ferrolearn | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FactorAnalysis | small_1Kx10 | 9.5 ms | 33.7 ms | 0.28x | 51.0 us | 18.9 us | **2.7x** | recon_rel | — | — | — |
| FactorAnalysis | tiny_50x5 | 1.6 ms | 166.4 us | **9.8x** | 35.5 us | 2.3 us | **15.7x** | recon_rel | — | — | — |
| FastICA | small_1Kx10 | 61.8 ms | 7.8 ms | **7.9x** | 103.1 us | 18.9 us | **5.5x** | recon_rel | 6.787e-01 | — | — |
| FastICA | tiny_50x5 | 8.5 ms | 119.9 us | **70.6x** | 22.1 us | 2.0 us | **11.0x** | recon_rel | 3.394e-01 | — | — |
| IncrementalPCA | medium_10Kx100 | 214.2 ms | 134.7 ms | 1.59x | 12.0 ms | 1.3 ms | **9.4x** | recon_rel | 9.722e-01 | — | — |
| IncrementalPCA | small_1Kx10 | 2.4 ms | 49.4 us | **48.1x** | 37.6 us | 10.1 us | **3.7x** | recon_rel | 6.969e-01 | — | — |
| IncrementalPCA | tiny_50x5 | 324.5 us | 3.9 us | **82.6x** | 23.7 us | 1.6 us | **14.4x** | recon_rel | 3.483e-01 | — | — |
| PCA | medium_10Kx100 | 33.6 ms | 15.3 ms | **2.2x** | 637.2 us | 2.0 ms | 0.32x | recon_rel | 9.698e-01 | 9.698e-01 | 1.00x |
| PCA | small_1Kx10 | 181.6 us | 49.9 us | **3.6x** | 35.4 us | 32.8 us | 1.08x | recon_rel | 6.787e-01 | 6.787e-01 | 1.00x |
| PCA | tiny_50x5 | 141.9 us | 22.5 us | **6.3x** | 23.6 us | 19.4 us | 1.22x | recon_rel | 3.394e-01 | 3.394e-01 | 1.00x |
| SparsePCA | small_1Kx10 | 246.5 ms | 423.0 ms | 0.58x | 483.6 us | 11.2 us | **43.3x** | recon_rel | 6.875e-01 | — | — |
| SparsePCA | tiny_50x5 | 9.5 ms | 1.3 ms | **7.4x** | 188.9 us | 1.6 us | **114.8x** | recon_rel | 3.541e-01 | — | — |
| TruncatedSVD | medium_10Kx100 | 26.2 ms | 14.9 ms | 1.76x | 424.2 us | 1.6 ms | 0.26x | recon_rel | 9.713e-01 | — | — |
| TruncatedSVD | small_1Kx10 | 706.6 us | 192.0 us | **3.7x** | 26.0 us | 7.2 us | **3.6x** | recon_rel | 6.790e-01 | — | — |
| TruncatedSVD | tiny_50x5 | 399.2 us | 5.8 us | **69.3x** | 21.3 us | 1.4 us | **15.5x** | recon_rel | 3.406e-01 | — | — |

### Kernel — 6 comparisons

| Algorithm | Dataset | sklearn fit | ferrolearn fit | fit speedup | sklearn predict | ferrolearn predict | predict speedup | metric | sklearn | ferrolearn | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Nystroem | medium_10Kx100 | 16.0 ms | 4.1 ms | **3.9x** | 18.1 ms | 63.4 ms | 0.29x | timing_only | — | — | — |
| Nystroem | small_1Kx10 | 72.2 ms | 3.7 ms | **19.3x** | 10.6 ms | 3.6 ms | **3.0x** | timing_only | — | — | — |
| Nystroem | tiny_50x5 | 1.2 ms | 353.1 us | **3.5x** | 378.0 us | 109.7 us | **3.4x** | timing_only | — | — | — |
| RBFSampler | medium_10Kx100 | 514.6 us | 290.8 us | 1.77x | 15.0 ms | 16.4 ms | 0.92x | timing_only | — | — | — |
| RBFSampler | small_1Kx10 | 200.6 us | 5.1 us | **39.2x** | 1.3 ms | 1.1 ms | 1.18x | timing_only | — | — | — |
| RBFSampler | tiny_50x5 | 284.8 us | 4.0 us | **71.4x** | 125.0 us | 50.5 us | **2.5x** | timing_only | — | — | — |

### Preprocess — 14 comparisons

| Algorithm | Dataset | sklearn fit | ferrolearn fit | fit speedup | sklearn predict | ferrolearn predict | predict speedup | metric | sklearn | ferrolearn | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MaxAbsScaler | medium_10Kx100 | 1.2 ms | 1.1 ms | 1.06x | 1.0 ms | 2.5 ms | 0.42x | rel_diff_vs_sklearn | 0.000e+00 | 0.000e+00 | — |
| MaxAbsScaler | small_1Kx10 | 71.9 us | 8.5 us | **8.4x** | 32.1 us | 11.9 us | **2.7x** | rel_diff_vs_sklearn | 0.000e+00 | 0.000e+00 | — |
| MaxAbsScaler | tiny_50x5 | 120.2 us | 2.0 us | **60.5x** | 57.1 us | 4.4 us | **13.1x** | rel_diff_vs_sklearn | 0.000e+00 | 0.000e+00 | — |
| MinMaxScaler | medium_10Kx100 | 998.3 us | 2.1 ms | 0.46x | 1.1 ms | 2.2 ms | 0.51x | rel_diff_vs_sklearn | 0.000e+00 | 1.146e-16 | — |
| MinMaxScaler | small_1Kx10 | 108.0 us | 15.2 us | **7.1x** | 36.8 us | 11.9 us | **3.1x** | rel_diff_vs_sklearn | 0.000e+00 | 1.088e-16 | — |
| MinMaxScaler | tiny_50x5 | 154.9 us | 2.3 us | **67.0x** | 60.0 us | 4.4 us | **13.7x** | rel_diff_vs_sklearn | 0.000e+00 | 8.444e-17 | — |
| PowerTransformer | small_1Kx10 | 30.1 ms | 1.7 ms | **17.5x** | 403.9 us | 114.9 us | **3.5x** | rel_diff_vs_sklearn | 0.000e+00 | 4.203e-01 | — |
| PowerTransformer | tiny_50x5 | 14.9 ms | 41.8 us | **357.0x** | 105.4 us | 3.6 us | **29.0x** | rel_diff_vs_sklearn | 0.000e+00 | 4.189e-01 | — |
| RobustScaler | medium_10Kx100 | 24.7 ms | 14.9 ms | 1.66x | 1.6 ms | 2.2 ms | 0.70x | rel_diff_vs_sklearn | 0.000e+00 | 0.000e+00 | — |
| RobustScaler | small_1Kx10 | 626.8 us | 115.3 us | **5.4x** | 36.9 us | 12.0 us | **3.1x** | rel_diff_vs_sklearn | 0.000e+00 | 4.251e-20 | — |
| RobustScaler | tiny_50x5 | 787.8 us | 4.7 us | **166.7x** | 67.4 us | 4.4 us | **15.5x** | rel_diff_vs_sklearn | 0.000e+00 | 2.450e-17 | — |
| StandardScaler | medium_10Kx100 | 2.7 ms | 1.9 ms | 1.40x | 1.4 ms | 3.3 ms | 0.42x | rel_diff_vs_sklearn | 0.000e+00 | 0.000e+00 | — |
| StandardScaler | small_1Kx10 | 215.9 us | 29.2 us | **7.4x** | 44.5 us | 31.8 us | 1.40x | rel_diff_vs_sklearn | 0.000e+00 | 0.000e+00 | — |
| StandardScaler | tiny_50x5 | 280.2 us | 49.9 us | **5.6x** | 75.0 us | 53.6 us | 1.40x | rel_diff_vs_sklearn | 0.000e+00 | 0.000e+00 | — |

### Regressor — 43 comparisons

| Algorithm | Dataset | sklearn fit | ferrolearn fit | fit speedup | sklearn predict | ferrolearn predict | predict speedup | metric | sklearn | ferrolearn | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ARDRegression | medium_10Kx100 | 770.0 ms | 49.2 ms | **15.7x** | 119.1 us | 72.6 us | 1.64x | r2 | 1.0000 | 1.0000 | +0.0000 |
| ARDRegression | small_1Kx10 | 707.7 us | 53.2 us | **13.3x** | 19.4 us | 1.2 us | **16.3x** | r2 | 1.0000 | 1.0000 | +0.0000 |
| ARDRegression | tiny_50x5 | 768.2 us | 5.1 us | **151.8x** | 18.2 us | 488 ns | **37.3x** | r2 | 1.0000 | 1.0000 | +0.0000 |
| BayesianRidge | medium_10Kx100 | 201.4 ms | 15.9 ms | **12.7x** | 105.2 us | 122.4 us | 0.86x | r2 | 1.0000 | 1.0000 | +0.0000 |
| BayesianRidge | small_1Kx10 | 385.4 us | 61.4 us | **6.3x** | 18.5 us | 1.2 us | **15.0x** | r2 | 1.0000 | 1.0000 | +0.0000 |
| BayesianRidge | tiny_50x5 | 358.0 us | 5.7 us | **62.9x** | 17.9 us | 489 ns | **36.6x** | r2 | 1.0000 | 1.0000 | -0.0000 |
| DecisionTreeRegressor | medium_10Kx100 | 431.2 ms | 241.9 ms | 1.78x | 316.0 us | 240.1 us | 1.32x | r2 | -0.5324 | -0.5521 | -0.0197 |
| DecisionTreeRegressor | small_1Kx10 | 3.2 ms | 1.7 ms | 1.92x | 34.7 us | 6.1 us | **5.7x** | r2 | 0.6236 | 0.6146 | -0.0090 |
| DecisionTreeRegressor | tiny_50x5 | 259.5 us | 22.6 us | **11.5x** | 23.8 us | 494 ns | **48.1x** | r2 | -2.2859 | -2.1071 | +0.1788 |
| ElasticNet | medium_10Kx100 | 4.2 ms | 10.7 ms | 0.39x | 89.6 us | 180.8 us | 0.50x | r2 | 0.8870 | 0.8870 | -0.0000 |
| ElasticNet | small_1Kx10 | 195.0 us | 97.5 us | **2.0x** | 20.5 us | 19.1 us | 1.07x | r2 | 0.8761 | 0.8761 | -0.0000 |
| ElasticNet | tiny_50x5 | 183.1 us | 35.1 us | **5.2x** | 18.9 us | 17.1 us | 1.11x | r2 | 0.8193 | 0.8193 | -0.0000 |
| ExtraTreesRegressor | medium_10Kx100 | 538.7 ms | 403.1 ms | 1.34x | 24.8 ms | 66.5 ms | 0.37x | r2 | 0.3724 | 0.3690 | -0.0034 |
| ExtraTreesRegressor | small_1Kx10 | 47.7 ms | 6.1 ms | **7.8x** | 25.2 ms | 2.1 ms | **11.9x** | r2 | 0.8793 | 0.8875 | +0.0082 |
| ExtraTreesRegressor | tiny_50x5 | 38.9 ms | 2.6 ms | **14.9x** | 14.3 ms | 34.6 us | **414.3x** | r2 | 0.2541 | 0.1803 | -0.0738 |
| GradientBoostingRegressor | small_1Kx10 | 122.2 ms | 53.1 ms | **2.3x** | 258.5 us | 82.0 us | **3.2x** | r2 | 0.9268 | 0.9269 | +0.0001 |
| GradientBoostingRegressor | tiny_50x5 | 16.2 ms | 1.0 ms | **15.5x** | 69.2 us | 4.6 us | **15.1x** | r2 | -0.1346 | -0.1405 | -0.0059 |
| HistGradientBoostingRegressor | medium_10Kx100 | 275.4 ms | 982.9 ms | 0.28x | 1.5 ms | 12.9 ms | 0.11x | r2 | 0.6349 | 0.6349 | +0.0000 |
| HistGradientBoostingRegressor | small_1Kx10 | 121.9 ms | 41.5 ms | **2.9x** | 565.1 us | 1.2 ms | 0.48x | r2 | 0.9394 | 0.9405 | +0.0012 |
| HistGradientBoostingRegressor | tiny_50x5 | 22.7 ms | 174.0 us | **130.3x** | 525.8 us | 2.7 us | **197.8x** | r2 | -0.2571 | -0.2571 | +0.0000 |
| HuberRegressor | medium_10Kx100 | 613.9 ms | 40.6 ms | **15.1x** | 101.2 us | 68.5 us | 1.48x | r2 | 1.0000 | 1.0000 | -0.0000 |
| HuberRegressor | small_1Kx10 | 3.7 ms | 112.0 us | **32.8x** | 28.3 us | 2.0 us | **14.0x** | r2 | 1.0000 | 1.0000 | -0.0000 |
| HuberRegressor | tiny_50x5 | 5.4 ms | 4.5 us | **1192.8x** | 18.3 us | 770 ns | **23.8x** | r2 | 1.0000 | 1.0000 | +0.0000 |
| KNeighborsRegressor | medium_10Kx100 | 645.7 us | 13.9 ms | 0.05x | 17.6 ms | 46.7 ms | 0.38x | r2 | 0.3173 | 0.3173 | +0.0000 |
| KNeighborsRegressor | small_1Kx10 | 316.2 us | 370.7 us | 0.85x | 14.1 ms | 7.4 ms | 1.92x | r2 | 0.7790 | 0.7790 | +0.0000 |
| KNeighborsRegressor | tiny_50x5 | 100.0 us | 6.3 us | **16.0x** | 14.1 ms | 12.3 us | **1150.8x** | r2 | 0.6307 | 0.6307 | +0.0000 |
| KernelRidge | small_1Kx10 | 32.8 ms | 33.6 ms | 0.98x | 270.3 us | 1.1 ms | 0.25x | r2 | 1.0000 | 1.0000 | +0.0000 |
| KernelRidge | tiny_50x5 | 179.5 us | 11.8 us | **15.2x** | 103.4 us | 2.1 us | **48.8x** | r2 | 0.9988 | 0.9988 | +0.0000 |
| Lasso | medium_10Kx100 | 36.8 ms | 12.6 ms | **2.9x** | 89.7 us | 172.4 us | 0.52x | r2 | 0.9997 | 0.9997 | +0.0000 |
| Lasso | small_1Kx10 | 188.4 us | 94.5 us | 1.99x | 20.5 us | 18.1 us | 1.13x | r2 | 0.9994 | 0.9994 | -0.0000 |
| Lasso | tiny_50x5 | 173.5 us | 36.4 us | **4.8x** | 18.4 us | 18.0 us | 1.02x | r2 | 0.9995 | 0.9995 | +0.0000 |
| LinearRegression | medium_10Kx100 | 97.0 ms | 5.6 ms | **17.4x** | 99.7 us | 166.3 us | 0.60x | r2 | 1.0000 | 1.0000 | +0.0000 |
| LinearRegression | small_1Kx10 | 225.2 us | 55.9 us | **4.0x** | 22.9 us | 17.6 us | 1.30x | r2 | 1.0000 | 1.0000 | +0.0000 |
| LinearRegression | tiny_50x5 | 177.5 us | 33.5 us | **5.3x** | 17.7 us | 16.4 us | 1.08x | r2 | 1.0000 | 1.0000 | +0.0000 |
| QuantileRegressor | medium_10Kx100 | 1.46 s | 1.35 s | 1.08x | 73.3 us | 71.1 us | 1.03x | r2 | -0.0017 | 1.0000 | +1.0017 |
| QuantileRegressor | small_1Kx10 | 18.7 ms | 2.5 ms | **7.6x** | 19.3 us | 1.3 us | **14.6x** | r2 | -0.0112 | 1.0000 | +1.0112 |
| QuantileRegressor | tiny_50x5 | 2.2 ms | 3.4 us | **641.6x** | 18.6 us | 477 ns | **38.9x** | r2 | -0.0488 | -0.0717 | -0.0229 |
| RandomForestRegressor | medium_10Kx100 | 1.58 s | 1.64 s | 0.96x | 25.0 ms | 47.9 ms | 0.52x | r2 | 0.3759 | 0.3810 | +0.0051 |
| RandomForestRegressor | small_1Kx10 | 76.0 ms | 13.8 ms | **5.5x** | 25.3 ms | 1.5 ms | **16.7x** | r2 | 0.8446 | 0.8415 | -0.0031 |
| RandomForestRegressor | tiny_50x5 | 66.1 ms | 1.6 ms | **41.5x** | 14.6 ms | 16.8 us | **868.8x** | r2 | -0.7390 | -0.8137 | -0.0747 |
| Ridge | medium_10Kx100 | 26.1 ms | 15.3 ms | 1.71x | 104.1 us | 211.7 us | 0.49x | r2 | 1.0000 | 1.0000 | +0.0000 |
| Ridge | small_1Kx10 | 257.5 us | 57.8 us | **4.5x** | 19.0 us | 17.8 us | 1.07x | r2 | 1.0000 | 1.0000 | +0.0000 |
| Ridge | tiny_50x5 | 252.7 us | 33.7 us | **7.5x** | 17.3 us | 18.6 us | 0.93x | r2 | 0.9988 | 0.9988 | +0.0000 |

## Summary — geometric mean speedup

| Family | n compared | fit geomean | predict geomean | accuracy parity (mean Δ) |
|---|---:|---:|---:|---:|
| classifier | 51 | 6.51x | 9.07x | +0.0052 |
| cluster | 15 | 1.30x | — | +0.0000 |
| decomp | 15 | 6.28x | 5.08x | — |
| kernel | 6 | 10.48x | 1.41x | — |
| preprocess | 14 | 9.70x | 2.66x | — |
| regressor | 43 | 6.43x | 4.18x | +0.0464 |

