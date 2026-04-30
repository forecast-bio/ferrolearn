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
| AdaBoostClassifier | small_1Kx10 | 50.6 ms | 10.8 ms | **4.7x** | 1.5 ms | 73.4 us | **20.6x** | accuracy | 91.50% | 92.00% | +0.50pp |
| AdaBoostClassifier | tiny_50x5 | 23.7 ms | 196.8 us | **120.5x** | 1.3 ms | 2.9 us | **460.4x** | accuracy | 90.00% | 100.00% | +10.00pp |
| BaggingClassifier | small_1Kx10 | 11.3 ms | 2.5 ms | **4.6x** | 10.8 ms | 69.2 us | **156.2x** | accuracy | 95.00% | 96.00% | +1.00pp |
| BaggingClassifier | tiny_50x5 | 11.2 ms | 498.5 us | **22.4x** | 10.8 ms | 2.3 us | **4671.6x** | accuracy | 80.00% | 90.00% | +10.00pp |
| BernoulliNB | medium_10Kx100 | 6.8 ms | 1.6 ms | **4.4x** | 1.6 ms | 329.6 us | **4.8x** | accuracy | 75.00% | 75.00% | +0.00pp |
| BernoulliNB | small_1Kx10 | 507.3 us | 24.0 us | **21.1x** | 99.3 us | 3.8 us | **26.4x** | accuracy | 77.50% | 77.50% | +0.00pp |
| BernoulliNB | tiny_50x5 | 381.7 us | 5.4 us | **71.0x** | 77.8 us | 754 ns | **103.1x** | accuracy | 100.00% | 100.00% | +0.00pp |
| ComplementNB | medium_10Kx100 | 1.3 ms | 1.0 ms | 1.32x | 146.2 us | 145.1 us | 1.01x | accuracy | 61.20% | 61.20% | +0.00pp |
| ComplementNB | small_1Kx10 | 427.2 us | 21.7 us | **19.7x** | 21.7 us | 2.8 us | **7.9x** | accuracy | 71.00% | 71.00% | +0.00pp |
| ComplementNB | tiny_50x5 | 359.0 us | 4.3 us | **83.7x** | 20.1 us | 676 ns | **29.7x** | accuracy | 30.00% | 30.00% | +0.00pp |
| DecisionTreeClassifier | medium_10Kx100 | 599.0 ms | 286.9 ms | **2.1x** | 227.3 us | 290.4 us | 0.78x | accuracy | 73.65% | 71.95% | -1.70pp |
| DecisionTreeClassifier | small_1Kx10 | 2.7 ms | 1.8 ms | 1.49x | 85.0 us | 32.3 us | **2.6x** | accuracy | 89.50% | 90.00% | +0.50pp |
| DecisionTreeClassifier | tiny_50x5 | 298.8 us | 105.6 us | **2.8x** | 33.3 us | 17.9 us | 1.86x | accuracy | 90.00% | 80.00% | -10.00pp |
| ExtraTreeClassifier | medium_10Kx100 | 7.9 ms | 10.0 ms | 0.79x | 293.3 us | 254.4 us | 1.15x | accuracy | 64.20% | 63.05% | -1.15pp |
| ExtraTreeClassifier | small_1Kx10 | 662.0 us | 351.0 us | 1.89x | 43.7 us | 8.4 us | **5.2x** | accuracy | 81.00% | 80.50% | -0.50pp |
| ExtraTreeClassifier | tiny_50x5 | 263.8 us | 11.2 us | **23.5x** | 31.9 us | 735 ns | **43.4x** | accuracy | 90.00% | 100.00% | +10.00pp |
| ExtraTreesClassifier | medium_10Kx100 | 132.7 ms | 63.7 ms | **2.1x** | 24.7 ms | 49.7 ms | 0.50x | accuracy | 93.90% | 93.75% | -0.15pp |
| ExtraTreesClassifier | small_1Kx10 | 90.7 ms | 4.1 ms | **22.0x** | 24.9 ms | 1.5 ms | **16.3x** | accuracy | 97.00% | 96.00% | -1.00pp |
| ExtraTreesClassifier | tiny_50x5 | 54.5 ms | 1.5 ms | **36.3x** | 14.1 ms | 11.2 us | **1252.1x** | accuracy | 90.00% | 90.00% | +0.00pp |
| GaussianNB | medium_10Kx100 | 4.3 ms | 2.7 ms | 1.61x | 647.2 us | 1.5 ms | 0.42x | accuracy | 81.55% | 81.55% | +0.00pp |
| GaussianNB | small_1Kx10 | 378.2 us | 148.8 us | **2.5x** | 136.2 us | 31.8 us | **4.3x** | accuracy | 89.00% | 89.00% | +0.00pp |
| GaussianNB | tiny_50x5 | 245.7 us | 79.5 us | **3.1x** | 38.9 us | 17.4 us | **2.2x** | accuracy | 100.00% | 100.00% | +0.00pp |
| GradientBoostingClassifier | small_1Kx10 | 134.9 ms | 51.1 ms | **2.6x** | 246.6 us | 82.2 us | **3.0x** | accuracy | 96.00% | 94.00% | -2.00pp |
| GradientBoostingClassifier | tiny_50x5 | 21.1 ms | 489.4 us | **43.1x** | 71.5 us | 3.4 us | **20.9x** | accuracy | 80.00% | 80.00% | +0.00pp |
| HistGradientBoostingClassifier | medium_10Kx100 | 302.5 ms | 942.8 ms | 0.32x | 1.5 ms | 14.2 ms | 0.11x | accuracy | 95.80% | 95.80% | +0.00pp |
| HistGradientBoostingClassifier | small_1Kx10 | 115.1 ms | 39.5 ms | **2.9x** | 602.3 us | 905.5 us | 0.67x | accuracy | 96.00% | 94.00% | -2.00pp |
| HistGradientBoostingClassifier | tiny_50x5 | 23.0 ms | 207.6 us | **110.7x** | 987.4 us | 2.8 us | **356.5x** | accuracy | 100.00% | 100.00% | +0.00pp |
| KNeighborsClassifier | medium_10Kx100 | 922.4 us | 14.2 ms | 0.07x | 17.9 ms | 43.9 ms | 0.41x | accuracy | 96.60% | 96.60% | +0.00pp |
| KNeighborsClassifier | small_1Kx10 | 385.5 us | 520.3 us | 0.74x | 14.4 ms | 6.6 ms | **2.2x** | accuracy | 91.50% | 91.50% | +0.00pp |
| KNeighborsClassifier | tiny_50x5 | 183.5 us | 107.8 us | 1.70x | 14.3 ms | 34.1 us | **418.8x** | accuracy | 90.00% | 90.00% | +0.00pp |
| LinearSVC | medium_10Kx100 | 48.8 ms | 20.7 ms | **2.4x** | 88.3 us | 68.8 us | 1.28x | accuracy | 83.60% | 83.70% | +0.10pp |
| LinearSVC | small_1Kx10 | 546.0 us | 223.2 us | **2.4x** | 28.7 us | 1.7 us | **16.7x** | accuracy | 83.50% | 83.50% | +0.00pp |
| LinearSVC | tiny_50x5 | 238.2 us | 6.2 us | **38.3x** | 25.4 us | 703 ns | **36.1x** | accuracy | 90.00% | 90.00% | +0.00pp |
| LogisticRegression | medium_10Kx100 | 750.4 ms | 46.5 ms | **16.1x** | 115.2 us | 153.0 us | 0.75x | accuracy | 83.50% | 83.45% | -0.05pp |
| LogisticRegression | small_1Kx10 | 1.8 ms | 336.0 us | **5.3x** | 26.6 us | 19.6 us | 1.35x | accuracy | 83.50% | 82.00% | -1.50pp |
| LogisticRegression | tiny_50x5 | 506.8 us | 86.3 us | **5.9x** | 25.0 us | 19.9 us | 1.25x | accuracy | 100.00% | 100.00% | +0.00pp |
| MultinomialNB | medium_10Kx100 | 1.9 ms | 1.2 ms | 1.56x | 156.1 us | 145.3 us | 1.07x | accuracy | 61.20% | 61.20% | +0.00pp |
| MultinomialNB | small_1Kx10 | 442.3 us | 23.1 us | **19.2x** | 23.5 us | 2.7 us | **8.6x** | accuracy | 70.50% | 70.50% | +0.00pp |
| MultinomialNB | tiny_50x5 | 381.6 us | 4.9 us | **77.4x** | 20.0 us | 684 ns | **29.3x** | accuracy | 30.00% | 30.00% | +0.00pp |
| NearestCentroid | medium_10Kx100 | 4.0 ms | 1.3 ms | **3.0x** | 513.8 us | 154.9 us | **3.3x** | accuracy | 69.15% | 69.15% | +0.00pp |
| NearestCentroid | small_1Kx10 | 313.2 us | 21.9 us | **14.3x** | 176.5 us | 2.3 us | **77.4x** | accuracy | 77.50% | 77.50% | +0.00pp |
| NearestCentroid | tiny_50x5 | 248.3 us | 8.4 us | **29.5x** | 327.5 us | 1.1 us | **293.5x** | accuracy | 100.00% | 100.00% | +0.00pp |
| QDA | medium_10Kx100 | — | — | — | — | — | — | accuracy | — | — | — |
| QDA | small_1Kx10 | — | — | — | — | — | — | accuracy | — | — | — |
| QDA | tiny_50x5 | — | — | — | — | — | — | accuracy | — | — | — |
| RandomForestClassifier | medium_10Kx100 | 265.6 ms | 195.4 ms | 1.36x | 24.8 ms | 29.9 ms | 0.83x | accuracy | 94.25% | 93.05% | -1.20pp |
| RandomForestClassifier | small_1Kx10 | 123.4 ms | 5.0 ms | **24.9x** | 24.8 ms | 867.1 us | **28.6x** | accuracy | 94.50% | 96.00% | +1.50pp |
| RandomForestClassifier | tiny_50x5 | 77.0 ms | 1.7 ms | **45.7x** | 14.1 ms | 25.5 us | **550.7x** | accuracy | 80.00% | 90.00% | +10.00pp |
| RidgeClassifier | medium_10Kx100 | 6.3 ms | 4.9 ms | 1.29x | 122.7 us | 139.1 us | 0.88x | accuracy | 83.35% | 83.35% | +0.00pp |
| RidgeClassifier | small_1Kx10 | 739.4 us | 47.6 us | **15.5x** | 71.6 us | 2.1 us | **33.8x** | accuracy | 83.00% | 83.00% | +0.00pp |
| RidgeClassifier | tiny_50x5 | 644.6 us | 5.7 us | **112.5x** | 25.4 us | 810 ns | **31.4x** | accuracy | 90.00% | 90.00% | +0.00pp |

### Cluster — 15 comparisons

| Algorithm | Dataset | sklearn fit | ferrolearn fit | fit speedup | sklearn predict | ferrolearn predict | predict speedup | metric | sklearn | ferrolearn | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AgglomerativeClustering | small_1Kx10 | 6.7 ms | 131.6 ms | 0.05x | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |
| AgglomerativeClustering | tiny_200x5 | 475.7 us | 1.1 ms | 0.43x | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |
| Birch | small_1Kx10 | 24.1 ms | 5.1 ms | **4.7x** | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |
| Birch | tiny_200x5 | 4.1 ms | 831.3 us | **5.0x** | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |
| DBSCAN | small_1Kx10 | 3.1 ms | 2.3 ms | 1.36x | — | — | — | ari | 0.0000 | 0.0000 | +0.0000 |
| DBSCAN | tiny_200x5 | 479.9 us | 70.2 us | **6.8x** | — | — | — | ari | 0.0000 | 0.0000 | +0.0000 |
| GaussianMixture | medium_5Kx20 | 57.5 ms | 574.0 ms | 0.10x | — | — | — | ari | 1.0000 | 0.8425 | -0.1575 |
| GaussianMixture | small_1Kx10 | 6.3 ms | 9.5 ms | 0.66x | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |
| GaussianMixture | tiny_200x5 | 1.9 ms | 1.2 ms | 1.60x | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |
| KMeans | medium_5Kx20 | 57.3 ms | 87.8 ms | 0.65x | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |
| KMeans | small_1Kx10 | 9.5 ms | 3.8 ms | **2.5x** | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |
| KMeans | tiny_200x5 | 4.5 ms | 220.8 us | **20.3x** | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |
| MiniBatchKMeans | medium_5Kx20 | 13.8 ms | 31.2 ms | 0.44x | — | — | — | ari | 1.0000 | 0.8348 | -0.1652 |
| MiniBatchKMeans | small_1Kx10 | 15.3 ms | 19.1 ms | 0.80x | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |
| MiniBatchKMeans | tiny_200x5 | 2.0 ms | 6.1 ms | 0.33x | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |

### Decomp — 15 comparisons

| Algorithm | Dataset | sklearn fit | ferrolearn fit | fit speedup | sklearn predict | ferrolearn predict | predict speedup | metric | sklearn | ferrolearn | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FactorAnalysis | small_1Kx10 | 9.7 ms | 33.5 ms | 0.29x | 51.6 us | 19.9 us | **2.6x** | recon_rel | — | — | — |
| FactorAnalysis | tiny_50x5 | 1.6 ms | 163.4 us | **9.7x** | 33.4 us | 2.1 us | **15.6x** | recon_rel | — | — | — |
| FastICA | small_1Kx10 | 25.3 ms | 13.0 ms | 1.95x | 53.4 us | 17.9 us | **3.0x** | recon_rel | 6.787e-01 | — | — |
| FastICA | tiny_50x5 | 8.0 ms | 122.2 us | **65.7x** | 23.9 us | 1.9 us | **12.6x** | recon_rel | 3.394e-01 | — | — |
| IncrementalPCA | medium_10Kx100 | 180.3 ms | 143.1 ms | 1.26x | 8.0 ms | 1.3 ms | **5.9x** | recon_rel | 9.722e-01 | — | — |
| IncrementalPCA | small_1Kx10 | 2.5 ms | 51.1 us | **48.0x** | 37.6 us | 10.0 us | **3.8x** | recon_rel | 6.969e-01 | — | — |
| IncrementalPCA | tiny_50x5 | 337.3 us | 3.9 us | **86.9x** | 22.7 us | 1.5 us | **14.7x** | recon_rel | 3.483e-01 | — | — |
| PCA | medium_10Kx100 | 3.7 ms | 14.6 ms | 0.26x | 418.4 us | 2.0 ms | 0.21x | recon_rel | 9.698e-01 | 9.698e-01 | 1.00x |
| PCA | small_1Kx10 | 184.9 us | 50.9 us | **3.6x** | 34.9 us | 28.9 us | 1.21x | recon_rel | 6.787e-01 | 6.787e-01 | 1.00x |
| PCA | tiny_50x5 | 145.7 us | 32.8 us | **4.4x** | 22.3 us | 19.7 us | 1.13x | recon_rel | 3.394e-01 | 3.394e-01 | 1.00x |
| SparsePCA | small_1Kx10 | 258.2 ms | 427.9 ms | 0.60x | 388.7 us | 11.0 us | **35.4x** | recon_rel | 6.875e-01 | — | — |
| SparsePCA | tiny_50x5 | 9.3 ms | 1.8 ms | **5.2x** | 194.0 us | 1.6 us | **120.8x** | recon_rel | 3.541e-01 | — | — |
| TruncatedSVD | medium_10Kx100 | 53.9 ms | 12.8 ms | **4.2x** | 415.4 us | 1.7 ms | 0.24x | recon_rel | 9.713e-01 | — | — |
| TruncatedSVD | small_1Kx10 | 732.4 us | 196.9 us | **3.7x** | 26.1 us | 9.5 us | **2.7x** | recon_rel | 6.790e-01 | — | — |
| TruncatedSVD | tiny_50x5 | 380.5 us | 5.7 us | **66.3x** | 19.4 us | 1.5 us | **12.6x** | recon_rel | 3.406e-01 | — | — |

### Kernel — 6 comparisons

| Algorithm | Dataset | sklearn fit | ferrolearn fit | fit speedup | sklearn predict | ferrolearn predict | predict speedup | metric | sklearn | ferrolearn | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Nystroem | medium_10Kx100 | 8.0 ms | 7.3 ms | 1.09x | 24.0 ms | 61.5 ms | 0.39x | timing_only | — | — | — |
| Nystroem | small_1Kx10 | 14.2 ms | 4.9 ms | **2.9x** | 8.0 ms | 5.6 ms | 1.44x | timing_only | — | — | — |
| Nystroem | tiny_50x5 | 1.1 ms | 224.2 us | **4.8x** | 247.0 us | 76.6 us | **3.2x** | timing_only | — | — | — |
| RBFSampler | medium_10Kx100 | 424.3 us | 521.0 us | 0.81x | 19.3 ms | 15.5 ms | 1.25x | timing_only | — | — | — |
| RBFSampler | small_1Kx10 | 138.1 us | 5.2 us | **26.6x** | 1.1 ms | 1.1 ms | 1.07x | timing_only | — | — | — |
| RBFSampler | tiny_50x5 | 185.9 us | 4.2 us | **44.5x** | 100.6 us | 51.1 us | 1.97x | timing_only | — | — | — |

### Preprocess — 14 comparisons

| Algorithm | Dataset | sklearn fit | ferrolearn fit | fit speedup | sklearn predict | ferrolearn predict | predict speedup | metric | sklearn | ferrolearn | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MaxAbsScaler | medium_10Kx100 | 1.1 ms | 1.1 ms | 0.95x | 1.0 ms | 2.1 ms | 0.48x | rel_diff_vs_sklearn | 0.000e+00 | 0.000e+00 | — |
| MaxAbsScaler | small_1Kx10 | 64.9 us | 8.5 us | **7.6x** | 30.5 us | 12.0 us | **2.5x** | rel_diff_vs_sklearn | 0.000e+00 | 0.000e+00 | — |
| MaxAbsScaler | tiny_50x5 | 133.0 us | 1.9 us | **68.9x** | 59.4 us | 4.3 us | **13.9x** | rel_diff_vs_sklearn | 0.000e+00 | 0.000e+00 | — |
| MinMaxScaler | medium_10Kx100 | 957.6 us | 2.0 ms | 0.49x | 1.3 ms | 2.2 ms | 0.56x | rel_diff_vs_sklearn | 0.000e+00 | 1.146e-16 | — |
| MinMaxScaler | small_1Kx10 | 100.1 us | 15.2 us | **6.6x** | 37.0 us | 12.0 us | **3.1x** | rel_diff_vs_sklearn | 0.000e+00 | 1.088e-16 | — |
| MinMaxScaler | tiny_50x5 | 166.7 us | 2.2 us | **75.7x** | 67.7 us | 4.3 us | **15.7x** | rel_diff_vs_sklearn | 0.000e+00 | 8.444e-17 | — |
| PowerTransformer | small_1Kx10 | 32.1 ms | 1.7 ms | **18.8x** | 399.4 us | 119.5 us | **3.3x** | rel_diff_vs_sklearn | 0.000e+00 | 4.203e-01 | — |
| PowerTransformer | tiny_50x5 | 15.1 ms | 53.7 us | **281.1x** | 114.2 us | 4.0 us | **28.5x** | rel_diff_vs_sklearn | 0.000e+00 | 4.189e-01 | — |
| RobustScaler | medium_10Kx100 | 24.6 ms | 14.3 ms | 1.72x | 1.4 ms | 1.9 ms | 0.73x | rel_diff_vs_sklearn | 0.000e+00 | 0.000e+00 | — |
| RobustScaler | small_1Kx10 | 706.3 us | 121.6 us | **5.8x** | 37.1 us | 11.9 us | **3.1x** | rel_diff_vs_sklearn | 0.000e+00 | 4.251e-20 | — |
| RobustScaler | tiny_50x5 | 416.8 us | 3.6 us | **117.1x** | 31.5 us | 2.3 us | **13.8x** | rel_diff_vs_sklearn | 0.000e+00 | 2.450e-17 | — |
| StandardScaler | medium_10Kx100 | 2.9 ms | 2.0 ms | 1.44x | 1.3 ms | 3.2 ms | 0.41x | rel_diff_vs_sklearn | 0.000e+00 | 0.000e+00 | — |
| StandardScaler | small_1Kx10 | 170.6 us | 29.8 us | **5.7x** | 54.2 us | 31.3 us | 1.73x | rel_diff_vs_sklearn | 0.000e+00 | 0.000e+00 | — |
| StandardScaler | tiny_50x5 | 263.4 us | 53.0 us | **5.0x** | 73.7 us | 49.2 us | 1.50x | rel_diff_vs_sklearn | 0.000e+00 | 0.000e+00 | — |

### Regressor — 43 comparisons

| Algorithm | Dataset | sklearn fit | ferrolearn fit | fit speedup | sklearn predict | ferrolearn predict | predict speedup | metric | sklearn | ferrolearn | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ARDRegression | medium_10Kx100 | 534.3 ms | 24.2 ms | **22.0x** | 90.2 us | 83.2 us | 1.08x | r2 | 1.0000 | 1.0000 | +0.0000 |
| ARDRegression | small_1Kx10 | 725.7 us | 54.8 us | **13.2x** | 18.8 us | 1.2 us | **15.5x** | r2 | 1.0000 | 1.0000 | +0.0000 |
| ARDRegression | tiny_50x5 | 749.9 us | 4.8 us | **157.6x** | 21.3 us | 553 ns | **38.5x** | r2 | 1.0000 | 1.0000 | +0.0000 |
| BayesianRidge | medium_10Kx100 | 195.0 ms | 44.5 ms | **4.4x** | 127.4 us | 72.4 us | 1.76x | r2 | 1.0000 | 1.0000 | +0.0000 |
| BayesianRidge | small_1Kx10 | 396.1 us | 60.9 us | **6.5x** | 19.7 us | 1.2 us | **16.9x** | r2 | 1.0000 | 1.0000 | +0.0000 |
| BayesianRidge | tiny_50x5 | 345.3 us | 4.8 us | **71.7x** | 19.1 us | 481 ns | **39.8x** | r2 | 1.0000 | 1.0000 | -0.0000 |
| DecisionTreeRegressor | medium_10Kx100 | 440.9 ms | 241.7 ms | 1.82x | 295.1 us | 323.8 us | 0.91x | r2 | -0.5324 | -0.5521 | -0.0197 |
| DecisionTreeRegressor | small_1Kx10 | 3.3 ms | 1.7 ms | 1.93x | 43.5 us | 6.9 us | **6.3x** | r2 | 0.6236 | 0.6146 | -0.0090 |
| DecisionTreeRegressor | tiny_50x5 | 245.0 us | 21.6 us | **11.3x** | 24.5 us | 526 ns | **46.6x** | r2 | -2.2859 | -2.1071 | +0.1788 |
| ElasticNet | medium_10Kx100 | 9.1 ms | 11.7 ms | 0.77x | 91.6 us | 183.5 us | 0.50x | r2 | 0.8870 | 0.8870 | -0.0000 |
| ElasticNet | small_1Kx10 | 196.1 us | 98.2 us | 2.00x | 20.6 us | 18.8 us | 1.10x | r2 | 0.8761 | 0.8761 | -0.0000 |
| ElasticNet | tiny_50x5 | 191.6 us | 37.3 us | **5.1x** | 21.5 us | 17.0 us | 1.26x | r2 | 0.8193 | 0.8193 | -0.0000 |
| ExtraTreesRegressor | medium_10Kx100 | 557.5 ms | 400.6 ms | 1.39x | 24.9 ms | 65.8 ms | 0.38x | r2 | 0.3724 | 0.3690 | -0.0034 |
| ExtraTreesRegressor | small_1Kx10 | 48.1 ms | 6.6 ms | **7.3x** | 24.8 ms | 2.0 ms | **12.1x** | r2 | 0.8793 | 0.8875 | +0.0082 |
| ExtraTreesRegressor | tiny_50x5 | 40.1 ms | 1.8 ms | **22.5x** | 14.0 ms | 23.4 us | **599.0x** | r2 | 0.2541 | 0.1803 | -0.0738 |
| GradientBoostingRegressor | small_1Kx10 | 124.9 ms | 53.4 ms | **2.3x** | 243.9 us | 76.7 us | **3.2x** | r2 | 0.9268 | 0.9269 | +0.0001 |
| GradientBoostingRegressor | tiny_50x5 | 16.8 ms | 1.0 ms | **16.1x** | 77.6 us | 4.8 us | **16.2x** | r2 | -0.1346 | -0.1405 | -0.0059 |
| HistGradientBoostingRegressor | medium_10Kx100 | 300.3 ms | 953.1 ms | 0.32x | 1.6 ms | 12.9 ms | 0.12x | r2 | 0.6349 | 0.6349 | +0.0000 |
| HistGradientBoostingRegressor | small_1Kx10 | 117.1 ms | 41.5 ms | **2.8x** | 733.0 us | 1.2 ms | 0.63x | r2 | 0.9394 | 0.9405 | +0.0012 |
| HistGradientBoostingRegressor | tiny_50x5 | 23.0 ms | 177.5 us | **129.7x** | 595.5 us | 2.6 us | **226.1x** | r2 | -0.2571 | -0.2571 | +0.0000 |
| HuberRegressor | medium_10Kx100 | 747.6 ms | 39.7 ms | **18.8x** | 91.1 us | 73.5 us | 1.24x | r2 | 1.0000 | 1.0000 | -0.0000 |
| HuberRegressor | small_1Kx10 | 3.8 ms | 71.7 us | **52.3x** | 19.2 us | 1.2 us | **16.1x** | r2 | 1.0000 | 1.0000 | -0.0000 |
| HuberRegressor | tiny_50x5 | 3.0 ms | 4.8 us | **625.9x** | 18.0 us | 717 ns | **25.1x** | r2 | 1.0000 | 1.0000 | +0.0000 |
| KNeighborsRegressor | medium_10Kx100 | 592.0 us | 13.1 ms | 0.05x | 16.3 ms | 45.4 ms | 0.36x | r2 | 0.3173 | 0.3173 | +0.0000 |
| KNeighborsRegressor | small_1Kx10 | 307.3 us | 337.3 us | 0.91x | 14.4 ms | 7.3 ms | 1.97x | r2 | 0.7790 | 0.7790 | +0.0000 |
| KNeighborsRegressor | tiny_50x5 | 171.5 us | 6.2 us | **27.4x** | 14.0 ms | 12.1 us | **1150.8x** | r2 | 0.6307 | 0.6307 | +0.0000 |
| KernelRidge | small_1Kx10 | 95.1 ms | 38.1 ms | **2.5x** | 281.8 us | 2.2 ms | 0.13x | r2 | 1.0000 | 1.0000 | +0.0000 |
| KernelRidge | tiny_50x5 | 167.6 us | 12.6 us | **13.3x** | 116.3 us | 2.4 us | **49.4x** | r2 | 0.9988 | 0.9988 | +0.0000 |
| Lasso | medium_10Kx100 | 33.0 ms | 12.2 ms | **2.7x** | 94.5 us | 177.5 us | 0.53x | r2 | 0.9997 | 0.9997 | +0.0000 |
| Lasso | small_1Kx10 | 213.2 us | 93.2 us | **2.3x** | 20.3 us | 17.6 us | 1.15x | r2 | 0.9994 | 0.9994 | -0.0000 |
| Lasso | tiny_50x5 | 193.6 us | 38.2 us | **5.1x** | 18.9 us | 17.7 us | 1.07x | r2 | 0.9995 | 0.9995 | +0.0000 |
| LinearRegression | medium_10Kx100 | 82.5 ms | 5.5 ms | **15.1x** | 101.4 us | 184.0 us | 0.55x | r2 | 1.0000 | 1.0000 | +0.0000 |
| LinearRegression | small_1Kx10 | 256.7 us | 57.9 us | **4.4x** | 17.6 us | 17.0 us | 1.03x | r2 | 1.0000 | 1.0000 | +0.0000 |
| LinearRegression | tiny_50x5 | 190.9 us | 35.4 us | **5.4x** | 20.8 us | 18.0 us | 1.16x | r2 | 1.0000 | 1.0000 | +0.0000 |
| QuantileRegressor | medium_10Kx100 | 1.44 s | 1.29 s | 1.11x | 84.2 us | 62.0 us | 1.36x | r2 | -0.0017 | 1.0000 | +1.0017 |
| QuantileRegressor | small_1Kx10 | 17.7 ms | 2.4 ms | **7.3x** | 23.2 us | 1.4 us | **16.8x** | r2 | -0.0112 | 1.0000 | +1.0112 |
| QuantileRegressor | tiny_50x5 | 1.9 ms | 3.6 us | **523.9x** | 18.2 us | 473 ns | **38.4x** | r2 | -0.0488 | -0.0717 | -0.0229 |
| RandomForestRegressor | medium_10Kx100 | 1.58 s | 1.69 s | 0.94x | 25.0 ms | 49.6 ms | 0.50x | r2 | 0.3759 | 0.3810 | +0.0051 |
| RandomForestRegressor | small_1Kx10 | 75.0 ms | 12.9 ms | **5.8x** | 25.1 ms | 1.5 ms | **17.1x** | r2 | 0.8446 | 0.8415 | -0.0031 |
| RandomForestRegressor | tiny_50x5 | 63.2 ms | 1.5 ms | **40.8x** | 14.5 ms | 53.6 us | **270.7x** | r2 | -0.7390 | -0.8137 | -0.0747 |
| Ridge | medium_10Kx100 | 18.9 ms | 16.3 ms | 1.16x | 183.1 us | 173.5 us | 1.06x | r2 | 1.0000 | 1.0000 | +0.0000 |
| Ridge | small_1Kx10 | 300.3 us | 56.8 us | **5.3x** | 17.9 us | 24.2 us | 0.74x | r2 | 1.0000 | 1.0000 | +0.0000 |
| Ridge | tiny_50x5 | 280.9 us | 33.7 us | **8.3x** | 18.9 us | 18.5 us | 1.02x | r2 | 0.9988 | 0.9988 | +0.0000 |

## Summary — geometric mean speedup

| Family | n compared | fit geomean | predict geomean | accuracy parity (mean Δ) |
|---|---:|---:|---:|---:|
| classifier | 51 | 6.74x | 9.57x | +0.0047 |
| cluster | 15 | 1.04x | — | -0.0215 |
| decomp | 15 | 4.94x | 4.44x | — |
| kernel | 6 | 4.94x | 1.30x | — |
| preprocess | 14 | 9.20x | 2.76x | — |
| regressor | 43 | 6.72x | 4.20x | +0.0464 |

