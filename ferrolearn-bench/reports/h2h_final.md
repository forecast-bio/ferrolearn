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
| AdaBoostClassifier | small_1Kx10 | 51.1 ms | 11.1 ms | **4.6x** | 1.6 ms | 45.2 us | **35.2x** | accuracy | 91.50% | 92.00% | +0.50pp |
| AdaBoostClassifier | tiny_50x5 | 24.2 ms | 198.6 us | **122.1x** | 1.3 ms | 2.9 us | **469.5x** | accuracy | 90.00% | 100.00% | +10.00pp |
| BaggingClassifier | small_1Kx10 | 11.3 ms | 2.6 ms | **4.4x** | 10.8 ms | 68.9 us | **156.7x** | accuracy | 95.00% | 96.00% | +1.00pp |
| BaggingClassifier | tiny_50x5 | 11.4 ms | 482.7 us | **23.5x** | 10.7 ms | 2.3 us | **4653.7x** | accuracy | 80.00% | 90.00% | +10.00pp |
| BernoulliNB | medium_10Kx100 | 7.2 ms | 1.6 ms | **4.5x** | 1.6 ms | 317.3 us | **5.0x** | accuracy | 75.00% | 75.00% | +0.00pp |
| BernoulliNB | small_1Kx10 | 511.5 us | 22.0 us | **23.3x** | 97.9 us | 3.4 us | **28.5x** | accuracy | 77.50% | 77.50% | +0.00pp |
| BernoulliNB | tiny_50x5 | 468.0 us | 5.1 us | **92.0x** | 84.7 us | 724 ns | **117.1x** | accuracy | 100.00% | 100.00% | +0.00pp |
| ComplementNB | medium_10Kx100 | 1.4 ms | 778.7 us | 1.80x | 123.6 us | 161.4 us | 0.77x | accuracy | 61.20% | 61.20% | +0.00pp |
| ComplementNB | small_1Kx10 | 444.3 us | 21.2 us | **21.0x** | 22.1 us | 2.6 us | **8.4x** | accuracy | 71.00% | 71.00% | +0.00pp |
| ComplementNB | tiny_50x5 | 360.1 us | 4.3 us | **83.3x** | 20.6 us | 671 ns | **30.7x** | accuracy | 30.00% | 30.00% | +0.00pp |
| DecisionTreeClassifier | medium_10Kx100 | 597.1 ms | 285.4 ms | **2.1x** | 229.0 us | 338.6 us | 0.68x | accuracy | 73.65% | 71.95% | -1.70pp |
| DecisionTreeClassifier | small_1Kx10 | 2.5 ms | 1.4 ms | 1.83x | 94.2 us | 22.3 us | **4.2x** | accuracy | 89.50% | 90.00% | +0.50pp |
| DecisionTreeClassifier | tiny_50x5 | 292.6 us | 86.2 us | **3.4x** | 40.5 us | 16.6 us | **2.4x** | accuracy | 90.00% | 80.00% | -10.00pp |
| ExtraTreeClassifier | medium_10Kx100 | 7.8 ms | 9.1 ms | 0.86x | 282.8 us | 263.3 us | 1.07x | accuracy | 64.20% | 63.45% | -0.75pp |
| ExtraTreeClassifier | small_1Kx10 | 847.1 us | 285.5 us | **3.0x** | 41.8 us | 8.1 us | **5.2x** | accuracy | 81.00% | 84.50% | +3.50pp |
| ExtraTreeClassifier | tiny_50x5 | 288.5 us | 13.6 us | **21.3x** | 25.6 us | 678 ns | **37.8x** | accuracy | 90.00% | 90.00% | +0.00pp |
| ExtraTreesClassifier | medium_10Kx100 | 129.0 ms | 63.2 ms | **2.0x** | 24.8 ms | 49.0 ms | 0.51x | accuracy | 93.90% | 93.75% | -0.15pp |
| ExtraTreesClassifier | small_1Kx10 | 89.2 ms | 3.7 ms | **24.0x** | 25.4 ms | 1.5 ms | **16.7x** | accuracy | 97.00% | 96.00% | -1.00pp |
| ExtraTreesClassifier | tiny_50x5 | 61.4 ms | 1.6 ms | **37.9x** | 14.1 ms | 12.1 us | **1163.2x** | accuracy | 90.00% | 90.00% | +0.00pp |
| GaussianNB | medium_10Kx100 | 4.1 ms | 2.1 ms | 1.89x | 607.7 us | 1.4 ms | 0.44x | accuracy | 81.55% | 81.55% | +0.00pp |
| GaussianNB | small_1Kx10 | 397.9 us | 140.8 us | **2.8x** | 49.9 us | 30.6 us | 1.63x | accuracy | 89.00% | 89.00% | +0.00pp |
| GaussianNB | tiny_50x5 | 250.0 us | 79.4 us | **3.1x** | 37.8 us | 18.2 us | **2.1x** | accuracy | 100.00% | 100.00% | +0.00pp |
| GradientBoostingClassifier | small_1Kx10 | 133.6 ms | 50.2 ms | **2.7x** | 243.6 us | 83.6 us | **2.9x** | accuracy | 96.00% | 94.00% | -2.00pp |
| GradientBoostingClassifier | tiny_50x5 | 20.4 ms | 487.8 us | **41.9x** | 73.8 us | 3.1 us | **23.8x** | accuracy | 80.00% | 80.00% | +0.00pp |
| HistGradientBoostingClassifier | medium_10Kx100 | 301.6 ms | 948.9 ms | 0.32x | 1.6 ms | 14.0 ms | 0.11x | accuracy | 95.80% | 95.80% | +0.00pp |
| HistGradientBoostingClassifier | small_1Kx10 | 120.3 ms | 39.8 ms | **3.0x** | 724.1 us | 913.4 us | 0.79x | accuracy | 96.00% | 94.00% | -2.00pp |
| HistGradientBoostingClassifier | tiny_50x5 | 23.1 ms | 271.3 us | **85.0x** | 643.1 us | 3.8 us | **170.6x** | accuracy | 100.00% | 100.00% | +0.00pp |
| KNeighborsClassifier | medium_10Kx100 | 918.9 us | 14.8 ms | 0.06x | 20.1 ms | 46.4 ms | 0.43x | accuracy | 96.60% | 96.60% | +0.00pp |
| KNeighborsClassifier | small_1Kx10 | 361.0 us | 520.5 us | 0.69x | 14.9 ms | 6.7 ms | **2.2x** | accuracy | 91.50% | 91.50% | +0.00pp |
| KNeighborsClassifier | tiny_50x5 | 191.9 us | 94.6 us | **2.0x** | 14.7 ms | 31.3 us | **470.0x** | accuracy | 90.00% | 90.00% | +0.00pp |
| LinearSVC | medium_10Kx100 | 45.1 ms | 21.9 ms | **2.1x** | 86.0 us | 75.1 us | 1.15x | accuracy | 83.60% | 83.70% | +0.10pp |
| LinearSVC | small_1Kx10 | 725.2 us | 334.7 us | **2.2x** | 35.7 us | 2.8 us | **12.6x** | accuracy | 83.50% | 83.50% | +0.00pp |
| LinearSVC | tiny_50x5 | 248.0 us | 6.1 us | **40.5x** | 25.4 us | 698 ns | **36.4x** | accuracy | 90.00% | 90.00% | +0.00pp |
| LogisticRegression | medium_10Kx100 | 717.6 ms | 31.2 ms | **23.0x** | 103.0 us | 193.0 us | 0.53x | accuracy | 83.50% | 83.45% | -0.05pp |
| LogisticRegression | small_1Kx10 | 765.7 us | 651.3 us | 1.18x | 26.7 us | 56.6 us | 0.47x | accuracy | 83.50% | 82.00% | -1.50pp |
| LogisticRegression | tiny_50x5 | 499.3 us | 96.3 us | **5.2x** | 30.5 us | 18.2 us | 1.67x | accuracy | 100.00% | 100.00% | +0.00pp |
| MultinomialNB | medium_10Kx100 | 2.1 ms | 1.2 ms | 1.67x | 154.4 us | 145.2 us | 1.06x | accuracy | 61.20% | 61.20% | +0.00pp |
| MultinomialNB | small_1Kx10 | 473.8 us | 23.5 us | **20.2x** | 24.5 us | 2.8 us | **8.7x** | accuracy | 70.50% | 70.50% | +0.00pp |
| MultinomialNB | tiny_50x5 | 386.1 us | 5.2 us | **74.2x** | 20.1 us | 657 ns | **30.6x** | accuracy | 30.00% | 30.00% | +0.00pp |
| NearestCentroid | medium_10Kx100 | 4.0 ms | 1.2 ms | **3.2x** | 540.3 us | 156.6 us | **3.4x** | accuracy | 69.15% | 69.15% | +0.00pp |
| NearestCentroid | small_1Kx10 | 309.1 us | 20.1 us | **15.4x** | 169.6 us | 2.3 us | **72.5x** | accuracy | 77.50% | 77.50% | +0.00pp |
| NearestCentroid | tiny_50x5 | 239.2 us | 11.8 us | **20.3x** | 236.2 us | 1.8 us | **131.4x** | accuracy | 100.00% | 100.00% | +0.00pp |
| QDA | medium_10Kx100 | — | — | — | — | — | — | accuracy | — | — | — |
| QDA | small_1Kx10 | — | — | — | — | — | — | accuracy | — | — | — |
| QDA | tiny_50x5 | — | — | — | — | — | — | accuracy | — | — | — |
| RandomForestClassifier | medium_10Kx100 | 276.0 ms | 193.5 ms | 1.43x | 25.0 ms | 29.6 ms | 0.85x | accuracy | 94.25% | 93.05% | -1.20pp |
| RandomForestClassifier | small_1Kx10 | 119.4 ms | 5.2 ms | **23.0x** | 25.2 ms | 874.0 us | **28.8x** | accuracy | 94.50% | 96.00% | +1.50pp |
| RandomForestClassifier | tiny_50x5 | 80.0 ms | 2.0 ms | **41.0x** | 14.0 ms | 32.3 us | **434.2x** | accuracy | 80.00% | 90.00% | +10.00pp |
| RidgeClassifier | medium_10Kx100 | 6.4 ms | 5.0 ms | 1.29x | 106.6 us | 124.9 us | 0.85x | accuracy | 83.35% | 83.35% | +0.00pp |
| RidgeClassifier | small_1Kx10 | 1.4 ms | 86.6 us | **16.1x** | 72.5 us | 5.6 us | **13.0x** | accuracy | 83.00% | 83.00% | +0.00pp |
| RidgeClassifier | tiny_50x5 | 675.9 us | 5.8 us | **115.9x** | 25.0 us | 743 ns | **33.6x** | accuracy | 90.00% | 90.00% | +0.00pp |

### Cluster — 15 comparisons

| Algorithm | Dataset | sklearn fit | ferrolearn fit | fit speedup | sklearn predict | ferrolearn predict | predict speedup | metric | sklearn | ferrolearn | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AgglomerativeClustering | small_1Kx10 | 6.3 ms | 132.5 ms | 0.05x | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |
| AgglomerativeClustering | tiny_200x5 | 554.3 us | 1.1 ms | 0.49x | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |
| Birch | small_1Kx10 | 22.7 ms | 5.1 ms | **4.4x** | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |
| Birch | tiny_200x5 | 3.3 ms | 902.1 us | **3.7x** | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |
| DBSCAN | small_1Kx10 | 3.1 ms | 2.3 ms | 1.36x | — | — | — | ari | 0.0000 | 0.0000 | +0.0000 |
| DBSCAN | tiny_200x5 | 577.9 us | 70.2 us | **8.2x** | — | — | — | ari | 0.0000 | 0.0000 | +0.0000 |
| GaussianMixture | medium_5Kx20 | 108.7 ms | 167.9 ms | 0.65x | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |
| GaussianMixture | small_1Kx10 | 9.0 ms | 6.3 ms | 1.42x | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |
| GaussianMixture | tiny_200x5 | 2.5 ms | 680.6 us | **3.7x** | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |
| KMeans | medium_5Kx20 | 44.0 ms | 30.6 ms | 1.44x | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |
| KMeans | small_1Kx10 | 9.5 ms | 2.4 ms | **3.9x** | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |
| KMeans | tiny_200x5 | 4.7 ms | 530.5 us | **8.8x** | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |
| MiniBatchKMeans | medium_5Kx20 | 12.8 ms | 31.4 ms | 0.41x | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |
| MiniBatchKMeans | small_1Kx10 | 16.0 ms | 18.4 ms | 0.87x | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |
| MiniBatchKMeans | tiny_200x5 | 2.1 ms | 6.4 ms | 0.33x | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |

### Decomp — 15 comparisons

| Algorithm | Dataset | sklearn fit | ferrolearn fit | fit speedup | sklearn predict | ferrolearn predict | predict speedup | metric | sklearn | ferrolearn | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FactorAnalysis | small_1Kx10 | 9.3 ms | 33.7 ms | 0.28x | 52.2 us | 19.4 us | **2.7x** | recon_rel | — | — | — |
| FactorAnalysis | tiny_50x5 | 1.5 ms | 158.9 us | **9.7x** | 35.7 us | 2.2 us | **16.1x** | recon_rel | — | — | — |
| FastICA | small_1Kx10 | 24.5 ms | 12.5 ms | 1.97x | 50.2 us | 17.6 us | **2.9x** | recon_rel | 6.787e-01 | — | — |
| FastICA | tiny_50x5 | 8.5 ms | 120.1 us | **70.7x** | 23.3 us | 2.1 us | **11.3x** | recon_rel | 3.394e-01 | — | — |
| IncrementalPCA | medium_10Kx100 | 194.7 ms | 119.3 ms | 1.63x | 8.0 ms | 1.4 ms | **5.7x** | recon_rel | 9.722e-01 | — | — |
| IncrementalPCA | small_1Kx10 | 2.3 ms | 53.1 us | **44.2x** | 35.1 us | 10.1 us | **3.5x** | recon_rel | 6.969e-01 | — | — |
| IncrementalPCA | tiny_50x5 | 325.7 us | 3.9 us | **83.3x** | 23.7 us | 1.6 us | **14.5x** | recon_rel | 3.483e-01 | — | — |
| PCA | medium_10Kx100 | 4.1 ms | 14.1 ms | 0.29x | 439.4 us | 1.9 ms | 0.23x | recon_rel | 9.698e-01 | 9.698e-01 | 1.00x |
| PCA | small_1Kx10 | 173.3 us | 50.6 us | **3.4x** | 30.5 us | 30.3 us | 1.01x | recon_rel | 6.787e-01 | 6.787e-01 | 1.00x |
| PCA | tiny_50x5 | 168.1 us | 24.3 us | **6.9x** | 24.9 us | 19.7 us | 1.26x | recon_rel | 3.394e-01 | 3.394e-01 | 1.00x |
| SparsePCA | small_1Kx10 | 258.2 ms | 384.0 ms | 0.67x | 348.1 us | 11.0 us | **31.7x** | recon_rel | 6.875e-01 | — | — |
| SparsePCA | tiny_50x5 | 9.4 ms | 1.1 ms | **8.2x** | 186.6 us | 1.6 us | **119.2x** | recon_rel | 3.541e-01 | — | — |
| TruncatedSVD | medium_10Kx100 | 19.0 ms | 7.8 ms | **2.4x** | 327.2 us | 870.6 us | 0.38x | recon_rel | 9.713e-01 | — | — |
| TruncatedSVD | small_1Kx10 | 700.5 us | 191.9 us | **3.7x** | 26.0 us | 7.9 us | **3.3x** | recon_rel | 6.790e-01 | — | — |
| TruncatedSVD | tiny_50x5 | 369.8 us | 5.8 us | **63.9x** | 21.6 us | 1.5 us | **14.0x** | recon_rel | 3.406e-01 | — | — |

### Kernel — 6 comparisons

| Algorithm | Dataset | sklearn fit | ferrolearn fit | fit speedup | sklearn predict | ferrolearn predict | predict speedup | metric | sklearn | ferrolearn | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Nystroem | medium_10Kx100 | 88.0 ms | 5.3 ms | **16.5x** | 23.7 ms | 61.3 ms | 0.39x | timing_only | — | — | — |
| Nystroem | small_1Kx10 | 8.0 ms | 4.2 ms | 1.91x | 8.0 ms | 5.6 ms | 1.43x | timing_only | — | — | — |
| Nystroem | tiny_50x5 | 800.5 us | 215.7 us | **3.7x** | 230.4 us | 77.0 us | **3.0x** | timing_only | — | — | — |
| RBFSampler | medium_10Kx100 | 400.1 us | 732.0 us | 0.55x | 23.1 ms | 15.4 ms | 1.50x | timing_only | — | — | — |
| RBFSampler | small_1Kx10 | 155.8 us | 5.1 us | **30.6x** | 1.1 ms | 1.0 ms | 1.07x | timing_only | — | — | — |
| RBFSampler | tiny_50x5 | 196.5 us | 4.0 us | **49.5x** | 94.9 us | 62.1 us | 1.53x | timing_only | — | — | — |

### Preprocess — 14 comparisons

| Algorithm | Dataset | sklearn fit | ferrolearn fit | fit speedup | sklearn predict | ferrolearn predict | predict speedup | metric | sklearn | ferrolearn | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MaxAbsScaler | medium_10Kx100 | 1.1 ms | 1.0 ms | 1.02x | 1.1 ms | 2.0 ms | 0.53x | rel_diff_vs_sklearn | 0.000e+00 | 0.000e+00 | — |
| MaxAbsScaler | small_1Kx10 | 67.8 us | 8.4 us | **8.0x** | 30.7 us | 11.9 us | **2.6x** | rel_diff_vs_sklearn | 0.000e+00 | 0.000e+00 | — |
| MaxAbsScaler | tiny_50x5 | 68.2 us | 1.2 us | **55.8x** | 34.7 us | 2.3 us | **15.3x** | rel_diff_vs_sklearn | 0.000e+00 | 0.000e+00 | — |
| MinMaxScaler | medium_10Kx100 | 1.0 ms | 1.8 ms | 0.57x | 1.3 ms | 2.1 ms | 0.61x | rel_diff_vs_sklearn | 0.000e+00 | 1.146e-16 | — |
| MinMaxScaler | small_1Kx10 | 97.2 us | 15.1 us | **6.5x** | 37.2 us | 11.9 us | **3.1x** | rel_diff_vs_sklearn | 0.000e+00 | 1.088e-16 | — |
| MinMaxScaler | tiny_50x5 | 89.6 us | 1.3 us | **71.5x** | 33.0 us | 2.3 us | **14.5x** | rel_diff_vs_sklearn | 0.000e+00 | 8.444e-17 | — |
| PowerTransformer | small_1Kx10 | 30.3 ms | 1.7 ms | **17.8x** | 386.4 us | 132.1 us | **2.9x** | rel_diff_vs_sklearn | 0.000e+00 | 4.203e-01 | — |
| PowerTransformer | tiny_50x5 | 15.7 ms | 43.1 us | **364.0x** | 101.6 us | 3.6 us | **27.9x** | rel_diff_vs_sklearn | 0.000e+00 | 4.189e-01 | — |
| RobustScaler | medium_10Kx100 | 25.0 ms | 14.2 ms | 1.76x | 1.4 ms | 1.9 ms | 0.72x | rel_diff_vs_sklearn | 0.000e+00 | 0.000e+00 | — |
| RobustScaler | small_1Kx10 | 625.0 us | 102.0 us | **6.1x** | 36.2 us | 11.9 us | **3.1x** | rel_diff_vs_sklearn | 0.000e+00 | 4.251e-20 | — |
| RobustScaler | tiny_50x5 | 462.0 us | 3.4 us | **135.6x** | 33.7 us | 2.2 us | **15.0x** | rel_diff_vs_sklearn | 0.000e+00 | 2.450e-17 | — |
| StandardScaler | medium_10Kx100 | 2.9 ms | 1.9 ms | 1.53x | 1.4 ms | 3.4 ms | 0.43x | rel_diff_vs_sklearn | 0.000e+00 | 0.000e+00 | — |
| StandardScaler | small_1Kx10 | 175.3 us | 28.7 us | **6.1x** | 43.2 us | 32.4 us | 1.33x | rel_diff_vs_sklearn | 0.000e+00 | 0.000e+00 | — |
| StandardScaler | tiny_50x5 | 183.0 us | 24.9 us | **7.3x** | 41.2 us | 27.3 us | 1.51x | rel_diff_vs_sklearn | 0.000e+00 | 0.000e+00 | — |

### Regressor — 43 comparisons

| Algorithm | Dataset | sklearn fit | ferrolearn fit | fit speedup | sklearn predict | ferrolearn predict | predict speedup | metric | sklearn | ferrolearn | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ARDRegression | medium_10Kx100 | 861.0 ms | 21.5 ms | **40.1x** | 106.3 us | 83.6 us | 1.27x | r2 | 1.0000 | 1.0000 | +0.0000 |
| ARDRegression | small_1Kx10 | 715.9 us | 51.2 us | **14.0x** | 20.0 us | 1.2 us | **17.3x** | r2 | 1.0000 | 1.0000 | +0.0000 |
| ARDRegression | tiny_50x5 | 707.9 us | 5.4 us | **130.6x** | 19.9 us | 522 ns | **38.1x** | r2 | 1.0000 | 1.0000 | +0.0000 |
| BayesianRidge | medium_10Kx100 | 196.5 ms | 15.3 ms | **12.8x** | 90.1 us | 82.4 us | 1.09x | r2 | 1.0000 | 1.0000 | +0.0000 |
| BayesianRidge | small_1Kx10 | 378.7 us | 58.5 us | **6.5x** | 20.5 us | 1.2 us | **17.3x** | r2 | 1.0000 | 1.0000 | +0.0000 |
| BayesianRidge | tiny_50x5 | 332.4 us | 5.4 us | **61.0x** | 16.7 us | 459 ns | **36.5x** | r2 | 1.0000 | 1.0000 | -0.0000 |
| DecisionTreeRegressor | medium_10Kx100 | 437.9 ms | 237.3 ms | 1.85x | 325.1 us | 246.5 us | 1.32x | r2 | -0.5324 | -0.5521 | -0.0197 |
| DecisionTreeRegressor | small_1Kx10 | 3.1 ms | 1.6 ms | 1.89x | 32.7 us | 6.1 us | **5.3x** | r2 | 0.6236 | 0.6146 | -0.0090 |
| DecisionTreeRegressor | tiny_50x5 | 240.9 us | 47.3 us | **5.1x** | 62.8 us | 587 ns | **106.9x** | r2 | -2.2859 | -2.1071 | +0.1788 |
| ElasticNet | medium_10Kx100 | 6.1 ms | 13.6 ms | 0.45x | 110.5 us | 194.6 us | 0.57x | r2 | 0.8870 | 0.8870 | -0.0000 |
| ElasticNet | small_1Kx10 | 235.1 us | 102.8 us | **2.3x** | 20.4 us | 17.5 us | 1.17x | r2 | 0.8761 | 0.8761 | -0.0000 |
| ElasticNet | tiny_50x5 | 183.2 us | 37.6 us | **4.9x** | 21.6 us | 17.0 us | 1.27x | r2 | 0.8193 | 0.8193 | -0.0000 |
| ExtraTreesRegressor | medium_10Kx100 | 554.6 ms | 427.4 ms | 1.30x | 25.3 ms | 64.1 ms | 0.40x | r2 | 0.3724 | 0.3690 | -0.0034 |
| ExtraTreesRegressor | small_1Kx10 | 50.2 ms | 6.5 ms | **7.7x** | 25.6 ms | 2.1 ms | **12.2x** | r2 | 0.8793 | 0.8875 | +0.0082 |
| ExtraTreesRegressor | tiny_50x5 | 45.3 ms | 1.7 ms | **26.7x** | 14.4 ms | 24.6 us | **584.7x** | r2 | 0.2541 | 0.1803 | -0.0738 |
| GradientBoostingRegressor | small_1Kx10 | 120.5 ms | 52.0 ms | **2.3x** | 237.4 us | 82.1 us | **2.9x** | r2 | 0.9268 | 0.9269 | +0.0001 |
| GradientBoostingRegressor | tiny_50x5 | 15.7 ms | 1.0 ms | **15.3x** | 71.0 us | 4.5 us | **15.9x** | r2 | -0.1346 | -0.1405 | -0.0059 |
| HistGradientBoostingRegressor | medium_10Kx100 | 272.2 ms | 980.6 ms | 0.28x | 1.4 ms | 12.8 ms | 0.11x | r2 | 0.6349 | 0.6349 | +0.0000 |
| HistGradientBoostingRegressor | small_1Kx10 | 130.3 ms | 40.9 ms | **3.2x** | 809.0 us | 1.2 ms | 0.68x | r2 | 0.9394 | 0.9405 | +0.0012 |
| HistGradientBoostingRegressor | tiny_50x5 | 23.5 ms | 186.8 us | **125.6x** | 567.3 us | 2.7 us | **208.7x** | r2 | -0.2571 | -0.2571 | +0.0000 |
| HuberRegressor | medium_10Kx100 | 976.6 ms | 44.4 ms | **22.0x** | 116.5 us | 81.2 us | 1.43x | r2 | 1.0000 | 1.0000 | -0.0000 |
| HuberRegressor | small_1Kx10 | 6.5 ms | 73.9 us | **88.3x** | 19.5 us | 1.2 us | **16.1x** | r2 | 1.0000 | 1.0000 | -0.0000 |
| HuberRegressor | tiny_50x5 | 4.6 ms | 4.9 us | **949.4x** | 17.5 us | 719 ns | **24.4x** | r2 | 1.0000 | 1.0000 | +0.0000 |
| KNeighborsRegressor | medium_10Kx100 | 686.5 us | 12.7 ms | 0.05x | 17.5 ms | 46.2 ms | 0.38x | r2 | 0.3173 | 0.3173 | +0.0000 |
| KNeighborsRegressor | small_1Kx10 | 301.3 us | 335.9 us | 0.90x | 14.3 ms | 7.5 ms | 1.90x | r2 | 0.7790 | 0.7790 | +0.0000 |
| KNeighborsRegressor | tiny_50x5 | 110.4 us | 6.4 us | **17.2x** | 14.3 ms | 12.2 us | **1172.2x** | r2 | 0.6307 | 0.6307 | +0.0000 |
| KernelRidge | small_1Kx10 | 40.9 ms | 33.0 ms | 1.24x | 267.9 us | 1.0 ms | 0.26x | r2 | 1.0000 | 1.0000 | +0.0000 |
| KernelRidge | tiny_50x5 | 167.9 us | 11.9 us | **14.1x** | 100.6 us | 2.2 us | **46.7x** | r2 | 0.9988 | 0.9988 | +0.0000 |
| Lasso | medium_10Kx100 | 43.8 ms | 17.1 ms | **2.6x** | 117.1 us | 200.3 us | 0.58x | r2 | 0.9997 | 0.9997 | +0.0000 |
| Lasso | small_1Kx10 | 200.6 us | 93.1 us | **2.2x** | 21.0 us | 18.9 us | 1.11x | r2 | 0.9994 | 0.9994 | -0.0000 |
| Lasso | tiny_50x5 | 188.2 us | 36.2 us | **5.2x** | 19.8 us | 16.8 us | 1.18x | r2 | 0.9995 | 0.9995 | +0.0000 |
| LinearRegression | medium_10Kx100 | 74.3 ms | 5.1 ms | **14.7x** | 96.0 us | 169.0 us | 0.57x | r2 | 1.0000 | 1.0000 | +0.0000 |
| LinearRegression | small_1Kx10 | 242.0 us | 56.7 us | **4.3x** | 17.9 us | 17.0 us | 1.05x | r2 | 1.0000 | 1.0000 | +0.0000 |
| LinearRegression | tiny_50x5 | 182.3 us | 35.0 us | **5.2x** | 18.1 us | 17.5 us | 1.03x | r2 | 1.0000 | 1.0000 | +0.0000 |
| QuantileRegressor | medium_10Kx100 | 1.44 s | 12.4 ms | **116.0x** | 84.0 us | 62.2 us | 1.35x | r2 | -0.0017 | -0.0012 | +0.0006 |
| QuantileRegressor | small_1Kx10 | 18.4 ms | 41.2 us | **447.5x** | 20.0 us | 1.2 us | **16.6x** | r2 | -0.0112 | -0.0194 | -0.0082 |
| QuantileRegressor | tiny_50x5 | 3.3 ms | 6.8 us | **476.6x** | 46.0 us | 1.4 us | **32.8x** | r2 | -0.0488 | -0.0717 | -0.0229 |
| RandomForestRegressor | medium_10Kx100 | 1.58 s | 1.68 s | 0.94x | 24.9 ms | 49.9 ms | 0.50x | r2 | 0.3759 | 0.3810 | +0.0051 |
| RandomForestRegressor | small_1Kx10 | 76.1 ms | 13.2 ms | **5.8x** | 25.3 ms | 1.6 ms | **16.2x** | r2 | 0.8446 | 0.8415 | -0.0031 |
| RandomForestRegressor | tiny_50x5 | 64.3 ms | 1.8 ms | **36.3x** | 14.1 ms | 24.3 us | **582.3x** | r2 | -0.7390 | -0.8137 | -0.0747 |
| Ridge | medium_10Kx100 | 24.2 ms | 13.5 ms | 1.79x | 97.3 us | 159.1 us | 0.61x | r2 | 1.0000 | 1.0000 | +0.0000 |
| Ridge | small_1Kx10 | 244.3 us | 59.9 us | **4.1x** | 18.7 us | 18.4 us | 1.01x | r2 | 1.0000 | 1.0000 | +0.0000 |
| Ridge | tiny_50x5 | 230.2 us | 34.5 us | **6.7x** | 17.7 us | 17.4 us | 1.02x | r2 | 0.9988 | 0.9988 | +0.0000 |

## Summary — geometric mean speedup

| Family | n compared | fit geomean | predict geomean | accuracy parity (mean Δ) |
|---|---:|---:|---:|---:|
| classifier | 51 | 6.75x | 8.88x | +0.0035 |
| cluster | 15 | 1.35x | — | +0.0000 |
| decomp | 15 | 5.16x | 4.56x | — |
| kernel | 6 | 6.78x | 1.26x | — |
| preprocess | 14 | 9.82x | 2.74x | — |
| regressor | 43 | 8.21x | 4.39x | -0.0006 |

