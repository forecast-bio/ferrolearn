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
| AdaBoostClassifier | small_1Kx10 | 51.3 ms | 10.8 ms | **4.7x** | 1.5 ms | 168.5 us | **9.1x** | accuracy | 91.50% | 72.50% | -19.00pp |
| AdaBoostClassifier | tiny_50x5 | 25.0 ms | 284.1 us | **87.8x** | 1.3 ms | 8.9 us | **150.6x** | accuracy | 90.00% | 100.00% | +10.00pp |
| BaggingClassifier | small_1Kx10 | 11.3 ms | 2.3 ms | **5.0x** | 10.8 ms | 74.4 us | **145.5x** | accuracy | 95.00% | 96.00% | +1.00pp |
| BaggingClassifier | tiny_50x5 | 11.3 ms | 446.1 us | **25.4x** | 10.8 ms | 2.4 us | **4545.7x** | accuracy | 80.00% | 90.00% | +10.00pp |
| BernoulliNB | medium_10Kx100 | 7.0 ms | 1.5 ms | **4.6x** | 1.6 ms | 320.7 us | **4.9x** | accuracy | 75.00% | 75.00% | +0.00pp |
| BernoulliNB | small_1Kx10 | 540.7 us | 22.7 us | **23.8x** | 109.2 us | 3.5 us | **31.5x** | accuracy | 77.50% | 77.50% | +0.00pp |
| BernoulliNB | tiny_50x5 | 418.5 us | 4.8 us | **87.2x** | 90.9 us | 783 ns | **116.1x** | accuracy | 100.00% | 100.00% | +0.00pp |
| ComplementNB | medium_10Kx100 | 1.6 ms | 942.7 us | 1.68x | 132.2 us | 146.9 us | 0.90x | accuracy | 61.20% | 61.20% | +0.00pp |
| ComplementNB | small_1Kx10 | 471.0 us | 22.9 us | **20.5x** | 27.5 us | 2.7 us | **10.1x** | accuracy | 71.00% | 71.00% | +0.00pp |
| ComplementNB | tiny_50x5 | 348.4 us | 4.1 us | **84.1x** | 19.6 us | 698 ns | **28.1x** | accuracy | 30.00% | 30.00% | +0.00pp |
| DecisionTreeClassifier | medium_10Kx100 | 596.7 ms | 288.1 ms | **2.1x** | 272.0 us | 309.3 us | 0.88x | accuracy | 73.65% | 71.95% | -1.70pp |
| DecisionTreeClassifier | small_1Kx10 | 2.5 ms | 1.3 ms | 1.93x | 44.8 us | 22.7 us | 1.97x | accuracy | 89.50% | 90.00% | +0.50pp |
| DecisionTreeClassifier | tiny_50x5 | 302.2 us | 83.6 us | **3.6x** | 33.1 us | 17.7 us | 1.87x | accuracy | 90.00% | 80.00% | -10.00pp |
| ExtraTreeClassifier | medium_10Kx100 | 8.1 ms | 10.0 ms | 0.81x | 281.2 us | 262.2 us | 1.07x | accuracy | 64.20% | 62.00% | -2.20pp |
| ExtraTreeClassifier | small_1Kx10 | 508.1 us | 272.0 us | 1.87x | 46.3 us | 6.3 us | **7.4x** | accuracy | 81.00% | 82.00% | +1.00pp |
| ExtraTreeClassifier | tiny_50x5 | 284.2 us | 11.3 us | **25.2x** | 32.7 us | 707 ns | **46.2x** | accuracy | 90.00% | 90.00% | +0.00pp |
| ExtraTreesClassifier | medium_10Kx100 | 127.2 ms | 62.9 ms | **2.0x** | 24.8 ms | 52.9 ms | 0.47x | accuracy | 93.90% | 93.75% | -0.15pp |
| ExtraTreesClassifier | small_1Kx10 | 89.5 ms | 4.0 ms | **22.5x** | 25.3 ms | 1.5 ms | **17.0x** | accuracy | 97.00% | 96.00% | -1.00pp |
| ExtraTreesClassifier | tiny_50x5 | 59.1 ms | 1.7 ms | **34.5x** | 14.1 ms | 12.4 us | **1139.7x** | accuracy | 90.00% | 90.00% | +0.00pp |
| GaussianNB | medium_10Kx100 | 4.3 ms | 2.9 ms | 1.47x | 682.6 us | 1.5 ms | 0.47x | accuracy | 81.55% | 81.55% | +0.00pp |
| GaussianNB | small_1Kx10 | 379.6 us | 156.5 us | **2.4x** | 51.0 us | 30.5 us | 1.67x | accuracy | 89.00% | 89.00% | +0.00pp |
| GaussianNB | tiny_50x5 | 250.5 us | 75.8 us | **3.3x** | 38.5 us | 17.0 us | **2.3x** | accuracy | 100.00% | 100.00% | +0.00pp |
| GradientBoostingClassifier | small_1Kx10 | 135.5 ms | 50.7 ms | **2.7x** | 234.1 us | 83.1 us | **2.8x** | accuracy | 96.00% | 94.00% | -2.00pp |
| GradientBoostingClassifier | tiny_50x5 | 21.4 ms | 550.1 us | **38.9x** | 70.7 us | 3.3 us | **21.7x** | accuracy | 80.00% | 80.00% | +0.00pp |
| HistGradientBoostingClassifier | medium_10Kx100 | 306.3 ms | 950.8 ms | 0.32x | 1.5 ms | 14.4 ms | 0.10x | accuracy | 95.80% | 95.80% | +0.00pp |
| HistGradientBoostingClassifier | small_1Kx10 | 119.0 ms | 39.9 ms | **3.0x** | 710.7 us | 893.7 us | 0.80x | accuracy | 96.00% | 94.00% | -2.00pp |
| HistGradientBoostingClassifier | tiny_50x5 | 23.1 ms | 264.3 us | **87.4x** | 650.1 us | 3.7 us | **176.2x** | accuracy | 100.00% | 100.00% | +0.00pp |
| KNeighborsClassifier | medium_10Kx100 | 856.3 us | 14.8 ms | 0.06x | 17.6 ms | 46.0 ms | 0.38x | accuracy | 96.60% | 96.60% | +0.00pp |
| KNeighborsClassifier | small_1Kx10 | 364.3 us | 516.5 us | 0.71x | 15.0 ms | 6.8 ms | **2.2x** | accuracy | 91.50% | 91.50% | +0.00pp |
| KNeighborsClassifier | tiny_50x5 | 202.9 us | 94.5 us | **2.1x** | 15.2 ms | 29.5 us | **515.2x** | accuracy | 90.00% | 90.00% | +0.00pp |
| LinearSVC | medium_10Kx100 | 46.1 ms | 14.14 s | 0.00x | 95.7 us | 74.8 us | 1.28x | accuracy | 83.60% | 62.55% | -21.05pp |
| LinearSVC | small_1Kx10 | 632.4 us | 11.4 ms | 0.06x | 35.0 us | 1.8 us | **19.5x** | accuracy | 83.50% | 83.00% | -0.50pp |
| LinearSVC | tiny_50x5 | 242.4 us | 89.9 us | **2.7x** | 27.2 us | 745 ns | **36.6x** | accuracy | 90.00% | 90.00% | +0.00pp |
| LogisticRegression | medium_10Kx100 | 837.1 ms | 22.5 ms | **37.3x** | 108.0 us | 190.9 us | 0.57x | accuracy | 83.50% | 83.45% | -0.05pp |
| LogisticRegression | small_1Kx10 | 1.9 ms | 343.6 us | **5.5x** | 27.1 us | 21.0 us | 1.29x | accuracy | 83.50% | 82.00% | -1.50pp |
| LogisticRegression | tiny_50x5 | 556.1 us | 88.4 us | **6.3x** | 37.7 us | 18.1 us | **2.1x** | accuracy | 100.00% | 100.00% | +0.00pp |
| MultinomialNB | medium_10Kx100 | 1.9 ms | 1.2 ms | 1.64x | 154.6 us | 148.5 us | 1.04x | accuracy | 61.20% | 61.20% | +0.00pp |
| MultinomialNB | small_1Kx10 | 492.6 us | 23.6 us | **20.8x** | 29.5 us | 2.8 us | **10.4x** | accuracy | 70.50% | 70.50% | +0.00pp |
| MultinomialNB | tiny_50x5 | 360.6 us | 4.8 us | **74.8x** | 20.4 us | 677 ns | **30.2x** | accuracy | 30.00% | 30.00% | +0.00pp |
| NearestCentroid | medium_10Kx100 | 4.3 ms | 1.3 ms | **3.2x** | 538.7 us | 154.1 us | **3.5x** | accuracy | 69.15% | 69.15% | +0.00pp |
| NearestCentroid | small_1Kx10 | 316.1 us | 71.5 us | **4.4x** | 242.9 us | 6.3 us | **38.8x** | accuracy | 77.50% | 77.50% | +0.00pp |
| NearestCentroid | tiny_50x5 | 213.7 us | 6.1 us | **35.1x** | 179.0 us | 853 ns | **209.9x** | accuracy | 100.00% | 100.00% | +0.00pp |
| QDA | medium_10Kx100 | — | — | — | — | — | — | accuracy | — | — | — |
| QDA | small_1Kx10 | — | — | — | — | — | — | accuracy | — | — | — |
| QDA | tiny_50x5 | — | — | — | — | — | — | accuracy | — | — | — |
| RandomForestClassifier | medium_10Kx100 | 294.1 ms | 210.9 ms | 1.39x | 25.1 ms | 29.1 ms | 0.86x | accuracy | 94.25% | 93.05% | -1.20pp |
| RandomForestClassifier | small_1Kx10 | 123.2 ms | 5.2 ms | **23.8x** | 26.1 ms | 900.5 us | **28.9x** | accuracy | 94.50% | 96.00% | +1.50pp |
| RandomForestClassifier | tiny_50x5 | 76.7 ms | 1.8 ms | **42.3x** | 14.5 ms | 28.3 us | **512.4x** | accuracy | 80.00% | 90.00% | +10.00pp |
| RidgeClassifier | medium_10Kx100 | 15.1 ms | 4.8 ms | **3.2x** | 120.4 us | 122.1 us | 0.99x | accuracy | 83.35% | 83.35% | +0.00pp |
| RidgeClassifier | small_1Kx10 | 986.5 us | 55.0 us | **17.9x** | 35.3 us | 3.4 us | **10.5x** | accuracy | 83.00% | 83.00% | +0.00pp |
| RidgeClassifier | tiny_50x5 | 671.0 us | 5.9 us | **114.2x** | 25.4 us | 796 ns | **31.9x** | accuracy | 90.00% | 90.00% | +0.00pp |

### Cluster — 15 comparisons

| Algorithm | Dataset | sklearn fit | ferrolearn fit | fit speedup | sklearn predict | ferrolearn predict | predict speedup | metric | sklearn | ferrolearn | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AgglomerativeClustering | small_1Kx10 | 6.5 ms | 132.9 ms | 0.05x | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |
| AgglomerativeClustering | tiny_200x5 | 501.1 us | 1.1 ms | 0.44x | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |
| Birch | small_1Kx10 | 23.1 ms | 5.2 ms | **4.5x** | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |
| Birch | tiny_200x5 | 4.7 ms | 843.6 us | **5.5x** | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |
| DBSCAN | small_1Kx10 | 3.1 ms | 2.3 ms | 1.35x | — | — | — | ari | 0.0000 | 0.0000 | +0.0000 |
| DBSCAN | tiny_200x5 | 486.2 us | 67.8 us | **7.2x** | — | — | — | ari | 0.0000 | 0.0000 | +0.0000 |
| GaussianMixture | medium_5Kx20 | 38.4 ms | 318.7 ms | 0.12x | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |
| GaussianMixture | small_1Kx10 | 7.2 ms | 21.1 ms | 0.34x | — | — | — | ari | 1.0000 | 0.8344 | -0.1656 |
| GaussianMixture | tiny_200x5 | 1.9 ms | 2.8 ms | 0.66x | — | — | — | ari | 1.0000 | 0.7270 | -0.2730 |
| KMeans | medium_5Kx20 | 53.9 ms | 85.7 ms | 0.63x | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |
| KMeans | small_1Kx10 | 9.4 ms | 3.8 ms | **2.5x** | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |
| KMeans | tiny_200x5 | 8.7 ms | 230.6 us | **37.8x** | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |
| MiniBatchKMeans | medium_5Kx20 | 13.2 ms | 10.9 ms | 1.20x | — | — | — | ari | 1.0000 | 0.8366 | -0.1634 |
| MiniBatchKMeans | small_1Kx10 | 16.4 ms | 7.2 ms | **2.3x** | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |
| MiniBatchKMeans | tiny_200x5 | 2.5 ms | 5.7 ms | 0.45x | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |

### Decomp — 15 comparisons

| Algorithm | Dataset | sklearn fit | ferrolearn fit | fit speedup | sklearn predict | ferrolearn predict | predict speedup | metric | sklearn | ferrolearn | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FactorAnalysis | small_1Kx10 | 9.5 ms | 33.9 ms | 0.28x | 51.1 us | 18.9 us | **2.7x** | recon_rel | — | — | — |
| FactorAnalysis | tiny_50x5 | 1.5 ms | 159.6 us | **9.7x** | 35.0 us | 2.1 us | **17.0x** | recon_rel | — | — | — |
| FastICA | small_1Kx10 | 61.2 ms | 7.8 ms | **7.9x** | 90.8 us | 18.4 us | **4.9x** | recon_rel | 6.787e-01 | — | — |
| FastICA | tiny_50x5 | 8.3 ms | 117.6 us | **70.3x** | 23.8 us | 2.0 us | **12.1x** | recon_rel | 3.394e-01 | — | — |
| IncrementalPCA | medium_10Kx100 | 285.9 ms | 119.6 ms | **2.4x** | 804.6 us | 1.2 ms | 0.69x | recon_rel | 9.722e-01 | — | — |
| IncrementalPCA | small_1Kx10 | 2.3 ms | 51.1 us | **46.0x** | 34.6 us | 10.2 us | **3.4x** | recon_rel | 6.969e-01 | — | — |
| IncrementalPCA | tiny_50x5 | 297.1 us | 4.1 us | **73.3x** | 24.2 us | 1.6 us | **15.1x** | recon_rel | 3.483e-01 | — | — |
| PCA | medium_10Kx100 | 6.2 ms | 26.4 ms | 0.24x | 550.4 us | 3.4 ms | 0.16x | recon_rel | 9.698e-01 | 9.698e-01 | 1.00x |
| PCA | small_1Kx10 | 164.1 us | 49.2 us | **3.3x** | 35.9 us | 33.4 us | 1.07x | recon_rel | 6.787e-01 | 6.787e-01 | 1.00x |
| PCA | tiny_50x5 | 166.2 us | 24.1 us | **6.9x** | 31.3 us | 19.1 us | 1.64x | recon_rel | 3.394e-01 | 3.394e-01 | 1.00x |
| SparsePCA | small_1Kx10 | 237.2 ms | 416.5 ms | 0.57x | 470.5 us | 10.9 us | **43.3x** | recon_rel | 6.875e-01 | — | — |
| SparsePCA | tiny_50x5 | 9.6 ms | 1.3 ms | **7.2x** | 181.1 us | 1.5 us | **118.7x** | recon_rel | 3.541e-01 | — | — |
| TruncatedSVD | medium_10Kx100 | 22.7 ms | 13.1 ms | 1.73x | 455.4 us | 1.7 ms | 0.27x | recon_rel | 9.713e-01 | — | — |
| TruncatedSVD | small_1Kx10 | 702.5 us | 192.1 us | **3.7x** | 26.2 us | 7.0 us | **3.7x** | recon_rel | 6.790e-01 | — | — |
| TruncatedSVD | tiny_50x5 | 394.1 us | 5.7 us | **68.8x** | 24.5 us | 1.6 us | **15.6x** | recon_rel | 3.406e-01 | — | — |

### Kernel — 6 comparisons

| Algorithm | Dataset | sklearn fit | ferrolearn fit | fit speedup | sklearn predict | ferrolearn predict | predict speedup | metric | sklearn | ferrolearn | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Nystroem | medium_10Kx100 | 100.0 ms | 4.6 ms | **21.5x** | 16.0 ms | 61.1 ms | 0.26x | timing_only | — | — | — |
| Nystroem | small_1Kx10 | 16.0 ms | 3.8 ms | **4.3x** | 7.1 ms | 4.6 ms | 1.56x | timing_only | — | — | — |
| Nystroem | tiny_50x5 | 856.3 us | 215.2 us | **4.0x** | 232.6 us | 78.2 us | **3.0x** | timing_only | — | — | — |
| RBFSampler | medium_10Kx100 | 538.3 us | 313.6 us | 1.72x | 16.5 ms | 15.9 ms | 1.04x | timing_only | — | — | — |
| RBFSampler | small_1Kx10 | 141.6 us | 5.2 us | **27.5x** | 1.1 ms | 1.0 ms | 1.07x | timing_only | — | — | — |
| RBFSampler | tiny_50x5 | 202.2 us | 4.0 us | **50.7x** | 94.7 us | 49.1 us | 1.93x | timing_only | — | — | — |

### Preprocess — 14 comparisons

| Algorithm | Dataset | sklearn fit | ferrolearn fit | fit speedup | sklearn predict | ferrolearn predict | predict speedup | metric | sklearn | ferrolearn | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MaxAbsScaler | medium_10Kx100 | 1.1 ms | 1.2 ms | 0.91x | 1.0 ms | 2.1 ms | 0.49x | rel_diff_vs_sklearn | 0.000e+00 | 0.000e+00 | — |
| MaxAbsScaler | small_1Kx10 | 68.9 us | 8.5 us | **8.1x** | 30.8 us | 12.1 us | **2.5x** | rel_diff_vs_sklearn | 0.000e+00 | 0.000e+00 | — |
| MaxAbsScaler | tiny_50x5 | 151.3 us | 2.0 us | **73.9x** | 56.7 us | 4.2 us | **13.6x** | rel_diff_vs_sklearn | 0.000e+00 | 0.000e+00 | — |
| MinMaxScaler | medium_10Kx100 | 960.8 us | 1.9 ms | 0.51x | 1.3 ms | 2.3 ms | 0.56x | rel_diff_vs_sklearn | 0.000e+00 | 1.146e-16 | — |
| MinMaxScaler | small_1Kx10 | 110.8 us | 15.2 us | **7.3x** | 35.5 us | 12.7 us | **2.8x** | rel_diff_vs_sklearn | 0.000e+00 | 1.088e-16 | — |
| MinMaxScaler | tiny_50x5 | 152.6 us | 2.2 us | **70.9x** | 60.6 us | 4.4 us | **13.8x** | rel_diff_vs_sklearn | 0.000e+00 | 8.444e-17 | — |
| PowerTransformer | small_1Kx10 | 31.1 ms | 1.7 ms | **17.9x** | 394.5 us | 115.4 us | **3.4x** | rel_diff_vs_sklearn | 0.000e+00 | 4.203e-01 | — |
| PowerTransformer | tiny_50x5 | 14.7 ms | 41.7 us | **352.0x** | 110.6 us | 4.4 us | **25.2x** | rel_diff_vs_sklearn | 0.000e+00 | 4.189e-01 | — |
| RobustScaler | medium_10Kx100 | 25.5 ms | 14.8 ms | 1.72x | 1.4 ms | 1.9 ms | 0.76x | rel_diff_vs_sklearn | 0.000e+00 | 0.000e+00 | — |
| RobustScaler | small_1Kx10 | 610.9 us | 107.4 us | **5.7x** | 38.2 us | 12.2 us | **3.1x** | rel_diff_vs_sklearn | 0.000e+00 | 4.251e-20 | — |
| RobustScaler | tiny_50x5 | 792.9 us | 4.7 us | **169.2x** | 69.7 us | 4.3 us | **16.3x** | rel_diff_vs_sklearn | 0.000e+00 | 2.450e-17 | — |
| StandardScaler | medium_10Kx100 | 4.3 ms | 1.8 ms | **2.3x** | 3.7 ms | 2.9 ms | 1.29x | rel_diff_vs_sklearn | 0.000e+00 | 0.000e+00 | — |
| StandardScaler | small_1Kx10 | 172.4 us | 28.6 us | **6.0x** | 41.7 us | 30.8 us | 1.35x | rel_diff_vs_sklearn | 0.000e+00 | 0.000e+00 | — |
| StandardScaler | tiny_50x5 | 274.5 us | 50.8 us | **5.4x** | 92.2 us | 48.9 us | 1.89x | rel_diff_vs_sklearn | 0.000e+00 | 0.000e+00 | — |

### Regressor — 43 comparisons

| Algorithm | Dataset | sklearn fit | ferrolearn fit | fit speedup | sklearn predict | ferrolearn predict | predict speedup | metric | sklearn | ferrolearn | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ARDRegression | medium_10Kx100 | 841.8 ms | 58.0 ms | **14.5x** | 120.3 us | 73.8 us | 1.63x | r2 | 1.0000 | 1.0000 | +0.0000 |
| ARDRegression | small_1Kx10 | 733.3 us | 53.4 us | **13.7x** | 18.5 us | 1.2 us | **15.1x** | r2 | 1.0000 | 1.0000 | +0.0000 |
| ARDRegression | tiny_50x5 | 684.1 us | 5.2 us | **131.2x** | 18.0 us | 531 ns | **33.9x** | r2 | 1.0000 | 1.0000 | +0.0000 |
| BayesianRidge | medium_10Kx100 | 210.2 ms | 16.7 ms | **12.6x** | 98.9 us | 71.8 us | 1.38x | r2 | 1.0000 | 1.0000 | +0.0000 |
| BayesianRidge | small_1Kx10 | 387.1 us | 57.7 us | **6.7x** | 20.1 us | 1.2 us | **17.1x** | r2 | 1.0000 | 1.0000 | +0.0000 |
| BayesianRidge | tiny_50x5 | 343.1 us | 4.8 us | **70.9x** | 16.8 us | 492 ns | **34.2x** | r2 | 1.0000 | 1.0000 | -0.0000 |
| DecisionTreeRegressor | medium_10Kx100 | 435.3 ms | 241.1 ms | 1.81x | 320.1 us | 237.6 us | 1.35x | r2 | -0.5324 | -0.5521 | -0.0197 |
| DecisionTreeRegressor | small_1Kx10 | 3.3 ms | 1.7 ms | 1.92x | 33.1 us | 7.1 us | **4.7x** | r2 | 0.6236 | 0.6146 | -0.0090 |
| DecisionTreeRegressor | tiny_50x5 | 455.8 us | 22.1 us | **20.6x** | 24.6 us | 522 ns | **47.2x** | r2 | -2.2859 | -2.1071 | +0.1788 |
| ElasticNet | medium_10Kx100 | 7.8 ms | 11.0 ms | 0.71x | 111.6 us | 173.7 us | 0.64x | r2 | 0.8870 | 0.8870 | -0.0000 |
| ElasticNet | small_1Kx10 | 205.4 us | 100.6 us | **2.0x** | 21.4 us | 18.1 us | 1.18x | r2 | 0.8761 | 0.8761 | -0.0000 |
| ElasticNet | tiny_50x5 | 179.5 us | 36.2 us | **5.0x** | 20.6 us | 17.2 us | 1.20x | r2 | 0.8193 | 0.8193 | -0.0000 |
| ExtraTreesRegressor | medium_10Kx100 | 546.6 ms | 393.9 ms | 1.39x | 24.8 ms | 68.5 ms | 0.36x | r2 | 0.3724 | 0.3690 | -0.0034 |
| ExtraTreesRegressor | small_1Kx10 | 46.6 ms | 6.2 ms | **7.5x** | 25.5 ms | 2.0 ms | **12.8x** | r2 | 0.8793 | 0.8875 | +0.0082 |
| ExtraTreesRegressor | tiny_50x5 | 47.1 ms | 1.8 ms | **26.9x** | 14.2 ms | 30.1 us | **471.1x** | r2 | 0.2541 | 0.1803 | -0.0738 |
| GradientBoostingRegressor | small_1Kx10 | 122.1 ms | 52.4 ms | **2.3x** | 242.4 us | 78.5 us | **3.1x** | r2 | 0.9268 | 0.9269 | +0.0001 |
| GradientBoostingRegressor | tiny_50x5 | 16.0 ms | 1.0 ms | **15.6x** | 72.5 us | 4.6 us | **15.6x** | r2 | -0.1346 | -0.1405 | -0.0059 |
| HistGradientBoostingRegressor | medium_10Kx100 | 282.0 ms | 955.3 ms | 0.30x | 1.5 ms | 12.8 ms | 0.12x | r2 | 0.6349 | 0.6349 | +0.0000 |
| HistGradientBoostingRegressor | small_1Kx10 | 118.3 ms | 41.3 ms | **2.9x** | 727.2 us | 1.2 ms | 0.63x | r2 | 0.9394 | 0.9405 | +0.0012 |
| HistGradientBoostingRegressor | tiny_50x5 | 23.8 ms | 176.3 us | **134.7x** | 556.2 us | 2.6 us | **213.1x** | r2 | -0.2571 | -0.2571 | +0.0000 |
| HuberRegressor | medium_10Kx100 | 848.7 ms | 39.8 ms | **21.3x** | 91.6 us | 72.8 us | 1.26x | r2 | 1.0000 | 1.0000 | -0.0000 |
| HuberRegressor | small_1Kx10 | 2.7 ms | 76.6 us | **35.7x** | 18.5 us | 1.2 us | **15.2x** | r2 | 1.0000 | 1.0000 | -0.0000 |
| HuberRegressor | tiny_50x5 | 3.5 ms | 7.0 us | **506.6x** | 26.1 us | 1.1 us | **23.7x** | r2 | 1.0000 | 1.0000 | +0.0000 |
| KNeighborsRegressor | medium_10Kx100 | 566.2 us | 13.0 ms | 0.04x | 19.3 ms | 45.5 ms | 0.42x | r2 | 0.3173 | 0.3173 | +0.0000 |
| KNeighborsRegressor | small_1Kx10 | 302.9 us | 319.1 us | 0.95x | 14.1 ms | 7.4 ms | 1.90x | r2 | 0.7790 | 0.7790 | +0.0000 |
| KNeighborsRegressor | tiny_50x5 | 253.0 us | 6.6 us | **38.4x** | 14.5 ms | 12.1 us | **1202.0x** | r2 | 0.6307 | 0.6307 | +0.0000 |
| KernelRidge | small_1Kx10 | 88.6 ms | 40.2 ms | **2.2x** | 309.6 us | 2.7 ms | 0.12x | r2 | 1.0000 | 0.9307 | -0.0692 |
| KernelRidge | tiny_50x5 | 160.5 us | 31.7 us | **5.1x** | 97.9 us | 6.6 us | **14.8x** | r2 | 0.9988 | 0.7963 | -0.2026 |
| Lasso | medium_10Kx100 | 25.8 ms | 12.4 ms | **2.1x** | 95.5 us | 168.7 us | 0.57x | r2 | 0.9997 | 0.9997 | +0.0000 |
| Lasso | small_1Kx10 | 208.7 us | 93.4 us | **2.2x** | 22.8 us | 18.8 us | 1.21x | r2 | 0.9994 | 0.9994 | -0.0000 |
| Lasso | tiny_50x5 | 182.3 us | 35.8 us | **5.1x** | 19.4 us | 16.7 us | 1.16x | r2 | 0.9995 | 0.9995 | +0.0000 |
| LinearRegression | medium_10Kx100 | 83.8 ms | 19.0 ms | **4.4x** | 90.3 us | 215.8 us | 0.42x | r2 | 1.0000 | 1.0000 | +0.0000 |
| LinearRegression | small_1Kx10 | 272.5 us | 56.7 us | **4.8x** | 18.9 us | 20.7 us | 0.91x | r2 | 1.0000 | 1.0000 | +0.0000 |
| LinearRegression | tiny_50x5 | 181.5 us | 47.8 us | **3.8x** | 17.4 us | 18.0 us | 0.97x | r2 | 1.0000 | 1.0000 | +0.0000 |
| QuantileRegressor | medium_10Kx100 | 1.44 s | 1.29 s | 1.12x | 77.6 us | 74.7 us | 1.04x | r2 | -0.0017 | 1.0000 | +1.0017 |
| QuantileRegressor | small_1Kx10 | 20.0 ms | 2.4 ms | **8.3x** | 21.9 us | 1.3 us | **16.5x** | r2 | -0.0112 | 1.0000 | +1.0112 |
| QuantileRegressor | tiny_50x5 | 2.1 ms | 7.3 us | **293.0x** | 53.2 us | 1.4 us | **37.0x** | r2 | -0.0488 | -0.0717 | -0.0229 |
| RandomForestRegressor | medium_10Kx100 | 1.60 s | 1.68 s | 0.95x | 24.9 ms | 47.3 ms | 0.53x | r2 | 0.3759 | 0.3810 | +0.0051 |
| RandomForestRegressor | small_1Kx10 | 79.8 ms | 12.9 ms | **6.2x** | 25.4 ms | 1.5 ms | **17.0x** | r2 | 0.8446 | 0.8415 | -0.0031 |
| RandomForestRegressor | tiny_50x5 | 62.8 ms | 1.6 ms | **39.3x** | 14.0 ms | 17.7 us | **790.9x** | r2 | -0.7390 | -0.8137 | -0.0747 |
| Ridge | medium_10Kx100 | 22.3 ms | 11.4 ms | 1.96x | 115.5 us | 214.6 us | 0.54x | r2 | 1.0000 | 1.0000 | +0.0000 |
| Ridge | small_1Kx10 | 247.3 us | 59.0 us | **4.2x** | 18.7 us | 20.7 us | 0.91x | r2 | 1.0000 | 1.0000 | +0.0000 |
| Ridge | tiny_50x5 | 222.1 us | 33.8 us | **6.6x** | 22.2 us | 16.6 us | 1.33x | r2 | 0.9988 | 0.9988 | +0.0000 |

## Summary — geometric mean speedup

| Family | n compared | fit geomean | predict geomean | accuracy parity (mean Δ) |
|---|---:|---:|---:|---:|
| classifier | 51 | 5.34x | 8.75x | -0.0059 |
| cluster | 15 | 1.17x | — | -0.0401 |
| decomp | 15 | 5.47x | 4.20x | — |
| kernel | 6 | 9.78x | 1.17x | — |
| preprocess | 14 | 10.09x | 2.97x | — |
| regressor | 43 | 6.38x | 4.09x | +0.0400 |

