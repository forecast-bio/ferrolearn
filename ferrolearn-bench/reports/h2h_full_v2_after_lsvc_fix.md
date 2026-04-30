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
| AdaBoostClassifier | small_1Kx10 | 51.4 ms | 11.3 ms | **4.6x** | 1.5 ms | 168.7 us | **9.1x** | accuracy | 91.50% | 72.50% | -19.00pp |
| AdaBoostClassifier | tiny_50x5 | 24.5 ms | 290.6 us | **84.2x** | 1.4 ms | 9.0 us | **151.1x** | accuracy | 90.00% | 100.00% | +10.00pp |
| BaggingClassifier | small_1Kx10 | 11.3 ms | 2.5 ms | **4.5x** | 10.8 ms | 72.5 us | **149.3x** | accuracy | 95.00% | 96.00% | +1.00pp |
| BaggingClassifier | tiny_50x5 | 11.3 ms | 511.2 us | **22.0x** | 11.0 ms | 2.3 us | **4691.1x** | accuracy | 80.00% | 90.00% | +10.00pp |
| BernoulliNB | medium_10Kx100 | 7.4 ms | 1.7 ms | **4.4x** | 1.8 ms | 333.3 us | **5.3x** | accuracy | 75.00% | 75.00% | +0.00pp |
| BernoulliNB | small_1Kx10 | 577.4 us | 24.5 us | **23.6x** | 109.1 us | 3.5 us | **31.2x** | accuracy | 77.50% | 77.50% | +0.00pp |
| BernoulliNB | tiny_50x5 | 404.0 us | 7.1 us | **56.6x** | 81.5 us | 777 ns | **104.9x** | accuracy | 100.00% | 100.00% | +0.00pp |
| ComplementNB | medium_10Kx100 | 1.7 ms | 1.1 ms | 1.58x | 137.0 us | 149.5 us | 0.92x | accuracy | 61.20% | 61.20% | +0.00pp |
| ComplementNB | small_1Kx10 | 462.0 us | 21.6 us | **21.4x** | 22.3 us | 2.8 us | **8.0x** | accuracy | 71.00% | 71.00% | +0.00pp |
| ComplementNB | tiny_50x5 | 353.8 us | 4.5 us | **77.8x** | 20.1 us | 647 ns | **31.0x** | accuracy | 30.00% | 30.00% | +0.00pp |
| DecisionTreeClassifier | medium_10Kx100 | 599.5 ms | 288.9 ms | **2.1x** | 230.6 us | 317.1 us | 0.73x | accuracy | 73.65% | 71.95% | -1.70pp |
| DecisionTreeClassifier | small_1Kx10 | 2.9 ms | 1.2 ms | **2.4x** | 39.7 us | 30.4 us | 1.31x | accuracy | 89.50% | 90.00% | +0.50pp |
| DecisionTreeClassifier | tiny_50x5 | 290.8 us | 101.8 us | **2.9x** | 26.0 us | 17.5 us | 1.49x | accuracy | 90.00% | 80.00% | -10.00pp |
| ExtraTreeClassifier | medium_10Kx100 | 8.1 ms | 9.9 ms | 0.82x | 304.0 us | 291.4 us | 1.04x | accuracy | 64.20% | 63.80% | -0.40pp |
| ExtraTreeClassifier | small_1Kx10 | 502.6 us | 291.5 us | 1.72x | 44.4 us | 6.7 us | **6.6x** | accuracy | 81.00% | 80.00% | -1.00pp |
| ExtraTreeClassifier | tiny_50x5 | 288.3 us | 10.3 us | **28.0x** | 24.9 us | 788 ns | **31.6x** | accuracy | 90.00% | 90.00% | +0.00pp |
| ExtraTreesClassifier | medium_10Kx100 | 136.1 ms | 64.1 ms | **2.1x** | 25.0 ms | 54.7 ms | 0.46x | accuracy | 93.90% | 93.75% | -0.15pp |
| ExtraTreesClassifier | small_1Kx10 | 89.8 ms | 4.5 ms | **20.1x** | 25.2 ms | 1.5 ms | **16.7x** | accuracy | 97.00% | 96.00% | -1.00pp |
| ExtraTreesClassifier | tiny_50x5 | 60.6 ms | 1.7 ms | **34.9x** | 14.3 ms | 11.0 us | **1303.9x** | accuracy | 90.00% | 90.00% | +0.00pp |
| GaussianNB | medium_10Kx100 | 4.5 ms | 2.4 ms | 1.91x | 621.9 us | 1.3 ms | 0.48x | accuracy | 81.55% | 81.55% | +0.00pp |
| GaussianNB | small_1Kx10 | 391.2 us | 155.5 us | **2.5x** | 56.6 us | 30.0 us | 1.89x | accuracy | 89.00% | 89.00% | +0.00pp |
| GaussianNB | tiny_50x5 | 248.4 us | 79.1 us | **3.1x** | 36.4 us | 17.3 us | **2.1x** | accuracy | 100.00% | 100.00% | +0.00pp |
| GradientBoostingClassifier | small_1Kx10 | 136.2 ms | 51.0 ms | **2.7x** | 240.1 us | 83.3 us | **2.9x** | accuracy | 96.00% | 94.00% | -2.00pp |
| GradientBoostingClassifier | tiny_50x5 | 20.4 ms | 484.9 us | **42.1x** | 76.6 us | 3.3 us | **23.1x** | accuracy | 80.00% | 80.00% | +0.00pp |
| HistGradientBoostingClassifier | medium_10Kx100 | 286.6 ms | 964.4 ms | 0.30x | 1.6 ms | 14.1 ms | 0.11x | accuracy | 95.80% | 95.80% | +0.00pp |
| HistGradientBoostingClassifier | small_1Kx10 | 115.6 ms | 40.1 ms | **2.9x** | 575.0 us | 951.5 us | 0.60x | accuracy | 96.00% | 94.00% | -2.00pp |
| HistGradientBoostingClassifier | tiny_50x5 | 23.3 ms | 419.8 us | **55.6x** | 1.0 ms | 6.3 us | **162.8x** | accuracy | 100.00% | 100.00% | +0.00pp |
| KNeighborsClassifier | medium_10Kx100 | 856.8 us | 15.3 ms | 0.06x | 14.7 ms | 45.4 ms | 0.32x | accuracy | 96.60% | 96.60% | +0.00pp |
| KNeighborsClassifier | small_1Kx10 | 377.6 us | 503.5 us | 0.75x | 14.7 ms | 6.7 ms | **2.2x** | accuracy | 91.50% | 91.50% | +0.00pp |
| KNeighborsClassifier | tiny_50x5 | 353.8 us | 88.2 us | **4.0x** | 14.9 ms | 32.6 us | **456.5x** | accuracy | 90.00% | 90.00% | +0.00pp |
| LinearSVC | medium_10Kx100 | 52.1 ms | 22.1 ms | **2.4x** | 82.8 us | 76.6 us | 1.08x | accuracy | 83.60% | 83.70% | +0.10pp |
| LinearSVC | small_1Kx10 | 613.9 us | 220.6 us | **2.8x** | 26.5 us | 1.8 us | **14.6x** | accuracy | 83.50% | 83.50% | +0.00pp |
| LinearSVC | tiny_50x5 | 238.5 us | 6.3 us | **37.6x** | 25.5 us | 710 ns | **35.9x** | accuracy | 90.00% | 90.00% | +0.00pp |
| LogisticRegression | medium_10Kx100 | 848.1 ms | 23.2 ms | **36.6x** | 110.2 us | 194.3 us | 0.57x | accuracy | 83.50% | 83.45% | -0.05pp |
| LogisticRegression | small_1Kx10 | 1.8 ms | 335.1 us | **5.3x** | 27.3 us | 19.8 us | 1.37x | accuracy | 83.50% | 82.00% | -1.50pp |
| LogisticRegression | tiny_50x5 | 545.3 us | 97.2 us | **5.6x** | 25.4 us | 20.7 us | 1.23x | accuracy | 100.00% | 100.00% | +0.00pp |
| MultinomialNB | medium_10Kx100 | 1.9 ms | 1.1 ms | 1.69x | 137.7 us | 148.2 us | 0.93x | accuracy | 61.20% | 61.20% | +0.00pp |
| MultinomialNB | small_1Kx10 | 443.8 us | 23.7 us | **18.7x** | 22.9 us | 2.7 us | **8.6x** | accuracy | 70.50% | 70.50% | +0.00pp |
| MultinomialNB | tiny_50x5 | 357.8 us | 4.7 us | **76.3x** | 22.0 us | 660 ns | **33.4x** | accuracy | 30.00% | 30.00% | +0.00pp |
| NearestCentroid | medium_10Kx100 | 3.8 ms | 1.3 ms | **3.0x** | 537.7 us | 153.1 us | **3.5x** | accuracy | 69.15% | 69.15% | +0.00pp |
| NearestCentroid | small_1Kx10 | 295.9 us | 26.9 us | **11.0x** | 611.8 us | 2.3 us | **266.2x** | accuracy | 77.50% | 77.50% | +0.00pp |
| NearestCentroid | tiny_50x5 | 213.7 us | 20.3 us | **10.5x** | 567.4 us | 3.5 us | **162.2x** | accuracy | 100.00% | 100.00% | +0.00pp |
| QDA | medium_10Kx100 | — | — | — | — | — | — | accuracy | — | — | — |
| QDA | small_1Kx10 | — | — | — | — | — | — | accuracy | — | — | — |
| QDA | tiny_50x5 | — | — | — | — | — | — | accuracy | — | — | — |
| RandomForestClassifier | medium_10Kx100 | 288.9 ms | 198.1 ms | 1.46x | 25.0 ms | 29.5 ms | 0.85x | accuracy | 94.25% | 93.05% | -1.20pp |
| RandomForestClassifier | small_1Kx10 | 123.5 ms | 5.1 ms | **24.3x** | 25.2 ms | 905.1 us | **27.9x** | accuracy | 94.50% | 96.00% | +1.50pp |
| RandomForestClassifier | tiny_50x5 | 82.3 ms | 1.9 ms | **43.1x** | 14.5 ms | 25.6 us | **565.4x** | accuracy | 80.00% | 90.00% | +10.00pp |
| RidgeClassifier | medium_10Kx100 | 6.7 ms | 5.0 ms | 1.34x | 125.6 us | 123.4 us | 1.02x | accuracy | 83.35% | 83.35% | +0.00pp |
| RidgeClassifier | small_1Kx10 | 729.4 us | 47.4 us | **15.4x** | 26.4 us | 2.2 us | **12.1x** | accuracy | 83.00% | 83.00% | +0.00pp |
| RidgeClassifier | tiny_50x5 | 661.1 us | 6.1 us | **108.7x** | 26.3 us | 829 ns | **31.7x** | accuracy | 90.00% | 90.00% | +0.00pp |

### Cluster — 15 comparisons

| Algorithm | Dataset | sklearn fit | ferrolearn fit | fit speedup | sklearn predict | ferrolearn predict | predict speedup | metric | sklearn | ferrolearn | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AgglomerativeClustering | small_1Kx10 | 6.3 ms | 134.7 ms | 0.05x | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |
| AgglomerativeClustering | tiny_200x5 | 484.7 us | 1.1 ms | 0.44x | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |
| Birch | small_1Kx10 | 23.6 ms | 5.2 ms | **4.6x** | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |
| Birch | tiny_200x5 | 3.5 ms | 642.4 us | **5.5x** | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |
| DBSCAN | small_1Kx10 | 3.2 ms | 2.3 ms | 1.38x | — | — | — | ari | 0.0000 | 0.0000 | +0.0000 |
| DBSCAN | tiny_200x5 | 471.9 us | 68.1 us | **6.9x** | — | — | — | ari | 0.0000 | 0.0000 | +0.0000 |
| GaussianMixture | medium_5Kx20 | 63.7 ms | 316.6 ms | 0.20x | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |
| GaussianMixture | small_1Kx10 | 6.4 ms | 20.9 ms | 0.31x | — | — | — | ari | 1.0000 | 0.8344 | -0.1656 |
| GaussianMixture | tiny_200x5 | 1.9 ms | 2.8 ms | 0.68x | — | — | — | ari | 1.0000 | 0.7270 | -0.2730 |
| KMeans | medium_5Kx20 | 51.1 ms | 89.5 ms | 0.57x | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |
| KMeans | small_1Kx10 | 10.0 ms | 3.8 ms | **2.6x** | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |
| KMeans | tiny_200x5 | 4.5 ms | 332.6 us | **13.5x** | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |
| MiniBatchKMeans | medium_5Kx20 | 13.3 ms | 9.4 ms | 1.41x | — | — | — | ari | 1.0000 | 0.8366 | -0.1634 |
| MiniBatchKMeans | small_1Kx10 | 17.0 ms | 6.8 ms | **2.5x** | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |
| MiniBatchKMeans | tiny_200x5 | 2.1 ms | 6.0 ms | 0.35x | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |

### Decomp — 15 comparisons

| Algorithm | Dataset | sklearn fit | ferrolearn fit | fit speedup | sklearn predict | ferrolearn predict | predict speedup | metric | sklearn | ferrolearn | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FactorAnalysis | small_1Kx10 | 9.8 ms | 33.6 ms | 0.29x | 47.2 us | 19.3 us | **2.4x** | recon_rel | — | — | — |
| FactorAnalysis | tiny_50x5 | 1.6 ms | 158.6 us | **9.9x** | 33.7 us | 2.2 us | **15.5x** | recon_rel | — | — | — |
| FastICA | small_1Kx10 | 62.6 ms | 7.7 ms | **8.2x** | 93.3 us | 17.7 us | **5.3x** | recon_rel | 6.787e-01 | — | — |
| FastICA | tiny_50x5 | 8.3 ms | 118.4 us | **70.5x** | 22.4 us | 1.9 us | **11.9x** | recon_rel | 3.394e-01 | — | — |
| IncrementalPCA | medium_10Kx100 | 186.3 ms | 126.2 ms | 1.48x | 8.0 ms | 1.2 ms | **6.9x** | recon_rel | 9.722e-01 | — | — |
| IncrementalPCA | small_1Kx10 | 2.5 ms | 49.5 us | **49.9x** | 53.5 us | 10.0 us | **5.3x** | recon_rel | 6.969e-01 | — | — |
| IncrementalPCA | tiny_50x5 | 292.0 us | 3.9 us | **74.0x** | 23.4 us | 1.6 us | **14.9x** | recon_rel | 3.483e-01 | — | — |
| PCA | medium_10Kx100 | 4.3 ms | 13.8 ms | 0.31x | 506.8 us | 1.8 ms | 0.29x | recon_rel | 9.698e-01 | 9.698e-01 | 1.00x |
| PCA | small_1Kx10 | 194.0 us | 49.9 us | **3.9x** | 34.2 us | 29.7 us | 1.15x | recon_rel | 6.787e-01 | 6.787e-01 | 1.00x |
| PCA | tiny_50x5 | 147.5 us | 24.5 us | **6.0x** | 22.7 us | 19.6 us | 1.16x | recon_rel | 3.394e-01 | 3.394e-01 | 1.00x |
| SparsePCA | small_1Kx10 | 240.8 ms | 432.3 ms | 0.56x | 426.1 us | 11.0 us | **38.6x** | recon_rel | 6.875e-01 | — | — |
| SparsePCA | tiny_50x5 | 9.1 ms | 1.8 ms | **5.2x** | 186.9 us | 1.6 us | **114.6x** | recon_rel | 3.541e-01 | — | — |
| TruncatedSVD | medium_10Kx100 | 75.2 ms | 6.9 ms | **10.9x** | 403.1 us | 861.2 us | 0.47x | recon_rel | 9.713e-01 | — | — |
| TruncatedSVD | small_1Kx10 | 728.1 us | 211.1 us | **3.4x** | 24.8 us | 6.9 us | **3.6x** | recon_rel | 6.790e-01 | — | — |
| TruncatedSVD | tiny_50x5 | 375.7 us | 5.7 us | **65.6x** | 19.7 us | 1.5 us | **13.2x** | recon_rel | 3.406e-01 | — | — |

### Kernel — 6 comparisons

| Algorithm | Dataset | sklearn fit | ferrolearn fit | fit speedup | sklearn predict | ferrolearn predict | predict speedup | metric | sklearn | ferrolearn | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Nystroem | medium_10Kx100 | 32.0 ms | 5.9 ms | **5.5x** | 14.2 ms | 60.9 ms | 0.23x | timing_only | — | — | — |
| Nystroem | small_1Kx10 | 8.0 ms | 4.9 ms | 1.62x | 12.0 ms | 6.1 ms | 1.97x | timing_only | — | — | — |
| Nystroem | tiny_50x5 | 1.1 ms | 363.2 us | **3.1x** | 371.7 us | 99.3 us | **3.7x** | timing_only | — | — | — |
| RBFSampler | medium_10Kx100 | 437.9 us | 293.7 us | 1.49x | 15.0 ms | 16.9 ms | 0.89x | timing_only | — | — | — |
| RBFSampler | small_1Kx10 | 135.0 us | 5.1 us | **26.7x** | 1.1 ms | 1.0 ms | 1.09x | timing_only | — | — | — |
| RBFSampler | tiny_50x5 | 260.6 us | 5.6 us | **46.4x** | 151.8 us | 92.1 us | 1.65x | timing_only | — | — | — |

### Preprocess — 14 comparisons

| Algorithm | Dataset | sklearn fit | ferrolearn fit | fit speedup | sklearn predict | ferrolearn predict | predict speedup | metric | sklearn | ferrolearn | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MaxAbsScaler | medium_10Kx100 | 1.1 ms | 1.1 ms | 0.96x | 1.0 ms | 2.1 ms | 0.50x | rel_diff_vs_sklearn | 0.000e+00 | 0.000e+00 | — |
| MaxAbsScaler | small_1Kx10 | 69.0 us | 8.5 us | **8.1x** | 33.7 us | 12.2 us | **2.8x** | rel_diff_vs_sklearn | 0.000e+00 | 0.000e+00 | — |
| MaxAbsScaler | tiny_50x5 | 75.4 us | 1.2 us | **61.8x** | 31.7 us | 2.2 us | **14.2x** | rel_diff_vs_sklearn | 0.000e+00 | 0.000e+00 | — |
| MinMaxScaler | medium_10Kx100 | 978.1 us | 1.8 ms | 0.53x | 1.3 ms | 2.2 ms | 0.58x | rel_diff_vs_sklearn | 0.000e+00 | 1.146e-16 | — |
| MinMaxScaler | small_1Kx10 | 96.4 us | 15.1 us | **6.4x** | 35.9 us | 12.0 us | **3.0x** | rel_diff_vs_sklearn | 0.000e+00 | 1.088e-16 | — |
| MinMaxScaler | tiny_50x5 | 92.3 us | 1.3 us | **71.6x** | 37.0 us | 2.3 us | **16.1x** | rel_diff_vs_sklearn | 0.000e+00 | 8.444e-17 | — |
| PowerTransformer | small_1Kx10 | 30.1 ms | 1.7 ms | **17.8x** | 392.1 us | 114.7 us | **3.4x** | rel_diff_vs_sklearn | 0.000e+00 | 4.203e-01 | — |
| PowerTransformer | tiny_50x5 | 15.7 ms | 42.6 us | **369.2x** | 101.9 us | 3.7 us | **27.8x** | rel_diff_vs_sklearn | 0.000e+00 | 4.189e-01 | — |
| RobustScaler | medium_10Kx100 | 24.0 ms | 14.6 ms | 1.64x | 1.3 ms | 2.0 ms | 0.68x | rel_diff_vs_sklearn | 0.000e+00 | 0.000e+00 | — |
| RobustScaler | small_1Kx10 | 660.9 us | 102.9 us | **6.4x** | 39.4 us | 12.0 us | **3.3x** | rel_diff_vs_sklearn | 0.000e+00 | 4.251e-20 | — |
| RobustScaler | tiny_50x5 | 448.4 us | 3.5 us | **129.5x** | 31.4 us | 2.3 us | **13.7x** | rel_diff_vs_sklearn | 0.000e+00 | 2.450e-17 | — |
| StandardScaler | medium_10Kx100 | 3.0 ms | 1.8 ms | 1.64x | 1.4 ms | 3.1 ms | 0.45x | rel_diff_vs_sklearn | 0.000e+00 | 0.000e+00 | — |
| StandardScaler | small_1Kx10 | 182.7 us | 27.7 us | **6.6x** | 42.7 us | 31.8 us | 1.34x | rel_diff_vs_sklearn | 0.000e+00 | 0.000e+00 | — |
| StandardScaler | tiny_50x5 | 214.3 us | 24.9 us | **8.6x** | 41.1 us | 26.4 us | 1.56x | rel_diff_vs_sklearn | 0.000e+00 | 0.000e+00 | — |

### Regressor — 43 comparisons

| Algorithm | Dataset | sklearn fit | ferrolearn fit | fit speedup | sklearn predict | ferrolearn predict | predict speedup | metric | sklearn | ferrolearn | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ARDRegression | medium_10Kx100 | 837.2 ms | 59.7 ms | **14.0x** | 116.2 us | 78.3 us | 1.48x | r2 | 1.0000 | 1.0000 | +0.0000 |
| ARDRegression | small_1Kx10 | 685.0 us | 53.6 us | **12.8x** | 19.2 us | 1.2 us | **16.5x** | r2 | 1.0000 | 1.0000 | +0.0000 |
| ARDRegression | tiny_50x5 | 744.9 us | 4.9 us | **151.3x** | 17.7 us | 505 ns | **35.0x** | r2 | 1.0000 | 1.0000 | +0.0000 |
| BayesianRidge | medium_10Kx100 | 283.7 ms | 21.6 ms | **13.1x** | 87.1 us | 82.9 us | 1.05x | r2 | 1.0000 | 1.0000 | +0.0000 |
| BayesianRidge | small_1Kx10 | 384.8 us | 60.7 us | **6.3x** | 16.9 us | 1.2 us | **13.8x** | r2 | 1.0000 | 1.0000 | +0.0000 |
| BayesianRidge | tiny_50x5 | 378.8 us | 4.9 us | **77.3x** | 18.2 us | 476 ns | **38.2x** | r2 | 1.0000 | 1.0000 | -0.0000 |
| DecisionTreeRegressor | medium_10Kx100 | 442.4 ms | 238.7 ms | 1.85x | 418.2 us | 235.8 us | 1.77x | r2 | -0.5324 | -0.5521 | -0.0197 |
| DecisionTreeRegressor | small_1Kx10 | 3.2 ms | 1.7 ms | 1.92x | 32.0 us | 6.4 us | **5.0x** | r2 | 0.6236 | 0.6146 | -0.0090 |
| DecisionTreeRegressor | tiny_50x5 | 448.9 us | 44.8 us | **10.0x** | 66.8 us | 1.6 us | **40.7x** | r2 | -2.2859 | -2.1071 | +0.1788 |
| ElasticNet | medium_10Kx100 | 6.4 ms | 10.8 ms | 0.59x | 97.9 us | 158.2 us | 0.62x | r2 | 0.8870 | 0.8870 | -0.0000 |
| ElasticNet | small_1Kx10 | 202.4 us | 91.7 us | **2.2x** | 19.9 us | 17.7 us | 1.13x | r2 | 0.8761 | 0.8761 | -0.0000 |
| ElasticNet | tiny_50x5 | 229.4 us | 36.9 us | **6.2x** | 19.4 us | 17.4 us | 1.12x | r2 | 0.8193 | 0.8193 | -0.0000 |
| ExtraTreesRegressor | medium_10Kx100 | 560.5 ms | 401.0 ms | 1.40x | 24.8 ms | 75.1 ms | 0.33x | r2 | 0.3724 | 0.3690 | -0.0034 |
| ExtraTreesRegressor | small_1Kx10 | 47.6 ms | 6.5 ms | **7.3x** | 25.2 ms | 2.0 ms | **12.6x** | r2 | 0.8793 | 0.8875 | +0.0082 |
| ExtraTreesRegressor | tiny_50x5 | 47.1 ms | 1.7 ms | **27.7x** | 14.2 ms | 24.0 us | **592.8x** | r2 | 0.2541 | 0.1803 | -0.0738 |
| GradientBoostingRegressor | small_1Kx10 | 123.1 ms | 52.0 ms | **2.4x** | 267.9 us | 77.7 us | **3.4x** | r2 | 0.9268 | 0.9269 | +0.0001 |
| GradientBoostingRegressor | tiny_50x5 | 15.3 ms | 1.0 ms | **14.9x** | 73.7 us | 4.9 us | **15.1x** | r2 | -0.1346 | -0.1405 | -0.0059 |
| HistGradientBoostingRegressor | medium_10Kx100 | 268.1 ms | 963.2 ms | 0.28x | 1.6 ms | 13.1 ms | 0.12x | r2 | 0.6349 | 0.6349 | +0.0000 |
| HistGradientBoostingRegressor | small_1Kx10 | 118.5 ms | 40.9 ms | **2.9x** | 1.2 ms | 1.2 ms | 1.07x | r2 | 0.9394 | 0.9405 | +0.0012 |
| HistGradientBoostingRegressor | tiny_50x5 | 24.4 ms | 360.7 us | **67.5x** | 752.7 us | 6.1 us | **124.2x** | r2 | -0.2571 | -0.2571 | +0.0000 |
| HuberRegressor | medium_10Kx100 | 534.5 ms | 57.1 ms | **9.4x** | 115.6 us | 87.8 us | 1.32x | r2 | 1.0000 | 1.0000 | -0.0000 |
| HuberRegressor | small_1Kx10 | 2.8 ms | 67.8 us | **40.8x** | 19.2 us | 1.3 us | **15.0x** | r2 | 1.0000 | 1.0000 | -0.0000 |
| HuberRegressor | tiny_50x5 | 2.4 ms | 5.7 us | **422.6x** | 21.8 us | 1.1 us | **20.0x** | r2 | 1.0000 | 1.0000 | +0.0000 |
| KNeighborsRegressor | medium_10Kx100 | 846.7 us | 13.6 ms | 0.06x | 17.3 ms | 45.7 ms | 0.38x | r2 | 0.3173 | 0.3173 | +0.0000 |
| KNeighborsRegressor | small_1Kx10 | 293.6 us | 318.3 us | 0.92x | 14.4 ms | 7.5 ms | 1.91x | r2 | 0.7790 | 0.7790 | +0.0000 |
| KNeighborsRegressor | tiny_50x5 | 185.4 us | 6.1 us | **30.1x** | 14.7 ms | 12.3 us | **1194.9x** | r2 | 0.6307 | 0.6307 | +0.0000 |
| KernelRidge | small_1Kx10 | 9.1 ms | 39.2 ms | 0.23x | 444.4 us | 2.7 ms | 0.16x | r2 | 1.0000 | 0.9307 | -0.0692 |
| KernelRidge | tiny_50x5 | 178.1 us | 31.6 us | **5.6x** | 104.6 us | 7.2 us | **14.5x** | r2 | 0.9988 | 0.7963 | -0.2026 |
| Lasso | medium_10Kx100 | 50.9 ms | 12.3 ms | **4.1x** | 105.4 us | 162.8 us | 0.65x | r2 | 0.9997 | 0.9997 | +0.0000 |
| Lasso | small_1Kx10 | 197.7 us | 98.9 us | 2.00x | 19.9 us | 17.7 us | 1.13x | r2 | 0.9994 | 0.9994 | -0.0000 |
| Lasso | tiny_50x5 | 233.1 us | 70.3 us | **3.3x** | 19.9 us | 17.5 us | 1.14x | r2 | 0.9995 | 0.9995 | +0.0000 |
| LinearRegression | medium_10Kx100 | 80.1 ms | 12.0 ms | **6.7x** | 122.7 us | 235.4 us | 0.52x | r2 | 1.0000 | 1.0000 | +0.0000 |
| LinearRegression | small_1Kx10 | 248.1 us | 55.9 us | **4.4x** | 17.1 us | 17.2 us | 0.99x | r2 | 1.0000 | 1.0000 | +0.0000 |
| LinearRegression | tiny_50x5 | 235.0 us | 34.7 us | **6.8x** | 17.5 us | 16.4 us | 1.06x | r2 | 1.0000 | 1.0000 | +0.0000 |
| QuantileRegressor | medium_10Kx100 | 1.46 s | 1.29 s | 1.14x | 71.7 us | 73.9 us | 0.97x | r2 | -0.0017 | 1.0000 | +1.0017 |
| QuantileRegressor | small_1Kx10 | 17.7 ms | 2.4 ms | **7.4x** | 18.3 us | 1.3 us | **14.6x** | r2 | -0.0112 | 1.0000 | +1.0112 |
| QuantileRegressor | tiny_50x5 | 2.4 ms | 3.4 us | **697.2x** | 19.0 us | 511 ns | **37.3x** | r2 | -0.0488 | -0.0717 | -0.0229 |
| RandomForestRegressor | medium_10Kx100 | 1.60 s | 1.69 s | 0.95x | 24.9 ms | 48.9 ms | 0.51x | r2 | 0.3759 | 0.3810 | +0.0051 |
| RandomForestRegressor | small_1Kx10 | 79.8 ms | 13.1 ms | **6.1x** | 25.1 ms | 1.5 ms | **16.5x** | r2 | 0.8446 | 0.8415 | -0.0031 |
| RandomForestRegressor | tiny_50x5 | 59.5 ms | 1.7 ms | **34.6x** | 14.6 ms | 17.7 us | **825.2x** | r2 | -0.7390 | -0.8137 | -0.0747 |
| Ridge | medium_10Kx100 | 20.8 ms | 14.3 ms | 1.46x | 135.2 us | 163.0 us | 0.83x | r2 | 1.0000 | 1.0000 | +0.0000 |
| Ridge | small_1Kx10 | 244.2 us | 57.7 us | **4.2x** | 20.3 us | 17.6 us | 1.15x | r2 | 1.0000 | 1.0000 | +0.0000 |
| Ridge | tiny_50x5 | 251.1 us | 35.0 us | **7.2x** | 19.1 us | 16.0 us | 1.19x | r2 | 0.9988 | 0.9988 | +0.0000 |

## Summary — geometric mean speedup

| Family | n compared | fit geomean | predict geomean | accuracy parity (mean Δ) |
|---|---:|---:|---:|---:|
| classifier | 51 | 6.72x | 8.57x | -0.0014 |
| cluster | 15 | 1.11x | — | -0.0401 |
| decomp | 15 | 6.01x | 5.16x | — |
| kernel | 6 | 6.08x | 1.18x | — |
| preprocess | 14 | 9.99x | 2.77x | — |
| regressor | 43 | 6.00x | 4.16x | +0.0400 |

