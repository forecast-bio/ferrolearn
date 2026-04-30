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
| AdaBoostClassifier | small_1Kx10 | 52.4 ms | 11.1 ms | **4.7x** | 1.6 ms | 45.5 us | **35.6x** | accuracy | 91.50% | 92.00% | +0.50pp |
| AdaBoostClassifier | tiny_50x5 | 23.9 ms | 191.8 us | **124.8x** | 1.3 ms | 3.1 us | **427.4x** | accuracy | 90.00% | 100.00% | +10.00pp |
| BaggingClassifier | small_1Kx10 | 11.3 ms | 2.3 ms | **4.9x** | 10.9 ms | 73.0 us | **148.7x** | accuracy | 95.00% | 96.00% | +1.00pp |
| BaggingClassifier | tiny_50x5 | 11.1 ms | 521.2 us | **21.3x** | 10.8 ms | 2.4 us | **4471.2x** | accuracy | 80.00% | 90.00% | +10.00pp |
| BernoulliNB | medium_10Kx100 | 7.4 ms | 1.9 ms | **4.0x** | 1.6 ms | 320.1 us | **5.1x** | accuracy | 75.00% | 75.00% | +0.00pp |
| BernoulliNB | small_1Kx10 | 521.9 us | 25.3 us | **20.6x** | 100.4 us | 3.5 us | **29.1x** | accuracy | 77.50% | 77.50% | +0.00pp |
| BernoulliNB | tiny_50x5 | 415.3 us | 5.1 us | **80.8x** | 100.8 us | 763 ns | **132.1x** | accuracy | 100.00% | 100.00% | +0.00pp |
| ComplementNB | medium_10Kx100 | 1.7 ms | 1.2 ms | 1.38x | 120.6 us | 148.0 us | 0.82x | accuracy | 61.20% | 61.20% | +0.00pp |
| ComplementNB | small_1Kx10 | 459.6 us | 21.2 us | **21.7x** | 28.5 us | 2.7 us | **10.5x** | accuracy | 71.00% | 71.00% | +0.00pp |
| ComplementNB | tiny_50x5 | 358.6 us | 4.3 us | **84.3x** | 24.8 us | 673 ns | **36.8x** | accuracy | 30.00% | 30.00% | +0.00pp |
| DecisionTreeClassifier | medium_10Kx100 | 600.6 ms | 293.7 ms | **2.0x** | 229.9 us | 318.1 us | 0.72x | accuracy | 73.65% | 71.95% | -1.70pp |
| DecisionTreeClassifier | small_1Kx10 | 2.8 ms | 1.7 ms | 1.61x | 43.4 us | 32.5 us | 1.34x | accuracy | 89.50% | 90.00% | +0.50pp |
| DecisionTreeClassifier | tiny_50x5 | 335.9 us | 92.2 us | **3.6x** | 32.2 us | 17.9 us | 1.80x | accuracy | 90.00% | 80.00% | -10.00pp |
| ExtraTreeClassifier | medium_10Kx100 | 8.6 ms | 10.7 ms | 0.81x | 301.9 us | 276.6 us | 1.09x | accuracy | 64.20% | 64.30% | +0.10pp |
| ExtraTreeClassifier | small_1Kx10 | 794.7 us | 275.9 us | **2.9x** | 91.6 us | 6.7 us | **13.6x** | accuracy | 81.00% | 85.50% | +4.50pp |
| ExtraTreeClassifier | tiny_50x5 | 303.7 us | 10.5 us | **29.0x** | 32.3 us | 813 ns | **39.7x** | accuracy | 90.00% | 90.00% | +0.00pp |
| ExtraTreesClassifier | medium_10Kx100 | 124.3 ms | 64.5 ms | 1.93x | 25.1 ms | 52.8 ms | 0.48x | accuracy | 93.90% | 93.75% | -0.15pp |
| ExtraTreesClassifier | small_1Kx10 | 93.1 ms | 4.0 ms | **23.4x** | 25.5 ms | 1.5 ms | **16.9x** | accuracy | 97.00% | 96.00% | -1.00pp |
| ExtraTreesClassifier | tiny_50x5 | 57.5 ms | 1.9 ms | **29.8x** | 14.2 ms | 13.3 us | **1061.7x** | accuracy | 90.00% | 90.00% | +0.00pp |
| GaussianNB | medium_10Kx100 | 4.4 ms | 2.6 ms | 1.67x | 653.3 us | 1.3 ms | 0.49x | accuracy | 81.55% | 81.55% | +0.00pp |
| GaussianNB | small_1Kx10 | 430.9 us | 206.2 us | **2.1x** | 67.3 us | 45.4 us | 1.48x | accuracy | 89.00% | 89.00% | +0.00pp |
| GaussianNB | tiny_50x5 | 248.0 us | 88.2 us | **2.8x** | 38.0 us | 18.6 us | **2.0x** | accuracy | 100.00% | 100.00% | +0.00pp |
| GradientBoostingClassifier | small_1Kx10 | 136.4 ms | 50.4 ms | **2.7x** | 248.1 us | 81.8 us | **3.0x** | accuracy | 96.00% | 94.00% | -2.00pp |
| GradientBoostingClassifier | tiny_50x5 | 21.2 ms | 485.1 us | **43.7x** | 77.0 us | 3.0 us | **25.8x** | accuracy | 80.00% | 80.00% | +0.00pp |
| HistGradientBoostingClassifier | medium_10Kx100 | 290.4 ms | 941.1 ms | 0.31x | 1.5 ms | 14.4 ms | 0.10x | accuracy | 95.80% | 95.80% | +0.00pp |
| HistGradientBoostingClassifier | small_1Kx10 | 123.1 ms | 39.6 ms | **3.1x** | 642.7 us | 938.8 us | 0.68x | accuracy | 96.00% | 94.00% | -2.00pp |
| HistGradientBoostingClassifier | tiny_50x5 | 24.0 ms | 193.3 us | **124.2x** | 616.1 us | 2.7 us | **230.2x** | accuracy | 100.00% | 100.00% | +0.00pp |
| KNeighborsClassifier | medium_10Kx100 | 863.0 us | 16.4 ms | 0.05x | 17.7 ms | 46.1 ms | 0.38x | accuracy | 96.60% | 96.60% | +0.00pp |
| KNeighborsClassifier | small_1Kx10 | 412.0 us | 530.0 us | 0.78x | 14.6 ms | 6.8 ms | **2.1x** | accuracy | 91.50% | 91.50% | +0.00pp |
| KNeighborsClassifier | tiny_50x5 | 165.9 us | 121.5 us | 1.37x | 14.2 ms | 39.3 us | **362.3x** | accuracy | 90.00% | 90.00% | +0.00pp |
| LinearSVC | medium_10Kx100 | 46.5 ms | 23.2 ms | **2.0x** | 96.1 us | 111.9 us | 0.86x | accuracy | 83.60% | 83.70% | +0.10pp |
| LinearSVC | small_1Kx10 | 651.3 us | 311.5 us | **2.1x** | 34.6 us | 2.8 us | **12.4x** | accuracy | 83.50% | 83.50% | +0.00pp |
| LinearSVC | tiny_50x5 | 230.2 us | 6.2 us | **36.9x** | 29.0 us | 719 ns | **40.3x** | accuracy | 90.00% | 90.00% | +0.00pp |
| LogisticRegression | medium_10Kx100 | 817.0 ms | 48.6 ms | **16.8x** | 156.8 us | 344.2 us | 0.46x | accuracy | 83.50% | 83.45% | -0.05pp |
| LogisticRegression | small_1Kx10 | 785.5 us | 332.5 us | **2.4x** | 25.0 us | 60.2 us | 0.42x | accuracy | 83.50% | 82.00% | -1.50pp |
| LogisticRegression | tiny_50x5 | 513.1 us | 89.2 us | **5.8x** | 25.5 us | 18.3 us | 1.40x | accuracy | 100.00% | 100.00% | +0.00pp |
| MultinomialNB | medium_10Kx100 | 2.5 ms | 1.4 ms | 1.78x | 149.8 us | 167.9 us | 0.89x | accuracy | 61.20% | 61.20% | +0.00pp |
| MultinomialNB | small_1Kx10 | 439.7 us | 24.0 us | **18.3x** | 22.8 us | 2.8 us | **8.2x** | accuracy | 70.50% | 70.50% | +0.00pp |
| MultinomialNB | tiny_50x5 | 361.1 us | 4.8 us | **75.2x** | 22.2 us | 682 ns | **32.5x** | accuracy | 30.00% | 30.00% | +0.00pp |
| NearestCentroid | medium_10Kx100 | 4.5 ms | 1.4 ms | **3.2x** | 537.5 us | 157.2 us | **3.4x** | accuracy | 69.15% | 69.15% | +0.00pp |
| NearestCentroid | small_1Kx10 | 330.1 us | 21.9 us | **15.1x** | 201.9 us | 2.3 us | **87.6x** | accuracy | 77.50% | 77.50% | +0.00pp |
| NearestCentroid | tiny_50x5 | 254.6 us | 4.7 us | **54.2x** | 502.5 us | 639 ns | **786.3x** | accuracy | 100.00% | 100.00% | +0.00pp |
| QDA | medium_10Kx100 | — | — | — | — | — | — | accuracy | — | — | — |
| QDA | small_1Kx10 | — | — | — | — | — | — | accuracy | — | — | — |
| QDA | tiny_50x5 | — | — | — | — | — | — | accuracy | — | — | — |
| RandomForestClassifier | medium_10Kx100 | 268.8 ms | 193.4 ms | 1.39x | 24.9 ms | 29.2 ms | 0.85x | accuracy | 94.25% | 93.05% | -1.20pp |
| RandomForestClassifier | small_1Kx10 | 118.6 ms | 5.4 ms | **21.9x** | 25.1 ms | 846.7 us | **29.6x** | accuracy | 94.50% | 96.00% | +1.50pp |
| RandomForestClassifier | tiny_50x5 | 79.1 ms | 1.9 ms | **40.8x** | 14.4 ms | 27.3 us | **527.3x** | accuracy | 80.00% | 90.00% | +10.00pp |
| RidgeClassifier | medium_10Kx100 | 15.1 ms | 4.8 ms | **3.1x** | 120.6 us | 159.2 us | 0.76x | accuracy | 83.35% | 83.35% | +0.00pp |
| RidgeClassifier | small_1Kx10 | 1.3 ms | 88.5 us | **15.1x** | 68.1 us | 5.6 us | **12.2x** | accuracy | 83.00% | 83.00% | +0.00pp |
| RidgeClassifier | tiny_50x5 | 697.9 us | 6.1 us | **114.5x** | 25.3 us | 861 ns | **29.4x** | accuracy | 90.00% | 90.00% | +0.00pp |

### Cluster — 15 comparisons

| Algorithm | Dataset | sklearn fit | ferrolearn fit | fit speedup | sklearn predict | ferrolearn predict | predict speedup | metric | sklearn | ferrolearn | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AgglomerativeClustering | small_1Kx10 | 6.9 ms | 132.4 ms | 0.05x | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |
| AgglomerativeClustering | tiny_200x5 | 550.4 us | 1.1 ms | 0.49x | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |
| Birch | small_1Kx10 | 23.0 ms | 5.3 ms | **4.4x** | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |
| Birch | tiny_200x5 | 4.6 ms | 849.4 us | **5.5x** | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |
| DBSCAN | small_1Kx10 | 3.2 ms | 2.3 ms | 1.39x | — | — | — | ari | 0.0000 | 0.0000 | +0.0000 |
| DBSCAN | tiny_200x5 | 529.5 us | 70.1 us | **7.6x** | — | — | — | ari | 0.0000 | 0.0000 | +0.0000 |
| GaussianMixture | medium_5Kx20 | 38.1 ms | 309.3 ms | 0.12x | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |
| GaussianMixture | small_1Kx10 | 6.6 ms | 21.1 ms | 0.31x | — | — | — | ari | 1.0000 | 0.8344 | -0.1656 |
| GaussianMixture | tiny_200x5 | 1.9 ms | 2.9 ms | 0.66x | — | — | — | ari | 1.0000 | 0.7270 | -0.2730 |
| KMeans | medium_5Kx20 | 56.0 ms | 86.0 ms | 0.65x | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |
| KMeans | small_1Kx10 | 10.0 ms | 4.0 ms | **2.5x** | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |
| KMeans | tiny_200x5 | 3.3 ms | 239.1 us | **13.7x** | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |
| MiniBatchKMeans | medium_5Kx20 | 13.4 ms | 12.0 ms | 1.12x | — | — | — | ari | 1.0000 | 0.8366 | -0.1634 |
| MiniBatchKMeans | small_1Kx10 | 17.5 ms | 7.3 ms | **2.4x** | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |
| MiniBatchKMeans | tiny_200x5 | 2.1 ms | 6.4 ms | 0.32x | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |

### Decomp — 15 comparisons

| Algorithm | Dataset | sklearn fit | ferrolearn fit | fit speedup | sklearn predict | ferrolearn predict | predict speedup | metric | sklearn | ferrolearn | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FactorAnalysis | small_1Kx10 | 9.3 ms | 34.1 ms | 0.27x | 49.8 us | 19.3 us | **2.6x** | recon_rel | — | — | — |
| FactorAnalysis | tiny_50x5 | 1.6 ms | 163.9 us | **9.5x** | 37.0 us | 2.2 us | **17.0x** | recon_rel | — | — | — |
| FastICA | small_1Kx10 | 26.5 ms | 13.4 ms | 1.98x | 49.2 us | 21.5 us | **2.3x** | recon_rel | 6.787e-01 | — | — |
| FastICA | tiny_50x5 | 8.8 ms | 121.2 us | **72.4x** | 25.0 us | 2.1 us | **11.7x** | recon_rel | 3.394e-01 | — | — |
| IncrementalPCA | medium_10Kx100 | 231.6 ms | 124.9 ms | 1.85x | 801.4 us | 1.3 ms | 0.63x | recon_rel | 9.722e-01 | — | — |
| IncrementalPCA | small_1Kx10 | 2.4 ms | 52.3 us | **46.1x** | 32.5 us | 10.0 us | **3.3x** | recon_rel | 6.969e-01 | — | — |
| IncrementalPCA | tiny_50x5 | 298.1 us | 3.8 us | **77.5x** | 24.3 us | 1.6 us | **15.5x** | recon_rel | 3.483e-01 | — | — |
| PCA | medium_10Kx100 | 3.7 ms | 14.7 ms | 0.25x | 439.9 us | 1.8 ms | 0.24x | recon_rel | 9.698e-01 | 9.698e-01 | 1.00x |
| PCA | small_1Kx10 | 211.7 us | 50.6 us | **4.2x** | 36.8 us | 29.4 us | 1.25x | recon_rel | 6.787e-01 | 6.787e-01 | 1.00x |
| PCA | tiny_50x5 | 150.9 us | 24.9 us | **6.1x** | 24.2 us | 19.3 us | 1.25x | recon_rel | 3.394e-01 | 3.394e-01 | 1.00x |
| SparsePCA | small_1Kx10 | 252.2 ms | 432.0 ms | 0.58x | 336.8 us | 10.4 us | **32.3x** | recon_rel | 6.875e-01 | — | — |
| SparsePCA | tiny_50x5 | 9.6 ms | 1.7 ms | **5.5x** | 196.9 us | 1.7 us | **114.3x** | recon_rel | 3.541e-01 | — | — |
| TruncatedSVD | medium_10Kx100 | 20.3 ms | 6.8 ms | **3.0x** | 4.0 ms | 791.4 us | **5.0x** | recon_rel | 9.713e-01 | — | — |
| TruncatedSVD | small_1Kx10 | 713.9 us | 200.4 us | **3.6x** | 25.2 us | 7.6 us | **3.3x** | recon_rel | 6.790e-01 | — | — |
| TruncatedSVD | tiny_50x5 | 414.5 us | 5.8 us | **71.6x** | 19.9 us | 1.5 us | **13.6x** | recon_rel | 3.406e-01 | — | — |

### Kernel — 6 comparisons

| Algorithm | Dataset | sklearn fit | ferrolearn fit | fit speedup | sklearn predict | ferrolearn predict | predict speedup | metric | sklearn | ferrolearn | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Nystroem | medium_10Kx100 | 40.0 ms | 4.5 ms | **8.9x** | 13.2 ms | 62.0 ms | 0.21x | timing_only | — | — | — |
| Nystroem | small_1Kx10 | 16.1 ms | 4.2 ms | **3.8x** | 16.0 ms | 6.1 ms | **2.6x** | timing_only | — | — | — |
| Nystroem | tiny_50x5 | 13.3 ms | 217.2 us | **61.1x** | 251.9 us | 77.4 us | **3.3x** | timing_only | — | — | — |
| RBFSampler | medium_10Kx100 | 413.5 us | 299.5 us | 1.38x | 15.2 ms | 16.9 ms | 0.90x | timing_only | — | — | — |
| RBFSampler | small_1Kx10 | 152.6 us | 5.3 us | **28.5x** | 1.1 ms | 1.1 ms | 1.06x | timing_only | — | — | — |
| RBFSampler | tiny_50x5 | 193.8 us | 3.9 us | **49.3x** | 83.5 us | 48.4 us | 1.72x | timing_only | — | — | — |

### Preprocess — 14 comparisons

| Algorithm | Dataset | sklearn fit | ferrolearn fit | fit speedup | sklearn predict | ferrolearn predict | predict speedup | metric | sklearn | ferrolearn | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MaxAbsScaler | medium_10Kx100 | 1.4 ms | 1.1 ms | 1.37x | 1.2 ms | 2.0 ms | 0.59x | rel_diff_vs_sklearn | 0.000e+00 | 0.000e+00 | — |
| MaxAbsScaler | small_1Kx10 | 70.2 us | 8.8 us | **8.0x** | 31.6 us | 12.2 us | **2.6x** | rel_diff_vs_sklearn | 0.000e+00 | 0.000e+00 | — |
| MaxAbsScaler | tiny_50x5 | 67.2 us | 1.3 us | **53.5x** | 32.8 us | 2.3 us | **14.4x** | rel_diff_vs_sklearn | 0.000e+00 | 0.000e+00 | — |
| MinMaxScaler | medium_10Kx100 | 1.6 ms | 1.9 ms | 0.84x | 1.7 ms | 4.7 ms | 0.37x | rel_diff_vs_sklearn | 0.000e+00 | 1.146e-16 | — |
| MinMaxScaler | small_1Kx10 | 100.2 us | 15.1 us | **6.6x** | 33.7 us | 12.1 us | **2.8x** | rel_diff_vs_sklearn | 0.000e+00 | 1.088e-16 | — |
| MinMaxScaler | tiny_50x5 | 92.3 us | 1.3 us | **71.4x** | 30.8 us | 2.4 us | **12.9x** | rel_diff_vs_sklearn | 0.000e+00 | 8.444e-17 | — |
| PowerTransformer | small_1Kx10 | 31.1 ms | 1.7 ms | **18.5x** | 388.1 us | 113.6 us | **3.4x** | rel_diff_vs_sklearn | 0.000e+00 | 4.203e-01 | — |
| PowerTransformer | tiny_50x5 | 14.8 ms | 41.7 us | **355.3x** | 101.2 us | 3.6 us | **28.1x** | rel_diff_vs_sklearn | 0.000e+00 | 4.189e-01 | — |
| RobustScaler | medium_10Kx100 | 26.9 ms | 14.2 ms | 1.90x | 1.4 ms | 2.1 ms | 0.65x | rel_diff_vs_sklearn | 0.000e+00 | 0.000e+00 | — |
| RobustScaler | small_1Kx10 | 639.0 us | 107.2 us | **6.0x** | 37.5 us | 12.0 us | **3.1x** | rel_diff_vs_sklearn | 0.000e+00 | 4.251e-20 | — |
| RobustScaler | tiny_50x5 | 454.5 us | 3.5 us | **129.4x** | 31.8 us | 2.3 us | **13.8x** | rel_diff_vs_sklearn | 0.000e+00 | 2.450e-17 | — |
| StandardScaler | medium_10Kx100 | 2.6 ms | 1.8 ms | 1.49x | 1.4 ms | 3.1 ms | 0.45x | rel_diff_vs_sklearn | 0.000e+00 | 0.000e+00 | — |
| StandardScaler | small_1Kx10 | 185.1 us | 28.1 us | **6.6x** | 40.6 us | 31.2 us | 1.30x | rel_diff_vs_sklearn | 0.000e+00 | 0.000e+00 | — |
| StandardScaler | tiny_50x5 | 168.2 us | 24.0 us | **7.0x** | 39.7 us | 27.3 us | 1.45x | rel_diff_vs_sklearn | 0.000e+00 | 0.000e+00 | — |

### Regressor — 43 comparisons

| Algorithm | Dataset | sklearn fit | ferrolearn fit | fit speedup | sklearn predict | ferrolearn predict | predict speedup | metric | sklearn | ferrolearn | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ARDRegression | medium_10Kx100 | 765.4 ms | 61.1 ms | **12.5x** | 122.7 us | 84.5 us | 1.45x | r2 | 1.0000 | 1.0000 | +0.0000 |
| ARDRegression | small_1Kx10 | 735.7 us | 51.8 us | **14.2x** | 18.7 us | 1.2 us | **15.9x** | r2 | 1.0000 | 1.0000 | +0.0000 |
| ARDRegression | tiny_50x5 | 732.8 us | 5.4 us | **134.7x** | 18.3 us | 505 ns | **36.2x** | r2 | 1.0000 | 1.0000 | +0.0000 |
| BayesianRidge | medium_10Kx100 | 206.9 ms | 23.6 ms | **8.7x** | 92.4 us | 81.7 us | 1.13x | r2 | 1.0000 | 1.0000 | +0.0000 |
| BayesianRidge | small_1Kx10 | 390.5 us | 59.5 us | **6.6x** | 16.9 us | 1.2 us | **14.1x** | r2 | 1.0000 | 1.0000 | +0.0000 |
| BayesianRidge | tiny_50x5 | 348.5 us | 5.4 us | **64.0x** | 17.8 us | 470 ns | **37.8x** | r2 | 1.0000 | 1.0000 | -0.0000 |
| DecisionTreeRegressor | medium_10Kx100 | 443.7 ms | 242.9 ms | 1.83x | 321.4 us | 286.3 us | 1.12x | r2 | -0.5324 | -0.5521 | -0.0197 |
| DecisionTreeRegressor | small_1Kx10 | 3.2 ms | 1.7 ms | 1.90x | 35.9 us | 6.5 us | **5.5x** | r2 | 0.6236 | 0.6146 | -0.0090 |
| DecisionTreeRegressor | tiny_50x5 | 447.5 us | 45.0 us | **9.9x** | 72.2 us | 1.7 us | **42.7x** | r2 | -2.2859 | -2.1071 | +0.1788 |
| ElasticNet | medium_10Kx100 | 7.4 ms | 15.3 ms | 0.48x | 148.3 us | 191.0 us | 0.78x | r2 | 0.8870 | 0.8870 | -0.0000 |
| ElasticNet | small_1Kx10 | 226.0 us | 98.0 us | **2.3x** | 20.1 us | 21.3 us | 0.95x | r2 | 0.8761 | 0.8761 | -0.0000 |
| ElasticNet | tiny_50x5 | 172.5 us | 34.7 us | **5.0x** | 18.3 us | 22.7 us | 0.81x | r2 | 0.8193 | 0.8193 | -0.0000 |
| ExtraTreesRegressor | medium_10Kx100 | 554.2 ms | 409.5 ms | 1.35x | 25.1 ms | 73.3 ms | 0.34x | r2 | 0.3724 | 0.3690 | -0.0034 |
| ExtraTreesRegressor | small_1Kx10 | 46.8 ms | 6.8 ms | **6.9x** | 25.2 ms | 2.1 ms | **11.8x** | r2 | 0.8793 | 0.8875 | +0.0082 |
| ExtraTreesRegressor | tiny_50x5 | 44.4 ms | 1.6 ms | **28.4x** | 14.0 ms | 24.7 us | **566.5x** | r2 | 0.2541 | 0.1803 | -0.0738 |
| GradientBoostingRegressor | small_1Kx10 | 123.1 ms | 51.7 ms | **2.4x** | 252.1 us | 81.5 us | **3.1x** | r2 | 0.9268 | 0.9269 | +0.0001 |
| GradientBoostingRegressor | tiny_50x5 | 15.6 ms | 1.0 ms | **15.0x** | 93.8 us | 4.9 us | **19.2x** | r2 | -0.1346 | -0.1405 | -0.0059 |
| HistGradientBoostingRegressor | medium_10Kx100 | 282.0 ms | 965.5 ms | 0.29x | 1.6 ms | 12.9 ms | 0.12x | r2 | 0.6349 | 0.6349 | +0.0000 |
| HistGradientBoostingRegressor | small_1Kx10 | 115.4 ms | 41.5 ms | **2.8x** | 666.1 us | 1.2 ms | 0.58x | r2 | 0.9394 | 0.9405 | +0.0012 |
| HistGradientBoostingRegressor | tiny_50x5 | 24.1 ms | 180.6 us | **133.6x** | 563.6 us | 2.8 us | **202.6x** | r2 | -0.2571 | -0.2571 | +0.0000 |
| HuberRegressor | medium_10Kx100 | 732.8 ms | 54.8 ms | **13.4x** | 115.6 us | 83.5 us | 1.38x | r2 | 1.0000 | 1.0000 | -0.0000 |
| HuberRegressor | small_1Kx10 | 7.0 ms | 69.4 us | **100.7x** | 22.6 us | 1.2 us | **18.8x** | r2 | 1.0000 | 1.0000 | -0.0000 |
| HuberRegressor | tiny_50x5 | 2.5 ms | 4.4 us | **557.3x** | 18.0 us | 786 ns | **22.9x** | r2 | 1.0000 | 1.0000 | +0.0000 |
| KNeighborsRegressor | medium_10Kx100 | 592.5 us | 15.1 ms | 0.04x | 19.5 ms | 45.3 ms | 0.43x | r2 | 0.3173 | 0.3173 | +0.0000 |
| KNeighborsRegressor | small_1Kx10 | 385.2 us | 316.1 us | 1.22x | 14.1 ms | 7.4 ms | 1.89x | r2 | 0.7790 | 0.7790 | +0.0000 |
| KNeighborsRegressor | tiny_50x5 | 120.9 us | 6.1 us | **19.8x** | 15.6 ms | 12.1 us | **1287.6x** | r2 | 0.6307 | 0.6307 | +0.0000 |
| KernelRidge | small_1Kx10 | 87.2 ms | 37.5 ms | **2.3x** | 382.2 us | 2.3 ms | 0.17x | r2 | 1.0000 | 1.0000 | +0.0000 |
| KernelRidge | tiny_50x5 | 172.9 us | 12.6 us | **13.7x** | 105.7 us | 2.2 us | **47.2x** | r2 | 0.9988 | 0.9988 | +0.0000 |
| Lasso | medium_10Kx100 | 18.6 ms | 31.8 ms | 0.59x | 163.3 us | 216.2 us | 0.76x | r2 | 0.9997 | 0.9997 | +0.0000 |
| Lasso | small_1Kx10 | 202.5 us | 100.4 us | **2.0x** | 19.2 us | 18.5 us | 1.04x | r2 | 0.9994 | 0.9994 | -0.0000 |
| Lasso | tiny_50x5 | 214.1 us | 34.2 us | **6.3x** | 18.6 us | 16.0 us | 1.16x | r2 | 0.9995 | 0.9995 | +0.0000 |
| LinearRegression | medium_10Kx100 | 81.4 ms | 5.2 ms | **15.7x** | 84.9 us | 167.7 us | 0.51x | r2 | 1.0000 | 1.0000 | +0.0000 |
| LinearRegression | small_1Kx10 | 228.6 us | 63.2 us | **3.6x** | 17.8 us | 20.4 us | 0.87x | r2 | 1.0000 | 1.0000 | +0.0000 |
| LinearRegression | tiny_50x5 | 214.4 us | 38.7 us | **5.5x** | 17.9 us | 16.6 us | 1.08x | r2 | 1.0000 | 1.0000 | +0.0000 |
| QuantileRegressor | medium_10Kx100 | 1.46 s | 1.31 s | 1.11x | 74.1 us | 67.4 us | 1.10x | r2 | -0.0017 | 1.0000 | +1.0017 |
| QuantileRegressor | small_1Kx10 | 17.5 ms | 2.4 ms | **7.3x** | 20.0 us | 1.2 us | **16.4x** | r2 | -0.0112 | 1.0000 | +1.0112 |
| QuantileRegressor | tiny_50x5 | 2.0 ms | 7.6 us | **269.1x** | 46.5 us | 1.4 us | **32.3x** | r2 | -0.0488 | -0.0717 | -0.0229 |
| RandomForestRegressor | medium_10Kx100 | 1.57 s | 1.65 s | 0.95x | 25.0 ms | 53.9 ms | 0.46x | r2 | 0.3759 | 0.3810 | +0.0051 |
| RandomForestRegressor | small_1Kx10 | 77.8 ms | 13.6 ms | **5.7x** | 25.7 ms | 1.5 ms | **17.0x** | r2 | 0.8446 | 0.8415 | -0.0031 |
| RandomForestRegressor | tiny_50x5 | 58.4 ms | 1.6 ms | **37.2x** | 13.9 ms | 17.4 us | **798.6x** | r2 | -0.7390 | -0.8137 | -0.0747 |
| Ridge | medium_10Kx100 | 12.7 ms | 5.3 ms | **2.4x** | 95.9 us | 158.3 us | 0.61x | r2 | 1.0000 | 1.0000 | +0.0000 |
| Ridge | small_1Kx10 | 262.2 us | 56.7 us | **4.6x** | 18.3 us | 18.7 us | 0.98x | r2 | 1.0000 | 1.0000 | +0.0000 |
| Ridge | tiny_50x5 | 321.6 us | 59.7 us | **5.4x** | 17.7 us | 16.8 us | 1.05x | r2 | 0.9988 | 0.9988 | +0.0000 |

## Summary — geometric mean speedup

| Family | n compared | fit geomean | predict geomean | accuracy parity (mean Δ) |
|---|---:|---:|---:|---:|
| classifier | 51 | 6.85x | 9.01x | +0.0039 |
| cluster | 15 | 1.08x | — | -0.0401 |
| decomp | 15 | 5.08x | 4.69x | — |
| kernel | 6 | 12.62x | 1.20x | — |
| preprocess | 14 | 10.31x | 2.61x | — |
| regressor | 43 | 6.30x | 4.22x | +0.0464 |

