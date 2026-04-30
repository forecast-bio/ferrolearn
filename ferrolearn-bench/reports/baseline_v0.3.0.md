# ferrolearn vs scikit-learn — head-to-head report

Each row is a single (algorithm, dataset) head-to-head: same canonical
dataset (sklearn `make_*`), same train/test split, same hyperparameters,
same quality metric, both libraries fit + predict in the same Python
process. Δ is `ferrolearn − sklearn` for the quality metric (positive
means ferrolearn is more accurate; for `recon_rel`, lower is better and
the cell shows `ferrolearn / sklearn` ratio).

### Classifier — 15 comparisons

| Algorithm | Dataset | sklearn fit | ferrolearn fit | fit speedup | sklearn predict | ferrolearn predict | predict speedup | metric | sklearn | ferrolearn | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DecisionTreeClassifier | medium_10Kx100 | 596.4 ms | 280.7 ms | **2.1x** | 241.1 us | 288.3 us | 0.84x | accuracy | 73.65% | 71.95% | -1.70pp |
| DecisionTreeClassifier | small_1Kx10 | 3.6 ms | 2.1 ms | 1.76x | 93.2 us | 34.7 us | **2.7x** | accuracy | 89.50% | 90.00% | +0.50pp |
| DecisionTreeClassifier | tiny_50x5 | 385.8 us | 119.1 us | **3.2x** | 49.2 us | 23.7 us | **2.1x** | accuracy | 90.00% | 80.00% | -10.00pp |
| GaussianNB | medium_10Kx100 | 5.3 ms | 2.3 ms | **2.3x** | 673.6 us | 1.4 ms | 0.50x | accuracy | 81.55% | 81.55% | +0.00pp |
| GaussianNB | small_1Kx10 | 366.6 us | 239.0 us | 1.53x | 49.0 us | 33.2 us | 1.48x | accuracy | 89.00% | 89.00% | +0.00pp |
| GaussianNB | tiny_50x5 | 236.4 us | 78.2 us | **3.0x** | 37.0 us | 16.8 us | **2.2x** | accuracy | 100.00% | 100.00% | +0.00pp |
| KNeighborsClassifier | medium_10Kx100 | 920.6 us | 17.4 ms | 0.05x | 17.9 ms | 44.2 ms | 0.40x | accuracy | 96.60% | 96.60% | +0.00pp |
| KNeighborsClassifier | small_1Kx10 | 378.1 us | 493.8 us | 0.77x | 1.3 ms | 6.5 ms | 0.20x | accuracy | 91.50% | 91.50% | +0.00pp |
| KNeighborsClassifier | tiny_50x5 | 174.1 us | 84.6 us | **2.1x** | 515.0 us | 28.1 us | **18.3x** | accuracy | 90.00% | 90.00% | +0.00pp |
| LogisticRegression | medium_10Kx100 | 697.9 ms | 48.7 ms | **14.3x** | 102.9 us | 188.6 us | 0.55x | accuracy | 83.50% | 83.45% | -0.05pp |
| LogisticRegression | small_1Kx10 | 763.2 us | 658.2 us | 1.16x | 78.7 us | 54.0 us | 1.46x | accuracy | 83.50% | 82.00% | -1.50pp |
| LogisticRegression | tiny_50x5 | 658.6 us | 135.0 us | **4.9x** | 34.2 us | 25.4 us | 1.35x | accuracy | 100.00% | 100.00% | +0.00pp |
| RandomForestClassifier | medium_10Kx100 | 270.4 ms | 202.0 ms | 1.34x | 24.7 ms | 34.5 ms | 0.72x | accuracy | 94.25% | 78.20% | -16.05pp |
| RandomForestClassifier | small_1Kx10 | 116.6 ms | 6.3 ms | **18.4x** | 25.2 ms | 1.4 ms | **18.5x** | accuracy | 94.50% | 92.00% | -2.50pp |
| RandomForestClassifier | tiny_50x5 | 79.3 ms | 1.7 ms | **47.8x** | 14.1 ms | 25.5 us | **551.9x** | accuracy | 80.00% | 90.00% | +10.00pp |

### Cluster — 3 comparisons

| Algorithm | Dataset | sklearn fit | ferrolearn fit | fit speedup | sklearn predict | ferrolearn predict | predict speedup | metric | sklearn | ferrolearn | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| KMeans | medium_5Kx20 | 37.0 ms | 85.8 ms | 0.43x | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |
| KMeans | small_1Kx10 | 9.0 ms | 3.8 ms | **2.4x** | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |
| KMeans | tiny_200x5 | 3.1 ms | 237.8 us | **12.9x** | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |

### Decomp — 3 comparisons

| Algorithm | Dataset | sklearn fit | ferrolearn fit | fit speedup | sklearn predict | ferrolearn predict | predict speedup | metric | sklearn | ferrolearn | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PCA | medium_10Kx100 | 3.6 ms | 14.0 ms | 0.26x | 441.8 us | 7.2 ms | 0.06x | recon_rel | 9.698e-01 | 9.698e-01 | 1.00x |
| PCA | small_1Kx10 | 161.2 us | 51.4 us | **3.1x** | 47.8 us | 30.2 us | 1.58x | recon_rel | 6.787e-01 | 6.787e-01 | 1.00x |
| PCA | tiny_50x5 | 200.1 us | 21.3 us | **9.4x** | 22.7 us | 21.1 us | 1.07x | recon_rel | 3.394e-01 | 3.394e-01 | 1.00x |

### Preprocess — 3 comparisons

| Algorithm | Dataset | sklearn fit | ferrolearn fit | fit speedup | sklearn predict | ferrolearn predict | predict speedup | metric | sklearn | ferrolearn | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| StandardScaler | medium_10Kx100 | 4.0 ms | 2.4 ms | 1.65x | 3.5 ms | 6.8 ms | 0.52x | rel_diff_vs_sklearn | 0.000e+00 | 0.000e+00 | — |
| StandardScaler | small_1Kx10 | 336.6 us | 69.6 us | **4.8x** | 104.4 us | 80.8 us | 1.29x | rel_diff_vs_sklearn | 0.000e+00 | 0.000e+00 | — |
| StandardScaler | tiny_50x5 | 257.1 us | 49.4 us | **5.2x** | 75.5 us | 49.5 us | 1.53x | rel_diff_vs_sklearn | 0.000e+00 | 0.000e+00 | — |

### Regressor — 12 comparisons

| Algorithm | Dataset | sklearn fit | ferrolearn fit | fit speedup | sklearn predict | ferrolearn predict | predict speedup | metric | sklearn | ferrolearn | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ElasticNet | medium_10Kx100 | 2.9 ms | 10.5 ms | 0.27x | 86.7 us | 185.0 us | 0.47x | r2 | 0.8870 | 0.8870 | -0.0000 |
| ElasticNet | small_1Kx10 | 192.4 us | 102.9 us | 1.87x | 22.6 us | 18.6 us | 1.21x | r2 | 0.8761 | 0.8761 | -0.0000 |
| ElasticNet | tiny_50x5 | 231.3 us | 36.9 us | **6.3x** | 19.3 us | 17.3 us | 1.12x | r2 | 0.8193 | 0.8193 | -0.0000 |
| Lasso | medium_10Kx100 | 27.6 ms | 15.3 ms | 1.80x | 84.3 us | 163.2 us | 0.52x | r2 | 0.9997 | 0.9997 | +0.0000 |
| Lasso | small_1Kx10 | 200.5 us | 91.8 us | **2.2x** | 24.9 us | 17.8 us | 1.40x | r2 | 0.9994 | 0.9994 | -0.0000 |
| Lasso | tiny_50x5 | 171.9 us | 40.1 us | **4.3x** | 24.7 us | 16.8 us | 1.46x | r2 | 0.9995 | 0.9995 | +0.0000 |
| LinearRegression | medium_10Kx100 | 23.1 ms | 5.4 ms | **4.3x** | 86.3 us | 168.6 us | 0.51x | r2 | 1.0000 | 1.0000 | +0.0000 |
| LinearRegression | small_1Kx10 | 256.2 us | 59.1 us | **4.3x** | 18.5 us | 17.7 us | 1.04x | r2 | 1.0000 | 1.0000 | +0.0000 |
| LinearRegression | tiny_50x5 | 191.9 us | 34.8 us | **5.5x** | 17.4 us | 19.1 us | 0.91x | r2 | 1.0000 | 1.0000 | +0.0000 |
| Ridge | medium_10Kx100 | 29.6 ms | 8.3 ms | **3.6x** | 95.7 us | 171.4 us | 0.56x | r2 | 1.0000 | 1.0000 | +0.0000 |
| Ridge | small_1Kx10 | 241.9 us | 56.5 us | **4.3x** | 17.9 us | 17.9 us | 1.00x | r2 | 1.0000 | 1.0000 | +0.0000 |
| Ridge | tiny_50x5 | 226.3 us | 37.9 us | **6.0x** | 17.5 us | 16.0 us | 1.09x | r2 | 0.9988 | 0.9988 | +0.0000 |

## Summary — geometric mean speedup

| Family | n compared | fit geomean | predict geomean | accuracy parity (mean Δ) |
|---|---:|---:|---:|---:|
| classifier | 15 | 2.52x | 2.13x | -0.0142 |
| cluster | 3 | 2.37x | — | +0.0000 |
| decomp | 3 | 1.96x | 0.47x | — |
| preprocess | 3 | 3.46x | 1.01x | — |
| regressor | 12 | 2.98x | 0.87x | -0.0000 |

