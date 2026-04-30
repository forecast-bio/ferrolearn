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
| DecisionTreeClassifier | medium_10Kx100 | 601.2 ms | 283.6 ms | **2.1x** | 207.9 us | 390.0 us | 0.53x | accuracy | 73.65% | 71.95% | -1.70pp |
| DecisionTreeClassifier | small_1Kx10 | 2.5 ms | 1.2 ms | **2.1x** | 36.3 us | 25.5 us | 1.42x | accuracy | 89.50% | 90.00% | +0.50pp |
| DecisionTreeClassifier | tiny_50x5 | 530.4 us | 188.7 us | **2.8x** | 69.1 us | 46.0 us | 1.50x | accuracy | 90.00% | 80.00% | -10.00pp |
| GaussianNB | medium_10Kx100 | 5.6 ms | 2.6 ms | **2.1x** | 672.3 us | 1.3 ms | 0.51x | accuracy | 81.55% | 81.55% | +0.00pp |
| GaussianNB | small_1Kx10 | 347.5 us | 150.5 us | **2.3x** | 48.8 us | 38.0 us | 1.28x | accuracy | 89.00% | 89.00% | +0.00pp |
| GaussianNB | tiny_50x5 | 242.7 us | 82.8 us | **2.9x** | 35.9 us | 19.4 us | 1.85x | accuracy | 100.00% | 100.00% | +0.00pp |
| KNeighborsClassifier | medium_10Kx100 | 1.2 ms | 17.9 ms | 0.07x | 17.8 ms | 44.6 ms | 0.40x | accuracy | 96.60% | 96.60% | +0.00pp |
| KNeighborsClassifier | small_1Kx10 | 405.3 us | 554.2 us | 0.73x | 1.3 ms | 6.6 ms | 0.20x | accuracy | 91.50% | 91.50% | +0.00pp |
| KNeighborsClassifier | tiny_50x5 | 161.6 us | 82.7 us | 1.95x | 457.6 us | 29.0 us | **15.8x** | accuracy | 90.00% | 90.00% | +0.00pp |
| LogisticRegression | medium_10Kx100 | 677.2 ms | 48.3 ms | **14.0x** | 108.3 us | 202.8 us | 0.53x | accuracy | 83.50% | 83.45% | -0.05pp |
| LogisticRegression | small_1Kx10 | 777.0 us | 327.6 us | **2.4x** | 28.7 us | 19.6 us | 1.46x | accuracy | 83.50% | 82.00% | -1.50pp |
| LogisticRegression | tiny_50x5 | 1.2 ms | 193.8 us | **6.1x** | 71.4 us | 53.7 us | 1.33x | accuracy | 100.00% | 100.00% | +0.00pp |
| RandomForestClassifier | medium_10Kx100 | 279.4 ms | 195.1 ms | 1.43x | 25.1 ms | 29.1 ms | 0.86x | accuracy | 94.25% | 93.05% | -1.20pp |
| RandomForestClassifier | small_1Kx10 | 117.0 ms | 5.2 ms | **22.5x** | 24.8 ms | 926.0 us | **26.8x** | accuracy | 94.50% | 96.00% | +1.50pp |
| RandomForestClassifier | tiny_50x5 | 81.5 ms | 1.7 ms | **47.3x** | 14.0 ms | 26.0 us | **537.0x** | accuracy | 80.00% | 90.00% | +10.00pp |

### Cluster — 3 comparisons

| Algorithm | Dataset | sklearn fit | ferrolearn fit | fit speedup | sklearn predict | ferrolearn predict | predict speedup | metric | sklearn | ferrolearn | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| KMeans | medium_5Kx20 | 36.2 ms | 86.8 ms | 0.42x | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |
| KMeans | small_1Kx10 | 10.3 ms | 3.7 ms | **2.8x** | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |
| KMeans | tiny_200x5 | 3.1 ms | 219.2 us | **14.1x** | — | — | — | ari | 1.0000 | 1.0000 | +0.0000 |

### Decomp — 3 comparisons

| Algorithm | Dataset | sklearn fit | ferrolearn fit | fit speedup | sklearn predict | ferrolearn predict | predict speedup | metric | sklearn | ferrolearn | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PCA | medium_10Kx100 | 3.6 ms | 13.7 ms | 0.26x | 458.3 us | 7.3 ms | 0.06x | recon_rel | 9.698e-01 | 9.698e-01 | 1.00x |
| PCA | small_1Kx10 | 179.3 us | 59.7 us | **3.0x** | 37.9 us | 30.9 us | 1.22x | recon_rel | 6.787e-01 | 6.787e-01 | 1.00x |
| PCA | tiny_50x5 | 158.5 us | 23.5 us | **6.8x** | 22.7 us | 19.7 us | 1.15x | recon_rel | 3.394e-01 | 3.394e-01 | 1.00x |

### Preprocess — 3 comparisons

| Algorithm | Dataset | sklearn fit | ferrolearn fit | fit speedup | sklearn predict | ferrolearn predict | predict speedup | metric | sklearn | ferrolearn | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| StandardScaler | medium_10Kx100 | 2.8 ms | 2.8 ms | 0.99x | 1.4 ms | 6.5 ms | 0.22x | rel_diff_vs_sklearn | 0.000e+00 | 0.000e+00 | — |
| StandardScaler | small_1Kx10 | 233.2 us | 35.2 us | **6.6x** | 56.3 us | 43.2 us | 1.30x | rel_diff_vs_sklearn | 0.000e+00 | 0.000e+00 | — |
| StandardScaler | tiny_50x5 | 177.4 us | 24.0 us | **7.4x** | 39.6 us | 25.2 us | 1.57x | rel_diff_vs_sklearn | 0.000e+00 | 0.000e+00 | — |

### Regressor — 12 comparisons

| Algorithm | Dataset | sklearn fit | ferrolearn fit | fit speedup | sklearn predict | ferrolearn predict | predict speedup | metric | sklearn | ferrolearn | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ElasticNet | medium_10Kx100 | 4.3 ms | 14.2 ms | 0.31x | 109.5 us | 194.3 us | 0.56x | r2 | 0.8870 | 0.8870 | -0.0000 |
| ElasticNet | small_1Kx10 | 197.2 us | 104.3 us | 1.89x | 19.0 us | 17.8 us | 1.07x | r2 | 0.8761 | 0.8761 | -0.0000 |
| ElasticNet | tiny_50x5 | 177.1 us | 51.6 us | **3.4x** | 19.1 us | 17.6 us | 1.08x | r2 | 0.8193 | 0.8193 | -0.0000 |
| Lasso | medium_10Kx100 | 36.5 ms | 14.9 ms | **2.5x** | 118.9 us | 211.6 us | 0.56x | r2 | 0.9997 | 0.9997 | +0.0000 |
| Lasso | small_1Kx10 | 205.8 us | 91.5 us | **2.2x** | 20.4 us | 17.7 us | 1.15x | r2 | 0.9994 | 0.9994 | -0.0000 |
| Lasso | tiny_50x5 | 181.1 us | 45.1 us | **4.0x** | 18.4 us | 16.8 us | 1.09x | r2 | 0.9995 | 0.9995 | +0.0000 |
| LinearRegression | medium_10Kx100 | 72.0 ms | 10.6 ms | **6.8x** | 87.5 us | 210.0 us | 0.42x | r2 | 1.0000 | 1.0000 | +0.0000 |
| LinearRegression | small_1Kx10 | 267.8 us | 67.3 us | **4.0x** | 17.5 us | 17.7 us | 0.99x | r2 | 1.0000 | 1.0000 | +0.0000 |
| LinearRegression | tiny_50x5 | 184.1 us | 34.5 us | **5.3x** | 18.7 us | 16.6 us | 1.12x | r2 | 1.0000 | 1.0000 | +0.0000 |
| Ridge | medium_10Kx100 | 27.3 ms | 6.9 ms | **3.9x** | 86.6 us | 205.0 us | 0.42x | r2 | 1.0000 | 1.0000 | +0.0000 |
| Ridge | small_1Kx10 | 253.1 us | 56.3 us | **4.5x** | 17.9 us | 16.6 us | 1.08x | r2 | 1.0000 | 1.0000 | +0.0000 |
| Ridge | tiny_50x5 | 239.9 us | 34.4 us | **7.0x** | 16.7 us | 15.9 us | 1.05x | r2 | 0.9988 | 0.9988 | +0.0000 |

## Summary — geometric mean speedup

| Family | n compared | fit geomean | predict geomean | accuracy parity (mean Δ) |
|---|---:|---:|---:|---:|
| classifier | 15 | 2.81x | 1.95x | -0.0016 |
| cluster | 3 | 2.53x | — | +0.0000 |
| decomp | 3 | 1.75x | 0.45x | — |
| preprocess | 3 | 3.65x | 0.77x | — |
| regressor | 12 | 3.10x | 0.83x | -0.0000 |

