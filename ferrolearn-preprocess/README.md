# ferrolearn-preprocess

Data preprocessing transformers for the
[ferrolearn](https://crates.io/crates/ferrolearn) machine learning framework.
Validated against scikit-learn 1.8.0 head-to-head — **bit-identical numerical
output** (relative diff ≤ 1e-16) for `StandardScaler`, `MinMaxScaler`,
`MaxAbsScaler`, and `RobustScaler`. See the
[workspace BENCHMARKS.md](../BENCHMARKS.md).

## Scalers

| Transformer | Description |
|-------------|-------------|
| `StandardScaler` | Zero-mean, unit-variance scaling |
| `MinMaxScaler` | Scale features to a given range (default [0, 1]) |
| `RobustScaler` | Median/IQR-based scaling, robust to outliers |
| `MaxAbsScaler` | Scale by maximum absolute value to [-1, 1] |
| `Normalizer` | Normalize each sample (row) to unit norm (L1, L2, max) |
| `PowerTransformer` | Yeo-Johnson power transform for Gaussian-like distributions |
| `QuantileTransformer` | Map to uniform or normal distribution via quantile matching |
| `KBinsDiscretizer` | Discretize continuous features into bins (uniform / quantile / kmeans) |

## Encoders

| Transformer | Description |
|-------------|-------------|
| `OneHotEncoder` | Encode categorical columns as binary indicator columns |
| `OrdinalEncoder` | Map categories to integers by order of appearance |
| `LabelEncoder` | Map labels to integer indices |
| `LabelBinarizer` / `MultiLabelBinarizer` | Binary indicator encoding for label vectors |
| `TargetEncoder` | Mean-target encoding for high-cardinality categoricals |
| `BinaryEncoder` | Binary base-2 encoding for high-cardinality categoricals |

## Imputers

| Transformer | Description |
|-------------|-------------|
| `SimpleImputer` | Fill missing (NaN) values: mean, median, most frequent, constant |
| `KNNImputer` | Fill missing values using k-nearest-neighbor average |
| `IterativeImputer` | Round-robin regression imputation (BayesianRidge by default) |

## Feature selection

| Transformer | Description |
|-------------|-------------|
| `VarianceThreshold` | Remove features with variance below a threshold |
| `SelectKBest` / `SelectPercentile` | Univariate feature selection |
| `SelectFromModel` | Threshold-based selection from any model with `coef_` / `feature_importances_` |
| `RFE` / `RFECV` | Recursive feature elimination |
| `SequentialFeatureSelector` | Forward / backward greedy selection |

## Feature engineering

| Transformer | Description |
|-------------|-------------|
| `PolynomialFeatures` | Polynomial and interaction feature expansion |
| `SplineTransformer` | B-spline basis expansion |
| `Binarizer` | Threshold features to {0, 1} |
| `FunctionTransformer` | Apply a user-provided function element-wise |
| `ColumnTransformer` | Apply different transformers to different column subsets |
| `RandomProjection` | Gaussian / sparse random projection (Johnson-Lindenstrauss) |

## Text feature extraction

| Transformer | Description |
|-------------|-------------|
| `CountVectorizer` | Bag-of-words frequency counts |
| `TfidfTransformer` | TF-IDF reweighting |

## Example

```rust
use ferrolearn_preprocess::StandardScaler;
use ferrolearn_core::FitTransform;
use ndarray::array;

let x = array![[1.0_f64, 10.0], [2.0, 20.0], [3.0, 30.0]];
let scaled = StandardScaler::<f64>::new().fit_transform(&x).unwrap();
// Each column now has mean ~= 0 and std ~= 1
```

All transformers implement `PipelineTransformer` for use inside a `Pipeline`.

## License

Licensed under either of [Apache License, Version 2.0](../LICENSE-APACHE) or
[MIT License](../LICENSE-MIT) at your option.
