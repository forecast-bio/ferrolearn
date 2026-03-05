# ferrolearn-preprocess

Data preprocessing transformers for the [ferrolearn](https://crates.io/crates/ferrolearn) machine learning framework.

## Scalers

| Transformer | Description |
|-------------|-------------|
| `StandardScaler` | Zero-mean, unit-variance scaling |
| `MinMaxScaler` | Scale features to a given range (default [0, 1]) |
| `RobustScaler` | Median/IQR-based scaling, robust to outliers |
| `MaxAbsScaler` | Scale by maximum absolute value to [-1, 1] |
| `Normalizer` | Normalize each sample (row) to unit norm |
| `PowerTransformer` | Yeo-Johnson power transform for Gaussian-like distributions |

## Encoders

| Transformer | Description |
|-------------|-------------|
| `OneHotEncoder` | Encode categorical columns as binary indicator columns |
| `OrdinalEncoder` | Map string categories to integers by order of appearance |
| `LabelEncoder` | Map string labels to integer indices |

## Imputers

| Transformer | Description |
|-------------|-------------|
| `SimpleImputer` | Fill missing (NaN) values using mean, median, most frequent, or constant |

## Feature selection

| Transformer | Description |
|-------------|-------------|
| `VarianceThreshold` | Remove features with variance below a threshold |
| `SelectKBest` | Keep the K features with highest ANOVA F-scores |
| `SelectFromModel` | Keep features whose model-derived importance exceeds a threshold |

## Feature engineering

| Transformer | Description |
|-------------|-------------|
| `PolynomialFeatures` | Generate polynomial and interaction features |
| `Binarizer` | Threshold features to binary values |
| `FunctionTransformer` | Apply a user-provided function element-wise |
| `ColumnTransformer` | Apply different transformers to different column subsets |

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

Licensed under either of [Apache License, Version 2.0](../LICENSE-APACHE) or [MIT License](../LICENSE-MIT) at your option.
