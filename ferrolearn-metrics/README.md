# ferrolearn-metrics

Evaluation metrics for the [ferrolearn](https://crates.io/crates/ferrolearn) machine learning framework.

## Classification metrics

| Function | Description |
|----------|-------------|
| `accuracy_score` | Fraction of correctly classified samples |
| `precision_score` | Positive predictive value (macro, micro, weighted, binary) |
| `recall_score` | Sensitivity / true positive rate |
| `f1_score` | Harmonic mean of precision and recall |
| `roc_auc_score` | Area under the ROC curve (binary) |
| `confusion_matrix` | Matrix of true vs. predicted counts |
| `log_loss` | Cross-entropy loss for probabilistic classifiers |

## Regression metrics

| Function | Description |
|----------|-------------|
| `mean_absolute_error` | Mean of absolute residuals |
| `mean_squared_error` | Mean of squared residuals |
| `root_mean_squared_error` | Square root of MSE |
| `r2_score` | Coefficient of determination |
| `mean_absolute_percentage_error` | MAPE (returned as a percentage, not a fraction) |
| `explained_variance_score` | Fraction of variance explained |

## Clustering metrics

| Function | Description |
|----------|-------------|
| `silhouette_score` | Mean silhouette coefficient over all samples |
| `adjusted_rand_score` | Adjusted Rand Index between two clusterings |
| `adjusted_mutual_info` | Adjusted Mutual Information |
| `davies_bouldin_score` | Davies-Bouldin Index (lower is better) |

## Example

```rust
use ferrolearn_metrics::{accuracy_score, f1_score, Average};
use ferrolearn_metrics::regression::r2_score;
use ndarray::array;

let y_true = array![0usize, 1, 2, 1, 0];
let y_pred = array![0usize, 1, 2, 0, 0];
let acc = accuracy_score(&y_true, &y_pred).unwrap();
let f1 = f1_score(&y_true, &y_pred, Average::Macro).unwrap();
```

## License

Licensed under either of [Apache License, Version 2.0](../LICENSE-APACHE) or [MIT License](../LICENSE-MIT) at your option.
