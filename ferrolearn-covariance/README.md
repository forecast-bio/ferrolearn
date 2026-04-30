# ferrolearn-covariance

Covariance estimators for the [ferrolearn](https://github.com/dollspace-gay/ferrolearn) machine learning framework.

A scikit-learn equivalent for Rust, providing:

- `EmpiricalCovariance` — maximum-likelihood covariance.
- `ShrunkCovariance` — fixed shrinkage toward a scaled identity.
- `LedoitWolf` — optimal Ledoit–Wolf shrinkage.
- `OAS` — Oracle Approximating Shrinkage.
- `MinCovDet` — Minimum Covariance Determinant via FAST-MCD.
- `EllipticEnvelope` — outlier detection on top of MCD.
- `GraphicalLasso` / `GraphicalLassoCV` — sparse inverse-covariance estimation.

Plus the function-style helpers `empirical_covariance`, `shrunk_covariance`, `ledoit_wolf`, `oas`, `log_likelihood`, `graphical_lasso`, and `fast_mcd`.

## License

Dual-licensed under MIT or Apache-2.0.
