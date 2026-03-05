# ferrolearn-sparse

Sparse matrix types for the [ferrolearn](https://crates.io/crates/ferrolearn) machine learning framework.

## Formats

| Type | Description |
|------|-------------|
| `CsrMatrix<T>` | Compressed Sparse Row — efficient for row slicing and matrix-vector products |
| `CscMatrix<T>` | Compressed Sparse Column — efficient for column slicing and transpose products |
| `CooMatrix<T>` | Coordinate (triplet) format — convenient for incremental construction |

All formats support:
- Conversion between formats and to/from dense `ndarray::Array2<T>`
- Scalar multiplication and element-wise addition
- Matrix-vector multiplication
- `CsrMatrix` implements the `ferrolearn_core::Dataset` trait

Backed by the [`sprs`](https://crates.io/crates/sprs) crate.

## Example

```rust
use ferrolearn_sparse::{CooMatrix, CsrMatrix};

// Build in COO format, then convert to CSR
let mut coo = CooMatrix::new(3, 3);
coo.push(0, 0, 1.0_f64);
coo.push(1, 2, 4.0);
coo.push(2, 1, 7.0);

let csr = CsrMatrix::from_coo(&coo).unwrap();
let dense = csr.to_dense();
assert_eq!(dense[[0, 0]], 1.0);
assert_eq!(dense[[1, 2]], 4.0);
```

## License

Licensed under either of [Apache License, Version 2.0](../LICENSE-APACHE) or [MIT License](../LICENSE-MIT) at your option.
