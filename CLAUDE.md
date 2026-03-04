# ferrolearn — Project Conventions

## Rust Edition & MSRV
- Edition: 2024
- MSRV: 1.85

## Import Paths
- Core traits: `use ferrolearn_core::{Fit, Predict, Transform, FitTransform}`
- Errors: `use ferrolearn_core::error::FerroError`
- Dataset: `use ferrolearn_core::dataset::Dataset`
- Array types: `use ndarray::{Array1, Array2, ArrayView1, ArrayView2}`
- Float bound: `use num_traits::Float`

## Error Handling
- All public functions return `Result<T, FerroError>`
- Use `thiserror` 2.0 for derive
- Never panic in library code
- Every error variant carries diagnostic context

## Numeric Generics
- Generic bound: `F: Float + Send + Sync + 'static`
- Support both f32 and f64
- Use `num_traits::{Zero, One}` where needed
- Default to f64 in examples and tests

## Testing Patterns
- Oracle fixtures: load JSON from `fixtures/`, compare with `float_cmp` ULP tolerance
- Property tests: `proptest` with `ProptestConfig::with_cases(256)`
- Fuzz: one target per public fit/transform/predict
- Compile-fail: `trybuild` for type-safety guarantees
- Every public function must have at least one unit test

## Naming Conventions
- Unfitted: `LinearRegression`, `StandardScaler`
- Fitted: `FittedLinearRegression`, `FittedStandardScaler`
- Traits for introspection: `HasCoefficients`, `HasFeatureImportances`, `HasClasses`
- Modules: snake_case matching the struct name (e.g., `linear_regression.rs`)

## Crate Dependencies (use these exact versions)
```toml
ndarray = "0.17"
faer = "0.24"
sprs = "0.11"
rayon = "1.11"
serde = { version = "1.0", features = ["derive"] }
num-traits = "0.2"
thiserror = "2.0"
approx = "0.5"
float-cmp = "0.10"
rand = "0.9"
```

## Code Style
- Run `cargo fmt` before committing
- Run `cargo clippy -p <crate> -- -D warnings` and fix all warnings
- Doc comments on every public item (structs, traits, functions, methods)
- Use `#[must_use]` on functions that return values that should not be ignored
- Prefer returning `Result` over panicking — even for "impossible" cases

## Workspace Structure
The workspace root `Cargo.toml` lists all member crates. Each crate has its own `Cargo.toml`.
When adding a new crate:
1. Create `<crate-name>/Cargo.toml` and `<crate-name>/src/lib.rs`
2. Add the crate to the workspace `members` list in the root `Cargo.toml`
3. If it depends on another workspace crate, use `path = "../<dep>"` dependencies

## Git Conventions
- Commit messages: imperative mood, concise subject line
- One logical change per commit
- Always run `cargo test -p <crate>` before committing
