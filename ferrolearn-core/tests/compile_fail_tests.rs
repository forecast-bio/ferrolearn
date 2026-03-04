/// Compile-fail tests for ferrolearn type-system safety guarantees.
///
/// These tests verify that unfitted models cannot call `predict()` or
/// `transform()`. Each `.rs` file in `tests/compile_fail/` is compiled
/// by `trybuild` and must fail to compile. A passing test means the
/// compiler correctly rejected the invalid code.
#[test]
fn compile_fail_tests() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/compile_fail/*.rs");
}
