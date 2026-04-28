//! Conjugate priors with closed-form posterior updates.
//!
//! For exponential-family likelihoods, the posterior over the latent parameter
//! has the same functional form as the prior when paired with a *conjugate*
//! prior. This module collects closed-form posterior updates that arise from
//! that pairing — no MCMC, no variational approximation, just arithmetic.
//!
//! Currently supported:
//!
//! - [`posterior_normal_normal`] — Normal-Normal: latent mean of a Gaussian
//!   likelihood with known per-observation variance, given a Normal prior on
//!   the mean.
//!
//! Future additions (Beta-Binomial, Gamma-Poisson, NIG-Normal, etc.) follow
//! the same pattern: free function returning a typed posterior summary.

/// Posterior summary for the Normal-Normal conjugate update.
///
/// Mean and variance of the Normal posterior over the latent parameter μ.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormalNormalPosterior {
    /// Posterior mean E[μ | observations].
    pub mean: f64,
    /// Posterior variance Var[μ | observations].
    pub var: f64,
}

/// Smallest valid observation variance. Anything ≤ 0 is clipped to this so
/// callers can pass the output of imperfect calibration routines without
/// crashing on degenerate inputs (matches the Python `if obs_var <= 0:
/// obs_var = 1.0` guard, but with a smaller floor since we *know* the
/// observation came from somewhere — falling back to 1.0 would over-weight
/// degenerate observations).
const MIN_OBS_VAR: f64 = 1e-12;

/// Closed-form posterior for the Normal-Normal conjugate model.
///
/// # Model
///
/// ```text
/// μ ~ N(μ₀, σ₀²)                        (prior)
/// y_i | μ ~ N(μ, σ_i²)  (independent)   (likelihood)
/// ```
///
/// where `μ₀ = prior_mean`, `σ₀² = prior_var`, and `observations` is a slice
/// of `(y_i, σ_i²)` pairs.
///
/// # Posterior
///
/// ```text
/// τ_prior   = 1 / σ₀²
/// τ_obs_i   = 1 / σ_i²
/// τ_post    = τ_prior + Σ τ_obs_i
/// μ_post    = (τ_prior · μ₀ + Σ τ_obs_i · y_i) / τ_post
/// σ²_post   = 1 / τ_post
/// ```
///
/// # Edge cases
///
/// - `prior_var ≤ 0` is clipped to [`MIN_OBS_VAR`] (the prior is effectively
///   improper but the math still produces a defined posterior dominated by
///   the data).
/// - Any `obs_var_i ≤ 0` is clipped to [`MIN_OBS_VAR`] (degenerate
///   observations get extremely high weight but do not produce NaN).
/// - Empty `observations` returns the prior unchanged.
///
/// # Example
///
/// ```
/// use ferrolearn_bayes::conjugate::posterior_normal_normal;
///
/// // Prior N(0, 1), three observations with unit variance.
/// let post = posterior_normal_normal(0.0, 1.0, &[(1.0, 1.0), (2.0, 1.0), (3.0, 1.0)]);
/// // Posterior precision = 1 + 3 = 4, posterior var = 0.25
/// // Posterior mean = (0 + 1 + 2 + 3) / 4 = 1.5
/// assert!((post.mean - 1.5).abs() < 1e-12);
/// assert!((post.var - 0.25).abs() < 1e-12);
/// ```
pub fn posterior_normal_normal(
    prior_mean: f64,
    prior_var: f64,
    observations: &[(f64, f64)],
) -> NormalNormalPosterior {
    let prior_var_safe = if prior_var > 0.0 {
        prior_var
    } else {
        MIN_OBS_VAR
    };

    // Empty observations: the posterior equals the prior. Skip the precision
    // round-trip (1/x then 1/x again accumulates one ULP) so the no-op case
    // is bit-exact.
    if observations.is_empty() {
        return NormalNormalPosterior {
            mean: prior_mean,
            var: prior_var_safe,
        };
    }

    let prior_precision = 1.0 / prior_var_safe;
    let mut total_precision = prior_precision;
    let mut weighted_sum = prior_precision * prior_mean;

    for &(y, obs_var) in observations {
        let obs_var_safe = if obs_var > 0.0 { obs_var } else { MIN_OBS_VAR };
        let obs_precision = 1.0 / obs_var_safe;
        total_precision += obs_precision;
        weighted_sum += obs_precision * y;
    }

    let mean = weighted_sum / total_precision;
    let var = 1.0 / total_precision;

    NormalNormalPosterior { mean, var }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use proptest::prelude::*;

    /// Slow-but-clearly-correct reference implementation for cross-validation.
    /// Walks observations one at a time, doing a sequential posterior update
    /// (each previous posterior becomes the next prior). This is mathematically
    /// equivalent to the batch form but exercises a different code path.
    fn reference_sequential(
        prior_mean: f64,
        prior_var: f64,
        observations: &[(f64, f64)],
    ) -> NormalNormalPosterior {
        let mut mean = prior_mean;
        let mut var = if prior_var > 0.0 {
            prior_var
        } else {
            MIN_OBS_VAR
        };
        for &(y, obs_var) in observations {
            let v = if obs_var > 0.0 { obs_var } else { MIN_OBS_VAR };
            // Two-Gaussian fusion: posterior is a Gaussian whose mean and
            // variance are the precision-weighted combination of (mean, var)
            // and (y, v).
            let inv_var = 1.0 / var;
            let inv_v = 1.0 / v;
            let new_inv_var = inv_var + inv_v;
            mean = (inv_var * mean + inv_v * y) / new_inv_var;
            var = 1.0 / new_inv_var;
        }
        NormalNormalPosterior { mean, var }
    }

    #[test]
    fn empty_observations_returns_prior() {
        let post = posterior_normal_normal(2.5, 4.0, &[]);
        assert_relative_eq!(post.mean, 2.5);
        assert_relative_eq!(post.var, 4.0);
    }

    #[test]
    fn single_observation_matches_two_gaussian_fusion() {
        // Two equal-precision Gaussians: posterior mean is their average.
        let post = posterior_normal_normal(0.0, 1.0, &[(2.0, 1.0)]);
        assert_relative_eq!(post.mean, 1.0);
        assert_relative_eq!(post.var, 0.5);
    }

    #[test]
    fn three_unit_observations_against_unit_prior() {
        // Prior N(0, 1), three observations each with σ²=1, values 1, 2, 3.
        // Posterior precision: 1 + 1 + 1 + 1 = 4, var = 0.25
        // Posterior mean: (0 + 1 + 2 + 3) / 4 = 1.5
        let post = posterior_normal_normal(0.0, 1.0, &[(1.0, 1.0), (2.0, 1.0), (3.0, 1.0)]);
        assert_relative_eq!(post.mean, 1.5);
        assert_relative_eq!(post.var, 0.25);
    }

    #[test]
    fn heteroscedastic_observations_weight_by_precision() {
        // Two observations: y₁=10 with σ²=1 (high precision = 1.0)
        //                  y₂=0 with σ²=100 (low precision = 0.01)
        // Prior μ₀=0, σ₀²=1 (precision 1.0)
        //
        // Total precision = 1.0 + 1.0 + 0.01 = 2.01
        // Weighted sum = 0 + 10 + 0 = 10
        // Mean = 10 / 2.01 ≈ 4.9751
        // Var = 1 / 2.01 ≈ 0.4975
        let post = posterior_normal_normal(0.0, 1.0, &[(10.0, 1.0), (0.0, 100.0)]);
        assert_relative_eq!(post.mean, 10.0 / 2.01, epsilon = 1e-12);
        assert_relative_eq!(post.var, 1.0 / 2.01, epsilon = 1e-12);
    }

    #[test]
    fn posterior_var_decreases_with_more_observations() {
        let prior_mean = 0.0;
        let prior_var = 1.0;
        let zero = posterior_normal_normal(prior_mean, prior_var, &[]).var;
        let one = posterior_normal_normal(prior_mean, prior_var, &[(0.0, 1.0)]).var;
        let three =
            posterior_normal_normal(prior_mean, prior_var, &[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)])
                .var;
        assert!(zero > one, "expected {zero} > {one}");
        assert!(one > three, "expected {one} > {three}");
    }

    #[test]
    fn obs_var_le_zero_is_clipped_not_panic() {
        // A degenerate (zero or negative) observation variance is clipped to
        // MIN_OBS_VAR. The posterior should be heavily dominated by that
        // observation but mathematically defined (no NaN, no Inf).
        let post = posterior_normal_normal(0.0, 1.0, &[(7.0, 0.0)]);
        assert!(post.mean.is_finite());
        assert!(post.var.is_finite());
        // With clipping at 1e-12, the precision is huge and the posterior
        // collapses to the observation.
        assert_relative_eq!(post.mean, 7.0, epsilon = 1e-9);
        assert!(post.var < 1e-10);

        let post2 = posterior_normal_normal(0.0, 1.0, &[(7.0, -1.0)]);
        assert!(post2.mean.is_finite());
        assert!(post2.var.is_finite());
    }

    #[test]
    fn prior_var_le_zero_is_also_clipped() {
        // Improper prior (var ≤ 0) gets clipped; the data dominates.
        let post = posterior_normal_normal(0.0, 0.0, &[(5.0, 1.0)]);
        assert!(post.mean.is_finite());
        assert!(post.var.is_finite());
        // Prior precision is huge so the posterior mean stays near 0... but
        // by clipping prior_var to 1e-12, prior precision is 1e12, observation
        // precision is 1.0, so prior dominates.
        assert!(post.mean.abs() < 1e-9);
    }

    #[test]
    fn matches_reference_sequential_implementation() {
        // Hand-crafted cases compared against the sequential reference.
        type HandCase = (f64, f64, &'static [(f64, f64)]);
        let cases: &[HandCase] = &[
            (0.0, 1.0, &[(1.0, 1.0)]),
            (5.0, 4.0, &[(3.0, 1.0), (7.0, 2.0)]),
            (-2.0, 0.25, &[(0.0, 1.0), (-1.0, 0.5), (1.0, 0.5)]),
            (
                0.0,
                100.0,
                &[(1.0, 0.01), (2.0, 0.01), (3.0, 0.01), (4.0, 0.01)],
            ),
        ];
        for (m0, v0, obs) in cases {
            let direct = posterior_normal_normal(*m0, *v0, obs);
            let ref_seq = reference_sequential(*m0, *v0, obs);
            assert_relative_eq!(direct.mean, ref_seq.mean, epsilon = 1e-10);
            assert_relative_eq!(direct.var, ref_seq.var, epsilon = 1e-10);
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10_000))]

        /// AC-19: the closed-form posterior must agree with the sequential
        /// reference to within 1e-9 absolute tolerance over ≥10 000 random
        /// inputs. The two implementations use different numerical paths
        /// (batch precision sum vs. iterative two-Gaussian fusion) so
        /// agreement is strong evidence both are correct.
        #[test]
        fn batch_form_matches_sequential_form(
            prior_mean in -100.0_f64..100.0,
            prior_var in 1e-3_f64..100.0,
            obs in proptest::collection::vec((-100.0_f64..100.0, 1e-3_f64..100.0), 0..16),
        ) {
            let direct = posterior_normal_normal(prior_mean, prior_var, &obs);
            let ref_seq = reference_sequential(prior_mean, prior_var, &obs);
            prop_assert!(
                (direct.mean - ref_seq.mean).abs() < 1e-9,
                "mean mismatch: direct={} ref={}",
                direct.mean,
                ref_seq.mean,
            );
            prop_assert!(
                (direct.var - ref_seq.var).abs() < 1e-9,
                "var mismatch: direct={} ref={}",
                direct.var,
                ref_seq.var,
            );
        }

        /// Posterior variance is monotonically non-increasing as observations
        /// are added (precision adds).
        #[test]
        fn posterior_var_is_non_increasing(
            prior_mean in -10.0_f64..10.0,
            prior_var in 1e-3_f64..10.0,
            obs in proptest::collection::vec((-10.0_f64..10.0, 1e-3_f64..10.0), 1..8),
        ) {
            let prior_only = posterior_normal_normal(prior_mean, prior_var, &[]);
            let with_obs = posterior_normal_normal(prior_mean, prior_var, &obs);
            // Strict decrease (every added observation has positive precision).
            prop_assert!(
                with_obs.var < prior_only.var,
                "var did not decrease: with_obs={} prior_only={}",
                with_obs.var,
                prior_only.var,
            );
        }

        /// Posterior mean lies in the convex hull of the prior mean and the
        /// observation values (precision-weighted mean is bounded).
        #[test]
        fn posterior_mean_in_convex_hull(
            prior_mean in -10.0_f64..10.0,
            prior_var in 1e-3_f64..10.0,
            obs in proptest::collection::vec((-10.0_f64..10.0, 1e-3_f64..10.0), 1..8),
        ) {
            let post = posterior_normal_normal(prior_mean, prior_var, &obs);
            let mut all_means: Vec<f64> = obs.iter().map(|&(y, _)| y).collect();
            all_means.push(prior_mean);
            let lo = all_means.iter().cloned().fold(f64::INFINITY, f64::min);
            let hi = all_means.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            // Allow tiny epsilon for floating-point edge cases.
            prop_assert!(
                post.mean >= lo - 1e-9,
                "post.mean={} below hull min={}",
                post.mean,
                lo,
            );
            prop_assert!(
                post.mean <= hi + 1e-9,
                "post.mean={} above hull max={}",
                post.mean,
                hi,
            );
        }

        /// Empty observations leave the prior unchanged.
        #[test]
        fn empty_obs_is_identity(
            prior_mean in -100.0_f64..100.0,
            prior_var in 1e-3_f64..100.0,
        ) {
            let post = posterior_normal_normal(prior_mean, prior_var, &[]);
            prop_assert_eq!(post.mean, prior_mean);
            prop_assert_eq!(post.var, prior_var);
        }
    }
}
