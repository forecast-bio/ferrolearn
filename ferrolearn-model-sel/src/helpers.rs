//! Construction helpers that mirror scikit-learn's `make_pipeline` and
//! `make_union` shorthand functions.
//!
//! These auto-generate step names of the form `step0`, `step1`, ... so that
//! callers don't have to invent meaningful names when they're not needed.

use ferrolearn_core::pipeline::{Pipeline, PipelineEstimator, PipelineStep, PipelineTransformer};
use num_traits::Float;

use crate::feature_union::FeatureUnion;

/// Build a [`Pipeline`] from a list of pre-boxed `PipelineStep`s and an
/// optional final [`PipelineEstimator`].
///
/// Auto-generates step names of the form `step0`, `step1`, ...
///
/// # Example
///
/// ```rust,no_run
/// use ferrolearn_model_sel::helpers::make_pipeline;
/// use ferrolearn_core::pipeline::Pipeline;
///
/// // In real code, `steps` would be filled with concrete transformers and
/// // `estimator` with the final fit step.
/// let _pipe: Pipeline<f64> = make_pipeline::<f64>(Vec::new(), None);
/// ```
#[must_use]
pub fn make_pipeline<F>(
    steps: Vec<Box<dyn PipelineStep<F>>>,
    estimator: Option<Box<dyn PipelineEstimator<F>>>,
) -> Pipeline<F>
where
    F: Float + Send + Sync + 'static,
{
    let mut pipe = Pipeline::<F>::new();
    for (i, step) in steps.into_iter().enumerate() {
        let name = format!("step{i}");
        pipe = pipe.step(&name, step);
    }
    if let Some(est) = estimator {
        pipe = pipe.estimator_step("estimator", est);
    }
    pipe
}

/// Build a [`FeatureUnion`] from a list of pre-boxed transformers.
///
/// Auto-generates names of the form `fu0`, `fu1`, ...
#[must_use]
pub fn make_union<F>(transformers: Vec<Box<dyn PipelineTransformer<F>>>) -> FeatureUnion<F>
where
    F: Float + Send + Sync + 'static,
{
    let mut fu = FeatureUnion::<F>::new();
    for (i, t) in transformers.into_iter().enumerate() {
        let name = format!("fu{i}");
        fu = fu.add(&name, t);
    }
    fu
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn make_pipeline_empty_runs() {
        // Just confirm the empty case constructs without panic.
        let _pipe: Pipeline<f64> = make_pipeline::<f64>(Vec::new(), None);
    }

    #[test]
    fn make_union_empty() {
        let fu: FeatureUnion<f64> = make_union::<f64>(Vec::new());
        assert_eq!(fu.n_transformers(), 0);
    }
}
