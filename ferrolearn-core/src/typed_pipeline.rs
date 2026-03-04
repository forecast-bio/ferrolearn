//! Compile-time type-safe pipeline using recursive tuple types.
//!
//! This module provides a zero-cost, type-safe pipeline that complements the
//! dynamic-dispatch [`Pipeline`](crate::pipeline::Pipeline). Unlike the dynamic
//! pipeline which constrains all intermediate data to `Array2<f64>`, this typed
//! pipeline verifies type compatibility between steps at compile time.
//!
//! # Design
//!
//! Pipeline steps are represented as nested structs:
//! `TypedPipelineStep<Step1, TypedPipelineStep<Step2, PipelineEnd>>`.
//! The compiler verifies that each step's output type matches the next step's
//! input type. If they don't match, you get a compile-time error instead of a
//! runtime surprise.
//!
//! # Examples
//!
//! ```ignore
//! use ferrolearn_core::typed_pipeline::TypedPipeline;
//!
//! // Build a pipeline with method chaining:
//! let pipeline = TypedPipeline::new()
//!     .then(StandardScaler::<f64>::new())  // transformer step
//!     .then(PCA::<f64>::new(5))            // another transformer
//!     .finish(LogisticRegression::<f64>::new()); // final estimator
//! ```
//!
//! # Type Safety Guarantee
//!
//! If step N outputs `Array2<f64>` but step N+1 expects `Array2<f32>`, the
//! compiler rejects the pipeline at build time. No runtime checks needed.

use crate::error::FerroError;
use crate::traits::{Fit, Predict, Transform};

// ---------------------------------------------------------------------------
// Core traits for typed pipeline steps
// ---------------------------------------------------------------------------

/// A step that can be fitted on input data and then transforms it.
///
/// This trait is the typed pipeline equivalent of a transformer. The key
/// difference from [`Transform`] is that it carries the output type as an
/// associated type, enabling the compiler to chain steps and verify type
/// compatibility.
///
/// # Type Parameters
///
/// - `Input`: The data type this step accepts for fitting and transforming.
pub trait TypedTransformStep<Input> {
    /// The data type produced after transformation.
    type Output;
    /// The fitted version of this step.
    type FittedStep: TypedFittedTransformStep<Input, Output = Self::Output>;

    /// Fit this step on the given input data, producing a fitted step.
    ///
    /// # Errors
    ///
    /// Returns a [`FerroError`] if the data is invalid or fitting fails.
    fn fit_step(&self, x: &Input) -> Result<Self::FittedStep, FerroError>;
}

/// A fitted transform step that can transform data.
///
/// This is the result of calling [`TypedTransformStep::fit_step`]. It holds
/// learned parameters and can transform new data.
///
/// # Type Parameters
///
/// - `Input`: The data type this fitted step accepts.
pub trait TypedFittedTransformStep<Input> {
    /// The data type produced after transformation.
    type Output;

    /// Transform the input data using the learned parameters.
    ///
    /// # Errors
    ///
    /// Returns a [`FerroError`] if the input has an incompatible shape.
    fn transform_step(&self, x: &Input) -> Result<Self::Output, FerroError>;
}

/// A step that can be fitted with input data and targets, then predicts.
///
/// This trait is the typed pipeline equivalent of a supervised estimator.
/// It serves as the final step of a pipeline.
///
/// # Type Parameters
///
/// - `Input`: The feature data type.
/// - `Target`: The target/label data type.
pub trait TypedEstimatorStep<Input, Target> {
    /// The fitted version of this estimator step.
    type FittedStep: TypedFittedEstimatorStep<Input>;

    /// Fit this estimator on the given input data and targets.
    ///
    /// # Errors
    ///
    /// Returns a [`FerroError`] if the data is invalid or fitting fails.
    fn fit_step(&self, x: &Input, y: &Target) -> Result<Self::FittedStep, FerroError>;
}

/// A fitted estimator step that can generate predictions.
///
/// This is the result of calling [`TypedEstimatorStep::fit_step`]. It holds
/// learned model parameters and can predict on new data.
///
/// # Type Parameters
///
/// - `Input`: The feature data type this fitted estimator accepts.
pub trait TypedFittedEstimatorStep<Input> {
    /// The prediction output type.
    type Output;

    /// Generate predictions for the given input data.
    ///
    /// # Errors
    ///
    /// Returns a [`FerroError`] if the input has an incompatible shape.
    fn predict_step(&self, x: &Input) -> Result<Self::Output, FerroError>;
}

// ---------------------------------------------------------------------------
// Blanket implementations: bridge existing Fit/Transform/Predict to typed traits
// ---------------------------------------------------------------------------

/// Wrapper that adapts a type implementing [`Fit<X, ()>`] + (fitted implements
/// [`Transform<X>`]) into a [`TypedTransformStep`].
///
/// This allows existing ferrolearn transformers (e.g., `StandardScaler`) to
/// participate in typed pipelines without modification.
pub struct TransformAdapter<T>(T);

impl<T> TransformAdapter<T> {
    /// Wrap an existing transformer for use in a typed pipeline.
    ///
    /// The wrapped type must implement `Fit<Input, ()>` where the fitted
    /// type implements `Transform<Input>`.
    pub fn new(inner: T) -> Self {
        Self(inner)
    }
}

impl<T, Input> TypedTransformStep<Input> for TransformAdapter<T>
where
    T: Fit<Input, (), Error = FerroError>,
    T::Fitted: Transform<Input, Error = FerroError>,
{
    type Output = <T::Fitted as Transform<Input>>::Output;
    type FittedStep = FittedTransformAdapter<T::Fitted>;

    fn fit_step(&self, x: &Input) -> Result<Self::FittedStep, FerroError> {
        let fitted = self.0.fit(x, &())?;
        Ok(FittedTransformAdapter(fitted))
    }
}

/// Fitted version of [`TransformAdapter`], wrapping an existing fitted
/// transformer.
pub struct FittedTransformAdapter<F>(F);

impl<F, Input> TypedFittedTransformStep<Input> for FittedTransformAdapter<F>
where
    F: Transform<Input, Error = FerroError>,
{
    type Output = F::Output;

    fn transform_step(&self, x: &Input) -> Result<Self::Output, FerroError> {
        self.0.transform(x)
    }
}

/// Wrapper that adapts a type implementing [`Fit<X, Y>`] + (fitted implements
/// [`Predict<X>`]) into a [`TypedEstimatorStep`].
///
/// This allows existing ferrolearn estimators (e.g., `LogisticRegression`) to
/// participate in typed pipelines without modification.
pub struct EstimatorAdapter<T>(T);

impl<T> EstimatorAdapter<T> {
    /// Wrap an existing estimator for use in a typed pipeline.
    ///
    /// The wrapped type must implement `Fit<Input, Target>` where the fitted
    /// type implements `Predict<Input>`.
    pub fn new(inner: T) -> Self {
        Self(inner)
    }
}

impl<T, Input, Target> TypedEstimatorStep<Input, Target> for EstimatorAdapter<T>
where
    T: Fit<Input, Target, Error = FerroError>,
    T::Fitted: Predict<Input, Error = FerroError>,
{
    type FittedStep = FittedEstimatorAdapter<T::Fitted>;

    fn fit_step(&self, x: &Input, y: &Target) -> Result<Self::FittedStep, FerroError> {
        let fitted = self.0.fit(x, y)?;
        Ok(FittedEstimatorAdapter(fitted))
    }
}

/// Fitted version of [`EstimatorAdapter`], wrapping an existing fitted
/// estimator.
pub struct FittedEstimatorAdapter<F>(F);

impl<F, Input> TypedFittedEstimatorStep<Input> for FittedEstimatorAdapter<F>
where
    F: Predict<Input, Error = FerroError>,
{
    type Output = F::Output;

    fn predict_step(&self, x: &Input) -> Result<Self::Output, FerroError> {
        self.0.predict(x)
    }
}

// ---------------------------------------------------------------------------
// Pipeline structure: recursive nested types
// ---------------------------------------------------------------------------

/// Marker for the end of a typed pipeline's transformer chain.
///
/// This is the "nil" element of the recursive type-level list. A pipeline
/// with no transformer steps has type `PipelineEnd`.
pub struct PipelineEnd;

/// A single step in a typed pipeline, forming a recursive chain.
///
/// The pipeline structure is a nested type:
/// ```text
/// TypedPipelineStep<Step1, TypedPipelineStep<Step2, PipelineEnd>>
/// ```
///
/// Steps are stored in reverse order (last-added is outermost) and
/// the recursive fit/transform implementations process them correctly.
pub struct TypedPipelineStep<Step, Rest> {
    /// The transformer step at this position.
    step: Step,
    /// The remaining steps (or [`PipelineEnd`]).
    rest: Rest,
}

// ---------------------------------------------------------------------------
// Recursive fitting for the transformer chain
// ---------------------------------------------------------------------------

/// Internal trait for recursively fitting a chain of transformer steps.
///
/// This trait is implemented by [`PipelineEnd`] (base case) and
/// [`TypedPipelineStep`] (recursive case). It is not intended for
/// direct use by library consumers.
pub trait FitTransformChain<Input> {
    /// The type of data produced after all transformations.
    type ChainOutput;
    /// The fitted version of this chain.
    type FittedChain: TransformChain<Input, ChainOutput = Self::ChainOutput>;

    /// Fit all steps in the chain sequentially, passing transformed data
    /// from each step to the next.
    ///
    /// # Errors
    ///
    /// Returns a [`FerroError`] if any step fails to fit or transform.
    fn fit_chain(&self, x: &Input) -> Result<(Self::FittedChain, Self::ChainOutput), FerroError>;
}

/// Internal trait for recursively transforming data through a fitted chain.
///
/// This trait is implemented by the fitted counterparts of [`PipelineEnd`]
/// and [`TypedPipelineStep`].
pub trait TransformChain<Input> {
    /// The type of data produced after all transformations.
    type ChainOutput;

    /// Transform data through all fitted steps in sequence.
    ///
    /// # Errors
    ///
    /// Returns a [`FerroError`] if any step fails to transform.
    fn transform_chain(&self, x: &Input) -> Result<Self::ChainOutput, FerroError>;
}

// Base case: PipelineEnd is the identity transformation.
impl<Input: Clone> FitTransformChain<Input> for PipelineEnd {
    type ChainOutput = Input;
    type FittedChain = FittedPipelineEnd;

    fn fit_chain(&self, x: &Input) -> Result<(Self::FittedChain, Self::ChainOutput), FerroError> {
        Ok((FittedPipelineEnd, x.clone()))
    }
}

/// Fitted version of [`PipelineEnd`] — the identity fitted chain.
pub struct FittedPipelineEnd;

impl<Input: Clone> TransformChain<Input> for FittedPipelineEnd {
    type ChainOutput = Input;

    fn transform_chain(&self, x: &Input) -> Result<Self::ChainOutput, FerroError> {
        Ok(x.clone())
    }
}

// Recursive case: fit the rest of the chain first, then fit this step on
// the transformed output.
impl<Step, Rest, Input> FitTransformChain<Input> for TypedPipelineStep<Step, Rest>
where
    Rest: FitTransformChain<Input>,
    Step: TypedTransformStep<Rest::ChainOutput>,
    Step::Output: Clone,
{
    type ChainOutput = Step::Output;
    type FittedChain = FittedTypedPipelineStep<Step::FittedStep, Rest::FittedChain>;

    fn fit_chain(&self, x: &Input) -> Result<(Self::FittedChain, Self::ChainOutput), FerroError> {
        // First, fit everything before this step.
        let (fitted_rest, intermediate) = self.rest.fit_chain(x)?;
        // Then fit this step on the intermediate output.
        let fitted_step = self.step.fit_step(&intermediate)?;
        // Transform the data through this step.
        let output = fitted_step.transform_step(&intermediate)?;
        Ok((
            FittedTypedPipelineStep {
                step: fitted_step,
                rest: fitted_rest,
            },
            output,
        ))
    }
}

/// A fitted step in the typed pipeline chain.
///
/// Holds the fitted step and the fitted rest of the chain.
pub struct FittedTypedPipelineStep<FittedStep, FittedRest> {
    /// The fitted transformer step.
    step: FittedStep,
    /// The fitted rest of the chain.
    rest: FittedRest,
}

impl<FittedStep, FittedRest, Input> TransformChain<Input>
    for FittedTypedPipelineStep<FittedStep, FittedRest>
where
    FittedRest: TransformChain<Input>,
    FittedStep: TypedFittedTransformStep<FittedRest::ChainOutput>,
{
    type ChainOutput = FittedStep::Output;

    fn transform_chain(&self, x: &Input) -> Result<Self::ChainOutput, FerroError> {
        let intermediate = self.rest.transform_chain(x)?;
        self.step.transform_step(&intermediate)
    }
}

// ---------------------------------------------------------------------------
// Complete pipeline: transformer chain + final estimator
// ---------------------------------------------------------------------------

/// A complete unfitted typed pipeline with a transformer chain and a final
/// estimator.
///
/// Created via the builder API: [`TypedPipeline::new()`] returns a
/// [`TypedPipelineBuilder`], which is finalized with
/// [`finish`](TypedPipelineBuilder::finish).
///
/// # Type Parameters
///
/// - `Chain`: The recursive transformer chain type.
/// - `Est`: The final estimator step type.
pub struct CompletePipeline<Chain, Est> {
    /// The transformer chain (possibly [`PipelineEnd`] if no transformers).
    chain: Chain,
    /// The final estimator step.
    estimator: Est,
}

/// A fitted typed pipeline that can generate predictions.
///
/// Created by calling `fit` on a [`CompletePipeline`].
///
/// # Type Parameters
///
/// - `FittedChain`: The fitted transformer chain type.
/// - `FittedEst`: The fitted estimator type.
pub struct FittedCompletePipeline<FittedChain, FittedEst> {
    /// The fitted transformer chain.
    chain: FittedChain,
    /// The fitted estimator.
    estimator: FittedEst,
}

impl<Chain, Est, Input, Target> Fit<Input, Target> for CompletePipeline<Chain, Est>
where
    Chain: FitTransformChain<Input>,
    Est: TypedEstimatorStep<Chain::ChainOutput, Target>,
{
    type Fitted = FittedCompletePipeline<Chain::FittedChain, Est::FittedStep>;
    type Error = FerroError;

    /// Fit the typed pipeline by fitting each transformer step in order,
    /// then fitting the final estimator on the transformed data.
    ///
    /// # Errors
    ///
    /// Returns a [`FerroError`] if any step fails to fit or transform.
    fn fit(&self, x: &Input, y: &Target) -> Result<Self::Fitted, FerroError> {
        let (fitted_chain, transformed) = self.chain.fit_chain(x)?;
        let fitted_est = self.estimator.fit_step(&transformed, y)?;
        Ok(FittedCompletePipeline {
            chain: fitted_chain,
            estimator: fitted_est,
        })
    }
}

impl<FittedChain, FittedEst, Input> Predict<Input>
    for FittedCompletePipeline<FittedChain, FittedEst>
where
    FittedChain: TransformChain<Input>,
    FittedEst: TypedFittedEstimatorStep<FittedChain::ChainOutput>,
{
    type Output = FittedEst::Output;
    type Error = FerroError;

    /// Generate predictions by transforming the input through all fitted
    /// transformer steps, then calling predict on the fitted estimator.
    ///
    /// # Errors
    ///
    /// Returns a [`FerroError`] if any step fails to transform or predict.
    fn predict(&self, x: &Input) -> Result<Self::Output, FerroError> {
        let transformed = self.chain.transform_chain(x)?;
        self.estimator.predict_step(&transformed)
    }
}

// ---------------------------------------------------------------------------
// Transform-only pipeline: a chain of transformers without a final estimator
// ---------------------------------------------------------------------------

/// A complete unfitted typed pipeline consisting only of transformer steps
/// (no final estimator).
///
/// Created via the builder API by calling
/// [`finish_transform`](TypedPipelineBuilder::finish_transform) instead of
/// [`finish`](TypedPipelineBuilder::finish).
///
/// This pipeline implements [`Fit<Input, ()>`] and the fitted version
/// implements [`Transform<Input>`].
///
/// # Type Parameters
///
/// - `Chain`: The recursive transformer chain type.
pub struct TransformOnlyPipeline<Chain> {
    /// The transformer chain.
    chain: Chain,
}

/// A fitted transform-only typed pipeline that can transform data.
///
/// Created by calling `fit` on a [`TransformOnlyPipeline`].
pub struct FittedTransformOnlyPipeline<FittedChain> {
    /// The fitted transformer chain.
    chain: FittedChain,
}

impl<Chain, Input> Fit<Input, ()> for TransformOnlyPipeline<Chain>
where
    Chain: FitTransformChain<Input>,
{
    type Fitted = FittedTransformOnlyPipeline<Chain::FittedChain>;
    type Error = FerroError;

    /// Fit the transform-only pipeline by fitting each step in order.
    ///
    /// # Errors
    ///
    /// Returns a [`FerroError`] if any step fails to fit.
    fn fit(&self, x: &Input, _y: &()) -> Result<Self::Fitted, FerroError> {
        let (fitted_chain, _) = self.chain.fit_chain(x)?;
        Ok(FittedTransformOnlyPipeline {
            chain: fitted_chain,
        })
    }
}

impl<FittedChain, Input> Transform<Input> for FittedTransformOnlyPipeline<FittedChain>
where
    FittedChain: TransformChain<Input>,
{
    type Output = FittedChain::ChainOutput;
    type Error = FerroError;

    /// Transform data through all fitted steps in sequence.
    ///
    /// # Errors
    ///
    /// Returns a [`FerroError`] if any step fails to transform.
    fn transform(&self, x: &Input) -> Result<Self::Output, FerroError> {
        self.chain.transform_chain(x)
    }
}

// ---------------------------------------------------------------------------
// Builder API
// ---------------------------------------------------------------------------

/// Entry point for building a typed pipeline.
///
/// Call [`TypedPipeline::new()`] to start, then chain `.then(step)` calls
/// for transformer steps, and finalize with `.finish(estimator)` to set
/// the final estimator.
///
/// # Examples
///
/// ```ignore
/// use ferrolearn_core::typed_pipeline::TypedPipeline;
///
/// let pipeline = TypedPipeline::new()
///     .then(StandardScaler::<f64>::new())
///     .then(PCA::<f64>::new(5))
///     .finish(LogisticRegression::<f64>::new());
/// ```
pub struct TypedPipeline;

impl TypedPipeline {
    /// Create a new typed pipeline builder with no steps.
    ///
    /// # Examples
    ///
    /// ```
    /// use ferrolearn_core::typed_pipeline::TypedPipeline;
    /// let builder = TypedPipeline::new();
    /// ```
    #[allow(clippy::new_ret_no_self)]
    pub fn new() -> TypedPipelineBuilder<PipelineEnd> {
        TypedPipelineBuilder { steps: PipelineEnd }
    }
}

/// Builder for constructing a typed pipeline step by step.
///
/// Add transformer steps with [`then`](TypedPipelineBuilder::then) and
/// finalize with [`finish`](TypedPipelineBuilder::finish) (for estimator
/// pipelines) or [`finish_transform`](TypedPipelineBuilder::finish_transform)
/// (for transform-only pipelines).
///
/// # Type Parameters
///
/// - `Steps`: The current recursive chain of transformer steps.
pub struct TypedPipelineBuilder<Steps> {
    /// The accumulated transformer steps.
    steps: Steps,
}

impl<Steps> TypedPipelineBuilder<Steps> {
    /// Add a transformer step to the pipeline.
    ///
    /// Steps are applied in the order they are added. The compiler verifies
    /// that this step's input type matches the previous step's output type.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use ferrolearn_core::typed_pipeline::TypedPipeline;
    ///
    /// let builder = TypedPipeline::new()
    ///     .then(my_scaler);
    /// ```
    pub fn then<S>(self, step: S) -> TypedPipelineBuilder<TypedPipelineStep<S, Steps>> {
        TypedPipelineBuilder {
            steps: TypedPipelineStep {
                step,
                rest: self.steps,
            },
        }
    }

    /// Finalize the pipeline with a final estimator step.
    ///
    /// The resulting [`CompletePipeline`] implements [`Fit`] and the fitted
    /// version implements [`Predict`].
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use ferrolearn_core::typed_pipeline::TypedPipeline;
    ///
    /// let pipeline = TypedPipeline::new()
    ///     .then(my_scaler)
    ///     .finish(my_classifier);
    /// ```
    pub fn finish<Est>(self, estimator: Est) -> CompletePipeline<Steps, Est> {
        CompletePipeline {
            chain: self.steps,
            estimator,
        }
    }

    /// Finalize the pipeline as a transform-only pipeline (no estimator).
    ///
    /// The resulting [`TransformOnlyPipeline`] implements [`Fit<Input, ()>`]
    /// and the fitted version implements [`Transform<Input>`].
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use ferrolearn_core::typed_pipeline::TypedPipeline;
    ///
    /// let pipeline = TypedPipeline::new()
    ///     .then(my_scaler)
    ///     .then(my_pca)
    ///     .finish_transform();
    /// ```
    pub fn finish_transform(self) -> TransformOnlyPipeline<Steps> {
        TransformOnlyPipeline { chain: self.steps }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Test fixtures: simple transformers and estimators --------------------

    /// A transformer that doubles all values in a `Vec<f64>`.
    struct Doubler;

    /// Fitted version of [`Doubler`].
    struct FittedDoubler;

    impl TypedTransformStep<Vec<f64>> for Doubler {
        type Output = Vec<f64>;
        type FittedStep = FittedDoubler;

        fn fit_step(&self, _x: &Vec<f64>) -> Result<FittedDoubler, FerroError> {
            Ok(FittedDoubler)
        }
    }

    impl TypedFittedTransformStep<Vec<f64>> for FittedDoubler {
        type Output = Vec<f64>;

        fn transform_step(&self, x: &Vec<f64>) -> Result<Vec<f64>, FerroError> {
            Ok(x.iter().map(|v| v * 2.0).collect())
        }
    }

    /// A transformer that adds a constant offset to all values.
    struct Adder {
        offset: f64,
    }

    /// Fitted version of [`Adder`].
    struct FittedAdder {
        offset: f64,
    }

    impl TypedTransformStep<Vec<f64>> for Adder {
        type Output = Vec<f64>;
        type FittedStep = FittedAdder;

        fn fit_step(&self, _x: &Vec<f64>) -> Result<FittedAdder, FerroError> {
            Ok(FittedAdder {
                offset: self.offset,
            })
        }
    }

    impl TypedFittedTransformStep<Vec<f64>> for FittedAdder {
        type Output = Vec<f64>;

        fn transform_step(&self, x: &Vec<f64>) -> Result<Vec<f64>, FerroError> {
            Ok(x.iter().map(|v| v + self.offset).collect())
        }
    }

    /// A transformer that negates all values.
    struct Negator;

    /// Fitted version of [`Negator`].
    struct FittedNegator;

    impl TypedTransformStep<Vec<f64>> for Negator {
        type Output = Vec<f64>;
        type FittedStep = FittedNegator;

        fn fit_step(&self, _x: &Vec<f64>) -> Result<FittedNegator, FerroError> {
            Ok(FittedNegator)
        }
    }

    impl TypedFittedTransformStep<Vec<f64>> for FittedNegator {
        type Output = Vec<f64>;

        fn transform_step(&self, x: &Vec<f64>) -> Result<Vec<f64>, FerroError> {
            Ok(x.iter().map(|v| -v).collect())
        }
    }

    /// A simple estimator that sums all values in the input.
    struct SumEstimator;

    /// Fitted version of [`SumEstimator`].
    struct FittedSumEstimator;

    impl TypedEstimatorStep<Vec<f64>, Vec<f64>> for SumEstimator {
        type FittedStep = FittedSumEstimator;

        fn fit_step(&self, _x: &Vec<f64>, _y: &Vec<f64>) -> Result<FittedSumEstimator, FerroError> {
            Ok(FittedSumEstimator)
        }
    }

    impl TypedFittedEstimatorStep<Vec<f64>> for FittedSumEstimator {
        type Output = f64;

        fn predict_step(&self, x: &Vec<f64>) -> Result<f64, FerroError> {
            Ok(x.iter().sum())
        }
    }

    /// A transformer that squares each element.
    struct Squarer;

    /// Fitted version of [`Squarer`].
    struct FittedSquarer;

    impl TypedTransformStep<Vec<f64>> for Squarer {
        type Output = Vec<f64>;
        type FittedStep = FittedSquarer;

        fn fit_step(&self, _x: &Vec<f64>) -> Result<FittedSquarer, FerroError> {
            Ok(FittedSquarer)
        }
    }

    impl TypedFittedTransformStep<Vec<f64>> for FittedSquarer {
        type Output = Vec<f64>;

        fn transform_step(&self, x: &Vec<f64>) -> Result<Vec<f64>, FerroError> {
            Ok(x.iter().map(|v| v * v).collect())
        }
    }

    /// A learning transformer that subtracts the mean (computed during fit).
    struct MeanCenterer;

    /// Fitted version of [`MeanCenterer`] that stores the learned mean.
    struct FittedMeanCenterer {
        mean: f64,
    }

    impl TypedTransformStep<Vec<f64>> for MeanCenterer {
        type Output = Vec<f64>;
        type FittedStep = FittedMeanCenterer;

        fn fit_step(&self, x: &Vec<f64>) -> Result<FittedMeanCenterer, FerroError> {
            if x.is_empty() {
                return Err(FerroError::InsufficientSamples {
                    required: 1,
                    actual: 0,
                    context: "MeanCenterer::fit_step".into(),
                });
            }
            let mean = x.iter().sum::<f64>() / x.len() as f64;
            Ok(FittedMeanCenterer { mean })
        }
    }

    impl TypedFittedTransformStep<Vec<f64>> for FittedMeanCenterer {
        type Output = Vec<f64>;

        fn transform_step(&self, x: &Vec<f64>) -> Result<Vec<f64>, FerroError> {
            Ok(x.iter().map(|v| v - self.mean).collect())
        }
    }

    /// An estimator that returns the mean of the input as the prediction.
    struct MeanPredictor;

    /// Fitted version of [`MeanPredictor`].
    struct FittedMeanPredictor;

    impl TypedEstimatorStep<Vec<f64>, Vec<f64>> for MeanPredictor {
        type FittedStep = FittedMeanPredictor;

        fn fit_step(
            &self,
            _x: &Vec<f64>,
            _y: &Vec<f64>,
        ) -> Result<FittedMeanPredictor, FerroError> {
            Ok(FittedMeanPredictor)
        }
    }

    impl TypedFittedEstimatorStep<Vec<f64>> for FittedMeanPredictor {
        type Output = f64;

        fn predict_step(&self, x: &Vec<f64>) -> Result<f64, FerroError> {
            if x.is_empty() {
                return Err(FerroError::InsufficientSamples {
                    required: 1,
                    actual: 0,
                    context: "MeanPredictor::predict_step".into(),
                });
            }
            Ok(x.iter().sum::<f64>() / x.len() as f64)
        }
    }

    /// A transformer that always fails during fit.
    struct FailingTransformer;

    impl TypedTransformStep<Vec<f64>> for FailingTransformer {
        type Output = Vec<f64>;
        type FittedStep = FittedDoubler; // never actually reached

        fn fit_step(&self, _x: &Vec<f64>) -> Result<FittedDoubler, FerroError> {
            Err(FerroError::NumericalInstability {
                message: "deliberate test failure".into(),
            })
        }
    }

    /// A transformer that changes the type from `Vec<f64>` to `Vec<i64>`.
    struct FloatToInt;

    /// Fitted version of [`FloatToInt`].
    struct FittedFloatToInt;

    impl TypedTransformStep<Vec<f64>> for FloatToInt {
        type Output = Vec<i64>;
        type FittedStep = FittedFloatToInt;

        fn fit_step(&self, _x: &Vec<f64>) -> Result<FittedFloatToInt, FerroError> {
            Ok(FittedFloatToInt)
        }
    }

    impl TypedFittedTransformStep<Vec<f64>> for FittedFloatToInt {
        type Output = Vec<i64>;

        fn transform_step(&self, x: &Vec<f64>) -> Result<Vec<i64>, FerroError> {
            Ok(x.iter().map(|v| *v as i64).collect())
        }
    }

    /// An estimator that accepts `Vec<i64>` input (to test type transitions).
    struct IntSumEstimator;

    /// Fitted version of [`IntSumEstimator`].
    struct FittedIntSumEstimator;

    impl TypedEstimatorStep<Vec<i64>, Vec<f64>> for IntSumEstimator {
        type FittedStep = FittedIntSumEstimator;

        fn fit_step(
            &self,
            _x: &Vec<i64>,
            _y: &Vec<f64>,
        ) -> Result<FittedIntSumEstimator, FerroError> {
            Ok(FittedIntSumEstimator)
        }
    }

    impl TypedFittedEstimatorStep<Vec<i64>> for FittedIntSumEstimator {
        type Output = i64;

        fn predict_step(&self, x: &Vec<i64>) -> Result<i64, FerroError> {
            Ok(x.iter().sum())
        }
    }

    // -- Tests ---------------------------------------------------------------

    #[test]
    fn test_single_transformer_with_estimator() {
        let pipeline = TypedPipeline::new().then(Doubler).finish(SumEstimator);

        let x = vec![1.0, 2.0, 3.0];
        let y = vec![0.0];

        let fitted = pipeline.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();

        // Doubled: [2.0, 4.0, 6.0], sum = 12.0
        assert!((pred - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_two_transformers_with_estimator() {
        let pipeline = TypedPipeline::new()
            .then(Doubler)
            .then(Adder { offset: 10.0 })
            .finish(SumEstimator);

        let x = vec![1.0, 2.0, 3.0];
        let y = vec![0.0];

        let fitted = pipeline.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();

        // Doubled: [2.0, 4.0, 6.0], +10: [12.0, 14.0, 16.0], sum = 42.0
        assert!((pred - 42.0).abs() < 1e-10);
    }

    #[test]
    fn test_three_transformers_with_estimator() {
        let pipeline = TypedPipeline::new()
            .then(Doubler)
            .then(Adder { offset: 1.0 })
            .then(Negator)
            .finish(SumEstimator);

        let x = vec![1.0, 2.0, 3.0];
        let y = vec![0.0];

        let fitted = pipeline.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();

        // Doubled: [2.0, 4.0, 6.0], +1: [3.0, 5.0, 7.0], negate: [-3.0, -5.0, -7.0]
        // sum = -15.0
        assert!((pred - (-15.0)).abs() < 1e-10);
    }

    #[test]
    fn test_estimator_only() {
        let pipeline = TypedPipeline::new().finish(SumEstimator);

        let x = vec![1.0, 2.0, 3.0];
        let y = vec![0.0];

        let fitted = pipeline.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();

        // No transformers, just sum = 6.0
        assert!((pred - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_transform_only_pipeline() {
        let pipeline = TypedPipeline::new()
            .then(Doubler)
            .then(Adder { offset: 5.0 })
            .finish_transform();

        let x = vec![1.0, 2.0, 3.0];

        let fitted = pipeline.fit(&x, &()).unwrap();
        let result = fitted.transform(&x).unwrap();

        // Doubled: [2.0, 4.0, 6.0], +5: [7.0, 9.0, 11.0]
        assert!((result[0] - 7.0).abs() < 1e-10);
        assert!((result[1] - 9.0).abs() < 1e-10);
        assert!((result[2] - 11.0).abs() < 1e-10);
    }

    #[test]
    fn test_learning_transformer() {
        let pipeline = TypedPipeline::new().then(MeanCenterer).finish(SumEstimator);

        let x = vec![10.0, 20.0, 30.0]; // mean = 20.0
        let y = vec![0.0];

        let fitted = pipeline.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();

        // Centered: [-10.0, 0.0, 10.0], sum = 0.0
        assert!(pred.abs() < 1e-10);
    }

    #[test]
    fn test_learning_transformer_uses_fit_data() {
        // Verify the fitted MeanCenterer uses the training mean, not the
        // predict-time mean.
        let pipeline = TypedPipeline::new().then(MeanCenterer).finish(SumEstimator);

        let x_train = vec![10.0, 20.0, 30.0]; // mean = 20.0
        let y_train = vec![0.0];

        let fitted = pipeline.fit(&x_train, &y_train).unwrap();

        // Predict on different data — should use training mean of 20.0
        let x_test = vec![25.0, 35.0]; // centered: [5.0, 15.0]
        let pred = fitted.predict(&x_test).unwrap();

        assert!((pred - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_transformer_fit_error_propagates() {
        let pipeline = TypedPipeline::new()
            .then(FailingTransformer)
            .finish(SumEstimator);

        let x = vec![1.0, 2.0, 3.0];
        let y = vec![0.0];

        let result = pipeline.fit(&x, &y);
        match result {
            Ok(_) => panic!("expected error, got Ok"),
            Err(FerroError::NumericalInstability { message }) => {
                assert_eq!(message, "deliberate test failure");
            }
            Err(other) => panic!("expected NumericalInstability, got: {other}"),
        }
    }

    #[test]
    fn test_error_in_second_step() {
        // First step succeeds, second step fails.
        let pipeline = TypedPipeline::new()
            .then(Doubler)
            .then(FailingTransformer)
            .finish(SumEstimator);

        let x = vec![1.0, 2.0, 3.0];
        let y = vec![0.0];

        let result = pipeline.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_type_changing_pipeline() {
        // Pipeline that changes types: Vec<f64> -> Vec<i64>
        let pipeline = TypedPipeline::new()
            .then(Doubler)
            .then(FloatToInt)
            .finish(IntSumEstimator);

        let x = vec![1.0, 2.0, 3.0];
        let y = vec![0.0];

        let fitted = pipeline.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();

        // Doubled: [2.0, 4.0, 6.0], to_int: [2, 4, 6], sum = 12
        assert_eq!(pred, 12);
    }

    #[test]
    fn test_squarer_transformer() {
        let pipeline = TypedPipeline::new().then(Squarer).finish(SumEstimator);

        let x = vec![1.0, 2.0, 3.0];
        let y = vec![0.0];

        let fitted = pipeline.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();

        // Squared: [1.0, 4.0, 9.0], sum = 14.0
        assert!((pred - 14.0).abs() < 1e-10);
    }

    #[test]
    fn test_pipeline_with_different_estimator() {
        let pipeline = TypedPipeline::new().then(Doubler).finish(MeanPredictor);

        let x = vec![1.0, 2.0, 3.0];
        let y = vec![0.0];

        let fitted = pipeline.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();

        // Doubled: [2.0, 4.0, 6.0], mean = 4.0
        assert!((pred - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_empty_input() {
        let pipeline = TypedPipeline::new().then(MeanCenterer).finish(SumEstimator);

        let x: Vec<f64> = vec![];
        let y = vec![0.0];

        // MeanCenterer should fail on empty input.
        let result = pipeline.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_predict_error_propagates() {
        let pipeline = TypedPipeline::new().finish(MeanPredictor);

        let x_train = vec![1.0, 2.0];
        let y_train = vec![0.0];

        let fitted = pipeline.fit(&x_train, &y_train).unwrap();

        // MeanPredictor fails on empty input.
        let x_test: Vec<f64> = vec![];
        let result = fitted.predict(&x_test);
        assert!(result.is_err());
    }

    #[test]
    fn test_transform_adapter_bridges_existing_types() {
        // Create a type that uses the existing Fit/Transform traits.
        struct OldStyleScaler;
        struct FittedOldStyleScaler {
            factor: f64,
        }

        impl Fit<Vec<f64>, ()> for OldStyleScaler {
            type Fitted = FittedOldStyleScaler;
            type Error = FerroError;

            fn fit(&self, _x: &Vec<f64>, _y: &()) -> Result<FittedOldStyleScaler, FerroError> {
                Ok(FittedOldStyleScaler { factor: 3.0 })
            }
        }

        impl Transform<Vec<f64>> for FittedOldStyleScaler {
            type Output = Vec<f64>;
            type Error = FerroError;

            fn transform(&self, x: &Vec<f64>) -> Result<Vec<f64>, FerroError> {
                Ok(x.iter().map(|v| v * self.factor).collect())
            }
        }

        let pipeline = TypedPipeline::new()
            .then(TransformAdapter::new(OldStyleScaler))
            .finish(SumEstimator);

        let x = vec![1.0, 2.0, 3.0];
        let y = vec![0.0];

        let fitted = pipeline.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();

        // Scaled by 3: [3.0, 6.0, 9.0], sum = 18.0
        assert!((pred - 18.0).abs() < 1e-10);
    }

    #[test]
    fn test_estimator_adapter_bridges_existing_types() {
        // Create a type that uses the existing Fit/Predict traits.
        struct OldStyleRegressor;
        struct FittedOldStyleRegressor;

        impl Fit<Vec<f64>, Vec<f64>> for OldStyleRegressor {
            type Fitted = FittedOldStyleRegressor;
            type Error = FerroError;

            fn fit(
                &self,
                _x: &Vec<f64>,
                _y: &Vec<f64>,
            ) -> Result<FittedOldStyleRegressor, FerroError> {
                Ok(FittedOldStyleRegressor)
            }
        }

        impl Predict<Vec<f64>> for FittedOldStyleRegressor {
            type Output = f64;
            type Error = FerroError;

            fn predict(&self, x: &Vec<f64>) -> Result<f64, FerroError> {
                Ok(x.iter().sum::<f64>() * 2.0)
            }
        }

        let pipeline = TypedPipeline::new()
            .then(Doubler)
            .finish(EstimatorAdapter::new(OldStyleRegressor));

        let x = vec![1.0, 2.0, 3.0];
        let y = vec![0.0];

        let fitted = pipeline.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();

        // Doubled: [2.0, 4.0, 6.0], sum*2 = 24.0
        assert!((pred - 24.0).abs() < 1e-10);
    }

    #[test]
    fn test_both_adapters_together() {
        // Use both adapters in the same pipeline.
        struct OldScaler;
        struct FittedOldScaler;

        impl Fit<Vec<f64>, ()> for OldScaler {
            type Fitted = FittedOldScaler;
            type Error = FerroError;

            fn fit(&self, _x: &Vec<f64>, _y: &()) -> Result<FittedOldScaler, FerroError> {
                Ok(FittedOldScaler)
            }
        }

        impl Transform<Vec<f64>> for FittedOldScaler {
            type Output = Vec<f64>;
            type Error = FerroError;

            fn transform(&self, x: &Vec<f64>) -> Result<Vec<f64>, FerroError> {
                Ok(x.iter().map(|v| v * 10.0).collect())
            }
        }

        struct OldPredictor;
        struct FittedOldPredictor;

        impl Fit<Vec<f64>, Vec<f64>> for OldPredictor {
            type Fitted = FittedOldPredictor;
            type Error = FerroError;

            fn fit(&self, _x: &Vec<f64>, _y: &Vec<f64>) -> Result<FittedOldPredictor, FerroError> {
                Ok(FittedOldPredictor)
            }
        }

        impl Predict<Vec<f64>> for FittedOldPredictor {
            type Output = f64;
            type Error = FerroError;

            fn predict(&self, x: &Vec<f64>) -> Result<f64, FerroError> {
                Ok(x.iter().sum())
            }
        }

        let pipeline = TypedPipeline::new()
            .then(TransformAdapter::new(OldScaler))
            .finish(EstimatorAdapter::new(OldPredictor));

        let x = vec![1.0, 2.0, 3.0];
        let y = vec![0.0];

        let fitted = pipeline.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();

        // Scaled by 10: [10.0, 20.0, 30.0], sum = 60.0
        assert!((pred - 60.0).abs() < 1e-10);
    }

    #[test]
    fn test_transform_only_single_step() {
        let pipeline = TypedPipeline::new().then(Doubler).finish_transform();

        let x = vec![1.0, 2.0, 3.0];

        let fitted = pipeline.fit(&x, &()).unwrap();
        let result = fitted.transform(&x).unwrap();

        assert!((result[0] - 2.0).abs() < 1e-10);
        assert!((result[1] - 4.0).abs() < 1e-10);
        assert!((result[2] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_four_step_pipeline() {
        let pipeline = TypedPipeline::new()
            .then(Doubler)
            .then(Adder { offset: 1.0 })
            .then(Squarer)
            .then(Negator)
            .finish(SumEstimator);

        let x = vec![1.0, 2.0];
        let y = vec![0.0];

        let fitted = pipeline.fit(&x, &y).unwrap();
        let pred = fitted.predict(&x).unwrap();

        // [1.0, 2.0] -> doubled [2.0, 4.0] -> +1 [3.0, 5.0]
        // -> squared [9.0, 25.0] -> negated [-9.0, -25.0] -> sum = -34.0
        assert!((pred - (-34.0)).abs() < 1e-10);
    }

    #[test]
    fn test_fitted_pipeline_predicts_on_new_data() {
        let pipeline = TypedPipeline::new()
            .then(Doubler)
            .then(Adder { offset: 1.0 })
            .finish(SumEstimator);

        let x_train = vec![1.0, 2.0, 3.0];
        let y_train = vec![0.0];

        let fitted = pipeline.fit(&x_train, &y_train).unwrap();

        // Predict on completely different data.
        let x_test = vec![10.0, 20.0];
        let pred = fitted.predict(&x_test).unwrap();

        // Doubled: [20.0, 40.0], +1: [21.0, 41.0], sum = 62.0
        assert!((pred - 62.0).abs() < 1e-10);
    }
}
