//! # ferrolearn-numerical
//!
//! Numerical foundations for the ferrolearn machine learning framework.
//!
//! This crate provides scipy-equivalent numerical primitives that other
//! ferrolearn crates depend on:
//!
//! - **[`sparse_eig`]** — Sparse eigensolvers (Lanczos iteration) for
//!   large symmetric matrices without forming them densely.
//! - **[`sparse_graph`]** — Graph algorithms on sparse adjacency matrices:
//!   Dijkstra shortest paths, connected components.
//! - **[`distributions`]** — Unified interface to statistical distributions
//!   (Normal, Chi-squared, F, t, Beta, Gamma, Dirichlet) with PDF/CDF/PPF.
//! - **[`optimize`]** — Trust-region Newton-CG optimizer for smooth
//!   unconstrained minimization, plus Brent's method for 1-D bounded
//!   minimization.
//! - **[`interpolate`]** — Cubic spline interpolation with natural and
//!   not-a-knot boundary conditions.
//! - **[`integrate`]** — Adaptive numerical quadrature (Simpson and
//!   Gauss-Legendre).

pub mod distributions;
pub mod integrate;
pub mod interpolate;
pub mod optimize;
pub mod sparse_eig;
pub mod sparse_graph;
pub mod special;
