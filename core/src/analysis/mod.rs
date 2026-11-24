//! What does a measured velocity field mean?
//!
//! - [`gradient`]: the velocity-gradient tensor at every grid point, and from
//!   it **vorticity** (local spin) and **divergence** (local expansion).
//! - [`critical`]: **critical points**, the places where the velocity is
//!   zero, classified by the eigenvalues of the gradient tensor (saddle,
//!   centre, focus, node). They are the "skeleton" of a flow pattern.
//! - [`fit`]: least-squares fits of simple models (uniform, vortex,
//!   source/sink, cylinder) to the field. Reports which one fits best and how
//!   well, as an R² score.
//!
//! Everything here works on any [`VectorField`](crate::VectorField), whether
//! measured by PIV or sampled from an exact flow with
//! [`VectorField::sample`](crate::VectorField::sample). Only `Valid` vectors
//! are used; outliers and empty windows are skipped.
//!
//! Units follow the field. For a PIV field in pixels per frame, vorticity
//! and divergence are in 1/frame, and fitted strengths (Γ, m) in px²/frame.
//! The calibration (mm per pixel, frame rate) is applied by the caller.

pub mod critical;
pub mod fit;
pub mod gradient;
mod least_squares; // private: an implementation detail of `fit`

pub use critical::{CriticalPoint, CriticalPointKind, find_critical_points, gradient_scale};
pub use fit::{FitReport, FlowModel, ModelFit, fit_flow_models};
pub use gradient::{
    GradientField, ScalarField, VelocityGradient, divergence, velocity_gradients, vorticity,
};
