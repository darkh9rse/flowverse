//! # flowverse-core
//!
//! The physics engine of FlowVerse. It knows nothing about the web: it takes
//! grayscale frames in and gives velocity fields out.
//!
//! Modules:
//! - [`image`]: a minimal grayscale [`Image`] type.
//! - [`flow`]: analytical potential-flow building blocks (the [`FlowElement`]
//!   trait with [`Uniform`], [`Source`], [`Vortex`], [`Doublet`], plus
//!   [`Cylinder`], and two exact viscous Navier–Stokes solutions,
//!   [`Couette`] and [`Poiseuille`] channel flows).
//! - [`synthetic`]: makes fake particle images moved by a known flow, so the
//!   PIV can be tested against an exact answer.
//! - [`piv`]: particle image velocimetry (FFT cross-correlation, Gaussian
//!   sub-pixel peak, normalized median outlier test).
//! - [`analysis`]: what the measured field means: vorticity, divergence,
//!   critical points, and which simple flow model fits it best.
//!
//! ## Coordinate convention (used everywhere in this crate)
//! `x` points right and `y` points **down**, like image rows. Pixel `(i, j)`
//! has its centre at the coordinate `(i, j)`. Velocities are in pixels per
//! unit time; with `dt = 1` that is "pixels per frame".

// `pub mod` makes each module (one file in src/) part of the public API.
pub mod analysis;
pub mod flow;
pub mod image;
pub mod piv;
pub mod synthetic;

// Re-export the most used types so callers can write
// `flowverse_core::Image` instead of `flowverse_core::image::Image`.
pub use flow::{Couette, Cylinder, Doublet, FlowElement, Poiseuille, Source, Uniform, Vortex};
pub use image::Image;
pub use piv::{PivConfig, Vector, VectorField, VectorStatus};
pub use synthetic::ParticleImageConfig;
