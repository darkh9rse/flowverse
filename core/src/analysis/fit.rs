//! Which simple flow does the measurement look like?
//!
//! Four candidate models are fitted to the measured field by least squares
//! (minimising the summed squared vector difference):
//!
//! | model    | parameters                          | count |
//! |----------|-------------------------------------|-------|
//! | uniform  | u, v                                | 2     |
//! | vortex   | centre x, y; circulation Γ; core rc | 4     |
//! | source   | centre x, y; strength m; core rc    | 4     |
//! | cylinder | centre x, y; stream u, v; radius R  | 5     |
//!
//! Each fit gets an R² score: the fraction of the measured motion the model
//! explains (1 = perfect, 0 = no better than "nothing moves").
//!
//! **Why "nothing moves" and not the mean.** The usual R² compares against
//! the mean of the data. For a uniform flow the mean *is* the whole signal,
//! so a perfect uniform fit would score about 0. We use the *uncentred* R²,
//! `1 − Σ|measured − model|² / Σ|measured|²`, which treats every model
//! fairly.
//!
//! **Resolution.** PIV vectors are averages over a window, so structures
//! smaller than about two windows get blurred. A cylinder of radius 35 px
//! measured with 32 px windows fits as R ≈ 28 px; with 16 px windows,
//! R ≈ 34 px. Choose windows of about half the smallest feature or less.
//!
//! **Simpler models win ties.** A cylinder with radius → 0 is a uniform
//! flow, so the cylinder can always match the uniform score. A model with
//! fewer parameters is chosen if its R² is within [`SIMPLER_MODEL_MARGIN`]
//! of the best (Occam's razor).

use std::f64::consts::PI;

use super::gradient::{ScalarField, divergence, grid_spacing, vorticity};
use super::least_squares::levenberg_marquardt;
use crate::flow::{Cylinder, FlowElement, Source, Uniform, Vortex};
use crate::piv::{VectorField, VectorStatus};

/// A model with fewer parameters is preferred if its R² is at most this
/// much below the best.
pub const SIMPLER_MODEL_MARGIN: f64 = 0.02;

/// A fitted model with its parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FlowModel {
    Uniform(Uniform),
    Vortex(Vortex),
    Source(Source),
    Cylinder(Cylinder),
}

impl FlowModel {
    /// Short name, for reports and JSON.
    pub fn name(&self) -> &'static str {
        match self {
            FlowModel::Uniform(_) => "uniform",
            FlowModel::Vortex(_) => "vortex",
            // A source with negative strength is a sink.
            FlowModel::Source(s) if s.strength < 0.0 => "sink",
            FlowModel::Source(_) => "source",
            FlowModel::Cylinder(_) => "cylinder",
        }
    }

    /// Number of free parameters in the fit.
    pub fn parameter_count(&self) -> usize {
        match self {
            FlowModel::Uniform(_) => 2,
            FlowModel::Vortex(_) | FlowModel::Source(_) => 4,
            FlowModel::Cylinder(_) => 5,
        }
    }
}

// A fitted model is itself a flow, so it can be sampled, drawn or used to
// make synthetic images.
impl FlowElement for FlowModel {
    fn velocity(&self, x: f64, y: f64) -> (f64, f64) {
        match self {
            FlowModel::Uniform(f) => f.velocity(x, y),
            FlowModel::Vortex(f) => f.velocity(x, y),
            FlowModel::Source(f) => f.velocity(x, y),
            FlowModel::Cylinder(f) => f.velocity(x, y),
        }
    }
}

/// One model's fit result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelFit {
    pub model: FlowModel,
    /// Uncentred R² (see module docs). Can be negative for a terrible fit.
    pub r_squared: f64,
}

/// All four fits plus the chosen best one.
#[derive(Debug, Clone, PartialEq)]
pub struct FitReport {
    /// In order: uniform, vortex, source, cylinder.
    pub fits: Vec<ModelFit>,
    /// Index into `fits` of the best model.
    pub best: usize,
    /// How many vectors were used.
    pub vector_count: usize,
}

impl FitReport {
    pub fn best(&self) -> &ModelFit {
        &self.fits[self.best]
    }
}

/// One valid measurement.
struct Sample {
    x: f64,
    y: f64,
    u: f64,
    v: f64,
}

/// Fit all four models to the field's valid vectors.
///
/// Returns `None` if there are fewer than 6 valid vectors, or if nothing
/// moves (with no motion there is nothing to explain).
pub fn fit_flow_models(field: &VectorField) -> Option<FitReport> {
    let samples: Vec<Sample> = field
        .vectors
        .iter()
        .filter(|v| v.status == VectorStatus::Valid)
        .map(|v| Sample {
            x: v.x as f64,
            y: v.y as f64,
            u: v.u as f64,
            v: v.v as f64,
        })
        .collect();
    let total: f64 = samples.iter().map(|s| s.u * s.u + s.v * s.v).sum();
    if samples.len() < 6 || total == 0.0 {
        return None;
    }
    let (hx, hy) = grid_spacing(field).unwrap_or((1.0, 1.0));
    let guesses = Guesses::new(field, &samples, hx, hy);

    // Each fit: starting parameters → least squares → turn into a model.
    let candidates: [(Vec<f64>, fn(&[f64]) -> FlowModel); 4] = [
        (guesses.uniform.to_vec(), uniform_model),
        (guesses.vortex.to_vec(), vortex_model),
        (guesses.source.to_vec(), source_model),
        (guesses.cylinder.to_vec(), cylinder_model),
    ];
    let fits: Vec<ModelFit> = candidates
        .into_iter()
        .map(|(p0, build)| {
            // Residuals: for each sample, (model u − measured u) and
            // (model v − measured v). So m = 2 × number of samples.
            let residuals = |p: &[f64], r: &mut [f64]| {
                let model = build(p);
                for (i, s) in samples.iter().enumerate() {
                    let (u, v) = model.velocity(s.x, s.y);
                    r[2 * i] = u - s.u;
                    r[2 * i + 1] = v - s.v;
                }
            };
            let p = levenberg_marquardt(&residuals, 2 * samples.len(), &p0);
            let model = build(&p);
            ModelFit {
                model,
                r_squared: r_squared(&model, &samples, total),
            }
        })
        .collect();

    // Highest R², then the simplest model within the margin of it.
    let best_r2 = fits
        .iter()
        .map(|f| f.r_squared)
        .fold(f64::NEG_INFINITY, f64::max);
    let best = (0..fits.len())
        .filter(|&i| fits[i].r_squared >= best_r2 - SIMPLER_MODEL_MARGIN)
        // Fewest parameters first. For equal counts (vortex vs source) the
        // higher R² wins.
        .min_by(|&i, &j| {
            let (fi, fj) = (&fits[i], &fits[j]);
            fi.model
                .parameter_count()
                .cmp(&fj.model.parameter_count())
                .then(fj.r_squared.total_cmp(&fi.r_squared))
        })
        .unwrap_or(0);

    Some(FitReport {
        fits,
        best,
        vector_count: samples.len(),
    })
}

fn r_squared(model: &FlowModel, samples: &[Sample], total: f64) -> f64 {
    let residual: f64 = samples
        .iter()
        .map(|s| {
            let (u, v) = model.velocity(s.x, s.y);
            (u - s.u).powi(2) + (v - s.v).powi(2)
        })
        .sum();
    1.0 - residual / total
}

// Parameter vector → model. Radii go through `abs` so the optimiser may wander
// through negative values without producing nonsense.
fn uniform_model(p: &[f64]) -> FlowModel {
    FlowModel::Uniform(Uniform { u: p[0], v: p[1] })
}
fn vortex_model(p: &[f64]) -> FlowModel {
    FlowModel::Vortex(Vortex::rankine(p[0], p[1], p[2], p[3].abs()))
}
fn source_model(p: &[f64]) -> FlowModel {
    FlowModel::Source(Source::rankine(p[0], p[1], p[2], p[3].abs()))
}
fn cylinder_model(p: &[f64]) -> FlowModel {
    FlowModel::Cylinder(Cylinder {
        x: p[0],
        y: p[1],
        u: p[2],
        v: p[3],
        radius: p[4].abs(),
    })
}

/// Starting points for the fits, read from the data.
///
/// Nonlinear least squares only finds the *nearest* minimum, so a good start
/// matters. Physics gives us one:
/// - A vortex's total vorticity equals its circulation Γ (Stokes' theorem),
///   its centre sits where vorticity concentrates, and inside the core the
///   vorticity is Γ/(π rc²), which gives rc.
/// - A source works the same way with divergence and strength m.
/// - A cylinder sits in a stream (≈ the median vector) and blocks it, so its
///   centre and size come from the region where the flow is slow.
struct Guesses {
    uniform: [f64; 2],
    vortex: [f64; 4],
    source: [f64; 4],
    cylinder: [f64; 5],
}

impl Guesses {
    fn new(field: &VectorField, samples: &[Sample], hx: f64, hy: f64) -> Self {
        let n = samples.len() as f64;
        let mean_u = samples.iter().map(|s| s.u).sum::<f64>() / n;
        let mean_v = samples.iter().map(|s| s.v).sum::<f64>() / n;
        let cx = samples.iter().map(|s| s.x).sum::<f64>() / n;
        let cy = samples.iter().map(|s| s.y).sum::<f64>() / n;
        let h = hx.max(hy);

        let vortex =
            concentration(field, &vorticity(field), hx * hy, h).unwrap_or([cx, cy, 0.0, h]);
        let source =
            concentration(field, &divergence(field), hx * hy, h).unwrap_or([cx, cy, 0.0, h]);

        // Cylinder: median stream (robust to the slow region near the body),
        // and the slow region's centroid and area.
        let median = |mut v: Vec<f64>| {
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };
        let su = median(samples.iter().map(|s| s.u).collect());
        let sv = median(samples.iter().map(|s| s.v).collect());
        let stream = su.hypot(sv);
        let slow: Vec<&Sample> = samples
            .iter()
            .filter(|s| s.u.hypot(s.v) < 0.3 * stream)
            .collect();
        let cylinder = if slow.is_empty() || stream == 0.0 {
            [cx, cy, su, sv, h]
        } else {
            let k = slow.len() as f64;
            [
                slow.iter().map(|s| s.x).sum::<f64>() / k,
                slow.iter().map(|s| s.y).sum::<f64>() / k,
                su,
                sv,
                // The slow region includes the stagnation zones in front of
                // and behind the body, so this overestimates R a little.
                (k * hx * hy / PI).sqrt().max(h),
            ]
        };

        Guesses {
            uniform: [mean_u, mean_v],
            vortex,
            source,
            cylinder,
        }
    }
}

/// From a scalar field (vorticity or divergence) concentrated in a blob:
/// [centre x, centre y, total = Σ value·area, core radius].
fn concentration(
    field: &VectorField,
    scalar: &ScalarField,
    cell_area: f64,
    h: f64,
) -> Option<[f64; 4]> {
    let known: Vec<(f64, f64, f64)> = scalar
        .values
        .iter()
        .zip(&field.vectors)
        .filter_map(|(s, v)| s.map(|s| (v.x as f64, v.y as f64, s)))
        .collect();
    let total: f64 = known.iter().map(|k| k.2).sum::<f64>() * cell_area;
    if known.is_empty() || total == 0.0 {
        return None;
    }
    // Look at the dominant sign only, and only near its peak. The centroid
    // of *all* positive values would be dragged towards the middle of the
    // image by noise everywhere.
    let sign = total.signum();
    let peak = known
        .iter()
        .map(|k| sign * k.2)
        .fold(f64::NEG_INFINITY, f64::max);
    if peak <= 0.0 {
        return None;
    }
    let strong: Vec<_> = known.iter().filter(|k| sign * k.2 >= 0.3 * peak).collect();
    let w: f64 = strong.iter().map(|k| sign * k.2).sum();
    let x = strong.iter().map(|k| sign * k.2 * k.0).sum::<f64>() / w;
    let y = strong.iter().map(|k| sign * k.2 * k.1).sum::<f64>() / w;
    // Inside a Rankine core the value is total / (π rc²).
    let rc = (total.abs() / (PI * peak)).sqrt().max(h);
    Some([x, y, total, rc])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sample `flow` on a 21×21 grid, 12 px apart (a 240 px square).
    fn sample<F: FlowElement>(flow: &F) -> VectorField {
        VectorField::sample(flow, 7.5, 7.5, 12.0, 21, 21)
    }

    fn fit(flow: &impl FlowElement) -> FitReport {
        fit_flow_models(&sample(flow)).expect("fit should run")
    }

    #[test]
    fn exact_uniform_flow_is_identified() {
        let r = fit(&Uniform { u: 1.2, v: -0.4 });
        assert_eq!(r.best().model.name(), "uniform");
        assert!(r.best().r_squared > 0.9999);
        let FlowModel::Uniform(u) = r.best().model else {
            unreachable!()
        };
        assert!((u.u - 1.2).abs() < 1e-6 && (u.v + 0.4).abs() < 1e-6);
    }

    #[test]
    fn exact_vortex_is_identified_with_its_parameters() {
        let truth = Vortex::rankine(118.0, 131.0, 400.0, 35.0);
        let r = fit(&truth);
        assert_eq!(r.best().model.name(), "vortex", "{r:?}");
        assert!(r.best().r_squared > 0.999);
        let FlowModel::Vortex(v) = r.best().model else {
            unreachable!()
        };
        assert!(
            (v.x - 118.0).abs() < 0.1 && (v.y - 131.0).abs() < 0.1,
            "{v:?}"
        );
        assert!(
            (v.circulation - 400.0).abs() < 2.0 && (v.core_radius - 35.0).abs() < 0.5,
            "{v:?}"
        );
    }

    #[test]
    fn exact_source_and_sink_are_identified() {
        let r = fit(&Source::rankine(110.0, 125.0, 350.0, 30.0));
        assert_eq!(r.best().model.name(), "source");
        assert!(r.best().r_squared > 0.999);
        let r = fit(&Source::rankine(130.0, 115.0, -350.0, 30.0));
        assert_eq!(r.best().model.name(), "sink");
        let FlowModel::Source(s) = r.best().model else {
            unreachable!()
        };
        assert!((s.strength + 350.0).abs() < 2.0, "{s:?}");
    }

    #[test]
    fn exact_cylinder_is_identified() {
        let truth = Cylinder {
            x: 125.0,
            y: 122.0,
            u: 1.0,
            v: 0.25,
            radius: 30.0,
        };
        let r = fit(&truth);
        assert_eq!(r.best().model.name(), "cylinder", "{r:?}");
        assert!(r.best().r_squared > 0.99);
        let FlowModel::Cylinder(c) = r.best().model else {
            unreachable!()
        };
        assert!(
            (c.x - 125.0).abs() < 1.0 && (c.y - 122.0).abs() < 1.0,
            "{c:?}"
        );
        assert!(
            (c.radius - 30.0).abs() < 1.0 && (c.u - 1.0).abs() < 0.02,
            "{c:?}"
        );
    }

    #[test]
    fn nothing_moving_gives_no_fit() {
        assert!(fit_flow_models(&sample(&Uniform { u: 0.0, v: 0.0 })).is_none());
    }
}
