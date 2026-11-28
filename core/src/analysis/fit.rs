//! Which simple flow does the measurement look like?
//!
//! Seven candidate models are fitted to the measured field by least squares
//! (minimising the summed squared vector difference):
//!
//! | model      | parameters                                   | count |
//! |------------|----------------------------------------------|-------|
//! | uniform    | u, v                                         | 2     |
//! | couette    | wall position, angle; shear rate             | 3     |
//! | vortex     | centre x, y; circulation Γ; core rc          | 4     |
//! | source     | centre x, y; strength m; core rc             | 4     |
//! | poiseuille | centreline position, angle; half-width; peak | 4     |
//! | cylinder   | centre x, y; stream u, v; radius R           | 5     |
//! | jet        | centreline position, angle; half-width;      | 5     |
//! |            | shear-layer thickness; peak                  |       |
//!
//! The first four plus the cylinder are ideal (inviscid) flows; Couette and
//! Poiseuille are viscous channel flows, exact Navier–Stokes solutions (see
//! [`crate::flow::Couette`]); the jet is a free stream with no walls (see
//! [`crate::flow::Jet`]). A line in the plane has two degrees of freedom
//! (position across it, and angle), which is why "wall position, angle"
//! counts 2: sliding the line along itself changes nothing.
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
//! **Physically impossible fits are ruled out.** A good R² is not enough:
//! a Poiseuille fit whose walls cross the measured area while fluid keeps
//! moving behind them can't be a channel (nothing flows inside a solid
//! wall). This is what tells a free jet from a channel when both profiles
//! look alike. See `ruled_out_reason`. Ruled-out fits keep their R² in
//! the report, with the reason, but can't be chosen as best (unless every
//! model is ruled out).
//!
//! **Simpler models win ties.** A cylinder with radius → 0 is a uniform
//! flow, so the cylinder can always match the uniform score. A model with
//! fewer parameters is chosen if its R² is within [`SIMPLER_MODEL_MARGIN`]
//! of the best (Occam's razor), *and* it leaves at most
//! [`SIMPLER_MODEL_RATIO`] times as much motion unexplained (plus a 0.1 %
//! floor). The second condition matters when every fit is excellent: R²
//! 0.984 vs 0.997 is a gap of only 0.013, but the simpler model misses five
//! times more of the motion, so it is not really a tie.

use std::f64::consts::PI;

use super::gradient::{ScalarField, divergence, grid_spacing, vorticity};
use super::least_squares::levenberg_marquardt;
use crate::flow::{Couette, Cylinder, FlowElement, Jet, Poiseuille, Source, Uniform, Vortex};
use crate::piv::{VectorField, VectorStatus};

/// A model with fewer parameters is preferred if its R² is at most this
/// much below the best.
pub const SIMPLER_MODEL_MARGIN: f64 = 0.02;

/// …and leaves at most this many times the best model's unexplained
/// fraction (1 − R²), plus [`TIE_FLOOR`].
pub const SIMPLER_MODEL_RATIO: f64 = 2.0;

/// Differences in unexplained motion below 0.1 % are noise-level ties.
const TIE_FLOOR: f64 = 0.001;

/// A fitted model with its parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FlowModel {
    Uniform(Uniform),
    Vortex(Vortex),
    Source(Source),
    Cylinder(Cylinder),
    Couette(Couette),
    Poiseuille(Poiseuille),
    Jet(Jet),
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
            FlowModel::Couette(_) => "couette",
            FlowModel::Poiseuille(_) => "poiseuille",
            FlowModel::Jet(_) => "jet",
        }
    }

    /// Number of free parameters in the fit.
    pub fn parameter_count(&self) -> usize {
        match self {
            FlowModel::Uniform(_) => 2,
            FlowModel::Couette(_) => 3,
            FlowModel::Vortex(_) | FlowModel::Source(_) | FlowModel::Poiseuille(_) => 4,
            FlowModel::Cylinder(_) | FlowModel::Jet(_) => 5,
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
            FlowModel::Couette(f) => f.velocity(x, y),
            FlowModel::Poiseuille(f) => f.velocity(x, y),
            FlowModel::Jet(f) => f.velocity(x, y),
        }
    }
}

/// One model's fit result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelFit {
    pub model: FlowModel,
    /// Uncentred R² (see module docs). Can be negative for a terrible fit.
    pub r_squared: f64,
    /// Why this model can't be the answer despite its R², if it can't.
    pub ruled_out: Option<&'static str>,
}

/// All seven fits plus the chosen best one.
#[derive(Debug, Clone, PartialEq)]
pub struct FitReport {
    /// In order: uniform, vortex, source, cylinder, couette, poiseuille, jet.
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

/// Fit all seven models to the field's valid vectors.
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
    // The channel models place their line relative to the middle of the data.
    let (cx, cy) = (guesses.middle[0], guesses.middle[1]);

    // Each fit: starting parameters → least squares → turn into a model.
    // `Box<dyn Fn>` holds "some function", either a plain `fn` or a closure
    // that remembers (cx, cy); a plain `fn` pointer can't capture values.
    let candidates: [(Vec<f64>, Box<dyn Fn(&[f64]) -> FlowModel>); 7] = [
        (guesses.uniform.to_vec(), Box::new(uniform_model)),
        (guesses.vortex.to_vec(), Box::new(vortex_model)),
        (guesses.source.to_vec(), Box::new(source_model)),
        (guesses.cylinder.to_vec(), Box::new(cylinder_model)),
        (
            guesses.couette.to_vec(),
            Box::new(move |p: &[f64]| couette_model(cx, cy, p)),
        ),
        (
            guesses.poiseuille.to_vec(),
            Box::new(move |p: &[f64]| poiseuille_model(cx, cy, p)),
        ),
        (
            guesses.jet.to_vec(),
            Box::new(move |p: &[f64]| jet_model(cx, cy, p)),
        ),
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
                ruled_out: ruled_out_reason(&model, &samples),
            }
        })
        .collect();

    // Only physically possible models compete (all of them if none is).
    let allowed: Vec<usize> = if fits.iter().all(|f| f.ruled_out.is_some()) {
        (0..fits.len()).collect()
    } else {
        (0..fits.len())
            .filter(|&i| fits[i].ruled_out.is_none())
            .collect()
    };
    // Highest R², then the simplest model within the margin of it.
    let best_r2 = allowed
        .iter()
        .map(|&i| fits[i].r_squared)
        .fold(f64::NEG_INFINITY, f64::max);
    let best = allowed
        .iter()
        .copied()
        .filter(|&i| {
            let (miss, best_miss) = (1.0 - fits[i].r_squared, 1.0 - best_r2);
            fits[i].r_squared >= best_r2 - SIMPLER_MODEL_MARGIN
                && miss <= SIMPLER_MODEL_RATIO * best_miss + TIE_FLOOR
        })
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

/// Speed (px per frame) that counts as "really moving": about 4× the random
/// error of PIV (≈ 0.06 px, see tests/piv_synthetic.rs). Fields here are
/// PIV displacements in pixels, before any calibration.
const MOVING_PX: f64 = 0.25;

/// A physical reason the fitted `model` is impossible, or `None`.
///
/// Poiseuille flow needs solid walls at n = ±h. If at least 10 % of the
/// vectors lie behind those walls (|n| > 1.1 h, a little margin for the
/// PIV window blurring the wall) and the fluid there typically moves (median
/// speed above [`MOVING_PX`] and above 2 % of the peak), there is no wall:
/// the fit is describing something else, usually a free jet.
fn ruled_out_reason(model: &FlowModel, samples: &[Sample]) -> Option<&'static str> {
    let FlowModel::Poiseuille(p) = model else {
        return None;
    };
    let (s, c) = p.angle.sin_cos();
    let mut behind: Vec<f64> = samples
        .iter()
        .filter(|q| (-(q.x - p.x) * s + (q.y - p.y) * c).abs() > 1.1 * p.half_width)
        .map(|q| q.u.hypot(q.v))
        .collect();
    if behind.len() * 10 < samples.len() {
        return None; // walls at or beyond the edge of the picture
    }
    behind.sort_by(f64::total_cmp);
    let median = behind[behind.len() / 2];
    (median > MOVING_PX && median > 0.02 * p.max_speed.abs())
        .then_some("fluid keeps moving beyond where the channel walls would be")
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

/// The point at signed distance `n` across a line at `angle` from `(cx, cy)`
/// (the normal is `(−sin, cos)`, matching [`crate::flow::Couette`]).
fn point_across(cx: f64, cy: f64, angle: f64, n: f64) -> (f64, f64) {
    let (s, c) = angle.sin_cos();
    (cx - n * s, cy + n * c)
}

/// Wrap an angle into (−π, π].
fn wrap(angle: f64) -> f64 {
    let a = angle.rem_euclid(2.0 * PI);
    if a > PI { a - 2.0 * PI } else { a }
}

/// p = [wall offset n0 from (cx, cy), angle, shear rate].
fn couette_model(cx: f64, cy: f64, p: &[f64]) -> FlowModel {
    let (x, y) = point_across(cx, cy, p[1], p[0]);
    // Turning the wall line by π gives the *same* flow (both the along-wall
    // direction and the across-wall distance flip sign), so report the angle
    // in (−π/2, π/2] to keep it unique.
    let mut angle = wrap(p[1]);
    if angle <= -PI / 2.0 {
        angle += PI;
    } else if angle > PI / 2.0 {
        angle -= PI;
    }
    FlowModel::Couette(Couette {
        x,
        y,
        angle,
        shear_rate: p[2],
    })
}

/// p = [centreline offset n0 from (cx, cy), angle, half-width, peak speed].
fn poiseuille_model(cx: f64, cy: f64, p: &[f64]) -> FlowModel {
    let (x, y) = point_across(cx, cy, p[1], p[0]);
    // A negative peak along `angle` is the same as a positive peak along
    // `angle + π` (the parabola is symmetric), so keep the peak positive.
    let (angle, max_speed) = if p[3] < 0.0 {
        (p[1] + PI, -p[3])
    } else {
        (p[1], p[3])
    };
    FlowModel::Poiseuille(Poiseuille {
        x,
        y,
        angle: wrap(angle),
        half_width: p[2].abs(),
        max_speed,
    })
}

/// p = [centreline offset n0 from (cx, cy), angle, half-width,
/// shear-layer thickness, peak speed]. Same symmetry as Poiseuille: keep
/// the peak positive.
fn jet_model(cx: f64, cy: f64, p: &[f64]) -> FlowModel {
    let (x, y) = point_across(cx, cy, p[1], p[0]);
    let (angle, max_speed) = if p[4] < 0.0 {
        (p[1] + PI, -p[4])
    } else {
        (p[1], p[4])
    };
    FlowModel::Jet(Jet {
        x,
        y,
        angle: wrap(angle),
        half_width: p[2].abs(),
        shear_thickness: p[3].abs(),
        max_speed,
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
    couette: [f64; 3],
    poiseuille: [f64; 4],
    jet: [f64; 5],
    /// Centroid of the vectors: the reference point for the channel lines.
    middle: [f64; 2],
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

        let (couette, poiseuille) = channel_guesses(samples, cx, cy, h);
        // Jet: start from the parabola. A parabola of half-width hp drops to
        // half its peak at hp/√2, which is where a jet's shear layers sit;
        // start with fairly thick shear layers and let the fit sharpen them.
        let [n0, angle, hp, peak] = poiseuille;
        let jet = [n0, angle, hp / 2f64.sqrt(), 0.25 * hp, peak];

        Guesses {
            uniform: [mean_u, mean_v],
            vortex,
            source,
            cylinder,
            couette,
            poiseuille,
            jet,
            middle: [cx, cy],
        }
    }
}

/// Starting points for Couette and Poiseuille.
///
/// 1. **Direction.** Channel flow is (nearly) parallel, so the flow
///    direction is the main axis of the velocity vectors: the angle θ that
///    maximises Σ(u cos θ + v sin θ)². That is the classic principal-axis
///    formula θ = ½ atan2(2Σuv, Σu² − Σv²). It works even when the fluid
///    moves both ways (Couette with the wall in view), unlike the mean.
/// 2. **Profile.** Project every vector onto that direction (speed `t`) and
///    note its distance `n` across it. Then fit `t = a + b n` (Couette) and
///    `t = a + b n + c n²` (Poiseuille) by ordinary linear least squares.
/// 3. **Read off the physics.** Line: shear rate = b, wall where t = 0, at
///    n = −a/b. Parabola: vertex (centreline) at n = −b/2c, peak
///    U = a − b²/4c, and the walls where t = 0, h = √(−U/c).
fn channel_guesses(samples: &[Sample], cx: f64, cy: f64, h: f64) -> ([f64; 3], [f64; 4]) {
    let (suu, svv, suv) = samples.iter().fold((0.0, 0.0, 0.0), |acc, s| {
        (acc.0 + s.u * s.u, acc.1 + s.v * s.v, acc.2 + s.u * s.v)
    });
    let mut angle = 0.5 * (2.0 * suv).atan2(suu - svv);
    let (sn, cs) = angle.sin_cos();
    let profile: Vec<(f64, f64)> = samples
        .iter()
        .map(|s| (-(s.x - cx) * sn + (s.y - cy) * cs, s.u * cs + s.v * sn))
        .collect();
    let extent = profile.iter().fold(h, |m, p| m.max(p.0.abs()));

    // Straight line t = a + b n.
    let line = polyfit::<2>(&profile);
    let couette = match line {
        Some([a, b]) if b.abs() > 1e-12 => {
            [(-a / b).clamp(-100.0 * extent, 100.0 * extent), angle, b]
        }
        _ => [0.0, angle, 0.0],
    };

    // Parabola t = a + b n + c n².
    let mut poiseuille = None;
    if let Some([a, b, c]) = polyfit::<3>(&profile) {
        if c != 0.0 {
            let n0 = -b / (2.0 * c);
            let peak = a - b * b / (4.0 * c);
            let h2 = -peak / c;
            if h2 > 0.0 && n0.abs() < 10.0 * extent {
                // Flow running along −angle: flip so the peak is positive.
                if peak < 0.0 {
                    angle += PI;
                }
                // Turning by π also flips the sign of n.
                let n0 = if peak < 0.0 { -n0 } else { n0 };
                poiseuille = Some([n0, angle, h2.sqrt(), peak.abs()]);
            }
        }
    }
    // No sensible parabola (e.g. the profile curves the wrong way): start
    // with a channel just wider than the data and the mean speed.
    let poiseuille = poiseuille.unwrap_or_else(|| {
        let mean = profile.iter().map(|p| p.1).sum::<f64>() / profile.len() as f64;
        [0.0, angle, extent + h, 1.5 * mean]
    });
    (couette, poiseuille)
}

/// Least-squares polynomial `t = p[0] + p[1] n + …` of `K` coefficients
/// through `(n, t)` points, via the normal equations. `const K: usize` is a
/// *const generic*: the array size is part of the type, chosen per call
/// (`polyfit::<2>` for a line, `polyfit::<3>` for a parabola).
fn polyfit<const K: usize>(points: &[(f64, f64)]) -> Option<[f64; K]> {
    // Scale n to about ±1 so the powers stay similar in size; otherwise
    // n⁴ ~ 10⁸ next to 1 makes the equations badly conditioned.
    let scale = points
        .iter()
        .fold(0.0f64, |m, p| m.max(p.0.abs()))
        .max(1e-9);
    let mut ata = [[0.0; K]; K];
    let mut atb = [0.0; K];
    for &(n, t) in points {
        let mut pw = [1.0; K];
        for k in 1..K {
            pw[k] = pw[k - 1] * n / scale;
        }
        for i in 0..K {
            atb[i] += pw[i] * t;
            for j in 0..K {
                ata[i][j] += pw[i] * pw[j];
            }
        }
    }
    // Gaussian elimination with partial pivoting.
    for col in 0..K {
        let pivot = (col..K).max_by(|&i, &j| ata[i][col].abs().total_cmp(&ata[j][col].abs()))?;
        if ata[pivot][col].abs() < 1e-12 {
            return None;
        }
        ata.swap(col, pivot);
        atb.swap(col, pivot);
        for row in col + 1..K {
            let f = ata[row][col] / ata[col][col];
            for k in col..K {
                ata[row][k] -= f * ata[col][k];
            }
            atb[row] -= f * atb[col];
        }
    }
    let mut x = [0.0; K];
    for row in (0..K).rev() {
        let rest: f64 = (row + 1..K).map(|k| ata[row][k] * x[k]).sum();
        x[row] = (atb[row] - rest) / ata[row][row];
    }
    // Undo the scaling: coefficient k multiplies (n/scale)^k.
    for (k, xk) in x.iter_mut().enumerate() {
        *xk /= scale.powi(k as i32);
    }
    Some(x)
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
    fn exact_couette_flow_is_identified() {
        // Still wall above the data, sloping 0.2 rad; fluid sheared below it.
        let truth = Couette {
            x: 60.0,
            y: 20.0,
            angle: 0.2,
            shear_rate: 0.012,
        };
        let r = fit(&truth);
        assert_eq!(r.best().model.name(), "couette", "{r:?}");
        assert!(r.best().r_squared > 0.9999);
        let FlowModel::Couette(c) = r.best().model else {
            unreachable!()
        };
        assert!(
            (c.angle - 0.2).abs() < 1e-4 && (c.shear_rate - 0.012).abs() < 1e-6,
            "{c:?}"
        );
        // The fitted wall point may sit anywhere along the true wall line,
        // so check its distance from that line instead of x and y.
        let gap = -(c.x - 60.0) * 0.2f64.sin() + (c.y - 20.0) * 0.2f64.cos();
        assert!(gap.abs() < 0.01, "wall {gap} px off");
    }

    #[test]
    fn exact_poiseuille_flow_is_identified_in_both_directions() {
        // A channel 180 px wide (walls at y ≈ 32 and 212), tilted slightly.
        for (angle, expect) in [(-0.15, -0.15), (PI - 0.15, PI - 0.15)] {
            let truth = Poiseuille {
                x: 120.0,
                y: 122.0,
                angle,
                half_width: 90.0,
                max_speed: 1.4,
            };
            let r = fit(&truth);
            assert_eq!(r.best().model.name(), "poiseuille", "{r:?}");
            assert!(r.best().r_squared > 0.999, "{r:?}");
            let FlowModel::Poiseuille(p) = r.best().model else {
                unreachable!()
            };
            assert!((wrap(p.angle - expect)).abs() < 1e-3, "{p:?}");
            assert!(
                (p.half_width - 90.0).abs() < 1.0 && (p.max_speed - 1.4).abs() < 0.01,
                "{p:?}"
            );
            let gap = -(p.x - 120.0) * angle.sin() + (p.y - 122.0) * angle.cos();
            assert!(gap.abs() < 0.5, "centreline {gap} px off");
        }
    }

    #[test]
    fn exact_jet_is_identified_and_not_called_a_channel() {
        // A top-hat jet: 120 px wide core, 8 px shear layers, still fluid
        // above and below it inside the 240 px field.
        let truth = Jet {
            x: 120.0,
            y: 118.0,
            angle: 0.1,
            half_width: 60.0,
            shear_thickness: 8.0,
            max_speed: 1.6,
        };
        let r = fit(&truth);
        assert_eq!(r.best().model.name(), "jet", "{r:?}");
        assert!(r.best().r_squared > 0.999, "{r:?}");
        let FlowModel::Jet(j) = r.best().model else {
            unreachable!()
        };
        assert!((j.angle - 0.1).abs() < 1e-3, "{j:?}");
        assert!(
            (j.half_width - 60.0).abs() < 0.5 && (j.shear_thickness - 8.0).abs() < 0.5,
            "{j:?}"
        );
        assert!((j.max_speed - 1.6).abs() < 0.01, "{j:?}");
        // And a real channel flow is still a channel (Poiseuille has one
        // parameter fewer, so it wins a near-tie).
        let r = fit(&Poiseuille {
            x: 120.0,
            y: 122.0,
            angle: 0.0,
            half_width: 90.0,
            max_speed: 1.4,
        });
        assert_eq!(r.best().model.name(), "poiseuille", "{r:?}");
    }

    #[test]
    fn a_channel_with_fluid_moving_behind_its_walls_is_ruled_out() {
        // Horizontal channel, walls at y = 60 and y = 140.
        let channel = FlowModel::Poiseuille(Poiseuille {
            x: 0.0,
            y: 100.0,
            angle: 0.0,
            half_width: 40.0,
            max_speed: 10.0,
        });
        // Vectors every 10 px from y = 0 to 200; inside the channel they
        // follow the parabola. Behind the walls: still, or moving at 0.6.
        let samples = |behind: f64| -> Vec<Sample> {
            (0..=20)
                .map(|k| {
                    let y = k as f64 * 10.0;
                    let n: f64 = y - 100.0;
                    let u = if n.abs() < 40.0 {
                        10.0 * (1.0 - (n / 40.0).powi(2))
                    } else {
                        behind
                    };
                    Sample {
                        x: 0.0,
                        y,
                        u,
                        v: 0.0,
                    }
                })
                .collect()
        };
        assert_eq!(ruled_out_reason(&channel, &samples(0.0)), None);
        assert_eq!(
            ruled_out_reason(&channel, &samples(0.05)),
            None,
            "PIV noise only"
        );
        assert!(ruled_out_reason(&channel, &samples(0.6)).is_some());
        // Other models are never ruled out by this check.
        let uniform = FlowModel::Uniform(Uniform { u: 1.0, v: 0.0 });
        assert_eq!(ruled_out_reason(&uniform, &samples(0.6)), None);
    }

    #[test]
    fn potential_flows_are_not_mistaken_for_channel_flows() {
        // A vortex has uniform vorticity in its core, like Couette shear,
        // so make sure the circular pattern still wins.
        let r = fit(&Vortex::rankine(118.0, 131.0, 400.0, 35.0));
        assert_eq!(r.best().model.name(), "vortex");
        let r = fit(&Uniform { u: 0.7, v: 0.9 });
        assert_eq!(r.best().model.name(), "uniform");
    }

    #[test]
    fn polyfit_recovers_a_parabola() {
        let pts: Vec<(f64, f64)> = (-20..=20)
            .map(|i| {
                (
                    i as f64 * 7.0,
                    2.0 - 0.5 * i as f64 * 7.0 + 0.01 * (i as f64 * 7.0).powi(2),
                )
            })
            .collect();
        let [a, b, c] = polyfit::<3>(&pts).unwrap();
        assert!((a - 2.0).abs() < 1e-9 && (b + 0.5).abs() < 1e-9 && (c - 0.01).abs() < 1e-12);
    }

    #[test]
    fn nothing_moving_gives_no_fit() {
        assert!(fit_flow_models(&sample(&Uniform { u: 0.0, v: 0.0 })).is_none());
    }
}
