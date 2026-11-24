//! Particle image velocimetry (PIV).
//!
//! The idea: cut frames A and B into small square *interrogation windows*.
//! For each window, find the shift that makes B's patch line up best with
//! A's patch. That shift is the local displacement of the fluid.
//!
//! Steps for each window:
//! 1. **Cross-correlation via FFT.** The correlation `C(s) = Σ A(p)·B(p+s)`
//!    is largest at the shift `s` where the particle patterns overlap.
//!    Computing it directly costs O(N⁴) for an N×N window. By the
//!    correlation theorem it equals `IFFT( conj(FFT(A)) · FFT(B) )`, which
//!    costs O(N² log N).
//! 2. **Gaussian sub-pixel peak.** The highest correlation value is only
//!    accurate to a whole pixel. Fitting a Gaussian through the peak and its
//!    two neighbours (separately in x and y) finds the true peak to about
//!    0.05 px. Before the fit, the loss-of-pairs bias is divided out
//!    (Westerweel 1997); see `Correlator::displacement`.
//! 3. **Normalized median test** (Westerweel & Scarano, 2005). A wrong peak
//!    gives a vector that disagrees with its neighbours. Each vector is
//!    compared with the median of its 8 neighbours and flagged as an outlier
//!    if it is too far off.
//!
//! Reference: Raffel et al., *Particle Image Velocimetry: A Practical Guide*.

use std::fmt;
use std::sync::Arc;

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};

use crate::flow::FlowElement;
use crate::image::Image;

/// Settings for [`analyze`].
#[derive(Debug, Clone, PartialEq)]
pub struct PivConfig {
    /// Interrogation window side length in pixels. Powers of two (16, 32,
    /// 64) make the FFT fastest. Displacements should stay below about a
    /// quarter of this (the "one-quarter rule"), or too few particles stay
    /// inside the window to give a clear peak.
    pub window_size: usize,
    /// Distance between neighbouring windows. `window_size / 2` (50 %
    /// overlap) is the usual choice: twice as many vectors in each
    /// direction, though neighbouring vectors are not independent.
    pub step: usize,
    /// Normalized-median threshold. Westerweel & Scarano recommend 2.
    pub median_threshold: f32,
    /// Normalized-median ε in pixels. It reflects the noise level of PIV
    /// (~0.1 px) and stops tiny, perfectly smooth fields from flagging
    /// everything because of a near-zero median residual.
    pub median_epsilon: f32,
}

impl Default for PivConfig {
    fn default() -> Self {
        PivConfig {
            window_size: 32,
            step: 16,
            median_threshold: 2.0,
            median_epsilon: 0.1,
        }
    }
}

/// Quality flag for one vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorStatus {
    /// Passed every check.
    Valid,
    /// Rejected by the normalized median test.
    Outlier,
    /// No usable correlation peak, e.g. the window was blank.
    NoPeak,
}

/// One measured displacement.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vector {
    /// Position of the window centre in pixel coordinates.
    pub x: f32,
    pub y: f32,
    /// Displacement from frame A to frame B in pixels. Divide by the time
    /// between frames to get a velocity.
    pub u: f32,
    pub v: f32,
    pub status: VectorStatus,
}

/// A regular grid of vectors, `cols × rows`, stored row-major like [`Image`].
#[derive(Debug, Clone, PartialEq)]
pub struct VectorField {
    pub cols: usize,
    pub rows: usize,
    pub vectors: Vec<Vector>,
}

impl VectorField {
    /// An exact field: `flow`'s velocity sampled on a `cols × rows` grid
    /// whose first point is `(x0, y0)` and whose points are `spacing` apart.
    /// Every vector is `Valid`. Useful for tests, and for drawing a fitted
    /// model next to the measurement.
    pub fn sample<F: FlowElement + ?Sized>(
        flow: &F,
        x0: f64,
        y0: f64,
        spacing: f64,
        cols: usize,
        rows: usize,
    ) -> Self {
        let mut vectors = Vec::with_capacity(cols * rows);
        for row in 0..rows {
            for col in 0..cols {
                let x = x0 + col as f64 * spacing;
                let y = y0 + row as f64 * spacing;
                let (u, v) = flow.velocity(x, y);
                vectors.push(Vector {
                    x: x as f32,
                    y: y as f32,
                    u: u as f32,
                    v: v as f32,
                    status: VectorStatus::Valid,
                });
            }
        }
        VectorField {
            cols,
            rows,
            vectors,
        }
    }

    /// The vector in grid column `col` and row `row`.
    pub fn get(&self, col: usize, row: usize) -> &Vector {
        assert!(
            col < self.cols && row < self.rows,
            "grid cell ({col}, {row}) out of range"
        );
        &self.vectors[row * self.cols + col]
    }
}

/// Reasons [`analyze`] can refuse to run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PivError {
    /// Frames A and B are not the same size.
    SizeMismatch,
    /// `window_size` below 4 or `step` of 0.
    BadConfig,
    /// The image is smaller than one window.
    ImageTooSmall,
}

impl fmt::Display for PivError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let msg = match self {
            PivError::SizeMismatch => "frames A and B have different sizes",
            PivError::BadConfig => "window_size must be at least 4 and step at least 1",
            PivError::ImageTooSmall => "image is smaller than one interrogation window",
        };
        f.write_str(msg)
    }
}

impl std::error::Error for PivError {}

/// Run PIV on a frame pair: correlate every window, then apply the
/// normalized median test.
pub fn analyze(a: &Image, b: &Image, cfg: &PivConfig) -> Result<VectorField, PivError> {
    if a.width() != b.width() || a.height() != b.height() {
        return Err(PivError::SizeMismatch);
    }
    let n = cfg.window_size;
    if n < 4 || cfg.step == 0 {
        return Err(PivError::BadConfig);
    }
    if a.width() < n || a.height() < n {
        return Err(PivError::ImageTooSmall);
    }

    // Number of windows that fit completely inside the image.
    let cols = (a.width() - n) / cfg.step + 1;
    let rows = (a.height() - n) / cfg.step + 1;

    let mut correlator = Correlator::new(n);
    let mut vectors = Vec::with_capacity(cols * rows);
    for row in 0..rows {
        for col in 0..cols {
            let x0 = col * cfg.step;
            let y0 = row * cfg.step;
            // The window covers pixels x0 ..= x0+n-1. Pixel centres sit on
            // whole numbers, so its centre is at x0 + (n-1)/2.
            let centre = (n as f32 - 1.0) / 2.0;
            let x = x0 as f32 + centre;
            let y = y0 as f32 + centre;
            let vector = match correlator.displacement(a, b, x0, y0) {
                Some((u, v)) => Vector {
                    x,
                    y,
                    u,
                    v,
                    status: VectorStatus::Valid,
                },
                None => Vector {
                    x,
                    y,
                    u: 0.0,
                    v: 0.0,
                    status: VectorStatus::NoPeak,
                },
            };
            vectors.push(vector);
        }
    }

    let mut field = VectorField {
        cols,
        rows,
        vectors,
    };
    normalized_median_test(&mut field, cfg.median_threshold, cfg.median_epsilon);
    Ok(field)
}

/// Flag vectors that disagree with their neighbours (Westerweel & Scarano 2005).
///
/// For each vector with value `U0` and its (up to 8) neighbours `Ui`:
/// - `Um` = median of the neighbours,
/// - `ri = |Ui - Um|` measures how much the neighbours scatter,
/// - `rm` = median of the `ri`,
/// - normalized residual `r0 = |U0 - Um| / (rm + ε)`.
///
/// Dividing by `rm` makes the test *adaptive*: a vortex core with large
/// natural variation is not punished, but a spike in a smooth region is.
/// The test runs on `u` and `v` separately, and a vector is an outlier if
/// either residual exceeds `threshold`.
///
/// All decisions are based on the original field and only applied at the
/// end, so flagging one vector cannot change the verdict on its neighbour.
/// Returns how many vectors were newly flagged.
pub fn normalized_median_test(field: &mut VectorField, threshold: f32, epsilon: f32) -> usize {
    let (cols, rows) = (field.cols, field.rows);
    let mut flagged = Vec::new();
    // Buffers reused for every vector: at most 8 neighbours.
    let mut nu = Vec::with_capacity(8);
    let mut nv = Vec::with_capacity(8);

    for row in 0..rows {
        for col in 0..cols {
            let centre = field.vectors[row * cols + col];
            if centre.status != VectorStatus::Valid {
                continue;
            }
            nu.clear();
            nv.clear();
            // Visit the 3×3 block around (col, row), clipped at the grid
            // edges. `saturating_sub(1)` stops at 0 instead of underflowing,
            // which would panic for an unsigned `usize`.
            for r in row.saturating_sub(1)..=(row + 1).min(rows - 1) {
                for c in col.saturating_sub(1)..=(col + 1).min(cols - 1) {
                    if r == row && c == col {
                        continue;
                    }
                    let n = field.vectors[r * cols + c];
                    if n.status != VectorStatus::NoPeak {
                        nu.push(n.u);
                        nv.push(n.v);
                    }
                }
            }
            // With fewer than 3 neighbours a median says little; leave it.
            if nu.len() < 3 {
                continue;
            }
            let ru = normalized_residual(centre.u, &mut nu, epsilon);
            let rv = normalized_residual(centre.v, &mut nv, epsilon);
            if ru > threshold || rv > threshold {
                flagged.push(row * cols + col);
            }
        }
    }

    for &i in &flagged {
        field.vectors[i].status = VectorStatus::Outlier;
    }
    flagged.len()
}

/// `|value - median| / (median residual + ε)` for one component.
/// Reorders `neighbours` (sorting is how we find the median).
fn normalized_residual(value: f32, neighbours: &mut [f32], epsilon: f32) -> f32 {
    let um = median(neighbours);
    for x in neighbours.iter_mut() {
        *x = (*x - um).abs();
    }
    let rm = median(neighbours);
    (value - um).abs() / (rm + epsilon)
}

/// Median of a non-empty slice (sorts it in place).
fn median(values: &mut [f32]) -> f32 {
    // `total_cmp` gives a full ordering for floats, including NaN, so the
    // sort can never panic. Plain `partial_cmp` would.
    values.sort_unstable_by(|a, b| a.total_cmp(b));
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        0.5 * (values[n / 2 - 1] + values[n / 2])
    }
}

/// Holds FFT plans and work buffers for one window size, so they are made
/// once per frame pair instead of once per window.
struct Correlator {
    n: usize,
    // `Arc<dyn Fft<f32>>` is how rustfft hands out plans: a shared pointer
    // to "some FFT algorithm", chosen by the planner for this size.
    forward: Arc<dyn Fft<f32>>,
    inverse: Arc<dyn Fft<f32>>,
    buf_a: Vec<Complex<f32>>,
    buf_b: Vec<Complex<f32>>,
}

impl Correlator {
    fn new(n: usize) -> Self {
        let mut planner = FftPlanner::new();
        Correlator {
            n,
            forward: planner.plan_fft_forward(n),
            inverse: planner.plan_fft_inverse(n),
            buf_a: vec![Complex::new(0.0, 0.0); n * n],
            buf_b: vec![Complex::new(0.0, 0.0); n * n],
        }
    }

    /// Sub-pixel displacement of the window whose top-left pixel is
    /// `(x0, y0)`, or `None` if there is no usable peak.
    fn displacement(&mut self, a: &Image, b: &Image, x0: usize, y0: usize) -> Option<(f32, f32)> {
        let n = self.n;
        load_window(a, x0, y0, n, &mut self.buf_a);
        load_window(b, x0, y0, n, &mut self.buf_b);

        fft2d(&mut self.buf_a, n, &*self.forward);
        fft2d(&mut self.buf_b, n, &*self.forward);
        // Correlation theorem: corr = IFFT(conj(FA) · FB). Store it in buf_a.
        for (fa, fb) in self.buf_a.iter_mut().zip(&self.buf_b) {
            *fa = fa.conj() * fb;
        }
        fft2d(&mut self.buf_a, n, &*self.inverse);
        // rustfft does not divide by n² on the inverse. The correlation is
        // scaled, but the peak position does not care.

        // Find the highest value.
        let (mut best, mut best_val) = (0usize, f32::NEG_INFINITY);
        for (i, c) in self.buf_a.iter().enumerate() {
            if c.re > best_val {
                best_val = c.re;
                best = i;
            }
        }
        // A blank window has zero correlation everywhere. Non-positive or
        // NaN peaks mean "no signal". `!(x > 0)` also catches NaN.
        if !(best_val > 0.0) {
            return None;
        }
        let (row, col) = (best / n, best % n);

        // Indices past n/2 are negative shifts that wrapped around:
        // e.g. n-1 means -1.
        let wrap = |k: usize| {
            if k > n / 2 {
                k as f32 - n as f32
            } else {
                k as f32
            }
        };

        // Loss-of-pairs bias correction (Westerweel 1997). At shift s only a
        // fraction (1 - |s|/n) of the window's particles still have their
        // partner inside the window, so the correlation is multiplied by
        // that tilted "tent". Across the three fit points the tilt pulls the
        // fitted peak towards zero shift: about -0.04 px at 32 px windows.
        // Dividing each sample by its weight removes the tilt before the fit.
        let weight = |k: usize| 1.0 - wrap(k % n).abs() / n as f32;
        // Correlation at row `r`, column `c`, wrapping around the edges
        // (the FFT correlation is *circular*: index n-1 sits next to 0),
        // with the weight divided out.
        let at =
            |r: usize, c: usize| self.buf_a[(r % n) * n + (c % n)].re / (weight(r) * weight(c));
        let c0 = at(row, col);
        let dx = gaussian_offset(at(row, col + n - 1), c0, at(row, col + 1));
        let dy = gaussian_offset(at(row + n - 1, col), c0, at(row + 1, col));

        Some((wrap(col) + dx, wrap(row) + dy))
    }
}

/// Copy an n×n window into a complex buffer, subtracting its mean.
///
/// Subtracting the mean matters. Without it, the uniform background
/// brightness correlates with itself at every shift and adds a broad hill
/// centred on zero shift. That hill drags the peak towards zero and biases
/// every result.
fn load_window(img: &Image, x0: usize, y0: usize, n: usize, out: &mut [Complex<f32>]) {
    let mut sum = 0.0f32;
    for y in 0..n {
        for x in 0..n {
            let p = img.get(x0 + x, y0 + y);
            out[y * n + x] = Complex::new(p, 0.0);
            sum += p;
        }
    }
    let mean = sum / (n * n) as f32;
    for c in out.iter_mut() {
        c.re -= mean;
    }
}

/// In-place 2-D FFT of an n×n row-major buffer.
///
/// A 2-D FFT is a 1-D FFT of every row followed by a 1-D FFT of every
/// column. rustfft only does 1-D, and its `process` transforms each
/// consecutive chunk of length n, which is every row at once. To do the
/// columns we *transpose* (columns become rows), transform the rows again,
/// and transpose back.
fn fft2d(buf: &mut [Complex<f32>], n: usize, fft: &dyn Fft<f32>) {
    fft.process(buf);
    transpose(buf, n);
    fft.process(buf);
    transpose(buf, n);
}

/// Transpose a square n×n matrix in place by swapping across the diagonal.
fn transpose(buf: &mut [Complex<f32>], n: usize) {
    for r in 0..n {
        for c in (r + 1)..n {
            buf.swap(r * n + c, c * n + r);
        }
    }
}

/// Sub-pixel peak offset from three samples: left `cm`, centre `c0`,
/// right `cp`.
///
/// A correlation peak of Gaussian particles is itself Gaussian. The log of a
/// Gaussian is a parabola, so fitting a parabola through the *logs* of the
/// three points gives the exact peak of a Gaussian:
///
/// `δ = (ln cm − ln cp) / (2 (ln cm − 2 ln c0 + ln cp))`
///
/// The logs need positive values. With mean-subtracted windows a neighbour
/// can be ≤ 0, so in that case we fall back to a plain parabola through the
/// raw values. The result is clamped to ±0.5 px. If the centre really is the
/// maximum, the true peak cannot be further away than that.
fn gaussian_offset(cm: f32, c0: f32, cp: f32) -> f32 {
    let delta = if cm > 0.0 && c0 > 0.0 && cp > 0.0 {
        let (lm, l0, lp) = (cm.ln(), c0.ln(), cp.ln());
        let denom = 2.0 * (lm - 2.0 * l0 + lp);
        if denom.abs() > f32::EPSILON {
            (lm - lp) / denom
        } else {
            0.0
        }
    } else {
        let denom = 2.0 * (cm - 2.0 * c0 + cp);
        if denom.abs() > f32::EPSILON {
            (cm - cp) / denom
        } else {
            0.0
        }
    };
    delta.clamp(-0.5, 0.5)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_fit_is_exact_for_a_sampled_gaussian() {
        // Sample a Gaussian with its true peak at 0.3 on the points -1, 0, 1.
        for &peak in &[0.3f32, -0.42, 0.0, 0.49] {
            let g = |x: f32| (-(x - peak).powi(2) / 1.7).exp();
            let est = gaussian_offset(g(-1.0), g(0.0), g(1.0));
            assert!((est - peak).abs() < 1e-4, "peak {peak}, got {est}");
        }
    }

    #[test]
    fn fft2d_then_inverse_returns_the_input_times_n_squared() {
        let n = 8;
        let mut planner = FftPlanner::new();
        let (fwd, inv) = (planner.plan_fft_forward(n), planner.plan_fft_inverse(n));
        let original: Vec<Complex<f32>> = (0..n * n)
            .map(|i| Complex::new((i as f32 * 0.37).sin(), 0.0))
            .collect();
        let mut buf = original.clone();
        fft2d(&mut buf, n, &*fwd);
        fft2d(&mut buf, n, &*inv);
        for (got, want) in buf.iter().zip(&original) {
            assert!((got.re / (n * n) as f32 - want.re).abs() < 1e-4);
        }
    }

    #[test]
    fn integer_shift_with_negative_component_is_found() {
        // Particle images moved by (+3, -2). The y shift is negative, so the
        // correlation peak lands at row index n-2 and must be unwrapped to -2.
        // We correlate one 32×32 window from the middle of a 64×64 frame.
        // Particles leave and enter at the window edges as in real data, so
        // the loss-of-pairs correction is meant to apply.
        use crate::flow::Uniform;
        use crate::synthetic::{ParticleImageConfig, generate_pair};
        let cfg = ParticleImageConfig {
            width: 64,
            height: 64,
            ..Default::default()
        };
        let (a, b) = generate_pair(&Uniform { u: 3.0, v: -2.0 }, &cfg);
        let (u, v) = Correlator::new(32).displacement(&a, &b, 16, 16).unwrap();
        // One window has ~0.05 px random error (see tests/piv_synthetic.rs
        // for the accuracy tests). Here we only need to rule out an
        // integer-pixel or sign mistake.
        assert!(
            (u - 3.0).abs() < 0.15 && (v + 2.0).abs() < 0.15,
            "got ({u}, {v})"
        );
    }

    #[test]
    fn blank_window_has_no_peak() {
        let img = Image::new(16, 16);
        assert_eq!(Correlator::new(16).displacement(&img, &img, 0, 0), None);
    }

    /// Build a smooth field (u = 1, v = 0.5) with one spike.
    fn field_with_spike(spike_at: (usize, usize)) -> VectorField {
        let (cols, rows) = (5, 5);
        let mut vectors = Vec::new();
        for row in 0..rows {
            for col in 0..cols {
                // Small, smooth variation so the field isn't perfectly flat.
                let u = 1.0 + 0.02 * col as f32;
                let v = 0.5 - 0.01 * row as f32;
                vectors.push(Vector {
                    x: col as f32,
                    y: row as f32,
                    u,
                    v,
                    status: VectorStatus::Valid,
                });
            }
        }
        let i = spike_at.1 * cols + spike_at.0;
        vectors[i].u = 6.0;
        vectors[i].v = -3.0;
        VectorField {
            cols,
            rows,
            vectors,
        }
    }

    #[test]
    fn median_test_flags_the_spike_only() {
        for &spike in &[(2, 2), (0, 0), (4, 1)] {
            let mut f = field_with_spike(spike);
            let count = normalized_median_test(&mut f, 2.0, 0.1);
            assert_eq!(count, 1, "spike at {spike:?}");
            assert_eq!(f.get(spike.0, spike.1).status, VectorStatus::Outlier);
        }
    }

    #[test]
    fn median_test_ignores_no_peak_neighbours() {
        let mut f = field_with_spike((2, 2));
        // A NoPeak cell holds a meaningless (0, 0). It must not count as a
        // neighbour, and it must not be flagged.
        f.vectors[0].status = VectorStatus::NoPeak;
        normalized_median_test(&mut f, 2.0, 0.1);
        assert_eq!(f.get(0, 0).status, VectorStatus::NoPeak);
        assert_eq!(f.get(1, 1).status, VectorStatus::Valid);
    }

    #[test]
    fn analyze_rejects_bad_input() {
        let a = Image::new(64, 64);
        let cfg = PivConfig::default();
        assert_eq!(
            analyze(&a, &Image::new(64, 32), &cfg),
            Err(PivError::SizeMismatch)
        );
        assert_eq!(
            analyze(&Image::new(16, 16), &Image::new(16, 16), &cfg),
            Err(PivError::ImageTooSmall)
        );
        let bad = PivConfig { step: 0, ..cfg };
        assert_eq!(analyze(&a, &a, &bad), Err(PivError::BadConfig));
    }
}
