//! Velocity gradients, vorticity and divergence by finite differences.

use rustfft::num_complex::Complex;

use crate::piv::{VectorField, VectorStatus};

/// The velocity-gradient tensor at one point:
///
/// ```text
///     ⎡ ∂u/∂x  ∂u/∂y ⎤
/// J = ⎣ ∂v/∂x  ∂v/∂y ⎦
/// ```
///
/// Near any point, the flow looks like `velocity ≈ J · (offset)`. So `J`
/// says whether the fluid is spinning, spreading out or being stretched
/// there.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VelocityGradient {
    pub dudx: f64,
    pub dudy: f64,
    pub dvdx: f64,
    pub dvdy: f64,
}

impl VelocityGradient {
    /// Vorticity `ω = ∂v/∂x − ∂u/∂y`: twice the local rotation rate.
    ///
    /// In this crate's y-down coordinates, positive ω matches a [`Vortex`]
    /// with positive Γ and appears as **clockwise** rotation on screen.
    ///
    /// [`Vortex`]: crate::Vortex
    pub fn vorticity(&self) -> f64 {
        self.dvdx - self.dudy
    }

    /// Divergence `∂u/∂x + ∂v/∂y`: the rate at which a small patch of fluid
    /// grows in area. It is zero for an incompressible 2-D flow. A video only
    /// sees a 2-D slice, though, so fluid moving towards or away from the
    /// camera shows up as non-zero divergence. That is how a source or sink
    /// appears.
    pub fn divergence(&self) -> f64 {
        self.dudx + self.dvdy
    }

    /// Determinant of J. Together with the trace (= divergence) it decides
    /// the type of a critical point.
    pub fn determinant(&self) -> f64 {
        self.dudx * self.dvdy - self.dudy * self.dvdx
    }

    /// The two eigenvalues of J, which may be complex.
    ///
    /// For a 2×2 matrix they are `λ = T/2 ± √(T²/4 − D)` with trace T and
    /// determinant D. A negative number under the root gives a complex pair:
    /// the flow spirals.
    pub fn eigenvalues(&self) -> [Complex<f64>; 2] {
        let half_t = 0.5 * self.divergence();
        let disc = half_t * half_t - self.determinant();
        if disc >= 0.0 {
            let r = disc.sqrt();
            [Complex::new(half_t + r, 0.0), Complex::new(half_t - r, 0.0)]
        } else {
            let i = (-disc).sqrt();
            [Complex::new(half_t, i), Complex::new(half_t, -i)]
        }
    }
}

/// One [`VelocityGradient`] per grid point of a [`VectorField`], or `None`
/// where it cannot be computed (an invalid vector with no usable
/// neighbours).
#[derive(Debug, Clone, PartialEq)]
pub struct GradientField {
    pub cols: usize,
    pub rows: usize,
    pub gradients: Vec<Option<VelocityGradient>>,
}

impl GradientField {
    pub fn get(&self, col: usize, row: usize) -> Option<VelocityGradient> {
        self.gradients[row * self.cols + col]
    }
}

/// One number per grid point (e.g. vorticity), `None` where unknown.
/// Same layout as the [`VectorField`] it came from.
#[derive(Debug, Clone, PartialEq)]
pub struct ScalarField {
    pub cols: usize,
    pub rows: usize,
    pub values: Vec<Option<f64>>,
}

impl ScalarField {
    pub fn get(&self, col: usize, row: usize) -> Option<f64> {
        self.values[row * self.cols + col]
    }
}

/// Grid spacing `(hx, hy)` in pixels, read from the vector positions.
/// `None` if the grid is a single row or column (no derivative possible).
pub(crate) fn grid_spacing(field: &VectorField) -> Option<(f64, f64)> {
    if field.cols < 2 || field.rows < 2 {
        return None;
    }
    let hx = (field.get(1, 0).x - field.get(0, 0).x) as f64;
    let hy = (field.get(0, 1).y - field.get(0, 0).y) as f64;
    (hx > 0.0 && hy > 0.0).then_some((hx, hy))
}

/// Compute the velocity gradient at every grid point.
///
/// Uses **central differences**, `f'(x) ≈ (f(x+h) − f(x−h)) / 2h`, which
/// are second-order accurate: the error shrinks with h². Where a neighbour
/// is missing (at the grid edge, or an outlier), it falls back to a
/// one-sided difference `(f(x+h) − f(x)) / h`. That is less accurate but
/// better than no value. A point whose own vector is invalid gets `None`.
pub fn velocity_gradients(field: &VectorField) -> GradientField {
    let (cols, rows) = (field.cols, field.rows);
    let mut gradients = vec![None; cols * rows];
    let Some((hx, hy)) = grid_spacing(field) else {
        return GradientField {
            cols,
            rows,
            gradients,
        };
    };

    // Velocity at (col, row) if it exists and is valid. Signed indices so
    // that col - 1 at col = 0 is simply "outside" rather than an underflow.
    let vel = |col: isize, row: isize| -> Option<(f64, f64)> {
        if col < 0 || row < 0 || col >= cols as isize || row >= rows as isize {
            return None;
        }
        let v = field.get(col as usize, row as usize);
        (v.status == VectorStatus::Valid).then_some((v.u as f64, v.v as f64))
    };

    for row in 0..rows as isize {
        for col in 0..cols as isize {
            let Some(centre) = vel(col, row) else {
                continue;
            };
            let along_x = derivative(centre, vel(col - 1, row), vel(col + 1, row), hx);
            let along_y = derivative(centre, vel(col, row - 1), vel(col, row + 1), hy);
            if let (Some((dudx, dvdx)), Some((dudy, dvdy))) = (along_x, along_y) {
                gradients[row as usize * cols + col as usize] = Some(VelocityGradient {
                    dudx,
                    dudy,
                    dvdx,
                    dvdy,
                });
            }
        }
    }
    GradientField {
        cols,
        rows,
        gradients,
    }
}

/// Derivative of (u, v) along one grid direction, from the centre value and
/// its two neighbours (`before` at −h, `after` at +h).
fn derivative(
    centre: (f64, f64),
    before: Option<(f64, f64)>,
    after: Option<(f64, f64)>,
    h: f64,
) -> Option<(f64, f64)> {
    let (lo, hi, span) = match (before, after) {
        (Some(b), Some(a)) => (b, a, 2.0 * h),
        (None, Some(a)) => (centre, a, h),
        (Some(b), None) => (b, centre, h),
        (None, None) => return None,
    };
    Some(((hi.0 - lo.0) / span, (hi.1 - lo.1) / span))
}

/// Vorticity at every grid point. See [`VelocityGradient::vorticity`].
pub fn vorticity(field: &VectorField) -> ScalarField {
    map_gradients(field, |g| g.vorticity())
}

/// Divergence at every grid point. See [`VelocityGradient::divergence`].
pub fn divergence(field: &VectorField) -> ScalarField {
    map_gradients(field, |g| g.divergence())
}

fn map_gradients(field: &VectorField, f: impl Fn(&VelocityGradient) -> f64) -> ScalarField {
    let grads = velocity_gradients(field);
    ScalarField {
        cols: grads.cols,
        rows: grads.rows,
        // `Option::map` applies `f` only when there is a value.
        values: grads.gradients.iter().map(|g| g.as_ref().map(&f)).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flow::{FlowElement, Source, Vortex};
    use std::f64::consts::PI;

    /// A linear flow u = a·x + b·y, v = c·x + d·y. Finite differences
    /// (central *and* one-sided) are exact for it, so every grid point,
    /// edges included, must give exactly (a, b, c, d).
    struct Linear(f64, f64, f64, f64);
    impl FlowElement for Linear {
        fn velocity(&self, x: f64, y: f64) -> (f64, f64) {
            (self.0 * x + self.1 * y, self.2 * x + self.3 * y)
        }
    }

    #[test]
    fn linear_flow_gradient_is_exact_everywhere() {
        let field = VectorField::sample(&Linear(0.1, -0.2, 0.3, 0.05), 3.0, 5.0, 8.0, 6, 5);
        let grads = velocity_gradients(&field);
        for g in &grads.gradients {
            let g = g.expect("all vectors valid, so all gradients exist");
            assert!((g.dudx - 0.1).abs() < 1e-5 && (g.dudy + 0.2).abs() < 1e-5);
            assert!((g.dvdx - 0.3).abs() < 1e-5 && (g.dvdy - 0.05).abs() < 1e-5);
        }
        let w = vorticity(&field).get(2, 2).unwrap();
        assert!((w - (0.3 - -0.2)).abs() < 1e-5);
        let d = divergence(&field).get(0, 4).unwrap();
        assert!((d - 0.15).abs() < 1e-5);
    }

    #[test]
    fn rankine_core_has_vorticity_gamma_over_pi_rc2_and_no_divergence() {
        let (gamma, rc) = (300.0, 40.0);
        let field = VectorField::sample(
            &Vortex::rankine(100.0, 100.0, gamma, rc),
            60.0,
            60.0,
            8.0,
            11,
            11,
        );
        let expected = gamma / (PI * rc * rc);
        // Inside the core (within 20 px of the centre) the field is
        // solid-body rotation, which is linear, so the estimate is exact.
        let w = vorticity(&field);
        let div = divergence(&field);
        for row in 3..8 {
            for col in 3..8 {
                assert!((w.get(col, row).unwrap() - expected).abs() < 1e-5);
                assert!(div.get(col, row).unwrap().abs() < 1e-5);
            }
        }
    }

    #[test]
    fn cored_source_has_divergence_m_over_pi_rc2() {
        let (m, rc) = (200.0, 30.0);
        let field = VectorField::sample(&Source::rankine(50.0, 50.0, m, rc), 34.0, 34.0, 8.0, 5, 5);
        let div = divergence(&field);
        let expected = m / (PI * rc * rc);
        assert!((div.get(2, 2).unwrap() - expected).abs() < 1e-5);
        assert!(vorticity(&field).get(2, 2).unwrap().abs() < 1e-5);
    }

    #[test]
    fn invalid_vectors_are_skipped_and_neighbours_fall_back_to_one_sided() {
        let mut field = VectorField::sample(&Linear(0.1, 0.0, 0.0, 0.2), 0.0, 0.0, 4.0, 5, 5);
        field.vectors[2 * 5 + 2].status = VectorStatus::Outlier;
        // Put garbage in the outlier to prove it is not used.
        field.vectors[2 * 5 + 2].u = 1000.0;
        let grads = velocity_gradients(&field);
        assert_eq!(grads.get(2, 2), None);
        let left = grads.get(1, 2).unwrap(); // its right neighbour is the outlier
        assert!((left.dudx - 0.1).abs() < 1e-5);
    }

    #[test]
    fn eigenvalues_of_rotation_and_strain() {
        // Pure rotation: eigenvalues ±iω/2.
        let rot = VelocityGradient {
            dudx: 0.0,
            dudy: -1.0,
            dvdx: 1.0,
            dvdy: 0.0,
        };
        let [a, b] = rot.eigenvalues();
        assert!(a.re.abs() < 1e-12 && (a.im - 1.0).abs() < 1e-12 && (b.im + 1.0).abs() < 1e-12);
        // Pure strain: eigenvalues 2 and -3.
        let strain = VelocityGradient {
            dudx: 2.0,
            dudy: 0.0,
            dvdx: 0.0,
            dvdy: -3.0,
        };
        let [a, b] = strain.eigenvalues();
        assert!((a.re - 2.0).abs() < 1e-12 && (b.re + 3.0).abs() < 1e-12);
    }
}
