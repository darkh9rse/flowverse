//! Critical points: where the velocity is zero, and what the flow does
//! around them.
//!
//! Close to a critical point the velocity is approximately `J · offset`, with
//! J the velocity-gradient tensor. The eigenvalues of J fix the local
//! pattern (the standard phase-portrait classification, e.g. Perry & Chong,
//! 1987):
//!
//! | eigenvalues               | pattern                         | example        |
//! |---------------------------|---------------------------------|----------------|
//! | real, opposite signs      | saddle                          | between two vortices |
//! | real, same sign           | node (all in / all out)         | source, sink   |
//! | complex, real part ≈ 0    | centre (closed circles)         | vortex         |
//! | complex, real part ≠ 0    | focus (spiral in / out)         | drain vortex   |
//!
//! Instead of the eigenvalues directly we use the equivalent trace
//! T = λ₁ + λ₂ (the divergence) and determinant D = λ₁ λ₂.

use super::gradient::{VelocityGradient, grid_spacing, velocity_gradients};
use crate::piv::{VectorField, VectorStatus};

/// What the flow looks like around a critical point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CriticalPointKind {
    /// Flow comes in along one direction and leaves along another.
    Saddle,
    /// Closed circular streamlines (a vortex).
    Center,
    /// Spiralling inwards (sink vortex, e.g. a drain).
    StableFocus,
    /// Spiralling outwards.
    UnstableFocus,
    /// Everything flows straight in (sink).
    StableNode,
    /// Everything flows straight out (source).
    UnstableNode,
    /// The gradient is (almost) singular, e.g. pure shear. No clear pattern.
    Degenerate,
}

impl CriticalPointKind {
    /// Lower-case name, for reports and JSON.
    pub fn name(&self) -> &'static str {
        match self {
            CriticalPointKind::Saddle => "saddle",
            CriticalPointKind::Center => "center",
            CriticalPointKind::StableFocus => "stable focus",
            CriticalPointKind::UnstableFocus => "unstable focus",
            CriticalPointKind::StableNode => "stable node",
            CriticalPointKind::UnstableNode => "unstable node",
            CriticalPointKind::Degenerate => "degenerate",
        }
    }
}

/// A located and classified critical point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CriticalPoint {
    pub x: f64,
    pub y: f64,
    pub kind: CriticalPointKind,
    /// Velocity gradient at the point (interpolated from the grid).
    pub gradient: VelocityGradient,
}

impl CriticalPoint {
    /// How sharply the flow changes around this point: `√|det J|`, the
    /// geometric mean of the eigenvalue sizes, in 1/time.
    pub fn strength(&self) -> f64 {
        self.gradient.determinant().abs().sqrt()
    }

    /// Whether this point stands out from measurement noise.
    ///
    /// Where the fluid barely moves (a dead zone, or inside a solid object),
    /// noise makes the velocity cross zero at random. That creates "critical
    /// points" with tiny gradients. We compare the point's strength with
    /// the field's typical gradient, `gradient_scale` (see
    /// [`gradient_scale`]), and keep it if it is at least `min_ratio` of that.
    /// A ratio of 0.2 separates real features from noise well in the
    /// synthetic tests.
    pub fn is_significant(&self, gradient_scale: f64, min_ratio: f64) -> bool {
        self.strength() >= min_ratio * gradient_scale
    }
}

/// Typical velocity gradient of a field: its 95th-percentile speed divided
/// by the grid spacing. A flow feature one grid cell across, moving at
/// typical speed, has a gradient of about this size.
pub fn gradient_scale(field: &VectorField) -> f64 {
    let Some((hx, hy)) = grid_spacing(field) else {
        return 0.0;
    };
    let mut speeds: Vec<f64> = field
        .vectors
        .iter()
        .filter(|v| v.status == VectorStatus::Valid)
        .map(|v| (v.u as f64).hypot(v.v as f64))
        .collect();
    if speeds.is_empty() {
        return 0.0;
    }
    speeds.sort_by(f64::total_cmp);
    let p95 = speeds[((speeds.len() as f64 * 0.95) as usize).min(speeds.len() - 1)];
    p95 / hx.max(hy)
}

/// |T| / (2√D) below this counts as a centre. Measured fields always have a
/// little divergence noise, so a perfect T = 0 never occurs.
const CENTER_DAMPING: f64 = 0.2;
/// |T| / (2√D) at or above this counts as a node. For exactly 1 the two
/// eigenvalues are equal (a "star" node, like a symmetric source). Noise
/// would push such a node randomly into the "focus" side, so we leave a
/// margin: a spiral too weak to see within 10 % is reported as a node.
const NODE_DAMPING: f64 = 0.9;
/// |D| below this fraction of |J|² is treated as zero (degenerate).
const DEGENERATE_DET: f64 = 1e-3;

/// Classify the flow pattern around a critical point with gradient `g`.
pub fn classify(g: &VelocityGradient) -> CriticalPointKind {
    let t = g.divergence();
    let d = g.determinant();
    let size2 = g.dudx * g.dudx + g.dudy * g.dudy + g.dvdx * g.dvdx + g.dvdy * g.dvdy;
    if size2 == 0.0 || d.abs() < DEGENERATE_DET * size2 {
        return CriticalPointKind::Degenerate;
    }
    if d < 0.0 {
        return CriticalPointKind::Saddle; // λ₁λ₂ < 0: real, opposite signs
    }
    // D > 0. Eigenvalues are complex when T² < 4D, i.e. when the "damping
    // ratio" ζ = |T| / (2√D) is below 1.
    let zeta = t.abs() / (2.0 * d.sqrt());
    let outward = t > 0.0;
    match (zeta, outward) {
        (z, _) if z < CENTER_DAMPING => CriticalPointKind::Center,
        (z, true) if z < NODE_DAMPING => CriticalPointKind::UnstableFocus,
        (z, false) if z < NODE_DAMPING => CriticalPointKind::StableFocus,
        (_, true) => CriticalPointKind::UnstableNode,
        (_, false) => CriticalPointKind::StableNode,
    }
}

/// Find and classify every critical point in a field.
///
/// The velocity is only known at grid points, so we treat each grid cell
/// (four neighbouring vectors) as a small patch where u and v vary
/// *bilinearly*. A zero can only lie in a cell where both u and v change
/// sign among the corners. In those cells, Newton's method finds the exact
/// zero of the bilinear patch, and the gradient there is interpolated from
/// the four corner gradients.
///
/// In regions with almost no flow, measurement noise also crosses zero, so
/// such regions can produce spurious points. Callers may want to ignore
/// points where the gradient is tiny compared to the flow.
pub fn find_critical_points(field: &VectorField) -> Vec<CriticalPoint> {
    let mut points: Vec<CriticalPoint> = Vec::new();
    let Some((hx, hy)) = grid_spacing(field) else {
        return points;
    };
    let grads = velocity_gradients(field);

    for row in 0..field.rows - 1 {
        for col in 0..field.cols - 1 {
            // Corners in the order (0,0), (1,0), (0,1), (1,1) of the cell.
            let corners = [
                (col, row),
                (col + 1, row),
                (col, row + 1),
                (col + 1, row + 1),
            ];
            if corners
                .iter()
                .any(|&(c, r)| field.get(c, r).status != VectorStatus::Valid)
            {
                continue;
            }
            let u = corners.map(|(c, r)| field.get(c, r).u as f64);
            let v = corners.map(|(c, r)| field.get(c, r).v as f64);
            if !changes_sign(&u) || !changes_sign(&v) {
                continue;
            }
            let Some((s, t)) = bilinear_zero(&u, &v) else {
                continue;
            };
            let Some(g) = interpolate_gradient(corners.map(|(c, r)| grads.get(c, r)), s, t) else {
                continue;
            };
            let origin = field.get(col, row);
            let x = origin.x as f64 + s * hx;
            let y = origin.y as f64 + t * hy;
            // A zero exactly on a shared edge or corner is found by every
            // cell that touches it. Keep only the first.
            let duplicate = points
                .iter()
                .any(|p| (p.x - x).abs() < 0.5 * hx && (p.y - y).abs() < 0.5 * hy);
            if !duplicate {
                points.push(CriticalPoint {
                    x,
                    y,
                    kind: classify(&g),
                    gradient: g,
                });
            }
        }
    }
    points
}

/// True if the four values are not all strictly positive or all strictly
/// negative. Zero counts as a sign change, so a zero exactly on a grid
/// point is found.
fn changes_sign(values: &[f64; 4]) -> bool {
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    min <= 0.0 && max >= 0.0
}

/// Value of the bilinear patch through corners `f = [f00, f10, f01, f11]`
/// at local coordinates (s, t) ∈ [0, 1]², written as
/// `a + b·s + c·t + d·s·t`. Returns (value, ∂/∂s, ∂/∂t).
fn bilinear(f: &[f64; 4], s: f64, t: f64) -> (f64, f64, f64) {
    let a = f[0];
    let b = f[1] - f[0];
    let c = f[2] - f[0];
    let d = f[3] - f[1] - f[2] + f[0];
    (a + b * s + c * t + d * s * t, b + d * t, c + d * s)
}

/// Newton's method for u(s, t) = v(s, t) = 0 inside the unit cell.
fn bilinear_zero(u: &[f64; 4], v: &[f64; 4]) -> Option<(f64, f64)> {
    let scale = u.iter().chain(v).fold(0.0f64, |m, x| m.max(x.abs()));
    if scale == 0.0 {
        return Some((0.5, 0.5)); // all four corners are zero: say the middle
    }
    let (mut s, mut t) = (0.5, 0.5);
    for _ in 0..30 {
        let (fu, us, ut) = bilinear(u, s, t);
        let (fv, vs, vt) = bilinear(v, s, t);
        if fu.abs().max(fv.abs()) < 1e-10 * scale {
            // Allow a hair outside [0, 1] for zeros exactly on an edge.
            let inside = |z: f64| (-1e-6..=1.0 + 1e-6).contains(&z);
            return (inside(s) && inside(t)).then_some((s.clamp(0.0, 1.0), t.clamp(0.0, 1.0)));
        }
        // Newton step: solve the 2×2 system [us ut; vs vt]·δ = −(fu, fv)
        // with Cramer's rule.
        let det = us * vt - ut * vs;
        if det.abs() < 1e-300 {
            return None;
        }
        s -= (fu * vt - ut * fv) / det;
        t -= (us * fv - fu * vs) / det;
        if !s.is_finite() || !t.is_finite() || s.abs() > 10.0 || t.abs() > 10.0 {
            return None; // ran far away: no zero in this cell
        }
    }
    None
}

/// Bilinearly interpolate the four corner gradients at (s, t).
fn interpolate_gradient(
    g: [Option<VelocityGradient>; 4],
    s: f64,
    t: f64,
) -> Option<VelocityGradient> {
    let [g00, g10, g01, g11] = [g[0]?, g[1]?, g[2]?, g[3]?];
    let w = [(1.0 - s) * (1.0 - t), s * (1.0 - t), (1.0 - s) * t, s * t];
    let mix = |f: fn(&VelocityGradient) -> f64| {
        w[0] * f(&g00) + w[1] * f(&g10) + w[2] * f(&g01) + w[3] * f(&g11)
    };
    Some(VelocityGradient {
        dudx: mix(|g| g.dudx),
        dudy: mix(|g| g.dudy),
        dvdx: mix(|g| g.dvdx),
        dvdy: mix(|g| g.dvdy),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flow::{Source, Uniform, Vortex};
    use CriticalPointKind::*;

    fn grad(dudx: f64, dudy: f64, dvdx: f64, dvdy: f64) -> VelocityGradient {
        VelocityGradient {
            dudx,
            dudy,
            dvdx,
            dvdy,
        }
    }

    #[test]
    fn classification_of_textbook_matrices() {
        assert_eq!(classify(&grad(1.0, 0.0, 0.0, -1.0)), Saddle);
        assert_eq!(classify(&grad(0.0, -1.0, 1.0, 0.0)), Center);
        assert_eq!(classify(&grad(-0.5, -1.0, 1.0, -0.5)), StableFocus);
        assert_eq!(classify(&grad(0.5, -1.0, 1.0, 0.5)), UnstableFocus);
        assert_eq!(classify(&grad(1.0, 0.0, 0.0, 2.0)), UnstableNode);
        assert_eq!(classify(&grad(1.0, 0.0, 0.0, 1.0)), UnstableNode); // star
        assert_eq!(classify(&grad(-1.0, 0.0, 0.0, -1.0)), StableNode);
        assert_eq!(classify(&grad(0.0, 1.0, 0.0, 0.0)), Degenerate); // shear
        assert_eq!(classify(&grad(0.0, 0.0, 0.0, 0.0)), Degenerate);
    }

    /// Sample `flow` on a 25×25 grid, 8 px apart, starting at (4, 4). The
    /// grid is offset so the singular points never land on grid nodes.
    fn sample<F: crate::FlowElement>(flow: &F) -> VectorField {
        VectorField::sample(flow, 4.0, 4.0, 8.0, 25, 25)
    }

    fn assert_single(points: &[CriticalPoint], kind: CriticalPointKind, x: f64, y: f64) {
        assert_eq!(points.len(), 1, "expected one point, got {points:?}");
        let p = points[0];
        assert_eq!(p.kind, kind);
        assert!(
            (p.x - x).abs() < 0.5 && (p.y - y).abs() < 0.5,
            "at ({}, {})",
            p.x,
            p.y
        );
    }

    #[test]
    fn vortex_is_a_center_source_unstable_node_sink_stable_node() {
        let (x, y) = (101.3, 97.6);
        assert_single(
            &find_critical_points(&sample(&Vortex::rankine(x, y, 300.0, 30.0))),
            Center,
            x,
            y,
        );
        assert_single(
            &find_critical_points(&sample(&Source::rankine(x, y, 300.0, 30.0))),
            UnstableNode,
            x,
            y,
        );
        assert_single(
            &find_critical_points(&sample(&Source::rankine(x, y, -300.0, 30.0))),
            StableNode,
            x,
            y,
        );
    }

    #[test]
    fn vortex_plus_sink_spirals_inwards() {
        // A drain: rotation plus inflow. Here ζ = m/Γ = 0.5, a clear spiral.
        let drain = (
            Vortex::rankine(90.0, 110.0, 300.0, 30.0),
            Source::rankine(90.0, 110.0, -150.0, 30.0),
        );
        assert_single(
            &find_critical_points(&sample(&drain)),
            StableFocus,
            90.0,
            110.0,
        );
    }

    #[test]
    fn two_corotating_vortices_have_a_saddle_between_them() {
        let pair = (
            Vortex::rankine(60.3, 100.2, 200.0, 12.0),
            Vortex::rankine(140.3, 100.2, 200.0, 12.0),
        );
        let points = find_critical_points(&sample(&pair));
        let count = |k| points.iter().filter(|p| p.kind == k).count();
        assert_eq!(count(Center), 2, "{points:?}");
        assert_eq!(count(Saddle), 1, "{points:?}");
        let saddle = points.iter().find(|p| p.kind == Saddle).unwrap();
        assert!((saddle.x - 100.3).abs() < 0.5 && (saddle.y - 100.2).abs() < 0.5);
    }

    #[test]
    fn noise_points_in_still_regions_are_not_significant() {
        // A vortex in a field that is otherwise still, plus a tiny ±0.01
        // wobble that creates spurious zero crossings far from the vortex.
        let mut field = sample(&Vortex::rankine(101.3, 97.6, 300.0, 30.0));
        for (i, v) in field.vectors.iter_mut().enumerate() {
            let far = (v.x - 101.3).hypot(v.y - 97.6) > 80.0;
            if far {
                let wobble = if (i * 7919) % 3 == 0 { 0.01 } else { -0.01 };
                v.u = wobble;
                v.v = -wobble * 0.7;
            }
        }
        let scale = gradient_scale(&field);
        let points = find_critical_points(&field);
        assert!(points.len() > 1, "the wobble should create spurious points");
        let strong: Vec<_> = points
            .iter()
            .filter(|p| p.is_significant(scale, 0.2))
            .collect();
        assert_eq!(strong.len(), 1, "{strong:?}");
        assert_eq!(strong[0].kind, Center);
    }

    #[test]
    fn uniform_flow_has_no_critical_points() {
        assert!(find_critical_points(&sample(&Uniform { u: 1.0, v: 0.3 })).is_empty());
    }
}
