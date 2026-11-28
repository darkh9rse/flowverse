//! Analytical 2-D potential-flow elements.
//!
//! Potential flow (inviscid, incompressible, irrotational) has a useful
//! property: the equations are *linear*, so the velocity of several elements
//! together is just the sum of their individual velocities. A few simple
//! building blocks can therefore describe many real flows, and fitting
//! measured PIV data to a sum of them is how FlowVerse will "classify" a flow.
//!
//! The formulas below are textbook (e.g. Anderson, *Fundamentals of
//! Aerodynamics*, ch. 3), written in the crate's image coordinates: `x`
//! right, `y` **down**. Because `y` is flipped compared to a maths textbook,
//! anything "counter-clockwise" in the textbook appears **clockwise on
//! screen**.

use std::f64::consts::PI;

/// Anything that can report a velocity at any point in the plane.
///
/// This is a *trait*: a set of methods a type promises to provide (similar to
/// an interface in other languages). PIV tests, the synthetic image generator
/// and later the flow fitter all work with "any `FlowElement`", so they don't
/// care which concrete flow they get.
pub trait FlowElement {
    /// Velocity `(u, v)` at the point `(x, y)`.
    fn velocity(&self, x: f64, y: f64) -> (f64, f64);
}

/// Below this squared distance from a singular point, elements return zero
/// velocity instead of infinity or NaN. Real PIV never resolves that close to
/// a singularity anyway.
const SINGULAR_R2: f64 = 1e-12;

/// Constant velocity everywhere: `(u, v)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Uniform {
    pub u: f64,
    pub v: f64,
}

impl FlowElement for Uniform {
    fn velocity(&self, _x: f64, _y: f64) -> (f64, f64) {
        // The leading underscore in `_x` tells the compiler "unused on
        // purpose", which silences the warning.
        (self.u, self.v)
    }
}

/// A source (positive `strength`) or sink (negative) at `(x, y)`.
///
/// `strength` is the volume flow rate per unit depth, m. Outside
/// `core_radius` fluid moves radially outward with speed `m / (2π r)`.
///
/// Inside the core the source is spread out evenly over a disc, like water
/// welling up from a pipe of finite size. The speed there grows linearly,
/// `m r / (2π rc²)`, so the divergence is a constant `m / (π rc²)`. This
/// mirrors the Rankine vortex core, and it removes the infinite speed at the
/// centre that would wreck synthetic images. `core_radius = 0` gives the
/// ideal point source.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Source {
    pub x: f64,
    pub y: f64,
    pub strength: f64,
    pub core_radius: f64,
}

impl Source {
    /// Ideal point source or sink (no core).
    pub fn point(x: f64, y: f64, strength: f64) -> Self {
        Source {
            x,
            y,
            strength,
            core_radius: 0.0,
        }
    }

    /// Source or sink spread uniformly over a disc of radius `core_radius`.
    pub fn rankine(x: f64, y: f64, strength: f64, core_radius: f64) -> Self {
        Source {
            x,
            y,
            strength,
            core_radius,
        }
    }
}

impl FlowElement for Source {
    fn velocity(&self, x: f64, y: f64) -> (f64, f64) {
        let (dx, dy) = (x - self.x, y - self.y);
        let r2 = dx * dx + dy * dy;
        if r2 < SINGULAR_R2 {
            return (0.0, 0.0);
        }
        // Radial unit vector is (dx, dy)/r and speed is m/(2πr), so the
        // velocity is m/(2π) * (dx, dy)/r². That form needs no square root.
        // Inside the core, r² is replaced by rc² (speed ∝ r).
        let rc2 = self.core_radius * self.core_radius;
        let k = self.strength / (2.0 * PI * r2.max(rc2));
        (k * dx, k * dy)
    }
}

/// A vortex at `(x, y)` with circulation Γ (`circulation`).
///
/// Outside `core_radius` this is the ideal point vortex: tangential speed
/// `Γ / (2π r)`. Inside it is a *Rankine* core that rotates like a solid
/// body, with speed `Γ r / (2π rc²)`. The ideal vortex has infinite speed at
/// the centre, which is impossible in a real fluid and would wreck any
/// synthetic image, so tests use a finite core. `core_radius = 0` gives the
/// pure point vortex.
///
/// Positive Γ rotates counter-clockwise in textbook axes, which appears
/// **clockwise on screen** because `y` points down.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vortex {
    pub x: f64,
    pub y: f64,
    pub circulation: f64,
    pub core_radius: f64,
}

impl Vortex {
    /// Ideal point vortex (no core).
    pub fn point(x: f64, y: f64, circulation: f64) -> Self {
        Vortex {
            x,
            y,
            circulation,
            core_radius: 0.0,
        }
    }

    /// Rankine vortex with a solid-body core of radius `core_radius`.
    pub fn rankine(x: f64, y: f64, circulation: f64, core_radius: f64) -> Self {
        Vortex {
            x,
            y,
            circulation,
            core_radius,
        }
    }
}

impl FlowElement for Vortex {
    fn velocity(&self, x: f64, y: f64) -> (f64, f64) {
        let (dx, dy) = (x - self.x, y - self.y);
        let r2 = dx * dx + dy * dy;
        if r2 < SINGULAR_R2 {
            return (0.0, 0.0);
        }
        let rc2 = self.core_radius * self.core_radius;
        // The tangential direction is the radial one turned 90°: (-dy, dx)/r.
        // Speed/r is Γ/(2π r²) outside the core and Γ/(2π rc²) inside.
        // Using speed/r again avoids a square root.
        let k = if r2 < rc2 {
            self.circulation / (2.0 * PI * rc2)
        } else {
            self.circulation / (2.0 * PI * r2)
        };
        (-k * dy, k * dx)
    }
}

/// A doublet at `(x, y)`: a source and sink of equal strength pushed together.
/// It is oriented along +x, with `strength` κ.
///
/// Potential `φ = κ (x - x0) / (2π r²)`, which gives
/// `u = κ (dy² - dx²) / (2π r⁴)` and `v = -2κ dx dy / (2π r⁴)`.
/// A doublet plus a uniform stream gives the classic flow around a cylinder.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Doublet {
    pub x: f64,
    pub y: f64,
    pub strength: f64,
}

impl FlowElement for Doublet {
    fn velocity(&self, x: f64, y: f64) -> (f64, f64) {
        let (dx, dy) = (x - self.x, y - self.y);
        let r2 = dx * dx + dy * dy;
        if r2 < SINGULAR_R2 {
            return (0.0, 0.0);
        }
        let k = self.strength / (2.0 * PI * r2 * r2);
        (k * (dy * dy - dx * dx), -2.0 * k * dx * dy)
    }
}

/// Flow past a solid circular cylinder of radius `radius` centred at
/// `(x, y)`, in a free stream `(u, v)` far away.
///
/// In potential flow this is a uniform stream plus a doublet of strength
/// `κ = 2π U R²` pointing along the stream. The combination makes the
/// circle r = R a streamline: no fluid crosses it. Our [`Doublet`] only
/// points along +x, so we rotate into the stream's frame (ξ along the
/// stream, η across it), use the textbook formulas
/// `u' = U (1 - R² (ξ² - η²) / r⁴)`, `v' = -2 U R² ξ η / r⁴`, and rotate
/// back.
///
/// Inside the cylinder the velocity is zero, because it is solid. In a real
/// video that region has no tracer particles, or shows the object's
/// stationary texture.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cylinder {
    pub x: f64,
    pub y: f64,
    pub u: f64,
    pub v: f64,
    pub radius: f64,
}

impl FlowElement for Cylinder {
    fn velocity(&self, x: f64, y: f64) -> (f64, f64) {
        let (dx, dy) = (x - self.x, y - self.y);
        let r2 = dx * dx + dy * dy;
        let speed = self.u.hypot(self.v);
        if r2 < self.radius * self.radius || speed == 0.0 {
            return (0.0, 0.0);
        }
        if r2 < SINGULAR_R2 {
            return (self.u, self.v); // radius 0: no cylinder, just the stream
        }
        // (c, s) = cos and sin of the stream direction.
        let (c, s) = (self.u / speed, self.v / speed);
        let xi = dx * c + dy * s;
        let eta = -dx * s + dy * c;
        let a = self.radius * self.radius / (r2 * r2);
        let u_rot = speed * (1.0 - a * (xi * xi - eta * eta));
        let v_rot = -2.0 * speed * a * xi * eta;
        (u_rot * c - v_rot * s, u_rot * s + v_rot * c)
    }
}

// ---------------------------------------------------------------------------
// Viscous channel flows: exact solutions of the Navier–Stokes equations
// ---------------------------------------------------------------------------
//
// Everything above is *potential* flow: no viscosity, and fluid slips freely
// along walls. The two flows below are the opposite: viscosity is what
// shapes them. They are two of the very few cases where the full
// Navier–Stokes equations can be solved exactly:
//
// For steady flow between parallel walls, the velocity points along the walls
// and only changes across them, u = u(n). Then the nonlinear term
// (u·∇)u is zero, and Navier–Stokes reduces to
//
//     μ d²u/dn² = dp/ds            (s along the walls, n across them)
//
// so u(n) is at most a parabola. Two special cases:
// - **Couette**: no pressure gradient, one wall slides. d²u/dn² = 0, so the
//   profile is a straight line (pure shear).
// - **Poiseuille**: both walls still, a pressure drop pushes the fluid.
//   d²u/dn² is a negative constant, so the profile is a parabola, zero at
//   the walls (no-slip) and fastest on the centreline.
//
// Both are described by a line in the image (a wall for Couette, the
// centreline for Poiseuille) through `(x, y)` at `angle` radians from the
// +x axis. Because y points down, a positive angle turns *clockwise* on
// screen. The flow runs along `(cos angle, sin angle)`, and the signed
// distance across the line is `n = −(x−x0)·sin(angle) + (y−y0)·cos(angle)`.

/// Signed distance `n` of `(x, y)` from the line through `(x0, y0)` at
/// `angle`, and the unit vector `(c, s)` along it.
fn across_line(x0: f64, y0: f64, angle: f64, x: f64, y: f64) -> (f64, f64, f64) {
    let (s, c) = angle.sin_cos();
    (-(x - x0) * s + (y - y0) * c, c, s)
}

/// Plane Couette flow: fluid sheared between a still wall and a sliding one.
///
/// The still wall is the line through `(x, y)` at `angle`. The velocity
/// points along the wall and grows linearly with distance `n` from it:
/// `u = shear_rate · n`. (With the moving wall a distance H away sliding at
/// speed U, `shear_rate = U / H`.) On the other side of the line (n < 0) the
/// fluid moves backwards, which is the same straight-line profile continued.
///
/// The shear rate is also the flow's vorticity magnitude (`−shear_rate` in
/// this crate's sign convention), the same everywhere: Couette flow
/// rotates fluid elements even though every particle moves in a straight
/// line. The wall shear stress is `μ · shear_rate`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Couette {
    pub x: f64,
    pub y: f64,
    pub angle: f64,
    pub shear_rate: f64,
}

impl FlowElement for Couette {
    fn velocity(&self, x: f64, y: f64) -> (f64, f64) {
        let (n, c, s) = across_line(self.x, self.y, self.angle, x, y);
        let speed = self.shear_rate * n;
        (speed * c, speed * s)
    }
}

/// Plane Poiseuille flow: pressure-driven flow in a channel between two
/// still walls.
///
/// The centreline passes through `(x, y)` at `angle`; the walls are
/// `half_width` away on each side. Inside, the speed along the channel is
/// `max_speed · (1 − n² / half_width²)`: a parabola that is zero at both
/// walls (no-slip) and largest on the centreline. Outside the walls the
/// velocity is zero (solid).
///
/// Useful facts that follow (h = half_width, U = max_speed, μ = viscosity):
/// - mean speed across the channel: 2U/3
/// - shear rate at the walls: 2U/h
/// - pressure gradient driving it: dp/ds = −2μU/h²
///
/// A negative `max_speed` means flow in the `−angle` direction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Poiseuille {
    pub x: f64,
    pub y: f64,
    pub angle: f64,
    pub half_width: f64,
    pub max_speed: f64,
}

impl Poiseuille {
    /// Average speed over the channel width, `2/3` of the peak.
    pub fn mean_speed(&self) -> f64 {
        2.0 * self.max_speed / 3.0
    }

    /// Velocity gradient at the walls, `2 U / h`. Multiply by the viscosity
    /// to get the wall shear stress.
    pub fn wall_shear_rate(&self) -> f64 {
        2.0 * self.max_speed / self.half_width
    }
}

impl FlowElement for Poiseuille {
    fn velocity(&self, x: f64, y: f64) -> (f64, f64) {
        let (n, c, s) = across_line(self.x, self.y, self.angle, x, y);
        let h = self.half_width;
        if h <= 0.0 || n.abs() >= h {
            return (0.0, 0.0); // inside a wall
        }
        let speed = self.max_speed * (1.0 - (n / h).powi(2));
        (speed * c, speed * s)
    }
}

/// A plane jet: a stream of fluid shooting into still surroundings, with
/// no walls (e.g. water from a slot nozzle into a tank).
///
/// The centreline passes through `(x, y)` at `angle` (same convention as
/// [`Poiseuille`]). Across the jet, the speed along it is
///
/// ```text
/// u(n) = U/2 · [ tanh((n + h)/δ) − tanh((n − h)/δ) ]
/// ```
///
/// with `U = max_speed`, `h = half_width` and `δ = shear_thickness`. Each
/// `tanh` is one **shear layer**: a thin zone, about 2δ thick, where the
/// fast jet rubs against the still fluid. Between them the core moves at
/// almost U when h ≫ δ (a "top-hat" jet, typical right after a nozzle).
/// When δ is comparable to h the profile becomes a smooth bell, like a jet
/// far downstream (the Bickley jet, u ∝ sech²).
///
/// This is a *model* profile (Michalke 1965), not an exact Navier–Stokes
/// solution: a real jet slowly spreads and slows down as it mixes. Over
/// the short stretch a video shows, it describes the shape well, and it
/// gives useful numbers:
/// - flow rate per unit depth Q = ∫u dn = 2hU, exactly, for any δ;
/// - peak vorticity in the shear layers, U/(2δ) (for h ≫ δ);
/// - the shear layers are unstable (Kelvin–Helmholtz): the fastest-growing
///   ripples have wavelength ≈ 14.1 δ. They roll up into the vortices often
///   seen along the edges of a jet.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Jet {
    pub x: f64,
    pub y: f64,
    pub angle: f64,
    pub half_width: f64,
    pub shear_thickness: f64,
    pub max_speed: f64,
}

/// Wavelength of the fastest-growing Kelvin–Helmholtz wave of a
/// `tanh(n/δ)` shear layer, in units of δ: 2π / 0.4446 (Michalke 1964,
/// temporal stability of the hyperbolic-tangent mixing layer).
pub const KH_WAVELENGTH_PER_THICKNESS: f64 = 2.0 * PI / 0.4446;

impl Jet {
    /// Volume flow rate per unit depth, `2 h U`.
    pub fn flow_rate(&self) -> f64 {
        2.0 * self.half_width * self.max_speed
    }

    /// Largest vorticity magnitude in the shear layers, `U / (2δ)`
    /// (accurate when the jet is much wider than its shear layers).
    pub fn shear_layer_vorticity(&self) -> f64 {
        self.max_speed / (2.0 * self.shear_thickness)
    }

    /// Expected spacing of the Kelvin–Helmholtz vortices, ≈ 14.1 δ.
    pub fn kh_wavelength(&self) -> f64 {
        KH_WAVELENGTH_PER_THICKNESS * self.shear_thickness
    }
}

impl FlowElement for Jet {
    fn velocity(&self, x: f64, y: f64) -> (f64, f64) {
        let (n, c, s) = across_line(self.x, self.y, self.angle, x, y);
        // δ → 0 is a sharp top-hat; keep it finite so we never divide by 0.
        let d = self.shear_thickness.abs().max(1e-9);
        let h = self.half_width;
        let speed = 0.5 * self.max_speed * (((n + h) / d).tanh() - ((n - h) / d).tanh());
        (speed * c, speed * s)
    }
}

/// Superposition: a pair of flow elements is itself a flow element whose
/// velocity is the sum. Pairs can be nested to combine more than two, e.g.
/// `(Uniform {..}, (source, sink))`.
///
/// This is a *generic* `impl`: one block of code that works for every pair
/// of types `A` and `B` that are both flow elements.
impl<A: FlowElement, B: FlowElement> FlowElement for (A, B) {
    fn velocity(&self, x: f64, y: f64) -> (f64, f64) {
        let (u1, v1) = self.0.velocity(x, y);
        let (u2, v2) = self.1.velocity(x, y);
        (u1 + u2, v1 + v2)
    }
}

/// How far a fluid particle starting at `(x, y)` travels in time `dt`.
///
/// This is **not** simply `velocity * dt`. In a vortex the particle follows a
/// curved path and the velocity changes along the way. We integrate the path
/// with the classic 4th-order Runge–Kutta method (RK4), split into
/// `RK4_STEPS` small steps. The synthetic image generator uses this to move
/// particles, and the tests use it as the "true" answer that PIV should
/// recover. PIV can only measure where a particle *ended up*, not its
/// instantaneous velocity.
///
/// `F: FlowElement + ?Sized` means "any type that implements FlowElement".
/// `?Sized` also allows trait objects (`&dyn FlowElement`), whose size isn't
/// known at compile time.
pub fn displacement<F: FlowElement + ?Sized>(flow: &F, x: f64, y: f64, dt: f64) -> (f64, f64) {
    const RK4_STEPS: usize = 16;
    let h = dt / RK4_STEPS as f64;
    let (mut px, mut py) = (x, y);
    for _ in 0..RK4_STEPS {
        // RK4 samples the velocity at four points across the step and
        // averages them with weights 1, 2, 2, 1. Its error per step is
        // proportional to h⁵, far below anything PIV can detect.
        let (k1x, k1y) = flow.velocity(px, py);
        let (k2x, k2y) = flow.velocity(px + 0.5 * h * k1x, py + 0.5 * h * k1y);
        let (k3x, k3y) = flow.velocity(px + 0.5 * h * k2x, py + 0.5 * h * k2y);
        let (k4x, k4y) = flow.velocity(px + h * k3x, py + h * k3y);
        px += h / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x);
        py += h / 6.0 * (k1y + 2.0 * k2y + 2.0 * k3y + k4y);
    }
    (px - x, py - y)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: are two numbers equal within an absolute tolerance?
    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn uniform_is_the_same_everywhere() {
        let f = Uniform { u: 1.5, v: -0.5 };
        assert_eq!(f.velocity(0.0, 0.0), (1.5, -0.5));
        assert_eq!(f.velocity(-300.0, 42.0), (1.5, -0.5));
    }

    #[test]
    fn source_flows_radially_with_speed_m_over_2_pi_r() {
        let s = Source::point(10.0, 20.0, 6.0);
        // A point 3 right and 4 down of the source: r = 5.
        let (u, v) = s.velocity(13.0, 24.0);
        let expected_speed = 6.0 / (2.0 * PI * 5.0);
        assert!(close(u.hypot(v), expected_speed, 1e-12));
        // Direction is outward along (3, 4)/5.
        assert!(close(u, expected_speed * 0.6, 1e-12));
        assert!(close(v, expected_speed * 0.8, 1e-12));
        // A sink (negative strength) points inward.
        let sink = Source {
            strength: -6.0,
            ..s
        };
        let (us, vs) = sink.velocity(13.0, 24.0);
        assert!(close(us, -u, 1e-12) && close(vs, -v, 1e-12));
    }

    #[test]
    fn point_vortex_is_tangential_with_speed_gamma_over_2_pi_r() {
        let w = Vortex::point(0.0, 0.0, 10.0);
        let (u, v) = w.velocity(4.0, 0.0);
        // Right of the centre, positive Γ points along +y (down on screen).
        assert!(close(u, 0.0, 1e-12));
        assert!(close(v, 10.0 / (2.0 * PI * 4.0), 1e-12));
        // Tangential means perpendicular to the radius: dot product is zero.
        let (u2, v2) = w.velocity(3.0, -7.0);
        assert!(close(u2 * 3.0 + v2 * -7.0, 0.0, 1e-12));
    }

    #[test]
    fn rankine_core_rotates_as_a_solid_body_and_matches_at_the_edge() {
        let rc = 5.0;
        let w = Vortex::rankine(0.0, 0.0, 2.0 * PI * rc, rc); // speed 1 at r = rc
        // Inside: speed grows linearly with r.
        let (_, v_half) = w.velocity(2.5, 0.0);
        assert!(close(v_half, 0.5, 1e-12));
        // At and just past the edge both formulas give speed 1, with no jump.
        let (_, v_edge) = w.velocity(rc, 0.0);
        assert!(close(v_edge, 1.0, 1e-12));
        let (_, v_out) = w.velocity(rc + 1e-9, 0.0);
        assert!(close(v_out, 1.0, 1e-6));
    }

    #[test]
    fn doublet_velocity_is_the_gradient_of_its_potential() {
        let d = Doublet {
            x: 1.0,
            y: -2.0,
            strength: 7.0,
        };
        // Analytical potential φ = κ dx / (2π r²).
        let phi = |x: f64, y: f64| {
            let (dx, dy) = (x - 1.0, y + 2.0);
            7.0 * dx / (2.0 * PI * (dx * dx + dy * dy))
        };
        // Compare u = ∂φ/∂x and v = ∂φ/∂y against central finite differences.
        let h = 1e-5;
        for &(x, y) in &[(4.0, 1.0), (-3.0, 0.5), (1.5, -6.0)] {
            let (u, v) = d.velocity(x, y);
            let du = (phi(x + h, y) - phi(x - h, y)) / (2.0 * h);
            let dv = (phi(x, y + h) - phi(x, y - h)) / (2.0 * h);
            assert!(close(u, du, 1e-7), "u at ({x},{y}): {u} vs {du}");
            assert!(close(v, dv, 1e-7), "v at ({x},{y}): {v} vs {dv}");
        }
    }

    #[test]
    fn singular_points_give_zero_not_nan() {
        assert_eq!(Source::point(0.0, 0.0, 1.0).velocity(0.0, 0.0), (0.0, 0.0));
        assert_eq!(Vortex::point(0.0, 0.0, 1.0).velocity(0.0, 0.0), (0.0, 0.0));
        assert_eq!(
            Doublet {
                x: 0.0,
                y: 0.0,
                strength: 1.0
            }
            .velocity(0.0, 0.0),
            (0.0, 0.0)
        );
    }

    #[test]
    fn displacement_in_uniform_flow_is_velocity_times_dt() {
        let (dx, dy) = displacement(&Uniform { u: 2.0, v: -3.0 }, 5.0, 5.0, 0.5);
        assert!(close(dx, 1.0, 1e-12) && close(dy, -1.5, 1e-12));
    }

    #[test]
    fn displacement_in_solid_body_rotation_follows_the_circle() {
        // Inside a Rankine core the angular speed is ω = Γ / (2π rc²).
        let rc = 50.0;
        let gamma = 2.0 * PI * rc * 5.0; // 5 px per unit time at r = rc
        let w = Vortex::rankine(0.0, 0.0, gamma, rc);
        let omega = gamma / (2.0 * PI * rc * rc);
        // Start at (r, 0): after dt the particle is at r(cos ωdt, sin ωdt).
        let r = 30.0;
        let dt = 1.0;
        let (dx, dy) = displacement(&w, r, 0.0, dt);
        let ex = r * (omega * dt).cos() - r;
        let ey = r * (omega * dt).sin();
        assert!(close(dx, ex, 1e-9), "{dx} vs {ex}");
        assert!(close(dy, ey, 1e-9), "{dy} vs {ey}");
    }

    #[test]
    fn cored_source_has_constant_divergence_inside_and_matches_outside() {
        let rc = 10.0;
        let m = 2.0 * PI * rc * 3.0; // speed 3 at the core edge
        let s = Source::rankine(0.0, 0.0, m, rc);
        assert!(close(s.velocity(5.0, 0.0).0, 1.5, 1e-12)); // linear inside
        assert!(close(s.velocity(rc, 0.0).0, 3.0, 1e-12));
        assert!(close(s.velocity(20.0, 0.0).0, 1.5, 1e-12)); // 1/r outside
        // Divergence ∂u/∂x + ∂v/∂y = m / (π rc²) inside the core.
        let h = 1e-4;
        let div = (s.velocity(2.0 + h, 1.0).0 - s.velocity(2.0 - h, 1.0).0
            + s.velocity(2.0, 1.0 + h).1
            - s.velocity(2.0, 1.0 - h).1)
            / (2.0 * h);
        assert!(close(div, m / (PI * rc * rc), 1e-6));
    }

    #[test]
    fn cylinder_matches_uniform_plus_doublet_and_has_no_flow_through_wall() {
        let (big_u, r) = (2.0, 5.0);
        let cyl = Cylinder {
            x: 1.0,
            y: 2.0,
            u: big_u,
            v: 0.0,
            radius: r,
        };
        let pair = (
            Uniform { u: big_u, v: 0.0 },
            Doublet {
                x: 1.0,
                y: 2.0,
                strength: 2.0 * PI * big_u * r * r,
            },
        );
        for &(x, y) in &[(9.0, 2.0), (-6.0, 7.0), (1.0, -10.0)] {
            let (a, b) = (cyl.velocity(x, y), pair.velocity(x, y));
            assert!(close(a.0, b.0, 1e-12) && close(a.1, b.1, 1e-12));
        }
        // Rotated stream: on the wall the velocity is tangential, and at the
        // sides (90° from the stream) the speed is 2U.
        let cyl = Cylinder {
            x: 0.0,
            y: 0.0,
            u: 1.2,
            v: 1.6,
            radius: r,
        };
        for k in 0..12 {
            let t = k as f64 * PI / 6.0 + 0.1;
            let (px, py) = (r * t.cos() * 1.000001, r * t.sin() * 1.000001);
            let (u, v) = cyl.velocity(px, py);
            assert!(
                close(u * t.cos() + v * t.sin(), 0.0, 1e-5),
                "radial flow at {t}"
            );
        }
        let side = cyl.velocity(-1.6 / 2.0 * r * 1.000001, 1.2 / 2.0 * r * 1.000001);
        assert!(close(side.0.hypot(side.1), 4.0, 1e-4));
        assert_eq!(cyl.velocity(1.0, 1.0), (0.0, 0.0)); // inside: solid
    }

    /// Second derivative of the along-flow speed across the line, by central
    /// differences. For a channel flow Navier–Stokes says μ·d²u/dn² = dp/ds,
    /// a constant, so this tells us the pressure gradient (divided by μ).
    fn profile_curvature<F: FlowElement>(f: &F, angle: f64, x: f64, y: f64) -> f64 {
        let (s, c) = angle.sin_cos();
        let along = |n: f64| {
            // Step a distance n across the line (normal = (−sin, cos)).
            let (u, v) = f.velocity(x - n * s, y + n * c);
            u * c + v * s
        };
        let d = 0.5;
        (along(d) - 2.0 * along(0.0) + along(-d)) / (d * d)
    }

    #[test]
    fn couette_is_linear_shear_with_no_slip_at_the_wall() {
        let angle = 0.3;
        let f = Couette {
            x: 50.0,
            y: 40.0,
            angle,
            shear_rate: 0.02,
        };
        let (s, c) = angle.sin_cos();
        // On the wall: no-slip, zero velocity.
        let on_wall = f.velocity(50.0 + 30.0 * c, 40.0 + 30.0 * s);
        assert!(close(on_wall.0, 0.0, 1e-12) && close(on_wall.1, 0.0, 1e-12));
        // 25 px away from the wall: speed 0.02 × 25 = 0.5, along the wall.
        let (u, v) = f.velocity(50.0 - 25.0 * s, 40.0 + 25.0 * c);
        assert!(close(u, 0.5 * c, 1e-12) && close(v, 0.5 * s, 1e-12));
        // No pressure gradient: the profile has no curvature anywhere.
        for (x, y) in [(10.0, 90.0), (80.0, 20.0)] {
            assert!(close(profile_curvature(&f, angle, x, y), 0.0, 1e-9));
        }
    }

    #[test]
    fn poiseuille_is_a_parabola_between_still_walls() {
        let angle = -0.4;
        let f = Poiseuille {
            x: 100.0,
            y: 80.0,
            angle,
            half_width: 40.0,
            max_speed: 1.5,
        };
        let (s, c) = angle.sin_cos();
        let at = |n: f64| f.velocity(100.0 - n * s, 80.0 + n * c);
        // Centreline: fastest, along the channel.
        let (u, v) = at(0.0);
        assert!(close(u, 1.5 * c, 1e-12) && close(v, 1.5 * s, 1e-12));
        // Halfway to a wall: 1 − (1/2)² = 3/4 of the peak.
        let (u, v) = at(20.0);
        assert!(close(u.hypot(v), 1.125, 1e-12));
        // At and beyond both walls: still (no-slip, then solid).
        for n in [40.0, -40.0, 55.0] {
            assert_eq!(at(n), (0.0, 0.0));
        }
        // Navier–Stokes: d²u/dn² = dp/ds / μ = −2U/h², the same everywhere
        // inside the channel.
        let expected = -2.0 * 1.5 / (40.0 * 40.0);
        for n in [-30.0, 0.0, 12.0] {
            let k = profile_curvature(&f, angle, 100.0 - n * s, 80.0 + n * c);
            assert!(close(k, expected, 1e-9), "curvature {k} at n = {n}");
        }
        assert!(close(f.mean_speed(), 1.0, 1e-12));
        assert!(close(f.wall_shear_rate(), 0.075, 1e-12));
    }

    #[test]
    fn jet_has_a_fast_core_shear_layers_and_still_surroundings() {
        let angle = 0.25;
        let jet = Jet {
            x: 0.0,
            y: 0.0,
            angle,
            half_width: 50.0,
            shear_thickness: 5.0,
            max_speed: 2.0,
        };
        let (s, c) = angle.sin_cos();
        let along = |n: f64| {
            let (u, v) = jet.velocity(-n * s, n * c);
            u * c + v * s
        };
        // Core: essentially U. Shear layer centres (n = ±h): U/2. Far
        // outside: still.
        assert!(close(along(0.0), 2.0, 1e-6));
        assert!(close(along(50.0), 1.0, 1e-3) && close(along(-50.0), 1.0, 1e-3));
        assert!(close(along(120.0), 0.0, 1e-9));
        // Flow rate: the integral of u across the jet is exactly 2hU.
        let dn = 0.05;
        let q: f64 = (-4000..4000).map(|i| along(i as f64 * dn) * dn).sum();
        assert!(close(q, jet.flow_rate(), 1e-6), "Q = {q}");
        assert!(close(jet.flow_rate(), 200.0, 1e-12));
        // Steepest slope (= |vorticity|) at the shear layer: U/(2δ).
        let slope = (along(50.0 - 0.01) - along(50.0 + 0.01)) / 0.02;
        assert!(
            close(slope, jet.shear_layer_vorticity(), 1e-4),
            "slope {slope}"
        );
        assert!(close(jet.kh_wavelength(), 70.66, 0.01));
    }

    #[test]
    fn a_pair_of_elements_adds_velocities() {
        let pair = (Uniform { u: 1.0, v: 0.0 }, Uniform { u: 0.5, v: -2.0 });
        assert_eq!(pair.velocity(3.0, 4.0), (1.5, -2.0));
    }
}
