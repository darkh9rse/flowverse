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

    #[test]
    fn a_pair_of_elements_adds_velocities() {
        let pair = (Uniform { u: 1.0, v: 0.0 }, Uniform { u: 0.5, v: -2.0 });
        assert_eq!(pair.velocity(3.0, 4.0), (1.5, -2.0));
    }
}
