//! Synthetic particle images: fake PIV recordings with a known answer.
//!
//! Real PIV films tiny tracer particles lit by a light sheet. Each particle
//! shows up as a small, roughly Gaussian blob. Here we scatter random blobs
//! over frame A, move every particle with a known [`FlowElement`], and draw
//! them again as frame B. Running PIV on (A, B) should give back the flow we
//! put in. This is the standard way to test PIV software (see Raffel et al.,
//! *Particle Image Velocimetry: A Practical Guide*, ch. 6).

use crate::flow::{FlowElement, displacement};
use crate::image::Image;

/// Settings for [`generate_pair`].
#[derive(Debug, Clone, PartialEq)]
pub struct ParticleImageConfig {
    pub width: usize,
    pub height: usize,
    /// Mean number of particles per pixel. 0.02 puts about 20 particles in a
    /// 32×32 PIV window, comfortably above the ~10 that PIV needs to work well.
    pub particle_density: f64,
    /// Particle image diameter in pixels, measured where the brightness has
    /// fallen to e⁻² of the peak. 2–4 px is ideal for PIV. Smaller particles
    /// cause "peak locking", where results snap to whole pixels, because a
    /// sub-pixel fit needs the peak spread over 3 or more pixels.
    pub particle_diameter: f64,
    /// Peak brightness of one particle.
    pub intensity: f32,
    /// Time between frame A and frame B.
    pub dt: f64,
    /// Particles are seeded this many pixels beyond every image edge, so
    /// particles that flow *into* the frame between A and B exist in A too
    /// (just off-screen). Set it larger than the biggest displacement.
    pub margin: f64,
    /// Seed for the random generator. The same seed always gives the same
    /// images, which keeps tests reproducible.
    pub seed: u64,
}

impl Default for ParticleImageConfig {
    fn default() -> Self {
        ParticleImageConfig {
            width: 256,
            height: 256,
            particle_density: 0.02,
            particle_diameter: 3.0,
            intensity: 1.0,
            dt: 1.0,
            margin: 16.0,
            seed: 1,
        }
    }
}

/// Make an image pair (frame A, frame B) where every particle in B has been
/// moved by `flow` for time `cfg.dt`.
pub fn generate_pair<F: FlowElement + ?Sized>(
    flow: &F,
    cfg: &ParticleImageConfig,
) -> (Image, Image) {
    let mut rng = SplitMix64::new(cfg.seed);

    // Seed area: the image plus a margin on every side.
    let x_min = -cfg.margin;
    let y_min = -cfg.margin;
    let span_x = cfg.width as f64 + 2.0 * cfg.margin;
    let span_y = cfg.height as f64 + 2.0 * cfg.margin;
    let count = (cfg.particle_density * span_x * span_y).round() as usize;

    let mut frame_a = Image::new(cfg.width, cfg.height);
    let mut frame_b = Image::new(cfg.width, cfg.height);
    for _ in 0..count {
        let x = x_min + rng.next_f64() * span_x;
        let y = y_min + rng.next_f64() * span_y;
        let (dx, dy) = displacement(flow, x, y, cfg.dt);
        draw_particle(&mut frame_a, x, y, cfg.particle_diameter, cfg.intensity);
        draw_particle(
            &mut frame_b,
            x + dx,
            y + dy,
            cfg.particle_diameter,
            cfg.intensity,
        );
    }
    (frame_a, frame_b)
}

/// Add one Gaussian particle image centred at `(cx, cy)`:
/// `I(r) = I0 · exp(-8 r² / d²)`. The factor 8 makes the brightness equal
/// `I0 · e⁻²` at `r = d/2`, matching how `particle_diameter` is defined.
fn draw_particle(img: &mut Image, cx: f64, cy: f64, diameter: f64, intensity: f32) {
    // Beyond r = d the brightness is below e⁻⁸ ≈ 0.03 % of the peak, so we
    // only visit pixels within that box. Drawing every particle onto every
    // pixel would be thousands of times slower.
    let reach = diameter.max(1.0);
    let x0 = (cx - reach).floor().max(0.0) as i64;
    let y0 = (cy - reach).floor().max(0.0) as i64;
    let x1 = ((cx + reach).ceil() as i64).min(img.width() as i64 - 1);
    let y1 = ((cy + reach).ceil() as i64).min(img.height() as i64 - 1);
    // If the particle is entirely off-screen, x1 < x0 and the loop is empty.
    let k = 8.0 / (diameter * diameter);
    for py in y0..=y1 {
        for px in x0..=x1 {
            let ddx = px as f64 - cx;
            let ddy = py as f64 - cy;
            let value = intensity as f64 * (-k * (ddx * ddx + ddy * ddy)).exp();
            img.add(px as usize, py as usize, value as f32);
        }
    }
}

/// A tiny pseudo-random number generator (SplitMix64, by Sebastiano Vigna).
///
/// We write our own instead of using the `rand` crate because we only need
/// uniform numbers. It keeps the dependency list (and the WebAssembly
/// download) small, and the output is identical on every platform and
/// version, so test results never change under us.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        // `wrapping_add` / `wrapping_mul` overflow on purpose. Plain `+` and
        // `*` would panic on overflow in debug builds.
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1). The top 53 bits fill an f64's mantissa exactly.
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flow::Uniform;

    #[test]
    fn particle_peak_is_at_its_centre_with_full_intensity() {
        let mut img = Image::new(9, 9);
        draw_particle(&mut img, 4.0, 4.0, 3.0, 1.0);
        assert!((img.get(4, 4) - 1.0).abs() < 1e-6);
        // One pixel away (r = 1) the formula gives exp(-8 · 1² / 3²).
        let expected = (-8.0f64 / 9.0).exp() as f32;
        assert!((img.get(5, 4) - expected).abs() < 1e-6);
        assert!(img.get(0, 0) < 1e-6);
    }

    #[test]
    fn zero_flow_gives_identical_frames() {
        let (a, b) = generate_pair(&Uniform { u: 0.0, v: 0.0 }, &ParticleImageConfig::default());
        assert_eq!(a, b);
        assert!(
            a.data().iter().any(|&p| p > 0.5),
            "frame should contain particles"
        );
    }

    #[test]
    fn same_seed_same_images_different_seed_different_images() {
        let flow = Uniform { u: 1.0, v: 0.0 };
        let cfg = ParticleImageConfig::default();
        assert_eq!(generate_pair(&flow, &cfg), generate_pair(&flow, &cfg));
        let other = ParticleImageConfig {
            seed: 2,
            ..cfg.clone()
        };
        assert_ne!(generate_pair(&flow, &cfg).0, generate_pair(&flow, &other).0);
    }

    #[test]
    fn rng_is_roughly_uniform() {
        let mut rng = SplitMix64::new(42);
        let n = 100_000;
        let mean: f64 = (0..n).map(|_| rng.next_f64()).sum::<f64>() / n as f64;
        assert!((mean - 0.5).abs() < 0.01, "mean {mean}");
    }
}
