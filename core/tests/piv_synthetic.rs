//! End-to-end tests: make synthetic particle images from a known flow, run
//! PIV on them, and check that the flow comes back out.
//!
//! Files in `tests/` are *integration tests*. Cargo compiles each one as a
//! separate program that uses the crate from the outside, only through its
//! public API, the same way the wasm wrapper will.

use flowverse_core::flow::displacement;
use flowverse_core::piv::analyze;
use flowverse_core::synthetic::generate_pair;
use flowverse_core::{
    FlowElement, ParticleImageConfig, PivConfig, Uniform, VectorField, VectorStatus, Vortex,
};

/// Compare every valid vector with the exact displacement at its position.
/// Returns (RMS error, largest error, fraction of vectors that are valid).
fn error_stats<F: FlowElement>(field: &VectorField, flow: &F, dt: f64) -> (f64, f64, f64) {
    let mut sum_sq = 0.0;
    let mut max_err: f64 = 0.0;
    let mut valid = 0usize;
    for vec in &field.vectors {
        if vec.status != VectorStatus::Valid {
            continue;
        }
        valid += 1;
        let (tu, tv) = displacement(flow, vec.x as f64, vec.y as f64, dt);
        let err = (vec.u as f64 - tu).hypot(vec.v as f64 - tv);
        sum_sq += err * err;
        max_err = max_err.max(err);
    }
    let rms = (sum_sq / valid.max(1) as f64).sqrt();
    (rms, max_err, valid as f64 / field.vectors.len() as f64)
}

#[test]
fn uniform_shift_is_recovered() {
    // A sub-pixel shift in both directions, with a negative v, exercises
    // the Gaussian fit and the wrap-around of negative peaks.
    let flow = Uniform { u: 2.3, v: -1.7 };
    let img_cfg = ParticleImageConfig::default();
    let (a, b) = generate_pair(&flow, &img_cfg);

    let field = analyze(&a, &b, &PivConfig::default()).expect("PIV should run");
    // 256 px, 32 px windows, 16 px step → (256-32)/16 + 1 = 15 per side.
    assert_eq!((field.cols, field.rows), (15, 15));

    let (rms, max_err, valid) = error_stats(&field, &flow, img_cfg.dt);
    println!(
        "uniform: rms {rms:.4} px, max {max_err:.4} px, valid {:.1}%",
        valid * 100.0
    );
    // Literature: well-seeded 32 px windows with 3 px particles give about
    // 0.02–0.05 px random error *per component*. The error here is the
    // vector magnitude, which is √2 larger. Measured: ≈ 0.06 px.
    assert_eq!(valid, 1.0, "a clean uniform flow should give no outliers");
    assert!(rms < 0.08, "RMS error {rms} px too large");
    assert!(max_err < 0.3, "worst vector off by {max_err} px");

    // The mean over all vectors averages out random error, so it must be
    // very close to the true shift.
    let n = field.vectors.len() as f64;
    let mean_u = field.vectors.iter().map(|v| v.u as f64).sum::<f64>() / n;
    let mean_v = field.vectors.iter().map(|v| v.v as f64).sum::<f64>() / n;
    assert!((mean_u - 2.3).abs() < 0.02, "mean u {mean_u}");
    assert!((mean_v + 1.7).abs() < 0.02, "mean v {mean_v}");
}

#[test]
fn vortex_is_recovered() {
    // Rankine vortex in the middle of the image. The 40 px core rotates as a
    // solid body; outside, speed falls off as 1/r.
    //
    // Γ is set by the velocity *gradient*, not the speed. Single-pass PIV
    // assumes the displacement is roughly constant inside a window. The
    // design rule (Keane & Adrian 1992) is that it should change by less
    // than ~5 % of the window size across one window: 1.6 px for 32 px.
    // Here the gradient is peak_speed / rc both inside the core and just
    // outside it, so the change across a window is 32 · 1.5 / 40 = 1.2 px.
    // (With 4 px/frame it would be 3.2 px, and the core error grows to
    // ~0.45 px. Fixing that needs iterative window deformation, a later
    // feature.)
    let (cx, cy, rc) = (127.5, 127.5, 40.0);
    let peak_speed = 1.5;
    let gamma = 2.0 * std::f64::consts::PI * rc * peak_speed;
    let flow = Vortex::rankine(cx, cy, gamma, rc);

    let img_cfg = ParticleImageConfig::default();
    let (a, b) = generate_pair(&flow, &img_cfg);
    let field = analyze(&a, &b, &PivConfig::default()).expect("PIV should run");

    // 1. Vectors match the exact displacement everywhere.
    let (rms, max_err, valid) = error_stats(&field, &flow, img_cfg.dt);
    println!(
        "vortex: rms {rms:.4} px, max {max_err:.4} px, valid {:.1}%",
        valid * 100.0
    );
    // Looser than the uniform case. Even within the design rule, the
    // velocity change across a window smears the correlation peak.
    // Measured: ≈ 0.1 px in the core, ≈ 0.05 px far from it.
    assert!(
        valid > 0.97,
        "too many outliers: only {:.1}% valid",
        valid * 100.0
    );
    assert!(rms < 0.12, "RMS error {rms} px too large");
    assert!(max_err < 0.5, "worst vector off by {max_err} px");

    // 2. The vortex centre is where the displacement is smallest.
    let slowest = field
        .vectors
        .iter()
        .filter(|v| v.status == VectorStatus::Valid)
        .min_by(|p, q| p.u.hypot(p.v).total_cmp(&q.u.hypot(q.v)))
        .unwrap();
    println!("vortex centre found at ({}, {})", slowest.x, slowest.y);
    assert!(
        (slowest.x as f64 - cx).abs() <= 1.0 && (slowest.y as f64 - cy).abs() <= 1.0,
        "centre found at ({}, {}), expected ({cx}, {cy})",
        slowest.x,
        slowest.y
    );

    // 3. Circulation from a line integral Γ = ∮ u·dl around a square loop
    //    of grid points outside the core. Stokes' theorem says it equals
    //    the total vorticity inside, which for this flow is exactly Γ.
    let gamma_measured = loop_circulation(&field, 3, 11);
    let gamma_expected = gamma * img_cfg.dt; // displacement units: px²
    let rel_err = (gamma_measured - gamma_expected).abs() / gamma_expected;
    println!(
        "circulation: measured {gamma_measured:.1}, expected {gamma_expected:.1}, error {:.2}%",
        rel_err * 100.0
    );
    assert!(rel_err < 0.05, "circulation off by {:.1}%", rel_err * 100.0);
}

/// Trapezoid-rule line integral of the displacement around the square of
/// grid cells from index `lo` to `hi` (both inclusive, in columns and rows).
///
/// The loop goes +x along the top row, +y down the right column, −x along
/// the bottom, −y up the left. In y-down coordinates that is the orientation
/// for which `Vortex` with positive Γ gives a positive integral.
fn loop_circulation(field: &VectorField, lo: usize, hi: usize) -> f64 {
    let mut path = Vec::new();
    for c in lo..hi {
        path.push((c, lo));
    }
    for r in lo..hi {
        path.push((hi, r));
    }
    for c in (lo + 1..=hi).rev() {
        path.push((c, hi));
    }
    for r in (lo + 1..=hi).rev() {
        path.push((lo, r));
    }

    let mut total = 0.0;
    for i in 0..path.len() {
        let p = field.get(path[i].0, path[i].1);
        let q = field.get(path[(i + 1) % path.len()].0, path[(i + 1) % path.len()].1);
        let (dx, dy) = ((q.x - p.x) as f64, (q.y - p.y) as f64);
        total += 0.5 * ((p.u + q.u) as f64 * dx + (p.v + q.v) as f64 * dy);
    }
    total
}

#[test]
fn large_displacement_needs_multi_pass() {
    // 13.4 px right, 9.7 px up: 16.5 px in total, about half a 32 px
    // window, twice the "quarter rule" limit of single-pass PIV.
    let flow = Uniform { u: 13.4, v: -9.7 };
    let img_cfg = ParticleImageConfig {
        // Seed particles well outside the frame so ones moving in from the
        // edge exist in frame B too.
        margin: 30.0,
        ..ParticleImageConfig::default()
    };
    let (a, b) = generate_pair(&flow, &img_cfg);

    // Single pass: a third of each window's particles leave it, so the peak
    // is weaker and noisier. On these clean synthetic images it degrades
    // (≈0.17 px RMS, ≈10 % outliers); on real, compressed video at the same
    // shift it collapsed completely (see core/examples/piv_frames.rs).
    let single = PivConfig {
        multi_pass: false,
        ..PivConfig::default()
    };
    let field = analyze(&a, &b, &single).unwrap();
    let (rms1, _, valid1) = error_stats(&field, &flow, img_cfg.dt);
    println!(
        "single pass: rms {rms1:.3} px, valid {:.1}%",
        valid1 * 100.0
    );

    // Multi-pass (the default): coarse windows predict the shift, the final
    // windows are offset by it, and the accuracy is that of a small shift.
    let field = analyze(&a, &b, &PivConfig::default()).unwrap();
    let (rms, max_err, valid) = error_stats(&field, &flow, img_cfg.dt);
    println!(
        "multi-pass: rms {rms:.3} px, max {max_err:.3} px, valid {:.1}%",
        valid * 100.0
    );
    assert!(
        valid1 < 0.95 && rms < 0.5 * rms1,
        "multi-pass should clearly beat single pass"
    );
    assert!(valid > 0.95, "only {:.1}% valid", valid * 100.0);
    assert!(rms < 0.1, "RMS error {rms} px too large");
}
