//! End-to-end analysis tests: known flow → synthetic particle images →
//! PIV → analysis. The analysis must recover what we put in, from a
//! *measured* field with real PIV noise, not just from exact samples.
//!
//! Every flow respects the single-pass PIV gradient rule (displacement
//! changes by < ~1.6 px across a 32 px window; see tests/piv_synthetic.rs).

use std::f64::consts::PI;

use flowverse_core::analysis::{
    CriticalPointKind, FlowModel, divergence, find_critical_points, fit_flow_models, vorticity,
};
use flowverse_core::piv::analyze;
use flowverse_core::synthetic::generate_pair;
use flowverse_core::{
    Couette, Cylinder, FlowElement, Jet, ParticleImageConfig, PivConfig, Poiseuille, Source,
    Uniform, VectorField, Vortex,
};

const CENTRE: f64 = 127.5;

fn measure<F: FlowElement>(flow: &F) -> VectorField {
    let (a, b) = generate_pair(flow, &ParticleImageConfig::default());
    analyze(&a, &b, &PivConfig::default()).expect("PIV should run")
}

/// Mean of the known values within `radius` of the image centre.
fn mean_near_centre(field: &VectorField, values: &[Option<f64>], radius: f64) -> f64 {
    let picked: Vec<f64> = field
        .vectors
        .iter()
        .zip(values)
        .filter(|(v, _)| (v.x as f64 - CENTRE).hypot(v.y as f64 - CENTRE) < radius)
        .filter_map(|(_, s)| *s)
        .collect();
    picked.iter().sum::<f64>() / picked.len() as f64
}

#[test]
fn measured_vortex_is_analysed() {
    // Peak 1.5 px/frame at the 40 px core edge.
    let (rc, gamma) = (40.0, 2.0 * PI * 40.0 * 1.5);
    let field = measure(&Vortex::rankine(CENTRE, CENTRE, gamma, rc));

    // Vorticity in the core is Γ/(π rc²); divergence is zero everywhere.
    let w = mean_near_centre(&field, &vorticity(&field).values, 25.0);
    let expected = gamma / (PI * rc * rc);
    println!("core vorticity {w:.4}, expected {expected:.4}");
    assert!((w - expected).abs() < 0.1 * expected);
    let d = divergence(&field);
    let max_div = d
        .values
        .iter()
        .flatten()
        .fold(0.0f64, |m, x| m.max(x.abs()));
    println!("max |divergence| {max_div:.4}");
    assert!(max_div < 0.3 * expected, "divergence should be ~0");

    // One critical point: a centre, at the vortex position. Its position
    // error is about (PIV noise) / (velocity gradient) = 0.04 / 0.0375
    // ≈ 1 px per axis, so allow 4 px (a quarter of the 16 px grid).
    let points = find_critical_points(&field);
    println!("critical points: {points:?}");
    let centres: Vec<_> = points
        .iter()
        .filter(|p| p.kind == CriticalPointKind::Center)
        .collect();
    assert_eq!(centres.len(), 1);
    assert!((centres[0].x - CENTRE).abs() < 4.0 && (centres[0].y - CENTRE).abs() < 4.0);

    // The fit picks the vortex and recovers Γ and rc.
    let report = fit_flow_models(&field).unwrap();
    print_report(&report);
    let FlowModel::Vortex(v) = report.best().model else {
        panic!("best fit is not a vortex")
    };
    assert!(report.best().r_squared > 0.97);
    assert!((v.x - CENTRE).abs() < 1.0 && (v.y - CENTRE).abs() < 1.0);
    assert!(
        (v.circulation - gamma).abs() < 0.03 * gamma,
        "Γ = {}",
        v.circulation
    );
    assert!(
        (v.core_radius - rc).abs() < 0.1 * rc,
        "rc = {}",
        v.core_radius
    );
}

#[test]
fn measured_source_and_sink_are_analysed() {
    let (rc, m) = (40.0, 2.0 * PI * 40.0 * 1.5);
    for sign in [1.0, -1.0] {
        let field = measure(&Source::rankine(CENTRE, CENTRE, sign * m, rc));
        let div = mean_near_centre(&field, &divergence(&field).values, 25.0);
        let expected = sign * m / (PI * rc * rc);
        println!("core divergence {div:.4}, expected {expected:.4}");
        assert!((div - expected).abs() < 0.1 * expected.abs());

        let kind = if sign > 0.0 {
            CriticalPointKind::UnstableNode
        } else {
            CriticalPointKind::StableNode
        };
        let points = find_critical_points(&field);
        println!("critical points: {points:?}");
        assert!(points.iter().any(|p| p.kind == kind
            && (p.x - CENTRE).abs() < 4.0 // same reasoning as the vortex
            && (p.y - CENTRE).abs() < 4.0));

        let report = fit_flow_models(&field).unwrap();
        print_report(&report);
        assert_eq!(
            report.best().model.name(),
            if sign > 0.0 { "source" } else { "sink" }
        );
        assert!(report.best().r_squared > 0.97);
        let FlowModel::Source(s) = report.best().model else {
            unreachable!()
        };
        assert!(
            (s.strength - sign * m).abs() < 0.03 * m,
            "m = {}",
            s.strength
        );
    }
}

#[test]
fn measured_uniform_flow_is_analysed() {
    let field = measure(&Uniform { u: 2.3, v: -1.7 });
    // Uniform flow has no critical points.
    assert!(find_critical_points(&field).is_empty());
    let report = fit_flow_models(&field).unwrap();
    print_report(&report);
    assert_eq!(report.best().model.name(), "uniform");
    assert!(report.best().r_squared > 0.999);
}

#[test]
fn measured_cylinder_is_analysed() {
    // Stream 1 px/frame past a 35 px cylinder. The speed doubles at the
    // sides and drops to zero at the wall.
    //
    // Window size matters here. A window centred just inside the wall still
    // contains moving fluid, so PIV makes the body look smaller. With exact
    // samples the fit returns R = 35.0. Through PIV it returns ~27–29 px with
    // 32 px windows, ~32–33 px with 24 px, and ~34 px with 16 px. Rule of
    // thumb: windows no larger than about half the smallest feature.
    let truth = Cylinder {
        x: CENTRE,
        y: CENTRE,
        u: 1.0,
        v: 0.0,
        radius: 35.0,
    };
    let (a, b) = generate_pair(&truth, &ParticleImageConfig::default());
    let cfg = PivConfig {
        window_size: 16,
        step: 8,
        ..PivConfig::default()
    };
    let field = analyze(&a, &b, &cfg).unwrap();

    let report = fit_flow_models(&field).unwrap();
    print_report(&report);
    let FlowModel::Cylinder(c) = report.best().model else {
        panic!("best fit is not a cylinder")
    };
    assert!(report.best().r_squared > 0.95);
    assert!(
        (c.x - CENTRE).abs() < 4.0 && (c.y - CENTRE).abs() < 4.0,
        "{c:?}"
    );
    assert!((c.radius - 35.0).abs() < 3.0, "R = {}", c.radius);
    assert!((c.u - 1.0).abs() < 0.05 && c.v.abs() < 0.05, "{c:?}");
}

#[test]
fn measured_couette_flow_is_analysed() {
    // Still wall along the top edge (y = 0), sliding wall far below. The
    // speed grows 0.01 px/frame per px, so 0.32 px across a 32 px window:
    // well inside the gradient rule, and 2.5 px/frame at the bottom.
    let shear = 0.01;
    let field = measure(&Couette {
        x: 0.0,
        y: 0.0,
        angle: 0.0,
        shear_rate: shear,
    });

    // Couette vorticity is the same everywhere: −(shear rate) with y down.
    let w = mean_near_centre(&field, &vorticity(&field).values, 80.0);
    println!("vorticity {w:.5}, expected {:.5}", -shear);
    assert!((w + shear).abs() < 0.1 * shear);

    let report = fit_flow_models(&field).unwrap();
    print_report(&report);
    let FlowModel::Couette(c) = report.best().model else {
        panic!("best fit is not Couette flow")
    };
    assert!(report.best().r_squared > 0.99);
    assert!((c.shear_rate - shear).abs() < 0.03 * shear, "{c:?}");
    assert!(c.angle.abs() < 0.01, "{c:?}");
    // The wall line (where the fluid is still) is at y = 0.
    assert!(c.y.abs() < 3.0, "wall at y = {}", c.y);
}

#[test]
fn measured_poiseuille_flow_is_analysed() {
    // Horizontal channel, walls at y = 17.5 and 237.5 (half-width 110 px).
    // Peak 1.5 px/frame; the steepest change is at the walls, 2U/h = 0.027
    // px/frame per px, i.e. 0.87 px across a window (gradient rule OK).
    let truth = Poiseuille {
        x: CENTRE,
        y: CENTRE,
        angle: 0.0,
        half_width: 110.0,
        max_speed: 1.5,
    };
    let field = measure(&truth);
    let report = fit_flow_models(&field).unwrap();
    print_report(&report);
    let FlowModel::Poiseuille(p) = report.best().model else {
        panic!("best fit is not Poiseuille flow")
    };
    assert!(report.best().r_squared > 0.98);
    assert!((p.max_speed - 1.5).abs() < 0.03 * 1.5, "{p:?}");
    assert!((p.half_width - 110.0).abs() < 5.0, "{p:?}");
    assert!((p.y - CENTRE).abs() < 2.0 && p.angle.abs() < 0.01, "{p:?}");
    // The derived physics follows: mean speed 2U/3, wall shear rate 2U/h.
    assert!((p.mean_speed() - 1.0).abs() < 0.03);
    assert!((p.wall_shear_rate() - 3.0 / 110.0).abs() < 0.1 * 3.0 / 110.0);
}

#[test]
fn measured_jet_is_analysed() {
    // A near-nozzle "top-hat" jet: 140 px wide core, 12 px shear layers,
    // peak 1 px/frame. Steepest change U/(2δ) ≈ 0.042 px/frame per px, so
    // 1.3 px across a window (gradient rule OK).
    //
    // Note: a jet with *thick* shear layers (δ comparable to h) is a smooth
    // bell, almost a parabola. From velocities alone it is then nearly
    // indistinguishable from Poiseuille flow, and the simpler-model rule
    // picks Poiseuille. Sharp shear layers are what give a jet away.
    let truth = Jet {
        x: CENTRE,
        y: CENTRE,
        angle: 0.0,
        half_width: 70.0,
        shear_thickness: 12.0,
        max_speed: 1.0,
    };
    // 16 px windows resolve the 12 px shear layers; 32 px windows smear
    // them to ~18 px, the profile looks like a parabola, and (with still
    // surroundings, so no wall check can object) Poiseuille wins the tie.
    let (a, b) = generate_pair(&truth, &ParticleImageConfig::default());
    let cfg = PivConfig {
        window_size: 16,
        step: 8,
        ..PivConfig::default()
    };
    let field = analyze(&a, &b, &cfg).unwrap();
    let report = fit_flow_models(&field).unwrap();
    print_report(&report);
    let FlowModel::Jet(j) = report.best().model else {
        panic!("best fit is not a jet")
    };
    assert!(report.best().r_squared > 0.98);
    assert!((j.max_speed - 1.0).abs() < 0.04, "{j:?}");
    assert!((j.half_width - 70.0).abs() < 3.0, "{j:?}");
    // 32 px PIV windows average across the shear layer and make it look
    // thicker than it is; allow for that.
    assert!(
        j.shear_thickness > 10.0 && j.shear_thickness < 18.0,
        "{j:?}"
    );
    // The flow rate 2hU does not depend on δ, so it survives the blur.
    assert!(
        (j.flow_rate() - 140.0).abs() < 0.05 * 140.0,
        "Q = {}",
        j.flow_rate()
    );
}

fn print_report(r: &flowverse_core::analysis::FitReport) {
    for f in &r.fits {
        println!(
            "  {:9} R² = {:.4}  {:?}",
            f.model.name(),
            f.r_squared,
            f.model
        );
    }
    println!("  best: {}", r.best().model.name());
}
