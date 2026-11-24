//! # flowverse-wasm
//!
//! A thin layer that lets JavaScript call `flowverse-core`. All physics lives
//! in the core; this crate only converts between JavaScript-friendly data
//! (byte arrays and JSON strings) and core types.
//!
//! Build: `wasm-pack build wasm --target web --out-dir ../web/pkg`
//!
//! ## Why JSON strings?
//! wasm-bindgen can pass numbers, strings and byte arrays across the
//! JS ↔ WebAssembly boundary cheaply. Rich structs would need more glue.
//! A JSON string is one copy, and `JSON.parse` in the browser is very fast.
//! It also keeps this API identical to what the web worker posts to the
//! page.
//!
//! ## Structure
//! Each exported function is a two-line wrapper around a plain Rust function
//! returning `Result<String, String>`. The plain functions can be tested
//! with `cargo test` on a normal computer, because creating a JavaScript
//! error (`JsError`) only works inside a browser.
//!
//! All results are in **pixels and frames**. The page applies the
//! calibration (mm per pixel, frames per second).

use flowverse_core::analysis::{self, CriticalPointKind, FitReport, FlowModel};
use flowverse_core::piv::{self, PivConfig};
use flowverse_core::synthetic::{self, ParticleImageConfig};
use flowverse_core::{
    Cylinder, Doublet, FlowElement, Image, Source, Uniform, Vector, VectorField, VectorStatus,
    Vortex,
};
use serde::Deserialize;
use serde_json::{Value, json};
use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// Exported functions (what JavaScript sees)
// ---------------------------------------------------------------------------

/// Run PIV on two frames of RGBA bytes (e.g. `ImageData.data` from a canvas).
///
/// `options` is JSON with any of `window_size`, `step`, `median_threshold`,
/// `median_epsilon`; missing ones use the defaults. Pass `"{}"` for all
/// defaults. Returns the vector field as JSON (see [`field_to_json`]).
#[wasm_bindgen]
pub fn run_piv(
    frame_a: &[u8],
    frame_b: &[u8],
    width: u32,
    height: u32,
    options: &str,
) -> Result<String, JsError> {
    run_piv_json(frame_a, frame_b, width as usize, height as usize, options)
        .map_err(|e| JsError::new(&e))
}

/// Vorticity, divergence, critical points and summary statistics for a
/// field produced by [`run_piv`].
#[wasm_bindgen]
pub fn analyze_field(field: &str) -> Result<String, JsError> {
    analyze_field_json(field).map_err(|e| JsError::new(&e))
}

/// Fit uniform / vortex / source / cylinder models to a field produced by
/// [`run_piv`] and report the best one with R² scores.
#[wasm_bindgen]
pub fn fit_flow(field: &str) -> Result<String, JsError> {
    fit_flow_json(field).map_err(|e| JsError::new(&e))
}

/// Make two synthetic particle frames moved by a known flow, for demos and
/// tests. Returns frame A's RGBA bytes followed by frame B's
/// (`2 · width · height · 4` bytes in total).
///
/// `flow` is JSON such as `{"type":"vortex","x":128,"y":128,
/// "circulation":377,"core_radius":40}`. See [`FlowSpec`] for every type.
#[wasm_bindgen]
pub fn synthetic_pair(flow: &str, width: u32, height: u32, seed: u32) -> Result<Vec<u8>, JsError> {
    synthetic_pair_bytes(flow, width as usize, height as usize, seed as u64)
        .map_err(|e| JsError::new(&e))
}

// ---------------------------------------------------------------------------
// Plain Rust implementations (testable without a browser)
// ---------------------------------------------------------------------------

/// PIV settings as they arrive from JavaScript. `Option` + `#[serde(default)]`
/// means every field may be left out.
#[derive(Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
struct PivOptions {
    window_size: Option<usize>,
    step: Option<usize>,
    median_threshold: Option<f32>,
    median_epsilon: Option<f32>,
}

pub fn run_piv_json(
    a: &[u8],
    b: &[u8],
    width: usize,
    height: usize,
    options: &str,
) -> Result<String, String> {
    let opts: PivOptions =
        serde_json::from_str(options).map_err(|e| format!("bad PIV options: {e}"))?;
    let d = PivConfig::default();
    let window_size = opts.window_size.unwrap_or(d.window_size);
    let cfg = PivConfig {
        window_size,
        // Default step: 50 % overlap of whatever window was chosen.
        step: opts.step.unwrap_or((window_size / 2).max(1)),
        median_threshold: opts.median_threshold.unwrap_or(d.median_threshold),
        median_epsilon: opts.median_epsilon.unwrap_or(d.median_epsilon),
    };
    let img_a = Image::from_rgba8(width, height, a).map_err(|e| format!("frame A: {e}"))?;
    let img_b = Image::from_rgba8(width, height, b).map_err(|e| format!("frame B: {e}"))?;
    let field = piv::analyze(&img_a, &img_b, &cfg).map_err(|e| e.to_string())?;
    Ok(field_to_json(&field, &cfg).to_string())
}

/// Status codes in the JSON: compact, and easy to switch on in JavaScript.
const STATUS_VALID: u8 = 0;
const STATUS_OUTLIER: u8 = 1;
const STATUS_NO_PEAK: u8 = 2;

/// The field as *parallel arrays* (`x[i]`, `y[i]`, `u[i]`, `v[i]`,
/// `status[i]` describe vector i, row-major). For thousands of vectors this
/// is much smaller and faster to parse than an array of objects.
fn field_to_json(field: &VectorField, cfg: &PivConfig) -> Value {
    let pick = |f: fn(&Vector) -> f32| field.vectors.iter().map(f).collect::<Vec<f32>>();
    let status: Vec<u8> = field
        .vectors
        .iter()
        .map(|v| match v.status {
            VectorStatus::Valid => STATUS_VALID,
            VectorStatus::Outlier => STATUS_OUTLIER,
            VectorStatus::NoPeak => STATUS_NO_PEAK,
        })
        .collect();
    json!({
        "cols": field.cols,
        "rows": field.rows,
        "window_size": cfg.window_size,
        "step": cfg.step,
        "x": pick(|v| v.x),
        "y": pick(|v| v.y),
        "u": pick(|v| v.u),
        "v": pick(|v| v.v),
        "status": status,
    })
}

/// The same parallel-array layout, read back in.
#[derive(Deserialize)]
struct FieldJson {
    cols: usize,
    rows: usize,
    x: Vec<f32>,
    y: Vec<f32>,
    u: Vec<f32>,
    v: Vec<f32>,
    status: Vec<u8>,
}

fn field_from_json(text: &str) -> Result<VectorField, String> {
    let f: FieldJson = serde_json::from_str(text).map_err(|e| format!("bad field JSON: {e}"))?;
    let n = f.cols * f.rows;
    if [f.x.len(), f.y.len(), f.u.len(), f.v.len(), f.status.len()]
        .iter()
        .any(|&len| len != n)
    {
        return Err(format!(
            "field arrays must all have cols*rows = {n} entries"
        ));
    }
    let vectors = (0..n)
        .map(|i| Vector {
            x: f.x[i],
            y: f.y[i],
            u: f.u[i],
            v: f.v[i],
            status: match f.status[i] {
                STATUS_VALID => VectorStatus::Valid,
                STATUS_OUTLIER => VectorStatus::Outlier,
                _ => VectorStatus::NoPeak,
            },
        })
        .collect();
    Ok(VectorField {
        cols: f.cols,
        rows: f.rows,
        vectors,
    })
}

pub fn analyze_field_json(text: &str) -> Result<String, String> {
    let field = field_from_json(text)?;
    let vort = analysis::vorticity(&field);
    let div = analysis::divergence(&field);
    let all_points = analysis::find_critical_points(&field);
    // Hide noise points in still regions (see CriticalPoint::is_significant).
    let scale = analysis::gradient_scale(&field);
    let (points, weak): (Vec<_>, Vec<_>) = all_points
        .into_iter()
        .partition(|p| p.is_significant(scale, 0.2));

    let valid: Vec<&Vector> = field
        .vectors
        .iter()
        .filter(|v| v.status == VectorStatus::Valid)
        .collect();
    let speeds: Vec<f64> = valid
        .iter()
        .map(|v| (v.u as f64).hypot(v.v as f64))
        .collect();
    let mean = |xs: &[f64]| {
        if xs.is_empty() {
            0.0
        } else {
            xs.iter().sum::<f64>() / xs.len() as f64
        }
    };
    let known_w: Vec<f64> = vort.values.iter().flatten().copied().collect();
    let known_d: Vec<f64> = div.values.iter().flatten().copied().collect();
    let max_abs = |xs: &[f64]| xs.iter().fold(0.0f64, |m, x| m.max(x.abs()));

    let critical: Vec<Value> = points
        .iter()
        .map(|p| {
            let [l1, l2] = p.gradient.eigenvalues();
            json!({
                "x": p.x,
                "y": p.y,
                "kind": p.kind.name(),
                "eigenvalues": [[l1.re, l1.im], [l2.re, l2.im]],
                "vorticity": p.gradient.vorticity(),
                "divergence": p.gradient.divergence(),
                "strength": p.strength(),
            })
        })
        .collect();

    // Count by kind, for quick summaries ("2 centres, 1 saddle").
    let kinds = [
        CriticalPointKind::Saddle,
        CriticalPointKind::Center,
        CriticalPointKind::StableFocus,
        CriticalPointKind::UnstableFocus,
        CriticalPointKind::StableNode,
        CriticalPointKind::UnstableNode,
        CriticalPointKind::Degenerate,
    ];
    let mut counts = serde_json::Map::new();
    for k in kinds {
        let n = points.iter().filter(|p| p.kind == k).count();
        if n > 0 {
            counts.insert(k.name().to_string(), json!(n));
        }
    }

    let out = json!({
        // `Option<f64>` becomes a number or `null` in JSON.
        "vorticity": vort.values,
        "divergence": div.values,
        "critical_points": critical,
        "stats": {
            "vector_count": field.vectors.len(),
            "valid_count": valid.len(),
            "valid_fraction": if field.vectors.is_empty() { 0.0 } else { valid.len() as f64 / field.vectors.len() as f64 },
            "mean_speed": mean(&speeds),
            "max_speed": speeds.iter().cloned().fold(0.0f64, f64::max),
            "mean_vorticity": mean(&known_w),
            "max_abs_vorticity": max_abs(&known_w),
            "mean_divergence": mean(&known_d),
            "max_abs_divergence": max_abs(&known_d),
            "critical_point_counts": counts,
            "weak_critical_points_hidden": weak.len(),
        },
    });
    Ok(out.to_string())
}

pub fn fit_flow_json(text: &str) -> Result<String, String> {
    let field = field_from_json(text)?;
    Ok(match analysis::fit_flow_models(&field) {
        Some(report) => report_to_json(&report),
        // Too few vectors, or nothing moves.
        None => json!({ "best": null, "best_index": null, "vector_count": 0, "fits": [] }),
    }
    .to_string())
}

fn report_to_json(report: &FitReport) -> Value {
    let fits: Vec<Value> = report
        .fits
        .iter()
        .map(|f| {
            json!({
                "model": f.model.name(),
                "r_squared": f.r_squared,
                "parameter_count": f.model.parameter_count(),
                "parameters": model_parameters(&f.model),
            })
        })
        .collect();
    json!({
        "best": report.best().model.name(),
        "best_index": report.best,
        "vector_count": report.vector_count,
        "fits": fits,
    })
}

fn model_parameters(model: &FlowModel) -> Value {
    match model {
        FlowModel::Uniform(f) => json!({ "u": f.u, "v": f.v }),
        FlowModel::Vortex(f) => {
            json!({ "x": f.x, "y": f.y, "circulation": f.circulation, "core_radius": f.core_radius })
        }
        FlowModel::Source(f) => {
            json!({ "x": f.x, "y": f.y, "strength": f.strength, "core_radius": f.core_radius })
        }
        FlowModel::Cylinder(f) => {
            json!({ "x": f.x, "y": f.y, "u": f.u, "v": f.v, "radius": f.radius })
        }
    }
}

/// A flow described in JSON. `#[serde(tag = "type")]` picks the variant
/// from the `"type"` field, e.g. `{"type": "uniform", "u": 1, "v": 0}`.
#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "lowercase", deny_unknown_fields)]
enum FlowSpec {
    Uniform {
        u: f64,
        v: f64,
    },
    Vortex {
        x: f64,
        y: f64,
        circulation: f64,
        #[serde(default)]
        core_radius: f64,
    },
    Source {
        x: f64,
        y: f64,
        strength: f64,
        #[serde(default)]
        core_radius: f64,
    },
    Doublet {
        x: f64,
        y: f64,
        strength: f64,
    },
    Cylinder {
        x: f64,
        y: f64,
        u: f64,
        v: f64,
        radius: f64,
    },
}

impl FlowElement for FlowSpec {
    fn velocity(&self, x: f64, y: f64) -> (f64, f64) {
        match *self {
            FlowSpec::Uniform { u, v } => Uniform { u, v }.velocity(x, y),
            FlowSpec::Vortex {
                x: x0,
                y: y0,
                circulation,
                core_radius,
            } => Vortex::rankine(x0, y0, circulation, core_radius).velocity(x, y),
            FlowSpec::Source {
                x: x0,
                y: y0,
                strength,
                core_radius,
            } => Source::rankine(x0, y0, strength, core_radius).velocity(x, y),
            FlowSpec::Doublet {
                x: x0,
                y: y0,
                strength,
            } => Doublet {
                x: x0,
                y: y0,
                strength,
            }
            .velocity(x, y),
            FlowSpec::Cylinder {
                x: x0,
                y: y0,
                u,
                v,
                radius,
            } => Cylinder {
                x: x0,
                y: y0,
                u,
                v,
                radius,
            }
            .velocity(x, y),
        }
    }
}

/// Largest synthetic frame we agree to make (4K UHD), so a typo can't ask the
/// browser for gigabytes.
const MAX_SYNTHETIC_PIXELS: usize = 3840 * 2160;

pub fn synthetic_pair_bytes(
    flow: &str,
    width: usize,
    height: usize,
    seed: u64,
) -> Result<Vec<u8>, String> {
    let spec: FlowSpec = serde_json::from_str(flow).map_err(|e| format!("bad flow JSON: {e}"))?;
    if width == 0 || height == 0 || width * height > MAX_SYNTHETIC_PIXELS {
        return Err(format!(
            "synthetic image size {width}x{height} out of range"
        ));
    }
    let cfg = ParticleImageConfig {
        width,
        height,
        seed,
        ..ParticleImageConfig::default()
    };
    let (a, b) = synthetic::generate_pair(&spec, &cfg);
    let mut bytes = a.to_rgba8();
    bytes.extend(b.to_rgba8());
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The full JavaScript path, minus the browser: JSON flow → synthetic
    /// RGBA frames → PIV → analysis → fit, all through the JSON API.
    #[test]
    fn vortex_round_trip_through_the_json_api() {
        let flow = r#"{"type":"vortex","x":127.5,"y":127.5,"circulation":377,"core_radius":40}"#;
        let bytes = synthetic_pair_bytes(flow, 256, 256, 1).unwrap();
        assert_eq!(bytes.len(), 2 * 256 * 256 * 4);
        let (a, b) = bytes.split_at(256 * 256 * 4);

        let field = run_piv_json(a, b, 256, 256, "{}").unwrap();
        let parsed: Value = serde_json::from_str(&field).unwrap();
        assert_eq!(parsed["cols"], 15);
        assert_eq!(parsed["step"], 16);

        let analysis: Value = serde_json::from_str(&analyze_field_json(&field).unwrap()).unwrap();
        assert_eq!(analysis["stats"]["critical_point_counts"]["center"], 1);
        assert!(analysis["stats"]["valid_fraction"].as_f64().unwrap() > 0.97);

        let fit: Value = serde_json::from_str(&fit_flow_json(&field).unwrap()).unwrap();
        assert_eq!(fit["best"], "vortex");
        let gamma = fit["fits"][1]["parameters"]["circulation"]
            .as_f64()
            .unwrap();
        assert!((gamma - 377.0).abs() < 15.0, "Γ = {gamma}");
    }

    #[test]
    fn options_choose_window_and_default_step_is_half() {
        let bytes = synthetic_pair_bytes(r#"{"type":"uniform","u":1,"v":0}"#, 128, 128, 3).unwrap();
        let (a, b) = bytes.split_at(128 * 128 * 4);
        let field: Value =
            serde_json::from_str(&run_piv_json(a, b, 128, 128, r#"{"window_size":16}"#).unwrap())
                .unwrap();
        assert_eq!(field["window_size"], 16);
        assert_eq!(field["step"], 8);
        assert_eq!(field["cols"], 15); // (128 - 16) / 8 + 1
    }

    #[test]
    fn bad_input_gives_readable_errors_not_panics() {
        assert!(
            run_piv_json(&[0; 16], &[0; 16], 2, 2, "{}")
                .unwrap_err()
                .contains("smaller")
        );
        assert!(
            run_piv_json(&[0; 15], &[0; 16], 2, 2, "{}")
                .unwrap_err()
                .contains("frame A")
        );
        assert!(
            run_piv_json(&[0; 16], &[0; 16], 2, 2, r#"{"windw":8}"#)
                .unwrap_err()
                .contains("options")
        );
        assert!(analyze_field_json("{").unwrap_err().contains("field JSON"));
        assert!(synthetic_pair_bytes(r#"{"type":"tornado"}"#, 8, 8, 1).is_err());
        assert!(
            synthetic_pair_bytes(r#"{"type":"uniform","u":1,"v":0}"#, 100_000, 100_000, 1).is_err()
        );
    }

    #[test]
    fn still_frames_fit_to_null() {
        let bytes = synthetic_pair_bytes(r#"{"type":"uniform","u":0,"v":0}"#, 64, 64, 1).unwrap();
        let (a, b) = bytes.split_at(64 * 64 * 4);
        let field = run_piv_json(a, b, 64, 64, "{}").unwrap();
        let fit: Value = serde_json::from_str(&fit_flow_json(&field).unwrap()).unwrap();
        assert!(fit["best"].is_null());
    }
}
