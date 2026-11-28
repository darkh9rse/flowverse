//! Run PIV on two real video frames from the command line.
//!
//! The website does this in the browser; this example lets you test the
//! same core on any video from a terminal, without a browser.
//!
//! 1. Pull two consecutive frames out of a video as 8-bit grayscale PGM
//!    files with ffmpeg (PGM is the simplest image format: a short text
//!    header, then one byte per pixel, so we can read it without any
//!    image library):
//!
//!    ```text
//!    ffmpeg -i video.mp4 -vf "select='eq(n\,100)+eq(n\,101)',format=gray" \
//!           -vsync 0 frame%d.pgm
//!    ```
//!
//! 2. Run:
//!
//!    ```text
//!    cargo run --release -p flowverse-core --example piv_frames -- frame1.pgm frame2.pgm [window]
//!    ```
//!
//! It prints the displacement statistics, the fitted flow models with their
//! R² scores, and the critical points. Add `--csv out.csv` at the end to
//! save the vectors (x, y, u, v, status) for plotting.

use std::error::Error;
use std::fs;

use flowverse_core::analysis::{find_critical_points, fit_flow_models, vorticity};
use flowverse_core::piv::analyze;
use flowverse_core::{Image, PivConfig, VectorStatus};

/// Read a binary (P5) PGM file into an [`Image`] with values 0..1.
fn read_pgm(path: &str) -> Result<Image, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    // The header is four whitespace-separated tokens: "P5", width, height,
    // max value. `#` starts a comment that runs to the end of the line.
    let mut tokens = Vec::new();
    let mut i = 0;
    while tokens.len() < 4 {
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i < bytes.len() && bytes[i] == b'#' {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        let start = i;
        while i < bytes.len() && !bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if start == i {
            return Err(format!("{path}: truncated PGM header").into());
        }
        tokens.push(std::str::from_utf8(&bytes[start..i])?.to_string());
    }
    i += 1; // exactly one whitespace byte separates the header from the pixels
    if tokens[0] != "P5" || tokens[3] != "255" {
        return Err(format!("{path}: need an 8-bit binary PGM (P5, max 255)").into());
    }
    let (w, h): (usize, usize) = (tokens[1].parse()?, tokens[2].parse()?);
    let pixels = bytes
        .get(i..i + w * h)
        .ok_or("PGM pixel data is too short")?;
    let data = pixels.iter().map(|&p| p as f32 / 255.0).collect();
    Ok(Image::from_vec(w, h, data)?)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: piv_frames A.pgm B.pgm [window] [--csv out.csv]");
        std::process::exit(2);
    }
    let a = read_pgm(&args[1])?;
    let b = read_pgm(&args[2])?;
    let window = args
        .get(3)
        .filter(|s| !s.starts_with("--"))
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(32);
    let cfg = PivConfig {
        window_size: window,
        step: window / 2,
        ..PivConfig::default()
    };
    println!(
        "frames {}×{} px, window {} px, step {} px",
        a.width(),
        a.height(),
        cfg.window_size,
        cfg.step
    );

    let field = analyze(&a, &b, &cfg)?;
    let valid: Vec<_> = field
        .vectors
        .iter()
        .filter(|v| v.status == VectorStatus::Valid)
        .collect();
    let speeds: Vec<f32> = valid.iter().map(|v| v.u.hypot(v.v)).collect();
    let mean = speeds.iter().sum::<f32>() / speeds.len().max(1) as f32;
    let max = speeds.iter().cloned().fold(0.0, f32::max);
    println!(
        "{} of {} vectors valid; speed mean {mean:.3} px/frame, max {max:.3}",
        valid.len(),
        field.vectors.len()
    );
    let w = vorticity(&field);
    let wmax = w
        .values
        .iter()
        .flatten()
        .fold(0.0f64, |m, x| m.max(x.abs()));
    println!("max |vorticity| {wmax:.4} per frame");

    match fit_flow_models(&field) {
        None => println!("no fit (nothing moved)"),
        Some(report) => {
            println!("model fits (best = {}):", report.best().model.name());
            for f in &report.fits {
                println!(
                    "  {:<11} R² = {:>7.4}  {:?}",
                    f.model.name(),
                    f.r_squared,
                    f.model
                );
            }
        }
    }
    for p in find_critical_points(&field) {
        println!("critical point: {:?} at ({:.1}, {:.1})", p.kind, p.x, p.y);
    }

    if let Some(pos) = args.iter().position(|s| s == "--csv") {
        let path = args.get(pos + 1).ok_or("--csv needs a file name")?;
        let mut out = String::from("x,y,u,v,status\n");
        for v in &field.vectors {
            out += &format!("{},{},{},{},{:?}\n", v.x, v.y, v.u, v.v, v.status);
        }
        fs::write(path, out)?;
        println!("vectors written to {path}");
    }
    Ok(())
}
