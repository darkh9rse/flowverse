# FlowVerse

Measures fluid flow from ordinary phone videos with PIV in the browser, then
classifies the flow and explains it.

## Architecture (Cargo workspace)
- core/: Rust library `flowverse-core`. PIV, analysis, flow fitting. No web code.
- wasm/: thin wrapper exposing the core to JavaScript with wasm-bindgen.
  Built with: wasm-pack build wasm --target web --out-dir ../web/pkg
  (output lands in web/pkg, which is git-ignored and rebuilt on deploy)
- web/: static website (plain HTML/JS). Upload a video, pick frames,
  run PIV via the wasm package, draw vectors, vorticity and moving particles.
  Video never leaves the user's device.
- functions/report/: one serverless function that sends ONLY computed numbers to
  an AI API for a plain-language report. The API key is an environment variable.
  The app must work fully without it (built-in text template fallback).

## Design rules
- Flow types implement a `FlowElement` trait (Uniform, Source, Vortex, Doublet).
- PIV: FFT cross-correlation (rustfft crate), Gaussian sub-pixel peak,
  normalized median outlier test.
- Every physics module has `cargo test` tests against known analytical answers,
  using synthetic particle images generated from FlowElements.
- Comment code for a learner new to Rust; explain non-obvious choices.
- Deploy target: Netlify or Cloudflare Pages (static site + one function).

## Commands
- Tests: `cargo test --workspace` (Rust), `npm test` (report function, Node 22+)
- Wasm: `bash scripts/netlify-build.sh` (same as Netlify and CI)
- Local site: `python3 -m http.server -d web 8000`, self-test at /test.html
