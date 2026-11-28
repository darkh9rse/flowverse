# FlowVerse

Measure fluid flow from an ordinary phone video.

FlowVerse turns two frames of a video of moving fluid into a velocity field
using **particle image velocimetry (PIV)**, then analyses it: vorticity,
divergence, critical points (vortex centres, saddles, sources, sinks), and
which simple flow it most resembles (uniform stream, vortex, source/sink, or
flow past a cylinder), with an R² score. An optional "Explain this flow"
button writes a plain-language report.

**Your video never leaves your device.** All measurement runs in your browser
in WebAssembly. Only if you press "Explain this flow" are a few computed
numbers (speeds, fit scores; never images) sent to the site's report
function. The page shows exactly what will be sent.

## Using it

1. Open the site and **upload or record a video**, or try a synthetic demo.
2. **Scrub to a frame** and press *Use as frame A*. Optionally choose
   *frame B*; if you don't, the next frame is used.
3. **Calibrate:** millimetres per pixel (measure a ruler in the video:
   length in mm ÷ length in pixels), the file's frame rate, and the
   slow-motion factor if you filmed in slow motion. Without mm/pixel,
   results are in pixels.
4. Press **Analyse flow**. You get velocity arrows, a vorticity colour map
   (red = clockwise, blue = anticlockwise), animated tracer particles, the
   best-fitting flow model with its parameters, and warnings about anything
   that looks wrong with the recording.

### Recording tips

- **Keep the camera still**: use a tripod or rest the phone on something.
  Camera movement looks like flow.
- **Film straight on**, perpendicular to the surface you're measuring.
- **Seed the fluid** with tracer particles that follow the water: ground
  pepper, coffee grounds, glitter, chalk dust. Aim for a dense, even speckle.
- **Light it brightly and evenly**, without glare spots. Bright light also
  means short exposures and sharp, unblurred particles.
- **Use the highest frame rate** available (slow-motion modes are great).
- Between frames A and B, particles should move **a few pixels to about 20
  px**. FlowVerse first measures with large windows and then refines with
  small ones (multi-pass PIV), so with the default 32 px window it can
  follow motions up to about a window size (32 px) if the video is large
  enough. Below ~0.5 px, noise dominates. Choose frames further apart for
  slow flow, closer together for fast flow.
- **Put a ruler in view** at the same distance as the flow, for calibration.
- If your phone's video won't open, record in "Most Compatible" / H.264
  instead of HEVC.

### What the analysis can and can't resolve

PIV measures the *average* motion inside each interrogation window, so
features smaller than about two windows are blurred. A 35 px cylinder
measured with 32 px windows fits as ~28 px; with 16 px windows, ~34 px. The
velocity should also change by less than ~5 % of the window size across one
window, so very tight, fast vortices need smaller windows or closer frames.

## How it works

| Part | What it does |
|---|---|
| `core/` | Rust library `flowverse-core`: grayscale images, potential-flow elements (`Uniform`, `Source`, `Vortex`, `Doublet`, `Cylinder`), a synthetic particle-image generator, PIV (FFT cross-correlation, Gaussian sub-pixel peak, loss-of-pairs bias correction, normalized median outlier test), and analysis (gradients, critical points, Levenberg–Marquardt model fits). No web code. |
| `wasm/` | Thin wasm-bindgen wrapper: RGBA frames in, JSON out. |
| `web/` | Static site (plain HTML/JS/CSS, no build step). PIV runs in a Web Worker so the page stays responsive. `test.html` is a self-test. |
| `functions/report/` | Netlify Function that sends the numeric summary to Claude for the plain-language report, with per-user rate limiting. The site falls back to a built-in text template whenever it is unavailable. |

Every physics module is tested against exact analytical answers, using
synthetic particle images made from known flows.

## Development setup

Requirements: [Rust via rustup](https://rustup.rs) and
[wasm-pack](https://github.com/wasm-bindgen/wasm-pack). Node.js 22+ is only
needed for the report function's tests.

```bash
# Rust tests (core physics + wasm JSON API)
cargo test --workspace

# Build the WebAssembly package into web/pkg
wasm-pack build wasm --release --target web --out-dir ../web/pkg

# Serve the site locally (any static server works)
python3 -m http.server -d web 8000
# then open http://localhost:8000 (and /test.html for the self-test)

# Report-function tests
npm install
npm test
```

Locally, without the function, "Explain this flow" uses the built-in
template. To try the AI report locally, use the
[Netlify CLI](https://docs.netlify.com/cli/get-started/) (`netlify dev`) with
`ANTHROPIC_API_KEY` set.

## Deploying to Netlify

1. Push the repository to GitHub and create a new Netlify site from it. The
   settings come from `netlify.toml`: the build runs
   `scripts/netlify-build.sh` (compiles the wasm package), publishes `web/`,
   and deploys `functions/report` at `/api/report`.
2. In **Site configuration → Environment variables**, set:
   - `ANTHROPIC_API_KEY`: optional. Netlify's AI Gateway already provides one
     automatically after the first production deploy, billed to your Netlify
     credits (Team settings → Billing → Usage). Set your own key to bill your
     Anthropic account instead.
   - `RATE_LIMIT_SALT`: a long random secret, used to hash visitor IPs for rate limiting.
   - Optional limits: `REPORT_LIMIT_PER_HOUR` (default 5 per visitor),
     `REPORT_LIMIT_PER_DAY` (10 per visitor), `REPORT_LIMIT_GLOBAL_PER_DAY`
     (20 for the whole site, which caps the AI bill). Past a limit, visitors
     get the built-in report instead.
3. Deploy. Visit `/test.html` on the live site to confirm WebAssembly works.

Netlify stops synchronous functions after 10 seconds by default. The report
function gives up at 9.5 s, and the page then shows the built-in report.

GitHub Actions (`.github/workflows/ci.yml`) runs formatting checks, all Rust
tests, the WebAssembly build and the function tests on every push.

## License

[MIT](LICENSE)
