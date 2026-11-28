# FlowVerse

Measure fluid flow from an ordinary phone video.

**Try it: [flowversse.netlify.app](https://flowversse.netlify.app)**. It
runs in any modern browser with nothing to install, and has built-in demos
if you have no video yet.

FlowVerse turns two frames of a video of moving fluid into a velocity field
using **particle image velocimetry (PIV)**, then analyses it: vorticity,
divergence, critical points (vortex centres, saddles, sources, sinks), and
which simple flow it most resembles, with an R² score:

| Ideal (inviscid) flows | Viscous channel flows (exact Navier–Stokes solutions) | Free flows |
|---|---|---|
| uniform stream, vortex, source/sink, flow past a cylinder | Couette (linear shear), Poiseuille (parabolic channel profile) | jet (fast core with two shear layers) |

Fits that are physically impossible are ruled out even if their R² is high:
for example, a "channel" whose walls would cross fluid that is visibly
moving. An optional "Explain this flow" button writes a plain-language
report.

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

### Real videos

Two processing steps make ordinary videos work better:

- **Multi-pass PIV.** Large windows first measure the rough motion; the
  final small windows are shifted by it. Motions up to about a window size
  are measured (with a single pass the limit is a quarter of that).
- **Background removal** (on by default). The per-pixel median of 15
  frames from about a second either side of A and B is the still scene: a
  stream bottom seen through clear water, tank walls, a printed timestamp.
  It is subtracted so only moving particles are measured. Needs a steady
  camera; turn it off for flows so slow that particles barely move in two
  seconds.

Tested on: a high-speed lab recording of a laminar jet (classified as a jet,
R² 0.97, 99 % reliable vectors), and a hand-held phone clip of a shallow
canal with foam (no simple model fits, correctly; background removal raised
the share of vectors following the current from 55 % to 64 % and the
measured surface motion by ~45 %, which the still stream bed had been
dragging towards zero). Filmed at an angle, speeds are distorted by
perspective; film straight down for numbers you can trust.

### What the analysis can and can't resolve

PIV measures the *average* motion inside each interrogation window, so
features smaller than about two windows are blurred. A 35 px cylinder
measured with 32 px windows fits as ~28 px; with 16 px windows, ~34 px. The
velocity should also change by less than ~5 % of the window size across one
window, so very tight, fast vortices need smaller windows or closer frames.

## Fluid mechanics behind it

FlowVerse started as an additional project for a Fluid Dynamics course. It
applies the course material to measured data instead of textbook formulas:

| Course topic | Where it appears in FlowVerse |
|---|---|
| Eulerian description, velocity field | PIV turns two frames into a velocity field **V**(x, y) on a grid |
| Pathlines | Animated tracer particles are moved through the measured field |
| Vorticity, rotational vs irrotational flow | Vorticity map ω = ∂v/∂x − ∂u/∂y, red and blue where the fluid spins |
| Continuity, incompressibility | Divergence ∂u/∂x + ∂v/∂y as a check: near zero for a 2-D incompressible flow |
| Potential flow and superposition | Uniform stream, source/sink and vortex fitted to the data; flow past a cylinder as uniform stream + doublet |
| Stagnation points | Critical points located and classified (vortex centre, saddle, source, sink) |
| Exact viscous solutions | Couette (linear shear) and plane Poiseuille (parabolic) profiles fitted, with a check that the implied walls are physically possible |
| Free shear flows | Jet model with a fast core and two shear layers |

Each model is tested against its analytical solution, using synthetic
particle images made from that exact flow (`cargo test`).

## How it works

| Part | What it does |
|---|---|
| `core/` | Rust library `flowverse-core`: grayscale images, potential-flow elements (`Uniform`, `Source`, `Vortex`, `Doublet`, `Cylinder`), viscous `Couette` and `Poiseuille` flows and a free `Jet`, a synthetic particle-image generator, background removal (temporal median), PIV (multi-pass with window offset, FFT cross-correlation, Gaussian sub-pixel peak, loss-of-pairs bias correction, normalized median outlier test), and analysis (gradients, critical points, Levenberg–Marquardt model fits with physical plausibility checks). No web code. |
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
