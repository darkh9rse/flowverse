// Self-test for the WebAssembly package (see test.html).
// `init` loads and compiles the .wasm file. The other imports are the
// functions exported from wasm/src/lib.rs with #[wasm_bindgen].
import init, { synthetic_pair, run_piv, analyze_field, fit_flow } from "./pkg/flowverse_wasm.js";

const W = 256, H = 256;
// Same vortex as the Rust tests: peak 1.5 px/frame at a 40 px core.
const TRUE_GAMMA = 2 * Math.PI * 40 * 1.5;
const flow = { type: "vortex", x: 127.5, y: 127.5, circulation: TRUE_GAMMA, core_radius: 40 };

const checks = document.getElementById("checks");
function check(name, ok, detail) {
  const li = document.createElement("li");
  li.className = ok ? "pass" : "fail";
  li.textContent = `${ok ? "PASS" : "FAIL"}: ${name} (${detail})`;
  checks.append(li);
  return ok;
}

try {
  await init();

  // Two frames, RGBA bytes back to back.
  const bytes = synthetic_pair(JSON.stringify(flow), W, H, 1);
  const frameA = bytes.subarray(0, W * H * 4);
  const frameB = bytes.subarray(W * H * 4);
  document.getElementById("frameA").getContext("2d")
    .putImageData(new ImageData(new Uint8ClampedArray(frameA), W, H), 0, 0);

  const t0 = performance.now();
  const fieldJson = run_piv(frameA, frameB, W, H, "{}");
  const t1 = performance.now();
  const analysis = JSON.parse(analyze_field(fieldJson));
  const fit = JSON.parse(fit_flow(fieldJson));
  const t2 = performance.now();
  const field = JSON.parse(fieldJson);

  // Draw the vectors, scaled up 6× so 1–2 px displacements are visible.
  const ctx = document.getElementById("vectors").getContext("2d");
  ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, W, H);
  ctx.lineWidth = 1;
  for (let i = 0; i < field.x.length; i++) {
    ctx.strokeStyle = field.status[i] === 0 ? "#1456c8" : "#d33";
    ctx.beginPath();
    ctx.moveTo(field.x[i], field.y[i]);
    ctx.lineTo(field.x[i] + 6 * field.u[i], field.y[i] + 6 * field.v[i]);
    ctx.stroke();
  }

  const best = fit.fits[fit.best_index];
  const gamma = best.parameters.circulation;
  const centres = analysis.critical_points.filter(p => p.kind === "center");
  const results = [
    check("PIV grid is 15×15", field.cols === 15 && field.rows === 15, `${field.cols}×${field.rows}`),
    check("≥ 97 % vectors valid", analysis.stats.valid_fraction >= 0.97, `${(100 * analysis.stats.valid_fraction).toFixed(1)} %`),
    check("one centre near (127.5, 127.5)",
      centres.length === 1 && Math.hypot(centres[0].x - 127.5, centres[0].y - 127.5) < 4,
      centres.map(p => `(${p.x.toFixed(1)}, ${p.y.toFixed(1)})`).join(", ") || "none"),
    check("best fit is a vortex", fit.best === "vortex", `${fit.best}, R² = ${best.r_squared.toFixed(4)}`),
    check("circulation within 3 %", gamma !== undefined && Math.abs(gamma - TRUE_GAMMA) < 0.03 * TRUE_GAMMA,
      `${gamma?.toFixed(1)} vs ${TRUE_GAMMA.toFixed(1)}`),
  ];
  const allPass = results.every(Boolean);
  document.title = allPass ? "PASS: FlowVerse wasm test" : "FAIL: FlowVerse wasm test";
  document.body.dataset.result = allPass ? "pass" : "fail";

  document.getElementById("out").textContent =
    `PIV ${(t1 - t0).toFixed(0)} ms, analysis + fit ${(t2 - t1).toFixed(0)} ms\n\n` +
    `Fits:\n` + fit.fits.map(f => `  ${f.model.padEnd(9)} R² = ${f.r_squared.toFixed(4)}`).join("\n") +
    `\n\nStats: ${JSON.stringify(analysis.stats, null, 2)}`;
} catch (err) {
  document.title = "FAIL: FlowVerse wasm test";
  document.body.dataset.result = "fail";
  document.getElementById("out").textContent = "Error: " + (err?.stack || err);
}
