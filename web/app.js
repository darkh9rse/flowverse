// FlowVerse page logic: load a video, pick two frames, calibrate, send the
// frames to the Web Worker for PIV, then show and explain the results.
//
// Nothing here uploads the video: frames are read from a local <video>
// element into canvases and processed on this device.

import { FlowView, vorticityColour, percentile } from "./render.js";
import { setupExplain } from "./explain.js";

const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------------------
// Worker: one long-lived worker, with a tiny request/response wrapper.
// ---------------------------------------------------------------------------

const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
let nextId = 1;
const pending = new Map();
worker.onmessage = ({ data }) => {
  const req = pending.get(data.id);
  if (!req) return;
  if (data.progress) return req.onProgress?.(data.progress);
  pending.delete(data.id);
  data.ok ? req.resolve(data.result) : req.reject(new Error(data.error));
};
worker.onerror = (e) => {
  for (const req of pending.values()) req.reject(new Error(e.message || "worker failed to start"));
  pending.clear();
};
function ask(message, transfer = [], onProgress) {
  return new Promise((resolve, reject) => {
    const id = nextId++;
    pending.set(id, { resolve, reject, onProgress });
    worker.postMessage({ id, ...message }, transfer);
  });
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

const state = {
  source: null, // "video" | "demo"
  frameIndex: 0,
  frameCount: 0,
  A: null, // { index, canvas }  full-resolution copy of the chosen frame
  B: null,
  last: null, // most recent analysis, for redraws and the report
};

const video = $("video");
const view = new FlowView($("view"));
setupExplain($("explain-area"));
window.flowverse = { state, view }; // handy for debugging in the console

// Respect "reduce motion": start with the particle animation off.
if (matchMedia("(prefers-reduced-motion: reduce)").matches) {
  $("layer-particles").checked = false;
  view.setLayer("particles", false);
}

// ---------------------------------------------------------------------------
// Step 1a: video loading and frame stepping
// ---------------------------------------------------------------------------

const fps = () => Math.max(1, parseFloat($("fps").value) || 30);

$("video-input").addEventListener("change", (e) => {
  const file = e.target.files?.[0];
  if (!file) return;
  if (video.src) URL.revokeObjectURL(video.src);
  state.source = "video";
  state.A = state.B = null;
  updateFrames();
  setStatus("Loading video…");
  // An object URL lets the <video> element read the local file directly.
  // The file is never uploaded anywhere.
  delete video.dataset.ready;
  state.frameIndex = -1;
  video.src = URL.createObjectURL(file);
  video.load();
});

video.addEventListener("loadedmetadata", async () => {
  // Some files (notably WebM written by browsers and screen recorders) don't
  // store their length, so `duration` is Infinity. Seeking far past the end
  // makes the browser scan the file and report the real duration.
  if (!Number.isFinite(video.duration)) {
    setStatus("Measuring video length…");
    await new Promise((resolve) => {
      const done = () => {
        if (!Number.isFinite(video.duration)) return;
        video.removeEventListener("durationchange", done);
        resolve();
      };
      video.addEventListener("durationchange", done);
      video.currentTime = 1e9;
    });
    delete video.dataset.ready; // force a fresh seek to frame 0 below
  }
  state.frameCount = Math.max(1, Math.floor(video.duration * fps()));
  $("scrub").max = state.frameCount - 1;
  $("video-panel").hidden = false;
  setStatus(`Video loaded: ${video.videoWidth}×${video.videoHeight}, ${video.duration.toFixed(1)} s.`);
  await showFrame(0);
});

video.addEventListener("error", () => {
  setStatus(
    "This browser cannot play that video. Phones often record HEVC (H.265), which some browsers can't decode. " +
      "Try recording in “Most Compatible” / H.264 mode, or open the page in another browser.",
    true,
  );
});

$("fps").addEventListener("change", () => {
  if (state.source !== "video" || !video.duration) return;
  state.frameCount = Math.max(1, Math.floor(video.duration * fps()));
  $("scrub").max = state.frameCount - 1;
  showFrame(Math.min(state.frameIndex, state.frameCount - 1));
});

// Seeking is asynchronous, and the slider can fire faster than the video
// can seek. Only the newest request matters, so there is one "wanted" frame
// and at most one seek loop running, which keeps going until the video
// shows the wanted frame.
let wantedFrame = 0;
let seeking = null; // the running seek loop's promise, or null

async function showFrame(index) {
  wantedFrame = Math.max(0, Math.min(state.frameCount - 1, index));
  $("scrub").value = wantedFrame;
  // Wait for any running loop. It follows `wantedFrame`, so it may already
  // bring us to the right frame.
  while (seeking) await seeking;
  if (state.frameIndex === wantedFrame && video.dataset.ready) return;
  // Note: `seeking` is assigned *before* anything awaits, and cleared in
  // `finally` *after*. Clearing it inside the loop instead would break
  // when the loop finishes without awaiting: it would clear `seeking`
  // before this line had even set it.
  seeking = seekLoop();
  try {
    await seeking;
  } finally {
    seeking = null;
  }
}

async function seekLoop() {
  while (state.frameIndex !== wantedFrame || !video.dataset.ready) {
    const target = wantedFrame;
    // Aim for the middle of the frame's time slot, so rounding never lands
    // us on the neighbouring frame.
    await seekVideo((target + 0.5) / fps());
    state.frameIndex = target;
    video.dataset.ready = "1";
  }
  $("frame-readout").textContent =
    `Frame ${state.frameIndex} of ${state.frameCount - 1} · ${(state.frameIndex / fps()).toFixed(3)} s`;
}

function seekVideo(time) {
  return new Promise((resolve) => {
    const done = () => {
      video.removeEventListener("seeked", done);
      // `seeked` can fire before the new frame is actually painted.
      // requestVideoFrameCallback waits for the paint where supported, with
      // a timeout in case no new frame comes (e.g. same frame).
      if ("requestVideoFrameCallback" in video) {
        const t = setTimeout(resolve, 250);
        video.requestVideoFrameCallback(() => { clearTimeout(t); resolve(); });
      } else {
        resolve();
      }
    };
    video.addEventListener("seeked", done);
    video.currentTime = Math.min(time, Math.max(0, video.duration - 1e-3));
  });
}

$("scrub").addEventListener("input", (e) => showFrame(parseInt(e.target.value, 10)));
$("prev-frame").addEventListener("click", () => showFrame(wantedFrame - 1));
$("next-frame").addEventListener("click", () => showFrame(wantedFrame + 1));

function grabFrame() {
  const c = document.createElement("canvas");
  c.width = video.videoWidth;
  c.height = video.videoHeight;
  c.getContext("2d").drawImage(video, 0, 0);
  return { index: state.frameIndex, canvas: c };
}

$("set-a").addEventListener("click", async () => {
  await showFrame(wantedFrame);
  state.A = grabFrame();
  // If B is not chosen, the frame after A is used when analysing. That is
  // done at analysis time rather than now, so the video never jumps away
  // from where the user is scrubbing.
  updateFrames();
});
$("set-b").addEventListener("click", async () => {
  await showFrame(wantedFrame);
  state.B = grabFrame();
  updateFrames();
});

function updateFrames() {
  $("frames").hidden = !(state.A || state.B);
  for (const [key, thumb, cap] of [["A", "thumb-a", "cap-a"], ["B", "thumb-b", "cap-b"]]) {
    const f = state[key];
    const t = $(thumb);
    if (f) {
      t.width = 320;
      t.height = Math.round((320 * f.canvas.height) / f.canvas.width);
      t.getContext("2d").drawImage(f.canvas, 0, 0, t.width, t.height);
      $(cap).textContent = state.source === "demo" ? `Frame ${key}: synthetic` : `Frame ${key}: #${f.index}`;
    } else {
      t.width = t.height = 0;
      $(cap).textContent = key === "B" && state.A ? "Frame B: the next frame (automatic)" : `Frame ${key}: not set`;
    }
  }
  $("analyze").disabled = !(state.A && (state.B || state.source === "video"));
}

// ---------------------------------------------------------------------------
// Step 1b: synthetic demo (no video needed)
// ---------------------------------------------------------------------------

const DEMO_W = 384, DEMO_H = 256;
const CX = 191.5, CY = 127.5;
const DEMO_FLOWS = {
  // Peak speed 1.5 px/frame at a 40 px core: within PIV's gradient limits.
  vortex: { type: "vortex", x: CX, y: CY, circulation: 2 * Math.PI * 40 * 1.5, core_radius: 40 },
  source: { type: "source", x: CX, y: CY, strength: 2 * Math.PI * 40 * 1.5, core_radius: 40 },
  sink: { type: "source", x: CX, y: CY, strength: -2 * Math.PI * 40 * 1.5, core_radius: 40 },
  cylinder: { type: "cylinder", x: CX, y: CY, u: 1.2, v: 0, radius: 40 },
  uniform: { type: "uniform", u: 2.3, v: -1.2 },
  // Viscous channel flows. Couette: still wall along the top edge, speed
  // rising 0.01 px/frame per px (2.5 px/frame at the bottom). Poiseuille:
  // horizontal channel nearly filling the frame, peak 1.5 px/frame.
  couette: { type: "couette", x: 0, y: 0, angle: 0, shear_rate: 0.01 },
  poiseuille: { type: "poiseuille", x: CX, y: CY, angle: 0, half_width: 120, max_speed: 1.5 },
};

$("demo-button").addEventListener("click", async () => {
  const flow = DEMO_FLOWS[$("demo-select").value];
  setStatus("Generating synthetic particle images…");
  try {
    const seed = 1 + Math.floor(Math.random() * 1e6);
    const bytes = await ask({ type: "synthetic", flow, width: DEMO_W, height: DEMO_H, seed });
    const toCanvas = (offset) => {
      const c = document.createElement("canvas");
      c.width = DEMO_W;
      c.height = DEMO_H;
      const px = new Uint8ClampedArray(bytes.buffer, offset, DEMO_W * DEMO_H * 4);
      c.getContext("2d").putImageData(new ImageData(px, DEMO_W, DEMO_H), 0, 0);
      return c;
    };
    state.source = "demo";
    $("video-panel").hidden = true;
    state.A = { index: 0, canvas: toCanvas(0) };
    state.B = { index: 1, canvas: toCanvas(DEMO_W * DEMO_H * 4) };
    // Demo frames are "1 frame apart at 30 fps", uncalibrated.
    $("fps").value = 30;
    $("slowmo").value = 1;
    updateFrames();
    setStatus("Synthetic frames ready. Press “Analyse flow”.");
    await analyze();
  } catch (err) {
    setStatus(`Demo failed: ${err.message}`, true);
  }
});

// ---------------------------------------------------------------------------
// Step 2: analysis
// ---------------------------------------------------------------------------

$("analyze").addEventListener("click", () => analyze());

async function analyze() {
  if (state.A && !state.B && state.source === "video") {
    if (state.A.index + 1 >= state.frameCount) {
      setStatus("Frame A is the last frame. Choose an earlier frame A, or a frame B.", true);
      return;
    }
    await showFrame(state.A.index + 1);
    state.B = grabFrame();
    updateFrames();
  }
  let { A, B } = state;
  if (!A || !B) return;
  if (A.index === B.index && state.source === "video") {
    setStatus("Frames A and B are the same frame. Pick two different frames.", true);
    return;
  }
  const notes = [];
  if (B.index < A.index) {
    [A, B] = [B, A]; // always measure forwards in time
    notes.push({ code: "swapped_frames", level: "info", text: "Frame B was before frame A, so they were swapped to measure forwards in time." });
  }

  // Time between the frames, in real seconds.
  const frames = B.index - A.index;
  const dt = frames / fps() / Math.max(1, parseFloat($("slowmo").value) || 1);

  // Scale down large frames for speed. pxScale = original px per analysis px.
  const maxW = parseInt($("max-width").value, 10);
  const fullW = A.canvas.width, fullH = A.canvas.height;
  const s = maxW && fullW > maxW ? maxW / fullW : 1;
  const w = Math.round(fullW * s), h = Math.round(fullH * s);
  const pxScale = fullW / w;
  const windowSize = parseInt($("window-size").value, 10);
  if (w < windowSize * 2 || h < windowSize * 2) {
    setStatus(`The frames (${w}×${h}) are too small for ${windowSize} px windows. Choose a smaller window.`, true);
    return;
  }
  const imgA = toImageData(A.canvas, w, h);
  const imgB = toImageData(B.canvas, w, h);
  notes.push(...imageWarnings(imgA, imgB, w));

  $("analyze").disabled = true;
  setStatus("Starting…");
  try {
    // Send copies (the worker takes ownership of transferred buffers, and we
    // still need frame A for drawing).
    const a = imgA.data.slice().buffer;
    const b = imgB.data.slice().buffer;
    const result = await ask(
      { type: "analyze", a, b, width: w, height: h, options: { window_size: windowSize } },
      [a, b],
      (msg) => setStatus(msg),
    );
    const calib = calibration(pxScale, dt);
    state.last = { ...result, frame: imgA, calib, frames, width: w, height: h };
    const warnings = [...notes, ...resultWarnings(result, windowSize, Math.min(w, h))];
    state.last.warnings = warnings;
    showResults(state.last, warnings);
    setStatus(`Done in ${Math.round(result.ms)} ms.`);
  } catch (err) {
    setStatus(`Analysis failed: ${err.message}`, true);
  } finally {
    $("analyze").disabled = false;
  }
}

function toImageData(canvas, w, h) {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(canvas, 0, 0, w, h);
  return ctx.getImageData(0, 0, w, h);
}

/** Unit conversion from analysis pixels and frame pairs to real units. */
function calibration(pxScale, dt) {
  const mm = parseFloat($("mm-per-px").value);
  const hasMm = mm > 0;
  // Length of one analysis pixel, in mm (calibrated) or original video px.
  const L = hasMm ? mm * pxScale : pxScale;
  const lu = hasMm ? "mm" : "px";
  return {
    calibrated: hasMm,
    dt,
    length: (px) => px * L, // position, radius
    speed: (disp) => (disp * L) / dt, // per second
    rate: (perPair) => perPair / dt, // vorticity, divergence: 1/s
    area: (a) => (a * L * L) / dt, // circulation, source strength
    lengthUnit: lu,
    speedUnit: `${lu}/s`,
    areaUnit: `${lu}²/s`,
  };
}

// ---------------------------------------------------------------------------
// Warnings for bad recordings
// ---------------------------------------------------------------------------

function imageWarnings(imgA, imgB, width) {
  const out = [];
  const n = imgA.width * imgA.height;
  // Brightness histogram (256 bins) of frame A, plus an identical-frame check.
  const hist = new Uint32Array(256);
  let sum = 0, sum2 = 0, same = true;
  const a = imgA.data, b = imgB.data;
  for (let i = 0; i < a.length; i += 4) {
    const g = Math.round(0.299 * a[i] + 0.587 * a[i + 1] + 0.114 * a[i + 2]);
    hist[g]++;
    sum += g / 255;
    sum2 += (g / 255) ** 2;
    if (same && (a[i] !== b[i] || a[i + 1] !== b[i + 1] || a[i + 2] !== b[i + 2])) same = false;
  }
  const quantile = (q) => {
    let acc = 0;
    for (let g = 0; g < 256; g++) if ((acc += hist[g]) >= q * n) return g / 255;
    return 1;
  };
  const mean = sum / n;
  const std = Math.sqrt(Math.max(0, sum2 / n - mean * mean));
  if (same) {
    out.push({ code: "identical_frames", level: "bad", text: "Frames A and B are identical. The video may repeat frames, or the frame rate setting is higher than the file's real frame rate." });
  }
  // A dark background with bright particles is ideal for PIV, so "dark"
  // means even the brightest 1 % of pixels are dim.
  if (quantile(0.99) < 0.25) out.push({ code: "dark", level: "warn", text: "The video is very dark. Add more light so the particles show clearly." });
  if (quantile(0.01) > 0.8) out.push({ code: "bright", level: "warn", text: "The video is very bright (washed out). Reduce exposure or lighting." });
  if (std < 0.03) out.push({ code: "low_texture", level: "warn", text: "Very little texture. PIV needs visible particles (pepper, glitter, coffee grounds) spread through the fluid." });
  if (hist[255] + hist[254] + hist[253] > 0.05 * n) out.push({ code: "glare", level: "warn", text: "Large overexposed areas (glare or reflections). They hide particles and can give false vectors." });
  if (width < 320) out.push({ code: "low_resolution", level: "warn", text: "Low resolution. Results will be coarse; record at a higher resolution if you can." });
  return out;
}

/** Largest displacement the multi-pass PIV can follow: a quarter of its
 * coarsest window. The core tries windows 4× and 2× the chosen size,
 * keeping those that fit at least twice into the shorter image side
 * (see `analyze` in core/src/piv.rs). */
function maxTrackable(windowSize, shortSide) {
  const coarsest = [4, 2, 1].map((k) => k * windowSize).find((w) => w === windowSize || 2 * w <= shortSide);
  return coarsest / 4;
}

function resultWarnings({ field, analysis, fit }, windowSize, shortSide) {
  const out = [];
  const st = analysis.stats;
  const speeds = [];
  for (let i = 0; i < field.u.length; i++) if (field.status[i] === 0) speeds.push(Math.hypot(field.u[i], field.v[i]));
  const p95 = percentile(speeds, 0.95);

  if (st.valid_fraction < 0.6) {
    out.push({ code: "few_valid", level: "bad", text: `Only ${pct(st.valid_fraction)} of vectors are reliable. Common causes: too few particles, motion blur, particles moving in and out of focus, or frames too far apart.` });
  } else if (st.valid_fraction < 0.85) {
    out.push({ code: "some_rejected", level: "warn", text: `${pct(1 - st.valid_fraction)} of vectors were rejected (red crosses). Results there are filled in from neighbours only.` });
  }
  const limit = maxTrackable(windowSize, shortSide);
  if (p95 > 0.8 * limit) {
    out.push({ code: "too_fast", level: "warn", text: `Particles move up to ${p95.toFixed(1)} px between frames, close to the ~${Math.round(limit)} px this analysis can follow. Pick frames closer together, or a larger window.` });
  }
  if (p95 < 0.05) {
    out.push({ code: "no_motion", level: "bad", text: "Almost no motion detected. Is the fluid moving? Try frames further apart." });
  } else if (p95 < 0.3) {
    out.push({ code: "too_slow", level: "warn", text: `Particles move only ${p95.toFixed(2)} px between frames, so measurement noise is large compared to the motion. Pick frames further apart.` });
  }
  const best = fit.best_index !== null ? fit.fits[fit.best_index] : null;
  if (best && best.model === "uniform" && best.r_squared > 0.9) {
    out.push({ code: "camera_motion", level: "info", text: "The whole picture moves the same way. If the fluid itself shouldn't move like that, the camera probably moved: use a tripod." });
  }
  if (best && best.r_squared < 0.5) {
    out.push({ code: "poor_fit", level: "info", text: `No simple model explains this flow well (best R² = ${best.r_squared.toFixed(2)}). It may be a combination of several patterns, or turbulent.` });
  }
  if (analysis.critical_points.length > 8) {
    out.push({ code: "many_points", level: "info", text: `${analysis.critical_points.length} critical points were found. Some may be noise rather than real features.` });
  }
  return out;
}

const pct = (x) => `${Math.round(100 * x)} %`;

// ---------------------------------------------------------------------------
// Step 3: results
// ---------------------------------------------------------------------------

for (const name of ["frame", "vorticity", "arrows", "particles", "points"]) {
  $(`layer-${name}`).addEventListener("change", (e) => view.setLayer(name, e.target.checked));
}
$("anim-speed").addEventListener("input", (e) => view.setSpeed(2 ** parseFloat(e.target.value)));

function showResults(r, warnings) {
  $("step-results").hidden = false;
  const ul = $("warnings");
  ul.replaceChildren(
    ...warnings.map((w) => {
      const li = document.createElement("li");
      li.className = w.level === "bad" ? "bad" : "";
      li.textContent = w.text;
      return li;
    }),
  );
  view.setData(r.frame, r.field, r.analysis);
  renderLegend(r);
  renderSummary(r);
  document.dispatchEvent(new CustomEvent("flowverse:results", { detail: r }));
  $("step-results").scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderLegend(r) {
  const c = r.calib;
  const wScale = c.rate(view.vorticityScale);
  const neg = vorticityColour(-1), pos = vorticityColour(1);
  const rgba = ([R, G, B, A]) => `rgba(${R},${G},${B},${(A / 255).toFixed(2)})`;
  $("legend").innerHTML = `
    <span class="scale">↺ ${fmt(-wScale)} <span class="bar" style="background:linear-gradient(90deg,${rgba(neg)},transparent,${rgba(pos)})"></span> ${fmt(wScale)} ↻ 1/s</span>
    <span>Vorticity: red = clockwise spin, blue = anticlockwise</span>
    <span>Arrow of one grid step = ${fmt(c.speed(view.arrowReference))} ${c.speedUnit}</span>`;
}

function renderSummary(r) {
  const { analysis, fit, calib: c } = r;
  const st = analysis.stats;
  const best = fit.best_index !== null ? fit.fits[fit.best_index] : null;
  const rows = (pairs) => pairs.map(([k, v]) => `<tr><th>${k}</th><td>${v}</td></tr>`).join("");

  let verdict;
  if (!best) {
    verdict = `<p class="verdict">No motion to classify.</p>`;
  } else if (best.r_squared < 0.5) {
    verdict = `<p class="verdict">No simple pattern fits well. Closest: <strong>${modelLabel(best)}</strong> (R² = ${best.r_squared.toFixed(2)}).</p>`;
  } else {
    verdict = `<p class="verdict">This looks like <strong>${modelLabel(best)}</strong>. The model explains ${pct(Math.max(0, best.r_squared))} of the measured motion (R² = ${best.r_squared.toFixed(3)}).</p>`;
  }

  const fitRows = fit.fits
    .map((f) => `<tr><td>${f.model}${f === best ? " ✓" : ""}</td><td>${f.r_squared.toFixed(3)}</td>
      <td><span class="r2" style="width:${Math.round(100 * Math.max(0, f.r_squared))}px"></span></td></tr>`)
    .join("");

  const counts = Object.entries(st.critical_point_counts).map(([k, n]) => `${n} ${k}`).join(", ") || "none";

  $("summary").innerHTML = `
    <div>
      ${verdict}
      ${best ? `<h3>Best-fit parameters</h3><table>${rows(modelParameters(best, c))}</table>` : ""}
      <h3>Model comparison</h3>
      <table><tr><th>Model</th><th>R²</th><th></th></tr>${fitRows}</table>
    </div>
    <div>
      <h3>Measurements</h3>
      <table>${rows([
        ["Mean speed", `${fmt(c.speed(st.mean_speed))} ${c.speedUnit}`],
        ["Max speed", `${fmt(c.speed(st.max_speed))} ${c.speedUnit}`],
        ["Max |vorticity|", `${fmt(c.rate(st.max_abs_vorticity))} 1/s`],
        ["Mean divergence", `${fmt(c.rate(st.mean_divergence))} 1/s`],
        ["Critical points", counts + (st.weak_critical_points_hidden ? ` (${st.weak_critical_points_hidden} weak ones in still regions hidden)` : "")],
        ["Reliable vectors", `${st.valid_count} of ${st.vector_count} (${pct(st.valid_fraction)})`],
        ["Time between frames", `${fmt(c.dt * 1000)} ms (${r.frames} frame${r.frames === 1 ? "" : "s"})`],
        ["Analysed at", `${r.width}×${r.height} px, ${r.field.window_size} px windows`],
      ])}</table>
      ${c.calibrated ? "" : `<p class="source">Enter millimetres per pixel to get real units.</p>`}
    </div>`;
}

function modelLabel(f) {
  return {
    uniform: "a uniform flow", vortex: "a vortex", source: "a source", sink: "a sink", cylinder: "flow past a cylinder",
    couette: "Couette (shear) flow", poiseuille: "Poiseuille (channel) flow",
  }[f.model] ?? f.model;
}

/** Best-fit parameters as [label, value] rows, in calibrated units. */
function modelParameters(f, c) {
  const p = f.parameters;
  const pos = () => [["Centre", `(${fmt(c.length(p.x))}, ${fmt(c.length(p.y))}) ${c.lengthUnit}`]];
  switch (f.model) {
    case "uniform":
      return [["Speed", `${fmt(c.speed(Math.hypot(p.u, p.v)))} ${c.speedUnit}`], ["Direction", direction(p.u, p.v)]];
    case "vortex":
      return [
        ...pos(),
        // Positive Γ is clockwise on screen (image y points down).
        ["Circulation Γ", `${fmt(c.area(Math.abs(p.circulation)))} ${c.areaUnit} (${p.circulation >= 0 ? "clockwise" : "anticlockwise"})`],
        ["Core radius", `${fmt(c.length(p.core_radius))} ${c.lengthUnit}`],
        ["Peak speed", `${fmt(c.speed(Math.abs(p.circulation) / (2 * Math.PI * Math.max(p.core_radius, 1e-9))))} ${c.speedUnit}`],
      ];
    case "source":
    case "sink":
      return [
        ...pos(),
        ["Strength m", `${fmt(c.area(p.strength))} ${c.areaUnit}`],
        ["Core radius", `${fmt(c.length(p.core_radius))} ${c.lengthUnit}`],
      ];
    case "cylinder":
      return [
        ...pos(),
        ["Radius", `${fmt(c.length(p.radius))} ${c.lengthUnit}`],
        ["Stream speed", `${fmt(c.speed(Math.hypot(p.u, p.v)))} ${c.speedUnit}`],
        ["Stream direction", direction(p.u, p.v)],
      ];
    // Viscous channel flows. `angle` is the flow line's direction (radians,
    // y down); (x, y) lies on the still wall / the centreline.
    case "couette":
      return [
        ["Shear rate", `${fmt(c.rate(Math.abs(p.shear_rate)))} 1/s`],
        ["Still wall through", `(${fmt(c.length(p.x))}, ${fmt(c.length(p.y))}) ${c.lengthUnit}`],
        ["Layers run", axis(p.angle)],
      ];
    case "poiseuille":
      return [
        ["Channel width", `${fmt(c.length(2 * p.half_width))} ${c.lengthUnit}`],
        ["Centreline through", `(${fmt(c.length(p.x))}, ${fmt(c.length(p.y))}) ${c.lengthUnit}`],
        ["Peak speed", `${fmt(c.speed(p.max_speed))} ${c.speedUnit}`],
        ["Mean speed (⅔ peak)", `${fmt(c.speed((2 * p.max_speed) / 3))} ${c.speedUnit}`],
        ["Wall shear rate", `${fmt(c.rate((2 * p.max_speed) / p.half_width))} 1/s`],
        ["Flow direction", direction(Math.cos(p.angle), Math.sin(p.angle))],
      ];
    default:
      return [];
  }
}

/** Screen direction of (u, v) as a compass-style word (y points down). */
function direction(u, v) {
  const names = ["right", "down-right", "down", "down-left", "left", "up-left", "up", "up-right"];
  const ang = Math.atan2(v, u);
  return names[(Math.round(ang / (Math.PI / 4)) + 8) % 8];
}

/** A line's direction as an angle from horizontal (0–180°, measured
 * anticlockwise on screen), e.g. "12° from horizontal". */
function axis(angle) {
  // Round first, then wrap, so a hair under 180° reads as 0°.
  const deg = Math.round((((-angle * 180) / Math.PI) % 180) + 180) % 180;
  return `${deg}° from horizontal`;
}

/** Three significant figures, without silly exponents for normal values. */
export function fmt(x) {
  if (!Number.isFinite(x)) return "–";
  const a = Math.abs(x);
  if (a !== 0 && (a < 0.001 || a >= 1e6)) return x.toExponential(2);
  return Number(x.toPrecision(3)).toString();
}

// ---------------------------------------------------------------------------

function setStatus(text, isError = false) {
  const el = $("status");
  el.textContent = text;
  el.classList.toggle("error", isError);
}
