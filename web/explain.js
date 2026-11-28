// "Explain this flow": a plain-language report of the results.
//
// The button sends a small summary of *computed numbers only* (no images,
// no video) to the site's /api/report function, which asks an AI model to
// explain them. If the function is missing (e.g. a local static server),
// not configured, rate-limited or failing, the page writes the report
// itself from a built-in template. The app works fully without the AI.

const REPORT_URL = "/api/report";
const TIMEOUT_MS = 20_000;

let latest = null; // { summary } for the current results

export function setupExplain(area) {
  document.addEventListener("flowverse:results", (e) => {
    latest = { summary: buildSummary(e.detail) };
    area.innerHTML = `
      <button class="button primary" type="button" id="explain-btn">Explain this flow</button>
      <details class="checklist"><summary>What gets sent?</summary>
        <p class="source">Only these computed numbers, never your video or images:</p>
        <pre class="report" id="explain-payload"></pre>
      </details>
      <div id="explain-out" aria-live="polite"></div>`;
    area.querySelector("#explain-payload").textContent = JSON.stringify(latest.summary, null, 2);
    area.querySelector("#explain-btn").addEventListener("click", () => explain(area));
  });
}

async function explain(area) {
  const btn = area.querySelector("#explain-btn");
  const out = area.querySelector("#explain-out");
  const { summary } = latest;
  btn.disabled = true;
  out.innerHTML = `<p class="source">Writing an explanation…</p>`;
  let reason;
  try {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), TIMEOUT_MS);
    const res = await fetch(REPORT_URL, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ summary }),
      signal: ctrl.signal,
    }).finally(() => clearTimeout(timer));
    const body = await res.json().catch(() => ({}));
    if (res.ok && typeof body.report === "string" && body.report.trim()) {
      show(out, body.report, "Written by an AI model from the numbers above. Check it against the measurements.");
      return;
    }
    reason = failureReason(res.status, body);
  } catch (err) {
    reason = err.name === "AbortError" ? "The AI explanation took too long." : "The AI explanation service couldn't be reached.";
  } finally {
    btn.disabled = false;
  }
  show(out, templateReport(summary), `${reason} This is the built-in explanation instead.`);
}

function failureReason(status, body) {
  if (status === 429) {
    const mins = Math.max(1, Math.round((body.retry_after ?? 3600) / 60));
    return `You've reached the limit for AI explanations (try again in about ${mins} min).`;
  }
  if (status === 404 || status === 405 || status === 501 || body.error === "not_configured") {
    return "AI explanations aren't set up on this site.";
  }
  return "The AI explanation isn't available right now.";
}

function show(out, text, note) {
  out.replaceChildren();
  const p = document.createElement("div");
  p.className = "report";
  p.textContent = text; // textContent, never innerHTML: the report is untrusted text
  const n = document.createElement("p");
  n.className = "source";
  n.textContent = note;
  out.append(p, n);
}

// ---------------------------------------------------------------------------
// Summary: the numbers sent to the function (schema: functions/report/summary.mjs)
// ---------------------------------------------------------------------------

export function buildSummary(r) {
  const { analysis, fit, calib: c } = r;
  const st = analysis.stats;
  const fits = fit.fits.map((f) => ({ model: f.model, r_squared: f.r_squared, parameters: physical(f, c) }));
  const best = fit.best_index !== null ? fit.fits[fit.best_index] : null;
  return {
    version: 1,
    units: { length: c.calibrated ? "mm" : "px", time: "s" },
    frame_interval_ms: c.dt * 1000,
    frames_apart: r.frames,
    image: { width: r.width, height: r.height },
    window_px: r.field.window_size,
    valid_fraction: st.valid_fraction,
    speed: { mean: c.speed(st.mean_speed), max: c.speed(st.max_speed) },
    vorticity: { max_abs: c.rate(st.max_abs_vorticity), mean: c.rate(st.mean_vorticity) },
    divergence: { mean: c.rate(st.mean_divergence), max_abs: c.rate(st.max_abs_divergence) },
    critical_points: st.critical_point_counts,
    // Positive circulation is clockwise on screen (image y points down).
    rotation: best?.model === "vortex" ? (best.parameters.circulation >= 0 ? "clockwise" : "anticlockwise") : null,
    best_model: best ? best.model : null,
    fits,
    warnings: (r.warnings ?? []).map((w) => w.code).filter(Boolean),
  };
}

/** Screen direction in degrees: 0 = right, 90 = up (image y points down). */
const directionDeg = (u, v) => ((Math.atan2(-v, u) * 180) / Math.PI + 360) % 360;

function physical(f, c) {
  const p = f.parameters;
  switch (f.model) {
    case "uniform":
      return { speed: c.speed(Math.hypot(p.u, p.v)), direction_deg: directionDeg(p.u, p.v) };
    case "vortex":
      return {
        x: c.length(p.x),
        y: c.length(p.y),
        circulation: c.area(Math.abs(p.circulation)),
        core_radius: c.length(p.core_radius),
        peak_speed: c.speed(Math.abs(p.circulation) / (2 * Math.PI * Math.max(p.core_radius, 1e-9))),
      };
    case "source":
    case "sink":
      return { x: c.length(p.x), y: c.length(p.y), strength: c.area(p.strength), core_radius: c.length(p.core_radius) };
    case "cylinder":
      return {
        x: c.length(p.x),
        y: c.length(p.y),
        radius: c.length(p.radius),
        stream_speed: c.speed(Math.hypot(p.u, p.v)),
        direction_deg: directionDeg(p.u, p.v),
      };
    case "couette":
      return {
        shear_rate: c.rate(Math.abs(p.shear_rate)),
        // Round before wrapping so a hair under 180° reads as 0°.
        axis_deg: Math.round(directionDeg(Math.cos(p.angle), Math.sin(p.angle))) % 180,
      };
    case "poiseuille":
      return {
        channel_width: c.length(2 * p.half_width),
        max_speed: c.speed(p.max_speed),
        mean_speed: c.speed((2 * p.max_speed) / 3),
        wall_shear_rate: c.rate((2 * p.max_speed) / p.half_width),
        direction_deg: directionDeg(Math.cos(p.angle), Math.sin(p.angle)),
      };
    default:
      return {};
  }
}

// ---------------------------------------------------------------------------
// Built-in template report (no AI)
// ---------------------------------------------------------------------------

const ADVICE = {
  identical_frames: "Frames A and B were identical, so nothing could be measured; check the frame rate setting.",
  swapped_frames: "Frame B came before frame A, so the two were swapped to measure forwards in time.",
  dark: "The video is dark. More light would make the particles clearer.",
  bright: "The video is washed out. Less light or exposure would help.",
  low_texture: "There is little visible texture. Add tracer particles such as pepper or glitter.",
  glare: "Glare or reflections hide parts of the picture. Light from the side or tilt the camera slightly.",
  low_resolution: "The video resolution is low, so the results are coarse.",
  few_valid: "Many vectors failed the quality checks, so treat the numbers with caution.",
  some_rejected: "Some vectors failed the quality checks and were left out.",
  too_fast: "Particles moved too far between the frames. Pick frames closer together.",
  too_slow: "Particles barely moved between the frames, so noise matters more. Pick frames further apart.",
  no_motion: "Almost no motion was detected.",
  camera_motion: "Everything moved together, which can also mean the camera moved. A tripod rules that out.",
  poor_fit: "None of the simple models fits well, so the flow is probably a mix of patterns or turbulent.",
  many_points: "Many critical points were found, so some are probably noise.",
};

const NAMES = {
  uniform: "uniform flow", vortex: "vortex", source: "source", sink: "sink", cylinder: "cylinder",
  couette: "Couette (shear) flow", poiseuille: "Poiseuille (channel) flow",
};

export function templateReport(s) {
  const L = s.units.length;
  const n = (x) => (x === null || x === undefined ? "?" : Number(Number(x).toPrecision(3)).toString());
  const words = (deg) => ["right", "up-right", "up", "up-left", "left", "down-left", "down", "down-right"][Math.round(deg / 45) % 8];
  const best = s.fits.find((f) => f.model === s.best_model);
  const out = [];

  if (!best) {
    out.push("Almost nothing moved between the two frames, so there is no flow pattern to describe.");
  } else {
    const p = best.parameters;
    const describe = {
      vortex: `The flow looks like a vortex: fluid circling around a centre, here ${s.rotation ?? "in one direction"} as seen in the video. Its core, the middle part that turns like a spinning disc, has a radius of about ${n(p.core_radius)} ${L}, and the fluid there moves at up to about ${n(p.peak_speed)} ${L}/s. Its circulation, a measure of the vortex's total strength, is about ${n(p.circulation)} ${L}²/s.`,
      source: `The flow looks like a source: fluid spreading outwards from one spot, as if welling up from below or coming out of a pipe. Its strength (how much fluid appears per second, per unit depth) is about ${n(p.strength)} ${L}²/s.`,
      sink: `The flow looks like a sink: fluid converging on one spot, as if draining away there. Its strength is about ${n(Math.abs(p.strength))} ${L}²/s flowing in.`,
      cylinder: `The flow looks like a stream passing a round obstacle about ${n(p.radius)} ${L} in radius. Far from it the fluid moves at about ${n(p.stream_speed)} ${L}/s towards the ${words(p.direction_deg)}; it splits around the obstacle, speeds up at its sides and slows down in front and behind.`,
      couette: `The flow looks like Couette flow: layers of fluid sliding past each other, as between a still wall and a moving one, along a line ${n(p.axis_deg)}° from horizontal. The speed grows steadily with distance from the still wall, by about ${n(p.shear_rate)} ${L}/s for every ${L} (the shear rate, ${n(p.shear_rate)} per second). Viscosity passes the moving wall's motion through the fluid layer by layer, and the fluid touching each wall moves with it (the no-slip condition). This is one of the few exact solutions of the Navier–Stokes equations.`,
      poiseuille: `The flow looks like Poiseuille flow: fluid pushed along a channel about ${n(p.channel_width)} ${L} wide, towards the ${words(p.direction_deg)}. The speed profile across the channel is a parabola: fastest in the middle, about ${n(p.max_speed)} ${L}/s, and zero at both walls, where friction (viscosity) holds the fluid still. The average speed is two thirds of the peak, about ${n(p.mean_speed)} ${L}/s. A steady pressure drop along the channel balances the friction; this is an exact solution of the Navier–Stokes equations.`,
      uniform: `Everything moves together in the same direction, towards the ${words(p.direction_deg)} at about ${n(p.speed)} ${L}/s.`,
    }[best.model];
    out.push(describe);

    const r2 = best.r_squared ?? 0;
    const how = r2 >= 0.9 ? "very well" : r2 >= 0.7 ? "reasonably well" : r2 >= 0.5 ? "only roughly" : "poorly";
    const others = s.fits
      .filter((f) => f !== best)
      .sort((a, b) => (b.r_squared ?? -1) - (a.r_squared ?? -1));
    const runnerUp = others[0];
    out.push(
      `This ${NAMES[best.model]} model fits the measurement ${how}: it explains ${Math.round(100 * Math.max(0, r2))} % of the measured motion (R² = ${n(r2)}).` +
        (runnerUp ? ` The next best model, a ${NAMES[runnerUp.model]}, explains ${Math.round(100 * Math.max(0, runnerUp.r_squared ?? 0))} %.` : "") +
        ` ${Math.round(100 * (s.valid_fraction ?? 0))} % of the measured vectors passed the quality checks.`,
    );
  }

  out.push(
    `Across the picture the average speed is ${n(s.speed.mean)} ${L}/s and the fastest is ${n(s.speed.max)} ${L}/s. ` +
      `The strongest local spin (vorticity) is ${n(s.vorticity.max_abs)} per second; the two frames were ${n(s.frame_interval_ms)} ms apart.`,
  );

  const advice = s.warnings.map((w) => ADVICE[w]).filter(Boolean);
  if (L === "px") advice.push("Lengths are in pixels because no calibration was entered; add millimetres per pixel for real units.");
  if (advice.length) out.push(advice.join(" "));
  return out.join("\n\n");
}
