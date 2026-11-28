// Web Worker: runs the WebAssembly PIV off the main thread, so the page
// stays responsive (scrolling, buttons, animations) while it computes.
//
// The page sends {id, type, ...}; we reply {id, ok, result} or
// {id, ok: false, error}, plus {id, progress} messages along the way.

import init, { run_piv, analyze_field, fit_flow, synthetic_pair, remove_background } from "./pkg/flowverse_wasm.js";

// Start loading the .wasm file immediately; each request awaits it.
const ready = init();

self.onmessage = async (event) => {
  const { id, type } = event.data;
  try {
    await ready;
    if (type === "analyze") {
      const { width, height, options, background } = event.data;
      let a = new Uint8Array(event.data.a), b = new Uint8Array(event.data.b);
      const t0 = performance.now();
      if (background) {
        // Optional: subtract the still scene (median of sampled frames).
        self.postMessage({ id, progress: "Removing the still background…" });
        const both = remove_background(new Uint8Array(background.frames), background.count, a, b, width, height);
        a = both.subarray(0, width * height * 4);
        b = both.subarray(width * height * 4);
      }
      self.postMessage({ id, progress: "Measuring motion (PIV)…" });
      const fieldJson = run_piv(a, b, width, height, JSON.stringify(options));
      self.postMessage({ id, progress: "Finding vortices and stagnation points…" });
      const analysis = JSON.parse(analyze_field(fieldJson));
      self.postMessage({ id, progress: "Fitting flow models…" });
      const fit = JSON.parse(fit_flow(fieldJson));
      const ms = performance.now() - t0;
      self.postMessage({ id, ok: true, result: { field: JSON.parse(fieldJson), analysis, fit, ms } });
    } else if (type === "synthetic") {
      const { flow, width, height, seed } = event.data;
      const bytes = synthetic_pair(JSON.stringify(flow), width, height, seed);
      // Transfer the buffer instead of copying it.
      self.postMessage({ id, ok: true, result: bytes }, [bytes.buffer]);
    } else {
      throw new Error(`unknown request type ${type}`);
    }
  } catch (err) {
    self.postMessage({ id, ok: false, error: String(err?.message ?? err) });
  }
};
