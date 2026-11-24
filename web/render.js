// Draws a measured flow field on a canvas: the video frame, a vorticity
// colour map, velocity arrows, critical points, and animated tracer particles.
//
// Everything is drawn in *analysis pixel* coordinates (the coordinates the
// PIV used). The canvas is scaled up for sharp lines on small frames and
// high-density screens.

const STATUS_VALID = 0;

export class FlowView {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    // The slow-to-draw layers (frame, colour map, arrows, points) are drawn
    // once into this offscreen canvas. Each animation frame only copies it
    // and draws the particles on top.
    this.staticLayer = document.createElement("canvas");
    this.layers = { frame: true, vorticity: true, arrows: true, particles: true, points: true };
    this.speedFactor = 1;
    this.particles = [];
    this.data = null;
    this.raf = 0;
    document.addEventListener("visibilitychange", () => this.#updateLoop());
    // Phones fire `resize` while scrolling (the address bar slides in and
    // out). Only redraw when the canvas size really changes.
    window.addEventListener("resize", () => {
      if (this.data && this.#resize()) this.#drawStatic();
    });
  }

  /** frame: ImageData at analysis size; field/analysis: worker results. */
  setData(frame, field, analysis) {
    const speeds = [];
    for (let i = 0; i < field.u.length; i++) {
      if (field.status[i] === STATUS_VALID) speeds.push(Math.hypot(field.u[i], field.v[i]));
    }
    const vort = analysis.vorticity.filter((w) => w !== null).map(Math.abs);
    this.data = {
      frame,
      field,
      analysis,
      // Robust scales: the 95th/98th percentile ignores a few wild values.
      speedScale: percentile(speeds, 0.95) || 1,
    };
    // Colour scale: the 98th-percentile |vorticity|, but never below half
    // the vorticity of a feature one grid step across. Without that floor,
    // a flow with no rotation at all would stretch pure noise to full
    // colour.
    const floor = (0.5 * this.data.speedScale) / field.step;
    this.data.vorticityScale = Math.max(percentile(vort, 0.98), floor) || 1;
    // Arrow length: the 95th-percentile speed draws 90 % of a grid step.
    this.arrowScale = (0.9 * field.step) / this.data.speedScale;
    this.frameCanvas = document.createElement("canvas");
    this.frameCanvas.width = frame.width;
    this.frameCanvas.height = frame.height;
    this.frameCanvas.getContext("2d").putImageData(frame, 0, 0);
    this.#resize();
    this.#resetParticles();
    this.#drawStatic();
    this.#updateLoop();
  }

  setLayer(name, on) {
    this.layers[name] = on;
    if (!this.data) return;
    this.#drawStatic();
    this.#updateLoop();
  }

  /** Particle speed multiplier (1 = default). */
  setSpeed(factor) {
    this.speedFactor = factor;
  }

  get vorticityScale() {
    return this.data?.vorticityScale ?? 1;
  }

  /** Displacement (px per frame pair) that draws an arrow one grid step long. */
  get arrowReference() {
    return this.data ? this.data.field.step / this.arrowScale : 1;
  }

  // ---- sizing -----------------------------------------------------------

  #resize() {
    const { frame } = this.data;
    // Scale so that one canvas pixel ≈ one screen pixel, at most 3×.
    const shown = this.canvas.getBoundingClientRect().width || frame.width;
    const want = (shown * (window.devicePixelRatio || 1)) / frame.width;
    const s = Math.min(3, Math.max(1, Math.round(want)));
    const changed =
      s !== this.scale || this.canvas.width !== frame.width * s || this.canvas.height !== frame.height * s;
    // Assigning width/height clears a canvas, even to the same value, so
    // only touch them when something changed.
    if (changed) {
      this.scale = s;
      for (const c of [this.canvas, this.staticLayer]) {
        c.width = frame.width * s;
        c.height = frame.height * s;
      }
    }
    return changed;
  }

  // ---- static layers ----------------------------------------------------

  #drawStatic() {
    const { frame, field, analysis } = this.data;
    const ctx = this.staticLayer.getContext("2d");
    ctx.setTransform(this.scale, 0, 0, this.scale, 0, 0);
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, frame.width, frame.height);

    if (this.layers.frame) {
      ctx.drawImage(this.frameCanvas, 0, 0);
      // Dim the video a little so overlays stand out.
      if (this.layers.vorticity || this.layers.arrows || this.layers.particles) {
        ctx.fillStyle = "rgba(0,0,0,0.35)";
        ctx.fillRect(0, 0, frame.width, frame.height);
      }
    }
    if (this.layers.vorticity) this.#drawVorticity(ctx, field, analysis.vorticity);
    if (this.layers.arrows) this.#drawArrows(ctx, field);
    if (this.layers.points) this.#drawCriticalPoints(ctx, analysis.critical_points);
    this.#present();
  }

  #drawVorticity(ctx, field, vorticity) {
    // One pixel per grid node in a tiny canvas, then stretched with
    // smoothing on, so the browser interpolates between nodes for free.
    const { cols, rows, step } = field;
    const tiny = document.createElement("canvas");
    tiny.width = cols;
    tiny.height = rows;
    const img = tiny.getContext("2d").createImageData(cols, rows);
    for (let i = 0; i < cols * rows; i++) {
      const w = vorticity[i];
      if (w === null) continue; // stays transparent
      const [r, g, b, a] = vorticityColour(w / this.data.vorticityScale);
      img.data.set([r, g, b, a], i * 4);
    }
    tiny.getContext("2d").putImageData(img, 0, 0);
    ctx.save();
    ctx.imageSmoothingEnabled = true;
    // Tiny-canvas pixel centres (i + 0.5) must land on node positions.
    const x0 = field.x[0] - step / 2;
    const y0 = field.y[0] - step / 2;
    ctx.drawImage(tiny, x0, y0, cols * step, rows * step);
    ctx.restore();
  }

  #drawArrows(ctx, field) {
    const k = this.arrowScale;
    const lw = Math.max(1, field.step / 14);
    ctx.lineCap = "round";
    for (let i = 0; i < field.x.length; i++) {
      const x = field.x[i];
      const y = field.y[i];
      if (field.status[i] !== STATUS_VALID) {
        // Rejected vector: a small red cross, so the user sees where PIV
        // failed instead of a silent gap.
        const s = field.step * 0.12;
        ctx.strokeStyle = "rgba(255,80,80,0.9)";
        ctx.lineWidth = lw;
        ctx.beginPath();
        ctx.moveTo(x - s, y - s); ctx.lineTo(x + s, y + s);
        ctx.moveTo(x + s, y - s); ctx.lineTo(x - s, y + s);
        ctx.stroke();
        continue;
      }
      const dx = field.u[i] * k;
      const dy = field.v[i] * k;
      const len = Math.hypot(dx, dy);
      if (len < 0.5) continue;
      // Dark outline first, then the bright arrow: readable on any background.
      for (const [colour, width] of [["rgba(0,0,0,0.6)", lw + 2], ["#ffd84a", lw]]) {
        ctx.strokeStyle = colour;
        ctx.lineWidth = width;
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(x + dx, y + dy);
        // Arrow head: two short lines at ±25° from the shaft.
        const head = Math.min(len * 0.4, field.step * 0.3);
        const ang = Math.atan2(dy, dx);
        for (const t of [ang + Math.PI - 0.45, ang + Math.PI + 0.45]) {
          ctx.moveTo(x + dx, y + dy);
          ctx.lineTo(x + dx + head * Math.cos(t), y + dy + head * Math.sin(t));
        }
        ctx.stroke();
      }
    }
  }

  #drawCriticalPoints(ctx, points) {
    const r = Math.max(5, this.data.field.step * 0.35);
    ctx.font = `${Math.max(11, r * 1.1)}px system-ui, sans-serif`;
    ctx.textBaseline = "middle";
    for (const p of points) {
      ctx.lineWidth = 2;
      for (const colour of ["rgba(0,0,0,0.7)", "#ffffff"]) {
        ctx.strokeStyle = colour;
        ctx.lineWidth = colour === "#ffffff" ? 2 : 4;
        ctx.beginPath();
        if (p.kind === "saddle") {
          ctx.moveTo(p.x - r, p.y - r); ctx.lineTo(p.x + r, p.y + r);
          ctx.moveTo(p.x + r, p.y - r); ctx.lineTo(p.x - r, p.y + r);
        } else if (p.kind.endsWith("node")) {
          ctx.moveTo(p.x, p.y - r); ctx.lineTo(p.x + r, p.y);
          ctx.lineTo(p.x, p.y + r); ctx.lineTo(p.x - r, p.y); ctx.closePath();
        } else {
          ctx.arc(p.x, p.y, r, 0, 2 * Math.PI);
        }
        ctx.stroke();
      }
      ctx.lineWidth = 3;
      ctx.strokeStyle = "rgba(0,0,0,0.7)";
      ctx.strokeText(p.kind, p.x + r + 4, p.y);
      ctx.fillStyle = "#fff";
      ctx.fillText(p.kind, p.x + r + 4, p.y);
    }
  }

  // ---- particles ---------------------------------------------------------

  #resetParticles() {
    const { frame } = this.data;
    const n = Math.round(Math.min(1500, Math.max(200, (frame.width * frame.height) / 450)));
    this.particles = Array.from({ length: n }, () => this.#spawn(true));
  }

  #spawn(randomAge) {
    const { field } = this.data;
    const xMin = field.x[0], yMin = field.y[0];
    const xMax = field.x[field.x.length - 1], yMax = field.y[field.y.length - 1];
    const life = 60 + Math.random() * 120;
    return {
      x: xMin + Math.random() * (xMax - xMin),
      y: yMin + Math.random() * (yMax - yMin),
      age: randomAge ? Math.random() * life : 0,
      life,
      trail: [],
    };
  }

  /** Bilinear interpolation of the measured displacement at (x, y),
   *  skipping invalid vectors. null where no data. */
  velocityAt(x, y) {
    const { field } = this.data;
    const gx = (x - field.x[0]) / field.step;
    const gy = (y - field.y[0]) / field.step;
    const c = Math.floor(gx), r = Math.floor(gy);
    if (c < 0 || r < 0 || c >= field.cols - 1 || r >= field.rows - 1) return null;
    const fx = gx - c, fy = gy - r;
    let u = 0, v = 0, wsum = 0;
    for (const [dc, dr, w] of [[0, 0, (1 - fx) * (1 - fy)], [1, 0, fx * (1 - fy)], [0, 1, (1 - fx) * fy], [1, 1, fx * fy]]) {
      const i = (r + dr) * field.cols + (c + dc);
      if (field.status[i] !== STATUS_VALID) continue;
      u += w * field.u[i];
      v += w * field.v[i];
      wsum += w;
    }
    // Mostly surrounded by bad vectors: don't guess.
    return wsum < 0.5 ? null : [u / wsum, v / wsum];
  }

  #stepParticles() {
    // The fastest typical particle moves ~1.2 analysis px per animation
    // frame at speed 1×, whatever the real flow speed. Smooth to watch.
    const k = (1.2 / this.data.speedScale) * this.speedFactor;
    for (let i = 0; i < this.particles.length; i++) {
      const p = this.particles[i];
      const vel = this.velocityAt(p.x, p.y);
      p.age++;
      if (!vel || p.age > p.life) {
        this.particles[i] = this.#spawn(false);
        continue;
      }
      p.trail.push(p.x, p.y);
      if (p.trail.length > 16) p.trail.splice(0, 2); // keep the last 8 points
      p.x += vel[0] * k;
      p.y += vel[1] * k;
    }
  }

  #drawParticles() {
    const ctx = this.ctx;
    ctx.lineWidth = 1.5;
    ctx.lineCap = "round";
    for (const p of this.particles) {
      if (p.trail.length < 2) continue;
      // Fade in at birth and out at death.
      const fade = Math.min(1, p.age / 15, (p.life - p.age) / 15);
      ctx.strokeStyle = `rgba(140,230,255,${(0.85 * fade).toFixed(3)})`;
      ctx.beginPath();
      ctx.moveTo(p.trail[0], p.trail[1]);
      for (let j = 2; j < p.trail.length; j += 2) ctx.lineTo(p.trail[j], p.trail[j + 1]);
      ctx.lineTo(p.x, p.y);
      ctx.stroke();
    }
  }

  // ---- drawing loop ------------------------------------------------------

  #present() {
    const ctx = this.ctx;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.drawImage(this.staticLayer, 0, 0);
    if (this.layers.particles) {
      ctx.setTransform(this.scale, 0, 0, this.scale, 0, 0);
      this.#drawParticles();
    }
  }

  #updateLoop() {
    const shouldRun = this.data && this.layers.particles && !document.hidden;
    if (shouldRun && !this.raf) {
      const tick = () => {
        this.#stepParticles();
        this.#present();
        this.raf = requestAnimationFrame(tick);
      };
      this.raf = requestAnimationFrame(tick);
    } else if (!shouldRun && this.raf) {
      cancelAnimationFrame(this.raf);
      this.raf = 0;
      if (this.data) this.#present();
    }
  }
}

/** Diverging colour for t in [-1, 1]: blue (negative) ↔ red (positive).
 *  Opacity grows with |t|, so calm regions show the video underneath. */
export function vorticityColour(t) {
  const m = Math.min(1, Math.abs(t));
  const alpha = Math.round(255 * 0.8 * Math.sqrt(m));
  return t >= 0 ? [230, 60, 50, alpha] : [40, 110, 235, alpha];
}

export function percentile(values, q) {
  if (!values.length) return 0;
  const sorted = Float64Array.from(values).sort();
  return sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))];
}
