// Netlify Function: POST /api/report
//
// Turns the numeric analysis summary into a plain-language report with
// Claude. Receives only computed numbers, never images or video (see
// summary.mjs). The website works fully without this function: when it is
// missing, unconfigured or failing, the page shows its built-in template.
//
// Environment variables (Netlify → Site configuration → Environment):
//   ANTHROPIC_API_KEY            required to enable AI reports
//   RATE_LIMIT_SALT              secret random string for hashing IPs
//   REPORT_LIMIT_PER_HOUR        per user, default 10
//   REPORT_LIMIT_PER_DAY         per user, default 30
//   REPORT_LIMIT_GLOBAL_PER_DAY  whole site, default 1000

import Anthropic from "@anthropic-ai/sdk";
import { getStore } from "@netlify/blobs";
import { createHandler } from "./handler.mjs";
import { createRateLimiter, memoryStore } from "./rate-limit.mjs";

const env = (name, fallback) => {
  const v = Number.parseInt(process.env[name] ?? "", 10);
  return Number.isFinite(v) && v >= 0 ? v : fallback;
};

let client = null;
function getClient() {
  if (!process.env.ANTHROPIC_API_KEY) return null;
  // Netlify cuts synchronous functions off after 10 s by default. Give up a
  // little earlier, without retries, so the page gets a clean error and
  // shows the built-in report instead.
  client ??= new Anthropic({ timeout: 9_500, maxRetries: 0 });
  return client;
}

// Netlify Blobs needs the request context, so the store is created on first
// use. If Blobs is unavailable (e.g. some local setups), fall back to memory,
// which still limits bursts within one function instance.
const fallbackStore = memoryStore();
function getLimiter() {
  let store;
  try {
    const blobs = getStore({ name: "report-rate-limits", consistency: "strong" });
    store = {
      get: (key) => blobs.get(key, { type: "json" }),
      set: (key, value) => blobs.setJSON(key, value),
    };
  } catch {
    store = fallbackStore;
  }
  return createRateLimiter({
    store,
    perHour: env("REPORT_LIMIT_PER_HOUR", 10),
    perDay: env("REPORT_LIMIT_PER_DAY", 30),
    globalPerDay: env("REPORT_LIMIT_GLOBAL_PER_DAY", 1000),
  });
}

if (!process.env.RATE_LIMIT_SALT) {
  console.warn("report: RATE_LIMIT_SALT is not set; using a fixed salt. Set it to a random secret.");
}

export default createHandler({
  getClient,
  Anthropic,
  getLimiter,
  salt: process.env.RATE_LIMIT_SALT ?? "flowverse-default-salt",
});

export const config = { path: "/api/report" };
