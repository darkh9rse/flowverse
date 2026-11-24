// Per-user rate limiting for the report function.
//
// "User" means a salted SHA-256 hash of the client's IP address. The raw IP
// is never stored, and without the salt (an environment variable) the hash
// can't be reversed by trying every IP.
//
// Counters live in a small key/value store with `get(key)` / `set(key, obj)`.
// On Netlify that is Netlify Blobs (see index.mjs); tests and local runs use
// the in-memory store below. Three limits apply:
//   - per user per hour (stops bursts),
//   - per user per day,
//   - for everyone per day (caps the total AI bill).
//
// Concurrent requests can race and let one or two extra requests through.
// That is acceptable for cost protection and keeps the code simple.

const HOUR = 3600_000;
const DAY = 24 * HOUR;

/** Salted SHA-256 of the IP, as hex. Uses Web Crypto (Node 18+, browsers). */
export async function userKey(salt, ip) {
  const bytes = new TextEncoder().encode(`${salt}|${ip}`);
  const hash = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(hash)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

/** A store that lives in memory (per function instance). */
export function memoryStore() {
  const map = new Map();
  return {
    get: async (key) => (map.has(key) ? structuredClone(map.get(key)) : null),
    set: async (key, value) => void map.set(key, structuredClone(value)),
  };
}

export function createRateLimiter({ store, perHour, perDay, globalPerDay, now = () => Date.now() }) {
  return {
    /**
     * Check and, if allowed, count one request.
     * Returns { allowed: true } or { allowed: false, scope, retryAfter } with
     * retryAfter in whole seconds.
     */
    async take(key) {
      const t = now();
      const hour = Math.floor(t / HOUR);
      const day = Math.floor(t / DAY);
      const untilNextHour = Math.ceil(((hour + 1) * HOUR - t) / 1000);
      const untilNextDay = Math.ceil(((day + 1) * DAY - t) / 1000);

      // Counters reset by keeping the bucket number they belong to.
      const user = (await store.get(`user/${key}`)) ?? {};
      const h = user.hour?.bucket === hour ? user.hour : { bucket: hour, count: 0 };
      const d = user.day?.bucket === day ? user.day : { bucket: day, count: 0 };
      const globalKey = `global/${day}`;
      const g = (await store.get(globalKey)) ?? { count: 0 };

      if (h.count >= perHour) return { allowed: false, scope: "hour", retryAfter: untilNextHour };
      if (d.count >= perDay) return { allowed: false, scope: "day", retryAfter: untilNextDay };
      if (g.count >= globalPerDay) return { allowed: false, scope: "global", retryAfter: untilNextDay };

      h.count++;
      d.count++;
      g.count++;
      await store.set(`user/${key}`, { hour: h, day: d });
      await store.set(globalKey, g);
      return { allowed: true };
    },
  };
}
