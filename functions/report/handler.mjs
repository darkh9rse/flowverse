// The report endpoint's logic, with every outside dependency passed in
// (AI client, error classes, rate limiter). index.mjs wires in the real
// ones; tests pass fakes. Uses only standard web APIs (Request, Response),
// so it runs the same on Netlify, in Node tests and in a browser.

import { sanitizeSummary, SYSTEM_PROMPT, userMessage } from "./summary.mjs";
import { userKey } from "./rate-limit.mjs";

export const MODEL = "claude-opus-5-5";
const MAX_BODY_BYTES = 16 * 1024;

const json = (status, body, headers = {}) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store", ...headers },
  });

/**
 * deps:
 *   getClient()  → an Anthropic client, or null when no API key is set
 *   Anthropic    → the SDK class (for its typed error classes)
 *   getLimiter() → a rate limiter (see rate-limit.mjs)
 *   salt         → secret salt for hashing IPs
 *   log          → console-like logger
 */
export function createHandler({ getClient, Anthropic, getLimiter, salt, log = console }) {
  return async function handler(req, context) {
    if (req.method !== "POST") return json(405, { error: "method_not_allowed" }, { allow: "POST" });

    const client = getClient();
    // Not an error for the user: the page falls back to its built-in text.
    if (!client) return json(503, { error: "not_configured" });

    const text = await req.text();
    if (new TextEncoder().encode(text).length > MAX_BODY_BYTES) return json(413, { error: "too_large" });
    let body;
    try {
      body = JSON.parse(text);
    } catch {
      return json(400, { error: "invalid_json" });
    }
    const checked = sanitizeSummary(body?.summary);
    if (!checked.ok) return json(400, { error: "invalid_summary", detail: checked.error });

    // Rate limit per user (hashed IP). Netlify provides the client IP as
    // context.ip; the header is a fallback.
    const ip = context?.ip ?? req.headers.get("x-nf-client-connection-ip") ?? "unknown";
    const verdict = await getLimiter().take(await userKey(salt, ip));
    if (!verdict.allowed) {
      return json(
        429,
        { error: "rate_limited", scope: verdict.scope, retry_after: verdict.retryAfter },
        { "retry-after": String(verdict.retryAfter) },
      );
    }

    try {
      const response = await client.beta.messages.create({
        model: MODEL,
        // Generous ceiling (thinking also uses tokens); the prompt asks for
        // under 250 words.
        max_tokens: 4000,
        // A short explanation of a handful of numbers: low effort is plenty
        // and keeps the reply fast, within serverless time limits.
        output_config: { effort: "low" },
        // If a safety classifier declines, retry inside the same call on a
        // fallback model chosen by the API.
        betas: ["server-side-fallback-2026-07-01"],
        fallbacks: "default",
        system: SYSTEM_PROMPT,
        messages: [{ role: "user", content: userMessage(checked.summary) }],
      });

      if (response.stop_reason === "refusal") {
        return json(502, { error: "refused" });
      }
      const report = response.content
        .filter((b) => b.type === "text")
        .map((b) => b.text)
        .join("")
        .trim();
      if (!report) return json(502, { error: "empty_response" });
      return json(200, { report, model: response.model, truncated: response.stop_reason === "max_tokens" });
    } catch (err) {
      // Most specific first. Details go to the log, not to the user.
      if (err instanceof Anthropic.AuthenticationError) {
        log.error("report: API key rejected");
        return json(503, { error: "not_configured" });
      }
      if (err instanceof Anthropic.RateLimitError) {
        return json(503, { error: "ai_busy" });
      }
      if (err instanceof Anthropic.APIConnectionTimeoutError) {
        return json(504, { error: "ai_timeout" });
      }
      if (err instanceof Anthropic.APIError) {
        log.error(`report: API error ${err.status}: ${err.message}`);
        return json(502, { error: "ai_error" });
      }
      log.error("report: unexpected error", err);
      return json(500, { error: "internal" });
    }
  };
}
