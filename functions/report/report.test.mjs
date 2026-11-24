// Runs the shared report-function test cases with Node's built-in runner:
//   node --test functions/
import { test } from "node:test";
import assert from "node:assert/strict";
import { cases } from "./report.cases.mjs";

for (const [name, fn] of Object.entries(cases)) {
  test(name, () => fn(assert));
}
