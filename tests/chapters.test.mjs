import test from "node:test";
import assert from "node:assert/strict";
import { renderApprovedChapters } from "../scripts/build-learning.mjs";

test("all nine original chapter texts render in both languages with pinned source provenance", () => {
 const out=renderApprovedChapters();
 assert.equal(out.schema,"wakeel-v022-bilingual-chapter-render/v1");
 assert.equal(out.count,18);
 assert.equal(Object.keys(out.chapters).length,9);
 for(let i=1;i<=9;i++){
  const c=out.chapters["CH-"+i];
  assert.ok(c.ar.html.startsWith("<h1>"));
  assert.ok(c.en.html.startsWith("<h1>"));
  assert.ok(c.ar.source_bytes > 1000);
  assert.ok(c.en.source_bytes > 1000);
  assert.match(c.ar.source_sha256,/^[0-9a-f]{64}$/);
  assert.match(c.en.source_sha256,/^[0-9a-f]{64}$/);
  assert.ok(c.ar.source_path.endsWith("/ar.md"));
  assert.ok(c.en.source_path.endsWith("/en.md"));
 }
});
test("chapter 9 scientific comparison tables and code blocks are retained", () => {
 const c=renderApprovedChapters().chapters["CH-9"];
 for(const lang of ["ar","en"]){
  assert.match(c[lang].html,/<table>/);
  assert.match(c[lang].html,/<pre><code/);
  assert.ok(!/<script[\s>]/i.test(c[lang].html));
 }
});
test("public HTML does not surface archival experiments or review source files", () => {
 const c=renderApprovedChapters();
 const combined=JSON.stringify(c);
 assert.ok(!combined.includes("runs.log\\\""));
 assert.ok(!combined.includes("results.json\\\""));
 for(const article of Object.values(c.chapters)){
  assert.ok(!("review" in article));
  assert.ok(!("results" in article));
 }
});
