import test from "node:test";
import assert from "node:assert/strict";
import {
  START, END, fitBigram, distribution,
  initialAssociations, swapAssociations, lookupAssociation, assessAssociation
} from "../src/models.mjs";

test("Unicode codepoints are counted, not UTF-16 surrogate halves", () => {
  const model = fitBigram("🤖🤖");
  assert.equal(model.sampleSize, 2);
  assert.deepEqual(model.vocabulary, ["⟨END⟩", "🤖"].sort());
  assert.equal(model.counts.get("🤖").get("🤖"), 1);
  assert.equal(model.counts.get("🤖").get(END), 1);
});
test("a transition model counts start and ending boundaries", () => {
  const model = fitBigram("aba");
  assert.equal(model.counts.get(START).get("a"), 1);
  assert.equal(model.counts.get("a").get("b"), 1);
  assert.equal(model.counts.get("a").get(END), 1);
  assert.equal(model.counts.get("b").get("a"), 1);
});
test("add-one smoothing normalizes the full next-character vocabulary", () => {
  const model = fitBigram("aba");
  const rows = distribution(model, "a", 1);
  assert.ok(Math.abs(rows.reduce((sum, r) => sum + r.probability, 0) - 1) < 1e-12);
  assert.equal(rows.find((r) => r.token === "b").probability, 0.4);
  assert.equal(rows.find((r) => r.token === "a").probability, 0.2);
});
test("zero smoothing assigns probability zero to unobserved transitions", () => {
  const model = fitBigram("aba");
  const rows = distribution(model, "a", 0);
  assert.equal(rows.find((r) => r.token === "a").probability, 0);
  assert.equal(rows.find((r) => r.token === "b").probability, 0.5);
});
test("out-of-sample context without smoothing has no claimed evidence", () => {
  const model = fitBigram("ab");
  assert.ok(distribution(model, "unknown", 0).every((r) => r.probability === 0));
  assert.ok(Math.abs(distribution(model, "unknown", 1).reduce((n,r) => n + r.probability,0) - 1) < 1e-12);
});
test("invalid samples and smoothing are rejected", () => {
  assert.throws(() => fitBigram(" "), RangeError);
  assert.throws(() => fitBigram("x".repeat(321)), RangeError);
  assert.throws(() => fitBigram(42), TypeError);
  assert.throws(() => distribution(fitBigram("ab"), "a", 0.5), RangeError);
});
test("association lookup changes under value swap but keys remain stable", () => {
  const before = initialAssociations();
  const after = swapAssociations(before);
  assert.deepEqual(before, { A: 1, B: 2 });
  assert.deepEqual(after, { A: 2, B: 1 });
  assert.equal(lookupAssociation(after, "A"), 2);
  assert.equal(lookupAssociation(after, "B"), 1);
  assert.deepEqual(swapAssociations(after), before);
});
test("counterfactual quiz checks the current state, not a memorized answer", () => {
  const before = initialAssociations();
  const after = swapAssociations(before);
  assert.equal(assessAssociation(before, "A", "1"), true);
  assert.equal(assessAssociation(after, "A", "1"), false);
  assert.equal(assessAssociation(after, "A", "2"), true);
  assert.throws(() => assessAssociation(after, "A", ""), RangeError);
  assert.throws(() => lookupAssociation(after, "C"), RangeError);
});