/**
 * Pedagogical, deterministic toy models. They do not execute or reproduce
 * the frozen scientific experiments in Wakeel v1.58.
 */
export const START = "⟨START⟩";
export const END = "⟨END⟩";
export const MAX_CODEPOINTS = 320;

/** Fit a character-level Bigram count table from a short Unicode string. */
export function fitBigram(raw) {
  if (typeof raw !== "string") throw new TypeError("Corpus must be text");
  const tokens = Array.from(raw.normalize("NFC").trim());
  if (!tokens.length) throw new RangeError("Corpus is empty");
  if (tokens.length > MAX_CODEPOINTS) throw new RangeError("Corpus exceeds limit");
  const vocabulary = [...new Set([...tokens, END])].sort();
  const counts = new Map();
  let previous = START;
  for (const token of [...tokens, END]) {
    if (!counts.has(previous)) counts.set(previous, new Map());
    const row = counts.get(previous);
    row.set(token, (row.get(token) || 0) + 1);
    previous = token;
  }
  return { vocabulary, contexts: [...counts.keys()], counts, sampleSize: tokens.length };
}

/** A probability distribution over every possible next token. */
export function distribution(model, previous, smoothing = 1) {
  if (!model || !Array.isArray(model.vocabulary) || !(model.counts instanceof Map)) {
    throw new TypeError("Invalid Bigram model");
  }
  if (smoothing !== 0 && smoothing !== 1) throw new RangeError("Unsupported smoothing");
  const row = model.counts.get(previous) || new Map();
  const observations = [...row.values()].reduce((a, b) => a + b, 0);
  const denominator = observations + smoothing * model.vocabulary.length;
  return model.vocabulary.map((token) => ({
    token,
    count: row.get(token) || 0,
    probability: denominator ? ((row.get(token) || 0) + smoothing) / denominator : 0
  })).sort((a, b) => b.probability - a.probability || a.token.localeCompare(b.token));
}

/** Key/value lookup analogy only; not Transformer attention or KV cache. */
export function initialAssociations() {
  return { A: 1, B: 2 };
}
export function swapAssociations(pairs) {
  assertPairs(pairs);
  return { A: pairs.B, B: pairs.A };
}
export function lookupAssociation(pairs, key) {
  assertPairs(pairs);
  if (key !== "A" && key !== "B") throw new RangeError("Unknown key");
  return pairs[key];
}
export function assessAssociation(pairs, key, answer) {
  if (answer !== "1" && answer !== "2") throw new RangeError("Select an answer");
  return lookupAssociation(pairs, key) === Number(answer);
}
function assertPairs(pairs) {
  if (!pairs || ![1, 2].includes(pairs.A) || ![1, 2].includes(pairs.B) || pairs.A === pairs.B) {
    throw new TypeError("Invalid association state");
  }
}