/**
 * Pedagogical, deterministic toy models. They do not execute or reproduce
 * the frozen scientific experiments in Wakeel v0.21.3.
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
/**
 * A bounded, single-transition additive-smoothing calculator.
 * Deliberately distinct from the frozen Chapter 3 corpus, tokenizer,
 * validation protocol and aggregate test losses.
 */
export function smoothedTransition(total,observed,vocabulary,alpha){
  if(!Number.isInteger(total)||total<0||total>100)throw new RangeError("Total transitions must be 0–100");
  if(!Number.isInteger(observed)||observed<0||observed>total)throw new RangeError("Observed transitions exceed total");
  if(!Number.isInteger(vocabulary)||vocabulary<2||vocabulary>32)throw new RangeError("Vocabulary size must be 2–32");
  if(![0.1,0.5,1].includes(alpha))throw new RangeError("Alpha must be a declared grid value");
  const probability=(observed+alpha)/(total+alpha*vocabulary);
  return {probability,loss:-Math.log(probability)};
}

/**
 * Explore the theoretical number of dense N-gram context/next-token cells.
 * This is a bounded arithmetic model, not an allocation, a training run,
 * a measured runtime/memory cost or a reproduction of Chapter 4 experiments.
 */
export function denseContextTableCells(contextVocabulary,outputVocabulary,order){
  if(!Number.isInteger(contextVocabulary)||contextVocabulary<2||contextVocabulary>64){
    throw new RangeError("Context vocabulary must be an integer from 2 to 64");
  }
  if(!Number.isInteger(outputVocabulary)||outputVocabulary<2||outputVocabulary>64){
    throw new RangeError("Output vocabulary must be an integer from 2 to 64");
  }
  if(!Number.isInteger(order)||order<0||order>3){
    throw new RangeError("Context order must be an integer from 0 to 3");
  }
  return contextVocabulary**order*outputVocabulary;
}

/**
 * One-dimensional quadratic-loss demonstration of a trainable parameter
 * versus a frozen one. NOT the Chapter 5 embedding MLP or its training data.
 * All arithmetic is bounded, synchronous and deterministic.
 */
export function toyScalarEmbeddingUpdate(initial,target,learningRate,steps){
 if(typeof initial!=="number"||!Number.isFinite(initial)||initial< -2||initial>2 ||
    typeof target!=="number"||!Number.isFinite(target)||target< -2||target>2){
  throw new RangeError("Scalar parameters must be finite numbers in [-2, 2]");
 }
 if(![0,0.25,0.5,1].includes(learningRate)){
  throw new RangeError("Learning rate must be one of 0, 0.25, 0.5, 1");
 }
 if(!Number.isInteger(steps)||steps<0||steps>20){
  throw new RangeError("Steps must be an integer from 0 to 20");
 }
 let trainable=initial;
 for(let i=0;i<steps;i++){
  trainable-=learningRate*(trainable-target);
 }
 const loss=(x)=>0.5*(x-target)**2;
 return {
  initial_loss:loss(initial),
  learned_value:trainable,
  learned_loss:loss(trainable),
  frozen_value:initial,
  frozen_loss:loss(initial)
 };
}

/**
 * Bounded scalar recurrence with zero intervening input. No RNN training,
 * classifier, or empirical recall probability is inferred from its state.
 */
export function toyMemoryRetention(initial,retention,delay){
 if(![-1,1].includes(initial))throw new RangeError("Initial state must be -1 or 1");
 if(![0,0.5,0.9,1].includes(retention))throw new RangeError("Unsupported retention factor");
 if(!Number.isInteger(delay)||delay<0||delay>16)throw new RangeError("Delay must be 0–16");
 const states=[initial];
 for(let i=0;i<delay;i++)states.push(states.at(-1)*retention);
 return {states,final_state:states.at(-1),magnitude:Math.abs(states.at(-1))};
}
