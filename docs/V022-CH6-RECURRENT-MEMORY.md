# Wakeel v0.22 — Chapter 6 recurrent memory and length generalization

## Scientific scope and provenance

The new bilingual view presents read-only numeric evidence from
`baseline/v0.21.3/content/chapter-06/results.json`. The build verifies the
exact SHA-256 against the immutable `PUBLIC-MANIFEST.json`, simulator version
`simulate_chapter6-3.2`, input hash of Chapter 3 examples, train-only
vocabulary provenance, and the start-token exclusion. No original experiment
has been retrained.

### Two distinct protocols

The synthetic task retrieves the first character after random filler and a
question mark. Evaluation consumes the question mark, and loss is evaluated
at the answer position only.

| | Fixed-length | Mixed-length |
|---|---|---|
| Training | Separate model for each delay and seed | One model per seed trained on delays 2,4,5,6,7,8 |
| Training examples | 120/class = 240/model | 60/class/delay = 720/model |
| Test examples | 40/class/delay | 40/class/delay |
| Epochs / learning rate | 300 / 0.3 | 300 / 0.3 |
| Seeds | 42,43,44 | 42,43,44 |
| Delay 16 | Trained at 16 | Held-out length |

| Filler length | Fixed mean binary accuracy | Mixed mean binary accuracy |
|---:|---:|---:|
| 2 | 100% | 83.3% |
| 4 | 100% | 83.3% |
| 5 | 66.7% | 83.3% |
| 6 | 50% | 83.3% |
| 7 | 50% | 83.3% |
| 8 | 50% | 83.3% |
| 16 | 50% | 83.3% |

At delay 16, the fixed seeds have accuracies 0.5,0.5,0.5;
the mixed seeds have 1,0.5,1. The source means are rounded to three
decimal places. The new view exposes the three seed results and their
test answer NLLs, the fixed untrained binary baseline, and the fixed
Oracle sanity reference. It does not conceal failed seeds.

The comparison does not isolate length diversity from training-set size:
240 sequences versus 720. Equal epochs and nominal learning rate do not
imply equal data exposure or compute. Fixed-delay results are not a
memory curve measured on a single trained model. Delay 16 is held out
only for the mixed protocol. Two successful seeds show that this setup
can generalize to the held-out delay; they do not establish universal
memory capacity, a causal mechanism, or superiority of an architecture.
Lengths beyond 16 were not evaluated.

### Real-message RNN results

| Language | Parameters | Mean validation NLL | Mean test NLL |
|---|---:|---:|---:|
| Arabic | 1358 | 3.1235 | 3.1899 |
| English | 1308 | 2.9291 | 2.9962 |

These are separate from the synthetic task. Each mean comes from three
recorded seeds.

## Independent scalar-state illustration

The browser computes h(t+1)=rho*h(t), with initial state -1 or 1,
rho in {0,0.5,0.9,1}, and integer step count 0..16. All intermediate
states are visible. Default h(0)=1, rho=0.9, t=4 gives 0.6561.
The formula is a zero-intervening-input recurrence, not the trained tanh
RNN; it has no optimizer, learned weights, interfering filler, or
classifier. The absolute state magnitude is not recall probability.
The original scientific results are never modified by this exercise.

## Verification and publication boundaries

- Validate protocol declarations, test counts, domain flags, seed-level
  scores, aggregate means and NLLs within source rounding, fixed Oracle
  and label-swap controls, and real-message RNN means.
- Unit tests cover exact evidence, invalid or altered protocols,
  inconsistent results, source hashes, and scalar recurrence boundaries.
- Chromium test covers bilingual data, delay and seed detail, interactive
  scalar traces, language switching, and isolation from Chapter 7.
- Project only numeric aggregates and source metadata into existing
  `data/learning.json`; preserve the six-asset public allowlist.
- No raw corpus, generated sequences, checkpoints or archived research
  artifacts are published by this projector. No frozen source changes.
- Independent scientific and translation review, human accessibility
  validation, production HTTPS, and full reproducible RNN training in
  the browser remain open. Automated tests do not certify these.
