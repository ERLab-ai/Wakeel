# Wakeel v0.22 — Chapter 3 frozen metrics and bounded smoothing lab

**Classification:** a SHA-verified, read-only scientific reference display next to a separate *computed teaching exercise*. This PR does **not** reproduce a complete Chapter 3 training run and does not independently certify its scientific conclusions.

## Immutable reference data

- Frozen source: `baseline/v0.21.3/content/chapter-03/results.json`, with its exact raw-byte digest checked against the original `PUBLIC-MANIFEST.json`.
- Projection runs **only at build time** in `scripts/frozen-evidence.mjs`.
- Allowlisted fields: the `validation_loss` and `test_loss` for Unigram, Count Bigram, Neural Bigram in Arabic and English; the Count Bigram validation-selected `alpha`; and one 3-decimal worked-example transition probability.
- Raw training corpora, individual samples, generated text, hidden-size details, model weights, optimization curves, test messages, and logs are **not** disclosed by this projector. The original authored chapter prose is separately rendered and SHA-checked by the existing chapter reader.

### Frozen values from the scientific source

| Language | Model | Validation loss | Test loss |
|---|---|---:|---:|
| Arabic | Unigram | 3.1310 | 3.1757 |
| Arabic | Count Bigram (selected α=0.5) | 2.6560 | 2.6380 |
| Arabic | Neural Bigram | 2.9268 | 3.0616 |
| English | Unigram | 2.9783 | 3.0255 |
| English | Count Bigram (selected α=0.1) | 2.2232 | 2.2234 |
| English | Neural Bigram | 2.7885 | 2.9087 |

**Scope:** these results use the chapter's declared, per-language datasets and tokenization. Do not directly compare absolute losses across languages or tokenizers as if they were controlled experimental conditions. The 200-epoch neural configuration is not evidence of the neural architecture's optimal performance; the count model performs better *in this specific frozen run*, not generally.

## A separate learner-controlled calculation

Formula, for a **single transition**:

```
P(next | context) = (c + α) / (n + α × V)
loss = −ln(P(next | context))
```

- `n`: number of observations of a context, integer from 0 to 100.
- `c`: number of the target successor, integer from 0 to `n`.
- `V`: size of the output vocabulary, integer from 2 to 32.
- `α`: exactly 0.1, 0.5 or 1, mirroring the declared validation grid in the frozen Chapter 3 experiment.
- The default `n=1, c=1, V=6, α=1` yields `2/7 ≈ 0.286`, matching a reported mini-example probability to three decimals.
- No text or corpus is trained, no model weights are updated, no learner score or progress is stored.

**Critical interpretation boundary:** the interactive output is the loss for **one artificially specified transition**, not the mean validation/test loss of the frozen corpus and not a replay of its original model. The labels, method and interpretation limitations are available in both languages.

## Test & disclosure gates

- Unit tests pin the original SHA, exact dataset metrics, validation-selected alpha and mini-example probability, and reject malformed values or tampered sources.
- Numeric tests validate the formula and reject fractional/negative counts, impossible counts, out-of-range vocabulary and undeclared alpha values.
- Chromium tests verify bilingual numeric output, learner input updates, invalid-input messaging, and Chapter 4 isolation.
- The static-publication inventory still consists of exactly **six allowlisted files**; the new JavaScript function is inside the existing `src/models.mjs`, and Chapter 3 metadata is inside `data/learning.json`.
- All original `baseline/v0.21.3/` bytes remain unchanged.

**Open, not falsely passed:** human scientific and translation review, NVDA/VoiceOver, high zoom and RTL mathematical readability, deployed HTTPS acceptance, and any statistically valid broader model comparison.
