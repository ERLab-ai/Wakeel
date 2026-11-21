# Wakeel v0.22 — Chapter 4 frozen context-order metrics and arithmetic exploration

**Classification:** a bilingual, SHA-verified read-only view of the **existing** Chapter 4 N-gram metrics, alongside a **distinct bounded mathematical exploration** of dense table size. Neither is a new scientific training run or an independent scientific sign-off.

## Scientific provenance

- Immutable input: `baseline/v0.21.3/content/chapter-04/results.json`.
- The source is verified as raw bytes using the `content/chapter-04/results.json` digest recorded in `baseline/v0.21.3/PUBLIC-MANIFEST.json`.
- Both the source-declared simulator (`simulate_chapter4-1.0`) and input hash for `content/chapter-03/examples.json` must match the frozen manifest.
- The build-only function `projectFrozenChapter4Evidence` projects **only** approved k=0..3 per-language aggregate fields: selected validation alpha, training/validation/test negative-log loss, validation/test unseen-context rates, theoretical dense cells and actually stored context–character records. It includes output/context vocabulary sizes and source SHA-256.
- **Excluded:** raw messages, case studies, followers, model generations, context strings and the Backoff results. The original source results file is not shipped. The full approved chapter *prose* remains available separately in the pre-existing SHA-pinned chapter reader.

## Read-only reference values

| Language | Context order | Validation loss | Test loss | Unseen validation contexts | Dense cells | Stored context–character records |
|---|---:|---:|---:|---:|---:|---:|
| Arabic | 0 | 3.1310 | 3.1757 | 0% | 38 | 37 |
| Arabic | 1 | 2.6560 | 2.6380 | 0.75% | 1,482 | 202 |
| Arabic | 2 | 2.2915 | 2.2239 | 23.31% | 57,798 | 318 |
| Arabic | 3 | 2.4659 | 2.3961 | 39.85% | 2,254,122 | 373 |
| English | 0 | 2.9783 | 3.0255 | 0% | 36 | 35 |
| English | 1 | 2.2232 | 2.2234 | 1.23% | 1,332 | 194 |
| English | 2 | 1.9625 | 1.9494 | 9.20% | 49,284 | 325 |
| English | 3 | 2.1224 | 2.0234 | 26.99% | 1,823,508 | 392 |

The source selected smoothing `alpha` using each model's validation set. These are *scoped historical outcomes*: for this reference corpus, moving to Trigram improved validation/test loss compared with Bigram, whereas 4-gram became sparser and had worse validation/test loss than Trigram. This is **not** evidence that the same optimal context length transfers to other training sets or models. The chapter acknowledges that the test set has been viewed before, so this must not be reported as independent generalization evidence.

### Backoff is deliberately excluded

Chapter 4 also includes **Stupid Backoff** experiments, but their scores are **unnormalized**. Their negative-log scores are **not directly comparable** to proper cross-entropy metrics from normalized N-gram distributions. The evidence table omits them rather than inventing a comparable probability interpretation.

## Learner-controlled arithmetic

A second, explicitly separate panel computes:

```
dense_cells = context_vocabulary_size ** k * output_vocabulary_size
```

Allowed inputs:
- Context vocabulary: integer **2–64**; includes the start symbol in this chapter.
- Output vocabulary: integer **2–64**; excludes the context-only start symbol.
- Context order `k`: integer **0–3**.

By default the controls take the frozen reference vocabulary sizes (Arabic 39 and 38, English 37 and 36) and use `k=2`. The result is exactly 57,798 for Arabic and 49,284 for English. Changing the values **does not** change the frozen corpus measurements above and does **not** allocate a table or measure runtime, memory usage, prediction accuracy, or test loss.

The published site continues to use exactly **six** allowlisted files and only extends `src/models.mjs`, `src/learning-view.mjs` and the existing `data/learning.json`. No student progress, scoring or research results are changed.

## Automated acceptance and open gates

- Tests validate the actual source SHA, simulator/input origin, per-language counts and theoretical formula, alpha chosen by validation, and consistent reference evaluation sizes.
- Negative tests reject source tampering, invalid model records, nonfinite losses, inconsistent dense cells, incorrect alpha, and unbounded inputs.
- A real Chromium test verifies the Arabic/English table values, source provenance, live calculator updates, bad-input handling, and chapter isolation.
- Human bilingual/scientific review, NVDA/VoiceOver, RTL math legibility, high zoom and external HTTPS acceptance remain **OPEN**. CI success does not certify them.

**Version boundary:** no file inside the immutable `baseline/v0.21.3/` directory is modified. v1.58/WKL releases are a separate workstream and not imported into Wakeel 2.0.
