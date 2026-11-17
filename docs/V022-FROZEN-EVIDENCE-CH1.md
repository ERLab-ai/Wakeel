# Wakeel v0.22 — Chapter 1 frozen aggregate evidence

**Scope:** one read-only, bilingual source-backed comparison in Chapter 1. This is NOT a new experiment, an interactive replay of the original simulator, a claim of human scientific approval, or a performance benchmark on unseen real-world messages.

## Approved source and derivation
- Immutable baseline: `baseline/v0.21.3/`.
- Input file: `content/chapter-01/results.json`, checked byte-for-byte against the exact SHA-256 in `PUBLIC-MANIFEST.json`.
- Projector: `scripts/frozen-evidence.mjs`, run only during `npm run build`.
- Allowed output, in the existing `dist/data/learning.json`: two languages (`ar`, `en`), two fixed splits (`seed`, `tests`), and exactly three numeric fields per split: `total`, `rules_correct`, and `nearest_example_correct`.
- Source identity and exact SHA-256 remain visible to the learner. Raw messages, individual row-level predictions, rule matches, logs, and original results JSON are never published by this new component.
- No new distribution file: the static website still consists of exactly six allowlisted assets.

## Learner-facing comparison
| Reference dataset | Messages | Rules correct | Nearest-example correct |
|---|---:|---:|---:|
| Original six messages used to write the rules | 6 | 6 | 6 |
| Nine alternate test phrasings | 9 | 4 | 8 |

The Arabic and English reference experiments have the same count values; the labels and source paragraphs are rendered independently in each language. The browser does not compute or alter the frozen results.

**Important scientific limits:** The six original messages were visible when writing the rules. The nine alternate phrasings are a small, task-specific reference set, not a probability sample. The comparison does not establish a generally superior method, statistical significance, production performance, or transfer to other datasets. No student progress, score, or mastery event is written from this evidence view.

## Verification
- Positive tests assert exact SHA-provenance and the expected aggregate values.
- Negative tests reject byte changes, malformed JSON, missing bilingual evidence, impossible counts and aggregates inconsistent with their underlying per-message correctness flags.
- Browser tests check Chapter 1 only, both languages, numeric counts, source provenance and isolation from Chapter 9.
- Build and publishing gates continue to reject raw results URLs and require exactly six public assets.
- This is an additional project-owned presentation/projection test, **not** an independent scientific peer review.

## Still open
- Human review of Arabic/English interpretation and reader accessibility (NVDA/VoiceOver), zoom and mobile visual presentation.
- A separately designed and validated interactive experiment tied to the chapter's original methodology.
- Source-backed projections for other chapters only after a scoped field-level disclosure review.
