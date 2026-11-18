# Wakeel v0.22 — Chapter 2 frozen tokenization evidence

**Purpose:** show that a token count depends on the chosen segmentation procedure, using a SHA-verified read-only projection of the approved Wakeel 2.0 `v0.21.3` Chapter 2 reference results.

## Provenance and display contract

- Frozen input: `baseline/v0.21.3/content/chapter-02/results.json`.
- The source bytes are checked against `baseline/v0.21.3/PUBLIC-MANIFEST.json` before any value is projected.
- The source declares `generated_from = content/chapter-02/examples.json` and pinned educational tokenizer `edu-tok-0.1`.
- Only `sections.tokenization.ar` and `sections.tokenization.en` are read for the display. Do **not** confuse them with `sections.units`: those are different examples.
- Each sample's declared `ws_keep_count` and `edu_count` must equal the length of its corresponding `ws_keep_tokens` and `edu_tokens` list.
- Publication contains only the counts, the tokenizer version and the frozen source SHA-256. No raw tokenizer output is added to the projected catalog. Other approved chapter text is separately already published in the bilingual chapter reader.
- The existing six-file public distribution allowlist remains unchanged. No source results JSON or historical experimental archives are published.

## Exact values from the approved source

| Reference sentence language | Whitespace-separated units (punctuation retained) | `edu-tok-0.1` educational units |
|---|---:|---:|
| Arabic | 5 | 8 |
| English | 7 | 9 |

These are **different reference sentences with related, but not identical, meanings**. Neither a cross-language performance ranking, a pricing conclusion nor a general tokenization-efficiency claim follows from these values. A new tokenizer or different wording could change the counts. The browser only replays recorded counts, not the tokenizer implementation.

## Safeguards

- Source SHA-256 verification at build time; any byte change rejects the build.
- Negative tests reject source-version drift, missing records, wrong tokenizer identifier, counts inconsistent with token arrays, and invalid source origin.
- A real Chromium acceptance test checks the bilingual Chapter 2 view, expansion persistence on language switch, provenance, exact counts and absence of the panel in Chapter 3.
- Original `baseline/v0.21.3/` files remain untouched; the earlier Chapter 1 aggregates stay intact.
- Human scientific and translation review, NVDA/VoiceOver acceptance, visual/math readability and deployed HTTPS remain independent open gates.

**Classification:** approved-source evidence display with bounded verification, not an independently peer-reviewed scientific replication.
