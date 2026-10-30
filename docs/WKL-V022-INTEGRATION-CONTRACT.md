# WKL v0.22 — Interactive Lab Integration Contract (draft)

**Date:** 2026-10-30  
**Status:** independent functional spike; not scientifically certified; no baseline migration.

## Source of truth

Original candidate: WKL-024BC / Wakeel v1.58.0 (2026-10-22), 651 source files, 332 deployable files, 602 JS tests, 24 Python tests, 250 static references and 331 local HTTP matches. Figures are recorded in the owner-provided verification report; they have **not** been reproduced in this repository. Browser/HTTPS/assistive technology and independent scientific/linguistic review remain OPEN.

No full WKL-024BC source is committed yet. Until a verified source import lands, v0.22 does not supersede the earlier release or change learning progress JSON v5.

## Artifact provenance contract

Each future lab result must be tagged one of:

- illustration: hand-designed representation; cannot be asserted as evidence of a learned property;
- computation: deterministic equation/model with declared parameters and domain;
- validated_experiment: passed the specified scientific validation gate, with validation record;
- frozen_evidence: rendered from a versioned immutable result with provenance hash; no suggestion it was re-run.

UI must show the result type, provenance, domain and limits; outputs cannot silently change type. Historic evidence hashes are immutable.

## Guardrails

1. Experimental controls must not mutate immutable WKL-024BC results, reference source records, or the archived nine laboratories.
2. A newly selected input invalidates the prior active observation until the computation runs; review history must distinguish pre-registered hypotheses from post-hoc explanation.
3. A bounded toy model can demonstrate only the declared law and input domain. Bigram is a count-based character model, not a neural LLM. Association lookup is not the mechanism of Transformer Attention.
4. The same scientific result must be rendered for both Arabic and English. Direction, wording, accessibility, units and constraints must be checked independently.
5. Preserve the v1.58 progress schema (JSON v5) and prevent prototype state from writing into its keys.
6. Only deploy the built public artifact, never internal reviews, test fixtures, backup packages or unpublished reviewer notes.

## Implementation sequencing

- P0 — import complete clean source with integrity manifest; verify a repeatable build; no scientific behavior changes.
- P1 — model/result classification, provenance metadata, explicit Arabic/English UI and component contracts.
- P2 — integrate with the existing Bigram lab, then the chapter nine key/value investigations **using original engines and frozen evidence**, not this analogy.
- P3 — integrate the learning journey, knowledge atlas and navigation across beginner and advanced levels.
- P4 — mobile/keyboard/screen-reader and deployed HTTPS acceptance; separate independent scientific and language reviews.

## Spike acceptance and limitations

Current branch accepts **only** a local count model, interactive state transitions, and structural/bilingual UI. Tests cover Unicode, probability normalization, smoothing, value swaps and failure cases. No automated browser or deployed HTTP smoke test has been completed by this spike. Existing WKL-024BC scientific tests are not claimed to run here.
