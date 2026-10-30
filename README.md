# وكيل | Wakeel

**Arabic-first and English-equivalent learning through experiments.**

## Repository status — October 30, 2026

This repository previously held only a placeholder README. The first feature branch introduces a **standalone v0.22 interactive-lab interface prototype** to validate bilingual interaction and scientific representation boundaries. It is **not** a migration of the existing Wakeel scientific platform and **must not** be described as v1.58 production content.

### Local run

Requires Node.js 20+ and Python 3.10+; no npm dependencies or external network calls.

~~~sh
npm run check
npm run serve
~~~

Open http://localhost:8080 and use the Arabic/English switch. Two small demonstrators are included:

1. **Character Bigram**: an actual count-based probability distribution computed from user-supplied text, with optional add-one smoothing.
2. **Key/value association**: an intentionally simple lookup counterfactual with a value-swap control. **Not Transformer Attention or a KV cache.**

The accessible static page uses semantic controls, supports RTL/LTR layout, and labels illustrative results as such. Browser/assistive-technology acceptance is not yet complete.

### Scientific baseline: not yet imported

The owner-provided canonical candidate is **Wakeel v1.58.0 / WKL-024BC**, delivered as an independently verified development source archive (651 source files) and public-site artifact (332 deployable files), dated October 22, 2026. Its recorded gates: 602/602 JavaScript, 24/24 Python, 250 static references, and 331/331 local HTTP checks. This **does not** mean a full scientific peer review, bilingual human approval, or deployed-browser acceptance.

The baseline is presently available in the project delivery bundle but **not yet represented in Git history here**. Do not replace or rewrite its original scientific engines, progress storage/JSON v5, review provenance, or licensed materials during import. Treat source imports and the new UI integration as separate reviewed changes. See [integration contract](docs/WKL-V022-INTEGRATION-CONTRACT.md).

### Current milestone

- [x] Branch-based lab runtime prototype, two interactive examples, bilingual interface, deterministic unit tests.
- [ ] Import the **complete** clean WKL-024BC source, preserving hashes and confidential/public distribution boundaries.
- [ ] Run the unchanged v1.58 release verification and record the results on the new repository.
- [ ] Integrate the new lab runtime with the original nine laboratories and existing 18 lesson investigations.
- [ ] Validate deployed HTTPS, keyboard and screen reader interactions, mobile behavior, human scientific and bilingual review.

**Legal:** The Tongjiang Lab project was studied for interaction patterns. No Tongjiang course code or lesson assets are copied here; this implementation is original.

**License:** No repository-wide license is asserted for newly created content until ownership and publication terms are confirmed; the imported v1.58 source includes its own MIT LICENSE and must retain its notices.
