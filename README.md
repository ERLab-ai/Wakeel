# وكيل 2.0 | Wakeel 2.0

**تعلم بناء النماذج اللغوية بالتجربة والفرضية والدليل.**  
**Learn language-model construction through experiments, falsifiable hypotheses, and evidence.**

## Project source of truth

The approved **Wakeel 2.0** educational/scientific baseline is **v0.21.3**, comprising:
- **44** journey stations across **9** authored chapters.
- **389** topics in a separately managed knowledge registry.
- Arabic/English station parity accepted by the project review.
- Frozen chapter 8/9 experiments, results, and review evidence. Do not silently recompute or modify these.
- Chapter 6/7 rerun acceptance remains scoped to the documented reference environment.

**Canonical owner-provided archive:** `WAKEEL-2-V0.21.3-SOURCE-PACK.zip`  
**Archive SHA-256:** `a97899ca74ec4a269b9a677c9618b0901de07dcbb9fc65b97059b2bd9eac6659`  
**Verified package:** 149 files (148 of them recorded in the internal SHA-256 manifest); local byte checks passed as of 2026-10-30.

**Important:** That complete approved source has **not yet been imported to this GitHub repository**. It must first pass public-distribution review: the source ZIP includes archived research and review materials that must not be deployed by accident. See [v0.22 integration contract](docs/WAKEEL2-V022-INTEGRATION-CONTRACT.md) and [P0 baseline import issue](https://github.com/ERLab-ai/Wakeel/issues/2).

The previously referenced **WKL-024BC / Wakeel v1.58.0** is a **different archive/workstream** and **not the approved Wakeel 2.0 baseline** for this repository. Never substitute it for v0.21.3.

## v0.22 prototype (draft PR #1)

This development branch currently holds a **standalone bilingual interface spike**, not a replacement for the approved source. Its two original teaching demonstrations are:

1. A character-level count-based Bigram with optional add-one smoothing.
2. A key/value mapping swap test, explicitly **not** Transformer Attention or experimentally validated evidence.

### Run locally

Requires Node.js >=20 and Python 3.10+ (no npm packages required):

```sh
npm run check
npm run serve
```

Open `http://localhost:8080`. The first development branch has 8 unit tests. Browser, assistive-technology and deployed-site acceptance are still open.

## Next milestones

- [ ] **P0:** inspect and import the real v0.21.3 scientific/educational source using a documented repository/release-file allowlist; verify all original hashes before modifying anything.
- [ ] Run `validate.py`, `selftest.py`, `build_atlas.py`, and chapter regression tests using the correct Python environment; report limits honestly.
- [ ] Bind the v0.22 user interface to the actual 44-station journey and 389-topic registry rather than creating competing content.
- [ ] Integrate an interactive scientific lab with clear provenance categories: illustration / computed demonstration / validated experiment / frozen evidence.
- [ ] Review accessibility, mobile support, bilingual scientific equivalence, public deployment and reproducibility.

## Licensing

Tongjiang Lab was reviewed as an interaction reference; **none of its course code or assets** were copied. Licensing for the repository as a whole must be established before public distribution of imported materials; preserve third-party notices.

This repository's `main` remains untouched by the v0.22 draft PR until review.
