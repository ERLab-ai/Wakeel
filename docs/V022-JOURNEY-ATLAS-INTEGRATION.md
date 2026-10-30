# Wakeel 2.0 — v0.22 journey and knowledge atlas integration

**Status:** Independent draft user-experience PR, built on merged scientific source v0.21.3.

## Authoritative inputs

The original source files remain unchanged at:
- `baseline/v0.21.3/data/WAKEEL-2-JOURNEYS-v0.3.json`
- `baseline/v0.21.3/data/WAKEEL-2-KNOWLEDGE-REGISTRY-v0.3.json`
- `baseline/v0.21.3/PUBLIC-MANIFEST.json`

`scripts/build-learning.mjs` produces `dist/data/learning.json`, with only learner-facing fields and SHA-provenance pointers. It copies only `index.html`, `styles.css`, `src/app.mjs`, `src/models.mjs`, and `src/learning-view.mjs` into the distribution. Scientific results JSON, raw scientific logs, archived P0 datasets, tests, and reviewer notes never enter `dist/`.

## Content boundaries

- The **44 stations** are authored learning activities. Each displays a why, a hypothesis, an attempt, outcomes, an interpretation, and discovery in the selected language; a deliverable is displayed **only if present** in both languages. For historical optional stations 31–36 the approved original intentionally omits both deliverables.
- The other **389 topics** are knowledge registry entries; they must not be described as completed explanatory lessons if they lack authoring. They remain independently accessible, without a prerequisite route.
- The **125 relations** are links between concepts, not numerical evidence of mastery.
- The **two toy learning widgets** do not exercise original frozen scientific experiment engines and are labeled as illustrative teaching demonstrations.

## Acceptance

### Automated
- 9 chapters / 44 uniquely ordered stations.
- 389 unique topic IDs / 125 valid relations.
- Arabic/English chapter title and goal present, topic titles paired.
- All six bilingual activity fields paired, with possible-outcome array lengths matched.
- Deliverable field **existence parity**, not mandatory presence.
- One station in precisely one chapter; station topics resolve to knowledge registry.
- No raw scientific results or review paths in built public distribution.
- Output is restricted to six explicitly named public assets.

### Review still required
- Browser navigation: `#station/1`, `#station/44`, `#topic/NLP-003`, direct open and back/forward.
- Verify search filtering, local keyboard use, mobile wrapping, live announcements and AR/EN switching.
- Full authored Markdown chapter rendering and human bilingual/scientific review.
- Separate real experiment display with explicit source labels, not the toy analogies.

## Reproduction

```sh
npm run check
npm run serve
```

Open http://localhost:8080. No npm install is required. The `dist/` directory is generated and should not be committed back into Git.

**Scientific source invariance:** no content under `baseline/v0.21.3/` is changed by the v0.22 generation. Windows `.gitattributes` disables line ending normalization for frozen source bytes.
