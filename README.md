# وكيل 2.0 | Wakeel 2.0

**من سؤال إنساني إلى نموذج لغوي قابل للقياس — تعلّم بالتجربة لا بالحفظ.**

## Scientific source and release state

**P0 is COMPLETE.** The curated scientific source of Wakeel 2.0 **v0.21.3** has been merged into `main` via [PR #3](https://github.com/ERLab-ai/Wakeel/pull/3), with the final verified source at `542e256`, and [Issue #2 closed](https://github.com/ERLab-ai/Wakeel/issues/2). The canonical frozen 149-file scientific ZIP remains out of GitHub history:

- Canonical source SHA-256: `a97899ca74ec4a269b9a677c9618b0901de07dcbb9fc65b97059b2bd9eac6659`.
- Curated in Git: **75 exact-byte source files + PUBLIC-MANIFEST.json**.
- Authored learner stations: **44**, organized in **9 chapters**.
- Knowledge registry: **389** topics and **125** recorded relationships.
- Published source is strictly separated from historical reviews, private signoff material and the deployable website.
- Chapter 8 and 9 scientific regressions and selftest passed on GitHub; chapter 6/7 experimental reruns remain bound to their reference environment.

## v0.22 experimental user experience (this draft PR)

The **v0.22** shell reads the real, approved knowledge/learning data from the repository, and generates a small browser-ready distribution.

### Local use

Requires Node.js 20+ and Python 3.10+, with no npm dependencies:

```sh
npm run check
npm run serve
```

Open [http://localhost:8080](http://localhost:8080). The site includes:

- **Journey:** all 44 authored stations, their bilingual hypotheses, actions, outcome alternatives, interpretations and discoveries. Chapter lists and direct `#station/44` links.
- **Atlas:** search all 389 topics by Arabic/English title or ID, view the 125 recorded links, open `#topic/NLP-003` directly without journey prerequisites. **Registry topics without authored station activities are explicitly labeled as non-lessons.**
- **Illustrative labs:** a count-based character Bigram and a tiny key/value counterfactual. These are **not** live replays of the frozen chapter-8/9 experiments.

The build outputs exactly **six allowlisted public files** under `dist/`: HTML, CSS, three app modules, and a generated learning catalog. This intentionally excludes scientific experiment logs, analysis results, internal review evidence, archived documents and Python tooling.

### Limits and next gates

This is a **reviewable bilingual interactive prototype**, not a deployed production learning platform:
- [x] Source imports and scientific verification in P0.
- [x] Journey/atlas derived from approved JSON; complete station/registry count and parity tests.
- [x] Windows checkout safeguard: `baseline/v0.21.3/** -text`; no frozen Git blob modifications.
- [ ] Review the browser flow and keyboard/screen-reader operation, RTL/LTR on mobile, and page-level scientific consistency.
- [ ] Render and review full chapter Markdown prose within the learning pages.
- [ ] Connect actual scientific engines and frozen output evidence to interactive learning labs without a false claim of retraining.
- [ ] Review website deployment allowlist and publish on HTTPS.

This branch is a separate draft [PR #4](https://github.com/ERLab-ai/Wakeel/pull/4). The earlier [PR #1](https://github.com/ERLab-ai/Wakeel/pull/1) remains a separate exploratory spike.

## Licensing and provenance

The content imported from Wakeel v0.21.3 is an approved **project-source subset**, with immutable source checksums. No Tongjiang Lab source code, diagrams or lesson assets were copied. Publishing this repository's source does not imply a license to republish it under unrelated terms.

See [v0.22 integration scope](docs/V022-JOURNEY-ATLAS-INTEGRATION.md) and [P0 publication decision](docs/PUBLICATION-DECISION-V0213.md).
