# قرار تصنيف ونشر مصدر وكيل 2.0 — v0.21.3

**Status:** APPROVED AS PUBLIC-SOURCE CANDIDATE; NOT YET COMMITTED. This is *not* approval to deploy source files on the public website or an assertion that content licensing has been resolved.

## Scope

The full approved v0.21.3 ZIP remains frozen with SHA-256 `a97899ca74ec4a269b9a677c9618b0901de07dcbb9fc65b97059b2bd9eac6659` (149 entries). We do not alter or replace it.

The public Git source candidate contains exactly **75 original-byte files**:
- **44** educational/scientific data and content,
- **27** source, schema, tests, and tools,
- **4** targeted machine-readable reproducibility inputs required by existing chapter-8/9 read-only tests:
  - `_ch9_p0/golden_ch9_p0.json`
  - `content/chapter-08/runs.log`
  - `content/chapter-09/runs.log`
  - `content/chapter-09/archive/v1.0/results.json`

**74 original files excluded**, including historic reviews/archived atlases, change logs, generated build outputs, and the original 149-file manifest (which must not be misrepresented as a 75-file manifest). A separate `PUBLIC-MANIFEST.json` records exact SHA-256 for the 75 public source files and the canonical full source ZIP digest.

The **post-release `signoff-v0.21.3/`** remains outside the public repository. Preserve it in the owner's controlled evidence archive. A public summary may cite its approved state and hashes, but must not reproduce internal review records by default.

## Independently verified: 75-file subset

From a fresh copy of the curated 75 files using Python 3.13.5 and NumPy 2.3.5:
- `validate.py`: PASS, 44 stations, 9 chapters, 389 topics, 125 relations.
- `build_atlas.py`: PASS.
- `test_chapter8.py`: 178/178 PASS (read-only).
- `test_chapter9.py`: 107/107 PASS (read-only).
- These are not new training runs. Full chapter 6/7 replay is still restricted to the reference environment; the full selftest was not independently completed in this curated subset run.

## Build candidate and check exact bytes

Obtain the owner-provided approved ZIP and `IMPORT-REPORT.json` in a trusted local workstation. Both are required and SHA-256 pinned.

```sh
python tools/prepare_public_v0213.py --archive /path/to/WAKEEL-2-V0.21.3-SOURCE-PACK.zip --classification /path/to/IMPORT-REPORT.json --output /tmp/WAKEEL2-V0213-CURATED-PUBLIC-SOURCE-CANDIDATE.zip
```

Extract the output **only** into a staging branch of `ERLab-ai/Wakeel` and run:
```sh
python tools/verify_public_v0213.py baseline/v0.21.3
cd baseline/v0.21.3
python tools/validate.py
python tools/build_atlas.py
python tools/test_chapter8.py
python tools/test_chapter9.py
```

The original source archive and private review evidence never need to be added to Git.

## Publication controls

- The repository is public: before committing the 75 source files, verify copyright and third-party distribution rights and confirm no personal/confidential data. A simple automated pattern scan found no obvious email, private-key PEM or credential strings, but **does not constitute full approval**.
- This is a **source allowlist**, not a hosted web allowlist. Public hosting uses a distinct, stricter build allowlist. Do not expose Python tools, raw scientific run logs, experiment archives, review notes or manifest internals as accessible website assets.
- No rewrite of the 75 original files is permitted; the public manifest is an additional record.
- Do not retroactively edit the approved scientific ZIP or move sign-off evidence into a public `signoff-v0.21.3/` folder without separate authorization.
- Until the actual source is committed to the correct branch and GitHub CI runs on it, PR #3 and Issue #2 remain OPEN.

**Workstreams:** v0.21.3 = frozen educational science. v0.22 = new user interface and interactive learning runtime. Avoid cross-import of unrelated WKL-024BC/v1.58 content.
