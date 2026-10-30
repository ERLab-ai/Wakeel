# Wakeel 2.0 — P0 publication decision and final editorial resolutions

**2026-10-30 — Signed-off selection:** 75 immutable science/source files, SHA-pinned to `WAKEEL-2-V0.21.3-SOURCE-PACK.zip`; 73 of 148 inner-manifest records intentionally excluded, plus original `MANIFEST.sha256.json` = 74 of 149 archive members excluded.

**Final inspection evidence:** 75/75 byte-identical, public manifest SHA-256 `919b552c0427740849483e5fe8e47edc638a80bfcb351f4a3beefcb251468cb9`, curated archive SHA-256 `77da30249ce9d4baa51d17e8166ae4efdf66c03604f64352578b2270f2c217e6`.

## Editorial issue closure

1. **Original `README.md` remains excluded.** GitHub repository uses an independently maintained README for product and contributor context. It is not the frozen educational README.
2. **`content/chapter-01/review-notes.md` remains excluded.** It is reviewer-facing, not learner content. Keep it with the controlled 149-file research archive.
3. **`content/chapter-08/test_invocation_replay.transcript.txt` remains excluded.** The read-only 178/178 suite passes without it; the replay mode is separate and can generate a fresh transcript when intentionally invoked. It is not a prerequisite of P0.

The post-release `signoff-v0.21.3/` remains a separately controlled archive, not published to GitHub.

## Local tests completed on selected 75-file subset

- `validate.py`: PASS (44 stations, 9 chapters, 389 topics, 125 links).
- `selftest.py`: PASS — 14 invalid mutations rejected, 2 healthy accepted, no baseline report modification.
- `build_atlas.py`: PASS.
- `test_chapter8.py`: 178/178 PASS, read-only.
- `test_chapter9.py`: 107/107 PASS, read-only.

This verifies a scoped source release, not browser deploy, cross-environment chapter-6/7 reruns or full independent scientific peer review.

## Safely stage actual ZIP to GitHub

The approved curated ZIP is available in this conversation and must be placed **outside** the cloned repository. On a trusted workstation with Git access and a clean checkout of branch `chore/v0213-baseline-import-p0`:

```sh
python tools/publish_curated_v0213.py --repo . --archive /path/to/WAKEEL2-V0213-CURATED-PUBLIC-SOURCE-CANDIDATE.zip
```

The default operation checks archive SHA-256, 76 safe members, embedded 75-file manifest and its pinned SHA-256, Git origin, current branch and clean work tree, then stages **only the 76 files** under `baseline/v0.21.3/`. Inspect with `git diff --cached --stat`. Then commit and push manually, or use `--push` in a first invocation from a **clean** checkout to commit/push in one operation.

**The original scientific source archive, private historical reviews, signoff evidence and local audit reports must not enter the Git staging area.**

## Source is not the deploy artifact

No raw `baseline/v0.21.3/` folder is served via a public web host. Create a distinct deployment allowlist for the actual browser-facing content and deny internal tools, review files, scientific run logs and archived P0 evidence. Do not claim P0 complete until 76 files are actually in GitHub and the mandatory `scientific-baseline` CI job passes there.

**Approved frozen identity:** `a97899ca74ec4a269b9a677c9618b0901de07dcbb9fc65b97059b2bd9eac6659`. Distinct `WKL-024BC/v1.58` is not part of Wakeel 2.0 v0.21.3.
