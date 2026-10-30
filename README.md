# وكيل 2.0 | Wakeel 2.0

**تعليم بناء النماذج اللغوية بالتجارب القابلة للفحص — بالعربية والإنجليزية.**  
**Learn how language models work through testable experiments — in Arabic and English.**

## Current state (2026-10-30)

The canonical approved educational/scientific baseline is **v0.21.3**, with **44 stations, 9 authored chapters, 389 knowledge topics and 125 links**. The v0.21.3 source archive is immutable:

- Original artifact: `WAKEEL-2-V0.21.3-SOURCE-PACK.zip`
- SHA-256: `a97899ca74ec4a269b9a677c9618b0901de07dcbb9fc65b97059b2bd9eac6659`
- Archive: 149 members (148 files covered by its internal manifest).
- Scientific G7 bilingual-journey parity: previously accepted in its scoped independent review.
- Scientific G8: chapter 6/7 reproducibility accepted only in the documented reference environment. Cross-environment replay is **not** certified.

**Publication:** A **75-file curated source subset** (plus its own immutable manifest) has been verified but is **not yet committed to this public GitHub repository**. Historic reviews, internal experimental archives, and `signoff-v0.21.3/` remain outside public GitHub. This README is repository documentation, **not** a modified copy of the frozen 2026 v0.21.3 release README.

## Source import and current development

- [P0 scientific source import — Issue #2](https://github.com/ERLab-ai/Wakeel/issues/2)
- [P0 safe import and source-policy PR #3](https://github.com/ERLab-ai/Wakeel/pull/3) (draft; the 75 source files are not yet in Git)
- [v0.22 interactive learning prototype PR #1](https://github.com/ERLab-ai/Wakeel/pull/1) (draft; toy examples, not a substitute for the scientific results)
- [Publication decisions](docs/PUBLICATION-DECISION-V0213.md)

The earlier `WKL-024BC / v1.58` workstream is a **different** version and is not the approved Wakeel 2.0 research source.

### Import curated source to this GitHub branch

On a trusted workstation with an authenticated Git checkout of `chore/v0213-baseline-import-p0`, save the already verified `WAKEEL2-V0213-CURATED-PUBLIC-SOURCE-CANDIDATE.zip` **outside the checkout**.

```sh
python tools/publish_curated_v0213.py --archive /path/to/WAKEEL2-V0213-CURATED-PUBLIC-SOURCE-CANDIDATE.zip --push
```

The command checks the exact ZIP and manifest hashes, all 75 included files, the current Git branch/remote and a clean worktree; it then commits only `baseline/v0.21.3/` and pushes that P0 branch. Do not upload the full historic 149-file ZIP or internal reviews directly.

## What's next

1. Pass **mandatory source CI** from a GitHub checkout containing those real 75 immutable files.
2. Connect the nine chapters, 44 stations and 389 topics to the bilingual v0.22 learning experience.
3. Implement honest interactive lab modes: illustration, bounded computation, validated experiment and immutable evidence replay.
4. Define a strictly separate **website** publication allowlist and perform accessibility and scientific review before live release.

**Licensing:** No new blanket repository license is implied by this README. All imported educational materials, external references, and dependencies must retain appropriate original notices; verify their distribution permissions before public release.
