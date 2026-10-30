# Wakeel 2.0 — v0.22 learning-lab integration contract

**Status:** draft; source-import review required.  
**Approved scientific baseline:** v0.21.3 (not the distinct WKL-024BC/v1.58 stream).

## Immutable source identity

Archive `WAKEEL-2-V0.21.3-SOURCE-PACK.zip`:
- SHA-256 `a97899ca74ec4a269b9a677c9618b0901de07dcbb9fc65b97059b2bd9eac6659`
- 149 zip entries, 148/148 manifest entries verified locally.
- Journey 44 stations / 9 chapters; knowledge registry 389 topics.
- Chapter 8/9 frozen outputs and scientific simulators must remain byte-for-byte intact.
- Chapter 6/7 rerun is reference-environment scoped; cross-environment replay not certified.

The source archive contains historical, review, data and output material: **do not publish the complete source ZIP directly as a hosted web tree.** Assess source distribution license/allowlist separately from the deployable public artifact.

## Why the initial PR is only a spike

The two static demos establish user interaction, deterministic inputs, RTL/LTR behavior and error states. Their outputs do **not** constitute independently reviewed scientific data from v0.21.3. No scientific numbers, simulator source, experiment reruns or evidence are imported/modified.

## Evidence classification

All future lab display outputs must declare exactly one of:
- `illustration`: pedagogical representation; no empirical validity claimed.
- `computation`: a declared equation or bounded deterministic calculation and parameters.
- `validated_experiment`: model output with recorded accepted tests and provenance.
- `frozen_evidence`: read-only output obtained from an approved file with hash.

For every result provide source/version, inputs, units, limitations and type; the UI cannot silently relabel outcomes.

## Non-negotiable invariants

1. Preserve scientific and review evidence; append new experimental runs separately with their own hashes.
2. 44 station IDs and orders and nine chapter identity are fixed until a new approved journey migration.
3. Arabic/English meaning parity holds for all real content and controls; swapping language must not alter scientific result.
4. Independent topic access remains possible; journey prerequisites are pedagogical, not access enforcement.
5. Existing chapter results are presented as archived measurements, not simulated live values.
6. A changed input invalidates the previous active observation until a new result has been calculated.
7. No Tongjiang source/assets are copied; Three.js may be used under its own license where needed.
8. Public deployment uses an explicit allowlist that excludes archived reviews, internal reports, test fixtures and unpublished scientific data by default.

## Phase gates

**P0 — baseline import:** verify ZIP and its manifest, review publication boundaries, import approved source content on separate branch, run unchanged validator/selftest/atlas and scientific regression in documented environment.

**P1 — one real lab:** select an approved Bigram chapter experiment and bind the UI to its actual outcomes; present a declared computational demo separately from frozen evidence.

**P2 — association experiment:** chapter 9 key/value task with authentic controlled-pair artifacts, restrictions on claims, and no fake Transformer training.

**P3 — full learning shell:** render 44 stations, nine bilingual chapters, knowledge atlas and topic navigation. No copying topic labels into a second inconsistent registry.

**P4 — quality/release:** scientific/translation review, interactive accessibility and mobile acceptance, reproducibility ledger, bounded offline distribution, controlled release.

## Sign-off

Draft PR #1 is **not** final v0.22 sign-off. Scientific baseline sign-off remains v0.21.3.
