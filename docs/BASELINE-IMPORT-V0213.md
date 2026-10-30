# Wakeel 2.0 — v0.21.3 exact-source import: P0

**State (2026-10-30):** The importer and its unit tests are in GitHub; the scientific source package is **not yet committed**. This is deliberately a separate PR from the v0.22 UI prototype.

## Canonical owner-approved reference

- Archive: WAKEEL-2-V0.21.3-SOURCE-PACK.zip
- SHA-256: a97899ca74ec4a269b9a677c9618b0901de07dcbb9fc65b97059b2bd9eac6659
- Archive members: 149, including MANIFEST.sha256.json
- Verified manifest entries: 148/148
- Uncompressed bytes: 4,594,746
- Structural validator: 44 journey stations / 9 chapters / 389 registry topics / 125 relations.
- Independent clean extraction: validator PASS, atlas build PASS, chapter 8 178/178 PASS, chapter 9 107/107 PASS.
- Chapter 6/7 replay acceptance is constrained to the documented reference environment. No claim of new cross-environment certification.

The distinct WKL-024BC/v1.58.0 archive is **not** the accepted Wakeel 2.0 scientific baseline; do not import it here.

## Safe local workflow

1. Obtain the approved v0.21.3 source pack from the project delivery.
2. Run the following commands **on a trusted workstation**, with this branch checked out and an initially empty destination:

    python tools/import_v0213.py --archive /path/to/WAKEEL-2-V0.21.3-SOURCE-PACK.zip --destination baseline/v0.21.3 --report import-verification.json

    cd baseline/v0.21.3
    python -m pip install -r requirements.txt
    python tools/validate.py
    python tools/selftest.py
    python tools/build_atlas.py
    python tools/test_chapter8.py
    python tools/test_chapter9.py

3. Independently inspect the complete 149-path manifest and the expected public-source publication rights before any Git push. **This repository is PUBLIC.**
4. In a separate integration PR, approve which files may be committed to a public source repository and, separately, which may be deployed as website files.

The importer refuses a different archive SHA, changed file SHA, unexpected file count, symlinks, unsafe paths and an already-filled destination. It writes no Git commit and transfers no files to GitHub.

## Public deployment is different from source import

Never serve the complete science directory as a web document root. The candidate contains historical reviews and scientific provenance (top-level archive/, review-v*/, _ch9_p0/, content chapter archives and logs) that require explicit consideration. A limited regex scan of the ZIP found no obvious private keys or email addresses, but is **not** a privacy, licensing or provenance certification.

Only allowlisted frontend bundles and reviewed learner content may appear in the hosted site. Never publish test logs, review notes, research snapshots and internal audit reports merely because they are part of a verifiable scientific source archive.

## P0 closure conditions

- Exact source committed with a reviewed source-publication allowlist and verified matching per-file hashes.
- Clean-checkout CI passing the scientific validator, selftest, atlas build and chapter 8/9 tests.
- Frozen approved evidence unchanged in Git.
- Public-deployment allowlist independently reviewed and enforced.
- A future PR integrates the real station/registry data into the user experience.

Until then, **P0 remains open**. CI success for importer-only commits does not prove the scientific source has been imported.
