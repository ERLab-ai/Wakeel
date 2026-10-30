# v0.22 — real-browser acceptance gate

This is an **automated Chromium E2E and accessibility-smoke gate**, not a claim of human visual approval or NVDA/VoiceOver certification.

The job `v0.22 actual browser acceptance (Chromium)` builds the *real* six-file `dist/` artifact from SHA-checked approved 18 bilingual chapters, then opens the page through the local **allowlisted HTTP server** in a headless Chromium browser.

## Automated coverage
- Initial Arabic `lang=ar dir=rtl`, nine chapters, 44 authored stations, 389 topics, 125 relations and 18 complete chapter source texts.
- Deep link to chapter 9, expanded full scientific prose, HTML tables and code blocks.
- Arabic/English switching must change `lang`/`dir` while preserving expanded prose and showing the correct language's SHA.
- Topic ID search, direct station 44, browser back navigation.
- Skip link keyboard focus, Enter-to-expand native `details`, search input keyboard, landmark/heading visibility through Chromium accessibility tree.
- 390px phone-sized viewport has no full-document horizontal overflow; scientific tables stay inside their own scrollable prose container. Both directions checked.
- Live preview must return HTTP 404 for raw scientific run logs, results, reviews, and backend tools.
- Browser JS exception list must be empty; screenshots captured for human review.

## Evidence and limitations
Screenshots in `wakeel-v022-browser-evidence` (7-day retention) document sample states, not proof of all mobile devices or full visual audit.

**Still OPEN:** human review of actual visual hierarchy/table legibility, RTL mixed Arabic/English math, zoom and reflow at 200%–400%, browser-specific behavior beyond Chromium, formal WCAG 2.2 audit, NVDA/VoiceOver speech/focus sequence, and usability of disabled/unavailable laboratory features. Keep PR #4 Draft until independently reviewed.

All frozen scientific content remains unchanged. No new training, scientific result, or evidence claim is made by this browser test.