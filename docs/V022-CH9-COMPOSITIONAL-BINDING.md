# Wakeel v0.22 — Chapter 9 compositional key–value binding

## Source and experimental scope

Read-only projection of the exact SHA-256-verified baseline/v0.21.3/content/chapter-09/results.json (simulate_chapter9-1.1). This version corrects documentation and read order without changing the recorded scientific numbers. The interface is bilingual and does not retrain the models.

Separate key/value/separator tokens replace the atomic records of Chapter 8. Training uses 512 loops, validation 128, development 128, and each of the confirmation and final in-distribution/length-16 splits 64 loops and 16 matched pairs. Data generation seeds are 9102 (P0), 9204 (confirmation), and 9203 (final). Model seeds are 42,43,44; the same examples are evaluated across seeds. The frozen budget is 1200 epochs and learning rate .003. The declared v1.1 sequence freezes models before reading all confirmation sets, then all final sets. This is source-reported chronology, not an external proof of historical access.

## Recorded final in-distribution means

| Model | Parameters | Q1 binary | Pair-both | Q2 binary (teacher forcing) | Both answers |
|---|---:|---:|---:|---:|---:|
| local_only | 561 | .5000 | 0 | .5000 | .2500 |
| fixed_query_prefix_reader | 961 | .7500 | 0 | 1 | .7500 |
| causal_self_attention_1h | 753 | .8125 | .3333 | .8125 | .6354 |
| causal_self_attention_2h | 945 | .8646 | .5000 | .9896 | .8542 |
| no_position | 753 | .6667 | 0 | .8333 | .5833 |
| no_position_2h | 945 | .9375 | .7500 | .9948 | .9323 |
| csa1h_concat | 1017 | .7500 | .0417 | .9687 | .7187 |

The two-layer no-position model's Q1 and pair-both means are .9375 and .7500 on the confirmation in-distribution set and final length-16 set too. These are limited synthetic results, not a general architecture ranking. The panel allows selecting the four splits and inspecting each seed. Teacher forcing supplies the true first answer; free rollout is shown separately (final in-distribution only). Interventions are the recorded final in-distribution results for seed 42 only; no cross-seed claim is made from them.

The deterministic count-only first-query ceiling is .75; pair-both ceiling is zero; second-query ceiling is 1 given the true first answer. These are ceilings for the count-only class, not all fixed-query architectures. In particular positional encoding can make the fixed-query reader order-sensitive.

## Descriptive factorial reading

| Depth | With positional encoding | Without |
|---|---:|---:|
| One layer | .8125 | .6667 |
| Two layers | .8646 | .9375 |

The recorded descriptive interaction is .2187 (21.87 percentage points). Depth changes parameter count 753→945, so its causal effect cannot be separated from capacity. Within-depth PE removal preserves parameter count. Three seeds share the same evaluation data; no significance test or independent sample inflation is claimed.

## Independent interactive calculation

Enumerate four equally weighted assignments (A,B) in {0,1}². Select the first queried key. The deterministic count reader sees a+b, predicts 1 only if the sum is 2, and otherwise predicts 0. Thus first-query accuracy is 75% and both matched contrasting worlds cannot be solved. The second reader computes a+b minus the displayed first answer. With the true answer it succeeds in all four cases; with the flipped answer it fails all four, and two predictions fall outside the binary domain. This arithmetic counterexample is not a trained model or a replication of the recorded intervention protocol.

## Engineering acceptance

The source projector verifies exact bytes, frozen protocol, recorded quarantine order and split overlap, nine reference segments, parameter counts, matched-pair counts, seed-level aggregate consistency and metric bounds. Numeric projection excludes raw data, attention traces and review metadata. Nine Node tests and a Chromium test cover source integrity, negative cases, separate splits, seeds, counterfactual arithmetic and chapter isolation. The Chapter 1 selector is narrowed to avoid mistaking Chapter 9 evidence for Chapter 1 evidence. The public asset allowlist remains six files; no frozen baseline edits. Independent human scientific/linguistic review, assistive technology, full model-training replication, integration review and production HTTPS remain open.
