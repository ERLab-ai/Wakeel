# Wakeel v0.22 — Chapter 8 causal self-attention and dual retrieval

## Sources and protocol

The public evidence is a read-only numeric projection of two separately SHA-verified files in the frozen v0.21.3 baseline:
- content/chapter-08/results.json (simulate_chapter8-1.2)
- content/chapter-08/factorial_isolation.json (simulate_chapter8_factorial-1.2), which references the exact SHA of the first file.

The dual-retrieval task uses two independent binary values encoded in atomic memory records. It tests record selection rather than learning key/value binding across separate tokens. The model is one head, one layer, not a complete Transformer. Training/validation/development have 512/128/128 loops, final in-distribution and length-16 tests each have 64 loops with two supervised answer positions per loop. Training lengths are 2,4,8; data seed 8102; model seeds 42,43,44; the frozen budget is 1200 epochs, lr 0.01. The final test was reread after a corrected weighted gradient aggregation; it is not a never-before-read test.

## Original final-test binary accuracy (three-seed mean)

| Model | In distribution | Reserved filler length 16 |
|---|---:|---:|
| fixed_query_prefix_reader | 1 | .9688 |
| causal_self_attention_1h | .875 | .875 |
| no_position | 1 | 1 |
| no_query_dependency | 1 | .9505 |
| csa_learned_absolute_pe | .8802 | .8333 |
| local_only | .5 | .5 |

Validation, development and final test remain selectable separately. Binary accuracy, full-vocabulary accuracy, full-vocabulary NLL and binary-normalized NLL are distinct measurements. The fixed reader and state-query reader differ in both query and output wiring. No architecture-wide superiority follows.

## Separate factorial confirmation

The confirmation data use seed 9204, 64 loops per split, zero recorded family overlap with the original segments, and no new selection. Parameter counts differ, so the 2×2 comparison is directional, not a clean parameter-matched causal identification.

| Cell | Parameters | Confirm in | Confirm length 16 |
|---|---:|---:|---:|
| fixed_query_residual | 680 | 1 | .9505 |
| state_query_residual | 736 | .875 | .875 |
| fixed_query_concat | 936 | 1 | .9714 |
| state_query_concat | 992 | .9583 | .9583 |

## Interactive scalar calculation

At query position 1 (zero-based), Q=1 and scalar K=V=[1,-1,x,y]. Scores are the values themselves (dimension one). Causal masking excludes positions 2 and 3 before normalization; their weights are exactly zero. Stable softmax over the allowed positions gives tanh(1) ≈ 0.761594 regardless of x,y. The unmasked negative control reads future values and can change the output. Both future values can be -2,-1,0,1,2. This illustration has no learned parameters, optimizer, full Transformer, classifier or accuracy estimate. It does not reproduce or re-certify the frozen training run.

## Verification

The projector checks both source digests and linkage, frozen budgets, split counts, causal-mask and prefix equivalence records, family overlap, target alignment, counterfactual data checks, oracle controls, weighted aggregate gradient records, metric bounds and seed means. Nine Node tests include all 25 causal future interventions and unmasked negative controls. A Chromium scenario verifies bilingual rendering, distinct confirmation/main tables, split selection and future intervention. The public six-asset allowlist and baseline stay unchanged. Independent scientific and linguistic review, assistive-technology acceptance, full training replication and production HTTPS remain open.
