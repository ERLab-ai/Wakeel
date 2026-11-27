# Wakeel v0.22 — Chapter 7 attention, retrieval and order discrimination

## Scope

The existing frozen Chapter 7 source, `baseline/v0.21.3/content/chapter-07/results.json`, is projected into the six-asset public learning catalog only after raw-byte SHA-256 verification against the original manifest. The input hash, simulator `simulate_chapter7-2.1`, split sizes, seed set, epoch budget and declared dimensions are checked. The projector validates seed means and the recorded positive controls, label-swap control and within-pair invariance. No training run is performed.

### A. Retrieval — held-out examples

100 training, 30 validation and 40 holdout examples per class, per seed and delay. The class token does not occur in the filler. These are separate trained models, not a single model evaluated at unseen lengths.

| Filler delay | RNN binary accuracy | Content reader accuracy | RNN NLL | Content reader NLL |
|---:|---:|---:|---:|---:|
| 2 | 1 | 1 | .0017 | .0035 |
| 4 | 1 | 1 | .0027 | .0067 |
| 5 | .5 | 1 | .8470 | .0100 |
| 6 | .5 | 1 | .7012 | .0159 |
| 7 | .5 | 1 | .7270 | .0243 |
| 8 | .5 | 1 | .7010 | .0456 |
| 16 | .5 | .792 | .7025 | .7113 |

The content reader can exploit the unique class token. Its higher accuracy at delay 16 does not imply better probabilistic quality: its NLL exceeds the balanced binary reference ln(2). That reference assigns all probability mass to the two possible answer tokens, not uniformly across the full output vocabulary. The known-position oracle is a distinct ceiling, not a learned attention model.

### B. Order discrimination — held-out matched pairs

100/30/40 train/validation/holdout pairs; both members share identical filler and differ in the order of the class symbols. For delays 4, 8, 16, the recorded RNN, learned positional reader, content-only reader and mean reader all score .5 on standard and shifted pairs. The mean and no-position readers have zero within-pair logit difference. Positive controls pos0 and pos_mid_shift score 1/.5 and .5/1 respectively; the latter was trained on shifted examples. The known-position controls do not learn where to search. NLL shown in the order table is the standard holdout NLL.

### C. Initialization sensitivity — exploratory validation, not holdout

| Setting | delay 4 | delay 8 | delay 16 |
|---|---:|---:|---:|
| Original scale | .5 | .5 | .5 |
| All groups ×3 | 1 | 1 | .911 |
| Keys/query only ×3 | .5 | .517 | .506 |
| Values only ×3 | .5 | .506 | .5 |
| Position only ×3 | .5 | .5 | .5 |

This is a deliberately scoped display of five settings; the source prose retains the extended component isolations and learning-rate sweep. No holdout score for the scaled model is inferred from validation. The original positional model's failure is configuration-specific, not structural impossibility. Raising learning rate alone did not reproduce the all-groups result in the recorded sweep. The cause is not isolated.

## Independent calculation

The two toy inputs are [1,-1,0,0] and [-1,1,0,0], optionally shifted to [0,0,1,-1] and [0,0,-1,1]. For each position, score = content scale × value + supplied-position bias (only at the selected index). The calculator normalizes scores via stable softmax and displays the weighted sum, all four weights per pair and known-position oracle value. Scale and bias are in {0,1,2,4}, selected index 0..3. This is an analytic single-query illustration, with scalar values, not a trained model, classifier, or full Transformer. No class probabilities or recall accuracy are computed. With zero position bias, permutation leaves the weighted sum unchanged. Providing a correct position changes the result but does not establish learned retrieval.

## Verification and limits

Nine Node tests cover exact recorded metrics, split identity, source integrity, seed means, controls, validation separation, and the bounded arithmetic. A Chromium scenario checks both languages, tables, controls and chapter isolation. The existing public asset allowlist and frozen research source are unchanged. No raw datasets, training curves or experimental archives are added to the catalog. Scientific/translation review by independent humans, production HTTPS, screen-reader acceptance and reproducible browser training remain open.
