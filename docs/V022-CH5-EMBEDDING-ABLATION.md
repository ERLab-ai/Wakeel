# Wakeel v0.22 — Chapter 5 learned embeddings and frozen ablation

**Classification:** SHA-pinned, read-only reproduction of *published numeric results* from the approved v0.21.3 reference; accompanied by a **separate single-scalar teaching illustration**, not an MLP training run. This work does not create new research evidence.

## Immutable scientific provenance
- Exact source bytes: `baseline/v0.21.3/content/chapter-05/results.json`, verified against `baseline/v0.21.3/PUBLIC-MANIFEST.json`.
- Source simulator must be `simulate_chapter5-1.0`; its `input_hash` must equal the manifest SHA for `content/chapter-03/examples.json`.
- The public projection contains **only** validated numeric architecture/parameter counts, six-point training loss curve, train/validation/test aggregate losses for learned vs frozen embeddings, seen/unseen validation counts/losses, source path, and SHA-256.
- No embedding matrices, corpus, predictions, generated samples, raw scientific JSON, or archived run logs are added to the **six-asset** public distribution. The existing authored chapter prose is independently rendered from SHA-pinned approved text.

## Verified architecture and ablation

The reference MLP consumes two previous character IDs, selects two embedding vectors of dimension eight, concatenates them (16), applies a 32-unit tanh hidden layer, and predicts over the vocabulary without the start-only token. Pinned seed/settings remain in the frozen source, **not** recreated by this browser view.

| Language | Total parameters | Embedding | Hidden | Output | Trainable with embeddings frozen |
|---|---:|---:|---:|---:|---:|
| Arabic | 2,110 | 312 | 544 | 1,254 | 1,798 |
| English | 2,028 | 296 | 544 | 1,188 | 1,732 |

Parameter identities are checked from the architecture before the aggregate is accepted:

```
Embedding = V_context × dimension
Hidden = (context_k × dimension + 1) × hidden_width
Output = (hidden_width + 1) × V_output
Total = Embedding + Hidden + Output
Trainable when embedding frozen = Total − Embedding
```

### Frozen reference metrics

| Language | Model | Validation NLL | Test NLL |
|---|---|---:|---:|
| Arabic | MLP, learned embeddings | 2.5579 | 2.7634 |
| Arabic | MLP, frozen random embeddings | 3.0632 | 3.1371 |
| Arabic | Count Trigram, separately pinned Chapter 4 source | **2.2915** | **2.2239** |
| English | MLP, learned embeddings | 2.1609 | 2.1759 |
| English | MLP, frozen random embeddings | 2.8913 | 2.9648 |
| English | Count Trigram, separately pinned Chapter 4 source | **1.9625** | **1.9494** |

Both original-source SHA-256 provenance paths appear in the learner UI whenever the Trigram comparator is present; it is **not silently reattributed to Chapter 5**.

### Validation seen vs unseen contexts

| Language | Contexts seen in training | Unseen in training |
|---|---:|---:|
| Arabic | 102 positions, NLL 2.2335 | 31 positions, NLL 3.6251 |
| English | 148 positions, NLL 2.0859 | 15 positions, NLL 2.9005 |

The count-weighted mean of the two strata is checked against Chapter 5's reported validation loss with a tolerance of 0.0002, accounting for source rounding. The training-loss curve is six checkpoints (1, 10, 50, 100, 200, 300 epochs) with the final point equal to the reported final train loss.

**Limits:** This reference experiment used a very small corpus. The Count Trigram performed better on this dataset; that is not an architectural law. The trainable-vs-frozen comparison describes this particular configuration and is **not evidence that learned dimensions acquired interpretable linguistic semantics**. The reused test set is not an independent generalization study, and the seen/unseen results do not by themselves establish the mechanism behind any difference.

## Standalone scalar gradient illustration

The learner changes a scalar's initial value `e`, target `t`, learning rate `η` and number of steps:

```
loss(e) = 0.5 × (e − t)^2
e_next = e − η × (e − t)       # trainable
e_next = e                    # frozen
```

Inputs are bounded: `e,t ∈ [-2,2]`, `η ∈ {0,0.25,0.5,1}`, integer steps `0..20`. The UI exposes rate choices 0/0.25/0.5/1 and step choices 0/1/2/5/10. Default `e=0,t=1,η=0.5,steps=1`: learned value 0.5, single-example loss 0.125; frozen value 0, loss 0.5. The frozen path is unaffected even after multiple steps.

This scalar quadratic is **not** the actual Chapter 5 MLP loss, vector embedding training or the source's scientific experiment. Comparing these scalar losses to the NLL validation/test table is invalid. No learner progress or scoring is stored.

## Acceptance and unresolved work
- Unit tests validate exact frozen values, parameter identities, seen/unseen weighted reconciliation, SHA mismatch rejection, incomplete fields and invalid gradient-demo inputs.
- Real Chromium tests examine the Chapter 5 evidence, scalar changes, invalid values, both languages and isolation to Chapter 5.
- The unchanged Linux/Windows, baseline integrity and published-asset checks remain mandatory. Original `baseline/v0.21.3/` byte content is never modified.
- **Still open:** independent scientific and bilingual human acceptance; NVDA/VoiceOver; 200%–400% zoom/RTL equation legibility; production HTTPS; a faithful and independently tested full MLP training replay.
