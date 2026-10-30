# Chapter Nine — When the Bags Match but the Binding Differs: From an Atomic Record to a Separate Key and Value

**This chapter is self-contained.** Simulation and table values come from `results.json` produced by `simulate_chapter9-1.1`; the logit-gap values used in the counter-tests of the fixed-query reader and the attention layers are quoted from the independent audit in `review-v1/` and were not recomputed in this review. The first execution of the experiment (`simulate_chapter9-1.0`, run fingerprint `c5f901ca7f8218fc`) is archived verbatim under `archive/v1.0` — the v1.1 re-run under the identical frozen protocol reproduced every simulation result without a single change, and corrected four documentation issues raised by the independent scientific and programmatic audit (its evidence sits in `review-v1/`). The split discipline was declared before any result: **train** (512 loops) → **validation** (128), the only set used to freeze the learning rate → a **development holdout** (128), reported but never selected on → a quarantined **confirmatory set** (generation seed 9204), read after models were frozen and before the final → a **reserved final test** (seed 9203; 64 in-distribution loops + 64 at filler length 16, never seen in training). In the first run the quarantined reads were interleaved per (model, seed) — with no selection or update after the freeze — and v1.1 made the order literally global: every confirmatory read completes before any final read, with identical numbers across both versions.

**Reading key:** A *key* (`K_A`/`K_B`) identifies the item being queried; a *value* (`V0`/`V1`) is its associated datum. *Positional encoding* adds information about token location; a *causal mask* prevents attention to future tokens; a *seed* specifies a reproducible model initialization. *First-query accuracy* here is correctness when choosing between `ANS0` and `ANS1`, not itself a probability over the complete 33-token vocabulary. Under *teacher forcing* the true first answer is supplied before predicting the second; under *free rollout* the model sees its own first prediction instead.

## 1. The puzzle: two identical counters, two different answers

In Chapter Eight the record was a single atomic token (`M_A0` or `M_A1`): a model may select the correct token, but that does not test its ability to **bind** a key to its value when they are two separate tokens. Chapter Nine decomposes the record:

| Loop | The two records | Filler | The two queries | First-query answer |
|---|---|---|---|---|
| a | `K_A V1 SEP` then `K_B V0 SEP` | `F3 F7` | `Q_A …` then `Q_B …` | ANS1 |
| b | `K_A V0 SEP` then `K_B V1 SEP` | `F3 F7` | `Q_A …` then `Q_B …` | ANS0 |

The two loops are **identical in the count of every token** at the first query: same keys, same two values, same delimiter, same filler, same query. What differs is the **adjacency relation**: which value follows which key. Any **deterministic** classifier that relies on token counts alone — however elaborate its computations — emits the same answer for both loops, so it must get one side right and the other wrong. This zero ceiling does not literally apply to a classifier that samples a random answer afresh on each prediction. We therefore built, for every (filler family, record order, query order), **counterfactual pairs** of exactly this kind; the decisive metric is getting **both members of the pair right**: for a deterministic order-blind classifier its ceiling is zero.

These pairs exist in all nine segments: 128 in train, 32 each in validation and development, and 16 in each of the four quarantined segments.

## 2. The measured ceiling: what token counts can and cannot do

We computed the empirical optimum for a classifier that sees only the **token counter** (no positions) at the first query, per segment:

| Metric | Value across all nine segments |
|---|---|
| Counter-classifier ceiling at the first query | 0.7500 |
| Ceiling for getting both pair members right using a deterministic counter classifier | 0.0000 |
| Structural oracle (reads `KEY VALUE SEP`) on both queries | 1.0000 |
| Structural oracle on counterfactual pairs | 1.0000 |
| Counter-classifier ceiling at the **second query** | 1.0000 |

Why 0.7500: classes where the two values coincide (`00` and `11` — half the samples) are solvable from counts alone; the contrasting classes (`01` and `10` — the other half) have identical counts across the pair, so no counter classifier exceeds 0.5000 there; the mixture gives 0.7500. The last row is an early warning we use in Section 7: at the second query, counting becomes **sufficient**, because the displayed first answer reveals the other key's value within this task's structure.

**An important documentation fix (from the independent audit):** the 0.7500 ceiling is proven for the counter **class**, and it is not a structural bound on any particular model. The fixed-query reader we meet below settles exactly there **as a measured fact**, but its architecture is not purely order-blind: its query is fixed, yet its keys and values carry positional encoding, so its attention scores depend on positions. The audit's counter-test measured the logit gap between the two members of a count-matched pair: 6.8e-05 for our model — and 3.5e-18 when the positional encoding is temporarily zeroed. So the model **carries** order sensitivity yet did not exploit it for learning within our budget; a measured outcome, not a structural incapacity.

**Check your understanding before examining the models:** If we swap `V0` and `V1` between `K_A` and `K_B` while preserving all token counts, can a deterministic count-only classifier answer the first query correctly in both loops? **Answer:** No; the count-only inputs are identical while the correct labels differ. Section 7 shows why the displayed first answer creates a different shortcut at the second query.

## 3. The models and the budget frozen before any result

Seven learned architectures, each answering a separate question, all trained on the identical path:

| Model | Parameters | The question it answers |
|---|---|---|
| local_only | 561 | Is the current query token alone enough? (measured, failure not assumed) |
| fixed_query_prefix_reader | 961 | Is a fixed-query prefix summary + current token enough? |
| causal_self_attention_1h | 753 | Is one causal-attention layer with sinusoidal positional encoding enough? |
| causal_self_attention_2h | 945 | What does a second causal-attention layer add on top? |
| no_position | 753 | Isolation: the same single layer without positional encoding |
| no_position_2h | 945 | Isolation: the same two layers without positional encoding |
| csa1h_concat | 1017 | Output-wire swap to concat (different parameter count, disclosed) |

The budget was frozen before reading any evaluation segment: Adam, 1200 epochs, one update per epoch, position-weighted gradient aggregation across length buckets. The learning-rate pool {0.003, 0.01, 0.03} was swept on validation alone under a pre-declared rule (each rate's candidate is its worst anchor; highest wins; ties → smallest), the three candidates tied at 0.7500, and **0.003** was selected. Full training runs total **30** (9 sweep + 21 final), of which **27 are unique configurations** — three configurations (the three anchors at seed 42 and the frozen rate) repeat between sweep and final with identical determinism. Every seed (42/43/44) is reported separately, then its mean and spread; nothing hides behind an average.

## 4. The primary evidence: the first query and the counterfactual pairs

On the reserved final test (filler lengths 2/4/8), first-query accuracy and the rate of getting both members of a pair right — mean of three seeds:

| Model | First-query accuracy | Both pair members | Both loop answers (teacher forcing) |
|---|---|---|---|
| local_only | 0.5000 | 0.0000 | 0.2500 |
| fixed_query_prefix_reader | 0.7500 | 0.0000 | 0.7500 |
| causal_self_attention_1h | 0.8125 | 0.3333 | 0.6354 |
| causal_self_attention_2h | 0.8646 | 0.5000 | 0.8542 |
| no_position | 0.6667 | 0.0000 | 0.5833 |
| no_position_2h | 0.9375 | 0.7500 | 0.9323 |
| csa1h_concat | 0.7500 | 0.0417 | 0.7187 |

Before the final, the quarantined confirmatory set (read first) supported **the main finding and broad ordering**, though not the exact ranking of every model: the fixed-query reader scored 0.7448 on confirmation versus 0.7500 for `csa1h_concat`, while they tied on the final set. On the central finding, first-query and pair-both scores were 0.9375 and 0.7500 for the two-layer no-position model, against 0.8594 and 0.5208 for the two-layer positional model. Seeds are reported apart, and the picture is richer than means:

| Model | Seed 42 | Seed 43 | Seed 44 | Pair-both (seed 42/43/44) |
|---|---|---|---|---|
| causal_self_attention_1h | 0.8750 | 0.7500 | 0.8125 | 0.5000 / 0.2500 / 0.2500 |
| causal_self_attention_2h | 0.7500 | 0.8438 | 1.0000 | 0.1250 / 0.3750 / 1.0000 |
| no_position | 0.7500 | 0.7500 | 0.5000 | 0.0000 / 0.0000 / 0.0000 |
| no_position_2h | 0.9375 | 0.8750 | 1.0000 | 0.7500 / 0.5000 / 1.0000 |

Two readings before any generalization. First, the pair metric is more honest than classification accuracy: a model at 0.8125 accuracy may sit at 0.3333 on pairs, because the countable half inflates accuracy while masking the order deficit. Second, the error anatomy says so literally: for the two-layer positional model (seed 42, on validation) classes `00` and `11` are at 1.0000 while `01` and `10` sit at 0.5000 — its errors are consistent with difficulty assigning each value to the corresponding key; the two-layer no-position model (same seed) got `00`, `01` and `11` at 1.0000 and `10` at 0.7500. The failure is not noise; it is a positional deficit exactly where order is the discriminator.

## 5. The factorial reading: depth interacts with positional encoding

Arranged as a 2×2 grid (seed means on the reserved final test):

| Architecture | With sinusoidal positional encoding | Without positional encoding | Effect of removing the encoding |
|---|---|---|---|
| One layer (753 parameters) | 0.8125 | 0.6667 | -0.1458 |
| Two layers (945 parameters) | 0.8646 | 0.9375 | 0.0729 |

| Comparison | Descriptive difference |
|---|---|
| Adding a second layer with positional encoding | 0.0521 |
| Adding a second layer without positional encoding | 0.2708 |
| Descriptive interaction (difference of differences) | 0.2187 |

**The effect of removing positional encoding flips sign with depth.** At one layer, **measured accuracy fell** to 0.6667 when positional encoding was removed; this does not prove absolute necessity. At two layers, **measured accuracy rose** to 0.9375 after removal, showing that encoding was not required to obtain this performance in this setting, without demonstrating general redundancy. This is the core teaching result: **the effect of one architectural choice can depend on another**, and any single slogan of the form "positional encoding matters / does not matter" ruins the lesson.

How can two layers extract order with no explicit encoding? In a single position-free layer, attention weights depend only on token identities, so the output at the query is a function of the prefix **counter**, not its order — a structural incapacity, measured (0.6667 below the ceiling; one seed at 0.5000). With two layers, the first layer's output at each position is a summary of what came **before** it; two different orderings of the values produce different summaries at different positions, so the second layer sees inputs that vary with order and can discriminate it. The attention maps support this reading as **evidence**, not causal attribution: in the two-layer no-position model, the first layer at the query distributes over the correct and the other value almost equally (0.3810 and 0.3810) — order-blind as the theory predicts — while the discrimination emerges in the second layer. The audit's counter-test measured order sensitivity directly: the single-layer no-position model has exactly zero logit gap between pair members, and the two-layer no-position model shows 6.5e-04.

**The limits of this reading are declared:** adding a layer raises parameters from 753 to 945, so the depth comparison is not isolated from capacity, and the correct statement is "added depth without positional encoding enabled order capture on this task" — no broader causal claim. The presence/absence comparison of the encoding within the same depth keeps the parameter count fixed (753 vs 753, and 945 vs 945) and is the cleaner probe for the encoding factor. The results are three seeds on a shared final set (64 loops and 16 pairs per seed) — not independent samples after averaging, no significance tests are used, and the seed spread is visible in the table above (two-layer models range from 0.7500 to 1.0000).

## 6. Negative results are part of the evidence

- **local_only at exactly 0.5000** in every seed and segment: the query token alone carries nothing about the values — measured, not assumed.
- **The fixed-query reader at 0.7500** for every seed of the in-distribution final test: it settles at the counter ceiling **on this split as a measured fact**; on confirmation it averaged 0.7448 (seed 44: 0.7344). See the Section 2 fix: its architecture carries positional sensitivity through its keys and values but did not exploit it here. Note that at the second query it jumps to 1.0000 — via the shortcut channel, not via binding.
- **no_position below the ceiling** (0.6667): a ceiling is an upper bound for a class, and a trained model may fall short of it; seed 43 dropped to 0.5000.
- **csa1h_concat at 0.7500**: doubling the output head's input width (1017 parameters) did not buy order capability — changing the wire is not changing the mechanism.

## 7. The second query: a legitimate shortcut that must be exposed, not counted as binding

At the second query, the first answer **is displayed in the input**: its token, together with its query token, reveals the first key's value by counting, and both value tokens are present, so the other key's value follows from counts alone. Hence the counter ceiling at the second query is **1.0000, measured on every segment** — the channel is structural to this task, not a generation flaw. We keep the evidence separate accordingly:

| Model | Second-query accuracy (teacher forcing) | Both answers |
|---|---|---|
| local_only | 0.5000 | 0.2500 |
| fixed_query_prefix_reader | 1.0000 | 0.7500 |
| causal_self_attention_1h | 0.8125 | 0.6354 |
| causal_self_attention_2h | 0.9896 | 0.8542 |
| no_position | 0.8333 | 0.5833 |
| no_position_2h | 0.9948 | 0.9323 |
| csa1h_concat | 0.9687 | 0.7187 |

These gains are **not** independent proof of binding. The double interventions at the second query (seed 42, on validation) separate memory readers from prior-answer readers — the other key's correct label never changes under any intervention:

| Model | No intervention | Flip record value only | Flip prior answer only | Both (coherent) |
|---|---|---|---|---|
| fixed_query_prefix_reader | 1.0000 | 0.5000 | 0.5000 | 1.0000 |
| causal_self_attention_2h | 0.9062 | 0.5938 | 0.5938 | 0.9062 |
| no_position | 1.0000 | 0.5000 | 0.5000 | 1.0000 |
| no_position_2h | 1.0000 | 0.6875 | 0.6875 | 1.0000 |

The fixed-query reader and the single-layer no-position model follow the prior answer almost completely: any single flip collapses them to 0.5000, and flipping both (which restores coherence) restores them fully. The binding models are less dependent but still affected by the contradiction between record and displayed answer — and these are **input** interventions measuring coherence sensitivity, not proofs of any internal mechanism. Under free rollout, a model that built its second answer on a borrowed first answer falls: the fixed-query reader drops to 0.7500 in free second-query accuracy — exactly its first-query accuracy. And an identity worth stating: the "both answers correct" rate under free rollout equals its teacher-forcing counterpart exactly for every model, because a correct generated first answer makes the two contexts identical, while a wrong one fails both metrics together.

## 8. Length 16: unseen in training, unopened before the freeze

On the quarantined length-16 segments (longest loop: 26 tokens), first-query accuracy, seed means:

| Model | final_ood16 |
|---|---|
| causal_self_attention_1h | 0.8125 |
| causal_self_attention_2h | 0.8333 |
| no_position | 0.6667 |
| no_position_2h | 0.9375 |
| fixed_query_prefix_reader | 0.7500 |

No collapse appears in **first-query accuracy** for the displayed architectures at length 16; the sinusoidal encoding is defined at positions unseen during training, and the two-layer no-position model keeps its full performance here (0.9375). This is a measurement at these lengths only, not a promise of further generalization.

## 9. The discipline that produced these numbers

Gates 1–10 of the chapter contract ran before any result was accepted: the input contract (no atomic fact; `KEY VALUE SEP` verified across all nine segments), the split contract (zero overlap across 72 filler families; sixteen-class balance; verbatim restoration of the five golden P0 digests), the pair builders with a mutation monitor proving the detectors fire, the references (structural oracle 1.0000 through the same evaluation path and label-flipping collapses it to 0.0000; the no-memory control at 0.5000), the causal mask (future weight 0.0 at machine level; suffix invariance; prefix/full equivalence), per-element central differences for every architecture before training and after partial training with worst error 2.3e-09, the weighted-aggregation gate with a negative test exposing unweighted aggregation, the loss contract (no answer token in the first query's context; the second query's target already visible as its predecessor in 0.5000 of loops — a declared channel probed by the interventions), the double intervention (each mode changes exactly its declared token), and measurement integrity (attention normalization independent per group summing to 1.0000 before rounding, with a negative test, and map capture changing no accuracy). Gate 11 is this text and its Arabic twin under automated checking; Gate 12: nothing outside the chapter folder was modified — the 36-station journey stands as is (a historical statement of the P1 execution; the merge was later approved in the v0.21 expansion).

## 10. Limits of the conclusions

Everything above holds at one budget (lr = 0.003, 1200 epochs, weighted full batch), filler lengths ≤16, and three seeds. The two-layer no-position model outperforming its positional twin (identical parameter count: 945) is a fact within this experiment, not a general rule that positional encoding hurts. Depth comparisons also change the parameter count, so depth is not causally isolated from capacity. High second-query performance is not counted as independent binding evidence because of the shortcut channel. Attention maps are evidence, not causal attributions. And the append-only run log documents the sequence and fingerprint chains of runs without by itself proving the absence of earlier access — a declared limit.

## 11. Reproduction and status

```bash
python -B tools/simulate_chapter9.py    # re-runs everything under the frozen protocol
python -B tools/test_chapter9.py        # read-only: classified checks + Mutation gate + this prose gate
python -B tools/test_chapter9.py --replay   # optional: temporary re-run matched by the triple classifier
```

Chapter Nine so far: the P0 data and protocol are approved, the P1 experiments are executed with their documentation corrected and both versions archived, and this text and its Arabic twin are complete. A historically documented state: at the close of P2 the chapter was not merged into the 36-station/7-chapter journey; the formal extension was then approved (v0.21) and the chapter merged into a 44-station/9-chapter journey (stations 41–44) — with no change to this chapter's results or protocol.
