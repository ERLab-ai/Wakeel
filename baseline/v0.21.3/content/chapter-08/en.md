# Chapter 8 — From a single-query reader to Causal Self-Attention

> **Chapter 7 ended with two measured abilities:** attention-based access that worked through a content shortcut, and an order-discrimination failure that was a configuration failure, not proven structural incapacity. This chapter asks the question Chapter 7 deferred: **Does each position need a state-derived query, and when does that become necessary or useful?** Per the approved contract: no assumed superiority of the more complex architecture, no result before the twelve gates pass, every number from its simulator — and the question stays open until the experiment measures it; it is not settled in its wording.

**This chapter is self-contained.** All of its numbers come from `results.json` produced by `simulate_chapter8-1.2`, and from `factorial_isolation.json` produced by the v0.16 factorial addendum in its v1.2 form (§5-e). The split discipline is explicit: **train** (512 loops) → **validation** (128), used only to freeze the learning rate → **development holdout** (128), reported but never used for selection → a **frozen final test** (64 in-distribution + 64 with a reserved filler length). **Read reliability:** it was read in the v0.16 round, then re-read under the v1.1 re-run authorized by the independent source audit (identically frozen protocol, no selection or tuning, with the v0.16 outcomes archived in `archive/v0.16` for comparison). **As of v0.16:** every simulator run is recorded in an append-only run log (`runs.log`) capturing the code fingerprint, settings, dataset segments, results hash, and completion time — it proves the sequence of runs and their chains, but not by itself the absence of earlier access; turning that into a fully audit-grade proof requires external archiving outside this platform's scope. **Each specific run is identified by its `run_id` together with its `source_sha256`, not by the version label alone** (reusing a version number with different source bytes can happen during maintenance, and the log records both).

---

## 0. Decisions declared before the experiment

| Decision | Value |
|---|---|
| Task | **Dual retrieval**: two keys A/B with two independent binary values — not assumed complementary; two questions with their answers inside the same loop |
| Records | Atomic symbols `M_A0/M_A1/M_B0/M_B1` — declared limitation: this tests **record selection**, not binding a key and a value written as separate symbols |
| Vocabulary | 32 tokens: 4 memory records + 2 queries + 2 answers + 24 fillers from a separate class-signal-free set |
| Loop structure | Two records (balanced order swaps) + filler + [query, answer, query, answer]; next-token prediction at the two query positions only |
| Filler lengths | Train/validation/dev: {2, 4, 8} mixed; **length 16 reserved** outside every development length |
| Positional encoding | Deterministic sinusoidal, not learned (to preempt "untrained positions" readings); learned absolute embedding is a declared separate comparison |
| Optimization | Adam with length-bucket accumulation (one update per epoch), 1200 epochs, one learning rate frozen for all models |
| Gradient aggregation (post-audit fix) | **Weighted by each bucket's supervised-position count**: L = Σ n_i·L_i / Σ n_i and gradient Σ n_i·∇L_i / Σ n_i — the independent source audit proved an unweighted-aggregation defect in v0.16; it was fixed and everything re-run (mandatory gate 5b) |
| Rate selection | Logged sweep on **validation only** (two models × three rates, sweep seed 42) with a declared rule: each rate is represented by its worst model; highest wins; ties → lower rate |
| Splits | 512 / 128 / 128 / 64 / 64 — at the level of the loop **family** (filler path) and all its transformations; no duplicate loops, no shared complete prefixes across segments; exact balance of the 16 structural classes in every block |
| Seeds | Model init 42/43/44; data generation 8102, independent — data shared across all models |
| Outputs | Full-vocabulary softmax plus a binary-normalized read on {0,1} — **two metrics, never substituted for each other**; NLL at answer positions only; teacher forcing and free rollout reported separately |
| Stance | No assumed superiority; results as measured — success or failure |

---

## 1. Station 1 — One question is not always enough

Chapter 7's reader owns **one learned query**: a single vector projected over the whole message to extract one summary, consulted when needed. For one question per message it worked — curiously, via a content shortcut. But real language carries multiple pieces of information per message, and every prediction position asks a different question: a name here, a tense there, a reference to something ten positions back.

So this chapter's hypothesis is narrow and declared: **in a dual-retrieval task — two different questions, at two different positions, about the same memory — is the fixed-query reader enough, or is a position-dependent query necessary for success or efficiency?** The goal is not to prove that self-attention is "smarter"; the contract forbids assuming that, and holds two outcomes equally admissible: the fixed reader succeeds with a small summary encoding both values, or both architectures fail to learn under the declared budget.

**The task itself is shortcut-proofed:** the two values are independent by construction (uniform over 00/01/10/11), so the second answer cannot be inferred from the first; question order and record order are swapped and balanced, so no answer can be inferred from position; fillers come from a class-signal-free symbol set, so no content shortcut as in Chapter 7; and the split is at the level of the complete loop family including all its transformations (question-order swap, record-order swap, value flip), with gate 6 checking 896 loops and gate 7 confirming zero family overlap across the five segments.

---

## 2. Station 2 — A query for every position: the definition

At position `t`, with `h_t = E[x_t] + PE_t` (deterministic sinusoidal):

```
Q_t = h_t W_Q ,  K_j = h_j W_K ,  V_j = h_j W_V
s_tj = Q_t·K_j / √8   for j ≤ t ;   −∞   for j > t
c_t = Σ_{j≤t} softmax(s_t,·)_j · V_j
logits_t = W_O·(h_t + c_t) + b_O
```

Three essential differences from Chapter 7's reader. **First:** the query is not one learned vector; `Q_t` is derived from the position's own representation, so every position "asks its own question". **Second:** reading aggregates over **all** previous positions `j ≤ t` with attention weights, not over one summary. **Third:** the residual output path is declared before any result: the head reads `h_t + c_t` together — the current query token and the summary of the past. This is the simplest possible statement: one head, one layer, width 8, no feed-forward, no multiple heads, **and no claim that this is a full Transformer**.

The comparison map against the single-query reader: the fixed reader is rebuilt here (`fixed_query_prefix_reader`) with the same sinusoidal representation and one learned static query, reapplied at any position where a prediction is needed, with an output head that sees (prefix summary, current query token) — so the only differences from the core mechanism are the query source and the output wiring. Alongside them: `local_only`, which sees only the current question token without records; `oracle_memory_lookup`, a reference ceiling that knows the question's key and reads its record through **the same evaluation path**; and two declared isolations — `no_position` (positional encoding removed) and `no_query_dependency` (query frozen to one shared vector; identical to the core architecture except the query wire). A declared separate comparison adds `csa_learned_absolute_pe` with a learned absolute positional embedding, carrying its declared caveat that reserved-length positions were never seen in training.

---

## 3. Station 3 — Why the causal mask? Gates before results

The causal mask forbids position `t` from seeing anything after it: in training (each position's prediction built from its past only), in evaluation, and in free generation. Its implementation here is **exact masking before the softmax** (−∞ added to blocked scores, so their weights are `exp(−∞) = 0` at the machine level, not merely small), with the loss on the two answer positions only (two per loop; 1024 supervised positions in the whole training set), and the question position never receives the answer symbol it should predict as input — gate 4 verifies this structurally on every loop.

A mask without verification is a claim; therefore the mechanical gates ran before any model number was read, and any gate failure aborts the simulator with no results:

| Gate | Measured | Value |
|---|---|---|
| Exact causal mask | max future weight / max allowed-row-sum error | 0.0 at machine level / 3.3e-16 |
| Past fixed under future change | max logit difference (sequences identical up to `t`, differing after — flipping future answers or filler; before and after training) | 0.0 |
| Prefix = full causal pass | max difference between prefix-only prediction and full-loop prediction under the mask | 8.9e-15 |
| Target alignment | supervised positions checked: end at `Q_A/Q_B`, targets in {0,1} only | 1024 positions |
| Central differences (every element, every parameter group, all six model classes, two distinct masks) | worst error before training / after partial training | 4.6e-11 / 5.8e-08 |
| Aggregated gradient across unequal buckets (gate 5b — post-audit addition) | central differences of the weighted aggregate Σ n_i·L_i / Σ n_i versus the gradient handed to the optimizer | 2.3e-07 |
| Counterfactual pairs (data) | loops checked: flipping A's value changes only A's answer; swapping questions moves positions, not records | 896 loops |
| Segment exclusivity | family overlap across the five segments | 0.0 |
| Positive oracle | `oracle_memory_lookup` accuracy on val/dev/test — and its label-flipped control | 1.000 / 0.000 |
| No-memory control | oracle accuracy on record-stripped loops (task undetermined) | 0.500 |
| Seed scope | every metric per seed plus mean and dispersion | recorded in every table below |
| Validation-only selection | frozen budget from a logged sweep; holdout/test never read for selection | lr = 0.01 |
| Prose regression | every number in both texts derived from `results.json` (automated check) | `tools/test_chapter8.py` |

A note on the finite-difference gate is worth pausing on: in an earlier draft of this simulator's derivatives the gate caught two real bugs — swapped operand order in the value-gradient of the attention aggregation, and summation over the wrong broadcast axis — before any model was trained on anything. **And the independent source audit proved a third defect that in-bucket finite differences cannot logically reveal:** the length-bucket gradient accumulation omitted the bucket-size weighting; gate 5b (central differences of the aggregated weighted loss across 6- and 10-position buckets) was added, the aggregation fixed, and everything re-run. **A fourth evidence audit then caught a reporting-path aggregation bug: the published attention masses summed to 0.5 instead of 1.0 per query type — a healthy softmax, healthy training, and healthy classification, with wrong interpretive numbers because the metric aggregation was wrong.** The layered lessons: **correct per-batch derivatives do not prove the training algorithm that aggregates them; correct training does not prove the metric aggregation used for interpretation — measurement and presentation correctness are tested separately from model correctness.** **And a fifth, from cross-environment replay: reproducibility does not mean byte-identical result files across environments — document the environment (Python/NumPy), separate its differences from scientific-output differences, and test scientific equivalence independently of run metadata.**

---

## 4. Station 4 — The dual-retrieval experiment

### The selection sweep (validation only — gate 11)

| Model | lr = 0.01 | lr = 0.03 | lr = 0.1 |
|---|---|---|---|
| fixed_query_prefix_reader | 1.000 (NLL 0.0002) | 1.000 (0.0001) | 1.000 (0.0) |
| causal_self_attention_1h | 1.000 (0.0002) | 1.000 (0.0) | 0.750 (0.4545) |

Under the declared rule the frozen rate is **lr = 0.01** with the budget frozen (1200 epochs) for all models and seeds; the development holdout and the final test were never read for any choice.

### Main results — binary accuracy on validation (per seed)

| Model | Params | Seed 42 | Seed 43 | Seed 44 | Mean ± std |
|---|---|---|---|---|---|
| local_only | 544 | 0.500 | 0.500 | 0.500 | 0.500 ± 0.0 |
| **fixed_query_prefix_reader** | **936** | **1.000** | **1.000** | **1.000** | **1.000 ± 0.0** |
| **causal_self_attention_1h** | **736** | **1.000** | 0.875 | 0.750 | **0.875 ± 0.1021** |
| no_position (isolation) | 736 | 1.000 | 1.000 | 1.000 | 1.000 ± 0.0 |
| no_query_dependency (isolation) | 680 | 1.000 | 1.000 | 1.000 | 1.000 ± 0.0 |
| csa_learned_absolute_pe (separate comparison) | 912 | 0.7617 | 1.000 | 0.875 | 0.8789 ± 0.0973 |
| oracle_memory_lookup (ceiling) | 0 | 1.000 through the same path (gate 8; label flip 0.000) | | | |

**And the first reading is the most important methodologically: the more complex architecture did not dominate.** The fixed-query reader solved the task completely on validation and on the development holdout (identical numbers: 1.000 for every seed) — meaning "a small static summary encoding both values" is **indeed possible** on this task, which is exactly the branch the contract required examining if the fixed reader tied or won. As for `causal_self_attention_1h`, it also succeeded — the per-position query **does learn** the task — but with visible seed dispersion: seed 43 erred at Q_A (0.750) and seed 44 at both positions (Q_A 0.875, Q_B 0.625), with final training losses of 0.0 / 0.1735 / 0.4545 across the seeds. **A success without control, and every per-seed number is displayed as measured.**

### Probability quality and free generation

| Model | Full-vocab NLL (validation, seed mean) | Note |
|---|---|---|
| local_only | 0.6933 | its normalized binary NLL is 0.6931 = ln2 — an ideal balanced marginal for independent values |
| fixed_query_prefix_reader | 0.0001 | |
| causal_self_attention_1h | 0.2094 | an average pulled by two successful seeds and one struggling seed |
| no_position | 0.0003 | |
| no_query_dependency | 0.0002 | |
| csa_learned_absolute_pe | 0.2094 | |

The standing measurement caveat: the balanced binary baseline is 0.6931 = ln2 only if probabilities are normalized over {0,1}; the full-vocabulary baseline is 3.4657 = ln32. The two scales are never equated, which is why they are displayed side by side.

Free generation (the **generated** first answer enters the second question's context) matched teacher forcing for the successful models: both positions correct on 1.000 of loops for the fixed reader and both isolations, but only 0.250 for `local_only` (0.5 × 0.5 — the independence of the two values visible in a single number), while `causal_self_attention_1h` produced 1.000 / 0.750 / 0.625 across its seeds. The rate at which the first generated token was an answer token was 1.000 for every learned model.

### The frozen final test — evaluated after freezing, re-evaluated in documented versioned runs

| Model | In-distribution: 42 / 43 / 44 | Mean ± std | Reserved length 16: 42 / 43 / 44 | Mean ± std |
|---|---|---|---|---|
| local_only | 0.500 / 0.500 / 0.500 | 0.500 ± 0.0 | 0.500 / 0.500 / 0.500 | 0.500 ± 0.0 |
| fixed_query_prefix_reader | 1.000 / 1.000 / 1.000 | 1.000 ± 0.0 | 0.9688 / 1.000 / 0.9375 | 0.9688 ± 0.0255 |
| causal_self_attention_1h | 1.000 / 0.875 / 0.750 | 0.875 ± 0.1021 | 1.000 / 0.875 / 0.750 | 0.875 ± 0.1021 |
| no_position | 1.000 / 1.000 / 1.000 | 1.000 ± 0.0 | 1.000 / 1.000 / 1.000 | 1.000 ± 0.0 |
| no_query_dependency | 1.000 / 1.000 / 1.000 | 1.000 ± 0.0 | 0.9375 / 0.9844 / 0.9297 | 0.9505 ± 0.0242 |
| csa_learned_absolute_pe | 0.7656 / 1.000 / 0.875 | 0.8802 ± 0.0958 | 0.750 / 1.000 / 0.750 | 0.8333 ± 0.1179 |

Every number here comes from the segment untouched throughout development. The per-length breakdown inside distribution (seed means on the test): the fixed reader and both isolations score 1.000 at all three trained lengths; `causal_self_attention_1h` and `csa_learned_absolute_pe` sit at 0.875 / 0.875 / 0.875 and 0.875 / 0.875 / 0.8854 — their difficulty is seed-level learning, not length.

---

## 5. Station 5 — Mechanism and generalization tests

### a) Presence of information vs. access to it

The information **exists** in the past by construction (gate 6 verifies every loop determines its answers), and **causal access** to it was established by three independent measurements: the positive oracle reads 1.000 through the same path and collapses to 0.000 when its labels are flipped; `local_only`, denied the records, stays at a measured 0.500, not an assumed one; and stripping the records makes the task undetermined — the oracle itself stops at 0.500 — while the learned models fall onto it (0.4727–0.5117) even as their learned probability quality collapses on out-of-training contexts (NLL: fixed reader 14.1326, `no_query_dependency` 12.888, `no_position` 3.9772, `causal_self_attention_1h` 3.5549, `csa_learned_absolute_pe` 2.2287): **near-chance accuracy with high confidence on an undetermined input — two metrics that cannot stand in for each other.**

### b) Causal intervention — flip a value and watch

The attention map is a **correlate**, not attribution; therefore the value of key A was flipped in every validation loop (against the same construction: identical filler path, question and record orders) and model behavior was measured at the two questions **by their key, not by their position**:

| Model (seed 42) | Follow new answer at Q_A | Prediction stability at Q_B | Mean binary shift at Q_A | at Q_B |
|---|---|---|---|---|
| causal_self_attention_1h | 1.000 | 1.000 | 0.9997 | 0.0003 |
| fixed_query_prefix_reader | 1.000 | 1.000 | 0.9998 | 0.0003 |
| no_position | 1.000 | 1.000 | 1.0 | 0.0 |
| no_query_dependency | 1.000 | 1.000 | 0.9996 | 0.0004 |

Initially-correct models (128 tracked loops) follow the flipped value at Q_A with a full binary shift (~1.0) and keep their Q_B prediction fixed at ~0.0003 — the reading comes causally from the **correct record**, not from a position or a marginal. On the frozen in-distribution test the pattern repeats (follow 1.000 for all four models; lowest Q_A shift 0.9996 for `causal_self_attention_1h`). **The limits of this attribution are declared:** the measurement above uses the seed-42 model, and the follow rate is conditional on the original prediction being correct (per-cell n0 values are recorded in the §5-e addendum); **and since flipping A's value changes both the A record and its earlier answer token when Q_A precedes Q_B, a Q_B shift by itself cannot localize which of the two inputs caused it — separating the record channel from the answer channel is a declared follow-up experiment.** And swapping question order did not drop the four models: 1.000 before the swap and 1.000 after — question identity tracks the symbol, not the position.

### c) The attention map: a correlate that misleads alone

Attention mass on the correct record versus the other record, at both question positions (validation, seed 42):

| Model | Q_A: correct / other | Q_B: correct / other |
|---|---|---|
| causal_self_attention_1h | 0.5 / 0.4492 | 0.752 / 0.224 |
| no_position | 0.6808 / 0.2554 | 0.6169 / 0.32 |
| no_query_dependency | 0.3928 / 0.3544 | 0.3336 / 0.3494 |
| fixed_query_prefix_reader | 0.3416 / 0.3494 | 0.3603 / 0.3489 |

Notice the gap between the two kinds of evidence: at Q_A the attention masses on the correct and the other record are close for the fixed reader (0.3416 versus 0.3494 — the other record literally higher) and for the query isolation (0.3928 versus 0.3544) — a weights-only reading would conclude the models do not clearly discriminate; yet the causal intervention flips their Q_A answer with a full shift. **Weights are a distributed reading distribution, not proof of source; attribution requires flipping the value and deleting the record — and both were done.** (These map numbers were corrected in v0.17.1: the earlier normalization divided by the combined count of both query types, making each type's categories sum to 0.5; each type now sums to 1.0, checked automatically.)

### d) Isolations: neither necessary nor dominant

- **Removing the explicit positional encoding** (`no_position`) did not weaken the model: 1.000 on validation, development holdout, and the in-distribution test, **and full generalization to the reserved length 16 (1.000 for every seed)** — because the task is content-addressed (an atomic record carries its key and value). **The limited conclusion is the only correct one:** the explicit positional encoding was not necessary for atomic-record retrieval in this task; this does not prove that all order information is unimportant, nor that the model became fully independent of sequence order — the causal mask itself, the appearance order of query and answer tokens, and the distinct prefixes all still carry structural order information.
- **Freezing the query** (`no_query_dependency`) — the isolation of "query depends on the current state" inside the very architecture of the core model — did not prevent solving: 1.000 in-distribution and 0.9505 ± 0.0242 on length 16. A static query plus output heads that see the current question token (through the residual) suffice for this task.
- **The learned positional embedding** (`csa_learned_absolute_pe`): 0.8789 ± 0.0973 in-distribution and 0.8333 ± 0.1179 on the reserved length — with one fully successful seed. The contract's caveat stands, declared: the longer-length positions were never seen in training, so reading this number as pure "positional generalization failure" is not possible here; that is why the deterministic sinusoidal encoding was chosen for the primary test.

**The isolation summary without cosmetics:** on this task, under one declared budget, neither mechanism — per-position queries, nor query dependence on the current state — was necessary, and the more complex architecture showed no superiority; it showed seed-level learning fragility that the simpler reader lacked. This is a result about **this task and this budget**, and it does not generalize to full Transformers or to human language.

### e) The factorial addendum (v0.16): is the gap due to the query source alone?

A methodological note adopted from the independent review: the main comparison above changes two factors at once — the query source (fixed/state) and the output wire (concat vs residual) — so its gap cannot be attributed to one factor. The 2×2 factorial grid was therefore completed with a fourth cell, `state_query_concat` (state-dependent query + concat output), **with no new selection whatsoever**: the budget is inherited frozen from this chapter's protocol (Adam, 1200 epochs, lr = 0.01, seeds 42/43/44, the same train/validation data), evaluated on **a brand-new fully independent confirmatory set** (generation seed 9204; 64 in-distribution loops + 64 at length 16; families entirely disjoint from every v0.15 segment) — the published final test was not reused for anything. The mechanism gates were re-run on the new cell (`simulate_chapter8_factorial-1.2`): future weight 0.0 at machine level, past fixed 0.0, central differences over all elements of 4.3e-11 fresh and 1.6e-09 after partial training, and the aggregated-gradient gate 5b at 6e-11.

| Cell (seed mean) | Params | Validation | Confirmatory in-dist. | Confirmatory: length 16 |
|---|---|---|---|---|
| fixed_query_residual | 680 | 1.000 ± 0.0 | 1.000 ± 0.0 | 0.9505 ± 0.0301 |
| state_query_residual | 736 | 0.875 ± 0.1021 | 0.875 ± 0.1021 | 0.875 ± 0.1021 |
| fixed_query_concat | 936 | 1.000 ± 0.0 | 1.000 ± 0.0 | 0.9714 ± 0.0258 |
| state_query_concat | 992 | 0.9583 ± 0.0589 | 0.9583 ± 0.0589 | 0.9583 ± 0.0589 |

**The factorial reading:** both fixed-query cells sit at ceiling regardless of the output wire, while the state-query cell partially recovered when the concat wire was added — so the learning dispersion belongs to the **combination** (state query × residual), not to the query source alone. **An interaction observed under these tested settings;** with an explicit declaration: the parameter counts are unequal (680/736/936/992), so the factorial read is directional, not a pure parameter isolation.

**And the counterfactual intervention — now measured for all cells and all three seeds on the confirmatory set — exposes the core of the difference** (cell: follow new answer at Q_A / prediction stability at Q_B):

| Cell | Seed 42 | Seed 43 | Seed 44 |
|---|---|---|---|
| fixed_query_residual | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| state_query_residual | 1.000 / 1.000 | 0.6667 / 1.000 | 0.8571 / 0.250 |
| fixed_query_concat | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| state_query_concat | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 0.500 |

In the two fixed cells: one clean mechanism in every seed (full follow, full stability, shift at Q_A ≈ 1.0 and at Q_B between 0.0001 and 0.0006). In the state-query residual cell — this chapter's main model — the struggling seeds are **not weaker copies of the same mechanism but measured, entangled behavior**: in seed 44 the probability shift moves to Q_B (0.4999) and stability drops to 0.250 — flipping A's value shakes the prediction at the wrong position in **48 of the 64 loops (75%; stability 0.250 means the prediction stayed put in only 16)**. This is what multi-seed intervention measurement settles where a single measurement cannot: **the fixed reader's advantage in this chapter is not "query-source superiority" but the cross-seed stability of its reading mechanism.**

---

## 6. Station 6 — Cost and what remains unsolved

**Structural cost:** a fixed query for one prediction costs O(T) attention scores at one position, while dense self-attention over all positions costs O(T²) to compute its scores. The actual time comparison depends on the number of queries needed, implementation, caching, and hardware; therefore the measured training times for every model and seed are recorded in `results.json` and not converted into any quality judgment. Declared parameters per model: 544 for the local reader, 936 for the fixed reader, 736 for self-attention and the position isolation, 680 for the query isolation, 912 for the learned-position comparison, 0 for the oracle — the comparison is not size-matched, and matching is a separate matter never claimed here.

**What remains unsolved (declared):** binding a key and value written as two separate symbols (atomicity deliberately excludes this route); multiple heads, layers, feed-forward blocks, and a full Transformer; the struggling seeds of `causal_self_attention_1h` were measured by the cross-seed intervention (§5-e: a measured entanglement — shift at the wrong position), while the full structural account of that dispersion remains a later matter; and independent language review plus user experience — separate release gates, not simulation results. Every result is bounded: one synthetic shortcut-proofed task, width 8, one head, three seeds displayed in full, one frozen budget.

---

## 7. Conclusion — and journey status

- **The per-position query learns and works:** `causal_self_attention_1h` solved the dual-retrieval task with derivative-verified causal masking (seed 42: 1.000 on validation, on the frozen in-distribution test, and at reserved length 16) — the mechanism this chapter introduces is real and learnable.
- **But its superiority was never assumed — and did not appear:** the fixed-query reader solved the task completely on every seed and every in-distribution segment, with better generalization to reserved length 16 (0.9688). The chapter becomes an example of its own discipline: the fixed reader's tie was adopted as a result, not as a simulation failure, and the branch the contract demanded was examined — yes, a small static summary could encode both values here. And the v0.16 factorial addendum (§5-e) completed the picture: the self-attention dispersion belongs to its **combination** (state query × residual wire), not to the query source alone, and the cross-seed intervention distinguished a clean mechanism in the fixed cells from entangled behavior in some self-attention seeds.
- **The isolations settled roles, not necessities:** without the explicit positional encoding and without query dependence, the task was fully solved — and reserved-length generalization improved when the explicit positional encoding was removed (1.000), within the declared limit above: this does not leave the model without any order information. The causal effect of reading the correct record was established by intervention (follow 1.000, stability 1.000), not by the weights, which nearly misled at Q_A.
- **And the bounds of all of the above:** one synthetic shortcut-proofed task, atomic symbols, one frozen budget, three fully-displayed seeds — and the final test was evaluated after freezing, then re-evaluated in explicitly versioned, no-selection replications documented in the run log (v0.16, then v1.1).

**The takeaway the learner keeps:** the existence of a more general mechanism does not make using it necessary for every task, nor guarantee easier training or better generalization.

**Journey status:** this chapter is developed **independently**; it is not added to the approved 36-station / 7-chapter journey until an official extension is approved that preserves existing station identities and selects new non-colliding identifiers. Nothing in the journey, registry, or atlas files was modified in this round (a historical state at the close of this chapter's reviews; its merge was later approved in the formal v0.21 expansion — 44 stations/9 chapters — with no change to its results).

**The bridge:** the mechanism that opens "every position asks its own question" is what multi-information routes and composed reasoning will need — but each next step gets a measurement stage of its own: separate key/value binding first, then multi-head and depth, each with a gate contract before its results. The approved wording of the lesson: **an architecture's expressiveness is not its learnability, and learning under one budget is not evidence of superiority beyond its task.**
