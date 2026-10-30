# Chapter 7 — How does a model reach earlier information without carrying all of it in its state?

> **Chapter 6 ended with three documented findings:** the retention ability exists and is measurable; learning difficulty grows with delay under a fixed budget; and the cliff is not a memory limit — a model trained on mixed lengths generalised to a held-out delay.
>
> This chapter's question: **the second ability — finding. Can a learned attention mechanism reach the required information, and how do we measure its gain and cost without confusing them with prior knowledge of the answer's location? Per the v0.12 audit contract: no assumed Attention advantage, and no result before the gradients are verified and the pairs are truly matched.**

**This chapter stands alone.** All texts and numbers are shown inside it, and its simulator is attached (`simulate_chapter7-2.1`).

---

## 0. Decisions declared before the experiment

| Decision | Value |
|---|---|
| Vocabulary and task | The same CH-6 synthetic vocabulary (32 symbols; output 31 without `<s>`), seeds 42/43/44 |
| Optimiser | **Full-batch gradient descent** — 300 epochs, lr 0.3; **the original experiments involve no hyperparameter tuning** — the later sensitivity experiments (§5 supplement) are exploratory on validation, evaluating declared settings |
| **Attention derivative** | `dS = A·(g − Σ A·g)` — **corrected** (v1.0 summed gdot without the attention weights — a P0 bug found by finite-difference checking in the v0.12 audit). A mandatory numerical-gradient gate now runs in the tests: worst measured absolute error **4.1e-11** across all parameter groups |
| Split | Retrieval: train 100/class, validation **30/class**, holdout **40/class**. Order: 100/30/40 **pairs** — the split unit is the pair. One shared exclusion ledger over full prefixes across train→validation→holdout, with sizes enforced by hard gates |
| **Matched pairs** | One filler path emits both counterparts together — [«أ»,«م»,f,«؟»]←«أ» and [«م»,«أ»,**same f**,«؟»]←«م» — and splits hold entire pairs (P0-2 fix: v1.0 sampled independent fillers per class, breaking the counterfactual) |
| Shared data | One dataset per (experiment, delay, seed) with a recorded fingerprint, used by every model |
| Models (named, with parameter counts) | RNN (**1,183**) — **attn_content**, content-only attention (**671**) — **attn_pos** with a learned positional table (**991**) — **mean** bag reader (**535**) — **pos0** positive control reading position 0 by construction (**535**) — **pos_mid_shift**, its counterpart trained on the shifted distribution |
| Evaluation path | One `evaluate` function from CH-6: consumes the full prefix including «؟»; binary + full-vocabulary argmax + test NLL |
| Reference | `OracleAt` knows the position in advance — a separate ceiling exercising the same path (1.000 on both standard and shifted, hard-gated) |
| Complexity | Single query O(n); full Self-Attention O(n²) — not implemented; no speed comparisons without specification |
| Declared stance | No assumed Attention advantage; results are shown as they come out — success or failure |

---

## 1. Three abilities, not one

**Retaining**, **finding**, and **using** — three different abilities. Chapter 6 established the first. This chapter tests the second: a mechanism that reaches the information's position instead of pushing it through every step.

---

## 2. The single-query reader

```
t_i = E[x_i] (+ P[i] learned positional vector)
k_i = W_k t_i ,  v_i = W_v t_i ,  q: learned query
scores_i = q·k_i / √8 ,  a = softmax(scores) ,  c = Σ a_i v_i
logits = W_o c + b
```

Pooling covers every prefix token **including «؟»** — the measurement position matches the training target position (mandatory parity and gradient tests in the test suite).

---

## 3. Experiment A — retrieval (same task family, revised comparison protocol)

The CH-6 task itself (100/class training instead of 120, plus a validation split) — **not a literal rerun**. Seed means (holdout):

| delay | RNN: binary / NLL | attn_content (671): binary / NLL | Oracle |
|---|---|---|---|
| 2 | 1.000 / 0.0017 | 1.000 / 0.0035 | 1.000 |
| 4 | 1.000 / 0.0027 | 1.000 / 0.0067 | 1.000 |
| 5 | 0.500 / 0.847 | **1.000** / 0.0100 | 1.000 |
| 6 | 0.500 / 0.7012 | **1.000** / 0.0159 | 1.000 |
| 7 | 0.500 / 0.727 | **1.000** / 0.0243 | 1.000 |
| 8 | 0.500 / 0.701 | **1.000** / 0.0456 | 1.000 |
| 16 | 0.500 / 0.7025 | **0.792** / 0.7113 | 1.000 |

The RNN is consistent with the CH-6 pattern (1.000 up to 4, failure from 5). **The content reader solves the task fully up to delay 8 and partially at 16** (per seed: 0.800/0.738/0.838).

**The declared reason — the uniqueness shortcut:** the class characters **never appear in the filler**, so attending to the unique character suffices, with no understanding of position. That loophole is exactly what experiment B removes.

**And a measurement note independent of it:** at delay 16 attn_content reached 0.792 accuracy with a loss of 0.7113 - **worse than the balanced-binary baseline ln2 ~= 0.6931**: higher classification accuracy does not mean better probability quality - two measures, not interchangeable.

---

## 4. Experiment B — order discrimination: removing the shortcut with matched pairs

**Corrected design:** each filler path emits **both counterparts together** — [«أ»,«م»,f,«؟»]←«أ» and [«م»,«أ»,**same f**,«؟»]←«م» — and splits hold entire pairs. (P0-2 fix: the previous version sampled independent fillers per class, so the two sides differed in filler as well as order; now a bag reader receives the literally same multiset in both orders.)

**Controls:** standard Oracle 1.000 and shifted Oracle 1.000 (the information is readable from both positions); label swap 0.000; **two learnable positive controls**: `pos0` reads position 0 by construction and reaches **1.000** on the standard split, and `pos_mid_shift` — trained on the shifted distribution of the same training pairs — reaches **1.000** on the shifted split. The pipeline can express the solution whenever position access is structural.

**Results (seed means, holdout standard / shifted):**

| Model (parameters) | delay 4 | delay 8 | delay 16 | NLL |
|---|---|---|---|---|
| RNN (1,183) | 0.500 / 0.500 | 0.500 / 0.500 | 0.500 / 0.500 | 0.702–0.703 |
| attn_pos (991) | 0.500 / 0.500 | 0.500 / 0.500 | 0.500 / 0.500 | 0.707–0.716 |
| attn_nopos (671) | 0.500 / 0.500 | 0.500 / 0.500 | 0.500 / 0.500 | 0.717 |
| mean bag (535) | 0.500 / 0.500 | 0.500 / 0.500 | 0.500 / 0.500 | 0.712–0.716 |
| **pos0 (positive control)** | **1.000** / 0.500 | **1.000** / 0.500 | **1.000** / 0.500 | 0.0017 |
| **pos_mid_shift (positive control)** | 0.500 / **1.000** | 0.500 / **1.000** | 0.500 / **1.000** | 1.40–1.42 (on standard) |

**Permutation invariance is proven at the logits level, not from accuracy:** the maximum within-pair logit difference (before and after training) is **0.0** for both mean and attn_nopos — they literally give the same answer to both orders, and their aggregate accuracy is **exactly 0.500** (a hard gate).

**The adopted negative result — now on verified gradients and truly matched data:** under a 300-epoch full-batch budget, **none of the four models learned order discrimination** — although the information is present, both references read it, and both positive controls prove the pipeline can express the solution. The failure is a documented learning failure of these settings. Compared with v0.12, the result is now stronger, not weaker: it previously rested on an incorrect derivative and unmatched fillers; it now rests on finite-difference-verified gradients (≤4.1e-11) and pairs differing only in the intended variable.

---

## 5. Isolations and costs

| Isolation | Result | What it establishes |
|---|---|---|
| Numerical gradient check (central differences) | ≤ 4.1e-11 | The training derivatives are correct — a precondition for any claim |
| Within-pair logit identity (mean/attn_nopos) | 0.0 | Structural order-blindness proven at the logits level |
| Pair shifted (deterministic transform of the same examples) | everyone 0.5 | No positional generalisation under this budget |
| pos0 / pos_mid_shift | 1.000 / 1.000 | The pipeline is capable — the failure is not a measurement artefact |
| Label swap | 0.000 | The evaluation path really compares against the label |

Costs (delay 4, seed 42; machine-dependent): RNN 13.46 s versus 0.08–0.34 s for the readers — the cost of the two formulations, not a quality verdict; the faster reader failed experiment B just as the slower one did.

**§5 supplement — initialisation sensitivity: when the gradients are correct and the failure persists.** An independent review of v0.13 retrained attn_pos (delay 4, seed 42, same protocol) obtaining 0.500 under the original initialisation and **1.000 with the initialisation scaled ×3** — on validation. We add the full experiment (exploratory, on **validation** — holdout was never used for selection) with **component isolation**:

| Scaling | delay 4 | delay 8 | delay 16 |
|---|---|---|---|
| ×1 (original) | 0.500 | 0.500 | 0.500 |
| **×3 on all groups** | **1.000** | **1.000** | **0.911** |
| keys+query only ×3 | 0.500 | 0.517 | 0.506 |
| values only ×3 | 0.500 | 0.506 | 0.500 |
| positions only ×3 | 0.500 | 0.500 | 0.500 |

Initial gradient norms (seed 42, delay 4): Wk=9.2e-05, wq=9.1e-05, Wv=0.0181, P=0.0212, Wo=0.0256 — roughly a two-order-of-magnitude gap between the attention-routing gradients and the output gradients (matching the review's measurement).

**Three-part reading:** (1) the original scale reproduces the failure; ×3 flips it to success at delays 4 and 8 and most seeds at 16 — **matching the independent review's measurement literally**. (2) **The isolation refutes the single-component hypothesis:** neither keys+query alone, nor values, nor positions reproduces the success — it requires the joint scaling, suggesting a total-signal-scale effect (an effective learning rate) rather than one weak pathway; testing a plain lr increase is the declared, not-yet-run follow-up. (3) The successful ×3 model does not generalise across positions (the review's measurement: 1.000 standard, 0.500 shifted — attributed, not re-measured here).

**The lesson adopted into WAKEEL:** a network's derivatives can be exactly correct (numerically checked ≤4.1e-11) and it can still fail to learn because the initialisation and optimiser do not generate enough signal — **derivative correctness ≠ optimisation-signal sufficiency**, and learnability is a capability measured separately from theoretical capability and from generalisation.

**Section 5 supplement 2 - is x3 merely an effective learning-rate increase? No.** With the original initialisation fixed and the learning rate raised alone (delay 4, on validation): 0.3 -> 0.500, 0.9 -> 0.500, 3.0 -> 0.500; 9.0 degraded its loss (2.21 in two seeds); 27.0 was unstable (NaN). **Raising the lr alone does not reproduce the x3 success** - the amplification changes the starting point, representation magnitudes, attention scores, the softmax distribution, and gradient scales, not just the update steps. The extended isolations deepen this: Embedding alone, output layer alone, E+Wo, E+P, kqv, kqp - **none achieved success comparable to the global x3 scaling** (they ranged 0.500-0.583); success appears only with **all groups x3**.

**The adopted scientific formulation:** none of the tested isolations sufficed for a comparable success, and the causal mechanism has not been conclusively isolated - the likelier hypothesis is an interaction across parameter groups that reshapes the optimisation landscape, not an effective learning-rate increase and not a single weak pathway.

---

## 6. Conclusion — three abilities

- **Retention:** established in Chapter 6, improved by mixing lengths into training.
- **Finding:** under this budget it happened only through a **unique content signature** (the uniqueness shortcut). When the discriminator was **position**, learning failed — and the negative result is now conditioned on verified correctness and on its own configuration: with the original initialisation the four models failed, while the sensitivity experiment showed the very same attn_pos learning the discrimination under x3 scaling on validation - **a configuration-specific failure, not an established structural incapacity**, resting on numerically checked derivatives and literally matched pairs.
- **Using:** a later question, never assumed to follow from finding.

**Declared limits:** one system (dimension 8, one head, one query, a learned-absolute positional table, 300 epochs, 3 seeds). Not tested: larger dimensions, multiple heads, lr schedules, fully mixed-length training on this task, and the full Transformer (O(n²)). Nothing about "attention" in general follows from the failure here — only about this documented configuration.

**The bridge:** when a model learns where to look (not only how to carry), multi-position access and causal attention open — a later stage with the same measuring contract: a controlled task, separate references, and every number from its simulator.
