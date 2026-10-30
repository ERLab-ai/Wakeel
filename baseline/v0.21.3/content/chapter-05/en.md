# Chapter 5 — How does the model learn a representation that helps it generalise?

> **Chapter 4 ended with a documented trade-off:** the longer the context, the better the discrimination — and the worse the scarcity. A 4-gram table needs 2,254,122 theoretical cells against 373 observed records, and its validation loss was worse than Trigram.
>
> This chapter's question: **can different contexts share learned parameters, instead of storing an independent table per character combination?** At what cost, and under what conditions? **We do not assume it will beat count-based Trigram.**

**This chapter can be read on its own.** All texts and numbers are shown inside, computed by a bundled tool.

---

## 0. Decisions declared before the experiment

| Decision | Value |
|---|---|
| Representation | characters; the same training vocabulary (39 AR / 37 EN) — **unchanged** |
| Context window | k = 2 (last two characters) |
| Embedding | a [V_context × 8] learnable table — one row per ID |
| Network | concat [16] → hidden [32] tanh → softmax over the output space (38 AR chars, no `<s>`) |
| Declared parameters | **2,110** AR / 2,028 EN — an architectural estimate before training |
| Data | the same Chapter-3 split (712/133/245 AR) — **no new test set**; the comparison is declared educational |
| Baseline | count-based Trigram (k=2) from Chapter 4: validation 2.292 / test 2.224 |
| Seed and settings | seed 42, 300 epochs, lr 0.5, full-batch — pinned in the simulator |

**Central rule:** the vocabulary does not disappear. IDs stay fixed, but each ID has a row in an Embedding table whose values change through optimisation. Vector proximity is not "understanding meaning" — that needs an independent test.

---

## 1. The limit of tables: why knowledge is not shared

From Chapter 4: context «لن » saw «أ» twice, and context «ين » saw «س» twice and «ن» once. A Trigram table treats each context as **an isolated island**: what it learned about «لن » transfers nothing to «ين » — however syntactically similar the cases.

Documented from our experiment: the trigram context of «أين القاعة؟» was seen three times, but never with the needed «ا» — its recorded followers are only «س» and «ن». If contexts had **representations** allowing comparison, part of the experience could transfer.

The practical question: can **shared parameters** across different contexts give a better prediction?

---

## 2. From ID to Embedding

In Chapter 4 each character was a raw number with no relations.

The intermediate solution: each ID gets an 8-dimensional row in an Embedding table:

| Concept | Precise description |
|---|---|
| One-hot | a vector of vocabulary length with a single 1 |
| Embedding | one-hot × [V_ctx × 8] matrix = a row selector |
| Learnable | the table's rows are parameters updated by optimisation like any other |

**Before training:** vectors are small random values with no relations. **After training:** they changed in whatever way reduced loss. **Forbidden naming:** calling dimensions pre-assigned "linguistic features" — that needs an independent test we do not provide here.

---

## 3. Our first MLP for next-character prediction

The approved architecture:

```
input: previous 2 token IDs
E: [V_ctx, 8]          # shared learnable embedding
concat: [16]
linear_1: [16, 32] + bias → tanh
linear_2: [32, V_out] + bias → softmax (no <s>)
```

**Training with teacher forcing:** every position of every training message is predicted from its true context — no future leakage: the target at position i is never used as input at or before i.

**Parameter count (computed from code):** Arabic — Embedding 312 + hidden 544 + output 1,254 = **2,110**. English 2,028.

---

## 4. The training experiment: a curve and a comparison

### Arabic (712 training positions)

| Epoch | Training loss |
|---|---|
| 1 | 3.638 |
| 10 | 3.409 |
| 50 | 3.059 |
| 100 | 2.801 |
| 200 | 2.284 |
| 300 | 1.989 |

### Final results — and the declared comparison

| Model | Validation | Test |
|---|---|---|
| Count-based Trigram (k=2, Chapter 4) | **2.292** | **2.224** |
| MLP (Emb 8 + hidden 32) | 2.558 | 2.763 |

**Trigram won on this small data.** This matches your prediction: we did not assume an MLP win — and it did not happen here. Candidate causes (more parameters, less data, limited tuning) are stated as **untested hypotheses**, not settled explanations.

---

## 5. The isolation experiment: frozen vs learned embeddings

Same architecture, same seed, one difference: in (a) the Embedding table is **frozen random** (never optimised); in (b) it is **learnable**. The trained-parameter counts differ — declared.

| Variant | Train | Validation | Test |
|---|---|---|---|
| Learned embeddings | 1.989 | 2.558 | 2.763 |
| Frozen random embeddings | 3.025 | 3.063 | 3.137 |

The gap (frozen val 3.063 vs learned val 2.558) isolates the value of training the embeddings: 312 parameters moving from random to trained reduced validation loss by 0.5 nats. Both variants remain above Unigram (3.131) and below Trigram (2.292).

**What this chapter proves:** embeddings **change only when learnable** — the frozen variant is unchanged by construction (verifiable by code inspection, no measurement needed). What it does not prove: that the change is "better" — the validation loss decides.

---

## 6. What actually generalises? — the discovery

We measured validation loss **separately** for contexts seen in training and contexts never seen:

| Validation contexts | NLL |
|---|---|
| seen in training | 2.234 |
| **unseen in training** | **3.625** |

**This gap (2.234 vs 3.625) is the chapter's finding:** shared parameters help on contexts similar to what was seen, and fail on the new. Bengio et al. (2003) supports the **research principle**: shared parameters enable prediction on contexts never literally seen. In our sample, the MLP had lower loss than the smoothed Trigram table on unseen contexts, clearly in English and marginally in Arabic, but was worse on seen contexts and overall. This demonstrates possible transfer through shared parameters in this implementation — it does not prove a specific linguistic relation was learned.

And generation with seed 42 produces linguistically invalid Arabic text («ع نن ين الاًبماستأعضلاًع…») — expected from 712 characters, and not used as evidence of quality.

### Cost comparison — no verdict

| | Trigram (k=2) | MLP |
|---|---|---|
| Storage | 318 observed records | 2,110 parameters |
| Inference | table lookup | matrix multiplications |
| Decision transparency | explicit (table) | implicit (weights) |

No winner is imposed: the two models represent the problem differently, each with its own domain of advantage.

---

## 7. A question we deliberately defer — and its limit

**Our model still sees a fixed window of two characters.** Any information before the window is lost entirely — even if decisive.

How could it retain information from before the window? That is Chapter 6's question: recurrent memory — then later attention, as an option with its own cost, not an inevitable solution.

---

## Check your understanding

**Standalone:**

1. What is the difference between One-hot and Embedding in size and content?
2. Why are Embedding dimensions not named "linguistic features"?
3. How many parameters does our model have? Give the breakdown.

**From the journey:**

4. Trigram beat the MLP here. Name two candidate causes — and what testing would be needed to prefer them.
5. In the isolation experiment: why was the frozen variant still able to learn at all?
6. The chapter ends with the fixed-window question. State the bridge to Chapter 6 in one sentence.

**Short answers:**

1. One-hot: vocabulary-length vector with one 1; Embedding: a dense 8-number row, learned — a row selector of the table.
2. Dimensions have no declared names, and interpreting them linguistically is a claim needing independent evidence.
3. 2,110: Embedding 312 + hidden (16×32+32)=544 + output (32×38+38)=1,254.
4. (a) small sample fits a table exactly, (b) limited MLP tuning. Preferring either needs bigger data and fair tuning — unavailable here.
5. The hidden layer can partially compensate a random representation — at slower learning; the frozen-embedding variant isolates the difference.
6. "Our model sees a fixed window; we need a mechanism retaining pre-window information" — the recurrent-state question.

---

## What this chapter does not claim

- **A very small sample:** 712 Arabic training positions. The learning curve belongs to this data.
- **No independent generalisation proof:** the test set was used in prior decisions; no learner-impact study.
- **No linguistic naming of dimensions:** interpreting Embedding dimensions is a claim without evidence here.
- **No interactive engine:** everything is simulation, documented.
- **Basic optimisation:** full-batch, fixed rate, 300 epochs — not carefully tuned; "Trigram won" may change with tuning, untested.

## The bridge to Chapter 6

We learned shared representations instead of a table per combination, but our model still sees a fixed window of two characters. **How does it retain information from before the window?** — Chapter 6's recurrent-memory question, with its limits tested before attention is introduced.
