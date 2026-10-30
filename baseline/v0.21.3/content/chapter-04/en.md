# Chapter 4 — When does context add predictive power? And when does it become a burden?

> **Chapter 3 ended with a documented limit:** after the space, 16 different characters followed as the next unit, and Bigram always gives one distribution. This chapter's question is not "is longer context better?" — it is sharper:
>
> **When do the previous units add predictive power? And when does adding context turn into data scarcity and extra cost?**

**This chapter can be read on its own.** Same messages, same character representation, same data splits.

---

## 0. Decisions declared before the experiment

| Decision | Value |
|---|---|
| Representation | characters; the same training vocabulary from Chapter 3 |
| Naming | `k=0` Unigram, `k=1` Bigram, `k=2` Trigram, `k=3` 4-gram — where k is the number of previous context units |
| Output space | all characters **except `<s>`** — a context-only symbol, never generated or scored as a target |
| Unseen context | additive smoothing yields a **uniform** distribution over the output space |
| α | chosen per model from the **validation** set only |
| Test | measured once per model — **declared**: we have seen this set before, so this is a teaching comparison, not independent generalisation proof |
| Backoff | a **separate** experiment with separate results |

The theoretical trade-off before any number: a dense table at context length k needs V^k × V_out cells. With a 39-character vocabulary: 39, then 1,521, then 59,319, then **2,313,441** theoretical cells — while the training corpus actually observed 37 / 202 / 318 / 373 contexts. The gap between those two numbers is the space of this chapter.

---

## 1. Experiment one: when does the model know the cases differ?

The two messages Bigram failed on:

| Message | Required after the first space |
|---|---|
| «أين القاعة؟» / "Which room is it in?" | «ا» / "r" |
| «لن أستطيع المجيء…» / "I can't make it tomorrow, apologies." | «أ» / "c" |

Computed from the training data — how often each candidate character followed these contexts:

The two languages differ structurally here, and the chapter reports each honestly:

**Arabic:** contexts coincide at one and two characters («ن » twice), and separate only at three — where «ين » was seen only three times (followers «س»×2, «ن»×1) and never with the needed «ا», so the model has no route to «أين القاعة؟», while «لن » saw «أ» twice and answers consistently.

**English:** the contexts already differ at two characters — "h " saw t×1 and r×1; "I " saw c×4, w×2, n×1 — so discrimination came earlier, though three characters still add precision ("ch " isolates r more sharply than "I " does).

**Reading:**

- At one character: both contexts are identical (a space), so the distributions are identical — no discrimination.
- At two characters: "ن " and "I " — in Arabic the two contexts still coincide; in English they differ but the candidates barely overlap with what is needed.
- At three: the contexts separate («ين » vs «لن »; "ch " vs "I c") — but **separation alone does not make the answer right**: after «ين » the model saw only «س» and «ن», never the needed «ا», so its distribution over the candidates will miss.

The lesson: **longer context discriminates the cases, but does not automatically make the answer correct** — correctness needs training data that witnessed the case.

---

## 2. Experiment two: the quantitative comparison k = 0..3

Same representation, same data, α per model from validation, test measured once:

### Arabic (712 training units)

| k | Model | α (from validation) | Train | Validation | Test | Unseen validation contexts |
|---|---|---|---|---|---|---|
| 0 | Unigram | 1.0 | 3.088 | 3.131 | 3.176 | 0% |
| 1 | Bigram | 0.5 | 2.144 | 2.656 | 2.638 | 0.8% |
| 2 | Trigram | 0.1 | 1.176 | **2.292** | **2.224** | 23.3% |
| 3 | 4-gram | 0.1 | 1.116 | 2.466 | 2.396 | **39.9%** |

### English (729 training units)

| k | Model | α (from validation) | Train | Validation | Test | Unseen validation contexts |
|---|---|---|---|---|---|---|
| 0 | Unigram | 1.0 | 3.017 | 2.978 | 3.026 | 0% |
| 1 | Bigram | 0.1 | 1.882 | 2.223 | 2.223 | 1.2% |
| 2 | Trigram | 0.1 | 1.197 | **1.963** | **1.949** | 9.2% |
| 3 | 4-gram | 0.1 | 1.074 | 2.122 | 2.023 | 27.0% |

**Reading — without a predetermined conclusion:**

1. **Training loss drops monotonically with every context increase** (Arabic: 3.09 → 2.14 → 1.18 → 1.12). Naturally: a finer model memorises better.
2. **Validation and test improve up to k=2, then worsen at k=3.** The computed cause: unseen-context share jumps from 0.8% to 23.3% to 39.9% — longer contexts mean fewer examples per context.
3. This is not a general law about context orders; it is the behaviour of this data at this size. With more data k=3 might improve — we did not test that.

**The permitted verdict:** in this experiment, Trigram was the best balance between added information and data scarcity, and 4-gram paid the sparsity price despite the best training loss.

### Cost: theoretical and measured

| k | Dense table cells (theoretical: V_ctx^k × V_out) | Unique contexts observed | Context–char pairs observed |
|---|---|---|---|
| 0 | 38 (EN) / 36 (AR) | 1 / 1 | 37 / 35 |
| 1 | 1,482 / 1,332 | 30 / 27 | 202 / 194 |
| 2 | 57,798 / 49,284 | 91 / 91 | 318 / 325 |
| 3 | 2,254,122 / 1,823,508 | 108 / 118 | 373 / 392 |

The dense table grows exponentially in theory; the actual implementation (dictionaries) is far smaller because it stores only what was observed — but the dense number shows why no table-based system can climb very far in context.

### Backoff — a separate experiment

Automatic fallback: when a trigram context is unseen or rare, fall back to bigram, then unigram, with weight 0.4. **A different method, shown separately:**

| | Validation | Test |
|---|---|---|
| Arabic (k=3 with Backoff) | 2.733 | 2.351 |
| English (k=3 with Backoff) | 1.820 | 1.680 |

Methodological warning: Stupid Backoff scores are **unnormalised** — they do not sum to one — so their average negative log is not a Cross-Entropy comparable with the proper models, and **we withdraw any "beat Trigram" verdict** built on those numbers. A fair comparison needs normalised scores or an interpolated model — deferred to a later round. The numbers above are directional only.

---

## 3. Generation across context levels

With a fixed seed (42) per level:

| k | Arabic | English |
|---|---|---|
| 0 | «لجمك يلل جبوعمجأر معلاينلأج ،فليابحأ حأغجمكاممتاف» | "petn wor fetnreIi smoewtp?g 'nowdeg" |
| 1 | «لالغء<unk>كن ةةىصماجت» | 'rour?<unk>rree' |
| 2 | «لديط» | "rell yrsCfcxmteao,tlrbyvsIf'Cnpyack I cannogsscWnCI witogiF?…" |
| 3 | «لدهف» | 'reuo' |

Notice: in Arabic, generation stops sooner as context grows (21←4←4 characters); in English the pattern is not a law — k=2 ran for 85 characters before hitting a period (truncated with…). `<unk>` means the model sampled a character never observed in training. Neither model is "better"; generation is a behavioural sample to read, not a quality metric — both expose the personality of their data.

---

## 4. Solutions we will explore later — a knowledge primer, not implementations

This chapter implemented **count-based N-grams only**. The following were neither taught nor run here; they appear in later chapters as options with their own costs:

| Architecture | What it adds | Its declared cost |
|---|---|---|
| A small neural network (we tried one in Chapter 3) | estimates parameters by optimisation instead of counting | did not win here, and its settings need tuning |
| An MLP over context | represents contexts as vectors and learns similarity between them | training and tuning cost; no explicit interpretable table |
| An RNN | a state that updates with every unit — theoretically unbounded context | difficulty retaining genuinely far context in practice |
| Causal attention | attends over the full context and balances units | computational and memory cost that grows — not a "free escape" from context cost |

**Forbidden reading:** "attention solves the context problem". Correct reading: each architecture offers a different way of using previous information, each with costs — we will build and measure.

---

## 5. The closing discovery

When we look at one character we do not see enough context. When we look at three characters the context may become more specific — **but the training examples that resemble it have become fewer**.

Could we build a model that benefits from longer context without needing a separate table for every possible combination?

That question opens Chapter 5: learned representations — instead of counting tables — then recurrent memory and attention, presenting none of them as the one inevitable solution.

---

## Check your understanding

**Standalone:**

1. Why is the k=3 training loss the lowest of all, while its validation loss is worse than k=2?
2. Context «لن » saw «أ» only twice. Does that make predicting «أ» trustworthy? Why do we say "consistent with the data" rather than "correct"?
3. What does a 4-gram do with a context it has never seen, under our declared policy?

**From the journey:**

4. In the discrimination experiment: why did two characters not solve the case while three did? And what remains missing even after separation?
5. Backoff improved English and only reduced damage in Arabic. What may we conclude — and what may we not?
6. Why did we qualify every comparison with "same representation, same evaluation data"? Name one specific risk of violating it.

**Short answers:**

1. Its training memorises better (finer contexts are rarer), but 39.9% of validation contexts were never seen — smoothing spreads them uniformly and validation loss rises. Scarcity, not a broken implementation.
2. Consistent with two training observations — far from statistical trust. The difference between "consistent" and "trustworthy" is the core of model measurement.
3. It produces a uniform distribution over the output space (1/V_out per character) — a declared decision preventing infinite loss, claiming no knowledge.
4. At two characters both contexts coincide ("ن " / "I "); at three they separate. Still missing: training data that witnessed the case — separation is necessary, not sufficient.
5. We may conclude: falling back to shorter context under scarcity reduced damage here. We may not conclude it as a general law — one experiment, two languages.
6. Because a different tokenizer changes the probabilities themselves, so any "win" would measure the tokenizer, not the model — we hit a close cousin of this in v0.5.1 and fixed it.

---

## What this chapter does not claim

- **A very small sample:** ~712 Arabic and ~729 English training units. The improvement/degradation curve belongs to this data.
- **No independent generalisation proof:** we have seen the test set in prior decisions; the comparison is declared educational, and independent proof needs a fresh held-out set.
- **No neural architectures implemented here:** MLP/RNN/attention mentioned as primers only.
- **Backoff uses Stupid Backoff, unnormalised:** its numbers are directional.
- **It does not prove human learning:** all metrics are over units.

## The bridge to Chapter 5

The chapter ends with an open question at its centre: we want a model that benefits from longer context **without a table per combination**. The boundary we learned to draw: between "storing what we saw" and "a representation that generalises".

Chapter 5 enters exactly through that boundary: learned representations — starting with the simplest: unit vectors learned during training instead of a fixed vocabulary, and their effect on the model itself before anything gets bigger.
