# Chapter 6 — How does a model remember what precedes its window? And why might it fail?

> **Chapter 5 ended with a documented trade-off:** shared parameters (Embedding + MLP) let part of the experience transfer across contexts, but the window stayed fixed — the last two characters only. Everything before them is invisible.
>
> This chapter's question: **is a recurrent state that carries an echo of what precedes the window enough? We build a first RNN, then measure its memory with a controlled task at known delays — assuming no success in advance.**

**This chapter stands alone.** All texts and numbers are shown inside it, and its simulator is attached (`simulate_chapter6-3.2`).

---

## 0. Decisions declared before the experiment

| Decision | Value |
|---|---|
| Vocabulary | The same training vocabulary across CH-3/4/5/6 — **39 Arabic / 37 English characters** (train only; unseen validation/test characters encode as `<unk>`: 1 Arabic, 3 English) |
| Output space | 38 Arabic / 36 English — **no `<s>`** (context only), the CH-5 convention |
| State reset | `h_0 = 0` at the start of **every message** — no leakage between messages |
| Network | `h_t = tanh(W_x e(x_t) + W_h h_{t-1} + b)` then `softmax(W_y h_t)` — E[V×8], W_x[16×8], W_h[16×16], W_y[V_out×16] |
| Parameters | **1,358** Arabic / 1,308 English |
| Real-corpus training | 3 seeds (42/43/44), 100 epochs, lr 0.1, full backpropagation through time per message |
| Synthetic task | class (أ or م) + random filler (27 letters, no أ/م/؟) of length delay + «؟» + **answer = the first character** |
| Loss mask | Loss on the answer position only — the filler is random and carries nothing to learn (see §5: the documented v0.9 failure) |
| Evaluation position | `predict_next_logits` **consumes every prefix token, including «؟»**, and only then the prediction is read — matching the training target position. (Off-by-one fix: the v2.1 probe read the prediction after the last filler and before «؟» — caught by an independent audit) |
| Task split | 120 training sequences/class and 40 test/class — separate draws (seeds 1000+ and 2000+), **and test examples contain no filler path literally present in any training sequence** (70 draws rejected at delay 2) |
| Delays | 2, 4, 5, 6, 7, 8, 16 — **to map the shape of the transition, not assume it** |
| 300 epochs, lr 0.3 | For the synthetic task, 3 seeds, with per-seed results shown |
| Generation | From the seed-42 model explicitly — **model-init and sampling seeds declared separately** (`model_init_seed=42`, `sampling_seed=42`), with a single `<s>` prefix as in training |
| Mixed-length test | One model per seed trained on delays {2,4,5,6,7,8} (60/class/delay), measured inside its domain and at a **held-out delay 16** outside it (§5 supplement) |
| W_h initialisation | Identity ×0.9 + noise — to let gradients cross the delay; a declared choice that enables the experiment, from which no claim about RNNs in general natural-language modelling follows |
| Sanity checks | **Hard — any failure stops the run with a non-zero exit:** explicit Oracle reference through the same evaluation path = 1.000; label-swap control = 0.000; untrained baseline within range; mastery at delays 2 and 4 ≥ 0.9 |

**Central rule:** every number in this chapter comes from `content/chapter-06/results.json` (input hash `08214db6…`). Text generation is a behavioural sample, never used as evidence.

---

## 1. The window's limit

From Chapter 4: «أين القاعة؟» has the trigram context «ين » — observed three times in training, but its recorded followers are only «س» and «ن»; the needed «ا» was never seen after it. A fixed two-character window cannot even tell «أين» apart from «لن».

And the deeper limit: two messages can match exactly on their last two characters while differing in one decisive earlier character. Any windowed model — table or MLP — **structurally cannot** see the difference. Question: does a recurrent state solve this?

---

## 2. Simple memory, no network

Before any learning: if we store the first character in an external variable at message start and read it back at «؟», classification accuracy is **perfect by construction**. This is our explicit reference — an Oracle that reads the first character from the prefix itself — but it exercises **the same evaluation path** as the network (the same accuracy and loss function): its 1.000 at every delay is a hard condition that stops the run if violated. With it comes a control: class-labelled-swapped evaluation drops the reference to 0.000 — proof that the evaluation path really compares against the label.

Early lesson: explicit memory is cheap **because we know in advance what to store**. In general language modelling we do not — and that is exactly what a recurrent state must learn: what to carry, what to drop.

---

## 3. A first RNN

The declared recurrence:

```
h_0 = 0                        (per message)
h_t = tanh(W_x @ e(x_t) + W_h @ h_{t-1} + b_h)
z_t = W_y @ h_t + b_y          (softmax over the output space)
```

The state `h` is a 16-number vector reused at every step — the window is now **the whole message**, in principle. Training uses full backpropagation through time (BPTT) per message.

On the real corpus (mean of 3 seeds):

| | Train (epoch 100) | Validation | Test |
|---|---|---|---|
| Arabic (27 messages) | 3.101 | **3.124** | 3.190 |
| English (23 messages) | 2.994 | **2.929** | 2.996 |

Sparing reading: on validation the RNN edges past Unigram by a hair (3.124 vs 3.131 Arabic; 2.929 vs 2.978 English) and stays below Bigram (2.656 / 2.223). On the Arabic test it is below Unigram (3.190 vs 3.176). **On 27 messages, a 16-unit RNN does not match counting tables at language modelling** — its real value is tested on a controlled task, not here.

And generation from the seed-42 model with sampling seed 42 (two separate declared seeds): linguistically invalid text («لجمف يلل تاوعمتأر معلاينلآج…») — expected, and not used as evidence.

---

## 4. The meeting-data comparison

Same split, same vocabulary, same output space (NLL):

| Model | Arabic: val / test | English: val / test |
|---|---|---|
| Unigram | 3.131 / 3.176 | 2.978 / 3.026 |
| Bigram | 2.656 / 2.638 | 2.223 / 2.223 |
| Trigram | **2.292 / 2.224** | **1.963 / 1.949** |
| MLP (CH-5) | 2.558 / 2.763 | 2.161 / 2.176 |
| RNN (3 seeds) | 3.124 / 3.190 | 2.929 / 2.996 |

Two declared caveats: RNN numbers are a mean of 3 seeds while the rest use one seed; and this is an educational comparison on small data, **not a general architectural ranking**. The useful message: in a small system, a Bigram table beats a 16-unit RNN at pure language modelling — frequent character transitions are stored directly, with no need to learn *how to store*.

---

## 5. The distance test — the core

**Design:** sequence = [class: «أ» or «م»] + random filler × delay + «؟» + answer (the first character itself). Training uses teacher forcing with the loss on **the prediction after «؟» only**. Evaluation goes through `predict_next_logits`: every prefix token — **including «؟»** — is consumed, and only then the prediction is read, so the measurement position matches the training position.

**Why the mask?** The first version (v0.9) trained on all positions, so the single answer target drowned in the noise of random fillers — accuracy pinned at 0.500, i.e. no learning. **Why the position fix?** The v2.1 probe called `forward(prefix up to «؟»)`, which does not consume the last token — so it read the prediction **after the last filler and before «؟»**. An independent audit caught this; re-measuring showed the classifications happened to agree at the previously tested delays, but the distributions differ — the corrected measurement is the one adopted here.

**The four measures** (all through one evaluation path):

- **binary:** forced choice between «أ» and «م» — random baseline ≈ 0.5.
- **argmax:** best class over the full output space (31) — baseline ≈ 1/31.
- **test NLL at the answer position** — separate from the training curve, per seed, with mean and standard deviation.
- **explicit reference:** Oracle through the same path = 1.000 (hard condition).

| delay | binary | per seed (42/43/44) | test NLL (mean ± std) | baseline binary |
|---|---|---|---|---|
| 2 | **1.000** | 1.0 / 1.0 / 1.0 | 0.0016 ± 0.0001 | 0.571 |
| 4 | **1.000** | 1.0 / 1.0 / 1.0 | 0.0030 ± 0.0007 | 0.513 |
| 5 | 0.667 | 0.5 / 0.5 / 1.0 | 0.5234 ± 0.3677 | 0.517 |
| 6 | 0.500 | 0.5 / 0.5 / 0.5 | 0.7763 ± 0.0931 | 0.496 |
| 7 | 0.500 | 0.5 / 0.5 / 0.5 | 0.7094 ± 0.0107 | 0.504 |
| 8 | 0.500 | 0.5 / 0.5 / 0.5 | 0.7103 ± 0.0113 | 0.504 |
| 16 | 0.500 | 0.5 / 0.5 / 0.5 | 0.7024 ± 0.0003 | 0.517 |

Class accuracies at mastery are identical (1.000/1.000) — no constant-answer pattern. At collapse there is a slight tilt toward «أ» (0.667 vs 0.333) — a tie-breaking artefact, not memory.

The **training** curve (answer NLL during training — not test loss; seed mean):

| delay | epoch 1 | 50 | 100 | 200 | 300 |
|---|---|---|---|---|---|
| 2 | 3.428 | 0.558 | 0.008 | 0.003 | 0.0016 |
| 4 | 3.429 | 0.763 | 0.884 | 0.9356 | **0.0031** |
| 5 | 3.430 | 0.789 | 0.722 | 0.813 | 0.526 |
| 8 | 3.434 | 0.784 | 0.720 | 0.707 | 0.7124 |
| 16 | 3.446 | 0.788 | 0.724 | 0.707 | 0.7025 |

**The scope of this comparison first:** everything above trains a **fresh model per delay** — seven independent learning tasks at a fixed budget, not one model whose memory is probed at seven distances. The permitted formulation: **with architecture, optimiser settings, and epoch budget held fixed, the task became harder to learn as the delay grew: all seeds succeeded at 2 and 4, diverged at 5, and none succeeded at 6 or beyond.** It does not follow that "the model's memory ends at the fifth character".

**Reading:**

1. **Full mastery at delays 2 and 4** — accuracy 1.000 in all three seeds, test loss near zero (0.0016/0.0030). The task is learnable with our plain components, measured at the trained position.
2. **The transition at delay 5 is not uniform:** two seeds fail completely (0.5) and one succeeds completely (1.0, with test loss 0.0058) — the loss standard deviation of 0.37 is the largest in the table. The edge is sharp and seed-dependent.
3. **From 6 onward no seed succeeds:** test loss settles at 0.70–0.71 ≈ **ln(2)** — the model predicts the marginal 50/50 on fresh test examples (measured test loss, not inferred from training).
4. **The training path is not a straight line:** at delay 4 the training loss *worsened* between epochs 50 and 200 (0.763→0.9356) then broke through to 0.0031 in the last 100 epochs. Full-batch gradient-descent plateaus (gradients accumulated over the whole set each epoch) — no momentum, no lr schedule.

**Test-set independence:** at delay 2 (filler space 27²=729) 70 test draws were rejected for a filler path literally present in training; the remainder has no literal overlap. At all other delays no draw was rejected. Independent draws alone do not guarantee "unseen sequences" — the overlap was excluded explicitly.

**§5 supplement — one model on a mixture of lengths: is the cliff a memory limit or a learning difficulty?**

After the audit established that the results above are seven independent tasks, we added the proposed experiment: **one model per seed trained on the mixture** of delays {2,4,5,6,7,8} (60 sequences/class/delay, with filler-path overlap excluded across all of them), then measured inside its domain and at **a held-out delay 16 the training never saw**:

| Seed | In domain (2–8) | Out of domain (16) |
|---|---|---|
| 42 | 1.000 | 1.000 |
| 43 | 0.500 | 0.500 |
| 44 | 1.000 | 1.000 |
| **Mean** | **0.833** | **0.833** |

The per-seed pattern is unambiguous: two of three seeds learn a strategy that carries the first character **and generalises to the unseen delay 16** (low test loss inside and outside the domain); the third settles on the marginal at every length. Permitted conclusion: **the measured cliff was a learning constraint at a fixed budget and a fixed length, not an established structural limit of memory.** — and the distinction between "learning at a length" and "generalising to untrained lengths" is the methodological entry to Chapter 7.

**What we do not say:** we do not generalise into a universal "RNN limit". This is one documented system: 16 tanh units, W_h init 0.9I, a 300-epoch budget, 3 seeds. Hidden-size variation, delays beyond 16, and gating mechanisms (LSTM/GRU) were not tested here.

---

## 6. Memory limits — and the bridge beyond the window

**What the experiment established:**

- Recurrent memory **exists and is measurable**: one bit carried across 4 characters with perfect accuracy and near-zero test loss, reproducible across seeds, measured at the correct answer position.
- **Learning difficulty grows with delay under a fixed budget** (seven independent tasks): full success at 2 and 4, a seed-dependent edge at 5, regular failure from 6 onward.
- **The cliff is not a memory limit:** when lengths are mixed in one model's training, two of three seeds generalise to the held-out delay 16 — the constraint is a declared-settings learning constraint, not a proven structural end.
- All four sanity conditions are hard (PASS): the explicit reference through the evaluation path is 1.000, the label-swap control is 0.000, the untrained baseline is within range (0.496–0.571), and there is no constant-answer pattern.

**What was not tested:** delays beyond 16, hidden-size variation, more seeds, gating mechanisms (LSTM/GRU — deferred positions, not implemented). Our limit: results of one system with declared settings.

**The bridge:** the failure to carry information across distance opens the next question: can we reach **directly** into earlier positions instead of pushing information through every step? That is the idea of attention — but no advantage is assumed: its fair comparison must run on **the same task and the same split/evaluation protocol**, clearly separating a reference that knows where the information sits in advance from a mechanism that must learn where to look. Every number will come from its simulator, as here.
