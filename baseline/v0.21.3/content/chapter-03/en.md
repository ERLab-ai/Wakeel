# Chapter 3 — How does the model learn to predict the next unit?

> **Chapter 2 ended with a question:** we encoded T1 with a 12-unit vocabulary and five of its eight units became `<unk>`. The question that opens this chapter: **what happens if we train a model on text where many words collapse into that one unknown token?**
>
> The answer you will discover by experiment: training cannot recover what the input representation erased. From that discovery we choose a better representation, then build our first generation model, then measure it with a number rather than an impression.

**This chapter can be read on its own.** All texts and numbers are shown inside, and every number is computed from its data by a bundled tool.

**Expected time:** 40–60 minutes.

---

## 0. Representation decisions, declared before any model

| Decision | Value |
|---|---|
| Unit level | **characters** — to escape the `<unk>` problem at word level |
| Start symbol | `<s>`, prepended to every message |
| Unknown policy | any character not seen in training becomes `<unk>`, and its occurrences are counted |
| Normalization | no diacritic stripping, no hamza folding; characters as written |
| Splitting | at message level before tokenization — never a random cut through a token stream |

**Declared split targets:** ~700 training units, 150 validation, 150 test. **Actual counts** (recorded, not forced): English 729 / 163 / 280; Arabic 712 / 133 / 245. The gap from targets is natural because we split whole messages; its effect on the metric is discussed below.

Why characters? This table answers it. Message T1, "Could we move the meeting to another day?":

| Representation | Units | Unknown |
|---|---|---|
| words (the 12-unit vocabulary from Chapter 2) | 8 | **5** |
| characters (the full training vocabulary) | 41 | **0** |

Characters solved coverage and created a length problem: 41 units instead of 8. That is the trade-off the two models will be built on, and Chapter 4 returns to it. (The single unknown English character across all nine test messages is the em-dash in T8 — it never appeared in training, so the unknown policy applied, as designed.)

---

## 1. A question that leads to the metric: which prediction is better?

Before any equation: suppose the correct next character is "a", and two models assign it different probabilities:

| | Probability given to "a" | Loss |
|---|---|---|
| Model A | 70% | ≈ 0.357 |
| Model B | 10% | ≈ 2.303 |

Model A gave a larger probability to what actually happened, so its loss is lower (loss here is −log of the probability). When we aggregate predictions over every position of a message, we need an average that summarises prediction quality:

**L = −(1/N) × Σ [ log P(xₜ | xₜ₋₁) ]**

- N: the number of transitions scored.
- xₜ: the unit that actually occurred; xₜ₋₁: the one before it.
- Natural logarithm; the unit is **nats per token**.

This metric is used in every comparison below, always with the same tokenizer and the same evaluation data.

---

## 2. Station one: a model with no context — the baseline

The weakest comparison we are willing to make is a model that spreads probability over characters without looking at any context (Unigram). We count it with simple smoothing and measure it:

| Model | Validation | Test |
|---|---|---|
| Unigram (English) | 3.009 | 3.056 |
| Unigram (Arabic) | 3.168 | 3.212 |

This is the **baseline**: in our experiment the context model landed below it. That is a measured result, not a law — poor estimation or scarce data can make a context model worse than no context at all.

---

## 3. Station two: Bigram by counting — a worked example

The idea: count every transition (character, following character) in the training text, then estimate probability by division. And so the loss does not explode on a transition never seen in training, we use **additive smoothing**:

P(b | a) = (count of a→b + α) ÷ (count of a + α × V)

where V is the vocabulary size and α is the smoothing strength.

### A small worked example (with its own mini vocabulary)

Mini training text: "hi hi". Its characters with `<s>`: `<s> h i (space) h i` — **5 transitions**, and a mini vocabulary of 4 units. With α = 1:

| Transition | Count | Computed P |
|---|---|---|
| `<s>` → h | 1 | (1+1)÷(1+6) = **0.286** |
| h → i | 2 | (2+1)÷(2+6) = **0.375** |
| i → space | 1 | (1+1)÷(1+6) = **0.286** |
| space → h | 1 | (1+1)÷(1+6) = **0.286** |

That is a complete Bigram in miniature: a table of numbers estimated from data by counting. (This example uses its own mini vocabulary; the real experiment below uses the full character vocabulary.)

### The real experiment

The full counting table is 37×37 cells, most of them zeros. We chose α on the **validation set** from three values, then measured the test set exactly once:

| Chosen on validation | Validation loss | Test loss |
|---|---|---|
| English: α = 0.1 (from 0.1 / 0.5 / 1.0) | 2.223 | **2.223** |
| Arabic: α = 0.5 (same grid) | 2.656 | **2.638** |

Compare with the baseline: English 2.978/3.026, Arabic 3.131/3.176. **Context genuinely helped** — and that is a measured result, not a law: poor estimation or scarce data can make a context model worse than no context.

This is the chapter's first quantitative comparison, fair by construction: same representation, same evaluation data, for both models.

---

## 4. Station three: generation, step by step

Generation here is **sampling** from the model's distribution with a fixed seed (42) — a different seed may legitimately produce different text.

Starting from "I'd" (the first three characters of a training message), the English model's first five actual steps:

| Step | Context | Chosen | Its probability |
|---|---|---|---|
| 1 | d | W | 0.229 |
| 2 | W | h | 0.635 |
| 3 | h | o | 0.118 |
| 4 | o | s | 0.060 |
| 5 | s | (space) | 0.370 |

The full output: "I'dWhos win's'tirome tindvvolf bhithee an". (The Arabic output from «أود» was «أودمالغء?كن ةةىصماجت».)

**This is text, not language — expected and declared:** a model trained on ~700 characters shows learning happening (locally coherent clusters, a stop at punctuation) but cannot produce clean language. The purpose is to see the mechanism — which is why we will not use text beauty as evidence of improvement; we return to the number in the next section.

---

## 5. Station four: is it improving? A numeric loss, not an impression

### The comparison used in this chapter

| Model | What it learns | What we measure |
|---|---|---|
| Unigram | character frequencies, no context | baseline |
| Count-based Bigram | next-character probabilities from transition counts | validation and test loss |
| Neural Bigram | parameters improved by descent on a loss function | training curve, validation and test loss |

### A small neural model

One layer: a one-hot input for the previous character, a softmax output over all vocabulary characters, small random initialisation with a pinned seed, gradient descent (learning rate 0.5, 200 full epochs over 729 transitions). This is the simplest possible network for the same task, and we compare its behaviour — not the word "neural beats counting".

### Results

| Model | English: validation / test | Arabic: validation / test |
|---|---|---|
| Unigram | 2.978 / 3.026 | 3.131 / 3.176 |
| Count Bigram (α from validation) | **2.223 / 2.223** | **2.656 / 2.638** |
| Neural Bigram | 2.789 / 2.909 | 2.927 / 3.062 |

**How to read this table — without overclaiming:**

1. Both models beat the baseline: context added information — a measured result on this data, not a law.
2. **Count-based Bigram won against the neural one in this specific experiment.** The described network (one-hot input, linear layer, softmax) can in principle represent the same conditional distributions the table represents; so the gap is not attributed to "small sample" alone. It can be partly explained by the estimation method and the optimisation settings (epochs, learning rate, initialisation, absence of tuning) — and we have not run the experiments needed to isolate each factor.
3. The neural training curve kept descending to epoch 200 (3.61 → 2.83 on training) but validation stayed worse than counting. We did not tune longer training or a bigger network, and we do not claim this is its ceiling.
4. Every comparison above uses the same tokenizer and the same evaluation data. Comparing two losses computed with different tokenizers is banned.

---

## 6. Station five: the real difference between counting and the network

Counting said "tally the transitions and divide"; the network said "start from random weights, measure the error, improve". The essential point: **both are models whose parameters were estimated from data**:

| | The counting table | The small network |
|---|---|---|
| Parameters | table cells | weights and biases |
| Estimation | direct counting + smoothing | iterative descent on a loss |
| Change during training | nothing changes after counting | weights change every epoch |
| Unseen units | covered by smoothing | covered by the softmax distribution |

The distinction that will separate future chapters is not "a model that learns versus one that does not", but **the method of estimating parameters** and the representational capacity that comes with it.

---

## 7. Station six: when does knowing the previous character fail?

In the test messages, after the character " " (space) the English data continued with **15 different characters**, and the Arabic data with **16**.

Now a distinction that matters after learning sampling-based generation:

- **Greedy decoding (always taking the highest-probability unit):** after the space the model answers the same every time — "t" in English, "ا" in Arabic — so it is wrong on every message where that character does not fit.
- **Sampling:** may answer differently depending on the seed, but **the distribution itself** remains the problem.

The real limit of Bigram is not "it picks one specific wrong character"; it is that it assigns **the same distribution to every position with the same previous character, no matter what came before**. The table never saw "Which room is it in?" or "I can't make it…" — it saw the space only.

And greedy generation on the training text shows this behaviourally: from "I'd" it produces "I'dI the the the…" in English, and from «أود» it produces «أودأسأسأس…» in Arabic — a loop made by the argmax after one character.

**The conclusion is deliberately scoped:** these two implementations failing here does not prove Bigram is theoretically helpless, nor that what comes next is the only fix. It proves that one previous character does not carry enough information here. The need for longer context has become an urgent, experiment-backed question — and that is Chapter 4's entrance.

---

## Check your understanding

**If you arrived here directly (without Chapter 1):**

1. What does α do in the counting Bigram? What happens to the loss of an unseen example without it?
2. Two models assign the correct token probabilities of 60% and 5%. Estimate each loss — which is better?

**If you came from the journey:**

3. Why did we choose characters for the first Bigram instead of the word vocabulary? And what did it cost?
4. Did counting lose to the network here, and what may we conclude from that?
5. Why did the model fail after the space, and what question does that failure open?

**Short answers:**

1. Smoothing guarantees no probability is exactly zero; without it, the loss of an unseen transition is infinite.
2. −log(0.6) ≈ 0.51 and −log(0.05) ≈ 3.0; the first is better because it gave a larger probability to what happened.
3. Because the small word vocabulary turned 5 of T1's 8 units into `<unk>`, while characters covered everything (0 unknown in T1's 41 characters; 1 unknown across all nine messages — the em-dash) at the cost of 41 units instead of 8.
4. Counting won in both languages on this small sample; you may not conclude that neural models are worse in general — both represent the same distribution with different estimation methods.
5. Because after the space the data continued with 15–16 different characters; greedy decoding gives one answer and sampling keeps the distribution narrow. The failure opens the longer-context question in Chapter 4.

---

## What this chapter does not claim

- **A very small sample:** ~712 training characters in Arabic and ~729 in English. The numbers belong to this data and these rules.
- **No comparison between the two languages:** each language was measured on its own data with its own tokenizer; nothing here may be read across the table as a language verdict.
- **The neural model was not fully tuned:** 200 epochs and one setting; we do not claim it reached a ceiling.
- **Generated text is not clean language, by design:** the purpose is to show the mechanism.
- **It does not prove human learning:** every metric here is over units, not over a learner's understanding.

## The bridge to Chapter 4

You now have: numbered character units, a probability table estimated by counting, a small neural counterpart behind it, and a fair numeric metric. And a documented limit: one previous character is not enough.

Chapter 4 starts at that limit: longer context — and we watch its effect on the loss and on the text, and discover what longer context costs in computation and memory.
