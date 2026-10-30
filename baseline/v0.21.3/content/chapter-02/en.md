# Chapter 2 — How does a computer write a sentence it has never written?

> **Chapter 1 ended at a clear limit:** our program chooses from four fixed replies, and templates can compose — but need a rule for every phrasing we anticipate. This chapter walks the same four steps: we want a new reply; we try a template; we ask about learning; and we are forced into the question of units.
>
> **This chapter can be read on its own.** Everything you need is here.

**Expected time:** 30–45 minutes.
**What you need:** nothing.

---

## 1. We write a new reply — by hand, then with a template

Take message T1 from Chapter 1: "Could we move the meeting to another day?" — case R1 (reschedule).

### Step one: by hand

Write an appropriate reply you have not seen before. Then write down **what information you needed** (the topic? a proposed alternative? a polite tone?) and **what decisions you made**. We are not asking you to describe how your hand picked each word; we are asking you to notice the size of the decision.

### Step two: with a template

Here is one template:

> Sorry for the bother — regarding "{topic}", we can {alternative}, and we'll send an updated invite.

The template knows nothing; someone has to supply the slot values. The three messages used in this experiment are shown in full so the chapter stays self-contained: T1 "Could we move the meeting to another day?", T5 "Can we shift the time to later?", T3 "Which room is it in?". These results are computed from the chapter's data:

| Message | Slots | Output |
|---|---|---|
| T1 | topic = "the meeting", alternative = "move it to another day" | "Sorry for the bother — regarding "the meeting", we can move it to another day, and we'll send an updated invite." |
| T5 | topic = "the time", alternative = "shift it later" | "Sorry for the bother — regarding "the time", we can shift it later, and we'll send an updated invite." |
| T3 | topic = "the meeting", alternative = **?** | **Failed: no value for [alternative]** |

Look at T3 "Which room is it in?": a logistics question, not a reschedule request. There is no "alternative" to propose, so the template produces nothing. And the slot values themselves were written by a human into the data; the template did not find them. Who finds slot values? That question will follow us.

### The template's limit

Each case may need its own template, but a general template or a conditional template system could cover more than one. The point is not the number of templates; it is that the cost of composing grows with phrasing variety — every unanticipated phrasing charges a fee.

### The third step of the bridge

At that point the fair question is: **could we learn a way to generate text from examples?** Step-by-step generation — producing the text one unit at a time — is one possible option, not the only one. We chose it for a practical reason: we can build it step by step and study its limits. And to build it, we first need to define the **units** it will produce.

---

## 2. What unit do we write with?

Suppose we chose step-by-step generation. The first practical question: the unit the program produces — what is it?

Take the English sentence: "I won't miss the meeting."

| Counting method | Result |
|---|---|
| Whitespace words (after removing punctuation) | 6 |
| Characters (no spaces) | 21 |
| Pieces after splitting contractions | 6 |

A human reads five words. Our program saw **six pieces**, because it removes the apostrophe and splits won't into won and t. (In the Arabic version of this chapter, the sentence «وسيشاركهم القرار في اجتماع الغد.» is one whitespace word carrying a prefix and pronouns — same lesson, language-authentic form.) The conclusion is the same: **the unit is a decision the tool's designer makes; it is not given in the text.**

---

## 3. Tokenization: naming the tool

The tool that splits text into units according to a defined method has a name: **Tokenizer**. Its decision is not a detail: the unit it chooses determines what any later model sees.

We will use an educational tokenizer pinned to a version (`edu-tok-0.1`) with only two published rules: split a leading "و" if at least four characters remain, and split a leading "ال" if at least two remain. These rules are invented for teaching and pinned so results repeat. Modern tokenizers use subword units and fallback paths — we mention that and do not implement it.

### Experiment 1: the same text, two methods

| Text | Method | Units |
|---|---|---|
| "I'll share the decision at tomorrow's meeting." | whitespace words (apostrophes kept) | 7 |
| "I'll share the decision at tomorrow's meeting." | punctuation removed, then whitespace | 9 |

The text never changed; the method did — and the count moved from 7 to 9, because removing the apostrophe splits I'll into i and ll, and tomorrow's into tomorrow and s. The two prefix rules of `edu-tok-0.1` add nothing here: they were written for Arabic, and a tokenizer carries its design origin with it. (In the Arabic version, the same experiment moves the count from 5 to 8, because those prefix rules do bite.)

### Experiment 2: two texts on the same topic, one pinned tokenizer

| Text | `edu-tok-0.1` |
|---|---|
| «وسيشاركهم القرار في اجتماع الغد.» | 8 |
| "I'll share the decision at tomorrow's meeting." | 9 |

The two texts share a topic; they are not exact meaning-equivalents, so the comparison is approximate. An observed difference: one token. What can we conclude? Nothing definitive. The difference depends on the tokenization method, the punctuation rules, and the wording of the two sentences — not on the "nature" of the language. The common conclusion — "language X is more expensive" — is banned here on two sentences and one tokenizer.

### And "cost" is not this sentence either

Token count affects some aspects of computation and consumption, but actual cost also depends on the model itself, the hardware, the implementation, and pricing if any. Do not turn the token counter into an economic verdict.

---

## 4. Vocabulary and token IDs — with the same tokenizer

For the program to think in units, it must turn them into numbers through a **vocabulary**: a list of accepted units, each with a fixed ID.

One consistency decision first: **we build the vocabulary with the same tokenizer we just studied** (`edu-tok-0.1`), not with a different method. Had we built it from whitespace words after teaching the tokenizer, the rules would have silently changed between two sections.

### Building the vocabulary

We applied `edu-tok-0.1` to the twelve examples, counted frequency, and took the top 12 units. `<unk>` gets ID 0 and is a special token — it is not counted in the vocabulary size:

| ID | Token | ID | Token |
|---|---|---|---|
| 0 | `<unk>` | 7 | we |
| 1 | the | 8 | sorry |
| 2 | meeting | 9 | t |
| 3 | i | 10 | be |
| 4 | tomorrow | 11 | there |
| 5 | can | 12 | is |

Three things the table itself reveals:

1. Lowercasing is part of this tokenizer's English handling: "Could" and "could" are one unit — which is why "could" in T1 maps to nothing: the word never appears in the twelve examples at all, so it has no ID, while "we" maps to 7.
2. **t** is in the vocabulary as a "word" — a direct consequence of splitting contractions. The tokenizer's design decisions create odd units. (In the Arabic version, the number-one unit is «ال»: the prefix rule splits it off every definite noun, so «الاجتماع» became «ال» + «اجتماع», and «اللقاء» broke into «ال» + «لقاء» — where «لقاء» is a real Arabic word that simply never appeared in the corpus, so it became `<unk>`. Naive rules create new units, and even real words go missing when unseen.)
3. Normalization rule: Arabic diacritics are not stripped or normalized — a diacritic stays attached to its word, so a diacriticised word is a different unit. Lowercasing applies to English only.

### Encoding a new message

T1: "Could we move the meeting to another day?"

| Token | could | we | move | the | meeting | to | another | day |
|---|---|---|---|---|---|---|---|---|
| ID | 0 | 7 | 0 | 1 | 2 | 0 | 0 | 0 |

Eight units, only three found in the vocabulary (ID 0 means `<unk>`). That is what a small vocabulary from small data produces: most of the new message becomes "unknown".

### What if we grow the vocabulary?

| Vocabulary size (excluding `<unk>`) | Coverage of the units in the nine messages |
|---|---|
| 8 | 32.3% |
| 12 | 40.3% |
| 20 | 50% |

Growth is real and slow: from 8 to 20 units raised coverage from 32% to 50%. Many words stayed out ("could", "move", "another", "day"…). Real vocabularies range from tens of thousands to hundreds of thousands of units (GPT-2's, for example, is 50,257), depending on the model, the tokenizer, the data, and engineering decisions. Modern tokenizers avoid `<unk>` with subword units and byte fallbacks — mentioned here, not implemented.

**A question we deliberately defer:** our vocabulary is a stored list that does not change. What is the difference between that and parameters that change during training? That is Chapter 3's question, and the answer will mark the line between everything built so far and what comes next.

---

## 5. Where we are, and where Chapter 3 is

| Step | Status |
|---|---|
| A new reply, different from the list | Tried by hand and by template; both limited to the phrasings we anticipate |
| Learning to generate | Step-by-step generation is the approach we chose to build — among options |
| Defining the unit | Done: Tokenizer, vocabulary, token IDs |
| **How does the model learn to choose the next unit?** | **Chapter 3's question** |

---

## Check your understanding

**If you arrived here directly (without Chapter 1):**

1. Explain in three sentences what a Tokenizer does.
2. What is the difference between **text**, a **token**, and a **token ID**? Use "won't" in your answer.
3. Why does the system need an explicit rule for units outside the vocabulary?

**If you came from the journey:**

4. Why did our system need tokenization? Start from "template" and end at "unit".
5. Message T9 in Chapter 1 failed both methods. Does the template or the vocabulary alone handle negation? What should any solution know before we judge it?

**Short answers:**

1. It splits text into units according to a defined, pinned method and turns them into something the program processes. The method is a decision, and results change when it changes.
2. Text is the sentence as a human wrote it; a token is the unit the tokenizer decided on — it may match a word, part of one, or two glued together (won't → won, t); the ID is the number assigned to that token inside a specific vocabulary — it belongs to the vocabulary, not the text.
3. Because the program cannot encode what is not in its list; with no explicit rule it either crashes or behaves unpredictably. The rule (`<unk>` here) is a declared engineering decision.
4. The template solves messages of the type it was designed for; the vocabulary is a representation, not a decision. The need came from phrasing variety, which led us to generation by units, which led to tokenization.
5. No. A template needs one template per negation phrasing; the vocabulary represents shape, not intent. Any solution first needs a representation of meaning or examples covering the construction — nothing in these two chapters proves a particular fix suffices.

**The test of understanding:** if you can explain to a friend why the same sentence yields different token counts under different tokenizers, and why a template is not enough — you understood the chapter rather than memorised it.

---

## What this chapter does not claim

- **The educational tokenizer has invented rules.** `edu-tok-0.1` is pinned for teaching; it is not a production tokenizer and does not represent modern subword algorithms. The vocabulary built with it inherits their quirks: «ال» is the top Arabic unit, "t" is an English unit, and «قاء» is a meaningless fragment.
- **The numbers are specific to these texts and rules.** Change a sentence or a rule and the results change. No verdict here on English, Arabic, or any language's efficiency.
- **Tokenization does not understand.** It is representation and an engineering decision; it does not register diacritics as meaning or prevent errors.
- **It does not prove learning.** The comprehension checks are questions and exercises; showing an effect on learners requires a study.
- **No claim that step-by-step generation is the only solution.** It is the option we chose to build and study.

## The bridge to Chapter 3

You leave knowing: the program now sees text as numbered units from a known vocabulary. But it does not yet know **how to choose the next unit**, or what would make a choice "right".

Chapter 3 starts there: we need a measurable objective for choosing the next unit, and a mechanism that learns from examples. And then the deferred difference appears: between a stored list that does not change, and parameters that change during training.
