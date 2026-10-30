# Chapter 1 — How do we get a computer to handle our language?

> **This chapter is one connected experience, not five lessons.** You start with one problem, try to solve it with rules, test where the rules break, then try a different approach to the same problem. The same messages stay with you to the last line, and you will meet them again in Chapter 2.

**Expected time:** 25–40 minutes, most of it thinking rather than reading.
**What you need:** nothing. No programming, no mathematics, no prior AI knowledge.

---

## 1. The problem: our messages do not look alike

A small team holds a weekly meeting. Short messages arrive:

| # | Message |
|---|---|
| S1 | I'd like to postpone tomorrow's meeting to next week. |
| S2 | Sorry, I can't attend tomorrow. |
| S3 | Where is the meeting? |
| S4 | I'll be there on time. |
| S5 | Can we push the meeting back by half an hour? |
| S6 | What's the meeting link? |

We want a program that reads a message and replies appropriately. Only four replies are available:

| Code | Situation | Reply |
|---|---|---|
| R1 | reschedule | We'll find another slot and send a new invite. |
| R2 | apology / cannot attend | Thanks for letting us know; we'll send you the notes. |
| R3 | logistics question | The meeting is in Room 2; the link is in the invite. |
| R4 | confirmation | Great, see you there. |

If nothing fits: "I didn't understand that. Could you say what it's about?"

**Notice one thing before you start.** These are not questions about language; this is a real work problem. Every approach you try later will be judged by whether it answers these messages, not by how elegant it sounds.

---

## 2. First attempt: write rules

The obvious approach: look for words that signal the situation. Rule: "if one of these words appears, the message belongs to this case."

The reference rules used throughout this chapter (in order; the first matching rule wins):

| Rule | If one of these appears | Case |
|---|---|---|
| RULE-1 | postpone, reschedule, push the meeting, delay | R1 |
| RULE-2 | sorry, can't attend, cannot attend, apolog* | R2 |
| RULE-3 | where, link, room, location | R3 |
| RULE-4 | i'll be there, i will attend, confirm, count me in | R4 |

### Before you try: predict

Pick exactly one, and write it down before reading on:
- **A)** The rules will handle all six messages.
- **B)** They will handle five.
- **C)** They will handle four or fewer.

The three outcomes cannot all hold. Any of them is reasonable before the test; the difference appears in the next step.

### Try it

Apply the rules to S1–S6. Result computed with exactly these rules: **6 out of 6**.

Every message contained one of the keywords, so it matched its case. If you picked A, the problem now looks solved. But note: the six messages are the very ones the rules were written while looking at. Passing what you have already seen is not a test.

---

## 3. The test: change the wording

The real problem is not the six messages; it is the messages we have not seen. Here are nine new ones, all about the same meeting, worded differently:

| # | Message | Intended case |
|---|---|---|
| T1 | Could we move the meeting to another day? | R1 |
| T2 | I'm sorry, I won't be able to attend. | R2 |
| T3 | Which room is it in? | R3 |
| T4 | I can't make it tomorrow, apologies. | R2 |
| T5 | Can we shift the time to later? | R1 |
| T6 | I'll join you tomorrow. | R4 |
| T7 | What's the location? | R3 |
| T8 | Sorry to bother you — what's the meeting link? | R3 |
| T9 | I won't miss it this time. | R4 |

### Computed result: 4 out of 9

<details>
<summary>Per-message detail (click to expand)</summary>

| # | What the program did | Intended case | Correct? |
|---|---|---|---|
| T1 | no rule matched → "I didn't understand" | R1 | no |
| T2 | RULE-2 (sorry) | R2 | yes |
| T3 | RULE-3 (room) | R3 | yes |
| T4 | RULE-2 (apolog*) | R2 | yes |
| T5 | no rule matched → "I didn't understand" | R1 | no |
| T6 | no rule matched → "I didn't understand" | R4 | no |
| T7 | RULE-3 (location) | R3 | yes |
| T8 | RULE-2 (sorry) → apology reply | R3 | **no, and confidently wrong** |
| T9 | no rule matched → "I didn't understand" | R4 | no |

</details>

Four kinds of failure, not one. This is the real lesson of the chapter:

**a) A synonym.** "move" and "shift" mean reschedule, but they were not in the keyword list. The rule did not reason wrongly; it simply lacked the expression.

**b) A phrasing we did not anticipate.** "I'll join you" carries a confirmation, but RULE-4 waits for "i'll be there". Same intent, different words.

**c) A match in the wrong place.** T8 is not an apology for absence; the sender apologises for the bother and asks for the link. But "sorry" appeared first, so the apology reply went out. The program did not merely fail to understand; it answered confidently and wrongly — worse than "I didn't understand", because a user may trust it.

**d) Negation.** T9 says "I won't miss it" — that is, I will attend. But "won't" usually appears in apologies, and "miss" is neutral. Neither the rules nor the next method understands that "won't + miss" means presence. Negation is not a keyword; it is an operation on meaning.

### Stop here

Before reading the alternative, answer for yourself: can this be fixed by adding words? Yes, partly. Add "move", "shift", "join" and T1, T5, T6 will pass. But T8 will remain: the issue is not a missing word, but that "sorry" is not sufficient evidence of an intent. And T9 will remain: negation needs understanding, not a list.

**A provisional conclusion, not a verdict on rules:** written rules work when phrasings are limited and known in advance, and become brittle when expression varies. This is not a defect of programming; it is a property of language. Meaning is spread across many words, and negation and ambiguity are part of language, not bugs in it.

---

## 4. The alternative: classifying by the nearest stored example

Instead of writing "if word X appears", we keep examples: a message and its correct case. When a new message arrives, we look for the **nearest stored example** and reuse its case.

### What this method is — and where its limit is

**It is:** instance-based learning — also called lazy learning — a recognised machine-learning approach: it stores the examples and defers computation to query time.

**Its limit in this implementation:** it does not abstract linguistic rules, does not train weights that change, and uses no neural network. The examples stay as they are; the only possible changes are adding examples or changing the similarity measure.

And we will return later to a question that separates two stages: what is the difference between storing examples and searching for the nearest, and learning parameters that change during training?

### How do we measure "nearest"?

We use one explicit definition, because any other definition gives different results:

1. Turn text into words: lowercase, remove punctuation, split on spaces.
2. Represent each message as a set of words (no duplicates).
3. The score = shared words ÷ all words in the two sets combined.
4. Take the highest score. On a tie, take the earlier example in the list.

**Worked example:** T3 "Which room is it in?" = {which, room, it, is, in}. E8 "Which room is the meeting in?" = {which, room, the, meeting, is, in}. Shared = {which, room, is, in} = 4, combined = 7 words, so the score is 4/7 ≈ 0.57. Every other example shares at most one or two words with T3, so E8 wins clearly.

The twelve examples:

<details>
<summary>Example list (click to expand)</summary>

| # | Example | Case |
|---|---|---|
| E1 | Can we postpone the meeting to tomorrow? | R1 |
| E2 | I need another slot for the meeting. | R1 |
| E3 | Should we delay the meeting? | R1 |
| E4 | Sorry, I can't attend. | R2 |
| E5 | I cannot attend, apologies. | R2 |
| E6 | Sorry, I won't be there tomorrow. | R2 |
| E7 | Where is the meeting? | R3 |
| E8 | Which room is the meeting in? | R3 |
| E9 | What's the meeting link? | R3 |
| E10 | I will attend tomorrow. | R4 |
| E11 | Count me in, I'll be there. | R4 |
| E12 | Confirming my attendance. | R4 |

</details>

### Before you try: predict

Pick exactly one — the ranges do not overlap:
- **A)** It will get 0 to 4 messages right.
- **B)** 5 to 7.
- **C)** 8 to 9.

### Computed result: 8 out of 9

<details>
<summary>Per-message detail (click to expand)</summary>

| # | Closest example | Shared words | Case | Correct? |
|---|---|---|---|---|
| T1 | E1 | meeting, the, to, we | R1 | yes |
| T2 | E6 | be, i, sorry, wont | R2 | yes |
| T3 | E8 | which, room, is, in | R3 | yes |
| T4 | E4 | cant, i | R2 | yes |
| T5 | E1 | can, the, to, we | R1 | yes |
| T6 | E10 | tomorrow | R4 | yes |
| T7 | E9 | the, whats | R3 | yes |
| T8 | E9 | link, meeting, the, whats | R3 | yes |
| T9 | E6 | i, wont | R2 | **no** |

</details>

**What changed?** T8, which tripped the rules, is solved here: "meeting", "link" and "whats" resemble E9 more than any apology example. This shows the method's strength: it does not look for a single word, but for **resemblance** between messages.

**What did not change?** T9 failed both methods. Similarity cannot see that "won't miss" is a negation of absence, hence a confirmation. The problem is not a shortage of examples or words; both methods work on shape, not meaning.

Note T5, though: its match with E1 rested on four shared words — can, we, the, to — and every one of them is a function word. The resemblance was real, but it carried no content: a small wording change could flip the nearest example.

### Do not trust the number before testing it

8 of 9 looks better than 4 of 9, but on nine messages. The number does not say the second method is better in general; it says it happened to succeed on this sample. Watch what happens when the examples change:

| Change in the examples | Result on T1–T9 |
|---|---|
| Examples as they are (12) | 8 of 9 |
| Remove the confirmation examples (R4) | 7 of 9 — fails T6 and T9 |
| Remove the apology examples (R2) | 7 of 9 — fails T2 and T4 |
| Keep only the first eight examples | 6 of 9 — fails T6, T8, T9 |

These numbers are computed the same way, not assumed. Their direct meaning: **the examples are the program's knowledge.** Remove a kind of example and the capability disappears. There is no "understanding" that compensates for the gap.

### So what does it mean?

- **Rules** suit cases that are limited, phrasings that are known, and situations where you must know **why** the program decided.
- **Nearest example** suits varied phrasing with labelled data, and situations where explaining every decision is not required — or not yet possible.
- **These two implementations** failed on T9 because both compare shape. That does not prove every rule system fails at negation, or every example-based classifier would — what we showed is the limit of these implementations on this sentence, not the impossibility of the approaches.

---

## 5. A new question, not a final answer

We succeeded in making the program choose a reply from four replies we wrote in advance. But what if we wanted a different reply, tailored to the details of each message?

We could try composing replies with rules and templates: "Sorry + [reason] + [alternative slot]". That is genuinely possible, and rules can assemble sentences we never wrote. But it needs a template for every message type, a rule for choosing the pieces, and it stays limited to the phrasings we can anticipate.

So could we instead train a model that learns to generate text from examples? And how could the task of writing a new sentence be broken into steps a computer can execute?

This is the question that opens Chapter 2. We will not answer it now. But notice it did not come from nowhere: it came from a clear limit of two innocent attempts — not from a claim that rules are helpless at everything.

---

## Check your understanding

Answer before looking:

1. New message: "Can we change the meeting to Thursday?" — do the reference rules handle it? Why?
2. Message: "Sorry, I'll be ten minutes late." — what mistake might the rules make? What kind is it?
3. If you added "move" to RULE-1, what improves? What does not change?
4. Why did both methods fail on T9? What would a fix need?
5. Order them: when do you choose rules, when examples, and when does neither suffice? Give an example of your own that is not in this chapter.

**Short answers:**

1. No. "change" and "Thursday" are not in RULE-1; it falls back. Cause: a new synonym, not faulty logic.
2. It may match RULE-2 ("sorry") and send the absence reply, while the message is a delay notice. Kind: a match in the wrong place — worse than no answer.
3. Only T1 improves. T5 ("shift") and T8 ("sorry" inside a logistics question) will not change; the first needs another synonym, the second needs intent discrimination, not another word.
4. Because "won't miss" is a negation that asserts attendance, and the two implementations we ran compare words rather than meaning. A fix needs a representation of negation or examples covering that construction — an open question, not something we proved here.
5. Rules when cases are limited and you want explanation; examples when phrasing varies and you have labelled data; neither when meaning lives in negation, ambiguity or context.

**The real test of understanding:** if you can describe, with a new example that does not appear here, what each method does, why it succeeds or fails in a given case, and what cannot be concluded from the experiment, you have understood the idea rather than the wording.

---

## What this chapter does not claim

- **It is not a scientific experiment.** 6 messages, 9 messages and 12 examples are a very small sample chosen by the author. Changing them changes the numbers.
- **It does not say learning from examples beats rules.** 8/9 versus 4/9 on this sample carries no general verdict, and another sample could reverse it. The examples were chosen after seeing the tests, so the comparison is a designed teaching demonstration, not an independent measurement.
- **It does not say rules are useless.** Many systems run successfully on limited rules, and rules can compose sentences through templates.
- **It does not prove that you learned.** The chapter offers ideas and exercises; showing an effect on learners requires a separate study, not one text.

## The bridge to Chapter 2

You leave knowing something specific: our program chooses from four fixed replies. Templates could assemble new replies, but that requires a rule for every phrasing we expect.

Chapter 2 opens with: can we learn generation from examples instead of writing a rule for every case? Then the question forces itself: what is the unit we write with? Only then do we reach tokenization — not before.
