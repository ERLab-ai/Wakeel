#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
simulate_chapter3.py — يحسب كل رقم في الفصل الثالث من بياناته.

ثلاثة نماذج على نفس التمثيل (محارف، نفس المفردة، نفس البيانات):
  Unigram        خط أساس بلا سياق (مع تنعيم)
  Bigram بالعدّ  تنعيم إضافي، ويُختار α على التحقق
  Bigram عصبي    طبقة واحدة softmax، تهيئة مثبتة البذرة، تحسين تدريجي

الاستخدام: python tools/simulate_chapter3.py [--lang ar]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "content" / "chapter-03" / "examples.json"
OUT = ROOT / "content" / "chapter-03" / "results.json"

UNK = "<unk>"
START = "<s>"


def char_seq(msg):
    return [START] + list(msg)


def build_vocab(texts):
    chars = set()
    for t in texts:
        chars.update(char_seq(t))
    vocab = sorted(chars)
    if UNK not in vocab:
        vocab.append(UNK)
    return {c: i for i, c in enumerate(vocab)}


def encode(msg, vocab):
    return [vocab.get(c, vocab[UNK]) for c in char_seq(msg)]


def bigram_counts(seqs, V):
    C = np.zeros((V, V))
    for s in seqs:
        for a, b in zip(s[:-1], s[1:]):
            C[a, b] += 1
    return C


def smoothed_loss(C, seqs, alpha, V, out_mask=None):
    """متوسط -log P على كل الانتقالات، بتنعيم إضافي على مساحة المخرجات فقط.

    out_mask: مصفوفة منطقية بطول V؛ True = وحدة مسموح توليدها وتقييمها.
    رموز السياق الخاصة (<s>) تُستخدم سياقاً ولا تدخل مساحة المخرجات."""
    V_out = int(out_mask.sum())
    denom = C.sum(axis=1, keepdims=True) + alpha * V_out
    prob = (C + alpha) / denom
    prob[:, ~out_mask] = 0.0
    prob /= prob.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore"):
        logP = np.log(prob)
    logP[prob == 0] = -np.inf
    n, loss = 0, 0.0
    for s in seqs:
        for a, b in zip(s[:-1], s[1:]):
            loss -= logP[a, b]
            n += 1
    return loss / max(n, 1), n


def unigram_loss(seqs_train, seqs_eval, V, alpha=1.0, out_mask=None):
    counts = np.zeros(V)
    for s in seqs_train:
        for c in s:
            counts[c] += 1
    V_out = int(out_mask.sum())
    p = (counts + alpha) / (counts.sum() + alpha * V_out)
    p[~out_mask] = 0.0
    p /= p.sum()
    logp = np.log(p)
    n, loss = 0, 0.0
    for s in seqs_eval:
        for b in s[1:]:
            loss -= logp[b]
            n += 1
    return loss / max(n, 1)


def train_neural(seqs, V, epochs, lr, seed, out_mask=None):
    """Bigram عصبي: one-hot(prev) -> softmax(W x + b) على مساحة المخرجات المسموحة."""
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.01, size=(V, V))
    b = np.zeros(V)
    pairs = [(a, b) for s in seqs for a, b in zip(s[:-1], s[1:]) if out_mask is None or out_mask[b]]
    curve = []
    for ep in range(1, epochs + 1):
        gW = np.zeros_like(W)
        gb = np.zeros(V)
        loss = 0.0
        for a, b in pairs:
            z = W[a] + b
            z = z - z.max()
            p = np.exp(z)
            p[~out_mask] = 0.0
            p /= p.sum()
            loss -= np.log(max(p[b], 1e-12))
            d = p.copy()
            d[b] -= 1.0
            d[~out_mask] = 0.0
            gW[a] += d
            gb += d
        n = len(pairs)
        W -= lr * gW / n
        b -= lr * gb / n
        if ep in (1, 10, 50, 100, 150, 200) or ep == epochs:
            curve.append({"epoch": ep, "train_loss": round(loss / n, 4)})
    return W, b, curve


def neural_loss(W, b, seqs, out_mask=None):
    n, loss = 0, 0.0
    for s in seqs:
        for a, b2 in zip(s[:-1], s[1:]):
            if out_mask is not None and not out_mask[b2]:
                continue
            z = W[a] + b
            z = z - z.max()
            p = np.exp(z)
            p[~out_mask] = 0.0
            p /= p.sum()
            loss -= np.log(max(p[b2], 1e-12))
            n += 1
    return loss / max(n, 1)


def generate(C, alpha, vocab, inv_vocab, prompt, max_chars, stop_chars, seed, mode="count", W=None, b=None, greedy=False, out_mask=None):
    rng = np.random.default_rng(seed)
    V = len(vocab)
    den = C.sum(axis=1, keepdims=True) + alpha * V
    P = (C + alpha) / den
    P[:, ~out_mask] = 0.0
    P /= P.sum(axis=1, keepdims=True)
    out = list(prompt)
    cur = vocab[START] if START in vocab else 0
    steps = []
    for _ in range(max_chars):
        if mode == "count":
            p = P[cur].copy()
        else:
            z = W[cur] + b
            z = z - z.max()
            p = np.exp(z)
            p[~out_mask] = 0.0
            p /= p.sum()
        if greedy:
            nxt = int(np.argmax(p))
        else:
            nxt = int(rng.choice(V, p=p))
        ch = inv_vocab[nxt]
        if ch in stop_chars:
            steps.append({"char": ch, "p": round(float(p[nxt]), 3), "stop": True})
            break
        if ch == UNK:
            out.append("?")
        else:
            out.append(ch)
        steps.append({"char": ch, "p": round(float(p[nxt]), 3)})
        cur = nxt
    return "".join(out), steps


def worked_example():
    """المثال اليدوي — بالسياسة الموحدة: <s> سياق فقط، ومساحة المخرجات 6 وحدات."""
    from collections import Counter
    mini = "سأحضر سأحضر"
    seq = char_seq(mini)
    transitions = list(zip(seq[:-1], seq[1:]))
    ctx = Counter(a for a, _ in transitions)
    cnt = Counter(transitions)
    v_out, alpha = 6, 1.0
    probs = {f"{a}→{b}": round((c + alpha) / (ctx[a] + alpha * v_out), 3)
             for (a, b), c in sorted(cnt.items())}
    return {"mini_text": mini, "chars": len(mini), "transitions": len(transitions),
            "mini_vocab_size": 7, "output_vocab_size": v_out, "alpha": alpha,
            "probabilities": probs, "start_symbol_generated": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["ar", "en"])
    a = ap.parse_args()
    src = json.loads(SRC.read_text(encoding="utf-8"))
    langs = [a.lang] if a.lang else ["ar", "en"]
    results = {"generated_from": str(SRC.relative_to(ROOT)).replace("\\", "/"), "worked_example": worked_example(), "languages": {}}

    for lang in langs:
        cfg = src["corpus"][lang]
        train_msgs, val_msgs, test_msgs = cfg["train"], cfg["val"], cfg["test"]
        vocab = build_vocab(train_msgs)
        V = len(vocab)
        inv = {i: c for c, i in vocab.items()}
        tr = [encode(m, vocab) for m in train_msgs]
        va = [encode(m, vocab) for m in val_msgs]
        te = [encode(m, vocab) for m in test_msgs]

        counts_actual = {"train": sum(len(s) - 1 for s in tr),
                         "val": sum(len(s) - 1 for s in va),
                         "test": sum(len(s) - 1 for s in te)}

        C = bigram_counts(tr, V)

        out_mask = np.array([c != START for c in vocab])  # <s> سياق فقط

        # Unigram خط الأساس
        uni_val = unigram_loss(tr, va, V, out_mask=out_mask)
        uni_test = unigram_loss(tr, te, V, out_mask=out_mask)

        # Bigram بالعدّ + اختيار α على التحقق
        grid = src["smoothing_grid"]
        val_losses = {str(al): round(smoothed_loss(C, va, al, V, out_mask=out_mask)[0], 4) for al in grid}
        best_alpha = min(val_losses, key=lambda k: val_losses[k])
        alpha = float(best_alpha)
        big_val, _ = smoothed_loss(C, va, alpha, V, out_mask=out_mask)
        big_test, _ = smoothed_loss(C, te, alpha, V, out_mask=out_mask)

        # Bigram عصبي: مخرج V_out فقط، ومدخل V الكاملة (بما فيها <s> سياقاً)
        W, b, curve = train_neural(tr, V, src["neural"]["epochs"], src["neural"]["lr"],
                                   src["neural"]["init_seed"], out_mask=out_mask)
        nn_val = neural_loss(W, b, va, out_mask)
        nn_test = neural_loss(W, b, te, out_mask)

        # توليد ثابت
        g = src["generation"]
        prompt = train_msgs[0][: g["prompt_chars"]]
        gen_count, steps_c = generate(C, alpha, vocab, inv, prompt, g["max_chars"], g["stop_chars"],
                                      src["seed"], "count", out_mask=out_mask)
        gen_nn, steps_n = generate(C, alpha, vocab, inv, prompt, g["max_chars"], g["stop_chars"],
                                   src["seed"], "neural", W, b, out_mask=out_mask)
        gen_greedy, _ = generate(C, alpha, vocab, inv, prompt, g["max_chars"], g["stop_chars"],
                                 src["seed"], "count", greedy=True, out_mask=out_mask)

        # فحص التغطية على مستوى المحارف لرسائل الاختبار (لبُissão المحطة الأولى)
        char_check = {"total_chars": 0, "unknown_chars": 0, "per_message": []}
        for m in test_msgs:
            body = list(m)
            unk = [c for c in body if c not in vocab]
            char_check["total_chars"] += len(body)
            char_check["unknown_chars"] += len(unk)
            char_check["per_message"].append({"text": m, "chars": len(body),
                                              "unknown": sorted(set(unk))})
        char_check["t1"] = char_check["per_message"][0]

        # مثال فشل: سياق واحد يستمر بطرق مختلفة في الاختبار
        ctx = {}
        for m in test_msgs:
            s = char_seq(m)
            for prv, nxt in zip(s[:-1], s[1:]):
                ctx.setdefault(prv, set()).add(nxt)
        entropy_rank = sorted(ctx.items(), key=lambda kv: -len(kv[1]))
        top_ctx, variants = entropy_rank[0]
        variants = sorted(variants)
        # أثر ذلك على النموذج: أكثر محرف متوقع بعد top_ctx
        den = C.sum(axis=1, keepdims=True) + alpha * V
        ci = vocab[top_ctx]
        Prow = (C[ci] + alpha) / den[ci]
        best_next = inv[int(np.argmax(Prow))]

        results_lang = {
            "char_level_check": char_check,
            "units_actual": counts_actual,
            "targets": src["split_targets"],
            "vocab_size_incl_unk_and_start": V,
            "unk_id": vocab[UNK],
            "loss_examples": {"correct_char": lang and None},
            "models": {
                "unigram": {"val_loss": round(uni_val, 4), "test_loss": round(uni_test, 4)},
                "count_bigram": {"alpha_grid": grid, "val_losses": val_losses,
                                 "chosen_alpha": alpha,
                                 "val_loss": round(big_val, 4), "test_loss": round(big_test, 4)},
                "neural_bigram": {"curve": curve, "val_loss": round(nn_val, 4),
                                  "test_loss": round(nn_test, 4)},
            },
            "generation": {"prompt": prompt, "count_text": gen_count, "count_steps_head": steps_c[:6],
                           "neural_text": gen_nn, "greedy_count_text": gen_greedy,
                           "decoding_note": "النصوص أعلاه أخذ عينات ببذرة مثبتة؛ وضع greedy يختار الأعلى احتمالاً دائماً"},
            "failure_demo": {"context_char": top_ctx, "distinct_next_chars_in_test": variants,
                             "count_model_prediction": best_next},
        }
        results["languages"][lang] = results_lang

        print(f"===== {lang} ===== units {counts_actual}")
        print(f"  unigram : val {uni_val:.3f} test {uni_test:.3f}")
        print(f"  count   : alpha*={alpha} val {big_val:.3f} test {big_test:.3f}")
        print(f"  neural  : val {nn_val:.3f} test {nn_test:.3f} | curve {curve}")
        print(f"  gen     : sample={gen_count!r} greedy={gen_greedy!r}")
        print(f"  failure : بعد «{top_ctx}» تأتي {variants} والنموذج يتوقع «{best_next}» دائماً")
        print(f"  char-check: {char_check['unknown_chars']}/{char_check['total_chars']} محرفاً مجهولاً")

    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print("written:", OUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
