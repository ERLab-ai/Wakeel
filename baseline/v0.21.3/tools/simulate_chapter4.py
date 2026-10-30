#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
simulate_chapter4.py — تجربتا الفصل الرابع من بياناته.

التجربة 1: تمايز السياق (1/2/3 محارف) على الرسالتين المرجعيتين.
التجربة 2: مقارنة كمية k = 0..3 (Unigram/Bigram/Trigram/4-gram)
           بنفس التمثيل والبيانات، α لكل نموذج من التحقق فقط.

السياسات المعلنة:
  * مساحة المخرجات تستبعد <s> (رمز سياق فقط).
  * سياق غير مرصود في التدريب: تنعيم إضافي → توزيع منتظم على مساحة المخرجات.
  * Backoff (استكمال نحساب Stupid Backoff بمعامل 0.4) تجربة منفصلة لا تدخل
    في جدول النماذج الأساسية.
  * حدود الرسائل محترمة: كل رسالة تُرمَّز باستقلال مع حشو <s> حتى k.

الاستخدام: python tools/simulate_chapter4.py [--lang ar]
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "content" / "chapter-03" / "examples.json"
OUT = ROOT / "content" / "chapter-04" / "results.json"
VERSION = "simulate_chapter4-1.0"

UNK = "<unk>"
START = "<s>"
KS = [0, 1, 2, 3]
ALPHA_GRID = [0.1, 0.5, 1.0]
BACKOFF_WEIGHT = 0.4


def body_chars(msg):
    """محارف الرسالة نفسها (بلا رمز البداية)."""
    return list(msg)


def padded(msg, k):
    """[<s>]*k ثم محارف الرسالة — حدود الرسائل محفوظة."""
    return [START] * k + body_chars(msg)


def ngrams(seqs, k, out_ids):
    """يولد (سياق tuple، الهدف) من كل رسالة باستقلال."""
    for s in seqs:
        p = [START] * k + list(s)
        for i in range(k, len(p)):
            ctx = tuple(p[i - k:i]) if k > 0 else ()
            yield ctx, p[i]


class NGram:
    """N-gram بالعدّ مع تنعيم إضافي؛ سياق غير مرصود → توزيع منتظم على المخرجات."""

    def __init__(self, k, train_seqs, vocab, out_ids, alpha):
        self.k = k
        self.alpha = alpha
        self.V_out = len(out_ids)
        self.out_pos = {full: pos for pos, full in enumerate(out_ids)}
        self.ctx_counts = Counter()
        self.pair_counts = Counter()
        for ctx, nxt in ngrams(train_seqs, k, out_ids):
            self.ctx_counts[ctx] += 1
            self.pair_counts[(ctx, nxt)] += 1
        self.sparse_records = len(self.pair_counts)
        self.observed_ctx_train = len(self.ctx_counts)

    def dist(self, ctx):
        """توزيع على مساحة المخرجات فقط. سياق غير مرصود → منتظم."""
        c = self.ctx_counts.get(ctx, 0)
        p = np.full(self.V_out, self.alpha / (c + self.alpha * self.V_out))
        if c > 0:
            for (c2, b), n in self.pair_counts.items():
                if c2 == ctx:
                    p[self.out_pos[b]] += n / (c + self.alpha * self.V_out)
        return p

    def loss(self, seqs, out_ids):
        total, n, unseen = 0.0, 0, 0
        for ctx, tgt in ngrams(seqs, self.k, out_ids):
            p = self.dist(ctx)
            total -= np.log(max(p[self.out_pos[tgt]], 1e-12))
            n += 1
            if self.ctx_counts.get(ctx, 0) == 0:
                unseen += 1
        return {"nll": round(total / max(n, 1), 4), "tokens": n,
                "unseen_contexts": unseen,
                "unseen_context_rate": round(unseen / max(n, 1), 4)}

    def greedy_next(self, ctx, out_ids):
        p = self.dist(tuple(ctx))
        return int(np.argmax(p)), p


def seqs_for(msgs, vocab, k):
    """ترميز رسائل إلى معرفات مع حشو <s> حتى k (الهدف لا يكون <s> أبداً)."""
    out = []
    for m in msgs:
        body = list(m)
        ids = [vocab.get(c, vocab[UNK]) for c in body]
        out.append(([vocab[START]] * k, ids))
    return out


def iter_eval(split, vocab, k, out_ids):
    for padded_ctx, ids in split:
        p = [START] * k + ids
        for i in range(k, len(p)):
            ctx = tuple(p[i - k:i]) if k > 0 else ()
            yield ctx, p[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["ar", "en"])
    a = ap.parse_args()
    src = json.loads(SRC.read_text(encoding="utf-8"))
    langs = [a.lang] if a.lang else ["ar", "en"]

    raw = SRC.read_bytes()
    input_hash = hashlib.sha256(raw).hexdigest()

    results = {"simulator_version": VERSION,
               "generated_from": str(SRC.relative_to(ROOT)).replace("\\", "/"),
               "input_hash": input_hash,
               "languages": {}}

    for lang in langs:
        cfg = src["corpus"][lang]
        train_msgs, val_msgs, test_msgs = cfg["train"], cfg["val"], cfg["test"]

        # مفردة السياق (من التدريب): المحارف + <s> + <unk>
        chars = set()
        for m in train_msgs:
            chars.update(m)
            chars.add(START)
        vocab = {c: i for i, c in enumerate(sorted(chars))}
        vocab.setdefault(UNK, len(vocab))
        inv_vocab = {i: c for c, i in vocab.items()}
        V = len(vocab)
        out_ids = [i for c, i in vocab.items() if c != START]  # مساحة المخرجات
        out_mask = np.zeros(V, dtype=bool)
        for i in out_ids:
            out_mask[i] = True

        train_seqs = [[vocab.get(c, vocab[UNK]) for c in m] for m in train_msgs]

        models = {}
        def ids_of(msgs):
            return [[vocab.get(c, vocab[UNK]) for c in m] for m in msgs]
        case_studies = []
        generation = []

        for k in KS:
            model = NGram(k, train_seqs, vocab, out_ids, alpha=None)
            # اختيار α على التحقق فقط
            val_sel = {}
            for al in ALPHA_GRID:
                model.alpha = al
                m_ = model.loss(ids_of(val_msgs), out_ids)
                val_sel[str(al)] = m_["nll"]
            best_alpha = float(min(val_sel, key=lambda x: val_sel[x]))
            model.alpha = best_alpha

            def eval_split(msgs):
                return model.loss(ids_of(msgs), out_ids)

            tr = eval_split(train_msgs)
            va = eval_split(val_msgs)
            te = eval_split(test_msgs)

            def unique_ctx(msgs):
                seen_ctx = set()
                for m in msgs:
                    p = [vocab[START]] * k + [vocab.get(c, vocab[UNK]) for c in m]
                    for i in range(k, len(p)):
                        seen_ctx.add(tuple(p[i - k:i]))
                return len(seen_ctx)

            dense_cells = (V ** k) * (V - 1) if k > 0 else (V - 1)
            models[f"k{k}"] = {
                "name": {0: "Unigram", 1: "Bigram", 2: "Trigram", 3: "4-gram"}[k],
                "alpha_from_validation": best_alpha,
                "val_losses_grid": val_sel,
                "train_nll": tr["nll"], "val_nll": va["nll"], "test_nll": te["nll"],
                "eval_tokens": te["tokens"],
                "observed_contexts_val": None,  # يُملأ أدناه
                "unseen_context_rate_val": va["unseen_context_rate"],
                "unseen_context_rate_test": te["unseen_context_rate"],
                "theoretical_dense_cells": int(dense_cells),
                "sparse_storage_records": model.sparse_records,
                "observed_unique_contexts_train": model.observed_ctx_train,
                "observed_pairs_train": model.sparse_records,
                "observed_unique_contexts_val": unique_ctx(val_msgs),
                "observed_unique_contexts_test": unique_ctx(test_msgs),
            }

        # التجربة 1: تمايز السياق على الرسالتين المرجعيتين
        ref = {
            "ar": ["أين القاعة؟", "لن أستطيع المجيء، أرجو أن تعذرني."],
            "en": ["Which room is it in?", "I can't make it tomorrow, apologies."],
        }[lang]
        for msg in ref:
            body = list(msg)
            # موضع الفضاء الأول (بعد الكلمة الأولى)
            sp = body.index(" ")
            true_next = body[sp + 1]  # مشتق من موضع حقيقي في النص، لا مكتوب يدوياً
            rows = []
            for k in (1, 2, 3):
                ctx_units = ([START] * k + body[:sp + 1])[-k:]
                ctx = tuple(vocab.get(c, vocab[UNK]) for c in ctx_units)
                model = NGram(k, train_seqs, vocab, out_ids, alpha=1.0)
                # كل المتابعين الفعليين في التدريب — بلا حصر بمرشحي رسالة الاختبار
                followers = {}
                for (c2, b2), n in model.pair_counts.items():
                    if c2 == ctx:
                        followers[inv_vocab[b2]] = n
                context_seen = model.ctx_counts.get(ctx, 0) > 0
                rows.append({"k": k, "context": "".join(ctx_units),
                             "context_seen_in_training": context_seen,
                             "context_training_count": model.ctx_counts.get(ctx, 0),
                             "training_followers": dict(sorted(followers.items(), key=lambda x: -x[1])),
                             "true_next": true_next,
                             "true_next_in_training_followers": true_next in followers,
                             "true_next_count": followers.get(true_next, 0)})
            case_studies.append({"message": msg, "true_next": true_next, "contexts": rows})

        # تجربة منفصلة: Stupid Backoff من k=3 نزولاً بمعامل 0.4
        pair_by_k, ctx_by_k = {}, {}
        for kk in (0, 1, 2, 3):
            cc, pp = Counter(), Counter()
            for m in train_msgs:
                p = [START] * kk + list(m)
                for i in range(kk, len(p)):
                    ctx2 = tuple(p[i - kk:i]) if kk > 0 else ()
                    cc[ctx2] += 1
                    pp[(ctx2, p[i])] += 1
            ctx_by_k[kk] = cc
            pair_by_k[kk] = pp

        def sb(ctx, tgt, kk, weight=0.4):
            if kk < 0:
                return 1.0 / V
            c = ctx_by_k[kk].get(ctx, 0)
            if c > 0 and pair_by_k[kk].get((ctx, tgt), 0) > 0:
                return pair_by_k[kk][(ctx, tgt)] / c
            shorter = ctx[1:] if kk > 0 else ()
            return weight * sb(shorter, tgt, kk - 1, weight)

        def sb_loss(msgs, weight=0.4):
            total, n = 0.0, 0
            for m in msgs:
                p3 = [START] * 3 + list(m)
                for i in range(3, len(p3)):
                    ctx = tuple(p3[i - 3:i])
                    s = sb(ctx, p3[i], 3, weight)
                    total -= np.log(max(min(s, 1.0), 1e-12))
                    n += 1
            return round(total / max(n, 1), 4)

        backoff = {"method": "Stupid Backoff (weight=0.4), k=3→0",
                   "separate_from_base_models": True,
                   "val_loss": sb_loss(val_msgs),
                   "test_loss": sb_loss(test_msgs)}

        # التوليد عبر المستويات (عينات ببذرة 42)
        seed = src.get("seed", 42) if "seed" in src else 42
        for k in KS:
            model = NGram(k, train_seqs, vocab, out_ids, alpha=1.0)
            model.alpha = models[f"k{k}"]["alpha_from_validation"]
            rng = np.random.default_rng(seed)
            ctx = tuple([vocab[START]] * k)
            out = []
            for _ in range(120):
                p = model.dist(ctx)
                nxt = int(rng.choice(len(p), p=p))
                ch = [c for c, i in vocab.items() if i == out_ids[nxt]][0]
                if ch in (".", "؟"):
                    break
                out.append(ch)
                ctx = tuple((list(ctx) + [out_ids[nxt]])[1:]) if k > 0 else ()
            generation.append({"k": k, "seed": seed, "decoding": "sampling",
                               "text": "".join(out)})
        # greedy للمقارنة
        model2 = NGram(2, train_seqs, vocab, out_ids, alpha=1.0)
        model2.alpha = models["k1"]["alpha_from_validation"]
        rng = np.random.default_rng(seed)
        ctx = tuple([vocab[START]] * 2)
        out = []
        for _ in range(80):
            p = model2.dist(ctx)
            nxt = int(np.argmax(p))
            ch = [c for c, i in vocab.items() if i == out_ids[nxt]][0]
            if ch in (".", "؟"):
                break
            out.append(ch)
            ctx = tuple((list(ctx) + [out_ids[nxt]])[1:])
        generation.append({"k": 2, "decoding": "greedy", "text": "".join(out)})

        results["languages"][lang] = {
            "vocab_size_incl_specials": V,
            "output_space": V - 1,
            "models": models,
            "backoff": backoff,
            "case_studies": case_studies,
            "generation": generation,
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")

    for lang in langs:
        L = results["languages"][lang]
        print(f"===== {lang} =====")
        for k in KS:
            m = L["models"][f"k{k}"]
            print(f'  k={k} {m["name"]:8} alpha={m["alpha_from_validation"]} '
                  f'train={m["train_nll"]} val={m["val_nll"]} test={m["test_nll"]} '
                  f'dense={m["theoretical_dense_cells"]:>9,} sparse={m["sparse_storage_records"]:>5,}')
        for cs in L["case_studies"]:
            print(f'  case {cs["message"][:24]!r} | الهدف بعد المسافة: {cs["true_next"]!r}')
            for row in cs["contexts"]:
                flw = row["training_followers"]
                flw_s = ", ".join(f"{ch}×{n}" for ch, n in list(flw.items())[:6]) or "لا شيء"
                print(f'    k={row["k"]} ctx={row["context"]!r} seen={row["context_seen_in_training"]} '
                      f'count={row["context_training_count"]} | الهدف: {row["true_next"]!r} '
                      f'في المتابعين: {row["true_next_in_training_followers"]} | متابعون: {flw_s}')
        b = L["backoff"]
        print(f'  backoff : val {b["val_loss"]} test {b["test_loss"]} (منفصلة)')
        for g in L["generation"]:
            print(f'  gen k={g["k"]} {g["decoding"]}: {g["text"][:60]!r}')
    print("written:", OUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
