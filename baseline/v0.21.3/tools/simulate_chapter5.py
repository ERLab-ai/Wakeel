#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
simulate_chapter5.py — MLP بالمحارف (k=2) مقابل Trigram بالعدّ.

المعمارية (المواصفة المعتمدة):
  E: [V_ctx, 8]  (Embedding قابل للتعلم أو مجمد حسب التجربة)
  concat: [16]
  linear_1: [16, 32] + bias, tanh
  linear_2: [32, V_out] + bias, softmax   (V_out يستبعد <s>)

teacher forcing على كل المواضع؛ بذور مثبتة؛ اختيار epoch على التحقق فقط.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "content" / "chapter-03" / "examples.json"
OUT = ROOT / "content" / "chapter-05" / "results.json"
VERSION = "simulate_chapter5-1.0"

START, UNK = "<s>", "<unk>"
K, EMB, HID = 2, 8, 32
EPOCHS, LR, SEED = 300, 0.5, 42


def body(msg):
    return list(msg)


def padded(msg, k=K):
    return [START] * k + body(msg)


def build_vocab(msgs):
    chars = set()
    for m in msgs:
        chars.update(m)
        chars.add(START)
    v = sorted(chars)
    if UNK not in v:
        v.append(UNK)
    return {c: i for i, c in enumerate(v)}


def positions(msgs, vocab, out_pos, k=K):
    """كل (سياق ids بطول k، موضع الهدف في مساحة المخرجات) — كل رسالة باستقلال."""
    X, Y = [], []
    for m in msgs:
        ids = [vocab[START]] * k + [vocab.get(c, vocab[UNK]) for c in body(m)]
        for i in range(k, len(ids)):
            X.append(ids[i - k:i])
            Y.append(out_pos[ids[i]])
    return np.array(X), np.array(Y)


def main():
    raw = SRC.read_bytes()
    input_hash = hashlib.sha256(raw).hexdigest()
    src = json.loads(SRC.read_text(encoding="utf-8"))
    t0 = time.perf_counter()

    results = {"simulator_version": VERSION, "input_hash": input_hash, "languages": {}}

    for lang in ("ar", "en"):
        cfg = src["corpus"][lang]
        train_msgs, val_msgs, test_msgs = cfg["train"], cfg["val"], cfg["test"]
        vocab = build_vocab(train_msgs)
        V_ctx, V_out = len(vocab), len(vocab) - 1  # <s> سياق فقط
        out_ids = [i for c, i in vocab.items() if c != START]

        out_ids = sorted(i for c, i in vocab.items() if c != START)
        out_pos = {full: pos for pos, full in enumerate(out_ids)}
        Xtr, Ytr = positions(train_msgs, vocab, out_pos)
        Xva, Yva = positions(val_msgs, vocab, out_pos)
        Xte, Yte = positions(test_msgs, vocab, out_pos)

        rng = np.random.default_rng(SEED)
        E = rng.normal(0, 0.1, size=(V_ctx, EMB))
        W1 = rng.normal(0, 0.1, size=(K * EMB, HID))
        b1 = np.zeros(HID)
        W2 = rng.normal(0, 0.1, size=(HID, V_out))
        b2 = np.zeros(V_out)

        # تدريب full-batch بسيط بمعدل ثابت — بدون مكتبة تعلم آلي
        curve = []
        for ep in range(1, EPOCHS + 1):
            Xe = E[Xtr].reshape(len(Xtr), K * EMB)
            H = np.tanh(Xe @ W1 + b1)
            Z = H @ W2 + b2
            Z = Z - Z.max(axis=1, keepdims=True)
            P = np.exp(Z)
            P /= P.sum(axis=1, keepdims=True)
            loss = -np.mean(np.log(P[np.arange(len(Ytr)), Ytr] + 1e-12))
            dZ = P.copy()
            dZ[np.arange(len(Ytr)), Ytr] -= 1.0
            dZ /= len(Ytr)
            gW2 = H.T @ dZ
            gb2 = dZ.sum(axis=0)
            dH = dZ @ W2.T * (1 - H ** 2)
            gE = np.zeros_like(E)
            for j in range(K):
                np.add.at(gE, Xtr[:, j], dH[:, j * EMB:(j + 1) * EMB])
            gW1 = Xe.T @ dH
            gb1 = dH.sum(axis=0)
            E -= LR * gE
            W1 -= LR * gW1
            b1 -= LR * gb1
            W2 -= LR * gW2
            b2 -= LR * gb2
            if ep in (1, 10, 50, 100, 200, 300):
                curve.append({"epoch": ep, "train_nll": round(float(loss), 4)})

        n_params_learned = int(E.size + W1.size + b1.size + W2.size + b2.size)

        def nll(X, Y):
            Xe = E[X].reshape(len(X), K * EMB)
            H = np.tanh(Xe @ W1 + b1)
            Z = H @ W2 + b2
            Z = Z - Z.max(axis=1, keepdims=True)
            P = np.exp(Z)
            P /= P.sum(axis=1, keepdims=True)
            return round(float(-np.mean(np.log(P[np.arange(len(Y)), Y] + 1e-12))), 4)

        # تجربة العزل: تضمينات مجمدة (نسخة عشوائية أولية بلا تدريب عليها)
        rng2 = np.random.default_rng(SEED)
        Ef = rng2.normal(0, 0.1, size=(V_ctx, EMB))
        W1f = rng2.normal(0, 0.1, size=(K * EMB, HID))
        b1f = np.zeros(HID)
        W2f = rng2.normal(0, 0.1, size=(HID, V_out))
        b2f = np.zeros(V_out)
        for ep in range(EPOCHS):
            Xe = Ef[Xtr].reshape(len(Xtr), K * EMB)
            H = np.tanh(Xe @ W1f + b1f)
            Z = H @ W2f + b2f
            Z = Z - Z.max(axis=1, keepdims=True)
            P = np.exp(Z)
            P /= P.sum(axis=1, keepdims=True)
            dZ = P.copy()
            dZ[np.arange(len(Ytr)), Ytr] -= 1.0
            dZ /= len(Ytr)
            gW2f = H.T @ dZ
            gb2f = dZ.sum(axis=0)
            dH = dZ @ W2f.T * (1 - H ** 2)
            for j in range(K):
                np.add.at(gE_f := np.zeros_like(Ef), Xtr[:, j], dH[:, j * EMB:(j + 1) * EMB])
            W2f -= LR * gW2f
            b2f -= LR * gb2f
        # تقييم المجمد على التدريب والتحقق
        def frozen_nll(X, Y):
            Xe = Ef[X].reshape(len(X), K * EMB)
            H = np.tanh(Xe @ W1f + b1f)
            Z = H @ W2f + b2f
            Z = Z - Z.max(axis=1, keepdims=True)
            P = np.exp(Z)
            P /= P.sum(axis=1, keepdims=True)
            return round(float(-np.mean(np.log(P[np.arange(len(Y)), Y] + 1e-12))), 4)
        frozen_train_nll = frozen_nll(Xtr, Ytr)
        frozen_val_nll = frozen_nll(Xva, Yva)
        frozen_test_nll = frozen_nll(Xte, Yte)
        trainable_frozen = int(W1f.size + b1f.size + W2f.size + b2f.size)

        # توليد بعينات ثابتة — فكّ فهرس الإخراج عبر out_ids (لا خلط بين الفهرس
        # المضغوط والكامل)، وتمرير السياق بالفهارس الكاملة.
        inv_full = {i: c for c, i in vocab.items()}

        def generate(model_E, model_W1, model_b1, model_W2, model_b2, prompt_chars, max_chars=60):
            rng = np.random.default_rng(SEED)
            ctx = [vocab[START]] * K + [vocab.get(c, vocab[UNK]) for c in prompt_chars]
            out = []
            for _ in range(max_chars):
                xc = np.array(ctx[-K:])
                Xe = model_E[xc].reshape(1, K * EMB)
                Hh = np.tanh(Xe @ model_W1 + model_b1)
                Z = Hh @ model_W2 + model_b2
                Z = Z - Z.max()
                P = np.exp(Z).ravel()
                P /= P.sum()
                nxt = int(rng.choice(V_out, p=P))
                full = out_ids[nxt]
                ch = inv_full[full]
                out.append("?" if ch == UNK else ch)
                ctx = ctx[1:] + [full]
            return "".join(out)

        gen_learned = generate(E, W1, b1, W2, b2, "أو" if lang == "ar" else "I'")
        gen_frozen = generate(Ef, W1f, b1f, W2f, b2f, "أو" if lang == "ar" else "I'")

        # تقسيم خسارة التحقق: سياقات مرصودة/غير مرصودة في التدريب
        train_ctx = set(tuple(x) for x in Xtr)
        seen_mask = np.array([tuple(x) in train_ctx for x in Xva])
        val_seen_nll = nll(Xva[seen_mask], Yva[seen_mask]) if seen_mask.any() else None
        val_unseen_nll = nll(Xva[~seen_mask], Yva[~seen_mask]) if (~seen_mask).any() else None

        results["languages"][lang] = {
            "units": {"train": len(Xtr), "val": len(Xva), "test": len(Xte)},
            "vocab": {"context": V_ctx, "output": V_out},
            "architecture": {"embedding_dim": EMB, "hidden": HID, "activation": "tanh",
                             "context_k": K, "output_excludes_start": True},
            "params": {"total_declared": n_params_learned,
                       "embedding": int(E.size), "hidden_layer": int(W1.size + b1.size),
                       "output_layer": int(W2.size + b2.size)},
            "training_curve": curve,
            "losses": {"train_final": curve[-1]["train_nll"] if curve else None,
                       "val_nll": nll(Xva, Yva), "test_nll": nll(Xte, Yte)},
            "val_seen_unseen_split": {"seen_nll": val_seen_nll, "unseen_nll": val_unseen_nll,
                                      "seen_count": int(seen_mask.sum()),
                                      "unseen_count": int((~seen_mask).sum())},
            "frozen_embedding_ablation": {"trainable": False,
                                          "train_nll": frozen_train_nll,
                                          "val_nll": frozen_val_nll,
                                          "test_nll": frozen_test_nll,
                                          "trainable_params": trainable_frozen,
                                          "note_ar": "التضمينات عشوائية ثابتة بلا تحسين",
                                          "note_en": "frozen random embeddings, never optimised"},
            "generation": {"prompt_chars": "أو" if lang == "ar" else "I'",
                           "learned": gen_learned, "frozen": gen_frozen,
                           "seed": SEED, "decoding": "sampling"},
        }
        elapsed = time.perf_counter() - t0
        results["languages"][lang]["timing_seconds"] = round(elapsed, 2)

        print(f"===== {lang} ===== params={n_params_learned}")
        print("  curve:", curve[-3:])
        print("  val:", nll(Xva, Yva), "test:", nll(Xte, Yte))
        print("  val seen/unseen:", val_seen_nll, "/", val_unseen_nll)
        print("  gen learned:", gen_learned[:50], "| frozen:", gen_frozen[:50])

    elapsed_total = time.perf_counter() - t0
    results["total_seconds"] = round(elapsed_total, 2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print("written:", OUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
