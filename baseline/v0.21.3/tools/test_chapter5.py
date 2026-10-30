#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_chapter5.py — انحدار أرقام الفصل الخامس المنشورة.

يغطي: عدد المعلمات (2,110/2,028) وفحص حساب المعلمات القابلة للتدريب في
النسخة المجمدة (1,798/1,732 = الكلي − حجم التضمين)، منحنى التدريب، خسائر
التحقق والاختبار، تقسيم المرصود/غير المرصود بإعادة بنائه **مستقلاً من المتن**،
قيم النسخة المجمدة، وخلو التوليد من رمز البداية.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SIM = ROOT / "tools" / "simulate_chapter5.py"
SRC = ROOT / "content" / "chapter-03" / "examples.json"
RESULTS = ROOT / "content" / "chapter-05" / "results.json"

START, UNK, K, EMB = "<s>", "<unk>", 2, 8

PUBLISHED = {
    "ar": {
        "params_total": 2110, "frozen_trainable": 1798,
        "curve_tail": {"200": 2.2843, "300": 1.9891},
        "val_nll": 2.5579, "test_nll": 2.7634,
        "seen_nll": 2.2335, "unseen_nll": 3.6251,
        "frozen_train_nll": 3.025, "frozen_val_nll": 3.0632, "frozen_test_nll": 3.1371,
    },
    "en": {
        "params_total": 2028, "frozen_trainable": 1732,
        "curve_tail": {"200": 2.2288, "300": 1.9084},
        "val_nll": 2.1609, "test_nll": 2.1759,
        "seen_nll": 2.0859, "unseen_nll": 2.9005,
        "frozen_train_nll": 2.961, "frozen_val_nll": 2.8913, "frozen_test_nll": 2.9648,
    },
}


def build_vocab(msgs):
    chars = set()
    for m in msgs:
        chars.update(m)
        chars.add(START)
    v = sorted(chars)
    if UNK not in v:
        v.append(UNK)
    return {c: i for i, c in enumerate(v)}


def contexts(msgs, vocab):
    """كل سياق تحقق (آخر معرفين) لكل رسالة — نفس اصطلاح المحاكي."""
    out = []
    for m in msgs:
        ids = [vocab[START]] * K + [vocab.get(c, vocab[UNK]) for c in m]
        for i in range(K, len(ids)):
            out.append(tuple(ids[i - K:i]))
    return out


def independent_split(lang):
    """إعادة بناء تقسيم تحقق مرصود/غير مرصود من المتن مباشرة (بلا نتائج المحاكي)."""
    src = json.loads(SRC.read_text(encoding="utf-8"))
    cfg = src["corpus"][lang]
    vocab = build_vocab(cfg["train"])
    train_ctx = set(contexts(cfg["train"], vocab))
    val_ctx = contexts(cfg["val"], vocab)
    seen = sum(1 for c in val_ctx if c in train_ctx)
    return seen, len(val_ctx) - seen


def main():
    proc = subprocess.run([sys.executable, str(SIM)], capture_output=True, text=True, encoding="utf-8")
    if proc.returncode != 0:
        print("فشل المحاكي:", (proc.stderr or "")[-400:])
        return 1
    r = json.loads(RESULTS.read_text(encoding="utf-8"))
    failures = []
    n = 0

    def check(label, ok):
        nonlocal n
        n += 1
        print(f'{"PASS" if ok else "FAIL":5} {label}')
        if not ok:
            failures.append(label)

    for lang, exp in PUBLISHED.items():
        L = r["languages"][lang]
        check(f"{lang}.params_total = {L['params']['total_declared']}",
              L["params"]["total_declared"] == exp["params_total"])
        # حساب المعلمات القابلة للتدريب في النسخة المجمدة: الكلي − حجم جدول التضمين
        emb = L["vocab"]["context"] * EMB
        check(f"{lang}.frozen_trainable = {L['frozen_embedding_ablation']['trainable_params']}"
              f" (= total − {emb})",
              L["frozen_embedding_ablation"]["trainable_params"] == exp["frozen_trainable"]
              and exp["params_total"] - exp["frozen_trainable"] == emb)
        for ep, nll in exp["curve_tail"].items():
            actual = [c["train_nll"] for c in L["training_curve"] if c["epoch"] == int(ep)][0]
            check(f"{lang}.curve[{ep}] = {actual}", round(actual, 3) == round(nll, 3))
        for key in ("val_nll", "test_nll"):
            check(f"{lang}.{key} = {L['losses'][key]}", L["losses"][key] == exp[key])
        fe = L["frozen_embedding_ablation"]
        for key in ("frozen_train_nll", "frozen_val_nll", "frozen_test_nll"):
            actual = fe[key.replace("frozen_", "")] if key != "frozen_train_nll" else fe["train_nll"]
            check(f"{lang}.{key} = {actual}", round(actual, 4) == round(exp[key], 4))
        # تقسيم مرصود/غير مرصود: إعادة بناء مستقلة من المتن ومطابقة العدادات
        split = L["val_seen_unseen_split"]
        seen_ind, unseen_ind = independent_split(lang)
        check(f"{lang}.split_counts مستقلة = {seen_ind}/{unseen_ind}"
              f" (المحاكي: {split['seen_count']}/{split['unseen_count']})",
              split["seen_count"] == seen_ind and split["unseen_count"] == unseen_ind
              and seen_ind + unseen_ind == len(contexts(
                  json.loads(SRC.read_text(encoding="utf-8"))["corpus"][lang]["val"],
                  build_vocab(json.loads(SRC.read_text(encoding="utf-8"))["corpus"][lang]["train"])))
              and seen_ind > 0 and unseen_ind > 0)
        check(f"{lang}.split_nll = {split['seen_nll']}/{split['unseen_nll']}",
              split["seen_nll"] == exp["seen_nll"] and split["unseen_nll"] == exp["unseen_nll"]
              and split["unseen_nll"] > split["seen_nll"])
        gen = L["generation"]
        check(f"{lang}.generation بلا رمز بداية",
              "<s>" not in gen["learned"] and "<s>" not in gen["frozen"])

    print()
    if failures:
        print("انحرافات:", failures)
        return 1
    print(f"كل أرقام الفصل الخامس المنشورة تطابق مخرجات المحاكي ({n} فحصاً).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
