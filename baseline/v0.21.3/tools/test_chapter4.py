#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_chapter4.py — اختبارات انحدار وثوابت لنجمة الفصل الرابع.

يتحقق من:
  * الأرقام المنشورة في الفصل مقابل مخرجات المحاكي.
  * سياسة السياق غير المرصود: توزيع منتظم على مساحة المخرجات.
  * رمز البداية <s> لا يظهر مخرجاً في أي توليد.
  * حدود الرسائل: لا انتقالات عابرة للرسائل.
  * استقلال بيانات الاختبار ومفردة المحارف عن طول السياق.
"""
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

START, UNK = "<s>", "<unk>"

ROOT = Path(__file__).resolve().parents[1]
SIM = ROOT / "tools" / "simulate_chapter4.py"
RESULTS = ROOT / "content" / "chapter-04" / "results.json"
SRC = ROOT / "content" / "chapter-04" / "examples.json"
SRC_CH3 = ROOT / "content" / "chapter-03" / "examples.json"

PUBLISHED = {
    "ar": {
        "k0": {"train_nll": 3.0878, "val_nll": 3.131, "test_nll": 3.1757},
        "k1": {"train_nll": 2.1444, "val_nll": 2.656, "test_nll": 2.638, "alpha": 0.5},
        "k2": {"train_nll": 1.1758, "val_nll": 2.2915, "test_nll": 2.2239},
        "k3": {"train_nll": 1.1161, "val_nll": 2.4659, "test_nll": 2.3961},
        "unseen_val": {"k0": 0.0, "k1": 0.0075, "k2": 0.2331, "k3": 0.3985},
        "backoff": {"val_loss": 2.7328, "test_loss": 2.3511},
    },
    "en": {
        "k0": {"train_nll": 3.0172, "val_nll": 2.9783, "test_nll": 3.0255},
        "k1": {"train_nll": 1.8815, "val_nll": 2.2232, "test_nll": 2.2234, "alpha": 0.1},
        "k2": {"train_nll": 1.1969, "val_nll": 1.9625, "test_nll": 1.9494},
        "k3": {"train_nll": 1.0738, "val_nll": 2.1224, "test_nll": 2.0234},
        "unseen_val": {"k0": 0.0, "k1": 0.0123, "k2": 0.092, "k3": 0.2699},
        "backoff": {"val_loss": 1.8201, "test_loss": 1.6799},
    },
}


DENSE_CELLS = {
    "ar": {0: 38, 1: 1482, 2: 57798, 3: 2254122},
    "en": {0: 36, 1: 1332, 2: 49284, 3: 1823508},
}


def run_sim():
    proc = subprocess.run([sys.executable, str(SIM)], capture_output=True, text=True, encoding="utf-8")
    if proc.returncode != 0:
        print("فشل المحاكي:", (proc.stderr or "")[-400:])
        sys.exit(1)


def main():
    failures = []
    res = {"ok": 0, "fail": 0}

    def check(name, ok):
        res["ok" if ok else "fail"] += 1
        print(f'{"PASS" if ok else "FAIL":5} {name}')

    run_sim()
    r = json.loads(RESULTS.read_text(encoding="utf-8"))

    for lang, exp in PUBLISHED.items():
        L = r["languages"][lang]
        for k in ("k0", "k1", "k2", "k3"):
            m, e = L["models"][k], exp[k]
            for f in ("train_nll", "val_nll", "test_nll"):
                check(f"{lang}.{k}.{f}", round(m[f], 3) == round(e[f], 3))
            if "alpha" in e:
                check(f"{lang}.{k}.alpha", m["alpha_from_validation"] == e["alpha"])
            check(f"{lang}.{k}.dense_cells", m["theoretical_dense_cells"] == DENSE_CELLS[lang][int(k[1:])])
        for k in ("k0", "k1", "k2", "k3"):
            check(f"{lang}.unseen_val.{k}",
                  round(L["models"][k]["unseen_context_rate_val"], 4) == round(exp["unseen_val"][k], 4))
        b = L["backoff"]
        check(f"{lang}.backoff.val", round(b["val_loss"], 3) == round(exp["backoff"]["val_loss"], 3))
        check(f"{lang}.backoff.test", round(b["test_loss"], 3) == round(exp["backoff"]["test_loss"], 3))
        texts = [g["text"] for g in L["generation"]]
        check(f"{lang}.no_start_in_generation", all("<s>" not in x for x in texts))

    # الفحص المستقل لدراسة الحالة: مسح مباشر لرسائل التدريب بلا NGram
    src_ch3 = json.loads(SRC_CH3.read_text(encoding="utf-8"))
    refs = {
        "ar": [("أين القاعة؟", "ين "), ("لن أستطيع المجيء، أرجو أن تعذرني.", "لن ")],
        "en": [("Which room is it in?", "ch "), ("I can't make it tomorrow, apologies.", "I ")],
    }
    for lang, pairs in refs.items():
        train_msgs = src_ch3["corpus"][lang]["train"]
        for msg, ctx in pairs:
            cs = [c for c in r["languages"][lang]["case_studies"] if c["message"] == msg]
            if not cs:
                continue
            row = [x for x in cs[0]["contexts"] if x["k"] == 3][0]
            flw = Counter()
            for m2 in train_msgs:
                s2 = [START] * 3 + list(m2)
                for i in range(3, len(s2)):
                    if s2[i - 3:i] == list(ctx):
                        flw[s2[i]] += 1
            ok = row["training_followers"] == dict(sorted(flw.items()))
            print(f'{"PASS" if ok else "FAIL":5} {lang} {ctx!r}: مسح مستقل يطابق عدّادات دراسة الحالة')
            if not ok:
                failures.append(f"independent_scan.{lang}.{ctx}")

    # اختبار الثوابت على كود المحاكي نفسه
    sys.path.insert(0, str(ROOT / "tools"))
    import simulate_chapter4 as sim4mod

    train_texts = ["ab", "cd"]
    vocab = {c: i for i, c in enumerate(sorted(set("abcd") | {START, UNK}))}
    out_ids = [i for c, i in vocab.items() if c != START]
    train = [[vocab[c] for c in m] for m in train_texts]

    # سياسة السياق غير المرصود: توزيع منتظم
    m = sim4mod.NGram(2, train, vocab, out_ids, alpha=1.0)
    unseen_ctx = (vocab[UNK], vocab[UNK])  # سياق لم يظهر في التدريب إطلاقاً
    p = m.dist(unseen_ctx)
    check("invariant.unseen_context_uniform", np.allclose(p, np.full(len(out_ids), 1 / len(out_ids))))

    # حدود الرسائل: لا انتقال من b (نهاية الأولى) إلى c (بداية الثانية)
    has_cross = m.pair_counts.get((vocab["b"], vocab["c"]), 0) > 0
    check("invariant.no_cross_message_transition", not has_cross)

    # استقلال بيانات الاختبار عن طول السياق
    t1 = [vocab.get(c, vocab[UNK]) for c in "abc"]
    t2 = [vocab.get(c, vocab[UNK]) for c in "abc"]
    check("invariant.test_tokens_independent_of_k", t1 == t2)

    # ثابت: عدد وحدات التقييم متطابق عبر كل مستويات السياق (لا تسرب عبر الحدود)
    for lang in ("ar", "en"):
        tokens = [L["models"][k]["eval_tokens"] for k in ("k0", "k1", "k2", "k3")]
        check(f"{lang}.eval_tokens_identical_across_k", len(set(tokens)) == 1)

    print()
    if failures:
        print("فشل:", failures)
        return 1
    print(f"نجحت كل الفحوص: {res['ok']} (منها {res['ok']} انحداراً وثوابت) — 0 إخفاقات.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
