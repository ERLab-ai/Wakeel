#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
simulate_chapter2.py — يحسب كل رقم في الفصل الثاني من بياناته.

قاعدة التوحيد (بعد مراجعة v0.5): المقطّع نفسه (edu-tok-0.1) يُستخدم في
التجارب وفي بناء المفردة وترميز الاختبارات. لا يتبدل المقطّع بين قسمين
بصمت.

الاستخدام:
    python tools/simulate_chapter2.py            # اللغتان
    python tools/simulate_chapter2.py --lang ar  # لغة واحدة
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "content" / "chapter-02" / "examples.json"
OUT = ROOT / "content" / "chapter-02" / "results.json"

AR_PUNCT = "؟،؛٪ـ«»”“…"
ASCII_PUNCT = ".,!?:;()[]{}\"'`—–/\\"
UNK = "<unk>"


def ws_keep(text):
    """تقسيم على المسافات فقط؛ الترقيم يبقى ملتحقاً."""
    return [w for w in text.split() if w]


def strip_punct(text, lower=False):
    if lower:
        text = text.casefold()
    for ch in AR_PUNCT + ASCII_PUNCT:
        text = text.replace(ch, " ")
    return [w for w in text.split() if w]


def edu_tokenize(text, lower=False):
    """edu-tok-0.1: مسافات + قاعدتا سابقات عربيتان. التشكيل يبقى ملتحقاً بكلمته (لا نطبّعه)."""
    out = []
    for w in strip_punct(text, lower=lower):
        if w.startswith("و") and len(w[1:]) >= 4:
            out.extend(["و", w[1:]])
        elif w.startswith("ال") and len(w[2:]) >= 2:
            out.extend(["ال", w[2:]])
        else:
            out.append(w)
    return out


def compose(template, slots):
    missing = [k for k, v in slots.items() if v in (None, "")]
    if missing:
        return None, missing
    text = template
    for k, v in slots.items():
        text = text.replace("{" + k + "}", v)
    return text, []


def unit_counts(sentence, lang):
    ws = ws_keep(sentence)
    chars = [c for c in sentence if not c.isspace()]
    if lang == "en":
        units = []
        for w in ws:
            parts = w.replace("\u2019", "'").split("'")
            units.extend(p for p in parts if p)
        return {"ws_words": len(ws), "characters_incl_punct": len(chars),
                "apostrophe_split_units": len(units), "tokens_ws": ws, "tokens_units": units}
    edu = edu_tokenize(sentence)
    return {"ws_words": len(ws), "characters_incl_punct": len(chars), "edu_units": len(edu),
            "tokens_ws": ws, "tokens_edu": edu}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["ar", "en"])
    a = ap.parse_args()
    src = json.loads(SRC.read_text(encoding="utf-8"))
    langs = [a.lang] if a.lang else ["ar", "en"]

    results = {"generated_from": str(SRC.relative_to(ROOT)).replace("\\", "/"),
               "pinned_tokenizer": src["tokenization"]["pinned_version"],
               "normalization_ar": "لا نزيل التشكيل ولا نوحّده؛ الحرف المشكول يبقى ضمن كلمته",
               "sections": {}}

    # 1) القالب
    for lang in langs:
        tpl = src["template"][lang]["text"]
        rows = {}
        for tid, slots in src["fills"][lang].items():
            text, missing = compose(tpl, slots)
            rows[tid] = {"slots": slots, "composed": text,
                         "status": "composed" if text else f"فشل: لا قيمة لـ {missing}"}
        results["sections"]["template"] = results["sections"].get("template", {})
        results["sections"]["template"][lang] = rows

    # 2) عدّ الوحدات
    for lang in langs:
        s = src["unit_counts"][lang]["sentence"]
        results["sections"]["units"] = results["sections"].get("units", {})
        results["sections"]["units"][lang] = unit_counts(s, lang)

    # 3) تجربتا التقطيع: (أ) النص نفسه بطريقتين (ب) نصان متقاربان بمقطّع مثبت
    for lang in langs:
        t = src["tokenization"]["texts"][lang]
        ws = ws_keep(t)
        edu = edu_tokenize(t)
        results["sections"]["tokenization"] = results["sections"].get("tokenization", {})
        results["sections"]["tokenization"][lang] = {
            "text": t,
            "ws_keep_count": len(ws), "ws_keep_tokens": ws,
            "edu_count": len(edu), "edu_tokens": edu,
        }

    # 4) المفردة: مبنية بمقطّع edu-tok-0.1 نفسه، و<unk> يحمل المعرّف 0
    for lang in langs:
        cfg = src["vocab"]
        corpus_key = "corpus" if lang == "ar" else "corpus_en"
        tests_key = "tests_ar" if lang == "ar" else "tests_en"
        first_seen, freq = {}, Counter()
        order = []
        for text in cfg[corpus_key]:
            for w in edu_tokenize(text, lower=(lang == "en")):
                if w not in first_seen:
                    first_seen[w] = len(first_seen)
                    order.append(w)
                freq[w] += 1
        ordered = sorted(freq, key=lambda w: (-freq[w], first_seen[w]))
        ids = {UNK: 0}
        ids.update({w: i + 1 for i, w in enumerate(ordered)})

        cov = {}
        for size in cfg["sizes"]:
            vocab = set(ordered[:size])
            total = known = 0
            unknown = []
            for text in cfg[tests_key]:
                for w in edu_tokenize(text, lower=(lang == "en")):
                    total += 1
                    if w in vocab:
                        known += 1
                    else:
                        unknown.append(w)
            cov[f"vocab_{size}"] = {"vocab_size": size, "unk_id": 0,
                                    "total_tokens": total, "covered": known,
                                    "coverage": round(known / total, 3),
                                    "unknown_sample": sorted(set(unknown))[:12]}

        t1 = cfg[tests_key][0]
        vocab12 = set(ordered[:12])
        t1_toks = edu_tokenize(t1, lower=(lang == "en"))
        enc = [{"token": w, "id": ids[w] if w in vocab12 else 0,
                "status": "known" if w in vocab12 else UNK} for w in t1_toks]

        results["sections"]["vocab"] = results["sections"].get("vocab", {})
        results["sections"]["vocab"][lang] = {
            "built_with": "edu-tok-0.1", "unk_id": 0,
            "top12_with_ids": [{"token": w, "id": ids[w]} for w in ordered[:12]],
            "t1_text": t1, "t1_tokens": t1_toks, "t1_encoding": enc,
            "coverage": cov,
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")

    for lang in langs:
        print(f"===== {lang} =====")
        for tid, r in results["sections"]["template"][lang].items():
            print(f"  قالب {tid}: {r['status']}")
        u = results["sections"]["units"][lang]
        print("  units:", {k: v for k, v in u.items() if not k.startswith("tokens_")})
        tk = results["sections"]["tokenization"][lang]
        print(f"  tokenization: ws_keep={tk['ws_keep_count']} edu={tk['edu_count']}")
        v = results["sections"]["vocab"][lang]
        print("  top12:", [(x["token"], x["id"]) for x in v["top12_with_ids"]])
        print("  coverage:", {k: x["coverage"] for k, x in v["coverage"].items()})
        print("  T1:", [(e["token"], e["id"]) for e in v["t1_encoding"]])
    print("written:", OUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
