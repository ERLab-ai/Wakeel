#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_chapter3.py — اختبارات انحدار لأرقام الفصل الثالث المنشورة.

1) يعيد تشغيل المحاكي ويقارن مخرجاته بالأرقام المطبوعة في الفصل.
2) يتحقق من المثال اليدوي (11 انتقالاً واحتمالاته) من الكود.
3) يتحقق من أن النصوص المنشورة تتضمن الأرقام الحاسمة.

أي انحراف يفشل الاختبار — فلا يتغير رقم في البيانات ويبقى في النص.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "content" / "chapter-03" / "results.json"
SIM = ROOT / "tools" / "simulate_chapter3.py"
AR = ROOT / "content" / "chapter-03" / "ar.md"
EN = ROOT / "content" / "chapter-03" / "en.md"

PUBLISHED = {
    "ar": {
        "units": {"train": 712, "val": 133, "test": 245},
        "unigram": {"val_loss": 3.131, "test_loss": 3.176},
        "count": {"chosen_alpha": 0.5, "val_loss": 2.656, "test_loss": 2.638},
        "neural": {"val_loss": 2.927, "test_loss": 3.062},
        "char_check": {"unknown_chars": 0, "total_chars": 245},
        "generation": "أودمالغء?كن ةةىصماجت",
        "greedy_prefix": "أودأس",
    },
    "en": {
        "units": {"train": 729, "val": 163, "test": 280},
        "unigram": {"val_loss": 2.978, "test_loss": 3.026},
        "count": {"chosen_alpha": 0.1, "val_loss": 2.223, "test_loss": 2.223},
        "neural": {"val_loss": 2.788, "test_loss": 2.909},
        "char_check": {"unknown_chars": 1, "total_chars": 280},
        "generation": "I'dWhos win's'tirome tindvvolf bhithee an",
        "greedy_prefix": "I'dI the the",
    },
}

WORKED = {
    "transitions": 11,
    "mini_vocab_size": 7,
    "output_vocab_size": 6,
    "probabilities": {"<s>→س": 0.286, "س→أ": 0.375, "أ→ح": 0.375,
                      "ح→ض": 0.375, "ض→ر": 0.375, "ر→ ": 0.286, " →س": 0.286},
}

PROSE_MUST_CONTAIN = {
    "ar": ["11 انتقالاً", "2.656", "2.638", "3.131", "3.176", "2.927", "3.062",
           "0.098", "0.358", "0.299", "0.041", "0.020", "أودأسأس", "0.286", "0.375"],
    "en": ["5 transitions", "2.223", "2.978", "3.026", "2.789", "2.909",
           "0.229", "0.635", "0.118", "0.060", "0.370", "I'dI the the", "0.286", "0.375"],
}

# أرقام الفصل الثاني (القسم الرابع بعد توحيد المقطّع) تُفحص في ملفه
PROSE_CH2_MUST_CONTAIN = {
    "ar": ["37.9", "43.1", "48.3", "اجتماع", "اللقاء", "edu-tok-0.1"],
    "en": ["32.3", "40.3", "50", "edu-tok-0.1"],
}


def norm(x):
    if isinstance(x, float):
        return round(x, 3)
    if isinstance(x, dict):
        return {k: norm(v) for k, v in x.items()}
    return x


def main():
    proc = subprocess.run([sys.executable, str(SIM)], capture_output=True, text=True, encoding="utf-8")
    if proc.returncode != 0:
        print("فشل المحاكي نفسه:", (proc.stderr or "")[-500:])
        return 1
    r = json.loads(RESULTS.read_text(encoding="utf-8"))
    failures = []

    for lang, exp in PUBLISHED.items():
        L = r["languages"][lang]
        checks = [
            ("units", L["units_actual"], exp["units"]),
            ("unigram", L["models"]["unigram"], exp["unigram"]),
            ("count.alpha", L["models"]["count_bigram"]["chosen_alpha"], exp["count"]["chosen_alpha"]),
            ("count.val", L["models"]["count_bigram"]["val_loss"], exp["count"]["val_loss"]),
            ("count.test", L["models"]["count_bigram"]["test_loss"], exp["count"]["test_loss"]),
            ("neural.val", L["models"]["neural_bigram"]["val_loss"], exp["neural"]["val_loss"]),
            ("neural.test", L["models"]["neural_bigram"]["test_loss"], exp["neural"]["test_loss"]),
            ("char.unknown", L["char_level_check"]["unknown_chars"], exp["char_check"]["unknown_chars"]),
            ("char.total", L["char_level_check"]["total_chars"], exp["char_check"]["total_chars"]),
        ]
        for name, actual, expected in checks:
            ok = norm(actual) == norm(expected)
            print(f'{"PASS" if ok else "FAIL":5} {lang}.{name}')
            if not ok:
                failures.append(f"{lang}.{name}: {actual!r} != {expected!r}")
        g = L["generation"]
        ok = g["count_text"] == exp["generation"]
        print(f'{"PASS" if ok else "FAIL":5} {lang}.generation.sample')
        if not ok:
            failures.append(f"{lang}.generation")
        ok = g["greedy_count_text"].startswith(exp["greedy_prefix"])
        print(f'{"PASS" if ok else "FAIL":5} {lang}.generation.greedy_prefix')
        if not ok:
            failures.append(f"{lang}.greedy_prefix: {g['greedy_count_text'][:30]!r}")

    w = r["worked_example"]
    for key, expected in WORKED.items():
        ok = norm(w[key]) == norm(expected)
        print(f'{"PASS" if ok else "FAIL":5} worked.{key}')
        if not ok:
            failures.append(f"worked.{key}: {w[key]!r}")
    ok = w["start_symbol_generated"] is False
    print(f'{"PASS" if ok else "FAIL":5} worked.start_symbol_generated')
    if not ok:
        failures.append("worked.start_symbol_generated")

    for lang, path in (("ar", AR), ("en", EN)):
        text = path.read_text(encoding="utf-8")
        missing = [s for s in PROSE_MUST_CONTAIN[lang] if s not in text]
        ok = not missing
        print(f'{"PASS" if ok else "FAIL":5} {lang}.prose يحوي الأرقام الحاسمة')
        if not ok:
            failures.append(f"{lang}.prose missing: {missing}")
    for lang in ("ar", "en"):
        ch2 = (ROOT / "content" / "chapter-02" / f"{lang}.md").read_text(encoding="utf-8")
        missing = [s for s in PROSE_CH2_MUST_CONTAIN[lang] if s not in ch2]
        ok = not missing
        print(f'{"PASS" if ok else "FAIL":5} chapter-02.{lang}.prose')
        if not ok:
            failures.append(f"chapter-02.{lang}.prose missing: {missing}")

    print()
    if failures:
        print("انحراف بين النص المنشور والبيانات:", failures)
        return 1
    print("كل أرقام الفصل الثالث المنشورة تطابق مخرجات المحاكي، والمثال اليدوي صحيح (11 انتقالاً).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
