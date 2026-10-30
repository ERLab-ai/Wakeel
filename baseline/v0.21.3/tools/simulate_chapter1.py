#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
simulate_chapter1.py — يحسب نتائج تجربة الفصل الأول من بياناتها.

كل رقم يُذكر في نص الفصل يجب أن يخرج من هنا. لا نتيجة مكتوبة بيد في النص
إلا وهي مخرجة من هذه الأداة على `content/chapter-01/examples.json`.

طريقتان:
  1) قواعد كلمات مفتاحية: أول قاعدة تحتوي إحدى كلماتها داخل النص.
  2) التعلم من أمثلة: أقرب مثال سابق بتشابه الكلمات (Jaccard)، ويُعاد ردّه.

الاستخدام:
    python tools/simulate_chapter1.py
    python tools/simulate_chapter1.py --lang ar
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "content" / "chapter-01" / "examples.json"
OUT = ROOT / "content" / "chapter-01" / "results.json"


ARABIC_PUNCT = "؟،؛٪ـ«»”“…"
ASCII_PUNCT = ".,!?:;()[]{}\"'`-–—/\\"


def tokens(text):
    text = text.casefold().replace("'", "").replace("\u2019", "")
    for ch in ARABIC_PUNCT + ASCII_PUNCT:
        text = text.replace(ch, " ")
    return [t for t in text.split() if t]


def rule_predict(text, rules):
    low = text.casefold()
    for r in rules:
        for kw in r["keywords"]:
            if kw.casefold() in low:
                return r["id"], r["intent"], kw
    return None, None, None


def nearest(text, examples):
    a = set(tokens(text))
    best = None
    for ex in examples:
        b = set(tokens(ex["text"]))
        inter, union = len(a & b), len(a | b)
        score = inter / union if union else 0.0
        if best is None or score > best[1]:
            best = (ex, score, sorted(a & b))
    return best


def run_lang(lang, data, intents, fallback):
    rules = data["rules"]
    seed = data["seed"]
    tests = data["tests"]
    examples = data["examples"]

    def rule_rows(items):
        rows = []
        for m in items:
            rid, intent, kw = rule_predict(m["text"], rules)
            rows.append({
                "id": m["id"], "text": m["text"], "intent": m["intent"],
                "matched_rule": rid, "matched_keyword": kw,
                "predicted": intent,
                "reply": intents[intent]["response_" + lang] if intent else fallback[lang],
                "correct": intent == m["intent"],
            })
        return rows

    def example_rows(items, pool=None):
        pool = pool if pool is not None else examples
        rows = []
        for m in items:
            ex, score, shared = nearest(m["text"], pool)
            rows.append({
                "id": m["id"], "text": m["text"], "intent": m["intent"],
                "nearest_example": ex["id"], "nearest_text": ex["text"],
                "similarity": round(score, 3), "shared_words": shared,
                "predicted": ex["intent"],
                "reply": intents[ex["intent"]]["response_" + lang],
                "correct": ex["intent"] == m["intent"],
            })
        return rows

    out = {}
    for name, items in (("seed", seed), ("tests", tests)):
        rr = rule_rows(items)
        er = example_rows(items)
        out[name] = {
            "total": len(items),
            "rule_correct": sum(r["correct"] for r in rr),
            "example_correct": sum(r["correct"] for r in er),
            "rule_rows": rr,
            "example_rows": er,
            "rule_matched_but_wrong": [r["id"] for r in rr if r["matched_rule"] and not r["correct"]],
            "rule_fallback_wrong": [r["id"] for r in rr if not r["matched_rule"] and not r["correct"]],
        }

    # اختبارات هدم: ماذا يحدث لو كانت الأمثلة ناقصة أو غير متوازنة؟
    seen, two_per = set(), []
    for ex in examples:
        if ex["intent"] not in seen:
            seen.add(ex["intent"])
        two_per.append(ex)
    two_per = [ex for i, ex in enumerate(two_per) if i < 8][:8]
    ablations = {
        "drop_confirm": [ex for ex in examples if ex["intent"] != "R4"],
        "first_eight": examples[:8],
        "no_apology": [ex for ex in examples if ex["intent"] != "R2"],
    }
    out["ablations"] = {}
    for name, pool in ablations.items():
        rows = example_rows(tests, pool)
        out["ablations"][name] = {
            "pool_size": len(pool),
            "correct": sum(r["correct"] for r in rows),
            "total": len(rows),
            "wrong": [r["id"] for r in rows if not r["correct"]],
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["ar", "en"], help="لغة واحدة فقط")
    a = ap.parse_args()

    src = json.loads(SRC.read_text(encoding="utf-8"))
    intents = {i["id"]: i for i in src["intents"]}
    langs = [a.lang] if a.lang else ["ar", "en"]
    results = {"generated_from": str(SRC.relative_to(ROOT)).replace("\\", "/"), "datasets": {}}

    for lang in langs:
        res = run_lang(lang, src["datasets"][lang], intents, src["fallback"])
        results["datasets"][lang] = res
        print(f"\n===== {lang.upper()} =====")
        for group in ("seed", "tests"):
            g = res[group]
            print(f"{group}: {g['rule_correct']}/{g['total']} قواعد | {g['example_correct']}/{g['total']} أمثلة")
            print("  id | intent | rule | ex | rule-ok ex-ok")
            for rr, er in zip(g["rule_rows"], g["example_rows"]):
                match = rr["matched_rule"] or f"—({rr['matched_keyword']})"
                print(f'  {rr["id"]:3} | {rr["intent"]} | {match:7} | {er["nearest_example"]:3} | '
                      f'{"Y" if rr["correct"] else "n"}      {"Y" if er["correct"] else "n"}')

    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("\nwritten:", OUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
