#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_chapter6.py — انحدار أرقام الفصل السادس المنشورة (v3.0).

أولاً: اختبارات وحدة مبكرة بلا محاكاة —
  (أ) تطابق موضع التقييم مع موضع التدريب: predict_next_logits(البادئة حتى ؟)
      == forward(البادئة + الجواب)[-1]، عند أطوال بادئات تغطي delay=0/2/4/8/16؛
  (ب) الفحص المضاد: لوغيتات موضع الحشو السابق (خلل v2.1) **لا** تساويها؛
  (ج) المرجع Oracle عبر مسار التقييم نفسه = 1.000، وتبديل التسميات = 0.000.
ثم: تشغيل المحاكي والتحقق من الثوابت المنشورة — بصمة المدخل محسوبة مستقلة،
المفردة من التدريب فقط، معادلة المعلمات، نظافة أبجدية الحشو، أرقام المهمة
عند التأخيرات السبعة (دقة لكل بذرة، NLL اختبار منفصل عن منحنى التدريب،
عدّاد رفض التداخلات)، وفحوص السلامة الصارمة (خروج المحاكي برمز غير صفري عند الإخفاق).
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SIM = ROOT / "tools" / "simulate_chapter6.py"
SRC = ROOT / "content" / "chapter-03" / "examples.json"
RESULTS = ROOT / "content" / "chapter-06" / "results.json"

PUBLISHED = {
    "ar": {"params": 1358, "v_ctx": 39, "v_out": 38, "unseen": 1,
           "mean_val": 3.1235, "mean_test": 3.1899,
           "seed42_val": 3.1182, "seed42_test": 3.1877},
    "en": {"params": 1308, "v_ctx": 37, "v_out": 36, "unseen": 3,
           "mean_val": 2.9291, "mean_test": 2.9962,
           "seed42_val": 2.923, "seed42_test": 2.9957},
}
# delay: (binary, argmax, A, B, per_seed_binary, nll_mean, nll_std, per_seed_nll,
#         untr_bin, untr_argmax, untr_nll, rejected, train_curve_last)
MEM = {
    2: (1.0, 1.0, 1.0, 1.0, [1.0, 1.0, 1.0], 0.0016, 0.0001, [0.0015, 0.0017, 0.0017],
        0.571, 0.083, 3.427, 70, 0.0016),
    4: (1.0, 1.0, 1.0, 1.0, [1.0, 1.0, 1.0], 0.003, 0.0007, [0.0022, 0.0039, 0.003],
        0.513, 0.067, 3.429, 0, 0.0031),
    5: (0.667, 0.667, 0.667, 0.667, [0.5, 0.5, 1.0], 0.5234, 0.3677, [0.8251, 0.7393, 0.0058],
        0.517, 0.079, 3.432, 0, 0.5262),
    6: (0.5, 0.5, 0.667, 0.333, [0.5, 0.5, 0.5], 0.7763, 0.0931, [0.7021, 0.7191, 0.9076],
        0.496, 0.058, 3.434, 0, 0.7812),
    7: (0.5, 0.5, 0.667, 0.333, [0.5, 0.5, 0.5], 0.7094, 0.0107, [0.7022, 0.7015, 0.7245],
        0.504, 0.079, 3.433, 0, 0.7061),
    8: (0.5, 0.5, 0.667, 0.333, [0.5, 0.5, 0.5], 0.7103, 0.0113, [0.7018, 0.7028, 0.7263],
        0.504, 0.058, 3.435, 0, 0.7124),
    16: (0.5, 0.5, 0.667, 0.333, [0.5, 0.5, 0.5], 0.7024, 0.0003, [0.7024, 0.7028, 0.702],
         0.517, 0.037, 3.444, 0, 0.7025),
}


def train_vocab(lang):
    src = json.loads(SRC.read_text(encoding="utf-8"))
    chars = set()
    for m in src["corpus"][lang]["train"]:
        chars.update(m)
        chars.add("<s>")
    v = sorted(chars)
    v.append("<unk>")
    return v, src


def unit_tests(failures, check):
    """اختبارات وحدة مبكرة — موضع التقييم، الفحص المضاد، الأوراكل عبر المسار."""
    sys.path.insert(0, str(SIM.parent))
    import simulate_chapter6 as s6

    # (أ)+(ب): تطابق المواضع عند أطوال تغطي delay=0/2/4/8/16 (طول البادئة = delay+2)
    m = s6.SimpleRNN(40, 39, seed=7)
    for delay in (0, 2, 4, 8, 16):
        prefix = [1 + (i % 30) for i in range(delay + 2)]   # ينتهي برمز «؟» افتراضي
        full = prefix + [2]                                  # الجواب الشكلي
        logits_eval = m.predict_next_logits(prefix)
        _, Zs_full = m.forward(full)
        _, Zs_old = m.forward(prefix)
        check(f"unit.parity delay={delay}: التقييم == موضع التدريب",
              np.array_equal(logits_eval, Zs_full[-1]))
        check(f"unit.anti_check delay={delay}: يختلف عن موضع v2.1 (قبل ؟)",
              not np.array_equal(logits_eval, Zs_old[-1]))

    # (ج): Oracle عبر مسار التقييم نفسه + فحص تبديل التسميات
    out_pos = {0: 0, 1: 1, 2: 2}
    oracle = s6.OracleMemory(out_pos, 3)
    rng = np.random.default_rng(5)
    prompts_a = [[0, 2, 2] for _ in range(6)]
    prompts_b = [[1, 2, 2] for _ in range(6)]
    acc, binary, nll = s6.evaluate(oracle, prompts_a, 0, out_pos, 0, 1)
    check(f"unit.oracle_through_eval_path = {binary}", binary == 1.0 and acc == 1.0)
    _, swap_binary, _ = s6.evaluate(oracle, prompts_a, 1, out_pos, 0, 1)
    check(f"unit.oracle_label_swap = {swap_binary}", swap_binary == 0.0)
    acc_b, binary_b, _ = s6.evaluate(oracle, prompts_b, 1, out_pos, 0, 1)
    check(f"unit.oracle_class_b = {binary_b}", binary_b == 1.0)


def main():
    failures = []
    n = 0

    def check(label, ok):
        nonlocal n
        n += 1
        print(f'{"PASS" if ok else "FAIL":5} {label}')
        if not ok:
            failures.append(label)

    unit_tests(failures, check)

    # ── تشغيل المحاكي (فحوصه الصارمة تجعله يخرج برمز غير صفري عند الإخفاق) ──
    proc = subprocess.run([sys.executable, str(SIM)], capture_output=True, text=True, encoding="utf-8")
    if proc.returncode != 0:
        print("فشل المحاكي أو فحوص السلامة الصارمة:",
              (proc.stderr or "")[-400:], (proc.stdout or "")[-300:])
        return 1
    r = json.loads(RESULTS.read_text(encoding="utf-8"))
    mp = r["memory_probe"]

    # ── بصمة المدخل محسوبة مستقلة ──
    check("input_hash = sha256(examples.json)",
          r["input_hash"] == hashlib.sha256(SRC.read_bytes()).hexdigest())

    # ── نظافة أبجدية الحشو من مصدر المحاكي ──
    sys.path.insert(0, str(SIM.parent))
    import simulate_chapter6 as s6
    check(f"filler_alphabet_size = {len(s6.FILLER)} (معلن 27)",
          len(s6.FILLER) == 27 and mp["filler_alphabet_size"] == 27)
    check("الحشو بلا أ/م/؟", not ({"أ", "م", "؟"} & set(s6.FILLER)))
    check("تصنيف المهمة = أ/م", mp["classes"] == ["أ", "م"])
    check("توثيق موضع التقييم موجود", "eval_position" in mp and "predict_next_logits" in mp["eval_position"])
    check("توثيق استبعاد التداخل موجود", "test_dedup" in mp)

    # ── المفردة من التدريب فقط وإعادة بنائها من المتن ──
    for lang, exp in PUBLISHED.items():
        L = r["languages"][lang]
        v, src = train_vocab(lang)
        unseen_ind = len({c for split in ("val", "test") for m in src["corpus"][lang][split]
                          for c in m if c not in v})
        check(f"{lang}.v_ctx = {L['vocab']['context']} (مستقلة: {len(v)})",
              L["vocab"]["context"] == exp["v_ctx"] == len(v))
        check(f"{lang}.v_out = {L['vocab']['output']} (= V_ctx − 1)",
              L["vocab"]["output"] == exp["v_out"] == exp["v_ctx"] - 1)
        check(f"{lang}.unseen_chars = {L['vocab_leak_check']['unseen_chars_in_val_test']}"
              f" (مستقلة: {unseen_ind})",
              L["vocab_leak_check"]["unseen_chars_in_val_test"] == exp["unseen"] == unseen_ind)
        check(f"{lang}.vocab_source = train_only",
              L["vocab_leak_check"]["vocab_source"] == "train_only")
        p = (exp["v_ctx"] * 8 + 16 * 8 + 16 * 16 + 16 + exp["v_out"] * 16 + exp["v_out"])
        check(f"{lang}.params = {L['rnn']['params']} (= معادلة الأبعاد: {p})",
              L["rnn"]["params"] == exp["params"] == p)
        check(f"{lang}.mean_val = {L['rnn']['mean_val']}", L["rnn"]["mean_val"] == exp["mean_val"])
        check(f"{lang}.mean_test = {L['rnn']['mean_test']}", L["rnn"]["mean_test"] == exp["mean_test"])
        s42 = L["rnn"]["seeds"]["42"]
        check(f"{lang}.seed42 val/test = {s42['val_nll']}/{s42['test_nll']}",
              s42["val_nll"] == exp["seed42_val"] and s42["test_nll"] == exp["seed42_test"])
        gen = L["rnn"]["generation"]
        check(f"{lang}.generation بذرتا تهيئة وسحب = "
              f"{gen['model_init_seed']}/{gen['sampling_seed']} وبادئة واحدة",
              gen["model_init_seed"] == 42 and gen["sampling_seed"] == 42
              and gen["start_symbols"] == 1)
        check(f"{lang}.generation بلا رمز بداية", "<s>" not in gen["text"])

    # ── المهمة الاصطناعية: القياس عند موضع الإجابة الصحيح ──
    check("random_answer_nll معلن 3.434 ≈ ln(31)",
          mp["random_answer_nll"] == 3.434 and abs(mp["random_answer_nll"] - 3.4339872) < 5.1e-4)
    check("فحص تبديل التسميات = 0.0", mp["oracle_label_swap_control"] == 0.0)
    check("التأخيرات المعلنة = 2/4/5/6/7/8/16", mp["delays"] == [2, 4, 5, 6, 7, 8, 16])
    for delay, (binary, argmax, a, b, ps_bin, nll_m, nll_s, ps_nll,
                untr_bin, untr_arg, untr_nll, rejected, curve_last) in MEM.items():
        blk = mp["delays_data"][f"delay_{delay}"]
        check(f"delay_{delay}.binary = {blk['accuracy_binary']} (معلن {binary})",
              blk["accuracy_binary"] == binary)
        check(f"delay_{delay}.argmax = {blk['accuracy_argmax']} (معلن {argmax})",
              blk["accuracy_argmax"] == argmax)
        check(f"delay_{delay}.الفئتان A/B = {blk['accuracy_class_A_binary']}/{blk['accuracy_class_B_binary']}",
              blk["accuracy_class_A_binary"] == a and blk["accuracy_class_B_binary"] == b)
        check(f"delay_{delay}.لكل بذرة = {blk['per_seed_binary']}",
              blk["per_seed_binary"] == ps_bin)
        check(f"delay_{delay}.NLL اختبار = {blk['test_answer_nll_mean']} ± {blk['test_answer_nll_std']}",
              blk["test_answer_nll_mean"] == nll_m and blk["test_answer_nll_std"] == nll_s)
        check(f"delay_{delay}.NLL لكل بذرة = {blk['per_seed_test_nll']}",
              blk["per_seed_test_nll"] == ps_nll)
        check(f"delay_{delay}.المرجع عبر مسار التقييم = 1.0",
              blk["oracle_binary_through_eval_path"] == 1.0)
        check(f"delay_{delay}.خط الأساس binary = {blk['untrained_baseline_binary']}",
              blk["untrained_baseline_binary"] == untr_bin)
        check(f"delay_{delay}.خط الأساس argmax = {blk['untrained_baseline_argmax']}",
              blk["untrained_baseline_argmax"] == untr_arg)
        check(f"delay_{delay}.خط الأساس NLL = {blk['untrained_baseline_test_nll']}",
              blk["untrained_baseline_test_nll"] == untr_nll)
        check(f"delay_{delay}.سحبات مرفوضة (تداخل) = {blk['test_draws_rejected_as_collisions']}",
              blk["test_draws_rejected_as_collisions"] == rejected)
        last = blk["train_curve_mean"][-1]["train_answer_nll"]
        check(f"delay_{delay}.منحنى التدريب الأخير = {last} (معلن {curve_last})",
              last == curve_last)
        check(f"delay_{delay}.n_test = 80", blk["n_test"] == 80)
    # تأكيد استقلال التأخير 2: 70 رفضاً هناك و0 في بقية التأخيرات
    check("تداخل التأخير 2 فقط (70 رفضاً، والبقية 0)",
          mp["delays_data"]["delay_2"]["test_draws_rejected_as_collisions"] == 70
          and all(mp["delays_data"][f"delay_{d}"]["test_draws_rejected_as_collisions"] == 0
                  for d in (4, 5, 6, 7, 8, 16)))
    # فحوص السلامة + البروفانس
    check("sanity_checks = PASS", mp["sanity_checks"] == ["PASS: كل الفحوص سليمة"])
    check("توثيق تهيئة W_h موجود", "wh_init" in r["provenance"])
    check("توثيق تصحيح التقييم موجود", "eval_fix" in r["provenance"])

    # ── الجزء ج: نموذج واحد على مزيج أطوال (تدقيق v0.11) ──
    mx = r["mixed_length_model"]
    check("مزيج الأطوال: تدريب 2/4/5/6/7/8 ومحجوز 16",
          mx["train_delays"] == [2, 4, 5, 6, 7, 8] and mx["holdout_delays"] == [16]
          and mx["train_per_class_per_delay"] == 60)
    for d in (2, 4, 5, 6, 7, 8, 16):
        blk = mx["means"][f"delay_{d}"]
        check(f"mixed.delay_{d}: binary = {blk['binary_mean']} "
              f"لكل بذرة {blk['per_seed_binary']} (in_domain={blk['in_domain']})",
              blk["binary_mean"] == 0.833 and blk["per_seed_binary"] == [1.0, 0.5, 1.0]
              and blk["in_domain"] == (d != 16))
        check(f"mixed.delay_{d}: NLL = {blk['test_nll_mean']}",
              blk["test_nll_mean"] == (0.3204 if d == 2 else 0.3201))

    print()
    if failures:
        print("انحرافات:", failures)
        return 1
    print(f"كل أرقام الفصل السادس المنشورة تطابق مخرجات المحاكي ({n} فحصاً).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
