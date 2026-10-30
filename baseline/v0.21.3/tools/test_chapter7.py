#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_chapter7.py — انحدار أرقام الفصل السابع المنشورة (v2.0).

بوابة 1 (إلزامية — تدقيق v0.12): فحص تدرج عددي بالفروق المركزية لكل مجموعات
معلمات الانتباه (بموضع وبلا موضع) وbag، على مدخلات غير بسيطة، بتضخيم معلمات —
أسوأ خطأ مطلق < 1e-7 (المقيس فعلياً ≈ 4e-11).
بوابة 2: أزواج الترتيب متقابلة حرفياً (نفس الحشو للترتيبين)، الفصل على مستوى
الزوج، عدم-التبادلية باللوغيتات = 0.0 قبل التدريب وبعده، ودقة الأعمى 0.500 بالضبط.
بوابة 3: التحكمان الإيجابيان pos0/pos_mid_shift = 1.000، الأوراكل 1.000/1.000،
تبديل التسميات 0.000.
ثم ثوابت التجربتين المنشورة (متوسطات البذور والمعلمات والبصمات والأحجام).
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SIM = ROOT / "tools" / "simulate_chapter7.py"
SRC = ROOT / "content" / "chapter-03" / "examples.json"
RESULTS = ROOT / "content" / "chapter-07" / "results.json"

RETRIEVAL = {
    2: {"rnn": (1.0, 0.0017), "attn_content": (1.0, 0.0035)},
    4: {"rnn": (1.0, 0.0027), "attn_content": (1.0, 0.0067)},
    5: {"rnn": (0.5, 0.847), "attn_content": (1.0, 0.01)},
    6: {"rnn": (0.5, 0.7012), "attn_content": (1.0, 0.0159)},
    7: {"rnn": (0.5, 0.727), "attn_content": (1.0, 0.0243)},
    8: {"rnn": (0.5, 0.701), "attn_content": (1.0, 0.0456)},
    16: {"rnn": (0.5, 0.7025), "attn_content": (0.792, 0.7113)},
}
ORDER = {
    4: {"rnn": (0.5, 0.5, 0.7018), "attn_pos": (0.5, 0.5, 0.7069),
        "attn_nopos": (0.5, 0.5, 0.7166), "mean": (0.5, 0.5, 0.7116),
        "pos0": (1.0, 0.5, 0.0017), "pos_mid_shift": (0.5, 1.0, 1.4191)},
    8: {"rnn": (0.5, 0.5, 0.7025), "attn_pos": (0.5, 0.5, 0.7147),
        "attn_nopos": (0.5, 0.5, 0.7166), "mean": (0.5, 0.5, 0.7151),
        "pos0": (1.0, 0.5, 0.0017), "pos_mid_shift": (0.5, 1.0, 1.4116)},
    16: {"rnn": (0.5, 0.5, 0.7026), "attn_pos": (0.5, 0.5, 0.7162),
         "attn_nopos": (0.5, 0.5, 0.7166), "mean": (0.5, 0.5, 0.7157),
         "pos0": (1.0, 0.5, 0.0017), "pos_mid_shift": (0.5, 1.0, 1.4034)},
}
PARAMS = {"rnn": 1183, "attn_pos": 991, "attn_nopos": 671, "mean": 535,
          "pos0": 535}


def gradcheck_gate(failures, check):
    """بوابة إلزامية: مشتقات الانتباه (بموضع/بدونه) وbag مقابل فروق مركزية."""
    sys.path.insert(0, str(SIM.parent))
    import simulate_chapter7 as s7

    V, V_out = 32, 31
    rng = np.random.default_rng(42)
    for use_pos in (True, False):
        m = s7.AttnReader(V, V_out, seed=42, use_pos=use_pos, amplify=3.0)
        seqs = []
        for _ in range(4):
            f = [int(x) for x in rng.integers(3, V, size=6)]
            seqs.append([0, 1] + f + [29] if not use_pos else
                        [1, 0] + f + [29])
        answers = [0, 1, 0, 1]
        _, g = m._loss_and_grads(seqs, answers)

        def num_grad(arr, idx, eps=1e-5):
            old = arr[idx]
            arr[idx] = old + eps
            lp = m._loss_and_grads(seqs, answers)[0]
            arr[idx] = old - eps
            lm = m._loss_and_grads(seqs, answers)[0]
            arr[idx] = old
            return (lp - lm) / (2 * eps)

        worst = 0.0
        for pname in ("E", "Wk", "Wv", "wq", "Wo", "bo") + (("P",) if use_pos else ()):
            arr = getattr(m, pname)
            grad = g[pname]
            flat = [tuple(i) for i in np.ndindex(arr.shape)]
            for i in rng.choice(len(flat), size=min(10, len(flat)), replace=False):
                worst = max(worst, abs(grad[flat[i]] - num_grad(arr, flat[i])))
        check(f"gradcheck.attn use_pos={use_pos}: أسوأ خطأ {worst:.2e} < 1e-7",
              worst < 1e-7)

    mean = s7.MeanReader(V, V_out, seed=42, amplify=3.0)
    seqs = [[0, 1] + [int(x) for x in rng.integers(3, V, size=6)] + [29] for _ in range(4)]
    _, g = mean._loss_and_grads(seqs, [0, 1, 0, 1])

    def num_grad(arr, idx, eps=1e-5):
        old = arr[idx]
        arr[idx] = old + eps
        lp = mean._loss_and_grads(seqs, [0, 1, 0, 1])[0]
        arr[idx] = old - eps
        lm = mean._loss_and_grads(seqs, [0, 1, 0, 1])[0]
        arr[idx] = old
        return (lp - lm) / (2 * eps)

    worst = 0.0
    for pname in ("E", "Wo", "bo"):
        arr = getattr(mean, pname)
        flat = [tuple(i) for i in np.ndindex(arr.shape)]
        for i in rng.choice(len(flat), size=min(10, len(flat)), replace=False):
            worst = max(worst, abs(g[pname][flat[i]] - num_grad(arr, flat[i])))
    check(f"gradcheck.mean: أسوأ خطأ {worst:.2e} < 1e-7", worst < 1e-7)


def main():
    failures = []
    n = 0

    def check(label, ok):
        nonlocal n
        n += 1
        print(f'{"PASS" if ok else "FAIL":5} {label}')
        if not ok:
            failures.append(label)

    gradcheck_gate(failures, check)

    # بوابة الأزواج: بنية متقابلة حرفياً + عمى بنيوي باللوغيتات
    sys.path.insert(0, str(SIM.parent))
    import simulate_chapter7 as s7
    import simulate_chapter6 as ch6
    synth_vocab = ch6.build_vocab(ch6.CLASS_A + ch6.CLASS_B + ch6.FILLER + ch6.QUERY)
    id_a, id_b = synth_vocab[ch6.CLASS_A], synth_vocab[ch6.CLASS_B]
    rng = np.random.default_rng(7)
    fill = [synth_vocab[c] for c in rng.choice(list(ch6.FILLER), size=8, replace=True)]
    a = [id_a, id_b] + fill + [synth_vocab[ch6.QUERY]]
    b = [id_b, id_a] + fill + [synth_vocab[ch6.QUERY]]
    check("unit.pairs.multiset متطابق داخل الزوج", sorted(a) == sorted(b))
    mean = s7.MeanReader(32, 31, seed=1)
    check("unit.pairs.mean_logit_identity",
          np.allclose(mean.predict_next_logits(a), mean.predict_next_logits(b), atol=1e-10))
    attn_np = s7.AttnReader(32, 31, seed=1, use_pos=False)
    check("unit.pairs.attn_nopos_logit_identity",
          np.allclose(attn_np.predict_next_logits(a), attn_np.predict_next_logits(b), atol=1e-9))
    attn_pos = s7.AttnReader(32, 31, seed=1, use_pos=True)
    check("unit.pairs.attn_pos_position_sensitive",
          not np.allclose(attn_pos.predict_next_logits(a),
                          attn_pos.predict_next_logits(b), atol=1e-9))
    # الإزاحة: تبديل موضعي حتمي يحفظ المتعددة والبطاقة
    pre = 4
    f = a[2:2 + 8]
    sh = f[:pre] + [a[0], a[1]] + f[pre:] + [a[-1]]
    check("unit.shift.preserves_multiset_and_query",
          sorted(sh) == sorted(a) and sh[-1] == a[-1] and sh[2:2 + 8] != a[2:2 + 8])

    # ── تشغيل المحاكي (بواباته صارمة) ──
    proc = subprocess.run([sys.executable, str(SIM)], capture_output=True, text=True, encoding="utf-8")
    if proc.returncode != 0:
        print("فشل المحاكي أو بواباته:", (proc.stderr or "")[-400:], (proc.stdout or "")[-300:])
        return 1
    r = json.loads(RESULTS.read_text(encoding="utf-8"))

    check("input_hash = sha256(examples.json)",
          r["input_hash"] == hashlib.sha256(SRC.read_bytes()).hexdigest())
    check("بروتوكول: full-batch + تصحيح المشتقة موثق + أزواج متقابلة",
          r["protocol"]["optimizer"] == "full-batch gradient descent"
          and "A·(g − Σ A·g)" in r["protocol"]["softmax_fix"])
    check("أحجام الاسترجاع 100/30/40 مفحوصة",
          all(ra["seed_42"]["sizes"] == {"train_per_class": 100, "val_per_class": 30,
                                         "holdout_per_class": 40}
              for ra in [r["experiments"]["retrieval"]["delays"][f"delay_{d}"]
                         for d in (2, 4, 5, 6, 7, 8, 16)]))
    check("أحجام الترتيب 100/30/40 زوجاً مفحوصة",
          all(ro["seed_42"]["sizes"] == {"train_pairs": 100, "val_pairs": 30,
                                         "holdout_pairs": 40}
              for ro in [r["experiments"]["order"]["delays"][f"delay_{d}"]
                         for d in (4, 8, 16)]))

    ra_all = r["experiments"]["retrieval"]["delays"]
    for d, exp in RETRIEVAL.items():
        blk = ra_all[f"delay_{d}"]["mean_across_seeds"]
        for name in ("rnn", "attn_content"):
            b, nll = exp[name]
            check(f"[أ] delay_{d}.{name} = {blk[name]['holdout_binary']}/{blk[name]['holdout_test_nll']}",
                  blk[name]["holdout_binary"] == b and blk[name]["holdout_test_nll"] == nll)
        check(f"[أ] delay_{d}.oracle = 1.0", ra_all[f"delay_{d}"]["seed_42"]["oracle_ceiling"] == 1.0)

    ro_all = r["experiments"]["order"]["delays"]
    for d, models in ORDER.items():
        blk = ro_all[f"delay_{d}"]["mean_across_seeds"]
        for name, (b, sh, nll) in models.items():
            check(f"[ب] delay_{d}.{name} = {blk[name]['holdout_binary']}/"
                  f"{blk[name]['shifted_pair_binary']}/{blk[name]['holdout_test_nll']}",
                  blk[name]["holdout_binary"] == b and blk[name]["shifted_pair_binary"] == sh
                  and blk[name]["holdout_test_nll"] == nll)
        c = ro_all[f"delay_{d}"]["seed_42"]["controls"]
        check(f"[ب] delay_{d}.controls: أوراكل 1.0/1.0، تبديل 0.0، تحكمان إيجابيان 1.0/1.0",
              c["oracle_standard"] == 1.0 and c["oracle_shifted"] == 1.0
              and c["oracle_label_swap"] == 0.0
              and c["positive_control_pos0_standard"] == 1.0
              and c["positive_control_pos_mid_shifted"] == 1.0)
        inv = ro_all[f"delay_{d}"]["seed_42"]["paired_invariance_maxdiff"]
        check(f"[ب] delay_{d}.عدم-التبادلية باللوغيتات = 0.0", inv == {"mean": 0.0, "attn_nopos": 0.0})
        for name, p in PARAMS.items():
            check(f"[ب] delay_{d}.{name}.params = {ro_all[f'delay_{d}']['seed_42'][name]['params']}",
                  ro_all[f"delay_{d}"]["seed_42"][name]["params"] == p)

    # حساسية التهيئة (v0.13.1 — استكشافية على التحقق؛ holdout غير مستخدم)
    sens = r["experiments"]["order"]["init_sensitivity"]
    check("sensitivity.eval_set = validation", "validation" in sens["eval_set"])
    SENS = {4: {"amp1": 0.5, "all3": 1.0, "kq_only3": 0.5, "v_only3": 0.5, "pos_only3": 0.5},
            8: {"amp1": 0.5, "all3": 1.0, "kq_only3": 0.517, "v_only3": 0.506, "pos_only3": 0.5},
            16: {"amp1": 0.5, "all3": 0.911, "kq_only3": 0.506, "v_only3": 0.5, "pos_only3": 0.5}}
    for d, mexp in SENS.items():
        m = sens["delays"][f"delay_{d}"]["mean_across_seeds"]
        for v, val in mexp.items():
            check(f"sens.delay_{d}.{v} = {m[v]} (معلن {val})", m[v] == val)
    gn = sens["delays"]["delay_4"]["grad_norms_at_init"]["seed_42"]
    check("sens.grad_norms@init مطابقة لقياس التدقيق (Wk=9.2e-05، wq=9.1e-05)",
          gn == {"Wk": 9.2e-05, "wq": 9.1e-05, "Wv": 0.018102, "P": 0.021161, "Wo": 0.025638})
    # مسبح معدل التعلم (تهيئة ×1): الرفع وحده لا يعيد النجاح؛ 9/27 غير مستقرة
    lr4 = sens["delays"]["delay_4"]
    LR_SWEEP = {"0.3": (0.5, True), "0.9": (0.5, True), "3.0": (0.5, True),
                "9.0": (0.5, True), "27.0": (None, False)}
    for lr_v, (b, stable) in LR_SWEEP.items():
        blk = lr4[f"lr_sweep_seed_42"][str(lr_v)]
        check(f"sens.lr_sweep.{lr_v} = {blk['val_binary']} (stable={blk['stable']})",
              blk["val_binary"] == b and blk["stable"] == stable)
    # عزلات موسعة: كل التركيبات الجزئية تفشل (تدقيق v0.13.1)
    b42s = sens["delays"]["delay_4"]["seed_42"]
    for k, v in {"E_only3": 0.5, "Wo_only3": 0.5, "E_Wo3": 0.5, "E_P3": 0.5,
                 "kqv3": 0.5, "kqp3": 0.5}.items():
        check(f"sens.iso.{k} = {b42s[k]['val_binary']}", b42s[k]["val_binary"] == v)

    check("بصمات مجموعات البيانات مسجلة في التجربتين",
          all(ra_all[f"delay_{d}"]["seed_42"].get("dataset_fingerprint")
              for d in (2, 4, 5, 6, 7, 8, 16))
          and all(ro_all[f"delay_{d}"]["seed_42"].get("dataset_fingerprint")
                  for d in (4, 8, 16)))

    print()
    if failures:
        print("انحرافات:", failures)
        return 1
    print(f"كل أرقام الفصل السابع المنشورة تطابق مخرجات المحاكي ({n} فحصاً).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
