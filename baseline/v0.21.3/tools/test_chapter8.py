#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_chapter8.py — انحدار الفصل الثامن وبوابته التحريرية (البوابة 12).

الأنماط (بعد تدقيق v0.16 المصدرية — فصل الفحص عن إعادة التوليد):
  * الافتراضي (قراءة-فقط، لا يشغّل المحاكيات ولا يكتب أي ملف):
      فحوص بنيوية على بيانات معاد توليدها، فروق مركزية مستقلة داخل دفعة،
      فحص مستقل للتدرج المُجمَّع الموزون عبر كتل غير متساوية (بوابة عيب v0.16)
      مع اختبار سلبي يثبت أن التجميع غير الموزون يُكتشف، مطابقة results.json
      وfactorial_isolation.json مع مرجع ذهبي منفصل (golden_v1_1.json)،
      تدقيق سلسلة runs.log، وبوابة النثر (مجموعة مشتقة + تكافؤ عددي ثنائي
      اللغة + إسناد حقلي عبر المرجع الذهبي).
  * --replay (اختياري صريح): يشغّل المحاكيَين بكتابة مؤقتة عبر
    WAKEEL_CH8_OUT_DIR ثم يقارن المخرجات المؤقتة بالمُ commits بعد استبعاد
    الحقول المتطايرة (total_seconds/train_seconds/run_log) — ملفات المستودع
    لا تُلمس. ملاحظة معلنة: إعادة التشغيل تعيد كشف مقاطع التقييم في العملية
    الجديدة فقط؛ لا يوجد تعديل ولا تسجيل على ملفات المستودع.
"""
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

# وضع القراءة-الصارم (شرط 5 في تدقيق الحزمة v0.17.1): لا كتابة bytecode إطلاقاً —
# يجب أن يسبق أي استيراد لمحاكيات الفصل.
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
SIM = ROOT / "tools" / "simulate_chapter8.py"
FSIM = ROOT / "tools" / "simulate_chapter8_factorial.py"
RESULTS = ROOT / "content" / "chapter-08" / "results.json"
FACT = ROOT / "content" / "chapter-08" / "factorial_isolation.json"
RUNLOG = ROOT / "content" / "chapter-08" / "runs.log"
GOLDEN = ROOT / "content" / "chapter-08" / "golden_v1_2.json"
AR = ROOT / "content" / "chapter-08" / "ar.md"
EN = ROOT / "content" / "chapter-08" / "en.md"

VOLATILE_RESULTS = ("run_log", "total_seconds")
VOLATILE_FACTORIAL = ("run_log", "total_seconds")
VOLATILE_SEED = ("train_seconds",)


def hash_tree(root):
    """بصمات كل ملفات الشجرة (شاملة كل شيء — شرط التدقيق: قبل/بعد مطابقة)."""
    out = {}
    for p in sorted(Path(root).rglob("*")):
        if p.is_file():
            out[str(p.relative_to(root)).replace("\\", "/")] = hashlib.sha256(
                p.read_bytes()).hexdigest()
    return out


# ═════════════════════════════ أدوات الفحص ═════════════════════════════
def numeric_tokens_of_table_lines(text):
    toks = []
    for line in text.splitlines():
        if line.strip().startswith("|"):
            for cell in line.split("|")[1:-1]:
                c = cell.strip().replace("%", "")
                for tok in re.findall(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", c):
                    toks.append(tok)
    return toks


def allowed_numbers(*docs):
    allowed = set()

    def add(v):
        v = float(v)
        for f in (f"{v:.4f}", f"{v:.3f}", f"{v:.2f}", f"{v:.1f}", f"{v:.0f}",
                  f"{v:g}", f"{v:.1e}", f"{v:.2e}", f"{v:.3e}", str(int(v))
                  if v == int(v) else f"{v:G}"):
            allowed.add(f)

    def walk(o):
        if isinstance(o, bool):
            return
        if isinstance(o, (int, float)):
            add(o)
        elif isinstance(o, dict):
            for k, v in o.items():
                if isinstance(k, str) and re.fullmatch(r"\d+(\.\d+)?", k):
                    add(float(k))
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    for doc in docs:
        walk(doc)
    add(math.log(2))
    add(math.log(32))
    add(math.sqrt(8))
    return allowed


def strip_volatile(doc, volatile_top, volatile_seed=(), volatile_paths=()):
    """نسخة مقارنة بلا حقول متطايرة (زمن حائط/سجل تشغيل) وبلا حقول مرتبطة
    ببيئة الإخراج (بصمة results.json في مجلد التشغيل نفسه)."""
    d = json.loads(json.dumps(doc))
    for k in volatile_top:
        d.pop(k, None)
    for path in volatile_paths:
        cur = d
        try:
            for k in path[:-1]:
                cur = cur[k]
            cur.pop(path[-1], None)
        except (KeyError, TypeError):
            pass

    def scrub(o):
        if isinstance(o, dict):
            for k in list(o):
                if k in volatile_seed:
                    del o[k]
                else:
                    scrub(o[k])
        elif isinstance(o, list):
            for v in o:
                scrub(v)
    scrub(d)
    return d


# ═════════════════════════════ الفحوص الحسابية المستقلة ═════════════════════════════
def gradcheck_single_batch(check, s8):
    """فروق مركزية داخل دفعة منفردة — لكل أصناف النماذج الست."""
    batches = s8.fd_batches()
    for name in s8.LEARNED_MODELS:
        m = s8.new_model(name, 42)
        worst = s8.gate5_finite_difference(m, batches)
        check(f"gradcheck.{name} (داخل دفعة): {worst:.2e} < 1e-6", worst < 1e-6)


def attention_norm_controls(check, s8):
    """تحكم مزدوج (طلب التدقيق v0.17.1): كتلة سليمة تُطبَّع، وكتلة فاسدة
    (مجموع 0.5 بدل 1) تُرفض باستثناء صريح — لا assert يُعطَّل مع python -O."""
    good = {"A": {"correct_record": 0.3, "other_record": 0.3, "filler": 0.2,
                  "prior_qa": 0.1, "self_query": 0.1},
            "B": {"correct_record": 0.4, "other_record": 0.2, "filler": 0.2,
                  "prior_qa": 0.1, "self_query": 0.1},
            "n": {"A": 1, "B": 1}}
    norm = s8._norm_att(good)
    check("unit.attention_norm_positive: كتلة سليمة (مجموع = عدد المواضع) تُطبَّع",
          abs(sum(norm["A"].values()) - 1.0) <= 1e-3
          and abs(sum(norm["B"].values()) - 1.0) <= 1e-3)
    bad = {"A": {"correct_record": 0.25, "other_record": 0.25, "filler": 0.0,
                 "prior_qa": 0.0, "self_query": 0.0},
           "B": {"correct_record": 0.5, "other_record": 0.3, "filler": 0.1,
                 "prior_qa": 0.05, "self_query": 0.05},
           "n": {"A": 4, "B": 4}}
    try:
        s8._norm_att(bad)
        rejected = False
    except RuntimeError:
        rejected = True
    check("unit.attention_norm_negative: كتلة مجموعها 0.5 تُرفض بـ RuntimeError "
          "(لا assert يُعطَّل مع -O)", rejected)


def gradcheck_aggregated(check, s8):
    """فحص عيب v0.16: التدرج المُجمَّع الموزون عبر كتل غير متساوية مقابل
    الفروق المركزية للخسارة الموزونة — مع اختبار سلبي يثبت أن التجميع
    غير الموزون يُكتشف."""
    b1 = s8._gate_batch({2: 5}, seed=941)   # 10 مواضع
    b2 = s8._gate_batch({8: 3}, seed=942)   # 6 مواضع (اتجاه عدم التساوي معاكس للمحاكي)
    groups = [(b1[0], b1[1], b1[2], 10), (b2[0], b2[1], b2[2], 6)]
    total = 16
    m = s8.new_model("causal_self_attention_1h", 42)
    _, grads = m.aggregate_grads(groups, total)

    def agg_loss():
        s = 0.0
        for X, qpos, ans, cnt in groups:
            leaves = {k: s8.leaf(v) for k, v in m.params.items()}
            logits, _ = s8.build_graph(None, X, m.spec, leaves=leaves)
            rows, fp, tg = s8._flat_supervision(qpos, ans)
            s += float(s8.nll_pick(logits, rows, fp, tg).v) * cnt
        return s / total

    eps = 1e-5
    worst_ok = 0.0
    for pname, arr in m.params.items():
        garr = grads[pname]
        for idx in np.ndindex(arr.shape):
            old = arr[idx]
            arr[idx] = old + eps
            lp = agg_loss()
            arr[idx] = old - eps
            lm = agg_loss()
            arr[idx] = old
            num = (lp - lm) / (2 * eps)
            worst_ok = max(worst_ok, abs(float(garr[idx]) - num))
    # التجميع غير الموزون (عيب v0.16): Σ∇L_i / Σn_i — اختبار سلبي: يجب أن يختلف
    acc_buggy = {k: np.zeros_like(v) for k, v in m.params.items()}
    leaves = {k: s8.leaf(v) for k, v in m.params.items()}
    for X, qpos, ans, cnt in groups:
        logits, _ = s8.build_graph(None, X, m.spec, leaves=leaves)
        rows, fp, tg = s8._flat_supervision(qpos, ans)
        g = s8.backward(s8.nll_pick(logits, rows, fp, tg))
        for k, n in leaves.items():
            gk = g.get(id(n))
            if gk is not None:
                acc_buggy[k] += gk          # بلا وزن — العيب القديم
    for k in acc_buggy:
        acc_buggy[k] /= total
    max_gap = max(float(np.max(np.abs(acc_buggy[k] - grads[k]))) for k in grads)
    check(f"gradcheck.المجمع الموزون (كتل 10/6 مواضع): {worst_ok:.2e} < 1e-6",
          worst_ok < 1e-6)
    check(f"gradcheck.اختبار سلبي: التجميع غير الموزون يختلف "
          f"(أقصى فجوة {max_gap:.2e} > 1e-4)", max_gap > 1e-4)


def structure_gates(check, s8):
    fresh = {n: s8.new_model(n, 42) for n in s8.SPECS}
    fresh["local_only"] = s8.new_model("local_only", 42)
    g1 = s8.gate1_causal_mask_exact(fresh)
    ok1 = g1 is not None and all(v["future_weight_max"] == 0.0 for v in g1.values())
    check("unit.gate1 وزن المستقبل 0.0 بالآلة لكل النماذج الانتباهية", bool(ok1))
    for n, m in fresh.items():
        g2 = s8.gate2_future_suffix_invariance(m)
        g3 = s8.gate3_prefix_vs_full(m)
        check(f"unit.gate2 {n} ثبات اللوغيتات ({g2:.1e})", g2 <= 1e-12)
        check(f"unit.gate3 {n} تطابق البادئة ({g3:.1e})", g3 <= 1e-9)
    segments, _ = s8.generate_datasets()
    g6 = s8.gate6_counterfactual_pairs(segments)
    check("unit.gate6 أزواج مضادة على كل الحلقات (896)",
          g6["value_flip_changes_only_A_answer"]
          and g6["query_order_swap_changes_positions_not_records"]
          and g6["loops_checked"] == 896)
    g5b = s8.gate5b_aggregated_gradient(s8.new_model("causal_self_attention_1h", 42))
    check(f"unit.gate5b التجميع الموزون (كتل 6/10): {g5b['worst_abs_error']:.2e} < 1e-6",
          g5b["worst_abs_error"] < 1e-6)
    return segments


# ═════════════════════════════ المطابقة مع المرجع الذهبي ═════════════════════════════
def golden_checks(check, r, fr, gold):
    for name, blk in gold["dual_lookup"].items():
        got = r["experiments"]["dual_lookup"].get(name)
        check(f"golden.dual_lookup.{name} موجود", got is not None)
        if got is None:
            continue
        for seed in ("42", "43", "44"):
            e = blk[seed]
            g = got[f"seed_{seed}"]
            check(f"golden.{name}.seed_{seed}: val={e['val']} dev={e['dev']} "
                  f"nll={e['val_nll']} params={e['params']}",
                  g["val"]["binary_accuracy"] == e["val"]
                  and g["dev_holdout"]["binary_accuracy"] == e["dev"]
                  and g["val"]["nll_full_vocab"] == e["val_nll"]
                  and g["params"] == e["params"]
                  and g["train_nll"] == e["train_nll"])
            t = got_final(r, name, seed)
            check(f"golden.final.{name}.seed_{seed}: in={e['test_in']} "
                  f"ood16={e['test_ood16']}",
                  t["test_in"]["binary_accuracy"] == e["test_in"]
                  and t["test_ood_length16"]["binary_accuracy"] == e["test_ood16"])
    for name, blk in gold["factorial_cells"].items():
        got = fr["experiments"]["factorial_cells"].get(name)
        check(f"golden.factorial.{name} موجود", got is not None)
        if got is None:
            continue
        m = got["mean_across_seeds"]
        check(f"golden.factorial.{name}: val={blk['val_mean']} "
              f"cin={blk['cin_mean']} cood={blk['cood_mean']} params={blk['params']}",
              m["val_binary_accuracy"]["mean"] == blk["val_mean"]
              and m["confirm_in_binary_accuracy"]["mean"] == blk["cin_mean"]
              and m["confirm_ood_length16_binary_accuracy"]["mean"] == blk["cood_mean"]
              and got["seed_42"]["params"] == blk["params"])
        for seed in ("42", "43", "44"):
            cf = got[f"seed_{seed}"]["counterfactual_confirm_in"]
            e = blk["counterfactual_confirm_in"][seed]
            check(f"golden.factorial.{name}.seed_{seed} تدخل: "
                  f"follow={e['follow']} stable={e['stable']}",
                  cf["A_follow_rate_when_initially_correct"] == e["follow"]
                  and cf["B_prediction_stability_rate"] == e["stable"])


def got_final(r, name, seed):
    return r["experiments"]["final_test"][name][f"seed_{seed}"]


# ═════════════════════════════ سجل التشغيل ═════════════════════════════
def runlog_checks(check, doc_pairs):
    lines = [json.loads(l) for l in RUNLOG.read_text(encoding="utf-8").splitlines()
             if l.strip()]
    check("run_log: سجلان على الأقل", len(lines) >= 2)
    for fname, fpath, doc, ver in doc_pairs:
        entry = next((e for e in reversed(lines)
                      if e["simulator_version"] == ver), None)
        check(f"run_log.{fname}: سطر يشير لأحدث إصدار",
              entry is not None and entry["simulator_version"] == ver)
        if entry is None:
            continue
        check(f"run_log.{fname}.results_sha256 = sha256(الملف الحالي)",
              entry["results_sha256"] == hashlib.sha256(
                  fpath.read_bytes()).hexdigest())
        stripped = {k: v for k, v in doc.items() if k != "run_log"}
        text_wo = json.dumps(stripped, ensure_ascii=False, indent=1) + "\n"
        e2 = {k: v for k, v in entry.items() if k not in ("run_id", "results_sha256")}
        recomputed = hashlib.sha256(
            (text_wo + json.dumps(e2, sort_keys=True, ensure_ascii=False))
            .encode("utf-8")).hexdigest()[:16]
        check(f"run_log.{fname}.run_id يُعاد حسابه ({recomputed})",
              recomputed == entry["run_id"] == doc.get("run_log", {}).get("run_id"))


# ═════════════════════════════ بوابة النثر ═════════════════════════════
def prose_gate(check, text, lang, allowed, versions):
    toks = numeric_tokens_of_table_lines(text)
    bad = sorted({t for t in toks if t not in allowed})
    check(f"prose.{lang}.tables كل أرقام الجداول ({len(toks)}) مشتقة من ملفات النتائج",
          not bad)
    if bad:
        print("  أرقام غير مشتقة:", bad[:12])
    required = {
        "ar": ["قناع سببي", "استعلام", "معلمات", "بذور", "التحقق",
               "اختبار نهائي", "تطويري"],
        "en": ["causal mask", "query", "parameters", "seeds", "validation",
               "final test", "development"],
    }[lang]
    for t in required:
        check(f"prose.{lang}.term مطلوب: «{t}»", t in text)
    forbidden = {
        "ar": ["عدم-التبادلية", "معدل تعلم فعلي", "أثبت تفوق", "يفوق دائماً"],
        "en": ["permutation invariance", "effective learning rate",
               "proves superiority", "always beats"],
    }[lang]
    for t in forbidden:
        check(f"prose.{lang}.forbidden ممنوع: «{t}»", t not in text)
    check(f"prose.{lang}.version إصدارا المحاكيَين مذكوران",
          all(v in text for v in versions))


def bilingual_parity(check, ar_text, en_text):
    """تكافؤ عرض الأرقام: جدولا اللغتين يحملان المجموعة نفسها من الأرقام
    (بنفس التكرارات) — فحص تكافؤ علمي هيكلي."""
    ca = Counter(numeric_tokens_of_table_lines(ar_text))
    ce = Counter(numeric_tokens_of_table_lines(en_text))
    only_ar = sorted({t for t in ca if ce[t] < ca[t]})
    only_en = sorted({t for t in ce if ca[t] < ce[t]})
    check(f"prose.parity تكافؤ أرقام الجداول ثنائي اللغة "
          f"(ع={sum(ca.values())}، إ={sum(ce.values())})",
          not only_ar and not only_en)
    if only_ar or only_en:
        print("  زائد في العربية:", only_ar[:8], "| زائد في الإنجليزية:", only_en[:8])


# ═════════════════════════════ إعادة التشغيل المؤقتة (--replay) ═════════════════════════════
# حقول مرتبطة ببيئة التشغيل: تُسجَّل فروقها صراحةً ولا تُفشل المطابقة العلمية
# (تدقيق v0.17.1 النهائي: قابلية إعادة الإنتاج ≠ تطابق حرفي عبر البيئات).
ENV_LINKED_RE = re.compile(
    r"provenance\.(runtime_versions(\..*)?|ch8_results_sha256)$")

# تشخيصات عائمة لبوابات القياس: تُقارن وفق تسامح كل بوابة المعتمد لا حرفياً —
# بيئات NumPy/BLAS مختلفة تنتج قيماً مختلفة في آخر الخانات مع بقاء كلتاهما
# داخل تسامح البوابة (تدقيق v0.17.1: 3.33e-16 مقابل 2.22e-16، و4.5e-11 مقابل 4.3e-11).
GATE_DIAG_TOLERANCES = [
    (re.compile(r"gates\.1_causal_mask_exact\..*row_sum_max_abs_error$"), 1e-12),
    (re.compile(r"gates\.2_future_suffix_invariance\.."), 1e-12),
    (re.compile(r"gates\.3_prefix_vs_full\.."), 1e-9),
    (re.compile(r"gates\.5_finite_difference_gradient\..*worst_abs_error$"), 1e-6),
    (re.compile(r"gates\.5b_aggregated_gradient\..*worst_abs_error$"), 1e-6),
    # Factorial simulator uses distinct paths for the same numerical diagnostics.
    (re.compile(r"gates\.5_finite_difference_fresh\.[^.]+$"), 1e-6),
    (re.compile(r"gates\.5_finite_difference_state_concat_briefly_trained$"), 1e-6),
    (re.compile(r"gates\.5b_aggregated_gradient_state_concat\.worst_abs_error$"), 1e-6),
    (re.compile(r"attention_sum_max_abs_error$"), 1e-3),
]


def classify_diff(path, old, new):
    """تصنيف فرق واحد: ('env'|'gate_diag'|'sci'). حقول البيئة أولاً، ثم تشخيصات
    البوابات العائمة، وكل ما عداه علمي صارم."""
    if ENV_LINKED_RE.search(path):
        return "env"
    for pat, tol in GATE_DIAG_TOLERANCES:
        if pat.search(path):
            try:
                fo, fn = float(old), float(new)
                ok = (0.0 <= fo <= tol) and (0.0 <= fn <= tol)
            except (TypeError, ValueError):
                ok = False
            return "gate_diag" if ok else "sci"
    return "sci"


def deep_diff(a, b, path=""):
    """فروق أوراق كاملة بين مستندين: [(مسار، قيمة_مرجعية، قيمة_جديدة)]."""
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            out += deep_diff(a.get(k), b.get(k), f"{path}.{k}" if path else str(k))
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((f"{path}[len {len(a)}!={len(b)}]", a, b))
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                out += deep_diff(x, y, f"{path}[{i}]")
    elif a != b:
        out.append((path, a, b))
    return out


def replay_classifier_unit(check):
    """فحص وحدوي لمنطق التصنيف الثلاثي (طلب تدقيق v0.17.1: فروق التشخيص العائم
    تُقارن بتسامح بوابتها، وفروق المقاييس العلمية صارمة)."""
    cases = [
        ("gates.1_causal_mask_exact.fixed_query_prefix_reader.row_sum_max_abs_error",
         3.33e-16, 2.22e-16, "gate_diag"),
        ("gates.5_finite_difference_gradient.causal_self_attention_1h."
         "briefly_trained_worst_abs_error", 5.8e-08, 4.3e-11, "gate_diag"),
        ("provenance.runtime_versions.python", "3.12.10", "3.13.5", "env"),
        ("provenance.ch8_results_sha256", "a" * 64, "b" * 64, "env"),
        ("experiments.dual_lookup.causal_self_attention_1h.seed_43.val."
         "binary_accuracy", 0.875, 0.9, "sci"),
        ("experiments.ablations.no_memory.fixed_query_prefix_reader."
         "nll_full_vocab", 14.1326, 14.2, "sci"),
        ("gates.5_finite_difference_fresh.state_query_concat",
         4.5e-11, 4.3e-11, "gate_diag"),
        ("gates.5_finite_difference_fresh.state_query_residual",
         4.2e-11, 4.5e-11, "gate_diag"),
        ("gates.5_finite_difference_state_concat_briefly_trained",
         1.653e-09, 1.608e-09, "gate_diag"),
        ("gates.5b_aggregated_gradient_state_concat.worst_abs_error",
         5e-11, 6e-11, "gate_diag"),
    ]
    for path, old, new, want in cases:
        got = classify_diff(path, old, new)
        check(f"unit.classify {path.rsplit('.', 1)[-1]} → {want}", got == want)
    # حالة رفض: تشخيص بوابة يتجاوز تسامحها يُصنف علمياً (يفشل التوقيع)
    got = classify_diff("gates.5_finite_difference_gradient.x.worst_abs_error",
                        5e-06, 2e-05)
    check(f"unit.classify رفض: تجاوز تسامح البوابة يُصنف sci ({got})", got == "sci")
    got = classify_diff("gates.5_finite_difference_fresh.state_query_concat",
                        1.1e-6, 2e-6)
    check("unit.classify رفض عاملية: بوابة 5 فوق التسامح → sci", got == "sci")


def replay(check, env, committed_hashes, tree_before):
    tmp = Path(tempfile.mkdtemp(prefix="wakeel_ch8_replay_"))
    try:
        renv = dict(env)
        renv["WAKEEL_CH8_OUT_DIR"] = str(tmp)
        renv["PYTHONDONTWRITEBYTECODE"] = "1"
        for sim, label in ((SIM, "ch8"), (FSIM, "factorial")):
            proc = subprocess.run([sys.executable, "-B", str(sim)],
                                  capture_output=True, text=True,
                                  encoding="utf-8", env=renv)
            check(f"replay.{label}: المحاكي نجح في المجلد المؤقت", proc.returncode == 0)
            if proc.returncode != 0:
                print((proc.stderr or "")[-400:])
                return
        r_new = json.loads((tmp / "results.json").read_text(encoding="utf-8"))
        f_new = json.loads((tmp / "factorial_isolation.json").read_text(encoding="utf-8"))
        r_old = json.loads(RESULTS.read_text(encoding="utf-8"))
        f_old = json.loads(FACT.read_text(encoding="utf-8"))
        # فصل المطابقة الثلاثي (تدقيق v0.17.1 النهائي): فروق بيئة التشغيل تُسجل
        # صراحةً؛ فروق تشخيص البوابات العائمة تُقارن بتسامح كل بوابة؛ وكل فرق
        # علمي يُفشل التوقيع ويُعرض بكامله — بلا حدود «أول عشرة».
        for label, new, old, volatile in (
                ("ch8", r_new, r_old, VOLATILE_RESULTS),
                ("factorial", f_new, f_old, VOLATILE_FACTORIAL)):
            diffs = deep_diff(strip_volatile(new, volatile, VOLATILE_SEED),
                              strip_volatile(old, volatile, VOLATILE_SEED))
            env_diffs, diag_diffs, sci_diffs = [], [], []
            for path, a, b in diffs:
                cls = classify_diff(path, a, b)
                {"env": env_diffs, "gate_diag": diag_diffs,
                 "sci": sci_diffs}[cls].append((path, a, b))
            check(f"replay.{label}: صفر فروق علمية في المقاييس ({len(sci_diffs)})",
                  not sci_diffs)
            for p, a, b in sci_diffs:
                print(f"  فرق علمي {p}: مرجعي={a!r} جديد={b!r}")
            for p, a, b in diag_diffs:
                print(f"  فرق تشخيص ضمن التسامح {p}: مرجعي={a!r} جديد={b!r}")
            check(f"replay.{label}: فروق تشخيص البوابات ضمن تسامحاتها "
                  f"({len(diag_diffs)} حقلاً مقبولاً)",
                  all(classify_diff(p, a, b) == "gate_diag" for p, a, b in diag_diffs))
            check(f"replay.{label}: فروق البيئة مسجلة صراحةً ({len(env_diffs)} حقلاً "
                  f"مرتبطاً ببيئة التشغيل ضمن provenance/runtime) وليست علمية",
                  True)
            for p, a, b in env_diffs:
                print(f"  فرق بيئة {p}: مرجعي={a!r} جديد={b!r}")
        after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in (RESULTS, FACT, RUNLOG)}
        check("replay: ملفات المستودع لم تُلمس (بصمات قبل/بعد متطابقة)",
              after == committed_hashes)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ═════════════════════════════════════ main ═════════════════════════════════════
def main():
    replay_mode = "--replay" in sys.argv
    failures = []
    n = 0
    tree_before = hash_tree(ROOT)
    self_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    print(f"test identity: test_chapter8.py sha256 = {self_sha}")

    def check(label, ok):
        nonlocal n
        n += 1
        print(f'{"PASS" if ok else "FAIL":5} {label}')
        if not ok:
            failures.append(label)

    sys.path.insert(0, str(SIM.parent))
    import simulate_chapter8 as s8
    import simulate_chapter8_factorial as s8f

    # ── 1) فحوص حسابية مستقلة (قراءة-فقط) ──
    segments = structure_gates(check, s8)
    gradcheck_single_batch(check, s8)
    gradcheck_aggregated(check, s8)
    attention_norm_controls(check, s8)
    replay_classifier_unit(check)

    # ── 2) قراءة المخرجات المُقرَّرة + المرجع الذهبي ──
    r = json.loads(RESULTS.read_text(encoding="utf-8"))
    fr = json.loads(FACT.read_text(encoding="utf-8"))
    gold = json.loads(GOLDEN.read_text(encoding="utf-8"))
    check("versions: simulate_chapter8-1.2 و simulate_chapter8_factorial-1.2",
          r["simulator_version"] == "simulate_chapter8-1.2"
          and fr["simulator_version"] == "simulate_chapter8_factorial-1.2")

    check("provenance.source_sha256 = sha256(simulate_chapter8.py)",
          r["provenance"]["source_sha256"] ==
          hashlib.sha256(SIM.read_bytes()).hexdigest())
    check("factorial provenance يربط أصول v1.1",
          fr["provenance"]["source_sha256"] ==
          hashlib.sha256(FSIM.read_bytes()).hexdigest()
          and fr["provenance"]["ch8_source_sha256"] ==
          hashlib.sha256(SIM.read_bytes()).hexdigest()
          and fr["provenance"]["ch8_results_sha256"] ==
          hashlib.sha256(RESULTS.read_bytes()).hexdigest())
    check("provenance.dataset_sha256 يطابق التوليد الحتمي",
          r["provenance"]["dataset_sha256"] ==
          {s: s8.seg_hash(segments[s]) for s in s8.SEGMENTS})
    check("أحجام المقاطع 512/128/128/64/64",
          r["gates"]["7_dataset_exclusivity"]["segment_sizes"] ==
          {"train": 512, "val": 128, "dev_holdout": 128,
           "test_in": 64, "test_ood": 64})
    check("بذرة البيانات مستقلة (8102) عن بذور النماذج (42/43/44)",
          r["protocol"]["seeds"]["data_generation"] == 8102
          and r["protocol"]["seeds"]["model_init"] == [42, 43, 44])
    check("ميزانية مجمدة 1200 عصراً وlr=0.01 مع توثيق تصحيح التجميع",
          r["protocol"]["frozen_budget"]["epochs"] == 1200
          and r["protocol"]["frozen_budget"]["lr"] == 0.01
          and "v1.1_fix" in r["protocol"]["training_aggregation"])
    check("لا اختيار على holdout/الاختبار",
          r["gates"]["11_validation_selection"]["holdout_used_for_selection"] is False)
    trials = r["gates"]["11_validation_selection"]["sweep_trials"]
    check("المسبح: 6 تجارب مسجلة على التحقق",
          len(trials) == 6 and all(t["lr"] in (0.01, 0.03, 0.1) for t in trials))
    best = max((0.01, 0.03, 0.1), key=lambda lr: (
        min(t["val_binary_accuracy"] for t in trials if t["lr"] == lr), -lr))
    check("قاعدة المسبح تعيد إنتاج lr المجمد",
          best == r["protocol"]["frozen_budget"]["lr"])
    g1 = r["gates"]["1_causal_mask_exact"]
    check("بوابة 1 مسجلة: 0.0 ومجاميع ≤1e-12",
          g1 is not None
          and all(v["future_weight_max"] == 0.0 for v in g1.values())
          and all(v["row_sum_max_abs_error"] <= 1e-12 for v in g1.values()))
    g2 = r["gates"]["2_future_suffix_invariance"]
    check("بوابة 2 مسجلة ≤1e-12 (جديدة ومدربة)",
          max(g2["fresh"].values()) <= 1e-12
          and max(g2["trained_seed42"].values()) <= 1e-12)
    g3 = r["gates"]["3_prefix_vs_full"]
    check("بوابة 3 مسجلة ≤1e-9",
          max(g3["fresh"].values()) <= 1e-9
          and max(g3["trained_seed42"].values()) <= 1e-9)
    g5 = r["gates"]["5_finite_difference_gradient"]
    check("بوابة 5 مسجلة <1e-6 (جديدة ومدرَّبة جزئياً)",
          all(v["untrained_worst_abs_error"] < 1e-6
              and v["briefly_trained_worst_abs_error"] < 1e-6
              for v in g5.values()))
    g5b = r["gates"]["5b_aggregated_gradient"]
    check("بوابة 5b مسجلة: التجميع الموزون عبر كتل 6/10 <1e-6",
          all(v["worst_abs_error"] < 1e-6 for v in g5b.values())
          and all(v["buckets_positions"] == [6, 10] for v in g5b.values()))
    check("بوابة 8: المرجع 1.000 وقلب التسميات 0.000",
          r["gates"]["8_positive_oracle"] == {
              "val_binary_accuracy": 1.0, "dev_holdout_binary_accuracy": 1.0,
              "test_in_binary_accuracy": 1.0,
              "val_label_flip_binary_accuracy": 0.0})
    check("بوابة 9: بلا ذاكرة المرجع 0.500 بالضبط",
          r["experiments"]["ablations"]["no_memory"]
          ["oracle_no_memory_binary_accuracy"] == 0.5)
    # بوابة تطبيع الانتباه (v0.17.1): مجموع فئات كل نوع سؤال = 1.0
    worst_att = 0.0
    for name in s8.LEARNED_MODELS:
        entry = r["experiments"]["dual_lookup"][name]["seed_42"]
        amaps = entry.get("attention_maps_val") or {}
        if not amaps:
            continue
        for q in ("A", "B"):
            worst_att = max(worst_att, abs(sum(amaps[q].values()) - 1.0))
        worst_att = max(worst_att, entry.get("attention_sum_max_abs_error", 0.0))
    check(f"بوابة تطبيع الانتباه: مجموع فئات كل نوع = 1.0 (أسوأ انحراف مقرَّب {worst_att})",
          worst_att <= 1e-3)
    for name in s8.LEARNED_MODELS:
        expected = s8.new_model(name, 42).params_count()
        got = r["experiments"]["dual_lookup"][name]["seed_42"]["params"]
        check(f"params.{name} = {expected}", got == expected)

    # نطاق البذور: إعادة حساب المتوسط/التشتت من قيم البذور المسجلة (بلا or True)
    for name in s8.LEARNED_MODELS:
        blk = r["experiments"]["dual_lookup"][name]
        seeds_ok = all(f"seed_{s}" in blk for s in (42, 43, 44))
        vals = [blk[f"seed_{s}"]["val"]["binary_accuracy"] for s in (42, 43, 44)]
        mean_rec = blk["mean_across_seeds"]["val_binary_accuracy"]
        check(f"seed_scope.{name}: 3 بذور + متوسط معاد حسابه "
              f"({mean_rec['mean']} = {round(float(np.mean(vals)), 4)})",
              seeds_ok
              and mean_rec["mean"] == round(float(np.mean(vals)), 4)
              and mean_rec["std"] == round(float(np.std(vals)), 4)
              and mean_rec["per_seed"] == vals)
        check(f"final_test.{name}: مقاييس كل بذرة موجودة",
              all(f"seed_{s}" in r["experiments"]["final_test"][name]
                  for s in (42, 43, 44)))

    # ── 3) الإضافة العاملية (قراءة-فقط) ──
    check("factorial.version = simulate_chapter8_factorial-1.2",
          fr["simulator_version"] == "simulate_chapter8_factorial-1.2")
    inh = fr["protocol"]["inherited_frozen_budget"]
    check("factorial الميزانية موروثة وبلا اختيار جديد",
          inh["lr"] == 0.01 and inh["epochs"] == 1200
          and inh["new_selection_performed"] is False)
    conf = fr["protocol"]["confirmatory_sets"]
    check("factorial تأكيدية 9204 بأحجام 64/64 وصفر تداخل",
          conf["data_generation_seed"] == 9204
          and conf["sizes"] == {"confirm_in": 64, "confirm_ood_length16": 64}
          and conf["family_overlap_with_v015_segments"] == 0)
    base_seg, base_fam = s8.generate_datasets()
    conf_seg, conf_fam = s8f.generate_confirm(set(base_fam))
    check("factorial بصمات التأكيدية تُعاد حتمياً",
          {s: s8.seg_hash(conf_seg[s]) for s in conf_seg} ==
          {k: fr["provenance"]["dataset_sha256"][k] for k in conf_seg})
    check("factorial انفصال العائلات يُعاد إثباته",
          not (set(conf_fam) & set(base_fam)))
    fg = fr["gates"]
    check("factorial بوابة 1/2/3/5/5b للخلية الجديدة ضمن الحدود",
          fg["1_causal_mask_exact"] is not None
          and max(fg["2_future_suffix_invariance"]["fresh"].values()) <= 1e-12
          and max(fg["3_prefix_vs_full"]["fresh"].values()) <= 1e-9
          and all(v < 1e-6 for v in fg["5_finite_difference_fresh"].values())
          and fg["5_finite_difference_state_concat_briefly_trained"] < 1e-6
          and fg["5b_aggregated_gradient_state_concat"]["worst_abs_error"] < 1e-6)
    gold["factorial_cells"] = gold.get("factorial_cells", {})

    # ── 4) المرجع الذهبي (إسناد حقلي للأرقام الرئيسية) ──
    golden_checks(check, r, fr, gold)

    # ── 5) سجل التشغيل ──
    runlog_checks(check, [
        ("ch8", RESULTS, r, "simulate_chapter8-1.2"),
        ("factorial", FACT, fr, "simulate_chapter8_factorial-1.2"),
    ])

    # ── 6) بوابة النثر ──
    ar_text = AR.read_text(encoding="utf-8")
    en_text = EN.read_text(encoding="utf-8")
    check("الفصلان موجودان وغير فارغين", len(ar_text) > 5000 and len(en_text) > 5000)
    allowed = allowed_numbers(r, fr, gold)
    versions = (r["simulator_version"], fr["simulator_version"])
    prose_gate(check, ar_text, "ar", allowed, versions)
    prose_gate(check, en_text, "en", allowed, versions)
    bilingual_parity(check, ar_text, en_text)

    # ── 7) إعادة التشغيل المؤقتة (اختيارية صريحة) ──
    if replay_mode:
        env = dict(os.environ)
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        committed = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in (RESULTS, FACT, RUNLOG)}
        replay(check, env, committed, tree_before)
    else:
        check("replay: غير مطلوب (افتراضي قراءة-فقط — شغّل --replay للمطابقة الحية)",
              True)

    # ── 8) شرط الكتابة-الصفرية الصارم: شجرة المستودع كاملة قبل/بعد ──
    # استثناء معلن: محضر التشغيل نفسه (test_invocation_replay.transcript.txt)
    # تكتبه أداة الالتقاط أثناء التشغيل في وضع replay، وتُنسخ إلى الحزمة بعد
    # انتهاء الفحص — فلا يُحسب تغيُّرها ضد الشيفرة.
    ignore_rel = ("content/chapter-08/test_invocation_replay.transcript.txt",) \
        if replay_mode else ()
    tree_after = hash_tree(ROOT)
    changed = sorted({k for k in set(tree_before) | set(tree_after)
                      if tree_before.get(k) != tree_after.get(k)
                      and k not in ignore_rel})
    check(f"no_write: شجرة المستودع كاملة بلا أي تغيير ({len(tree_before)} ملفاً؛ "
          f"مستثنى: محضر التشغيل)", not changed)
    if changed:
        print("  ملفات متغيرة:", changed[:10])

    print()
    if failures:
        print("انحرافات:", failures)
        return 1
    print(f"كل فحوص الفصل الثامن نجحت ({n} فحصاً) — "
          f"{'بما فيها إعادة التشغيل المؤقتة' if replay_mode else 'وضع قراءة-فقط'}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
