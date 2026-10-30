#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_chapter9.py — انحدار الفصل التاسع P1/P2 وبوابة النثر (تصحيح v0.19).

الأنماط:
  * الافتراضي (قراءة-فقط، لا يشغّل المحاكي ولا يكتب أي ملف — فحص كتابة-صفرية
    صارم على شجرة المستودع كاملة قبل/بعد): إعادة توليد البيانات ومطابقة بصمات
    المقاطع التسعة مع results.json ومع مرجع P0 الذهبي، فحوص بنيوية مستقلة،
    إعادة حساب مستقلة لبوابات الآلية على نماذج جديدة، تصنيف كل فحص
    environment / gate-diagnostic / science بتسامحات معلنة، بوابة Mutation،
    **تحقق التطابق العددي v1.0→v1.1** (الأرشيف مقابل الجاري)، وبوابة النثر
    (البوابة 11): كل أرقام جداول النصين مشتقة من results.json + مصطلحات
    إلزامية وممنوعة + تكافؤ أرقام الجداول ثنائي اللغة.
  * --replay (اختياري صريح): يشغّل المحاكي بكتابة مؤقتة عبر WAKEEL_CH9_OUT_DIR
    ثم يقارن بالمُقرَّرة بالمصنف الثلاثي الموروث من الفصل الثامن (تدقيق v0.17.1):
    فروق بيئة التشغيل تُسجَّل صراحةً ولا تفشل المطابقة؛ تشخيصات البوابات العائمة
    تُقارن بتسامح كل بوابة؛ وكل فرق علمي يُفشل التوقيع ويُعرض بكامله.
"""
import copy
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

# وضع القراءة-الصارم: لا كتابة bytecode إطلاقاً — يسبق أي استيراد للمحاكي.
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
SIM = ROOT / "tools" / "simulate_chapter9.py"
RESULTS = ROOT / "content" / "chapter-09" / "results.json"
RUNLOG = ROOT / "content" / "chapter-09" / "runs.log"
P0_GOLDEN = ROOT / "_ch9_p0" / "golden_ch9_p0.json"
AR = ROOT / "content" / "chapter-09" / "ar.md"
EN = ROOT / "content" / "chapter-09" / "en.md"
OUTLINE = ROOT / "content" / "chapter-09" / "outline.md"
ARCHIVE_V10 = ROOT / "content" / "chapter-09" / "archive" / "v1.0" / "results.json"

TOL_FD = 1e-6
TOL_LOGIT = 1e-9
TOL_ZERO = 1e-12
TOL_MEAN = 5e-4
TOL_ATT = 5e-4
SIM_VERSION = "simulate_chapter9-1.1"

EXPECTED_STAGE_ORDER = [
    "generate_all", "gates_1_2_3_4_7_8_non_blind", "gates_5_6_fresh",
    "lr_sweep_on_val", "FREEZE_BUDGET", "final_training_val_dev_probes",
    "gates_trained_5_9", "interventions_val", "FREEZE_MODELS",
    "blind_structural_gates", "confirmatory_read_all_models_all_seeds",
    "final_read_all_models_all_seeds",
    "post_freeze_interventions_attention_final_in", "write_results"]
EXPECTED_SIZES = {"train": 512, "val": 128, "dev_holdout": 128,
                  "p0_probe_in": 64, "p0_probe_ood_length16": 64,
                  "confirm_in": 64, "confirm_ood16": 64,
                  "final_in": 64, "final_ood16": 64}
EXPECTED_PAIRS = {"val": 32, "dev_holdout": 32, "p0_probe_in": 16,
                  "p0_probe_ood_length16": 16, "confirm_in": 16,
                  "confirm_ood16": 16, "final_in": 16, "final_ood16": 16}
EXPECTED_TRAINING_RUNS = {"sweep_calls": 9, "final_calls": 21,
                          "full_training_calls": 30,
                          "duplicated_between_sweep_and_final": 3,
                          "unique_configurations": 27}

CHECKS = []


def record(category, name, ok):
    CHECKS.append((category, name, bool(ok)))
    if not ok:
        print(f"FAIL [{category}] {name}")


def expect(condition, message):
    if not condition:
        raise AssertionError(message)


def hash_tree(root):
    out = {}
    for p in sorted(Path(root).rglob("*")):
        if p.is_file():
            out[str(p.relative_to(root)).replace("\\", "/")] = \
                hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def strip_volatile(doc):
    d = copy.deepcopy(doc)
    d.pop("run_log", None)
    d.pop("total_seconds", None)

    def walk(o):
        if isinstance(o, dict):
            o.pop("train_seconds", None)
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(d)
    return d


# ═══════════ المصنف الثلاثي الموروث من الفصل الثامن (تدقيق v0.17.1) ═══════════
# حقول مرتبطة ببيئة التشغيل: تُسجَّل فروقها صراحةً ولا تُفشل المطابقة العلمية.
ENV_LINKED_RE = re.compile(r"provenance\.runtime_versions(\..*)?$")

# تشخيصات عائمة لبوابات القياس: تُقارن وفق تسامح كل بوابة لا حرفياً — بيئات
# NumPy/BLAS مختلفة تنتج قيماً مختلفة في آخر الخانات مع بقاء كلتاهما داخل
# تسامح البوابة.
GATE_DIAG_TOLERANCES = [
    (re.compile(r"gates\.5_causal_mask\.mask_exact\..*row_sum_max_abs_error$"), 1e-12),
    (re.compile(r"gates\.5_causal_mask\.future_suffix_invariance\.(fresh|trained_seed42)\.[^.]+$"), 1e-9),
    (re.compile(r"gates\.5_causal_mask\.prefix_vs_full\.(fresh|trained_seed42)\.[^.]+$"), 1e-9),
    (re.compile(r"gates\.6_finite_difference_gradient\."), 1e-6),
    (re.compile(r"gates\.6b_aggregated_gradient\.[^.]+\.worst_abs_error$"), 1e-6),
    (re.compile(r"gates\.9_measurement_integrity\.rounded_sum_max_abs_error$"), 1e-3),
]
# حقول «حد أدنى» بدل سقف: قيمتها يجب أن تبقى فوق عتبة الكشف (لا نطاق [0,tol]).
GATE_DIAG_MIN_BOUNDS = [
    (re.compile(r"gates\.6b_aggregated_gradient\.[^.]+\.negative_unweighted_worst_abs_error$"), 1e-4),
]


def classify_diff(path, old, new):
    """تصنيف فرق واحد: ('env'|'gate_diag'|'sci')."""
    if ENV_LINKED_RE.search(path):
        return "env"
    for pat, tol in GATE_DIAG_MIN_BOUNDS:
        if pat.search(path):
            try:
                fo, fn = float(old), float(new)
                return "gate_diag" if (fo >= tol and fn >= tol) else "sci"
            except (TypeError, ValueError):
                return "sci"
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


# ═══════════════════════════ أدوات بوابة النثر (موروثة من CH-8) ═══════════════════════════
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
                    allowed.add(k)          # مفاتيح رقمية خام: "00","01","10","11","2",...
                    add(float(k))
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    for doc in docs:
        walk(doc)
    add(math.log(2))
    add(math.log(33))
    add(math.sqrt(8))
    for i in range(65):                     # ثوابت بنيوية: مفردة/أطوال/أعداد/معرفات
        allowed.add(str(i))
    return allowed


def prose_gate(check, text, lang, allowed, version):
    toks = numeric_tokens_of_table_lines(text)
    bad = sorted({t for t in toks if t not in allowed})
    check(f"prose.{lang}.tables كل أرقام الجداول ({len(toks)}) مشتقة من results.json",
          not bad)
    if bad:
        print("  أرقام غير مشتقة:", bad[:12])
    required = {
        "ar": ["السؤال الأول", "مضادة", "سقف", "تفاعل", "ترميز موضعي",
               "بذور", "محجوز", "السؤال الثاني", "تدخل", "مقيس"],
        "en": ["first query", "counterfactual pair", "ceiling", "interaction",
               "positional encoding", "seeds", "reserved", "second query",
               "intervention", "measured"],
    }[lang]
    for t in required:
        check(f"prose.{lang}.term مطلوب: «{t}»", t in text)
    forbidden = {
        "ar": ["العمق وحده", "العامل السببي الوحيد", "سقف بنيوي للقارئ الثابت",
               "يثبت أن العمق", "يفوق دائماً"],
        "en": ["depth alone", "solely caused by depth",
               "structural ceiling for the fixed reader", "proves that depth",
               "always beats"],
    }[lang]
    for t in forbidden:
        check(f"prose.{lang}.forbidden ممنوع: «{t}»", t not in text)
    check(f"prose.{lang}.version إصدار المحاكي مذكور", version in text)


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


# ═════════════════ مدقق النتائج العلمي (النسخة الحقيقية والطفرات) ═════════════════
def science_validator(results, segments, seg_digests, golden_digests, param_counts):
    """يرفع AssertionError عند أي خلل — على النسخة الحقيقية يجب أن يمر، وعلى كل
    نسخة مطفَّرة يجب أن يُكشف."""
    prov = results["provenance"]
    expect(prov["dataset_sha256"] == seg_digests,
           "dataset digests mismatch vs regeneration")
    expect(all(seg_digests[s] == golden_digests[s] for s in golden_digests),
           "P0 golden digests mismatch")
    for block in (results["gates"]["4_references"]["non_blind"]["segments"],
                  results["gates"]["4_references"]["blind_post_freeze"]["segments"]):
        for sp, rec in block.items():
            expect(rec["bag_ceiling_first_query"] == 0.75,
                   f"bag ceiling != 0.75 on {sp}")
            expect(rec["bag_matched_pair_both_correct_ceiling_first_query"] == 0.0,
                   f"bag paired-both ceiling != 0 on {sp}")
            expect(rec["oracle_binary_both_queries"] == 1.0,
                   f"oracle != 1.0 on {sp}")
            expect(rec["oracle_matched_pair_both_correct"] == 1.0,
                   f"oracle paired != 1.0 on {sp}")
    tr = results["protocol"]["training_runs"]
    for k, v in EXPECTED_TRAINING_RUNS.items():
        expect(tr[k] == v, f"training_runs.{k} = {tr[k]} != {v}")

    def sane(v):
        return isinstance(v, (int, float)) and np.isfinite(v) and 0.0 <= v <= 1.0

    def check_block(block, where):
        for met in ("first_query_binary_accuracy", "second_query_binary_accuracy",
                    "both_answers_correct", "first_query_paired_both_correct"):
            expect(sane(block[met]), f"metric out of range {met} @ {where}")
        for met in ("nll_full_vocab", "nll_binary_normalized"):
            expect(np.isfinite(block[met]) and block[met] >= 0,
                   f"nll invalid {met} @ {where}")

    seeds = [42, 43, 44]
    models = results["protocol"]["model_configs"]
    exp = results["experiments"]
    for name in models:
        for s in seeds:
            expect(f"seed_{s}" in exp["models_non_blind"][name]["seeds"],
                   f"missing seed {s} for {name} (non-blind)")
            for seg in ("val", "dev_holdout"):
                check_block(exp["models_non_blind"][name]["seeds"]
                            [f"seed_{s}"][seg], f"{name}:{s}:{seg}")
            expect(f"seed_{s}" in exp["confirmatory_post_freeze"][name],
                   f"missing seed {s} for {name} (confirm)")
            expect(f"seed_{s}" in exp["final_test_post_freeze"][name],
                   f"missing seed {s} for {name} (final)")
            for seg in ("confirm_in", "confirm_ood16"):
                check_block(exp["confirmatory_post_freeze"][name]
                            [f"seed_{s}"][seg], f"{name}:{s}:{seg}")
            for seg in ("final_in", "final_ood16"):
                check_block(exp["final_test_post_freeze"][name]
                            [f"seed_{s}"][seg], f"{name}:{s}:{seg}")

        def check_means(seed_map, segs, where):
            for seg in segs:
                for met in ("first_query_binary_accuracy",
                            "first_query_paired_both_correct",
                            "second_query_binary_accuracy",
                            "both_answers_correct"):
                    vals = [seed_map[f"seed_{s}"][seg][met] for s in seeds]
                    rec = seed_map["mean_across_seeds"][f"{seg}_{met}"]
                    expect(abs(rec["mean"] - float(np.mean(vals))) <= TOL_MEAN,
                           f"mean mismatch {where} {seg} {met}")
                    expect(abs(rec["std"] - float(np.std(vals))) <= TOL_MEAN,
                           f"std mismatch {where} {seg} {met}")
                    expect(rec["per_seed"] == vals,
                           f"per_seed mismatch {where} {seg} {met}")
        check_means(exp["models_non_blind"][name]["seeds"],
                    ("val", "dev_holdout"), f"{name}:non-blind")
        check_means(exp["confirmatory_post_freeze"][name],
                    ("confirm_in", "confirm_ood16"), f"{name}:confirm")
        check_means(exp["final_test_post_freeze"][name],
                    ("final_in", "final_ood16"), f"{name}:final")
        expect(exp["models_non_blind"][name]["seeds"]["seed_42"]["params"]
               == param_counts[name], f"param count mismatch {name}")

    def check_pairs(seed_map, segs, where):
        for seg in segs:
            got = seed_map["seed_42"][seg]["matched_pairs"]
            expect(got == EXPECTED_PAIRS[seg],
                   f"matched_pairs {where} {seg}: {got} != {EXPECTED_PAIRS[seg]}")
    for name in models:
        check_pairs(exp["models_non_blind"][name]["seeds"],
                    ("val", "dev_holdout", "p0_probe_in", "p0_probe_ood_length16"),
                    name)
        check_pairs(exp["confirmatory_post_freeze"][name],
                    ("confirm_in", "confirm_ood16"), name)
        check_pairs(exp["final_test_post_freeze"][name],
                    ("final_in", "final_ood16"), name)

    # القراءة العاملية الوصفية مطابقة للمتوسطات المسجلة (تصحيح v1.1)
    fr = exp.get("factorial_reading_descriptive")
    expect(fr is not None, "factorial_reading_descriptive missing")
    if fr is not None:
        def fmean(model):
            return exp["final_test_post_freeze"][model]["mean_across_seeds"][
                "final_in_first_query_binary_accuracy"]["mean"]
        for cell, model in (("one_layer_with_pe", "causal_self_attention_1h"),
                            ("one_layer_no_pe", "no_position"),
                            ("two_layers_with_pe", "causal_self_attention_2h"),
                            ("two_layers_no_pe", "no_position_2h")):
            expect(fr["cells"][cell] == fmean(model),
                   f"factorial cell {cell} != recorded mean")
        expect(fr["descriptive_interaction"] == round(
            (fr["cells"]["two_layers_no_pe"] - fr["cells"]["one_layer_no_pe"])
            - (fr["cells"]["two_layers_with_pe"] - fr["cells"]["one_layer_with_pe"]), 4),
            "descriptive interaction mismatch")

    sweep = exp["selection_sweep"]
    trials = sweep["trials"]
    grid = results["protocol"]["frozen_budget"]["grid"]
    anchors = results["protocol"]["frozen_budget"]["anchors"]
    expect(len(trials) == len(grid) * len(anchors), "sweep trial count wrong")
    best = max(grid, key=lambda lr: (
        min(t["val_first_query_binary_accuracy"]
            for t in trials if t["lr"] == lr), -lr))
    expect(sweep["frozen_lr"] == best,
           f"frozen lr {sweep['frozen_lr']} != rule outcome {best}")
    expect(results["protocol"]["quarantine"]["stage_order"]
           == EXPECTED_STAGE_ORDER, "stage order mismatch")
    so = results["protocol"]["quarantine"]["stage_order"]
    expect(so.index("confirmatory_read_all_models_all_seeds")
           < so.index("final_read_all_models_all_seeds"),
           "global confirmatory read must precede final read")
    for where in ("interventions_val_seed42",
                  "interventions_final_in_seed42_post_freeze"):
        for name, iv in exp[where].items():
            expect(sane(iv["baseline_q2_binary_correct"]),
                   f"intervention baseline out of range {where}/{name}")
            for mode, rec in iv["modes"].items():
                expect(sane(rec["q2_binary_accuracy"]),
                       f"intervention {mode} out of range {where}/{name}")
    for where in ("attention_maps_val_seed42",
                  "attention_maps_final_in_seed42_post_freeze"):
        for name, layers in exp[where].items():
            if not layers:
                continue
            for lk, groups in layers.items():
                for gk, cats in groups.items():
                    expect(abs(sum(cats.values()) - 1.0) <= TOL_ATT,
                           f"attention sum off {where}/{name}/{lk}/{gk}")


def run_mutation_gate(results, segments, seg_digests, golden_digests, param_counts):
    """بوابة Mutation: كل انحراف علمي حقيقي يجب أن يُكشف بواسطة المدقق نفسه."""
    def mutate_and_expect(desc, fn):
        r = copy.deepcopy(results)
        fn(r)
        try:
            science_validator(r, segments, seg_digests, golden_digests,
                              param_counts)
            record("science", f"mutation_detected:{desc}", False)
        except AssertionError:
            record("science", f"mutation_detected:{desc}", True)

    m = results["experiments"]["models_non_blind"]["causal_self_attention_1h"]
    cur = m["seeds"]["seed_42"]["val"]["first_query_binary_accuracy"]
    mutate_and_expect(
        "tamper val first_query accuracy",
        lambda r: r["experiments"]["models_non_blind"]["causal_self_attention_1h"]
        ["seeds"]["seed_42"]["val"].__setitem__(
            "first_query_binary_accuracy", 0.0 if cur > 0 else 0.5))
    mutate_and_expect(
        "tamper confirm mean",
        lambda r: r["experiments"]["confirmatory_post_freeze"]
        ["causal_self_attention_1h"]["mean_across_seeds"]
        ["confirm_in_first_query_binary_accuracy"].__setitem__("mean", 0.1234))
    mutate_and_expect(
        "tamper bag ceiling",
        lambda r: r["gates"]["4_references"]["non_blind"]["segments"]
        ["val"].__setitem__("bag_ceiling_first_query", 0.8))
    mutate_and_expect(
        "tamper oracle accuracy",
        lambda r: r["gates"]["4_references"]["blind_post_freeze"]["segments"]
        ["final_in"].__setitem__("oracle_binary_both_queries", 0.9))
    mutate_and_expect(
        "tamper frozen lr",
        lambda r: r["experiments"]["selection_sweep"].__setitem__(
            "frozen_lr", 0.003 if r["experiments"]["selection_sweep"]
            ["frozen_lr"] != 0.003 else 0.01))
    mutate_and_expect(
        "tamper final_in digest",
        lambda r: r["provenance"]["dataset_sha256"]
        .__setitem__("final_in", "0" * 64))
    mutate_and_expect(
        "drop a seed block",
        lambda r: r["experiments"]["models_non_blind"]["no_position"]["seeds"]
        .pop("seed_43"))
    mutate_and_expect(
        "tamper param count",
        lambda r: r["experiments"]["models_non_blind"]["local_only"]["seeds"]
        ["seed_42"].__setitem__("params", 999))
    mutate_and_expect(
        "swap confirm/final stage order",
        lambda r: r["protocol"]["quarantine"]["stage_order"].__setitem__(
            slice(10, 12),
            ["final_read_all_models_all_seeds",
             "confirmatory_read_all_models_all_seeds"]))
    mutate_and_expect(
        "intervention accuracy out of range",
        lambda r: r["experiments"]["interventions_val_seed42"]
        ["causal_self_attention_2h"]["modes"]["both"].__setitem__(
            "q2_binary_accuracy", 1.5))
    mutate_and_expect(
        "tamper factorial reading cell",
        lambda r: r["experiments"]["factorial_reading_descriptive"]["cells"]
        .__setitem__("two_layers_no_pe", 0.99))
    mutate_and_expect(
        "tamper training-run count",
        lambda r: r["protocol"]["training_runs"]
        .__setitem__("unique_configurations", 30))


def replay_classifier_unit(check):
    """فحص وحدوي لمنطق التصنيف الثلاثي الموروث (فروق التشخيص بتسامح بوابتها،
    وفروق المقاييس العلمية صارمة، وفروق البيئة مسجلة)."""
    cases = [
        ("gates.5_causal_mask.mask_exact.causal_self_attention_2h_layer2."
         "row_sum_max_abs_error", 3.33e-16, 2.22e-16, "gate_diag"),
        ("gates.6_finite_difference_gradient.causal_self_attention_1h."
         "briefly_trained_worst_abs_error", 2.33e-10, 4.3e-11, "gate_diag"),
        ("gates.6b_aggregated_gradient.no_position_2h."
         "negative_unweighted_worst_abs_error", 0.021, 0.019, "gate_diag"),
        ("provenance.runtime_versions.python", "3.12.10", "3.13.5", "env"),
        ("provenance.runtime_versions.numpy", "2.2.6", "2.3.1", "env"),
        ("experiments.final_test_post_freeze.causal_self_attention_2h.seed_43."
         "final_in.first_query_binary_accuracy", 0.8594, 0.9, "sci"),
        ("experiments.interventions_val_seed42.no_position.modes.memory_only."
         "q2_binary_accuracy", 0.5, 0.6, "sci"),
    ]
    for path, old, new, want in cases:
        got = classify_diff(path, old, new)
        check(f"unit.classify {path.rsplit('.', 1)[-1]} → {want}", got == want)
    got = classify_diff("gates.6_finite_difference_gradient.x."
                        "untrained_worst_abs_error", 5e-06, 2e-05)
    check(f"unit.classify رفض: تجاوز تسامح البوابة يُصنف sci ({got})", got == "sci")
    got = classify_diff("gates.6b_aggregated_gradient.x."
                        "negative_unweighted_worst_abs_error", 0.02, 1e-6)
    check("unit.classify رفض: فقدان كشف التجميع غير الموزون → sci", got == "sci")


# ═════════════════════════════════════ main ═════════════════════════════════════
def main():
    print(f"test_chapter9.py sha256: "
          f"{hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}")
    results = json.loads(RESULTS.read_text(encoding="utf-8"))

    tree_before = hash_tree(ROOT)

    # ── environment ──
    record("environment", "results.json readable with simulator_version 1.1",
           results.get("simulator_version") == SIM_VERSION)
    import platform
    for k in ("python", "numpy"):
        v = results["provenance"]["runtime_versions"].get(k, "")
        ok = bool(v) and re.fullmatch(r"[0-9][0-9A-Za-z.\-+]*", str(v)) is not None
        record("environment", f"runtime version recorded well-formed [{k}]", ok)
        if str(v) != {"python": platform.python_version(),
                      "numpy": np.__version__}[k]:
            print(f"  [env] ملاحظة بيئة: {k} المسجل {v} يختلف عن الحالي — "
                  f"يُسجَّل ولا يُفشل (مصنف الفصل الثامن)")
    src_hash = hashlib.sha256(SIM.read_bytes()).hexdigest()
    record("environment", "source_sha256 matches tools/simulate_chapter9.py",
           results["provenance"]["source_sha256"] == src_hash)
    log_lines = [json.loads(x) for x in
                 RUNLOG.read_text(encoding="utf-8").splitlines() if x.strip()]
    last = log_lines[-1]
    record("environment", "runs.log last entry == v1.1",
           last["simulator_version"] == SIM_VERSION)
    record("environment", "runs.log run_id == results run_id",
           last["run_id"] == results["run_log"]["run_id"])
    record("environment", "runs.log results hash == results.json bytes",
           last["results_sha256"] == hashlib.sha256(
               RESULTS.read_bytes()).hexdigest())
    record("environment", "runs.log source fingerprint == recorded",
           last["source_sha256"] == results["provenance"]["source_sha256"])
    v10_entry = next((e for e in log_lines
                      if e["simulator_version"] == "simulate_chapter9-1.0"), None)
    record("environment", "runs.log keeps the v1.0 entry (append-only chain)",
           v10_entry is not None and v10_entry["results_sha256"]
           == hashlib.sha256(ARCHIVE_V10.read_bytes()).hexdigest())

    # ── إعادة توليد البيانات (قراءة فقط) ──
    sys.path.insert(0, str(ROOT / "tools"))
    import simulate_chapter9 as sim
    segments, owner = sim.generate_all()
    seg_digests = {s: sim.seg_digest(segments[s]) for s in sim.SEGMENTS}
    golden = json.loads(P0_GOLDEN.read_text(encoding="utf-8"))
    golden_digests = golden["split_sha256"]

    record("science", "dataset digests == regeneration (all 9 segments)",
           seg_digests == results["provenance"]["dataset_sha256"])
    record("science", "P0 five splits == P0 golden file",
           all(seg_digests[s] == golden_digests[s] for s in sim.P0_LENGTHS))

    # ── التطابق العددي v1.0 → v1.1 (وعد corrections_v1_1) ──
    old = json.loads(ARCHIVE_V10.read_text(encoding="utf-8"))
    diffs = deep_diff(strip_volatile(old), strip_volatile(results))
    top_meta = {"simulator_version", "stage", "run_log", "total_seconds"}
    ignored_prefixes = ("protocol.", "provenance.", "gates.11_prose_discipline",
                        "experiments.factorial_reading_descriptive")

    def ignored(p):
        return (p in top_meta or p.startswith(ignored_prefixes)
                or p.endswith(".spec.desc"))   # أوصاف النماذج توثيق لا أرقام
    sci = [(p, a, b) for p, a, b in diffs if not ignored(p)]
    record("science",
           f"v1.0→v1.1 numerical identity: صفر فروق علمية/عددية "
           f"({len(sci)} متبقية من {len(diffs)} فرقاً موثقاً)",
           not sci)
    for p, a, b in sci[:10]:
        print(f"  فرق علمي متبقٍ {p}: {a!r} → {b!r}")

    # ── gate-diagnostic: بوابات البيانات مستقلة ──
    try:
        sim.gate1_input_contract(segments, sim.SEGMENTS)
        record("gate-diagnostic", "gate1 input contract on all 9 segments", True)
    except RuntimeError as e:
        record("gate-diagnostic", f"gate1 input contract: {e}", False)
    try:
        sim._balance_check(segments, sim.SEGMENTS)
        record("gate-diagnostic", "gate2 16-class balance on all 9 segments", True)
    except RuntimeError as e:
        record("gate-diagnostic", f"gate2 balance: {e}", False)
    fams = {}
    for fam, sp in owner.items():
        fams.setdefault(sp, set()).add(fam)
    overlap_ok = all(not (fams[a] & fams[b])
                     for i, a in enumerate(sim.SEGMENTS)
                     for b in sim.SEGMENTS[i + 1:])
    record("gate-diagnostic", "family disjointness across 9 segments (72 families)",
           overlap_ok and len(owner) == 72)
    record("gate-diagnostic", "segment sizes as declared",
           {s: len(segments[s]) for s in sim.SEGMENTS} == EXPECTED_SIZES)
    try:
        for sp in sim.SEGMENTS:
            sim.first_query_paired_bags(segments[sp])
        record("gate-diagnostic", "gate3 matched 01/10 pairs on all segments", True)
    except RuntimeError as e:
        record("gate-diagnostic", f"gate3 pairs: {e}", False)
    try:
        sim.gate7_loss_contract(segments["train"])
        record("gate-diagnostic", "gate7 loss contract (train)", True)
    except RuntimeError as e:
        record("gate-diagnostic", f"gate7: {e}", False)
    try:
        sim.gate8_double_intervention(segments["train"][:128] + segments["val"])
        record("gate-diagnostic", "gate8 double intervention structure", True)
    except RuntimeError as e:
        record("gate-diagnostic", f"gate8: {e}", False)

    # ── gate-diagnostic: بوابات الآلة مستقلة على نماذج جديدة ──
    fresh = {n: sim.new_model(n, 42) for n in sim.LEARNED_MODELS}
    g5 = sim.gate5_causal_mask_exact(fresh)
    record("gate-diagnostic", "gate5 causal mask exact (fresh, all archs)",
           g5 is not None)
    for n in sim.LEARNED_MODELS:
        record("gate-diagnostic", f"gate5b suffix invariance fresh [{n}]",
               sim.gate5b_future_suffix_invariance(fresh[n]) <= TOL_LOGIT)
        record("gate-diagnostic", f"gate5c prefix==full fresh [{n}]",
               sim.gate5c_prefix_vs_full(fresh[n]) <= TOL_LOGIT)
    small = sim._gate_batch({2: 2}, seed=123)
    for n in sim.LEARNED_MODELS:
        err = sim.gate6_finite_difference(fresh[n], [small])
        record("gate-diagnostic", f"gate6 independent FD fresh [{n}]",
               err <= TOL_FD)
    rec6b = results["gates"]["6b_aggregated_gradient"]
    record("gate-diagnostic", "gate6b recorded worst <= tol (all archs)",
           all(v["worst_abs_error"] <= TOL_FD for v in rec6b.values()))
    record("gate-diagnostic", "gate6b negative unweighted exposed (all archs)",
           all(v["negative_unweighted_worst_abs_error"] > 10 * TOL_FD
               for v in rec6b.values()))
    g9 = results["gates"]["9_measurement_integrity"]
    record("gate-diagnostic", "gate9 rounded attention sums within tol",
           g9["rounded_sum_max_abs_error"] <= TOL_ATT)
    record("gate-diagnostic", "gate9 negative off-by-one count rejected",
           g9["negative_off_by_one_count_rejected"] is True)
    record("gate-diagnostic", "gate9 partial subset had unequal group counts",
           all(g9["partial_subset_unequal_group_counts"].values()))

    # ── science: المدقق الشامل + بوابة Mutation ──
    param_counts = {n: sim.new_model(n, 42).params_count()
                    for n in sim.LEARNED_MODELS}
    record("science", "parameter counts == fresh recomputation",
           param_counts == results["protocol"]["parameter_counts"])
    try:
        science_validator(results, segments, seg_digests, golden_digests,
                          param_counts)
        record("science", "full science validator passes on committed results",
               True)
    except AssertionError as e:
        record("science", f"science validator: {e}", False)
    run_mutation_gate(results, segments, seg_digests, golden_digests,
                      param_counts)

    # ── science: المصنف الثلاثي — فحص وحدوي ──
    replay_classifier_unit(lambda name, ok: record("gate-diagnostic", name, ok))

    # ── بوابة النثر (البوابة 11) — بعد التأليف ──
    ar_text = AR.read_text(encoding="utf-8")
    en_text = EN.read_text(encoding="utf-8")
    record("science", "الفصلان ar.md/en.md موجدان وغير فارغين",
           len(ar_text) > 5000 and len(en_text) > 5000)
    record("science", "outline.md موثق",
           OUTLINE.exists() and len(OUTLINE.read_text(encoding="utf-8")) > 1000)
    allowed = allowed_numbers(results)
    prose_gate(lambda name, ok: record("science", name, ok),
               ar_text, "ar", allowed, SIM_VERSION)
    prose_gate(lambda name, ok: record("science", name, ok),
               en_text, "en", allowed, SIM_VERSION)
    bilingual_parity(lambda name, ok: record("science", name, ok),
                     ar_text, en_text)

    # ── replay (اختياري صريح) — بالمصنف الثلاثي الموروث ──
    if "--replay" in sys.argv:
        with tempfile.TemporaryDirectory(prefix="wakeel-ch9-replay-") as tmp:
            env = dict(os.environ, WAKEEL_CH9_OUT_DIR=tmp, PYTHONUTF8="1",
                       PYTHONDONTWRITEBYTECODE="1")
            proc = subprocess.run([sys.executable, "-B", str(SIM)],
                                  env=env, cwd=str(ROOT), capture_output=True,
                                  text=True, encoding="utf-8")
            record("environment", "replay simulator exit code 0",
                   proc.returncode == 0)
            if proc.returncode != 0:
                print((proc.stderr or "")[-2000:])
            else:
                replay_doc = json.loads(
                    (Path(tmp) / "results.json").read_text(encoding="utf-8"))
                diffs = deep_diff(strip_volatile(replay_doc),
                                  strip_volatile(results))
                env_d, diag_d, sci_d = [], [], []
                for path, a, b in diffs:
                    {"env": env_d, "gate_diag": diag_d,
                     "sci": sci_d}[classify_diff(path, a, b)].append((path, a, b))
                record("environment",
                       f"replay: صفر فروق علمية ({len(sci_d)})", not sci_d)
                for p, a, b in sci_d:
                    print(f"  فرق علمي {p}: مرجعي={a!r} جديد={b!r}")
                record("environment",
                       f"replay: تشخيصات البوابات ضمن تسامحاتها "
                       f"({len(diag_d)} حقلاً)",
                       all(classify_diff(p, a, b) == "gate_diag"
                           for p, a, b in diag_d))
                for p, a, b in diag_d:
                    print(f"  فرق تشخيص ضمن التسامح {p}: {a!r} ↔ {b!r}")
                print(f"  [env] فروق بيئة مسجلة صراحةً: {len(env_d)} حقلاً "
                      f"داخل provenance/runtime_versions")
                for p, a, b in env_d:
                    print(f"  فرق بيئة {p}: {a!r} ↔ {b!r}")
                record("environment", "replay wrote nothing into repo",
                       hash_tree(ROOT) == tree_before)

    # ── الكتابة-الصفرية الصارمة ──
    record("environment", "zero-write: repo tree identical before/after",
           hash_tree(ROOT) == tree_before)

    passed = sum(1 for _, _, ok in CHECKS if ok)
    by_cat = {}
    for cat, _, ok in CHECKS:
        by_cat.setdefault(cat, [0, 0])
        by_cat[cat][0] += int(ok)
        by_cat[cat][1] += 1
    for cat in ("environment", "gate-diagnostic", "science"):
        if cat in by_cat:
            print(f"[{cat}] {by_cat[cat][0]}/{by_cat[cat][1]}")
    if passed == len(CHECKS):
        print(f"كل فحوص الفصل التاسع نجحت ({len(CHECKS)} فحصاً)")
        return 0
    print(f"فشلت {len(CHECKS) - passed} من {len(CHECKS)} فحصاً")
    return 1


if __name__ == "__main__":
    sys.exit(main())
