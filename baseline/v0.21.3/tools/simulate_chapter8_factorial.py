#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
simulate_chapter8_factorial.py — إضافة v0.16: إكمال تجربة العزل العاملَري 2×2 (v1.0)

تغلق ملاحظة المراجعة المستقلة P1 «المقارنة الرئيسية غير معزولة معمارياً»:
المقارنة بين `fixed_query_prefix_reader` و`causal_self_attention_1h` تغيّر
عاملين معاً (مصدر الاستعلام × سلك الإخراج) — فتكمل هذه الإضافة الخلية الرابعة
`state_query_concat` (استعلام تابع للحالة + خرج concat) لتكتمل الشبكة:

    fixed_query_residual  = no_query_dependency        (680 معلمة)
    state_query_residual  = causal_self_attention_1h   (736 معلمة)
    fixed_query_concat    = fixed_query_prefix_reader  (936 معلمة)
    state_query_concat    = جديدة                      (992 معلمة)

الانضباط:
* **لا اختيار جديد إطلاقاً**: الميزانية موروثة مجمّدة من بروتوكول v0.15
  (Adam، 1200 عصراً، lr = 0.01، نفس بذور النماذج، نفس بيانات التدريب/التحقق
  المولَّدة حتمياً ببذرة 8102) — لا مسبح جديد ولا ضبط على أي مقطع.
* **مجموعة تأكيدية مستقلة جديدة**: بذرة توليد 9204 (مختلفة عن 8102)، بنفس
  سياسة الأحجام والتوازن، مع إثبات انفصال العائلات عن كل مقاطع v0.15 —
  فلا يُعاد استخدام الاختبار النهائي المنشور في أي شيء.
* بوابات الآلية تُعاد على الخلية الجديدة (قناع دقيق، ثبات الماضي، تطابق
  البادئة، فروق مركزية لكل عنصر لكل المجموعات) قبل التدريب وبعده.
* **التدخل المضاد (قلب قيمة A) يُقاس لكل الخلايا ولكل البذور الثلاث** —
  إغلاقاً لملاحظة P2 (إسناد التدخل كان على بذرة 42 فقط).
* تصريح معلن بعدم تساوي المعلمات بين الخلايا (680/736/936/992): القراءة
  عاملية اتجاهية، ولا تُنسب الفروق إلى عامل واحد دون الآخر.

كل رقم يُقاس هنا ويُكتب في factorial_isolation.json؛ لا افتراض مسبق
لنتيجة أي خلية.
"""
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import simulate_chapter8 as s8  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
_OUTDIR = Path(os.environ.get("WAKEEL_CH8_OUT_DIR",
                              str(ROOT / "content" / "chapter-08")))
OUT = _OUTDIR / "factorial_isolation.json"
RUN_LOG = _OUTDIR / "runs.log"
CH8_RESULTS = _OUTDIR / "results.json"
CH8_SIM = ROOT / "tools" / "simulate_chapter8.py"
VERSION = "simulate_chapter8_factorial-1.2"

CONFIRM_DATA_SEED = 9204                     # بذرة تأكيدية جديدة — ليست 8102
CONFIRM_IN_LENGTHS = {2: 16, 4: 16, 8: 32}   # 64 حلقة تأكيدية داخل-التوزيع
CONFIRM_OOD_LENGTHS = {16: 64}               # 64 خارج-التوزيع (طول محجوز)
SEEDS = s8.MODEL_SEEDS
FROZEN_LR = 0.01                             # موروث من بروتوكول v0.15 المجمد
FROZEN_EPOCHS = s8.EPOCHS

CELLS = {
    "fixed_query_residual": {"alias": "no_query_dependency"},
    "state_query_residual": {"alias": "causal_self_attention_1h"},
    "fixed_query_concat": {"alias": "fixed_query_prefix_reader"},
    "state_query_concat": {
        "alias": "جديدة",
        "arch": "attention", "pos_mode": "sin", "query_mode": "state",
        "out_mode": "concat", "out_dim": 2 * s8.EMB_DIM},
}


class CellModel(s8.WakeelModel):
    """خلية عاملية بمواصفة صريحة (نفس آلية s8 بالكامل)."""

    def __init__(self, name, spec, seed):
        self.name = name
        self.seed = seed
        self.spec = spec
        rng = np.random.default_rng(seed)
        self.params = s8.init_params(rng, spec)


def cell_spec(name):
    alias = CELLS[name].get("alias")
    if alias in s8.SPECS:
        return dict(s8.SPECS[alias])
    return dict(CELLS[name])


def new_cell(name, seed):
    return CellModel(name, cell_spec(name), seed)


def generate_confirm(base_fams):
    """مجموعة تأكيدية جديدة: نفس سياسة v0.15 (عائلات/توازن/دفاتر) ببذرة جديدة،
    مع انفصال إلزامي عن كل عائلات v0.15 (إعادة سحب عند أي تصادم)."""
    rng = np.random.default_rng(CONFIRM_DATA_SEED)
    segments = {"confirm_in": [], "confirm_ood": []}
    fam_ledger = {}
    plan = [("confirm_in", CONFIRM_IN_LENGTHS), ("confirm_ood", CONFIRM_OOD_LENGTHS)]
    for seg, lengths in plan:
        for ln, count in sorted(lengths.items()):
            if count % 16 != 0:
                raise RuntimeError("كتلة تأكيدية ليست من مضاعفات 16 للتوازن التام")
            for _ in range(count // 16):
                for idx in rng.permutation(16):
                    cls = s8.STRUCT_CLASSES[int(idx)]
                    for _try in range(1_000_000):
                        path = tuple(int(x) for x in rng.integers(0, s8.VOCAB_N_FILLER, size=ln))
                        fam = (ln, path)
                        if fam in fam_ledger or fam in base_fams:
                            continue
                        fam_ledger[fam] = seg
                        break
                    else:
                        raise RuntimeError("فضاء العائلات التأكيدية مستنزف")
                    loop = s8.build_loop(ln, cls, path)
                    segments[seg].append(loop)
    # التوازن التام داخل كل كتلة
    for seg, loops in segments.items():
        by_len = {}
        for lp in loops:
            by_len.setdefault(lp["length"], []).append(lp)
        for ln, lps in sorted(by_len.items()):
            counts = {c: 0 for c in s8.STRUCT_CLASSES}
            for lp in lps:
                counts[tuple(lp["cls"])] += 1
            if not all(v == len(lps) // 16 for v in counts.values()):
                raise RuntimeError(f"توازن مكسور في التأكيدية {seg} L={ln}")
    # إثبات الانفصال النهائي:
    overlap = set(fam_ledger) & set(base_fams)
    if overlap:
        raise RuntimeError(f"تداخل عائلات مع v0.15: {len(overlap)}")
    return segments, fam_ledger


def main():
    t0 = time.perf_counter()
    print(f"== {VERSION} ==")

    # 1) البيانات: تدريب/تحقق v0.15 (حتمية) + تأكيدية جديدة منفصلة
    base_segments, base_fam = s8.generate_datasets()
    confirm, confirm_fam = generate_confirm(set(base_fam))
    if len(confirm["confirm_in"]) != 64 or len(confirm["confirm_ood"]) != 64:
        raise RuntimeError("أحجام المجموعة التأكيدية مخالفة للخطة 64/64")
    base_hashes = {s: s8.seg_hash(base_segments[s]) for s in s8.SEGMENTS}
    confirm_hashes = {s: s8.seg_hash(confirm[s]) for s in confirm}
    print("[بيانات] تأكيدية جديدة 64+64 — انفصال كامل عن عائلات v0.15")

    # 2) بوابات الآلية على الخلية الجديدة (جديدة) — قبل أي تدريب
    fresh_sc = new_cell("state_query_concat", SEEDS[0])
    fresh_all = {n: new_cell(n, SEEDS[0]) for n in CELLS}
    g1 = s8.gate1_causal_mask_exact(fresh_all)
    g2 = {n: s8.gate2_future_suffix_invariance(m) for n, m in fresh_all.items()}
    g3 = {n: s8.gate3_prefix_vs_full(m) for n, m in fresh_all.items()}
    for n in fresh_all:
        if g2[n] > s8.GATE_LOGIT_TOL or g3[n] > s8.GATE_LOGIT_TOL:
            raise RuntimeError(f"بوابة 2/3 فشلت على {n}: {g2[n]:.3e}/{g3[n]:.3e}")
    g5 = {n: round(s8.gate5_finite_difference(fresh_all[n], s8.fd_batches()), 12)
          for n in fresh_all}
    for n, err in g5.items():
        if err >= s8.GATE_GRAD_TOL:
            raise RuntimeError(f"بوابة 5 فشلت على {n}: {err:.3e}")
    # بوابة 5b (v1.1): التدرج المجمع عبر كتل غير متساوية — الخلية الجديدة:
    g5b_sc = s8.gate5b_aggregated_gradient(fresh_all["state_query_concat"])
    if g5b_sc["worst_abs_error"] >= s8.GATE_GRAD_TOL:
        raise RuntimeError(f"بوابة 5b فشلت: {g5b_sc['worst_abs_error']:.3e}")
    # نموذج مُدرَّب جزئياً ثم فحص ثانٍ (نفس أثر v0.15):
    s8.train_batch_steps(fresh_all["state_query_concat"], s8.fd_batches()[1],
                         lr=0.05, epochs=150)
    g5_trained = round(s8.gate5_finite_difference(fresh_all["state_query_concat"],
                                                  s8.fd_batches()), 12)
    if g5_trained >= s8.GATE_GRAD_TOL:
        raise RuntimeError(f"بوابة 5 (مدرَّب جزئياً) فشلت: {g5_trained:.3e}")
    print(f"[بوابات] 1/2/3/5 ناجحة على الخلايا الأربع (state_query_concat: "
          f"جديدة {g5['state_query_concat']:.2e}، مدرَّبة جزئياً {g5_trained:.2e})")

    # 3) التدريب والتقييم: 4 خلايا × 3 بذور — ميزانية موروثة مجمّدة، بلا اختيار
    train_loops = base_segments["train"]
    val_loops = base_segments["val"]
    cells = {}
    trained_store = {}
    for name in CELLS:
        per_seed = {}
        for seed in SEEDS:
            m = new_cell(name, seed)
            info = s8.train_model(m, train_loops, FROZEN_LR, FROZEN_EPOCHS)
            trained_store[(name, seed)] = m
            ev_val = s8.evaluate(m, val_loops)
            ev_cin = s8.evaluate(m, confirm["confirm_in"])
            ev_cood = s8.evaluate(m, confirm["confirm_ood"])
            per_seed[f"seed_{seed}"] = {
                "params": m.params_count(),
                "train_nll": info["final_train_nll"],
                "stable": info["stable"],
                "train_seconds": info["train_seconds"],
                "val": ev_val,
                "confirm_in": ev_cin,
                "confirm_ood_length16": ev_cood,
                "free_rollout_confirm_in": s8.free_rollout(m, confirm["confirm_in"]),
                # التدخل المضاد لكل بذرة (إغلاق P2):
                "counterfactual_val": s8.counterfactual_value_swap(m, val_loops),
                "counterfactual_confirm_in":
                    s8.counterfactual_value_swap(m, confirm["confirm_in"]),
            }
            print(f"[خلية] {name} seed={seed}: val={ev_val['binary_accuracy']} "
                  f"confirm_in={ev_cin['binary_accuracy']} "
                  f"confirm_ood16={ev_cood['binary_accuracy']} "
                  f"params={m.params_count()} ({info['train_seconds']}s)")
        per_seed["mean_across_seeds"] = s8._mean_eval_block(
            [per_seed[f"seed_{s}"] for s in SEEDS], "val")
        for sec in ("confirm_in", "confirm_ood_length16"):
            per_seed["mean_across_seeds"].update(
                s8._mean_eval_block([per_seed[f"seed_{s}"] for s in SEEDS], sec))
        cells[name] = per_seed

    # بوابتا 2/3 على النماذج المدربة (بذرة 42):
    g2_trained = {n: s8.gate2_future_suffix_invariance(
        trained_store[(n, SEEDS[0])]) for n in CELLS}
    g3_trained = {n: s8.gate3_prefix_vs_full(
        trained_store[(n, SEEDS[0])]) for n in CELLS}
    for n in CELLS:
        if g2_trained[n] > s8.GATE_LOGIT_TOL or g3_trained[n] > s8.GATE_LOGIT_TOL:
            raise RuntimeError(f"بوابتا 2/3 على المدرب فشلتا لـ{n}: "
                               f"{g2_trained[n]:.3e}/{g3_trained[n]:.3e}")

    # 4) التجميع والكتابة + سجل التشغيل
    src_path = Path(__file__).resolve()
    results = {
        "simulator_version": VERSION,
        "purpose": "إغلاق ملاحظة P1 (مقارنة غير معزولة معمارياً): شبكة عاملية 2×2 "
                   "لمصدر الاستعلام × سلك الإخراج، بمجموعة تأكيدية مستقلة جديدة",
        "protocol": {
            "inherited_frozen_budget": {"lr": FROZEN_LR, "epochs": FROZEN_EPOCHS,
                                        "seeds": list(SEEDS),
                                        "train_val": "بيانات v0.15 نفسها (بذرة 8102)",
                                        "new_selection_performed": False,
                                        "note": "لا مسبح جديد ولا ضبط على أي مقطع؛ "
                                                "الميزانية موروثة مجمّدة من v0.15"},
            "confirmatory_sets": {
                "data_generation_seed": CONFIRM_DATA_SEED,
                "sizes": {"confirm_in": 64, "confirm_ood_length16": 64},
                "family_overlap_with_v015_segments": 0,
                "policy": "نفس سياسة العائلات والتوازن؛ بذرة مختلفة؛ لا يُعاد استخدام "
                          "الاختبار النهائي المنشور في أي شيء"},
            "cells": {n: {"params_spec": cell_spec(n),
                          "alias": CELLS[n].get("alias")} for n in CELLS},
            "parameter_matching": "غير متساوية: 680/736/936/992 — القراءة عاملية "
                                  "اتجاهية؛ لا تُنسب الفروق إلى عامل واحد دون الآخر "
                                  "(تصريح بالعجز عن الضبط في هذه الإضافة)",
            "eval_path": "مسار v0.15 نفسه (s8.evaluate / free_rollout / "
                         "counterfactual_value_swap) بلا تعديل",
            "stance": "لا افتراض مسبق لنتيجة أي خلية",
        },
        "gates": {
            "1_causal_mask_exact": g1,
            "2_future_suffix_invariance": {"fresh": g2, "trained_seed42": g2_trained},
            "3_prefix_vs_full": {"fresh": g3, "trained_seed42": g3_trained},
            "5_finite_difference_fresh": g5,
            "5_finite_difference_state_concat_briefly_trained": g5_trained,
            "5b_aggregated_gradient_state_concat": g5b_sc,
            "tolerances": {"logits": s8.GATE_LOGIT_TOL, "grad": s8.GATE_GRAD_TOL},
        },
        "experiments": {"factorial_cells": cells},
        "provenance": {
            "ch8_results_sha256": hashlib.sha256(
                CH8_RESULTS.read_bytes()).hexdigest() if CH8_RESULTS.exists() else None,
            "ch8_source_sha256": hashlib.sha256(CH8_SIM.read_bytes()).hexdigest(),
            "source_sha256": hashlib.sha256(src_path.read_bytes()).hexdigest(),
            "dataset_sha256": {**base_hashes, **confirm_hashes},
            "runtime_versions": {"python": platform.python_version(),
                                 "numpy": np.__version__},
            "config_fingerprint": _fp({
                "version": VERSION, "confirm_seed": CONFIRM_DATA_SEED,
                "budget": [FROZEN_LR, FROZEN_EPOCHS], "seeds": sorted(SEEDS),
                "cells": sorted(CELLS)}),
        },
        "total_seconds": round(time.perf_counter() - t0, 2),
    }
    results_text = json.dumps(results, ensure_ascii=False, indent=1) + "\n"
    entry = {
        "simulator_version": VERSION,
        "source_sha256": results["provenance"]["source_sha256"],
        "config_fingerprint": results["provenance"]["config_fingerprint"],
        "dataset_sha256": results["provenance"]["dataset_sha256"],
        "inherited_frozen_budget": results["protocol"]["inherited_frozen_budget"],
        "python": platform.python_version(),
        "numpy": np.__version__,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_seconds": results["total_seconds"],
        "note": "إضافة v0.16 — شبكة عاملية بمجموعة تأكيدية مستقلة؛ سجل ملحق بلا "
                "إثبات ذاتي لانعدام وصول سابق",
    }
    run_id = hashlib.sha256((results_text +
                             json.dumps(entry, sort_keys=True, ensure_ascii=False)
                             ).encode("utf-8")).hexdigest()[:16]
    results["run_log"] = {"path": "content/chapter-08/runs.log", "run_id": run_id}
    final_text = json.dumps(results, ensure_ascii=False, indent=1) + "\n"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(final_text, encoding="utf-8", newline="\n")
    entry["run_id"] = run_id
    entry["results_sha256"] = hashlib.sha256(final_text.encode("utf-8")).hexdigest()
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(RUN_LOG, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"run_log: {run_id}")
    try:
        shown = OUT.relative_to(ROOT)
    except ValueError:
        shown = OUT
    print(f"written: {shown} | {results['total_seconds']}s")
    return 0


def _fp(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False)
                          .encode("utf-8")).hexdigest()[:16]


if __name__ == "__main__":
    sys.exit(main())
