#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
validate.py — فحص آلي لأطلس «وكيل».

يُشغَّل على مدخلات محددة، ويكتب تقريره في مسار صريح. الربط بالمدخلات عبر بصمة
SHA-256، حتى لا تصلح وثيقة قديمة كدليل على مدخلات جديدة.

الاستخدام:
    python tools/validate.py
    python tools/validate.py --report out/tmp/check.json
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

from common import fingerprint

ROOT = Path(__file__).resolve().parents[1]

DEFAULTS = {
    "registry": ROOT / "data" / "WAKEEL-2-KNOWLEDGE-REGISTRY-v0.3.json",
    "journeys": ROOT / "data" / "WAKEEL-2-JOURNEYS-v0.3.json",
    "contracts": ROOT / "data" / "WAKEEL-2-ENTRY-CONTRACTS-v0.3.json",
    "report": ROOT / "out" / "validation-report.json",
}
SCHEMAS = {
    "registry": ROOT / "schema" / "wakeel-registry.schema.json",
    "journeys": ROOT / "schema" / "wakeel-journey.schema.json",
    "contracts": ROOT / "schema" / "wakeel-entry-contracts.schema.json",
}


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


class Checker:
    def __init__(self):
        self.errors = []
        self.warnings = []

    def err(self, check, msg):
        self.errors.append({"check": check, "message": msg})

    def warn(self, check, msg):
        self.warnings.append({"check": check, "message": msg})

    def schema(self, instance, schema_path, label):
        schema = load(schema_path)
        v = Draft202012Validator(schema, format_checker=FormatChecker())
        for e in sorted(v.iter_errors(instance), key=lambda e: list(e.absolute_path)):
            loc = "/".join(str(x) for x in e.absolute_path) or "(root)"
            self.err(f"schema:{label}", f"{loc}: {e.message}")


def check_graph(c, reg, topics):
    rtypes = reg["relation_types"]
    triples = set()
    adj = defaultdict(list)
    for r in reg["relations"]:
        s, d, typ = r["source"], r["target"], r["type"]
        if typ not in rtypes:
            c.err("relations", f"نوع غير معروف {typ} في {s}->{d}")
        for x in (s, d):
            if x not in topics:
                c.err("relations", f"طرف غير موجود: {x} في {s}->{d} ({typ})")
        if s == d:
            c.err("relations", f"حلقة ذاتية: {s} ({typ})")
        key = (s, d, typ)
        if key in triples:
            c.err("relations", f"علاقة مكررة: {s}->{d} ({typ})")
        triples.add(key)
        if r.get("added_in") and r.get("review") not in ("accepted", "edited", "provisional"):
            c.err("relations", f"حافة مضافة بلا حالة مراجعة صريحة: {s}->{d} ({typ})")
        if r.get("review") == "provisional" and not r.get("review_note_ar"):
            c.err("relations", f"حافة provisional بلا سبب: {s}->{d}")
        if typ in rtypes and rtypes[typ]["is_dependency"]:
            adj[s].append(d)

    for (s, d, typ) in sorted(triples):
        if rtypes.get(typ, {}).get("symmetric") and (d, s, typ) in triples:
            c.warn("relations", f"علاقة متناظرة مخزنة بالاتجاهين: {s}<->{d} ({typ})")

    color, cycles = {}, []

    def dfs(n, stack):
        color[n] = 1
        stack.append(n)
        for m in adj.get(n, []):
            if color.get(m) == 1:
                cycles.append(stack[stack.index(m):] + [m])
            elif color.get(m, 0) == 0:
                dfs(m, stack)
        stack.pop()
        color[n] = 2

    for n in list(adj):
        if color.get(n, 0) == 0:
            dfs(n, [])
    for cyc in cycles:
        c.err("cycles", "دورة اعتماد: " + " -> ".join(cyc))


CANONICAL_STATION_ORDER = [
    "FND-002", "FND-004", "FND-013", "NLP-001", "ML-001",
    "LLM-002", "NLP-002", "NLP-003",
    "DAT-001", "NLP-004", "LLM-003", "LLM-007", "LLM-009", "DAT-008", "LLM-008", "LLM-010",
    "LLM-011", "DLN-017", "LLM-013", "DLN-014", "LLM-014",
    "DLN-010", "TRN-005", "TRN-004", "TRN-006",
    "LLM-021", "EVA-002", "INF-001", "INF-002", "EVA-007",
    # التوسعة الرسمية v0.13.1: محطات الفصل السابع (31-36) — هويات المحطات 1-30 لم تُمس
    "AGT-013", "HIS-010", "AGT-007", "LLM-016", "DLN-012", "TRN-003",
    # التوسعة الرسمية v0.21 (قرار الدمج المعتمد 8+9): محطات الفصلين الثامن
    # والتاسع (37-44) بمعرفات موضوعات قائمة غير مستخدمة — هويات 1-36 لم تُمس
    "LLM-004", "DLN-004", "DAT-015", "EVA-009",
    "DAT-014", "EVA-005", "DLN-002", "DAT-016",
]


def check_journeys(c, jrn, topics):
    # حماية هوية الرحلة: أي حذف أو تبديل غير مصرح به للمحطات الثلاثين يفشل الفحص.
    actual_order = [s["topic_id"] for s in jrn["journeys"][0]["stations"]] if jrn.get("journeys") else []
    if actual_order and actual_order != CANONICAL_STATION_ORDER:
        missing = [t for t in CANONICAL_STATION_ORDER if t not in actual_order]
        extra = [t for t in actual_order if t not in CANONICAL_STATION_ORDER]
        c.err("journey", f"ترتيب/هوية المحطات غير مطابق للمعتمد؛ ناقص: {missing}؛ زائد: {extra}")
    if len({j["id"] for j in jrn["journeys"]}) != len(jrn["journeys"]):
        c.err("journey", "معرّف رحلة مكرر")

    used_in = defaultdict(list)
    legacy_stations = []
    for j in jrn["journeys"]:
        # نطاق التكرار: داخل الرحلة الواحدة فقط. إعادة استخدام الموضوع بين رحلات مسموحة.
        seen_here = set()
        orders = [s["order"] for s in j["stations"]]
        if orders != list(range(1, len(orders) + 1)):
            c.err("journey", f'{j["id"]}: ترتيب المحطات غير متصل 1..n')
        for s in j["stations"]:
            tid = s["topic_id"]
            if tid not in topics:
                c.err("journey", f'{j["id"]}#{s["order"]}: موضوع غير موجود {tid}')
            if tid in seen_here:
                c.err("journey", f'{j["id"]}: الموضوع مكرر داخل الرحلة {tid}')
            seen_here.add(tid)
            used_in[tid].append(j["id"])
            if s["role"] not in jrn["station_roles"]:
                c.err("journey", f'{j["id"]}#{s["order"]}: دور غير معروف')
            if s["role"] == "core":
                if not (s.get("deliverable_ar") or "").strip():
                    c.err("journey", f'{j["id"]}#{s["order"]}: محطة أساسية بلا مخرج')
                if not (s.get("activity") or {}):
                    c.err("journey", f'{j["id"]}#{s["order"]}: محطة أساسية بلا نشاط')
            # (تصحيح v0.21 — قرار الدمج) عقد النشاط يُفحص لكل محطة ذات نشاط،
            # أساسية كانت أو اختيارية؛ كان فحص expects_ar محصوراً في الأساسية
            # فكانت المحطات الاختيارية الثلاث تُغفل (اكتشاف مراجعة الدمج).
            act = s.get("activity") or {}
            if act:
                for f in ("sees_ar", "tries_ar", "discovers_ar"):
                    if not (act.get(f) or "").strip():
                        c.err("journey", f'{j["id"]}#{s["order"]}: نشاط ناقص ({f})')
                if act.get("expects_ar"):
                    legacy_stations.append(s["order"])
                else:
                    for f in ("hypothesis_ar", "interpretation_ar"):
                        if not str(act.get(f) or "").strip():
                            c.err("journey", f'{j["id"]}#{s["order"]}: نشاط بلا {f}')
                    if len(act.get("possible_outcomes_ar") or []) < 2:
                        c.err("journey", f'{j["id"]}#{s["order"]}: نتائج محتملة أقل من اثنتين')
                    # (v0.21.2 — قرار G7 المعتمد) التكافؤ اللغوي الكامل بلا استثناءات
                    # صار مفروضاً في check_language_parity مع اختبار حذف سلبي ذاتي.

        # الفصول: تغطية كاملة، بلا تداخل، وبترتيب المحطات
        chapters = j.get("chapters") or []
        if not chapters:
            c.err("journey", f'{j["id"]}: بلا فصول تعليمية')
        covered = []
        for ch in chapters:
            for o in ch["station_orders"]:
                covered.append(o)
        if sorted(covered) != sorted(orders):
            c.err("journey", f'{j["id"]}: فصول لا تغطي المحطات مرة واحدة بالضبط')
        if len(covered) != len(set(covered)):
            c.err("journey", f'{j["id"]}: محطة في أكثر من فصل')
        for ch in chapters:
            if ch["station_orders"] != sorted(ch["station_orders"]):
                c.err("journey", f'{j["id"]}/{ch["id"]}: ترتيب المحطات داخل الفصل غير تصاعدي')
            # (v0.21.2) تكافؤ عناوين/أهداف كل الفصول مفروض في check_language_parity
            cont = ch.get("content") or {}
            for lang in ("ar", "en"):
                rel = cont.get(lang)
                if rel and not (ROOT / rel).is_file():
                    c.err("journey", f'{j["id"]}/{ch["id"]}: ملف المحتوى غير موجود {rel}')
            if cont.get("status") in ("authored", "reviewed") and not (cont.get("ar") and cont.get("en")):
                c.err("journey", f'{j["id"]}/{ch["id"]}: فصل موسوم authored بلا نسختين')

        for b in j.get("optional_branches_ar", []):
            if b["topic_id"] not in topics:
                c.err("journey", f'{j["id"]}: فرع اختياري غير موجود {b["topic_id"]}')
        if not j.get("exit_evidence_ar"):
            c.err("journey", f'{j["id"]}: بلا أدلة خروج قابلة للفحص')

    if legacy_stations:
        c.warn("journey", f"{len(legacy_stations)} محطة بعقد نشاط قديم (expects) تحتاج ترحيلاً: {legacy_stations}")
    reused = {t: js for t, js in used_in.items() if len(js) > 1}
    return used_in, reused


def language_parity_errors(jrn_doc):
    """(v0.21.2 — قرار G7 المعتمد: بلا استثناءات) تكافؤ ثنائي كامل للرحلة:
    كل محطة (السياق/المخرج/حقول النشاط الستة) وكل فصل (العنوان/الهدف)
    وحقول الرحلة (الهدف/السؤال/ملاحظة الاستقلال/أدلة الخروج/الملاحظات/
    الفروع) وحقول الملف (قاعدة الفصل/أدوار المحطات)."""
    errs = []
    if not (jrn_doc.get("separation_rule_en") or "").strip():
        errs.append("parity: separation_rule_en مفقود")
    roles_en = jrn_doc.get("station_roles_en") or {}
    for r in ("core", "optional", "reference"):
        if not (roles_en.get(r) or "").strip():
            errs.append(f"parity: station_roles_en.{r} مفقود")
    for j in jrn_doc["journeys"]:
        for f in ("goal_en", "starts_with_question_en", "independence_note_en"):
            if not (j.get(f) or "").strip():
                errs.append(f'parity: {j["id"]} بلا {f}')
        if not (j.get("exit_evidence_en") or []):
            errs.append(f'parity: {j["id"]} بلا exit_evidence_en')
        for s in j["stations"]:
            o = s["order"]
            if not (s.get("why_here_en") or "").strip():
                errs.append(f"parity: المحطة {o} بلا why_here_en")
            d_ar, d_en = s.get("deliverable_ar"), s.get("deliverable_en")
            if (d_ar is None) != (d_en is None):
                errs.append(f"parity: المحطة {o} deliverable وجوده غير متطابق بين اللغتين")
            elif d_ar is not None and not str(d_en).strip():
                errs.append(f"parity: المحطة {o} بلا deliverable_en")
            act = s.get("activity") or {}
            for f in ("sees_en", "tries_en", "hypothesis_en",
                      "interpretation_en", "discovers_en"):
                if not (act.get(f) or "").strip():
                    errs.append(f"parity: المحطة {o} نشاط بلا {f}")
            if len(act.get("possible_outcomes_en") or []) < 2:
                errs.append(f"parity: المحطة {o} possible_outcomes_en أقل من اثنتين")
        for ch in j.get("chapters") or []:
            for f in ("title_en", "goal_en"):
                if not (ch.get(f) or "").strip():
                    errs.append(f'parity: {ch["id"]} بلا {f}')
        for n in j.get("methodology_notes_en") or []:
            if not (n.get("note_en") or "").strip():
                errs.append(f'parity: methodology_notes_en[{n.get("topic_id")}] بلا note_en')
        for b in j.get("optional_branches_en") or []:
            if not (b.get("when_en") or "").strip():
                errs.append(f'parity: optional_branches_en[{b.get("topic_id")}] بلا when_en')
    return errs


def check_language_parity(c, jrn_doc):
    for msg in language_parity_errors(jrn_doc):
        c.err("language_parity", msg)


def parity_negative_selftest(c, jrn_doc):
    """(قرار G7، شرط 3) حذف أي حقل إنجليزي يجب أن يُكتشف — طفرات حذف ممثلة
    لنشاط المحطة وسياقها والفصل والرحلة وعدم تطابق وجود المخرج."""
    import copy

    def errs_after(mutate):
        d = copy.deepcopy(jrn_doc)
        mutate(d)
        return language_parity_errors(d)

    checks = [
        ("station activity field deletion",
         lambda d: d["journeys"][0]["stations"][4]["activity"].pop("hypothesis_en"),
         "hypothesis_en"),
        ("station context field deletion",
         lambda d: d["journeys"][0]["stations"][20].pop("why_here_en"),
         "why_here_en"),
        ("chapter field deletion",
         lambda d: d["journeys"][0]["chapters"][0].pop("goal_en"),
         "goal_en"),
        ("journey field deletion",
         lambda d: d["journeys"][0].pop("goal_en"),
         "goal_en"),
        ("deliverable presence mismatch",
         lambda d: d["journeys"][0]["stations"][16].update(deliverable_en=None),
         "deliverable"),
    ]
    for name, mutate, needle in checks:
        errs = errs_after(mutate)
        if not any(needle in e for e in errs):
            c.err("language_parity",
                  f"اختبار الحذف السلبي أعمى عن {name} ({needle})")


def station_semantic_errors(jrn):
    """(v0.21/v0.21.2) أخطاء دلالة المحطات S1-S4 — قائمة رسائل؛ تستخدمها
    الفحوص الرسمية والاختبار السلبي الذاتي معاً."""
    import math
    errs = []
    st = {s["order"]: s for j in jrn["journeys"] for s in j["stations"]}

    def _sm(xs):
        m = max(xs)
        es = [math.exp(x - m) for x in xs]
        t = sum(es)
        return [e / t for e in es]

    full = _sm([1.0, 2.0, 3.0])
    masked = _sm([1.0, 2.0]) + [0.0]
    if not (masked[2] == 0.0 and full[0] != masked[0] and full[1] != masked[1]):
        errs.append("برهان إعادة التطبيع (المحطة 19) فشل حسابياً")
    blob19 = json.dumps(st[19], ensure_ascii=False)
    a19 = st[19]["activity"]
    if "دون أي أثر على أوزان الماضي" in blob19 or not (
            "يعيد التطبيع" in a19["interpretation_ar"]
            or "يعيد تطبيع" in a19["interpretation_ar"]):
        errs.append("المحطة 19: تغير الأوزان المرئية ليس دليل خطأ؛ يجب تفسير إعادة التطبيع")
    if "تغير الخرج رغم تثبيت" in blob19:
        errs.append("المحطة 19: الحالة الثالثة يجب أن تكون ضابط تكافؤ المسارين لا تثبيت اللاحقة المبتور")
    if "مساري التقييم" not in a19["possible_outcomes_ar"][2]:
        errs.append("المحطة 19: الحالة الثالثة يجب أن تختبر تكافؤ مساري التقييم صراحة")
    a22 = st[22]["activity"]
    if 31 * 32 + 8 != 1000:
        errs.append("حساب الدفعات (المحطة 22) غير سليم")
    if "تُرى أكثر من غيرها" in json.dumps(a22, ensure_ascii=False) \
            or "دفعة أخيرة من 8" not in a22["possible_outcomes_ar"][0]:
        errs.append("المحطة 22: التقريب لا يعني تكراراً؛ الدفعة الأخيرة 8 أمثلة")
    blob20 = json.dumps(st[20], ensure_ascii=False)
    s20 = st[20]
    a20 = s20["activity"]
    for banned in ("الانتباه نفسه لا يعرف الترتيب", "تحتاج إدخالاً صريحاً",
                   "مهما تعددت طبقاته", "في كل البنى"):
        if banned in blob20:
            errs.append(f"المحطة 20: عبارة أوسع من النتيجة («{banned}»)")
    # متطلبات إيجابية (v0.21.2): علامات النطاق الصحيح يجب أن تبقى في حقولها —
    # استبدال الحقل بأي صياغة غير مقيّدة يُفقد العلامة فيُكتشف
    if not ("الواحدة" in s20["why_here_ar"] and "هذه المهمة" in s20["why_here_ar"]):
        errs.append("المحطة 20: لماذا_هنا يجب أن يقيد العمى بالطبقة الواحدة وبهذه المهمة")
    if "كما قاس الفصل التاسع" not in a20["discovers_ar"]:
        errs.append("المحطة 20: يكتشف يجب أن يحمل تحفظ العمق (إشارة الفصل التاسع)")
    if not all(k in a20["hypothesis_ar"] for k in
               ("طبقة انتباه واحدة", "موضع السؤال", "متساويتين في تعداد")):
        errs.append("المحطة 20: الحكم مقيد بطبقة واحدة عند موضع السؤال وببادئتين متساويتين تعداداً")
    a41 = st[41]["activity"]
    if "لا يثبت" not in a41["interpretation_ar"] or "مجموعة الرموز" in a41["interpretation_ar"]:
        errs.append("المحطة 41: فشل الأزواج لا يحدد بمفرده آلية الخطأ")
    # (v0.21.3 — تصحيحا مراجعة الاعتماد) وحدتا قياس ومصطلح علمي
    blob28_en = json.dumps(st[28], ensure_ascii=False)
    if "one token" in blob28_en or "one case" not in st[28]["activity"]["hypothesis_en"]:
        errs.append("المحطة 28: وحدة القياس الإنجليزية يجب أن تكون case مطابقة للعربية")
    if "uselessness" in json.dumps(st[43], ensure_ascii=False):
        errs.append("المحطة 43: غياب الضرورة ≠ انعدام الفائدة — استخدم lack of necessity")
    return errs


def check_station_semantics(c, jrn):
    for msg in station_semantic_errors(jrn):
        c.err("station_semantics", msg)


def semantic_guard_negative_selftest(c, jrn):
    """(v0.21.2 — طلب التدقيق المستقل §4) اختبار سلبي للحارس نفسه: خمس طفرات
    في الذاكرة، كل واحدة يجب أن تُكتشف؛ نجاة أي منها خطأ صريح في الفحص."""
    import copy
    mutations = {
        "st20_discovers_overgeneral": lambda st: st[20]["activity"].__setitem__(
            "discovers_ar",
            "الانتباه السببي لا يستطيع معرفة الترتيب مهما تعددت طبقاته."),
        "st20_why_overgeneral": lambda st: st[20].__setitem__(
            "why_here_ar", "الانتباه نفسه لا يعرف الترتيب."),
        "st19_outcome_illposed": lambda st: st[19]["activity"]
            ["possible_outcomes_ar"].__setitem__(
                2, "تغير الخرج رغم تثبيت الرموز اللاحقة"),
        "st22_repetition_claim": lambda st: st[22]["activity"].__setitem__(
            "interpretation_ar",
            "التقريب للأعلى يعني أن بعض الأمثلة تُرى أكثر من غيرها."),
        "st41_overclaim": lambda st: st[41]["activity"].__setitem__(
            "interpretation_ar",
            "الدقة بلا أزواج تعني حلاً من مجموعة الرموز لا من علاقاتها."),
    }
    for name, mutate in mutations.items():
        j2 = copy.deepcopy(jrn)
        st2 = {s["order"]: s for s in j2["journeys"][0]["stations"]}
        mutate(st2)
        if not station_semantic_errors(j2):
            c.err("station_semantics",
                  f"الحارس الدلالي أعمى عن طفرة {name} — يجب أن تُكتشف")


def check_contracts(c, con, topics, journeys_by_id):
    for x in con["contracts"]:
        tid = x["topic_id"]
        if tid not in topics:
            c.err("contracts", f"عقد لموضوع غير موجود: {tid}")
            continue
        if not x.get("inline_explanations"):
            c.err("contracts", f"{tid}: بلا شرح داخل الصفحة (خطر بوابة)")
        if not x.get("forbidden_gates_ar"):
            c.err("contracts", f"{tid}: بلا قائمة بوابات ممنوعة")
        if not x.get("proof_of_independence_ar"):
            c.err("contracts", f"{tid}: بلا دليل استقلال")
        jid, order = x.get("linked_journey_id"), x.get("linked_station_order")
        if jid:
            j = journeys_by_id.get(jid)
            if j is None:
                c.err("contracts", f"{tid}: linked_journey_id غير موجود {jid}")
            else:
                station = next((s for s in j["stations"] if s["order"] == order), None)
                if station is None:
                    c.err("contracts", f"{tid}: المحطة {order} غير موجودة في {jid}")
                elif station["topic_id"] != tid:
                    c.err("contracts", f"{tid}: المحطة {order} في {jid} تشير إلى {station['topic_id']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="فحص أطلس وكيل")
    ap.add_argument("--registry", default=str(DEFAULTS["registry"]))
    ap.add_argument("--journeys", default=str(DEFAULTS["journeys"]))
    ap.add_argument("--contracts", default=str(DEFAULTS["contracts"]))
    ap.add_argument("--report", default=str(DEFAULTS["report"]))
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    files = {"registry": a.registry, "journeys": a.journeys, "contracts": a.contracts}
    fp = fingerprint(files)

    reg, jrn, con = load(a.registry), load(a.journeys), load(a.contracts)
    c = Checker()
    c.schema(reg, SCHEMAS["registry"], "registry")
    c.schema(jrn, SCHEMAS["journeys"], "journeys")
    c.schema(con, SCHEMAS["contracts"], "contracts")

    topics = {t["id"]: t for t in reg["topics"]}
    domains = {d["id"]: d for d in reg["domains"]}
    groups = {g["id"]: g for g in reg["domain_groups"]}

    if len(topics) != len(reg["topics"]):
        c.err("ids", "معرّفات موضوعات مكررة")
    if len(domains) != len(reg["domains"]):
        c.err("ids", "معرّفات مجالات مكررة")

    for t in reg["topics"]:
        if t["domain_id"] not in domains:
            c.err("domains", f'{t["id"]}: مجال غير معروف {t["domain_id"]}')
        if t["id"].split("-")[0] != t["domain_id"]:
            c.err("domains", f'{t["id"]}: بادئة المعرّف لا تطابق domain_id')

    seen_in_group = defaultdict(list)
    for g in reg["domain_groups"]:
        for d in g["domain_ids"]:
            seen_in_group[d].append(g["id"])
            if d not in domains:
                c.err("groups", f"{g['id']}: مجال غير موجود {d}")
    for d in domains:
        if d not in seen_in_group:
            c.err("groups", f"المجال {d} غير ممثل في أي مجموعة")
        elif len(seen_in_group[d]) > 1:
            c.err("groups", f"المجال {d} في أكثر من مجموعة: {seen_in_group[d]}")
    for d in reg["domains"]:
        if d["group_id"] not in groups:
            c.err("groups", f'{d["id"]}: group_id غير معروف')
        elif d["id"] not in groups[d["group_id"]]["domain_ids"]:
            c.err("groups", f'{d["id"]}: group_id لا يطابق عضوية المجموعة')

    forbidden = {"prerequisites", "related", "topic_count", "counts"}
    for t in reg["topics"]:
        bad = forbidden & set(t)
        if bad:
            c.err("single_source", f'{t["id"]}: حقول ممنوعة (نسخة من العلاقات) {sorted(bad)}')
    for d in reg["domains"]:
        if "topic_count" in d:
            c.err("single_source", f'المجال {d["id"]}: topic_count مخزَّن بدل اشتقاقه')

    check_graph(c, reg, topics)
    used_in, reused = check_journeys(c, jrn, topics)
    check_station_semantics(c, jrn)
    semantic_guard_negative_selftest(c, jrn)
    check_language_parity(c, jrn)
    parity_negative_selftest(c, jrn)
    check_contracts(c, con, topics, {j["id"]: j for j in jrn["journeys"]})

    hist_ids = [h["id"] for h in reg["id_history"]]
    if len(hist_ids) != len(set(hist_ids)):
        c.err("history", "معرّف مكرر في id_history")
    merged = {h["id"] for h in reg["id_history"] if h["state"] == "merged"}
    for h in reg["id_history"]:
        if h["state"] == "merged":
            if h["id"] in topics:
                c.err("history", f'{h["id"]}: مدموج لكنه لا يزال موضوعاً قائماً')
            if h.get("merged_into") not in topics:
                c.err("history", f'{h["id"]}: merged_into غير موجود')
            if not h.get("rationale_ar"):
                c.err("history", f'{h["id"]}: دمج بلا تبرير')
        if h["state"] == "renamed" and h["id"] not in topics:
            c.err("history", f'{h["id"]}: موسوم renamed لكنه غير موجود')
    for tid in topics:
        if tid in merged:
            c.err("history", f"{tid}: معرّف مدموج أُعيد استخدامه")

    per_domain = Counter(t["domain_id"] for t in reg["topics"])
    coverage = Counter(t["coverage"] for t in reg["topics"])
    deg = Counter()
    for r in reg["relations"]:
        deg[r["source"]] += 1
        deg[r["target"]] += 1
    isolated = [i for i in topics if deg[i] == 0]
    flagship = [i for i, t in topics.items() if t["coverage"] == "flagship_journey"]
    frontier = [i for i, t in topics.items() if t["coverage"] == "research_frontier"]
    station_topics = {s["topic_id"] for j in jrn["journeys"] for s in j["stations"]}
    isolated_stations = sorted(station_topics & set(isolated))
    review_counts = Counter(r.get("review") for r in reg["relations"] if r.get("added_in"))

    report = {
        "valid": not c.errors,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_fingerprint": fp,
        "inputs": {k: Path(v).name for k, v in files.items()},
        "schema_version": reg["schema"],
        "counts": {
            "domains": len(domains),
            "groups": len(groups),
            "topics": len(topics),
            "relations": len(reg["relations"]),
            "relation_types": len(reg["relation_types"]),
            "coverage": dict(coverage),
            "journeys": len(jrn["journeys"]),
            "stations": sum(len(j["stations"]) for j in jrn["journeys"]),
            "topics_in_journeys": len(station_topics),
            "entry_contracts": len(con["contracts"]),
            "added_edges_by_review": dict(review_counts),
        },
        "per_domain": dict(per_domain),
        "graph": {
            "isolated_topics": len(isolated),
            "isolated_ratio": round(len(isolated) / len(topics), 4),
            "flagship_total": len(flagship),
            "flagship_candidates_isolated": len([i for i in flagship if deg[i] == 0]),
            "isolated_stations": isolated_stations,
            "frontier_total": len(frontier),
            "frontier_isolated": len([i for i in frontier if deg[i] == 0]),
            "max_degree": max(deg.values()) if deg else 0,
        },
        "journeys": {
            "cross_journey_reuse_allowed": True,
            "reused_topics": reused,
        },
        "errors": c.errors,
        "warnings": c.warnings,
    }

    rp = Path(a.report)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    if not a.quiet:
        print(json.dumps({k: report[k] for k in ("valid", "input_fingerprint", "counts", "graph")},
                         ensure_ascii=False, indent=1))
        if c.warnings:
            print(f"\nتحذيرات ({len(c.warnings)}):")
            for w in c.warnings:
                print("  -", w["check"], ":", w["message"])
        if c.errors:
            print(f"\nأخطاء ({len(c.errors)}):")
            for e in c.errors:
                print("  -", e["check"], ":", e["message"])
        else:
            print("\nكل الفحوص نجحت.")
    return 0 if not c.errors else 1


if __name__ == "__main__":
    sys.exit(main())
