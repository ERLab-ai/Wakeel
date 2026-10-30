#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_v02.py — يحوّل سجل المعرفة v0.1 إلى v0.2 بقرارات تحكيم معلنة.

مبادئ البناء:
  * `relations` هي المصدر الوحيد للحقيقة. لا تُخزَّن نسخة داخل الموضوع.
  * `topic_count` لا يُخزَّن؛ يُشتق في المدقق وفي مولد الأطلس.
  * المعرّفات لا تُحذف من التاريخ: الموضوع المدمج يبقى في `id_history` بحالة merged.
  * أنواع العلاقات من مفردة واحدة موثقة في `relation_types`.

كل قرار تحكيم مكتوب هنا صراحة، لا مخفياً في تعديل يدوي على الملف.
"""
import json
import sys
from pathlib import Path

from common import write_json

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "inputs" / "WAKEEL-2-KNOWLEDGE-REGISTRY-v0.1.json"
REVIEW = ROOT / "data" / "inputs" / "WAKEEL-2-V02-EDGE-REVIEW.json"
OUT = ROOT / "data" / "WAKEEL-2-KNOWLEDGE-REGISTRY-v0.3.json"
REPORT = ROOT / "out" / "build-report.json"
DATE = "2026-10-25"

# ---------------------------------------------------------------- المجموعات العليا
GROUPS = [
    ("GRP-FOUND", "الأسس", "Foundations", ["FND", "MAT", "DAT", "HIS"]),
    ("GRP-METHODS", "مناهج الذكاء", "AI Methods", ["SYM", "ML", "DLN", "RLC"]),
    ("GRP-MODELS", "النماذج والتوليد", "Models and Generation",
     ["NLP", "LLM", "TRN", "INF", "CVG", "AUD", "MMU"]),
    ("GRP-SYSTEMS", "الأنظمة والفعل", "Systems and Action", ["AGT", "ROB", "SYS"]),
    ("GRP-QUALITY", "الجودة والبحث", "Quality, Society and Research",
     ["EVA", "HUM", "RES", "APP"]),
]

# ---------------------------------------------------------------- مفردة أنواع العلاقات
RELATION_TYPES = {
    "motivates": {
        "title_ar": "يحرّك", "direction_ar": "من المشكلة/الحاجة إلى السؤال الذي تُنتجه",
        "symmetric": False, "is_dependency": False,
        "example_ar": "الحاجة إلى التعامل مع اللغة ← التعلم من أمثلة ← النمذجة اللغوية",
    },
    "reveals_limit": {
        "title_ar": "يكشف حداً", "direction_ar": "من حل إلى السؤال الذي يظهر عند فشله",
        "symmetric": False, "is_dependency": False,
        "example_ar": "Bigram ← محدودية السياق ← نموذج يستخدم سياقاً أطول",
    },
    "hard_implementation": {
        "title_ar": "شرط تنفيذ", "direction_ar": "من المتطلب إلى ما يتطلبه عند التنفيذ فقط",
        "symmetric": False, "is_dependency": True,
        "example_ar": "المشتقات ← الانتشار العكسي عند برمجته عددياً",
    },
    "just_in_time": {
        "title_ar": "تمهيد موجز", "direction_ar": "من المفهوم المساعد إلى الموضوع الذي يحتاجه",
        "symmetric": False, "is_dependency": True,
        "example_ar": "معرّف الرمز ← نشاط التضمين",
    },
    "contrast": {
        "title_ar": "تقابل", "direction_ar": "من نهج إلى نهج بديل يصلح للمقارنة",
        "symmetric": True, "is_dependency": False,
        "example_ar": "القواعد المكتوبة ↔ التعلم من الأمثلة",
    },
    "related": {
        "title_ar": "مرتبط", "direction_ar": "ارتباط معرفي بلا ترتيب مفروض",
        "symmetric": True, "is_dependency": False,
        "example_ar": "التحليل الصرفي ↔ التقطيع تحت الكلمة",
    },
    "historical_context": {
        "title_ar": "سياق تاريخي", "direction_ar": "من اللاحق إلى محطته التاريخية",
        "symmetric": False, "is_dependency": False,
        "example_ar": "Transformer ← مراحل تطور الانتباه",
    },
    "explains": {
        "title_ar": "يفسّر", "direction_ar": "من الأعمق تفسيراً إلى الظاهرة التي يفسرها",
        "symmetric": False, "is_dependency": False,
        "example_ar": "الاحتمالات ← توزيع التنبؤ والتوليد",
    },
    "clarifies": {
        "title_ar": "يوضّح تمييزاً", "direction_ar": "من المفهوم إلى المفاهيم المتشابهة التي يميّزها",
        "symmetric": True, "is_dependency": False,
        "example_ar": "البرنامج ↔ النموذج ↔ البيانات",
    },
    "specializes": {
        "title_ar": "تخصيص", "direction_ar": "من الحالة الخاصة إلى المفهوم العام",
        "symmetric": False, "is_dependency": False,
        "example_ar": "مثال عدائي ← اختبار المتانة",
    },
}

# ---------------------------------------------------------------- قرارات الدمج
MERGES = {
    "AGT-018": {
        "into": "EVA-011",
        "case": 1,
        "rationale_ar": "المفهوم واحد: نص غير موثوق يتجاوز حدود التعليمات. الاختلاف في سياق التطبيق (وكيل/استرجاع) لا في المسألة العلمية.",
        "evidence_ar": "السؤالان يصفان الآلية نفسها، وكانت بينهما حافة just_in_time أي أن النموذج نفسه ربط الموضوع بنسخته.",
    },
    "TRN-018": {
        "into": "EVA-003",
        "case": 2,
        "rationale_ar": "تلوث المقاييس ظاهرة قياس واحدة. موضعها الطبيعي في مجال التقييم، ومنشأها في التدريب يُشرح داخل الموضوع.",
        "evidence_ar": "EVA-003 يضم التلوث أصلاً في عنوانه وسؤاله، وTRN-018 يعيد الحالة نفسها من جهة التدريب.",
    },
    "RES-014": {
        "into": "SYM-015",
        "case": 3,
        "rationale_ar": "الرمزي العصبي برنامج واحد؛ وجود نسخة في مجال البحث لا يضيف مسألة، بل يضيف زاوية نشر.",
        "evidence_ar": "العنوانان مترادفان تقريباً، والسؤالان يطلبان الشيء نفسه بصياغتين.",
    },
    "RES-015": {
        "into": "LLM-020",
        "case": 4,
        "rationale_ar": "سؤال واحد: هل تكون وحدة التنبؤ غير رمزية؟ الإطار البحثي يوصف داخل الموضوع نفسه.",
        "evidence_ar": "«وحدة التنبؤ إذا لم تكن رمزاً» و«حالة أو مفهوم لا رمز» سؤال واحد بلفظين.",
    },
    "ROB-013": {
        "into": "MMU-013",
        "case": 6,
        "rationale_ar": "VLA بنية متعددة الوسائط تُطبَّق في الروبوتات. الروبوت تطبيق لا مفهوم مستقل.",
        "evidence_ar": "العنوانان يتطابقان مصطلحياً، والسؤالان يصفان الانتقال من الإدراك إلى الفعل.",
    },
}

# ---------------------------------------------------------------- إعادة تسمية وضبط حدود
RENAMES = {
    "EVA-011": {
        "title_ar": "حقن المطالبات وحدود الثقة",
        "title_en": "Prompt injection and trust boundaries",
        "guiding_question_ar": "كيف يهاجم نص غير موثوق حدود تعليمات النظام، ومتى يصل أثره إلى فعّل الوكيل؟",
    },
    "MMU-013": {
        "title_ar": "نماذج الرؤية واللغة والفعل",
        "title_en": "Vision-language-action models",
        "guiding_question_ar": "كيف تتحول الملاحظات متعددة الوسائط إلى سياسة فعل قابلة للتنفيذ؟",
    },
    "NLP-008": {
        "title_ar": "تحديات معالجة العربية",
        "title_en": "Arabic NLP challenges",
        "guiding_question_ar": "كيف تؤثر اللواصق والتشكيل واختلاف الكتابة واللهجات في المعالجة؟",
    },
    "INF-012": {
        "title_ar": "التشغيل المحلي مقابل الاستضافة",
        "title_en": "Local versus hosted inference",
        "guiding_question_ar": "متى نختار تشغيل النموذج محلياً، وما الذي ندفعه ثمناً لذلك؟",
    },
    "SYS-015": {
        "title_ar": "مفاتيح المزود وحدود الاعتماد (BYOK)",
        "title_en": "Provider keys and BYOK trust",
        "guiding_question_ar": "من يملك المفتاح، وما الذي يتعرض له عند الاعتماد على مزود خارجي؟",
    },
    "LLM-020": {
        "title_ar": "وحدات التنبؤ غير الرمزية",
        "title_en": "Non-token prediction units: latent and concept models",
        "guiding_question_ar": "متى تكون وحدة التنبؤ حالة كامنة أو مفهوماً بدل رمز نصي، وهل يلزم أن يكون التوليد تسلسلياً؟",
    },
}

# ---------------------------------------------------------------- موضوع جديد نتيجة التفكيك
NEW_TOPICS = [
    {
        "id": "SYS-018",
        "domain_id": "SYS",
        "title_ar": "التصميم دون اتصال والتدهور الآمن",
        "title_en": "Offline-first behaviour and graceful degradation",
        "guiding_question_ar": "كيف يظل المنتج صالحاً للاستخدام حين تنقطع الخدمة أو يتعذر الوصول إلى المزود؟",
        "coverage": "shared_foundation",
        "rationale_ar": "كانت مدمجة في SYS-015 مع BYOK؛ وهما مسألتان مختلفتان: ملكية المفتاح مقابل سلوك النظام عند فقد الاتصال.",
    },
]

# ---------------------------------------------------------------- علاقات التحكيم الجديدة
NEW_RELATIONS = [
    ("EVA-013", "EVA-005", "specializes", "الحالة العدائية تخصيص لاختبار المتانة الأوسع"),
    ("DAT-007", "NLP-008", "specializes", "التطبيع إجراء بيانات داخل تحديات العربية الأوسع"),
    ("NLP-016", "EVA-008", "specializes", "تقييم العربية حالة خاصة من التقييم عبر اللغات"),
    ("AUD-013", "EVA-008", "specializes", "تقييم الكلام العربي حالة خاصة من التقييم عبر اللغات"),
    ("DLN-021", "ML-019", "specializes", "GNN حالة خاصة من التعلم على الرسوم"),
    ("DLN-006", "TRN-001", "hard_implementation", "صياغة هدف لمهمة تحتاج مفهوم الخسارة والهدف"),
    ("DLN-006", "TRN-001", "related", "المفهوم النظري وربطه بالإجراء الهندسي"),
    ("HUM-015", "DAT-013", "related", "بطاقة النموذج وبطاقة البيانات: كيانان مختلفان بوظيفة توثيق واحدة"),
    ("MAT-017", "EVA-002", "hard_implementation", "حساب Perplexity يحتاج الانتروبيا المتقاطعة"),
    # --- وصل الجبهات البحثية بجارها العلمي المباشر فقط (بلا علاقات اعتماد) ---
    ("RES-013", "RLC-010", "related", "نموذج العالم هو الأساس الذي يستخدمه التعلم المعتمد على النموذج"),
    ("LLM-019", "LLM-002", "contrast", "توليد غير تلقائي الانحدار مقابل توليد رمزاً برمز"),
    ("LLM-020", "LLM-002", "contrast", "وحدة تنبؤ كامنة/مفاهيمية مقابل وحدة رمزية"),
    ("MMU-010", "MMU-001", "related", "النموذج الموحد حالة متقدمة من دمج الوسائط"),
    ("MMU-013", "ROB-001", "related", "سياسة الرؤية واللغة والفعل داخل نظام مجسد"),
    ("EVA-020", "EVA-009", "specializes", "التفسير الآلي الداخلي منهج داخل التفسير الأوسع"),
    ("DLN-020", "ML-020", "related", "التعلم المستمر والتحديث عند وصول بيانات جديدة: سياقان مختلفان للاحتفاظ بالمعرفة"),
    ("DLN-023", "ML-016", "related", "التعلم الفوقي والنقل: كلاهما استثمار معرفة سابقة في مهمة جديدة"),
    ("SYM-015", "SYM-007", "contrast", "الاستدلال بالقواعد المكتوبة مقابل الاستدلال بتعلم التمثيلات مع قيود"),
    ("RES-012", "RES-017", "related", "البحث الآلي عن البنى والاكتشاف الآلي: أتمتة البحث مقابل أتمتة النتائج"),
    ("RES-016", "EVA-001", "specializes", "تقييم ادعاءات العمومية حالة من الادعاء القابل للفحص"),
    ("RES-011", "RES-010", "related", "الاستدلال وقت الاختبار محور توسع موازٍ لقوانين توسع التدريب"),
    # --- وصل موضوعات الرحلة المعزولة بجارها التعريفي المباشر (مرشحة لمراجعة بشرية) ---
    ("MAT-001", "MAT-007", "hard_implementation", "المشتقة تُعرَّف على دالة"),
    ("MAT-004", "MAT-005", "hard_implementation", "الضرب الداخلي عملية على متجهات"),
    ("MAT-015", "MAT-017", "hard_implementation", "الانتروبيا المتقاطعة تُحسب بلوغاريتم"),
    ("MAT-016", "MAT-017", "hard_implementation", "الانتروبيا أصل مقياس الخسارة"),
    ("MAT-003", "LLM-002", "just_in_time", "التسلسل كترتيب قبل الحديث عن توليد متتابع"),
    ("DLN-001", "DLN-003", "hard_implementation", "التنشيط يُطبَّق على مخرج العصبون"),
    ("DLN-010", "TRN-004", "hard_implementation", "حلقة التدريب تعمل بدفعات وخطوات"),
    ("DLN-011", "DLN-018", "hard_implementation", "بنية Transformer تستخدم التطبيع والروابط المتبقية"),
    ("DLN-014", "DLN-018", "hard_implementation", "بنية Transformer تحتاج تمثيلاً لترتيب المواضع"),
    ("DLN-016", "LLM-012", "hard_implementation", "نمذجة RNN اللغوية تحتاج شبكة متكررة"),
    ("LLM-007", "LLM-006", "specializes", "Bigram حالة خاصة من N-gram عند n=2"),
    ("HIS-009", "LLM-006", "historical_context", "تاريخ النماذج قبل Transformer يمر بالنماذج الإحصائية"),
    ("DAT-004", "TRN-002", "hard_implementation", "تجهيز المدونة يبدأ بالتنظيف والتحقق"),
    ("FND-006", "FND-010", "hard_implementation", "برمجة بايثون تحتاج الشروط والحلقات والدوال"),
    ("NLP-007", "NLP-003", "related", "فك الترميز الوجه الآخر لبناء المقطع"),
    ("FND-014", "AGT-010", "clarifies", "تمييز النموذج/النظام يوازي تمييز النموذج/الوكيل/المنظومة"),
    ("EVA-007", "EVA-001", "related", "الادعاء غير المسند من أنواع الادعاء غير القابلة للفحص"),
]

SOURCING_POLICY = {
    "current_state_ar": "لا موضوع مرتبط بمصدر فردي في هذا الإصدار. الحقل موجود فارغاً ومعه حالة صريحة، ولا يُملأ بمصادر مُختلقة.",
    "required_before_publication_ar": "لكل موضوع مصدر مؤسس واحد على الأقل، وتاريخ للادعاء إن كان بحثياً.",
    "frontier_rule_ar": "موضوعات research_frontier لا تُعرض كحقائق نهائية؛ يلزمها مصدر وتاريخ وقيود معلنة.",
    "fields": {"sources": "قائمة مصادر فردية (فارغة الآن)", "source_status": "unassigned | partial | verified"},
}


def main():
    src = json.loads(SRC.read_text(encoding="utf-8"))
    topics = src["topics"]
    relations = src["relations"]

    before = {
        "topics": len(topics),
        "relations": len(relations),
        "coverage": {},
    }
    for t in topics:
        before["coverage"][t["coverage"]] = before["coverage"].get(t["coverage"], 0) + 1

    # فحص الازدواج قبل الإزالة: يجب أن يطابق المصدر المركزي تماماً
    norm = lambda s: "hard_implementation" if s == "hard_for_implementation" else s
    central = {(r["source"], r["target"], norm(r["type"])) for r in relations}
    mirrored = set()
    for t in topics:
        for p in t.get("prerequisites", []):
            mirrored.add((p["topic_id"], t["id"], norm(p["type"])))
        for r in t.get("related", []):
            mirrored.add((t["id"], r["topic_id"], norm(r["relation"])))
    assert central == mirrored, "النسخة الموزعة لا تطابق المصدر المركزي — التحويل غير آمن"

    # 1) العلاقات: إعادة توجيه المدمج، وإسقاط ما يصبح حلقة ذاتية
    redirects, dropped_self = [], []
    new_relations = []
    for r in relations:
        s = MERGES.get(r["source"], {}).get("into", r["source"])
        d = MERGES.get(r["target"], {}).get("into", r["target"])
        if s != r["source"] or d != r["target"]:
            redirects.append({"from": r, "to": [s, d, r["type"]]})
        if s == d:
            dropped_self.append({"original": r, "reason_ar": "طرفا العلاقة صارا الموضوع نفسه بعد الدمج"})
            continue
        new_relations.append({"source": s, "target": d, "type": r["type"]})

    # 2) تطبيق قرارات المراجعة المستقلة (26 اعتماد، 10 تعديل، 1 حذف، 1 تحكيم)
    review = json.loads(REVIEW.read_text(encoding="utf-8"))
    decisions = {}
    for dec in review["decisions"]:
        e = dec["edge"]
        decisions[(e["source"], e["target"], e["type"])] = dec

    existing = {(r["source"], r["target"], r["type"]) for r in new_relations}
    audit = {"accepted": [], "edited": [], "removed": [], "provisional": [], "unmatched": [], "collisions": []}
    for s, d, typ, why in NEW_RELATIONS:
        s = MERGES.get(s, {}).get("into", s)
        d = MERGES.get(d, {}).get("into", d)
        dec = decisions.get((s, d, typ))
        if dec is None:
            audit["unmatched"].append([s, d, typ])
            continue
        st = dec["status"]
        if st == "remove":
            audit["removed"].append({"edge": [s, d, typ], "reason_ar": dec["reason_ar"]})
            continue
        if st == "edit":
            p = dec["proposed_edge"]
            ns, nd, ntyp = p["source"], p["target"], p["type"]
            rel = {"source": ns, "target": nd, "type": ntyp, "added_in": "v0.2",
                   "rationale_ar": p.get("rationale_ar", why), "review": "edited",
                   "was_type": typ, "review_reason_ar": dec["reason_ar"]}
            audit["edited"].append({"from": [s, d, typ], "to": [ns, nd, ntyp]})
        elif st == "review":
            ns, nd, ntyp = s, d, typ
            rel = {"source": ns, "target": nd, "type": ntyp, "added_in": "v0.2",
                   "rationale_ar": why, "review": "provisional", "review_note_ar": dec["reason_ar"]}
            audit["provisional"].append([ns, nd, ntyp])
        elif st == "accept":
            ns, nd, ntyp = s, d, typ
            rel = {"source": ns, "target": nd, "type": ntyp, "added_in": "v0.2",
                   "rationale_ar": why, "review": "accepted", "review_reason_ar": dec["reason_ar"]}
            audit["accepted"].append([ns, nd, ntyp])
        else:
            raise SystemExit(f"حالة مراجعة غير معروفة: {st}")
        key = (ns, nd, ntyp)
        if key in existing:
            audit["collisions"].append([ns, nd, ntyp])
            continue
        new_relations.append(rel)
        existing.add(key)

    got = {k: len(audit[k]) for k in ("accepted", "edited", "removed", "provisional")}
    expected = {
        "accepted": review["counts"]["accept"],
        "edited": review["counts"]["edit"],
        "removed": review["counts"]["remove"],
        "provisional": review["counts"]["review"],
    }
    if got != expected:
        raise SystemExit(f"عدد قرارات المراجعة لا يطابق الملف: {got} != {expected}")
    if audit["unmatched"]:
        raise SystemExit(f"حواف مضافة بلا قرار مراجعة: {audit['unmatched']}")

    # 2) الموضوعات
    out_topics = []
    for t in topics:
        if t["id"] in MERGES:
            continue
        item = {
            "id": t["id"],
            "domain_id": t["domain_id"],
            "title_ar": t["title_ar"],
            "title_en": t["title_en"],
            "guiding_question_ar": t["guiding_question_ar"],
            "coverage": t["coverage"],
            "independent_entry": True,
            "status": "inventory_proposal",
            "source_status": "unassigned",
            "sources": [],
        }
        if t["id"] in RENAMES:
            item.update(RENAMES[t["id"]])
            item["renamed_in"] = "v0.2"
        out_topics.append(item)

    for nt in NEW_TOPICS:
        item = {k: v for k, v in nt.items() if k != "rationale_ar"}
        item.update({"independent_entry": True, "status": "inventory_proposal",
                     "source_status": "unassigned", "sources": [], "added_in": "v0.2"})
        out_topics.append(item)

    # 3) المجالات: بلا topic_count، ومع group_id
    group_of = {d: g[0] for g in GROUPS for d in g[3]}
    out_domains = []
    for dm in src["domains"]:
        out_domains.append({
            "id": dm["id"],
            "group_id": group_of[dm["id"]],
            "title_ar": dm["title_ar"],
            "title_en": dm["title_en"],
        })

    # 4) سجل المعرّفات
    id_history = []
    for old, m in MERGES.items():
        src_topic = next(t for t in topics if t["id"] == old)
        id_history.append({
            "id": old,
            "state": "merged",
            "merged_into": m["into"],
            "previous_titles": {"title_ar": src_topic["title_ar"], "title_en": src_topic["title_en"]},
            "previous_coverage": src_topic["coverage"],
            "case": m["case"],
            "rationale_ar": m["rationale_ar"],
            "evidence_ar": m["evidence_ar"],
            "decided": DATE,
        })
    for old, r in RENAMES.items():
        src_topic = next(t for t in topics if t["id"] == old)
        id_history.append({
            "id": old, "state": "renamed", "decided": DATE,
            "previous_titles": {"title_ar": src_topic["title_ar"], "title_en": src_topic["title_en"]},
            "previous_question_ar": src_topic["guiding_question_ar"],
        })
    for nt in NEW_TOPICS:
        id_history.append({"id": nt["id"], "state": "added", "decided": DATE,
                           "rationale_ar": nt["rationale_ar"]})

    registry = {
        "schema": "wakeel.knowledge-registry.v0.3",
        "schema_uri": "schema/wakeel-registry.schema.json",
        "date": DATE,
        "status": "PLANNING_DRAFT_NOT_IMPLEMENTATION",
        "supersedes": "wakeel.knowledge-registry.v0.2",
        "revision": "r2-review-applied",
        "scope": src["scope"],
        "fundamental_rule": src["fundamental_rule"],
        "single_source_of_truth": {
            "rule_ar": "العلاقات تُخزَّن في relations وحدها. لا تُنسخ داخل الموضوع. ما يظهر في صفحة الموضوع مشتق عند القراءة.",
            "reason_ar": "v0.1 كان يخزن كل علاقة مرتين (prerequisites وrelated) بفردتين مختلفتين لاسم النوع، فكان الانحراف الصامت ممكناً.",
            "derived_at_read_time": ["prerequisites(topic)", "related(topic)", "topic_count(domain)", "coverage_counts"],
        },
        "domain_groups": [
            {"id": gid, "title_ar": gar, "title_en": gen, "domain_ids": ids}
            for gid, gar, gen, ids in GROUPS
        ],
        "domains": out_domains,
        "relation_types": RELATION_TYPES,
        "relations": new_relations,
        "edge_review": {
            "source": "data/inputs/WAKEEL-2-V02-EDGE-REVIEW.json",
            "scope_ar": "مراجعة مستقلة للحواف المضافة في v0.2؛ لا تُعد تصديقاً بمصدر علمي فردي.",
            "counts": {k: review["counts"][k] for k in ("accept", "edit", "remove", "review")},
            "applied": {k: len(audit[k]) for k in ("accepted", "edited", "removed", "provisional")},
            "provisional_edges": audit["provisional"],
        },
        "topics": out_topics,
        "id_history": id_history,
        "sourcing_policy": SOURCING_POLICY,
        "reference_basis": src["reference_basis"],
        "constraints": src["constraints"],
        "notes": [
            "topic_count يُشتق ولا يُخزَّن؛ المدقق يتحقق من تطابق أي مجموع معلن في الوثائق.",
            "العلاقات تصف المعرفة العلمية فقط؛ ترتيب الرحلات يعيش في wakeel.journey-graph.v0.2 ولا يُدمج هنا.",
            "الحقول prerequisites/related أُزيلت من الموضوعات؛ الأداة tools/validate.py تطبع العرض المشتق للتحقق.",
            "المعرّفات المدموجة محفوظة في id_history ولا تُعاد للاستخدام.",
            "كل حافة مضافة في v0.2 تحمل حقل review: accepted أو edited أو provisional، مع سبب المراجعة.",
            "الحواف provisional غير معتمدة للمحتوى؛ لا تُعرض كحقيقة في الدروس قبل حسمها.",
        ],
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    write_json(OUT, registry)

    after = {
        "topics": len(out_topics),
        "relations": len(new_relations),
        "coverage": {},
        "domains": len(out_domains),
    }
    for t in out_topics:
        after["coverage"][t["coverage"]] = after["coverage"].get(t["coverage"], 0) + 1

    report = {
        "before": before,
        "after": after,
        "merge_redirects": redirects,
        "dropped_self_loops": dropped_self,
        "edge_review": audit,
        "output": OUT.relative_to(ROOT).as_posix(),
    }
    write_json(REPORT, report)

    print(json.dumps({"before": before, "after": after}, ensure_ascii=False, indent=1))
    print("redirects:", len(redirects), "dropped self-loops:", len(dropped_self))
    print("written:", OUT)


if __name__ == "__main__":
    sys.exit(main())
