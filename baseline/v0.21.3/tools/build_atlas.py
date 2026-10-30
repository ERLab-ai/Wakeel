#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_atlas.py — يولّد مستند الأطلس من السجل والرحلة والعقود.

قبل التوليد: يُشغّل المدقق على نفس المدخلات، ويرفض التوليد إن فشل الفحص أو إن
كان تقرير التحقق لا يخص بصمة المدخلات الحالية. لا وثيقة تصف مدخلات متغيرة
بشارة تحقق قديمة.
"""
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

from common import fingerprint, write_text_lf

ROOT = Path(__file__).resolve().parents[1]
REG = ROOT / "data" / "WAKEEL-2-KNOWLEDGE-REGISTRY-v0.3.json"
JRN = ROOT / "data" / "WAKEEL-2-JOURNEYS-v0.3.json"
CON = ROOT / "data" / "WAKEEL-2-ENTRY-CONTRACTS-v0.3.json"
VAL = ROOT / "out" / "validation-report.json"
BLD = ROOT / "out" / "build-report.json"
TPL = ROOT / "templates" / "atlas.tpl.md"
OUT = ROOT / "out" / "WAKEEL-2-KNOWLEDGE-ATLAS-v0.7.md"
VALIDATOR = ROOT / "tools" / "validate.py"

COV_LABEL = {
    "flagship_journey": "مرشحة للرحلة الأولى",
    "shared_foundation": "تأسيسية مشتركة",
    "expansion": "توسع في الفروع",
    "research_frontier": "جبهة بحثية",
}
ROLE_LABEL = {"core": "أساسية", "optional": "اختيارية", "reference": "مرجعية"}
REVIEW_LABEL = {"accepted": "معتمدة", "edited": "معدّلة", "provisional": "معلّقة"}


def cell(s):
    return str(s).replace("|", "\\|").replace("\n", " ").strip()


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(cell(x) for x in r) + " |")
    return "\n".join(out)


def components(topic_ids, relations):
    parent = {i: i for i in topic_ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for r in relations:
        a, b = find(r["source"]), find(r["target"])
        if a != b:
            parent[a] = b
    return Counter(find(i) for i in topic_ids)


def run_validator():
    proc = subprocess.run([sys.executable, str(VALIDATOR), "--report", str(VAL),
                           "--registry", str(REG), "--journeys", str(JRN),
                           "--contracts", str(CON), "--quiet"],
                          capture_output=True, text=True, encoding="utf-8")
    if proc.returncode != 0:
        print("رفض التوليد: فشل الفحص البنيوي.")
        print(proc.stdout or "", proc.stderr or "")
        return None
    return json.loads(VAL.read_text(encoding="utf-8"))


def main():
    val = run_validator()
    if val is None:
        return 1
    fp_now = fingerprint({"registry": REG, "journeys": JRN, "contracts": CON})
    if val.get("input_fingerprint") != fp_now:
        print("رفض التوليد: تقرير التحقق لا يخص المدخلات الحالية.")
        return 1

    reg = json.loads(REG.read_text(encoding="utf-8"))
    jrn = json.loads(JRN.read_text(encoding="utf-8"))
    con = json.loads(CON.read_text(encoding="utf-8"))
    bld = json.loads(BLD.read_text(encoding="utf-8")) if BLD.exists() else {}

    topics = {t["id"]: t for t in reg["topics"]}
    domains = {d["id"]: d for d in reg["domains"]}
    groups = reg["domain_groups"]
    per_domain = Counter(t["domain_id"] for t in reg["topics"])
    coverage = Counter(t["coverage"] for t in reg["topics"])
    rtypes = reg["relation_types"]
    rel = reg["relations"]
    total = len(topics)

    # ---------------------------------------------------------------- 1) التغييرات
    hist = reg["id_history"]
    merges = [h for h in hist if h["state"] == "merged"]
    renames = [h for h in hist if h["state"] == "renamed"]
    added = [h for h in hist if h["state"] == "added"]
    added_rel = [r for r in rel if r.get("added_in") == "v0.2"]
    ac = Counter(r.get("review") for r in added_rel)
    changes = table(
        ["التغيير", "العدد", "الأثر"],
        [
            ["مصدر واحد للحقيقة", "كل الموضوعات", "حُذف `prerequisites` و`related` من كل موضوع؛ لا نسخة محلية للعلاقات"],
            ["توحيد أنواع العلاقات", f"{len(rtypes)} أنواع", "`hard_for_implementation` أُلغي"],
            ["دمج موضوعات مكررة", len(merges), "المعرّفات محفوظة في `id_history`"],
            ["إعادة تسمية وضبط حدود", len(renames), "منها `LLM-020` بعد مراجعة نطاقه"],
            ["تفكيك مفهوم مختلط", len(added), "`SYS-018` منفصل عن BYOK"],
            ["مجموعات المجالات", len(groups), "في البيانات لا في المحادثة"],
            ["اشتقاق `topic_count`", "22 مجالاً", "لا يُخزَّن"],
            ["حواف مضافة", len(added_rel), f"معتمدة {ac.get('accepted', 0)}، معدّلة {ac.get('edited', 0)}، معلّقة {ac.get('provisional', 0)}"],
            ["فصول الرحلة", sum(len(j.get("chapters", [])) for j in jrn["journeys"]), "تجمع المحطات في وحدات تعلم"],
            ["أنشطة المحطات", sum(1 for j in jrn["journeys"] for s in j["stations"] if s.get("activity")), "يرى/يجرب/فرضية المتعلم/نتائج محتملة/تفسير/يكتشف"],
            ["عقود دخول مستقل", len(con["contracts"]), "مع مخطط JSON خاص بها"],
            ["فصول مؤلَّفة", sum(1 for j in jrn["journeys"] for ch in j["chapters"] if (ch.get("content") or {}).get("status") in ("authored", "reviewed")), "الفصول الستة بنسختيهما، مع بيانات ونتائج محسوبة"],
        ],
    )

    # ---------------------------------------------------------------- 2) التغطية
    cov_rows = []
    for key in ("flagship_journey", "shared_foundation", "expansion", "research_frontier"):
        n = coverage.get(key, 0)
        cov_rows.append([f"`{key}`", COV_LABEL[key], n, f"{n / total:.1%}"])
    cov_rows.append(["**المجموع**", "", f"**{total}**", "100%"])

    # ---------------------------------------------------------------- 3) خريطة المجالات
    blocks = []
    for g in groups:
        blocks.append(f"### {g['title_ar']} / {g['title_en']}\n")
        blocks.append(table(["الرمز", "المجال", "English", "الموضوعات"],
                            [[f"`{d}`", domains[d]["title_ar"], domains[d]["title_en"], per_domain[d]]
                             for d in g["domain_ids"]]))
        blocks.append("")
        for d in g["domain_ids"]:
            dm = domains[d]
            sub = [t for t in reg["topics"] if t["domain_id"] == d]
            blocks.append(f"#### {d} — {dm['title_ar']} / {dm['title_en']} ({len(sub)} موضوعاً)\n")
            blocks.append(table(
                ["المعرّف", "الموضوع", "السؤال الذي يبدأ منه التعلم", "التغطية"],
                [[f"`{t['id']}`", f"**{t['title_ar']}** / {t['title_en']}",
                  t["guiding_question_ar"], f"`{t['coverage']}`"] for t in sub]))
            blocks.append("")
    domain_map = "\n".join(blocks)

    # ---------------------------------------------------------------- 4) العلاقات
    type_rows = []
    for k, v in rtypes.items():
        nature = "اعتماد" if v["is_dependency"] else ("متناظر" if v["symmetric"] else "دلالي")
        type_rows.append([f"`{k}`", v["title_ar"], v["direction_ar"], nature, v.get("example_ar", "")])
    type_table = table(["النوع", "المعنى", "الاتجاه", "الطبيعة", "مثال"], type_rows)

    by_type = Counter(r["type"] for r in rel)
    dist = "، ".join(f"`{k}` {v}" for k, v in by_type.most_common())
    comps = components(list(topics), rel)
    stats = val.get("graph", {})
    graph_stats = table(
        ["مؤشر الشبكة", "القيمة", "كيف يُقرأ"],
        [
            ["إجمالي العلاقات", len(rel), "منها حواف مضافة في v0.2 بمراجعة صريحة"],
            ["توزيعها", dist, "الاعتماد أكثر من نصف الشبكة"],
            ["موضوعات بلا أي علاقة", f'{stats.get("isolated_topics", "?")} من {total}', "أغلبها من فئة التوسع التي لم تُراجع"],
            ["مكوّنات مترابطة", len(comps), f"أكبرها {max(comps.values())} عقدة"],
            ["موضوعات مرشحة للرحلة بلا علاقة", f'{stats.get("flagship_candidates_isolated", "?")} من {coverage.get("flagship_journey", 0)}',
             "ليست من المحطات المختارة؛ لا محطة معزولة"],
            ["جبهات بحثية بلا علاقة", f'{stats.get("frontier_isolated", "?")} من {coverage.get("research_frontier", 0)}', "وُصلت بجيرانها في المراجعة"],
            ["أعلى درجة ارتباط", stats.get("max_degree", "?"), "الشبكة ما زالت رقيقة"],
        ],
    )

    # ---------------------------------------------------------------- 5) الرحلة
    j = jrn["journeys"][0]
    stations = {s["order"]: s for s in j["stations"]}
    js = [f"**{j['title_ar']}** — `{j['id']}` ({j['status']})\n"]
    js.append(f"**السؤال الذي تبدأ منه:** {j['starts_with_question_ar']}\n")
    js.append(f"**الهدف:** {j['goal_ar']}\n")
    js.append(f"> {j['independence_note_ar']}\n")
    js.append(f"> {jrn['separation_rule_ar']}\n")
    for ch in j["chapters"]:
        orders = ch["station_orders"]
        js.append(f"### {ch['title_ar']} — المحطات {orders[0]}–{orders[-1]}\n")
        js.append(f"**هدف الفصل:** {ch['goal_ar']}\n")
        cont = ch.get("content") or {}
        if cont.get("ar") or cont.get("en"):
            js.append(f"**محتوى الفصل:** [العربية]({cont.get('ar', '')}) · [English]({cont.get('en', '')}) — الحالة: `{cont.get('status')}`\n")
        js.append(table(
            ["#", "الدور", "الموضوع", "لماذا هنا", "المخرج"],
            [[o, ROLE_LABEL.get(stations[o]["role"], stations[o]["role"]),
              f"`{stations[o]['topic_id']}` {topics[stations[o]['topic_id']]['title_ar']}",
              stations[o]["why_here_ar"], stations[o].get("deliverable_ar") or "—"] for o in orders]))
        js.append("")
        js.append("**أنشطة المحطات في هذا الفصل:**\n")
        for o in orders:
            s = stations[o]
            act = s.get("activity") or {}
            if not act:
                continue
            js.append(f"**{o}. `{s['topic_id']}` — {topics[s['topic_id']]['title_ar']}** ({ROLE_LABEL.get(s['role'], s['role'])})\n")
            js.append(f"- **يرى:** {act.get('sees_ar', '—')}")
            js.append(f"- **يجرب:** {act.get('tries_ar', '—')}")
            if act.get("hypothesis_ar"):
                js.append(f"- **فرضية المتعلم قبل التجربة:** {act['hypothesis_ar']}")
                outs = act.get("possible_outcomes_ar") or []
                js.append("- **نتائج محتملة:** " + " / ".join(f"«{x}»" for x in outs))
                js.append(f"- **تفسير كل احتمال:** {act.get('interpretation_ar', '—')}")
            elif act.get("expects_ar"):
                js.append(f"- **نتيجة مفترضة (عقد قديم، يحتاج ترحيلاً):** {act['expects_ar']}")
            js.append(f"- **يكتشف:** {act.get('discovers_ar', '—')}")
            js.append("")
    js.append("**أدلة الإنجاز:**\n")
    for e in j["exit_evidence_ar"]:
        js.append(f"- {e}")
    if j.get("methodology_notes_ar"):
        js.append("\n**ملاحظات منهجية (ليست محطات):**\n")
        for m in j["methodology_notes_ar"]:
            js.append(f"- `{m['topic_id']}` {topics[m['topic_id']]['title_ar']} — {m['note_ar']}")
    js.append("\n**فروع اختيارية:**\n")
    for b in j.get("optional_branches_ar", []):
        js.append(f"- `{b['topic_id']}` {topics[b['topic_id']]['title_ar']} — {b['when_ar']}")
    journey_section = "\n".join(js)

    # ---------------------------------------------------------------- 6) عقود الدخول
    cs = []
    for c in con["contracts"]:
        t = topics[c["topic_id"]]
        cs.append(f"### `{c['topic_id']}` — {t['title_ar']}\n")
        cs.append(f"**سؤال الدخول:** {c['entry_question_ar']}\n")
        cs.append(f"**لماذا يستحق الاستقلال:** {c['why_standalone_ar']}\n")
        cs.append("**ما يُشرح داخل الصفحة بدل حجبه:**\n")
        cs.append(table(["المفهوم", "الوعد"],
                        [[x["concept_ar"], x["promise_ar"]] for x in c["inline_explanations"]]))
        cs.append("\n**مستويات العمق:**\n")
        cs.append(table(["المستوى", "ما يحتاجه"], [[x["level"], x["needs_ar"]] for x in c["depth_levels"]]))
        cs.append("\n**بوابات ممنوعة:**\n")
        for g in c["forbidden_gates_ar"]:
            cs.append(f"- {g}")
        cs.append(f"\n**دليل الاستقلال:** {c['proof_of_independence_ar']}")
        if c.get("linked_journey_id"):
            cs.append(f"**الرابط بالرحلة:** `{c['linked_journey_id']}` المحطة {c['linked_station_order']} (اختياري، ليس شرط دخول).")
        cs.append("")
    contract_section = "\n".join(cs)

    # ---------------------------------------------------------------- 7) الفحص الآلي
    checks = table(
        ["الفحص", "ماذا يفشل عنده البناء"],
        [
            ["مطابقة JSON Schema", "حقل ناقص أو زائد أو نوع خاطئ، بما فيه إعادة إدخال `prerequisites` أو `topic_count`"],
            ["مصدر حقيقة واحد", "ظهور نسخة للعلاقات أو عدد مخزّن داخل أي عنصر"],
            ["سلامة المعرّفات", "معرّف مكرر، بادئة لا تطابق المجال، معرّف مدموج أُعيد استخدامه"],
            ["حالة المراجعة", "حافة مضافة بلا `review`، أو `provisional` بلا سبب"],
            ["العلاقات", "طرف غير موجود، نوع غير معروف، حلقة ذاتية، ثلاثية مكررة"],
            ["الدورات", "دورة في أي نوع اعتماد، مجتمعةً لا منفردة"],
            ["المجموعات", "مجال بلا مجموعة، أو في مجموعتين"],
            ["الرحلة: التكرار", "موضوع مكرر **داخل** الرحلة؛ إعادة الاستخدام بين رحلات مسموحة"],
            ["الرحلة: الفصول", "فصل لا يغطي محطاته، أو تداخل فصول، أو ترتيب غير تصاعدي"],
            ["الرحلة: الأنشطة", "محطة أساسية بلا `sees/tries/expects/discovers` أو بلا مخرج"],
            ["عقود الدخول", "عقد بلا شرح داخلي أو بلا بوابات ممنوعة، أو رابط رحلة لا يطابق موضوعه"],
        ],
    )
    limits = table(
        ["ما لا يثبته المدقق", "الجهة التي تثبته"],
        [
            ["الصحة العلمية لعنوان موضوع أو سؤال افتتاحي", "مراجعة بشرية / مصدر فردي"],
            ["صحة العلاقة العلمية نفسها، لا مجرد إعلانها وتبريرها", "المراجعة المستقلة (أُجريت للحواف المضافة)"],
            ["التكرار الدلالي بين موضوعين", "تحكيم تحريري"],
            ["جودة الشرح والتكافؤ بين العربية والإنجليزية", "تحرير المحتوى قبل النشر"],
            ["جودة تصميم الرحلة التعليمية", "تجربة متعلمين، لا فحص بنيوي"],
        ],
    )
    validation_section = (
        checks
        + f"\n\n**نتيجة الفحص:** ناجح — أخطاء: {len(val.get('errors', []))}، تحذيرات: {len(val.get('warnings', []))}.\n"
        + f"\n**بصمة المدخلات:** `{val['input_fingerprint'][:16]}…` (تاريخ الفحص: {val['generated_at']}). "
        + "المولّد يعيد الفحص ويرفض التوليد إذا لم تطابق البصمة، فلا تصلح شارة تحقق قديمة لمدخلات جديدة.\n"
        + "\n### حدود المدقق — ما لا يثبته\n\n"
        + limits
        + "\n\n**الخلاصة:** المدقق يضمن أن النموذج لا يناقض نفسه. لا يضمن أن المعرفة صحيحة."
    )

    refs = "\n".join(f"{i + 1}. <{u}>" for i, u in enumerate(reg["reference_basis"]))
    counts_line = (f"{len(domains)} مجالاً في {len(groups)} مجموعات، و{total} موضوعاً، "
                   f"و{len(rel)} علاقة موثّقة، و{len(rtypes)} أنواع علاقة")

    repl = {
        "{{DATE}}": reg["date"],
        "{{CHANGES_TABLE}}": changes,
        "{{COUNTS_LINE}}": counts_line,
        "{{COVERAGE_TABLE}}": table(["الرمز", "المعنى", "العدد", "النسبة"], cov_rows),
        "{{DOMAIN_MAP}}": domain_map,
        "{{RELATION_TYPES_TABLE}}": type_table,
        "{{GRAPH_STATS}}": graph_stats,
        "{{JOURNEY_SECTION}}": journey_section,
        "{{CONTRACT_SECTION}}": contract_section,
        "{{VALIDATION_SECTION}}": validation_section,
        "{{REFERENCE_LIST}}": refs,
    }
    text = TPL.read_text(encoding="utf-8")
    for k, v in repl.items():
        text = text.replace(k, v)
    left = re.findall(r"\{\{[A-Z_]+\}\}", text)
    if left:
        print("ERROR: placeholders left:", left)
        return 1
    write_text_lf(OUT, text + "\n")
    print("written:", OUT.relative_to(ROOT), "| topics:", total, "| relations:", len(rel),
          "| fingerprint:", fp_now[:12])
    return 0


if __name__ == "__main__":
    sys.exit(main())
