#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
selftest.py — اختبار سلبي وإيجابي للمدقق.

كل حالة تُكتب في مجلد مؤقت، مع تقرير مؤقت. لا يلمس هذا الاختبار
out/validation-report.json ولا أي ملف من ملفات المشروع.

الغرض: إثبات أن المدقق يرفض الفساد، ويقبل الحالات المشروعة — ومنها إعادة
استخدام الموضوع نفسه في رحلتين، وهي قاعدة مؤسسة لا خطأ.
"""
import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "tools" / "validate.py"
REG = ROOT / "data" / "WAKEEL-2-KNOWLEDGE-REGISTRY-v0.3.json"
JRN = ROOT / "data" / "WAKEEL-2-JOURNEYS-v0.3.json"
CON = ROOT / "data" / "WAKEEL-2-ENTRY-CONTRACTS-v0.3.json"


def run(dirpath, reg, jrn, con):
    (dirpath / "reg.json").write_text(json.dumps(reg, ensure_ascii=False), encoding="utf-8")
    (dirpath / "jrn.json").write_text(json.dumps(jrn, ensure_ascii=False), encoding="utf-8")
    (dirpath / "con.json").write_text(json.dumps(con, ensure_ascii=False), encoding="utf-8")
    p = subprocess.run([
        sys.executable, str(VALIDATOR),
        "--registry", str(dirpath / "reg.json"),
        "--journeys", str(dirpath / "jrn.json"),
        "--contracts", str(dirpath / "con.json"),
        "--report", str(dirpath / "report.json"),
        "--quiet",
    ], capture_output=True, text=True, encoding="utf-8")
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def main():
    base_reg = json.loads(REG.read_text(encoding="utf-8"))
    base_jrn = json.loads(JRN.read_text(encoding="utf-8"))
    base_con = json.loads(CON.read_text(encoding="utf-8"))

    cases = []

    def add(name, reg=None, jrn=None, con=None, valid=False):
        cases.append((name, reg or base_reg, jrn or base_jrn, con or base_con, valid))

    r = copy.deepcopy(base_reg); r["domains"][0]["topic_count"] = 17
    add("topic_count مخزّن", reg=r)
    r = copy.deepcopy(base_reg); r["topics"][0]["prerequisites"] = [{"topic_id": "FND-002"}]
    add("نسخة علاقات داخل الموضوع", reg=r)
    r = copy.deepcopy(base_reg); r["relations"].append(copy.deepcopy(r["relations"][0]))
    add("علاقة مكررة", reg=r)
    r = copy.deepcopy(base_reg); r["relations"].append({"source": "MAT-007", "target": "MAT-001", "type": "hard_implementation"})
    add("دورة اعتماد", reg=r)
    r = copy.deepcopy(base_reg); r["relations"].append({"source": "FND-001", "target": "ZZZ-999", "type": "related"})
    add("طرف غير موجود", reg=r)
    r = copy.deepcopy(base_reg)
    r["id_history"].append({"id": "FND-001", "state": "merged", "merged_into": "NOPE-001",
                            "decided": "2026-10-25", "rationale_ar": "اختبار"})
    add("merged_into غير موجود", reg=r)
    r = copy.deepcopy(base_reg); r["relations"].append({"source": "FND-001", "target": "FND-002", "type": "causes_everything"})
    add("نوع علاقة غير معروف", reg=r)
    r = copy.deepcopy(base_reg); r["domain_groups"][0]["domain_ids"].remove("FND")
    add("مجال بلا مجموعة", reg=r)
    r = copy.deepcopy(base_reg); r["relations"].append({"source": "FND-001", "target": "FND-002", "type": "related",
                                                       "added_in": "v0.2", "rationale_ar": "بلا مراجعة"})
    add("حافة مضافة بلا حالة مراجعة", reg=r)

    j = copy.deepcopy(base_jrn)
    first = j["journeys"][0]
    first["stations"][1]["topic_id"] = first["stations"][0]["topic_id"]
    add("موضوع مكرر داخل الرحلة", jrn=j)

    j = copy.deepcopy(base_jrn)
    j["journeys"][0]["chapters"][0]["station_orders"] = [1, 2, 3, 4]
    add("فصل لا يغطي محطاته", jrn=j)

    j = copy.deepcopy(base_jrn)
    del j["journeys"][0]["stations"][0]["activity"]
    add("محطة أساسية بلا نشاط", jrn=j)

    j = copy.deepcopy(base_jrn)
    j["journeys"][0]["stations"][3]["topic_id"] = "ZZZ-001"
    add("محطة بموضوع غير موجود", jrn=j)

    c = copy.deepcopy(base_con)
    c["contracts"][0]["linked_station_order"] = 99
    add("رابط رحلة بمحطة خاطئة", con=c)

    # --- حالات يجب أن تُقبل ---
    add("السجل والرحلة الحاليان", valid=True)

    j = copy.deepcopy(base_jrn)
    j2 = copy.deepcopy(j["journeys"][0])
    j2["id"] = "JRN-CV-001"
    j2["title_ar"] = "رحلة اختبار ثانية"
    j2["title_en"] = "Second test journey"
    s1 = copy.deepcopy(j2["stations"][0]); s1.update({"order": 1, "role": "reference", "deliverable_ar": None, "deliverable_en": None})
    s2 = copy.deepcopy(j2["stations"][5]); s2.update({"order": 2, "role": "reference", "deliverable_ar": None, "deliverable_en": None})
    j2["stations"] = [s1, s2]
    j2["chapters"] = [{"id": "CH-1", "title_ar": "اختبار", "title_en": "Test",
                       "station_orders": [1, 2],
                       "goal_ar": "اختبار إعادة استخدام موضوع بين رحلتين",
                       "goal_en": "Testing topic reuse across two journeys"}]
    j["journeys"].append(j2)
    add("إعادة استخدام موضوع بين رحلتين (مسموحة)", jrn=j, valid=True)

    failures = []
    with tempfile.TemporaryDirectory(prefix="wakeel_selftest_") as td:
        tmp = Path(td)
        for i, (name, reg, jrn, con, valid) in enumerate(cases):
            d = tmp / f"case_{i}"
            d.mkdir()
            code, out = run(d, reg, jrn, con)
            ok = (code == 0) if valid else (code != 0)
            tag = "EXPECT-VALID" if valid else "EXPECT-REJECT"
            print(f'{"PASS" if ok else "FAIL":5} {tag:14} {name}')
            if not ok:
                failures.append(name)
                print(out[-700:])

    print()
    if failures:
        print(f"فشل الاختبار الذاتي في {len(failures)} حالة: {failures}")
        return 1
    print(f"نجح الاختبار الذاتي: {sum(1 for c in cases if not c[4])} حالة فاسدة رُفضت، "
          f"{sum(1 for c in cases if c[4])} حالة سليمة قُبلت، والتقرير الأساسي لم يُلمس.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
