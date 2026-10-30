#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
common.py — أدوات مشتركة: بصمة دلالية وكتابة ثابتة البايتات.

سبب الوجود: بصمة الملفات الخام تختلف بين Windows (CRLF) وLinux (LF) رغم
تطابق المحتوى. البصمة هنا تُحسب على JSON موحّد التمثيل (مفاتيح مرتبة، بلا
مسافات زائدة)، والكتابة تُثبّت نهاية السطر على LF في كل الأنظمة.
"""
import hashlib
import json
from pathlib import Path


def canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def fingerprint(paths):
    """بصمة دلالية: تتجاهل ترتيب المفاتيح والمسافات ونهايات الأسطر."""
    h = hashlib.sha256()
    for key, p in paths.items():
        obj = json.loads(Path(p).read_text(encoding="utf-8"))
        h.update(key.encode("utf-8"))
        h.update(canonical(obj).encode("utf-8"))
    return h.hexdigest()


def write_json(path, obj):
    text = json.dumps(obj, ensure_ascii=False, indent=1) + "\n"
    Path(path).write_text(text, encoding="utf-8", newline="\n")


def write_text_lf(path, text):
    Path(path).write_text(text, encoding="utf-8", newline="\n")
