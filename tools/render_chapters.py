#!/usr/bin/env python3
"""Render only the approved nine bilingual Wakeel chapters; no raw HTML allowed.

The 18 original Markdown byte sequences are verified against the immutable
75-file PUBLIC-MANIFEST before rendering. The rendered HTML is static build
output, never a replacement for the approved scientific chapter source.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "baseline" / "v0.21.3"
MD = MarkdownIt("commonmark", {"html": False, "linkify": False, "typographer": False})
MD.enable("table")

def fail(message: str):
    raise ValueError("Approved chapter render rejected: " + message)

def render_chapters(source: Path = SOURCE):
    manifest = json.loads((source / "PUBLIC-MANIFEST.json").read_text("utf-8"))
    if manifest.get("source_version") != "v0.21.3" or manifest.get("count_public_source_files") != 75:
        fail("wrong approved source")
    result = {}
    for i in range(1, 10):
        chapter = "CH-" + str(i)
        translations = {}
        for language in ("ar", "en"):
            path = f"content/chapter-{i:02d}/{language}.md"
            expected = manifest.get("files", {}).get(path)
            if not isinstance(expected, str) or len(expected) != 64:
                fail(f"missing source digest for {path}")
            raw = (source / path).read_bytes()
            actual = hashlib.sha256(raw).hexdigest()
            if actual != expected:
                fail(f"chapter bytes differ from the frozen source: {path}")
            content = raw.decode("utf-8")
            if not content.strip() or len(content) < 1000:
                fail(f"chapter unexpectedly empty: {path}")
            # MarkdownIt escapes inline raw HTML (including <script>) and
            # rejects unsafe javascript: URLs; table syntax is enabled.
            html = MD.render(content)
            if not html.strip().startswith("<h1>"):
                fail(f"expected first-level authored chapter title: {path}")
            translations[language] = {
                "html": html,
                "source_sha256": actual,
                "source_path": path,
                "source_bytes": len(raw),
            }
        result[chapter] = translations
    return {
        "schema": "wakeel-v022-bilingual-chapter-render/v1",
        "count": 18,
        "chapters": result,
    }

if __name__ == "__main__":
    try:
        print(json.dumps(render_chapters(), ensure_ascii=False, separators=(",", ":")))
    except (OSError, UnicodeError, ValueError, KeyError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
