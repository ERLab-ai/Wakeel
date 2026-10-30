"""Security and immutability tests for build-only Markdown conversion."""
import json
from pathlib import Path
import tempfile
import unittest
from render_chapters import MD, SOURCE, render_chapters

class ChapterRendererTests(unittest.TestCase):
    def test_approved_chapters_are_complete_and_bilingual(self):
        result=render_chapters()
        self.assertEqual(result["count"],18)
        self.assertEqual(len(result["chapters"]),9)
        for chapter in result["chapters"].values():
            for lang in ("ar","en"):
                self.assertTrue(chapter[lang]["html"].startswith("<h1>"))
                self.assertGreater(chapter[lang]["source_bytes"],1000)

    def test_raw_html_and_script_urls_do_not_become_active_markup(self):
        html=MD.render('<script>alert("x")</script>\n\n[evil](javascript:alert(1))\n\n|x|y|\n|-|-|\n|1|2|')
        self.assertNotIn("<script",html)
        self.assertNotIn('href="javascript:',html)
        self.assertIn("&lt;script&gt;",html)
        self.assertIn("<table>",html)

    def test_modified_original_chapter_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="wakeel-render-mut-") as d:
            p=Path(d)
            (p/"PUBLIC-MANIFEST.json").write_bytes((SOURCE/"PUBLIC-MANIFEST.json").read_bytes())
            for i in range(1,10):
                for lang in ("ar","en"):
                    relative=Path("content")/f"chapter-{i:02d}"/f"{lang}.md"
                    target=p/relative
                    target.parent.mkdir(parents=True,exist_ok=True)
                    target.write_bytes((SOURCE/relative).read_bytes())
            target=p/"content/chapter-09/en.md"
            target.write_bytes(target.read_bytes()+b"\nUNSIGNED ADDITION")
            with self.assertRaisesRegex(ValueError,"chapter bytes differ"):
                render_chapters(p)

if __name__=="__main__":
    unittest.main()
