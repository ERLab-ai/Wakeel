"""Real Chromium browser acceptance for Wakeel v0.22 preview.

Run: python -m unittest discover -s tests -p 'test_browser.py' -v
Requires Playwright Chromium installed. Static/axe-independent checks do not
replace a human screen-reader or visual accessibility audit.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import re
import subprocess
import time
import unittest
from urllib.request import urlopen
from urllib.error import URLError
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1]
BASE="http://127.0.0.1:8087"
OUT=ROOT/"artifacts"/"browser"

class BrowserGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        OUT.mkdir(parents=True,exist_ok=True)
        cls.server=subprocess.Popen(
            ["node","scripts/serve.mjs"],cwd=ROOT,
            env={**os.environ,"PORT":"8087"},
            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        for _ in range(80):
            if cls.server.poll() is not None:
                raise RuntimeError("Preview server exited before browser test")
            try:
                with urlopen(BASE+"/",timeout=1) as response:
                    if response.status==200:
                        break
            except (URLError,TimeoutError):
                time.sleep(0.2)
        else:
            raise RuntimeError("Preview server never became ready")
        cls.pw=sync_playwright().start()
        cls.browser=cls.pw.chromium.launch(headless=True,args=["--no-sandbox"])

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.pw.stop()
        cls.server.terminate()
        try:cls.server.wait(timeout=5)
        except subprocess.TimeoutExpired:cls.server.kill()

    def setUp(self):
        self.page=self.browser.new_page(
            viewport={"width":1365,"height":850},device_scale_factor=1)
        self.errors=[]
        self.page.on("pageerror",lambda error:self.errors.append(str(error)))

    def tearDown(self):
        try:self.assertEqual(self.errors,[],"JavaScript runtime errors")
        finally:self.page.close()

    def visit(self,fragment=""):
        self.page.goto(BASE+"/"+fragment,wait_until="networkidle",timeout=30000)
        self.page.locator("#chapter-nav .chapter-choice").first.wait_for()
        self.page.locator("#topic-results .topic-result").first.wait_for()

    def test_01_authentic_journey_atlas_and_arabic_default(self):
        self.visit()
        self.assertEqual(self.page.locator("html").get_attribute("lang"),"ar")
        self.assertEqual(self.page.locator("html").get_attribute("dir"),"rtl")
        self.assertEqual(self.page.locator("#chapter-nav a").count(),9)
        self.assertEqual(self.page.locator(".nav a[href='#journey']").inner_text(),"الرحلة")
        self.assertEqual(self.page.locator(".nav a[href='#atlas']").inner_text(),"الأطلس")
        self.assertEqual(self.page.locator(".nav").get_attribute("aria-label"),"التنقل الرئيسي")
        self.assertNotIn("لم يُنقل",self.page.locator("footer").inner_text())
        self.assertEqual(self.page.locator("#station-nav a").count(),5)
        catalog=self.page.evaluate("""async()=> {
          const j=await(await fetch('/data/learning.json')).json();
          return {counts:j.counts,articles:Object.keys(j.chapter_articles||{}).length,
            complete:Object.values(j.chapter_articles||{}).every(c=>c.ar?.html&&c.en?.html)};
        }""")
        self.assertEqual(catalog["counts"],{"chapters":9,"stations":44,"topics":389,"relations":125})
        self.assertEqual(catalog["articles"],9)
        self.assertTrue(catalog["complete"])
        self.page.screenshot(path=str(OUT/"arabic-desktop.png"))

    def test_02_direct_chapter_9_full_science_and_language_switch(self):
        self.visit("#chapter/9")
        detail=self.page.locator(".chapter-reader")
        self.assertTrue(detail.is_visible())
        self.assertTrue(detail.evaluate("(d)=>d.open"),"Direct chapter link should expand the article")
        self.assertEqual(self.page.locator(".chapter-prose").get_attribute("dir"),"rtl")
        self.assertGreaterEqual(self.page.locator(".chapter-prose table").count(),1)
        self.assertGreaterEqual(self.page.locator(".chapter-prose pre").count(),1)
        self.assertEqual(self.page.locator(".chapter-prose h1").count(),1)
        arabic_source=self.page.locator(".chapter-provenance").inner_text()
        self.page.screenshot(path=str(OUT/"arabic-chapter-9.png"))
        self.page.locator("#locale").select_option("en")
        self.assertEqual(self.page.locator("html").get_attribute("lang"),"en")
        self.assertEqual(self.page.locator("html").get_attribute("dir"),"ltr")
        self.assertEqual(self.page.locator(".nav a[href='#journey']").inner_text(),"Journey")
        self.assertEqual(self.page.locator(".nav a[href='#atlas']").inner_text(),"Atlas")
        self.assertEqual(self.page.locator(".nav").get_attribute("aria-label"),"Main navigation")
        self.assertNotIn("not yet been imported",self.page.locator("footer").inner_text())
        self.assertEqual(self.page.locator(".chapter-prose").get_attribute("dir"),"ltr")
        self.assertNotEqual(arabic_source,self.page.locator(".chapter-provenance").inner_text())
        self.assertTrue(self.page.locator(".chapter-reader").evaluate("(d)=>d.open"),
                        "Full-chapter expansion should persist across language switches")
        self.page.screenshot(path=str(OUT/"english-chapter-9.png"))

    def test_03_direct_topic_search_and_history(self):
        self.visit("#topic/NLP-003")
        self.assertIn("NLP-003",self.page.locator("#topic-detail").inner_text())
        self.page.locator("#topic-search").fill("NLP-003")
        results=self.page.locator("#topic-results a.topic-result")
        self.assertEqual(results.count(),1)
        self.assertIn("NLP-003",results.first.inner_text())
        self.page.goto(BASE+"/#station/44",wait_until="networkidle")
        self.assertIn("44 / 44",self.page.locator("#station-content .eyebrow").inner_text())
        self.page.go_back(wait_until="networkidle")
        self.assertIn("NLP-003",self.page.locator("#topic-detail").inner_text())

    def test_04_keyboard_access_and_accessibility_semantics(self):
        self.visit("#chapter/9")
        self.page.goto(BASE+"/",wait_until="networkidle")
        self.page.keyboard.press("Tab")
        self.assertIn("skip",self.page.evaluate("document.activeElement.className"))
        self.page.locator(".chapter-reader summary").focus()
        self.page.keyboard.press("Enter")
        self.assertTrue(self.page.locator(".chapter-reader").evaluate("(d)=>d.open"))
        self.page.locator("#topic-search").focus()
        self.page.keyboard.type("FND-001")
        self.assertEqual(self.page.locator("#topic-results a.topic-result").count(),1)
        self.assertEqual(self.page.locator("#topic-search").get_attribute("type"),"search")
        cdp=self.page.context.new_cdp_session(self.page)
        tree=cdp.send("Accessibility.getFullAXTree")["nodes"]
        roles=[node.get("role",{}).get("value","") for node in tree]
        self.assertIn("main",roles)
        self.assertIn("navigation",roles)
        self.assertIn("heading",roles)

    def test_05_mobile_overflow_and_scrollable_science_tables(self):
        self.page.set_viewport_size({"width":390,"height":844})
        self.visit("#chapter/9")
        doc=self.page.evaluate("({width:document.documentElement.scrollWidth, viewport:innerWidth})")
        self.assertLessEqual(doc["width"],doc["viewport"]+2,
                             "Horizontal document overflow on 390px phone")
        tables=self.page.locator(".chapter-prose table")
        self.assertGreater(tables.count(),0)
        first=tables.first
        first.scroll_into_view_if_needed()
        container=self.page.locator(".chapter-prose")
        self.assertEqual(container.evaluate("(e)=>getComputedStyle(e).overflowX"),"auto")
        self.page.screenshot(path=str(OUT/"arabic-mobile-table.png"))
        self.page.locator("#locale").select_option("en")
        doc=self.page.evaluate("({width:document.documentElement.scrollWidth, viewport:innerWidth})")
        self.assertLessEqual(doc["width"],doc["viewport"]+2,
                             "Horizontal document overflow on 390px English phone")
        self.page.screenshot(path=str(OUT/"english-mobile.png"))

    def test_06_no_public_science_archives_and_no_active_user_markup(self):
        self.visit()
        for name in [
          "/baseline/v0.21.3/content/chapter-09/results.json",
          "/baseline/v0.21.3/content/chapter-09/runs.log",
          "/review-v0.21.2/MANIFEST.json",
          "/tools/render_chapters.py"
        ]:
            self.assertEqual(self.page.request.get(BASE+name).status,404,name)
        self.assertEqual(self.page.locator(".chapter-prose script").count(),0)
        self.assertEqual(self.page.locator(".chapter-prose iframe").count(),0)
        self.assertEqual(self.page.locator(".chapter-prose img").count(),0)

if __name__=="__main__":
    unittest.main(verbosity=2)
