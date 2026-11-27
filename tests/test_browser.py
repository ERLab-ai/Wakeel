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

    def test_07_hash_navigation_focus_and_malformed_uri(self):
        self.visit()
        self.page.locator("#chapter-nav a.chapter-choice").nth(1).click()
        self.page.wait_for_function(
            "() => document.activeElement?.matches('#station-content .station-main-title')")
        self.assertEqual(self.page.locator("#station-content .station-main-title").get_attribute("tabindex"),"-1")
        self.page.locator("#topic-results a.topic-result").nth(5).click()
        self.page.wait_for_function(
            "() => document.activeElement?.matches('#topic-detail .station-main-title')")
        self.page.evaluate("location.hash = '#topic/%E0%A4%A'")
        self.page.wait_for_timeout(100)
        self.assertTrue(self.page.locator("#topic-detail .station-main-title").is_visible())

    def test_08_chapter_one_frozen_aggregates_in_both_languages(self):
        self.visit("#chapter/1")
        evidence=self.page.locator(".classification-evidence")
        self.assertEqual(evidence.count(),1)
        evidence.locator("summary").click()
        self.assertTrue(evidence.evaluate("(e)=>e.open"))
        rows=evidence.locator(".evidence-table tbody tr")
        self.assertEqual(rows.count(),2)
        self.assertIn("4 / 9",rows.nth(1).inner_text())
        self.assertIn("8 / 9",rows.nth(1).inner_text())
        self.assertIn("SHA-256:",evidence.locator(".evidence-provenance").inner_text())
        catalog=self.page.evaluate("""async() => {
          const j=await (await fetch('/data/learning.json')).json();
          return {evidence:j.evidence_replays?.['CH-1'],
            hasRows:JSON.stringify(j.evidence_replays).includes('rule_rows')};
        }""")
        self.assertEqual(catalog["evidence"]["datasets"]["ar"]["tests"]["rules_correct"],4)
        self.assertFalse(catalog["hasRows"])
        self.page.locator("#locale").select_option("en")
        self.assertIn("Frozen Chapter 1",self.page.locator(".evidence-summary").inner_text())
        self.assertTrue(self.page.locator(".classification-evidence").evaluate("(e)=>e.open"))
        self.assertIn("Correct with rules",self.page.locator(".evidence-table").inner_text())
        self.page.goto(BASE+"/#chapter/9",wait_until="networkidle")
        self.assertEqual(self.page.locator(".classification-evidence").count(),0)

    def test_09_chapter_two_tokenization_evidence_is_language_scoped(self):
        self.visit("#chapter/2")
        evidence=self.page.locator(".tokenization-evidence")
        self.assertEqual(evidence.count(),1)
        evidence.locator("summary").click()
        self.assertTrue(evidence.evaluate("(e)=>e.open"))
        counts=evidence.locator(".evidence-table tbody tr")
        self.assertEqual(counts.count(),2)
        self.assertEqual(counts.nth(0).locator("td").inner_text(),"5")
        self.assertEqual(counts.nth(1).locator("td").inner_text(),"8")
        self.assertIn("edu-tok-0.1",evidence.locator(".evidence-provenance").inner_text())
        self.page.locator("#locale").select_option("en")
        evidence=self.page.locator(".tokenization-evidence")
        self.assertTrue(evidence.evaluate("(e)=>e.open"))
        self.assertIn("frozen text-unit comparison",evidence.locator("summary").inner_text())
        counts=evidence.locator(".evidence-table tbody tr")
        self.assertEqual(counts.nth(0).locator("td").inner_text(),"7")
        self.assertEqual(counts.nth(1).locator("td").inner_text(),"9")
        projection=self.page.evaluate("""async() => {
          const j=await (await fetch('/data/learning.json')).json();
          return j.evidence_replays?.['CH-2'];
        }""")
        self.assertEqual(projection["tokenizer"],"edu-tok-0.1")
        self.assertNotIn("edu_tokens",str(projection))
        self.page.goto(BASE+"/#chapter/3",wait_until="networkidle")
        self.assertEqual(self.page.locator(".tokenization-evidence").count(),0)

    def test_10_chapter_three_metric_and_smoothing_in_both_languages(self):
        self.visit("#chapter/3")
        panel=self.page.locator(".bigram-evidence")
        self.assertEqual(panel.count(),1)
        panel.locator("summary").click()
        self.assertTrue(panel.evaluate("(node)=>node.open"))
        rows=panel.locator(".evidence-table tbody tr")
        self.assertEqual(rows.count(),3)
        self.assertIn("2.6560",rows.nth(1).inner_text())
        self.assertIn("2.6380",rows.nth(1).inner_text())
        self.assertIn("0.5",panel.inner_text())
        self.assertIn("SHA-256",panel.locator(".evidence-provenance").inner_text())
        output=panel.locator(".smoothing-output")
        self.assertIn("0.2857",output.inner_text())
        inputs=panel.locator(".smoothing-field input")
        self.assertEqual(inputs.count(),3)
        inputs.nth(1).fill("0")
        self.assertIn("0.1429",output.inner_text())
        inputs.nth(1).fill("2")
        self.assertIn("0 ≤ c ≤ n",output.inner_text())
        inputs.nth(1).fill("1")
        panel.locator(".smoothing-field select").select_option("0.1")
        self.assertIn("0.6875",output.inner_text())
        self.page.locator("#locale").select_option("en")
        panel=self.page.locator(".bigram-evidence")
        self.assertTrue(panel.evaluate("(node)=>node.open"))
        rows=panel.locator(".evidence-table tbody tr")
        self.assertIn("2.2232",rows.nth(1).inner_text())
        self.assertIn("2.2234",rows.nth(1).inner_text())
        self.assertIn("0.1",panel.inner_text())
        self.assertIn("single-transition",panel.inner_text())
        self.page.goto(BASE+"/#chapter/4",wait_until="networkidle")
        self.assertEqual(self.page.locator(".bigram-evidence").count(),0)

    def test_11_chapter_four_context_sparsity_bilingual_and_bounded(self):
        self.visit("#chapter/4")
        panel=self.page.locator(".context-evidence")
        self.assertEqual(panel.count(),1)
        panel.locator("summary").click()
        self.assertTrue(panel.evaluate("(e)=>e.open"))
        self.assertIn("SHA-256",panel.locator(".evidence-provenance").inner_text())
        rows=panel.locator(".evidence-table tbody tr")
        self.assertEqual(rows.count(),4)
        self.assertIn("2.2239",rows.nth(2).inner_text())
        self.assertIn("2.3961",rows.nth(3).inner_text())
        self.assertIn("23.31%",rows.nth(2).inner_text())
        self.assertIn("57,798",rows.nth(2).inner_text())
        self.assertIn("2,254,122",rows.nth(3).inner_text())
        self.assertNotIn("Backoff",panel.locator(".evidence-table").inner_text())
        output=panel.locator(".context-output")
        self.assertIn("57,798",output.inner_text())
        panel.locator(".context-order").select_option("3")
        self.assertIn("2,254,122",output.inner_text())
        vocab=panel.locator(".context-contextVocabulary")
        vocab.fill("2")
        self.assertIn("304",output.inner_text())
        vocab.fill("1")
        self.assertIn("2–64",output.inner_text())
        vocab.fill("39")
        panel.locator(".context-order").select_option("2")
        self.assertIn("57,798",output.inner_text())
        self.page.locator("#locale").select_option("en")
        panel=self.page.locator(".context-evidence")
        self.assertTrue(panel.evaluate("(e)=>e.open"))
        self.assertIn("context length and data sparsity",panel.locator("summary").inner_text())
        rows=panel.locator(".evidence-table tbody tr")
        self.assertIn("1.9494",rows.nth(2).inner_text())
        self.assertIn("2.0234",rows.nth(3).inner_text())
        self.assertIn("49,284",rows.nth(2).inner_text())
        self.assertIn("1,823,508",rows.nth(3).inner_text())
        self.assertIn("49,284",panel.locator(".context-output").inner_text())
        self.page.goto(BASE+"/#chapter/5",wait_until="networkidle")
        self.assertEqual(self.page.locator(".context-evidence").count(),0)

    def test_12_chapter_five_ablation_and_scalar_update_are_bilingual(self):
        self.visit("#chapter/5")
        panel=self.page.locator(".embedding-evidence")
        self.assertEqual(panel.count(),1)
        panel.locator("summary").click()
        self.assertTrue(panel.evaluate("(e)=>e.open"))
        self.assertIn("SHA-256",panel.locator(".evidence-provenance").first.inner_text())
        self.assertEqual(panel.locator(".evidence-table").count(),3)
        metrics=panel.locator(".evidence-table").nth(0).locator("tbody tr")
        self.assertEqual(metrics.count(),3)
        self.assertIn("2.5579",metrics.nth(0).inner_text())
        self.assertIn("2.7634",metrics.nth(0).inner_text())
        self.assertIn("2,110",metrics.nth(0).inner_text())
        self.assertIn("3.0632",metrics.nth(1).inner_text())
        self.assertIn("1,798",metrics.nth(1).inner_text())
        self.assertIn("2.2239",metrics.nth(2).inner_text())
        strata=panel.locator(".evidence-table").nth(1).locator("tbody tr")
        self.assertIn("102",strata.nth(0).inner_text())
        self.assertIn("31",strata.nth(1).inner_text())
        curve=panel.locator(".evidence-table").nth(2).locator("tbody tr")
        self.assertEqual(curve.count(),6)
        self.assertIn("1.9891",curve.last.inner_text())
        self.assertIn("0.5000",panel.locator(".embedding-output").inner_text())
        self.assertIn("0.1250",panel.locator(".embedding-output").inner_text())
        panel.locator(".embedding-steps").select_option("2")
        self.assertIn("0.7500",panel.locator(".embedding-output").inner_text())
        panel.locator(".embedding-initial").fill("3")
        self.assertIn("−2",panel.locator(".embedding-output").inner_text())
        panel.locator(".embedding-initial").fill("0")
        self.page.locator("#locale").select_option("en")
        panel=self.page.locator(".embedding-evidence")
        self.assertTrue(panel.evaluate("(e)=>e.open"))
        metrics=panel.locator(".evidence-table").nth(0).locator("tbody tr")
        self.assertIn("2.1609",metrics.nth(0).inner_text())
        self.assertIn("2.1759",metrics.nth(0).inner_text())
        self.assertIn("2.8913",metrics.nth(1).inner_text())
        self.assertIn("1.9494",metrics.nth(2).inner_text())
        self.assertIn("scalar quadratic-loss",panel.inner_text())
        self.page.goto(BASE+"/#chapter/6",wait_until="networkidle")
        self.assertEqual(self.page.locator(".embedding-evidence").count(),0)

    def test_13_chapter_six_memory_protocols_and_scalar_state(self):
        self.visit("#chapter/6")
        panel=self.page.locator(".memory-evidence")
        self.assertEqual(panel.count(),1)
        panel.locator("summary").click()
        self.assertTrue(panel.evaluate("(e)=>e.open"))
        self.assertIn("SHA-256",panel.locator(".evidence-provenance").inner_text())
        self.assertIn("3.1899",panel.locator(".memory-corpus").inner_text())
        self.assertIn("240",panel.inner_text())
        self.assertIn("720",panel.inner_text())
        rows=panel.locator(".memory-accuracy tbody tr")
        self.assertEqual(rows.count(),7)
        self.assertIn("100.0%",rows.first.inner_text())
        self.assertIn("50.0%",rows.last.inner_text())
        self.assertIn("83.3%",rows.last.inner_text())
        self.assertIn("طول محجوز",rows.last.inner_text())
        seeds=panel.locator(".memory-seeds tbody tr")
        self.assertEqual(seeds.count(),3)
        self.assertIn("0.7024",panel.locator(".memory-stats").inner_text())
        self.assertIn("0.3201",panel.locator(".memory-stats").inner_text())
        self.assertIn("100.0%",seeds.first.inner_text())
        self.assertNotIn("100.0%",seeds.nth(1).inner_text())
        panel.locator(".memory-delay").select_option("2")
        self.assertIn("ضمن أطوال التدريب",panel.locator(".memory-stats").inner_text())
        self.assertIn("0.0016",panel.locator(".memory-stats").inner_text())
        self.assertIn("0.656100",panel.locator(".memory-output").inner_text())
        self.assertEqual(panel.locator(".memory-trace tbody tr").count(),5)
        panel.locator(".memory-steps").select_option("16")
        self.assertIn("0.185302",panel.locator(".memory-output").inner_text())
        self.assertEqual(panel.locator(".memory-trace tbody tr").count(),17)
        panel.locator(".memory-initial").select_option("-1")
        self.assertIn("-0.185302",panel.locator(".memory-output").inner_text())
        panel.locator(".memory-retention").select_option("1")
        self.assertIn("-1.000000",panel.locator(".memory-output").inner_text())
        self.page.locator("#locale").select_option("en")
        panel=self.page.locator(".memory-evidence")
        self.assertTrue(panel.evaluate("(e)=>e.open"))
        self.assertIn("2.9962",panel.locator(".memory-corpus").inner_text())
        self.assertIn("Held-out length",panel.locator(".memory-accuracy tbody tr").last.inner_text())
        self.assertIn("State magnitude is not a recall probability",panel.inner_text())
        self.page.goto(BASE+"/#chapter/7",wait_until="networkidle")
        self.assertEqual(self.page.locator(".memory-evidence").count(),0)

    def test_14_chapter_seven_attention_order_and_validation_scope(self):
        self.visit("#chapter/7")
        panel=self.page.locator(".attention-evidence")
        self.assertEqual(panel.count(),1)
        panel.locator("summary").click()
        self.assertTrue(panel.evaluate("(e)=>e.open"))
        self.assertIn("SHA-256",panel.locator(".evidence-provenance").inner_text())
        retrieval=panel.locator(".attention-retrieval tbody tr")
        self.assertEqual(retrieval.count(),7)
        self.assertIn("79.2%",retrieval.last.inner_text())
        self.assertIn("0.7113",retrieval.last.inner_text())
        order=panel.locator(".attention-order tbody tr")
        self.assertEqual(order.count(),6)
        self.assertIn("100.0%",order.nth(4).inner_text())
        self.assertIn("50.0%",order.nth(4).inner_text())
        self.assertIn("1.4191",order.nth(5).inner_text())
        panel.locator(".attention-delay").select_option("16")
        self.assertIn("1.4034",order.nth(5).inner_text())
        self.assertIn("91.1%",panel.locator(".attention-validation tbody tr").nth(1).inner_text())
        self.assertIn("تحقق استكشافي فقط",panel.locator(".attention-validation caption").inner_text())
        trace=panel.locator(".attention-trace tbody tr")
        self.assertEqual(trace.count(),4)
        panel.locator(".attention-scale").select_option("0")
        self.assertIn("0.0000 / 0.0000",panel.locator(".attention-output").inner_text())
        panel.locator(".attention-bias").select_option("4")
        self.assertIn("0.9306 / -0.9306",panel.locator(".attention-output").inner_text())
        panel.locator(".attention-shift").select_option("1")
        self.assertIn("0.0000 / 0.0000",panel.locator(".attention-output").inner_text())
        panel.locator(".attention-index").select_option("2")
        self.assertIn("0.9306 / -0.9306",panel.locator(".attention-output").inner_text())
        self.page.locator("#locale").select_option("en")
        panel=self.page.locator(".attention-evidence")
        self.assertTrue(panel.evaluate("(e)=>e.open"))
        self.assertIn("exploratory validation only",panel.locator(".attention-validation caption").inner_text())
        self.assertIn("Higher accuracy need not mean better probabilities",panel.inner_text())
        self.assertIn("79.2%",panel.locator(".attention-retrieval tbody tr").last.inner_text())
        self.page.goto(BASE+"/#chapter/8",wait_until="networkidle")
        self.assertEqual(self.page.locator(".attention-evidence").count(),0)

    def test_15_chapter_eight_causal_mask_and_confirmation(self):
        self.visit("#chapter/8")
        panel=self.page.locator(".causal-evidence")
        self.assertEqual(panel.count(),1)
        panel.locator("summary").click()
        self.assertTrue(panel.evaluate("(e)=>e.open"))
        self.assertEqual(panel.locator(".evidence-provenance").count(),2)
        main=panel.locator(".causal-main tbody tr")
        self.assertEqual(main.count(),6)
        self.assertIn("100.00%",main.nth(0).inner_text())
        self.assertIn("87.50%",main.nth(1).inner_text())
        panel.locator(".causal-split").select_option("test_ood")
        self.assertIn("96.88%",main.nth(0).inner_text())
        self.assertIn("100.00%",main.nth(2).inner_text())
        confirm=panel.locator(".causal-confirm tbody tr")
        self.assertEqual(confirm.count(),4)
        self.assertIn("992",confirm.nth(3).inner_text())
        panel.locator(".causal-confirm-split").select_option("confirm_ood")
        self.assertIn("97.14%",confirm.nth(2).inner_text())
        self.assertIn("95.83%",confirm.nth(3).inner_text())
        trace=panel.locator(".causal-trace tbody tr")
        self.assertEqual(trace.count(),4)
        self.assertEqual(trace.nth(2).locator("td").last.inner_text(),"0.000000")
        initial=panel.locator(".causal-output").inner_text()
        self.assertIn("0.761594",initial)
        panel.locator(".causal-future1").select_option("-2")
        self.assertEqual(panel.locator(".causal-output").inner_text(),initial)
        panel.locator(".causal-mask").select_option("0")
        self.assertNotEqual(panel.locator(".causal-output").inner_text(),initial)
        self.assertNotEqual(trace.nth(2).locator("td").last.inner_text(),"0.000000")
        panel.locator(".causal-mask").select_option("1")
        self.assertEqual(panel.locator(".causal-output").inner_text(),initial)
        self.page.locator("#locale").select_option("en")
        panel=self.page.locator(".causal-evidence")
        self.assertTrue(panel.evaluate("(e)=>e.open"))
        self.assertIn("new confirmation set",panel.locator(".causal-confirm caption").inner_text())
        self.assertIn("not a never-before-read test",panel.inner_text())
        self.assertIn("not a full Transformer",panel.inner_text())
        self.page.goto(BASE+"/#chapter/9",wait_until="networkidle")
        self.assertEqual(self.page.locator(".causal-evidence").count(),0)

    def test_16_chapter_nine_binding_and_count_shortcut(self):
        self.visit("#chapter/9")
        panel=self.page.locator(".binding-evidence")
        self.assertEqual(panel.count(),1)
        panel.locator("summary").click()
        self.assertTrue(panel.evaluate("(e)=>e.open"))
        self.assertIn("SHA-256",panel.locator(".evidence-provenance").inner_text())
        rows=panel.locator(".binding-results tbody tr")
        self.assertEqual(rows.count(),7)
        self.assertIn("93.75%",rows.nth(5).inner_text())
        self.assertIn("75.00%",rows.nth(5).inner_text())
        panel.locator(".binding-seed").select_option("44")
        self.assertIn("100.00%",rows.nth(5).inner_text())
        panel.locator(".binding-seed").select_option("mean")
        panel.locator(".binding-split").select_option("confirm_in")
        self.assertIn("74.48%",rows.nth(1).inner_text())
        panel.locator(".binding-split").select_option("final_ood16")
        self.assertIn("93.75%",rows.nth(5).inner_text())
        self.assertEqual(panel.locator(".binding-free tbody tr").count(),7)
        self.assertEqual(panel.locator(".binding-interventions tbody tr").count(),7)
        output=panel.locator(".binding-output")
        self.assertIn("75.00%",output.inner_text())
        self.assertIn("0.00%",output.inner_text())
        self.assertIn("100.00%",output.inner_text())
        self.assertEqual(panel.locator(".binding-trace tbody tr").count(),4)
        panel.locator(".binding-prior").select_option("flipped")
        self.assertNotIn("100.00%",output.inner_text())
        self.assertIn("خارج المجال الثنائي",panel.locator(".binding-trace").inner_text())
        panel.locator(".binding-query").select_option("B")
        self.assertIn("75.00%",output.inner_text())
        self.page.locator("#locale").select_option("en")
        panel=self.page.locator(".binding-evidence")
        self.assertTrue(panel.evaluate("(e)=>e.open"))
        self.assertIn("not all fixed-query architectures",panel.inner_text())
        self.assertIn("seed 42 only",panel.locator(".binding-interventions caption").inner_text())
        self.page.goto(BASE+"/#chapter/1",wait_until="networkidle")
        self.assertEqual(self.page.locator(".binding-evidence").count(),0)
        self.assertEqual(self.page.locator(".classification-evidence").count(),1)

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
