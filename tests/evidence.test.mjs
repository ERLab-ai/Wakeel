import test from "node:test";
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {createHash} from "node:crypto";
import {projectFrozenChapter1Evidence} from "../scripts/frozen-evidence.mjs";

const path="content/chapter-01/results.json";
const base="baseline/v0.21.3/";
const original=readFileSync(base+path);
const manifest=JSON.parse(readFileSync(base+"PUBLIC-MANIFEST.json","utf8"));
const digest=bytes=>createHash("sha256").update(bytes).digest("hex");

test("Chapter 1 reference evidence is derived from original SHA-verified bytes",()=>{
 const output=projectFrozenChapter1Evidence(original,manifest);
 assert.equal(output.schema,"wakeel-v022-frozen-aggregate/v1");
 assert.equal(output.chapter_id,"CH-1");
 assert.equal(output.source_path,path);
 assert.equal(output.source_sha256,manifest.files[path]);
 assert.equal(output.status,"approved-source-aggregate-not-independent-review");
 assert.deepEqual(output.datasets.ar,{
  seed:{total:6,rules_correct:6,nearest_example_correct:6},
  tests:{total:9,rules_correct:4,nearest_example_correct:8}
 });
 assert.deepEqual(output.datasets.en,output.datasets.ar);
});

test("public evidence projection excludes individual messages, IDs and research logs",()=>{
 const out=projectFrozenChapter1Evidence(original,manifest);
 const serialized=JSON.stringify(out);
 for(const forbidden of [
  "rule_rows","example_rows","matched_keyword","nearest_text",
  "shared_words","runs.log","S1","T1","matched_rule"
 ])assert.equal(serialized.includes('"'+forbidden+'"'),false,forbidden);
 assert.deepEqual(Object.keys(out).sort(),[
  "chapter_id","datasets","schema","source_path","source_sha256","status"
 ].sort());
});

test("refuse a byte-modified scientific results file or mismatched source manifest",()=>{
 const mutated=Buffer.from(original);
 mutated[0]=mutated[0]===0x7b?0x5b:0x7b;
 assert.throws(()=>projectFrozenChapter1Evidence(mutated,manifest),/SHA-256 mismatch/);
 assert.throws(()=>projectFrozenChapter1Evidence(original,{...manifest,source_version:"v0.21.2"}),/source version/);
 assert.throws(()=>projectFrozenChapter1Evidence(original,{...manifest,files:{}}),/source SHA-256/);
});

test("even a re-hashed synthetic fixture cannot contradict row-level correctness",()=>{
 const fake=JSON.parse(original.toString("utf8"));
 fake.datasets.ar.tests.rule_correct=5;
 const bytes=Buffer.from(JSON.stringify(fake));
 const trusted={...manifest,files:{...manifest.files,[path]:digest(bytes)}};
 assert.throws(()=>projectFrozenChapter1Evidence(bytes,trusted),/aggregate\/row mismatch/);
});

test("reject missing English evidence, invalid counts and non-JSON bytes",()=>{
 const fake=JSON.parse(original.toString("utf8"));
 delete fake.datasets.en;
 let bytes=Buffer.from(JSON.stringify(fake));
 let trusted={...manifest,files:{...manifest.files,[path]:digest(bytes)}};
 assert.throws(()=>projectFrozenChapter1Evidence(bytes,trusted),/missing language en/);
 fake.datasets.en=JSON.parse(original.toString("utf8")).datasets.en;
 fake.datasets.ar.seed.total=-1;
 bytes=Buffer.from(JSON.stringify(fake));
 trusted={...manifest,files:{...manifest.files,[path]:digest(bytes)}};
 assert.throws(()=>projectFrozenChapter1Evidence(bytes,trusted),/invalid total/);
 bytes=Buffer.from("not json");
 trusted={...manifest,files:{...manifest.files,[path]:digest(bytes)}};
 assert.throws(()=>projectFrozenChapter1Evidence(bytes,trusted),/invalid JSON/);
});
