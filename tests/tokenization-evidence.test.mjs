import test from "node:test";
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {createHash} from "node:crypto";
import {projectFrozenChapter2Evidence} from "../scripts/frozen-evidence.mjs";

const sourcePath="content/chapter-02/results.json";
const base="baseline/v0.21.3/";
const original=readFileSync(base+sourcePath);
const manifest=JSON.parse(readFileSync(base+"PUBLIC-MANIFEST.json","utf8"));
const sha=bytes=>createHash("sha256").update(bytes).digest("hex");
const fixture=raw=>{
 const bytes=Buffer.from(JSON.stringify(raw));
 return {bytes,trust:{...manifest,files:{...manifest.files,[sourcePath]:sha(bytes)}}};
};

test("Chapter 2 counts originate from manifest SHA-pinned reference outputs",()=>{
 const data=projectFrozenChapter2Evidence(original,manifest);
 assert.equal(data.source_sha256,manifest.files[sourcePath]);
 assert.equal(data.source_path,sourcePath);
 assert.equal(data.chapter_id,"CH-2");
 assert.equal(data.tokenizer,"edu-tok-0.1");
 assert.equal(data.schema,"wakeel-v022-frozen-tokenization/v1");
 assert.equal(data.status,"approved-source-counts-not-independent-review");
 assert.deepEqual(data.samples,{
  ar:{whitespace_units:5,educational_tokenizer_units:8},
  en:{whitespace_units:7,educational_tokenizer_units:9}
 });
});

test("Chapter 2 output contains only allowed counts and provenance",()=>{
 const data=projectFrozenChapter2Evidence(original,manifest);
 const str=JSON.stringify(data);
 for(const bad of ["وسيشاركهم","tomorrow's","ws_keep_tokens","edu_tokens",
   "normalization_ar","template","vocab","sections","runs.log"]){
  assert.equal(str.includes(bad),false,bad);
 }
 assert.deepEqual(Object.keys(data).sort(),
  ["chapter_id","samples","schema","source_path","source_sha256","status","tokenizer"].sort());
});

test("Chapter 2 rejects source byte changes, wrong version and missing manifest digest",()=>{
 const mutated=Buffer.from(original);mutated[0]=0x5b;
 assert.throws(()=>projectFrozenChapter2Evidence(mutated,manifest),/SHA-256 mismatch/);
 assert.throws(()=>projectFrozenChapter2Evidence(original,{...manifest,source_version:"v0.21.2"}),/source version/);
 assert.throws(()=>projectFrozenChapter2Evidence(original,{...manifest,files:{}}),/missing source SHA-256/);
});

test("Chapter 2 rejects inconsistent count fields even with a fixture-rehashed source",()=>{
 let raw=JSON.parse(original.toString("utf8"));
 raw.sections.tokenization.ar.edu_count=7;
 let f=fixture(raw);
 assert.throws(()=>projectFrozenChapter2Evidence(f.bytes,f.trust),/count\/token array mismatch/);
 raw=JSON.parse(original.toString("utf8"));
 raw.sections.tokenization.en.ws_keep_tokens.pop();
 f=fixture(raw);
 assert.throws(()=>projectFrozenChapter2Evidence(f.bytes,f.trust),/count\/token array mismatch/);
 raw=JSON.parse(original.toString("utf8"));
 raw.sections.tokenization.ar.ws_keep_count=-1;
 f=fixture(raw);
 assert.throws(()=>projectFrozenChapter2Evidence(f.bytes,f.trust),/invalid ar\/ws_keep_count/);
});

test("Chapter 2 rejects wrong tokenizer, absent language, invalid and unrecognized source",()=>{
 let raw=JSON.parse(original.toString("utf8"));
 raw.pinned_tokenizer="unknown-tokenizer";let f=fixture(raw);
 assert.throws(()=>projectFrozenChapter2Evidence(f.bytes,f.trust),/tokenizer version mismatch/);
 raw=JSON.parse(original.toString("utf8"));
 delete raw.sections.tokenization.en; f=fixture(raw);
 assert.throws(()=>projectFrozenChapter2Evidence(f.bytes,f.trust),/missing reference sentence en/);
 raw=JSON.parse(original.toString("utf8"));
 raw.generated_from="../unapproved.json";f=fixture(raw);
 assert.throws(()=>projectFrozenChapter2Evidence(f.bytes,f.trust),/unexpected evidence origin/);
 const bytes=Buffer.from("bad json");
 assert.throws(()=>projectFrozenChapter2Evidence(bytes,fixture(raw).trust),/SHA-256 mismatch/);
});
