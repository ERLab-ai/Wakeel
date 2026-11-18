import test from "node:test";
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {createHash} from "node:crypto";
import {smoothedTransition} from "../src/models.mjs";
import {projectFrozenChapter3Evidence} from "../scripts/frozen-evidence.mjs";

const file="content/chapter-03/results.json";
const base="baseline/v0.21.3/";
const original=readFileSync(base+file);
const manifest=JSON.parse(readFileSync(base+"PUBLIC-MANIFEST.json","utf8"));
const fixture=changed=>{
 const bytes=Buffer.from(JSON.stringify(changed));
 const digest=createHash("sha256").update(bytes).digest("hex");
 return {bytes,trust:{...manifest,files:{...manifest.files,[file]:digest}}};
};

test("Chapter 3 provenance and exact frozen validation/test metrics",()=>{
 const data=projectFrozenChapter3Evidence(original,manifest);
 assert.equal(data.schema,"wakeel-v022-frozen-bigram-metrics/v1");
 assert.equal(data.chapter_id,"CH-3");
 assert.equal(data.status,"approved-source-metrics-not-independent-review");
 assert.equal(data.source_sha256,manifest.files[file]);
 assert.equal(data.source_path,file);
 assert.deepEqual(data.datasets.ar,{
  unigram:{validation_loss:3.131,test_loss:3.1757},
  count_bigram:{validation_loss:2.656,test_loss:2.638,chosen_alpha:0.5},
  neural_bigram:{validation_loss:2.9268,test_loss:3.0616}
 });
 assert.deepEqual(data.datasets.en,{
  unigram:{validation_loss:2.9783,test_loss:3.0255},
  count_bigram:{validation_loss:2.2232,test_loss:2.2234,chosen_alpha:0.1},
  neural_bigram:{validation_loss:2.7885,test_loss:2.9087}
 });
 assert.deepEqual(data.mini_example,{
  alpha:1,total_context_count:1,observed_next_count:1,
  output_vocabulary:6,reported_probability:0.286
 });
});

test("Chapter 3 public projection never includes corpora, token sequences or generated samples",()=>{
 const data=projectFrozenChapter3Evidence(original,manifest);
 const text=JSON.stringify(data);
 for(const token of ["worked_example","mini_text","generation","train_loss","curve",
  "per_message","train","units_actual","count_text","prompt",
  "char_level_check","failure_demo","val_losses","alpha_grid","<unk>"]){
  assert.equal(text.includes('"'+token+'"'),false,token);
 }
 assert.deepEqual(Object.keys(data).sort(),
  ["schema","chapter_id","status","source_path","source_sha256","mini_example","datasets"].sort());
});

test("Chapter 3 rejects corrupted source bytes or missing/incorrect source version",()=>{
 const changed=Buffer.from(original);
 changed[0]=0x5b;
 assert.throws(()=>projectFrozenChapter3Evidence(changed,manifest),/SHA-256 mismatch/);
 assert.throws(()=>projectFrozenChapter3Evidence(original,{...manifest,source_version:"v0.21.2"}),/source version/);
 assert.throws(()=>projectFrozenChapter3Evidence(original,{...manifest,files:{}}),/missing source SHA-256/);
});

test("Chapter 3 rejects missing bilingual metrics and invalid losses, even when fixture is rehashed",()=>{
 let raw=JSON.parse(original.toString("utf8"));
 delete raw.languages.en.models.neural_bigram;
 let f=fixture(raw);
 assert.throws(()=>projectFrozenChapter3Evidence(f.bytes,f.trust),/missing en\/neural_bigram/);
 raw=JSON.parse(original.toString("utf8"));
 raw.languages.ar.models.unigram.test_loss=-1; f=fixture(raw);
 assert.throws(()=>projectFrozenChapter3Evidence(f.bytes,f.trust),/invalid ar\/unigram\/test_loss/);
 raw=JSON.parse(original.toString("utf8"));
 raw.languages.en.models.count_bigram.chosen_alpha=0.5; f=fixture(raw);
 assert.throws(()=>projectFrozenChapter3Evidence(f.bytes,f.trust),/selected alpha\/validation loss mismatch/);
 raw=JSON.parse(original.toString("utf8"));
 raw.languages.en.models.count_bigram.alpha_grid=[0.1,1]; f=fixture(raw);
 assert.throws(()=>projectFrozenChapter3Evidence(f.bytes,f.trust),/validation alpha grid drift/);
});

test("Chapter 3 rejects a corrupted mini example, incorrect origin and invalid JSON",()=>{
 let raw=JSON.parse(original.toString("utf8"));
 raw.worked_example.probabilities["<s>→س"]=0.9;
 let f=fixture(raw);
 assert.throws(()=>projectFrozenChapter3Evidence(f.bytes,f.trust),/worked-example probability mismatch/);
 raw=JSON.parse(original.toString("utf8"));
 raw.generated_from="other.json";f=fixture(raw);
 assert.throws(()=>projectFrozenChapter3Evidence(f.bytes,f.trust),/unexpected evidence origin/);
 const bytes=Buffer.from("not json");
 const digest=createHash("sha256").update(bytes).digest("hex");
 assert.throws(()=>projectFrozenChapter3Evidence(bytes,
  {...manifest,files:{...manifest.files,[file]:digest}}),/invalid JSON/);
});

test("bounded smoothing formula reproduces mini example to the published precision",()=>{
 const p=smoothedTransition(1,1,6,1);
 assert.equal(p.probability,2/7);
 assert.equal(p.probability.toFixed(3),"0.286");
 assert.ok(Math.abs(p.loss-Math.log(7/2))<1e-12);
 assert.equal(smoothedTransition(2,2,6,1).probability,3/8);
});

test("more smoothing lowers the observed event probability in the fixed example",()=>{
 const light=smoothedTransition(1,1,6,0.1);
 const heavy=smoothedTransition(1,1,6,1);
 assert.equal(light.probability,1.1/1.6);
 assert.ok(light.probability>heavy.probability);
 assert.ok(light.loss<heavy.loss);
 assert.ok(Math.abs(smoothedTransition(0,0,6,0.5).probability-1/6)<1e-12);
});

test("single-transition calculator rejects impossible inputs or arbitrary alpha",()=>{
 for(const args of [
  [-1,0,6,1],[101,1,6,1],[2,-1,6,1],[2,3,6,1],
  [2,1,1,1],[2,1,33,1],[2.5,1,6,1],
  [2,1.5,6,1],[2,1,6,0],[2,1,6,0.3],
  [NaN,0,6,1],[Infinity,1,6,1]
 ])assert.throws(()=>smoothedTransition(...args),RangeError,JSON.stringify(args));
});
