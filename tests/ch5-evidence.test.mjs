import test from "node:test";
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {createHash} from "node:crypto";
import {toyScalarEmbeddingUpdate} from "../src/models.mjs";
import {projectFrozenChapter5Evidence} from "../scripts/frozen-evidence.mjs";

const sourcePath="content/chapter-05/results.json";
const root="baseline/v0.21.3/";
const original=readFileSync(root+sourcePath);
const manifest=JSON.parse(readFileSync(root+"PUBLIC-MANIFEST.json","utf8"));
const changed=()=>JSON.parse(original.toString("utf8"));
const fixture=raw=>{
 const bytes=Buffer.from(JSON.stringify(raw));
 const digest=createHash("sha256").update(bytes).digest("hex");
 return {bytes,trust:{...manifest,files:{...manifest.files,[sourcePath]:digest}}};
};
const project=()=>projectFrozenChapter5Evidence(original,manifest);

test("Chapter 5 published metrics are SHA-pinned and bilingual",()=>{
 const p=project();
 assert.equal(p.schema,"wakeel-v022-frozen-embedding-ablation/v1");
 assert.equal(p.chapter_id,"CH-5");
 assert.equal(p.source_path,sourcePath);
 assert.equal(p.source_sha256,manifest.files[sourcePath]);
 assert.equal(p.status,"approved-source-metrics-not-independent-review");
 assert.deepEqual(p.datasets.ar.learned,{train_loss:1.9891,validation_loss:2.5579,test_loss:2.7634});
 assert.deepEqual(p.datasets.en.learned,{train_loss:1.9084,validation_loss:2.1609,test_loss:2.1759});
 assert.deepEqual(p.datasets.ar.frozen,{train_loss:3.025,validation_loss:3.0632,test_loss:3.1371});
 assert.deepEqual(p.datasets.en.frozen,{train_loss:2.961,validation_loss:2.8913,test_loss:2.9648});
});

test("Chapter 5 parameter accounting follows documented architecture",()=>{
 const p=project();
 assert.deepEqual(p.datasets.ar.architecture,{
  context_vocabulary:39,output_vocabulary:38,context_order:2,embedding_dim:8,hidden:32
 });
 assert.deepEqual(p.datasets.ar.parameters,{
  total:2110,embedding:312,hidden:544,output:1254,trainable_frozen_embeddings:1798
 });
 assert.deepEqual(p.datasets.en.parameters,{
  total:2028,embedding:296,hidden:544,output:1188,trainable_frozen_embeddings:1732
 });
});

test("Chapter 5 source learning curves and seen/unseen validation strata reconcile",()=>{
 const p=project();
 assert.deepEqual(p.datasets.ar.training_curve.map(x=>x.epoch),[1,10,50,100,200,300]);
 assert.deepEqual(p.datasets.ar.training_curve.map(x=>x.loss),[
  3.638,3.4085,3.0588,2.8013,2.2843,1.9891
 ]);
 assert.deepEqual(p.datasets.ar.validation_contexts,{
  seen:{count:102,loss:2.2335},unseen:{count:31,loss:3.6251}
 });
 assert.deepEqual(p.datasets.en.validation_contexts,{
  seen:{count:148,loss:2.0859},unseen:{count:15,loss:2.9005}
 });
 for(const lang of ["ar","en"]){
  const values=p.datasets[lang];
  const v=values.validation_contexts;
  const average=(v.seen.count*v.seen.loss+v.unseen.count*v.unseen.loss)/
    (v.seen.count+v.unseen.count);
  assert.ok(Math.abs(average-values.learned.validation_loss)<0.0002);
 }
});

test("Chapter 5 public projection excludes raw scientific datasets and generations",()=>{
 const p=project(),serialized=JSON.stringify(p);
 assert.deepEqual(Object.keys(p).sort(),[
  "schema","chapter_id","status","source_path","source_sha256","datasets"
 ].sort());
 for(const field of ['"generation"','"prompt_chars"','"learned"','"frozen_embedding_ablation"',
  '"seed"','"timing_seconds"','"input_hash"','"units"','"note_ar"',
  '"simulator_version"','"activation"','"total_seconds"']){
  if(field==='"learned"')continue; // only allow aggregate learned losses, not generated text
  assert.equal(serialized.includes(field),false,field);
 }
});

test("Chapter 5 fails closed on mutated bytes, wrong version and simulator/input provenance",()=>{
 const tampered=Buffer.from(original);tampered[0]=0x5b;
 assert.throws(()=>projectFrozenChapter5Evidence(tampered,manifest),/SHA-256 mismatch/);
 assert.throws(()=>projectFrozenChapter5Evidence(original,{...manifest,source_version:"v0.21.2"}),
   /source version/);
 assert.throws(()=>projectFrozenChapter5Evidence(original,{...manifest,files:{}}),
   /missing source SHA-256/);
 let raw=changed();raw.input_hash="f".repeat(64);
 let f=fixture(raw);
 assert.throws(()=>projectFrozenChapter5Evidence(f.bytes,f.trust),/simulator\/input provenance/);
 raw=changed();raw.simulator_version="wrong";f=fixture(raw);
 assert.throws(()=>projectFrozenChapter5Evidence(f.bytes,f.trust),/simulator\/input provenance/);
});

test("Chapter 5 refuses re-hashed fixtures with inconsistent trainable parameters",()=>{
 let raw=changed();raw.languages.ar.params.embedding=311;
 let f=fixture(raw);
 assert.throws(()=>projectFrozenChapter5Evidence(f.bytes,f.trust),/parameter accounting mismatch ar/);
 raw=changed();raw.languages.en.frozen_embedding_ablation.trainable_params=2028;
 f=fixture(raw);
 assert.throws(()=>projectFrozenChapter5Evidence(f.bytes,f.trust),/parameter accounting mismatch en/);
 raw=changed();raw.languages.ar.architecture.hidden=24;
 f=fixture(raw);
 assert.throws(()=>projectFrozenChapter5Evidence(f.bytes,f.trust),/unexpected architecture ar/);
});

test("Chapter 5 refuses distorted curves and nonreconciling validation strata",()=>{
 let raw=changed();raw.languages.en.training_curve.at(-1).train_nll=0.5;
 let f=fixture(raw);
 assert.throws(()=>projectFrozenChapter5Evidence(f.bytes,f.trust),/training curve\/final loss mismatch en/);
 raw=changed();raw.languages.ar.val_seen_unseen_split.unseen_count=30;
 f=fixture(raw);
 assert.throws(()=>projectFrozenChapter5Evidence(f.bytes,f.trust),/validation split mismatch ar/);
 raw=changed();raw.languages.en.val_seen_unseen_split.unseen_nll=10;
 f=fixture(raw);
 assert.throws(()=>projectFrozenChapter5Evidence(f.bytes,f.trust),/weighted validation loss mismatch en/);
 raw=changed();delete raw.languages.en;
 f=fixture(raw);
 assert.throws(()=>projectFrozenChapter5Evidence(f.bytes,f.trust),/missing language en/);
});

test("scalar example computes a real gradient step, keeping the frozen parameter fixed",()=>{
 const p=toyScalarEmbeddingUpdate(0,1,0.5,1);
 assert.deepEqual(p,{
  initial_loss:0.5,learned_value:0.5,learned_loss:0.125,
  frozen_value:0,frozen_loss:0.5
 });
 const more=toyScalarEmbeddingUpdate(0,1,0.5,2);
 assert.equal(more.learned_value,0.75);
 assert.equal(more.learned_loss,0.03125);
 assert.equal(more.frozen_value,0);
 assert.equal(more.frozen_loss,0.5);
 const same=toyScalarEmbeddingUpdate(0,1,0.5,0);
 assert.equal(same.learned_value,same.frozen_value);
 assert.equal(same.learned_loss,same.frozen_loss);
});

test("bounded scalar update never claims improvement when frozen or rate zero",()=>{
 const p=toyScalarEmbeddingUpdate(-1,2,0,20);
 assert.equal(p.learned_value,-1);
 assert.equal(p.learned_loss,p.frozen_loss);
 const optimal=toyScalarEmbeddingUpdate(0,1,1,1);
 assert.equal(optimal.learned_value,1);
 assert.equal(optimal.learned_loss,0);
});

test("scalar illustration rejects invalid and unbounded inputs",()=>{
 for(const args of [
  [-2.1,1,0.5,1],[2.1,1,0.5,1],[0,-3,0.5,1],[0,3,0.5,1],
  [NaN,1,0.5,1],[Infinity,1,0.5,1],[0,1,0.01,1],
  [0,1,-0.1,1],[0,1,0.5,-1],[0,1,0.5,21],[0,1,0.5,2.5],
  ["0",1,0.5,1]
 ])assert.throws(()=>toyScalarEmbeddingUpdate(...args),RangeError,String(args));
});
