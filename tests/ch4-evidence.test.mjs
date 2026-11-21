import test from "node:test";
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {createHash} from "node:crypto";
import {denseContextTableCells} from "../src/models.mjs";
import {projectFrozenChapter4Evidence} from "../scripts/frozen-evidence.mjs";

const file="content/chapter-04/results.json";
const root="baseline/v0.21.3/";
const original=readFileSync(root+file);
const manifest=JSON.parse(readFileSync(root+"PUBLIC-MANIFEST.json","utf8"));
const fixture=source=>{
 const bytes=Buffer.from(JSON.stringify(source));
 const digest=createHash("sha256").update(bytes).digest("hex");
 return {bytes,trust:{...manifest,files:{...manifest.files,[file]:digest}}};
};
const altered=()=>JSON.parse(original.toString("utf8"));

test("Chapter 4 reference has exact verified source identity and four context orders",()=>{
 const projected=projectFrozenChapter4Evidence(original,manifest);
 assert.equal(projected.schema,"wakeel-v022-frozen-context-orders/v1");
 assert.equal(projected.chapter_id,"CH-4");
 assert.equal(projected.source_sha256,manifest.files[file]);
 assert.equal(projected.status,"approved-source-metrics-not-independent-review");
 assert.equal(projected.source_path,file);
 assert.deepEqual(projected.datasets.ar.orders.map(x=>x.k),[0,1,2,3]);
 assert.deepEqual(projected.datasets.en.orders.map(x=>x.k),[0,1,2,3]);
 assert.deepEqual([projected.datasets.ar.context_vocabulary,
  projected.datasets.ar.output_vocabulary],[39,38]);
 assert.deepEqual([projected.datasets.en.context_vocabulary,
  projected.datasets.en.output_vocabulary],[37,36]);
});

test("Chapter 4 frozen results demonstrate scoped sparsity trade-off, not universal optimality",()=>{
 const p=projectFrozenChapter4Evidence(original,manifest);
 const a=p.datasets.ar.orders,e=p.datasets.en.orders;
 assert.deepEqual(a.map(x=>x.test_loss),[3.1757,2.638,2.2239,2.3961]);
 assert.deepEqual(e.map(x=>x.test_loss),[3.0255,2.2234,1.9494,2.0234]);
 assert.deepEqual(a.map(x=>x.unseen_validation_rate),[0,0.0075,0.2331,0.3985]);
 assert.deepEqual(e.map(x=>x.unseen_validation_rate),[0,0.0123,0.092,0.2699]);
 assert.deepEqual(a.map(x=>x.dense_cells),[38,1482,57798,2254122]);
 assert.deepEqual(e.map(x=>x.dense_cells),[36,1332,49284,1823508]);
 assert.deepEqual(a.map(x=>x.sparse_records),[37,202,318,373]);
 assert.deepEqual(e.map(x=>x.sparse_records),[35,194,325,392]);
 assert.deepEqual(a.map(x=>x.selected_alpha),[1,0.5,0.1,0.1]);
 assert.deepEqual(e.map(x=>x.selected_alpha),[1,0.1,0.1,0.1]);
});

test("Chapter 4 projection cannot leak unnormalized Backoff scores or individual messages",()=>{
 const out=projectFrozenChapter4Evidence(original,manifest);
 const str=JSON.stringify(out);
 for(const fragment of ['"backoff"','"case_studies"','"generation"','"training_followers"',
   '"message"','"true_next"','"input_hash"','"val_losses_grid"','"train_nll"',
   '"generated_from"','"text"','"prompt"','"context_training_count"']){
  assert.equal(str.includes(fragment),false,fragment);
 }
 assert.deepEqual(Object.keys(out).sort(),[
  "schema","chapter_id","status","source_path","source_sha256","datasets"
 ].sort());
});

test("Chapter 4 rejects modified or incorrect source provenance",()=>{
 const tampered=Buffer.from(original);tampered[0]=0x5b;
 assert.throws(()=>projectFrozenChapter4Evidence(tampered,manifest),/SHA-256 mismatch/);
 assert.throws(()=>projectFrozenChapter4Evidence(original,{...manifest,source_version:"v0.21.2"}),
  /source version/);
 const raw=altered();raw.input_hash="0".repeat(64);
 const f=fixture(raw);
 assert.throws(()=>projectFrozenChapter4Evidence(f.bytes,f.trust),/simulator\/input provenance/);
 const raw2=altered();raw2.simulator_version="wrong";
 const g=fixture(raw2);
 assert.throws(()=>projectFrozenChapter4Evidence(g.bytes,g.trust),/simulator\/input provenance/);
});

test("Chapter 4 rejects corrupt data, missing bilingual rows and impossible dense size",()=>{
 let data=altered();delete data.languages.en.models.k3;
 let f=fixture(data);
 assert.throws(()=>projectFrozenChapter4Evidence(f.bytes,f.trust),/missing model en\/k3/);
 data=altered();data.languages.ar.models.k2.theoretical_dense_cells++;
 f=fixture(data);
 assert.throws(()=>projectFrozenChapter4Evidence(f.bytes,f.trust),/dense-table mismatch ar\/k2/);
 data=altered();data.languages.en.models.k0.sparse_storage_records=9999;
 f=fixture(data);
 assert.throws(()=>projectFrozenChapter4Evidence(f.bytes,f.trust),/invalid sparse records en\/k0/);
 data=altered();data.languages.ar.models.k3.unseen_context_rate_val=1.2;
 f=fixture(data);
 assert.throws(()=>projectFrozenChapter4Evidence(f.bytes,f.trust),/invalid unseen rate ar\/k3/);
 data=altered();data.languages.ar.models.k1.test_nll=-1;
 f=fixture(data);
 assert.throws(()=>projectFrozenChapter4Evidence(f.bytes,f.trust),/invalid loss ar\/k1/);
});

test("Chapter 4 rejects invalid validation selection and unequal evaluation sizes",()=>{
 let raw=altered();raw.languages.ar.models.k2.alpha_from_validation=0.5;
 let f=fixture(raw);
 assert.throws(()=>projectFrozenChapter4Evidence(f.bytes,f.trust),/invalid validation selection ar\/k2/);
 raw=altered();raw.languages.en.models.k1.eval_tokens=281;
 f=fixture(raw);
 assert.throws(()=>projectFrozenChapter4Evidence(f.bytes,f.trust),/unequal test token counts en/);
 raw=altered();raw.languages.en.models.k3.val_losses_grid["1.0"]=null;
 f=fixture(raw);
 assert.throws(()=>projectFrozenChapter4Evidence(f.bytes,f.trust),/invalid validation selection en\/k3/);
});

test("bounded dense-table formula matches scientific reference sizes",()=>{
 assert.equal(denseContextTableCells(39,38,0),38);
 assert.equal(denseContextTableCells(39,38,1),1482);
 assert.equal(denseContextTableCells(39,38,2),57798);
 assert.equal(denseContextTableCells(39,38,3),2254122);
 assert.equal(denseContextTableCells(37,36,2),49284);
 assert.equal(denseContextTableCells(37,36,3),1823508);
 assert.equal(denseContextTableCells(2,3,3),24);
});

test("dense-table calculation refuses invalid or unbounded inputs",()=>{
 for(const args of [
  [1,38,2],[65,38,2],[39,1,2],[39,65,2],
  [39,38,-1],[39,38,4],[39,38,1.5],
  [39.1,38,2],[39,38.2,2],[NaN,38,2],[Infinity,38,2]
 ])assert.throws(()=>denseContextTableCells(...args),RangeError,JSON.stringify(args));
});
