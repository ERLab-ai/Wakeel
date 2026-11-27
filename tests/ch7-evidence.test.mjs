import test from "node:test";
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {createHash} from "node:crypto";
import {projectFrozenChapter7Evidence} from "../scripts/frozen-evidence.mjs";
import {toyAttentionOrder} from "../src/models.mjs";
const root="baseline/v0.21.3/",file="content/chapter-07/results.json";
const original=readFileSync(root+file),manifest=JSON.parse(readFileSync(root+"PUBLIC-MANIFEST.json","utf8"));
const project=()=>projectFrozenChapter7Evidence(original,manifest);
const changed=()=>JSON.parse(original.toString("utf8"));
const fixture=raw=>{
 const bytes=Buffer.from(JSON.stringify(raw));
 return {bytes,trust:{...manifest,files:{...manifest.files,[file]:createHash("sha256").update(bytes).digest("hex")}}};
};
const reject=(raw,pattern)=>{const f=fixture(raw);assert.throws(()=>projectFrozenChapter7Evidence(f.bytes,f.trust),pattern);};
test("Chapter 7 verifies source and keeps the retrieval shortcut evidence",()=>{
 const p=project();
 assert.equal(p.schema,"wakeel-v022-frozen-attention-order/v1");
 assert.equal(p.chapter_id,"CH-7");
 assert.equal(p.source_sha256,manifest.files[file]);
 assert.deepEqual(p.seeds,[42,43,44]);
 assert.deepEqual(p.retrieval.map(x=>x.delay),[2,4,5,6,7,8,16]);
 assert.deepEqual(p.retrieval.map(x=>x.rnn.accuracy),[1,1,0.5,0.5,0.5,0.5,0.5]);
 assert.deepEqual(p.retrieval.map(x=>x.attn_content.accuracy),[1,1,1,1,1,1,0.792]);
 const r=p.retrieval.at(-1);
 assert.equal(r.attn_content.nll,0.7113);
 assert.ok(r.attn_content.nll>Math.log(2));
 assert.equal(r.rnn.nll,0.7025);
 assert.deepEqual(r.attn_content.runs.map(x=>x.accuracy),[0.8,0.738,0.838]);
 assert.ok(p.retrieval.every(x=>x.oracle_accuracy===1));
});
test("Order holdout preserves negative findings and distinct known-position controls",()=>{
 const p=project();
 assert.deepEqual(p.order.map(x=>x.delay),[4,8,16]);
 for(const row of p.order){
  for(const m of ["rnn","attn_pos","attn_nopos","mean"]){
   assert.equal(row.models[m].accuracy,0.5);assert.equal(row.models[m].shifted_accuracy,0.5);
  }
  assert.equal(row.models.pos0.accuracy,1);assert.equal(row.models.pos0.shifted_accuracy,0.5);
  assert.equal(row.models.pos_mid_shift.accuracy,0.5);assert.equal(row.models.pos_mid_shift.shifted_accuracy,1);
 }
});
test("Initialization sensitivity is explicitly validation only",()=>{
 const p=project();assert.deepEqual(p.validation_sensitivity.map(x=>x.variants.all3),[1,1,0.911]);
 assert.deepEqual(p.validation_sensitivity.map(x=>x.variants.amp1),[0.5,0.5,0.5]);
 assert.deepEqual(p.validation_sensitivity.map(x=>x.variants.kq_only3),[0.5,0.517,0.506]);
 assert.deepEqual(p.validation_sensitivity.map(x=>x.variants.v_only3),[0.5,0.506,0.5]);
 assert.ok(p.validation_sensitivity.every(x=>x.variants.pos_only3===0.5));
 assert.ok(!JSON.stringify(p.validation_sensitivity).includes("holdout"));
});
test("Projection excludes raw datasets and internal training data",()=>{
 const p=project(),str=JSON.stringify(p);
 assert.deepEqual(Object.keys(p).sort(),["schema","chapter_id","status","source_path","source_sha256","seeds","retrieval","order","validation_sensitivity"].sort());
 for(const key of ["dataset_fingerprint","curve","grad_norms_at_init","train_seconds","protocol","input_hash","per_class_binary","lr_sweep_seed_42","total_seconds"])assert.ok(!str.includes('"'+key+'"'),key);
});
test("Source hash, version, protocol and split are fail closed",()=>{
 const bad=Buffer.from(original);bad[0]=0x5b;
 assert.throws(()=>projectFrozenChapter7Evidence(bad,manifest),/SHA-256 mismatch/);
 assert.throws(()=>projectFrozenChapter7Evidence(original,{...manifest,source_version:"v0.21.2"}),/source\/version/);
 let r=changed();r.protocol.splits.holdout_pairs=39;reject(r,/protocol mismatch/);
 r=changed();r.input_hash="f".repeat(64);reject(r,/protocol mismatch/);
 r=changed();r.experiments.order.init_sensitivity.eval_set="holdout";reject(r,/sensitivity split mismatch/);
});
test("Invalid aggregate, oracle, paired invariance and positive controls reject rehashed fixtures",()=>{
 let r=changed();r.experiments.retrieval.delays.delay_16.mean_across_seeds.attn_content.holdout_binary=0.9;reject(r,/mean mismatch/);
 r=changed();r.experiments.retrieval.delays.delay_2.seed_42.oracle_ceiling=0;reject(r,/retrieval oracle mismatch/);
 r=changed();r.experiments.order.delays.delay_4.seed_42.paired_invariance_maxdiff.mean=0.1;reject(r,/order controls mismatch/);
 r=changed();r.experiments.order.delays.delay_4.seed_42.pos0.shifted_pair_binary=1;reject(r,/positive control mismatch/);
 r=changed();r.experiments.order.init_sensitivity.delays.delay_16.mean_across_seeds.all3=1;reject(r,/sensitivity mean mismatch/);
});
test("Toy content-only weighted sums are permutation invariant and normalized",()=>{
 for(const shifted of [false,true])for(const scale of [0,1,2,4]){
  const p=toyAttentionOrder(scale,0,0,shifted);
  assert.ok(Math.abs(p.first.weighted_sum-p.second.weighted_sum)<1e-12);
  for(const row of [p.first,p.second])assert.ok(Math.abs(row.weights.reduce((a,b)=>a+b,0)-1)<1e-12);
  assert.equal(p.first.oracle_value,1);assert.equal(p.second.oracle_value,-1);
 }
});
test("Toy supplied position separates counterparts but must move with shifted inputs",()=>{
 let p=toyAttentionOrder(0,4,0,false);
 assert.ok(p.first.weighted_sum>0.9&&p.second.weighted_sum< -0.9);
 p=toyAttentionOrder(0,4,0,true);
 assert.equal(p.first.weighted_sum,0);assert.equal(p.second.weighted_sum,0);
 p=toyAttentionOrder(0,4,2,true);
 assert.ok(p.first.weighted_sum>0.9&&p.second.weighted_sum< -0.9);
 assert.equal(p.target_index,2);
});
test("Toy rejects unbounded or untyped controls",()=>{
 for(const args of [[3,0,0,false],[0,100,0,false],[0,0,-1,false],[0,0,4,false],[0,0,0.5,false],[0,0,0,1],[NaN,0,0,false]])assert.throws(()=>toyAttentionOrder(...args),RangeError);
});
