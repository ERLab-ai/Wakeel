import test from "node:test";
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {createHash} from "node:crypto";
import {projectFrozenChapter8Evidence} from "../scripts/frozen-evidence.mjs";
import {toyCausalPrefix} from "../src/models.mjs";
const root="baseline/v0.21.3/",pathA="content/chapter-08/results.json",pathB="content/chapter-08/factorial_isolation.json";
const a=readFileSync(root+pathA),b=readFileSync(root+pathB),manifest=JSON.parse(readFileSync(root+"PUBLIC-MANIFEST.json","utf8"));
const project=()=>projectFrozenChapter8Evidence(a,b,manifest);
const decode=x=>JSON.parse(x.toString("utf8"));
const hash=x=>createHash("sha256").update(x).digest("hex");
function rejectChanged(rawA,rawB,pattern){
 const x=Buffer.from(JSON.stringify(rawA));
 // Preserve the source linkage unless the caller intentionally alters it.
 const y=Buffer.from(JSON.stringify(rawB));
 const trust={...manifest,files:{...manifest.files,[pathA]:hash(x),[pathB]:hash(y)}};
 assert.throws(()=>projectFrozenChapter8Evidence(x,y,trust),pattern);
}
test("Chapter 8 main and new confirmation sets retain separate identities",()=>{
 const p=project();
 assert.equal(p.schema,"wakeel-v022-frozen-causal-dual-lookup/v1");
 assert.equal(p.chapter_id,"CH-8");
 assert.equal(p.source_sha256,manifest.files[pathA]);
 assert.equal(p.confirmation_source_sha256,manifest.files[pathB]);
 assert.deepEqual(p.seeds,[42,43,44]);
 assert.equal(p.main.length,6);assert.equal(p.confirmation.length,4);
 const find=k=>p.main.find(x=>x.model===k);
 assert.equal(find("fixed_query_prefix_reader").test_in.binary_accuracy.mean,1);
 assert.equal(find("causal_self_attention_1h").test_in.binary_accuracy.mean,0.875);
 assert.deepEqual(find("causal_self_attention_1h").test_in.binary_accuracy.per_seed,[1,0.875,0.75]);
 assert.equal(find("fixed_query_prefix_reader").test_ood.binary_accuracy.mean,0.9688);
 assert.equal(find("no_position").test_ood.binary_accuracy.mean,1);
 assert.equal(find("no_query_dependency").test_ood.binary_accuracy.mean,0.9505);
 assert.equal(find("local_only").test_in.nll_full_vocab.mean,0.6933);
 assert.equal(find("local_only").test_in.nll_binary_normalized.mean,0.6931);
});
test("Factorial cells retain unequal parameter counts and new-set means",()=>{
 const p=project();
 assert.deepEqual(p.confirmation.map(x=>x.params),[680,736,936,992]);
 assert.deepEqual(p.confirmation.map(x=>x.confirm_in.binary_accuracy.mean),[1,0.875,1,0.9583]);
 assert.deepEqual(p.confirmation.map(x=>x.confirm_ood.binary_accuracy.mean),[0.9505,0.875,0.9714,0.9583]);
 assert.ok(p.confirmation.every(x=>!("test_in" in x)&&!("validation" in x)));
});
test("Projection discloses no raw datasets or training traces",()=>{
 const p=project(),s=JSON.stringify(p);
 assert.deepEqual(Object.keys(p).sort(),["schema","chapter_id","status","source_path","source_sha256","confirmation_source_path","confirmation_source_sha256","seeds","main","confirmation"].sort());
 for(const k of ["curve","train_seconds","attention_maps_val","dataset_sha256","runtime_versions","run_log","provenance","protocol","per_query","gates"])assert.ok(!s.includes('"'+k+'"'),k);
});
test("Both original byte hashes are required",()=>{
 const bad=Buffer.from(a);bad[0]=0x5b;
 assert.throws(()=>projectFrozenChapter8Evidence(bad,b,manifest),/SHA-256 mismatch/);
 const bad2=Buffer.from(b);bad2[0]=0x5b;
 assert.throws(()=>projectFrozenChapter8Evidence(a,bad2,manifest),/SHA-256 mismatch/);
 assert.throws(()=>projectFrozenChapter8Evidence(a,b,{...manifest,source_version:"v0.21.2"}),/source\/version mismatch/);
});
test("Rehashed fixtures reject invalid source linkage and confirmation protocol",()=>{
 let r=decode(a),f=decode(b);
 f.provenance.ch8_results_sha256="0".repeat(64);rejectChanged(r,f,/source linkage mismatch/);
 r=decode(a);f=decode(b);f.protocol.confirmatory_sets.family_overlap_with_v015_segments=1;
 f.provenance.ch8_results_sha256=hash(Buffer.from(JSON.stringify(r)));
 rejectChanged(r,f,/confirmation protocol mismatch/);
});
test("Rehashed fixtures reject future leakage, overlap, inconsistent means",()=>{
 for(const [mutate,pattern] of [
  [r=>r.gates["1_causal_mask_exact"].causal_self_attention_1h.future_weight_max=0.001,/causal mask mismatch/],
  [r=>r.gates["7_dataset_exclusivity"].family_overlap_counts["train|val"]=1,/family overlap/],
  [r=>r.gates["5b_aggregated_gradient"].causal_self_attention_1h.worst_abs_error=0.01,/aggregated gradient mismatch/],
  [r=>r.experiments.final_test.causal_self_attention_1h.mean_across_seeds.test_in_binary_accuracy.mean=1,/mean mismatch/]
 ]){
  const r=decode(a),f=decode(b);mutate(r);
  f.provenance.ch8_results_sha256=hash(Buffer.from(JSON.stringify(r)));rejectChanged(r,f,pattern);
 }
});
test("Causal mask exactly excludes all 25 possible future interventions",()=>{
 const ref=Math.tanh(1);
 for(const x of [-2,-1,0,1,2])for(const y of [-2,-1,0,1,2]){
  const p=toyCausalPrefix(x,y,true);
  assert.deepEqual(p.weights.slice(2),[0,0]);
  assert.equal(p.future_weight,0);
  assert.equal(p.output,p.prefix_output);
  assert.ok(Math.abs(p.output-ref)<1e-12);
  assert.ok(Math.abs(p.weights.reduce((a,b)=>a+b,0)-1)<1e-12);
 }
});
test("Unmasked counter-control changes under future intervention",()=>{
 const p=toyCausalPrefix(2,2,false),q=toyCausalPrefix(-2,-2,false);
 assert.ok(p.future_weight>0&&q.future_weight>0);
 assert.ok(p.output>p.prefix_output&&q.output<q.prefix_output);
 assert.equal(p.prefix_output,q.prefix_output);
 assert.ok(Math.abs(p.weights.reduce((a,b)=>a+b,0)-1)<1e-12);
});
test("Toy causal reader rejects unsupported inputs",()=>{
 for(const args of [[3,0,true],[0,0,1],["1",0,true],[0,NaN,false],[0,0.5,true]])assert.throws(()=>toyCausalPrefix(...args),RangeError);
});
