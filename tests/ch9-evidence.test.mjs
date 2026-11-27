import test from "node:test";
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {createHash} from "node:crypto";
import {projectFrozenChapter9Evidence} from "../scripts/frozen-evidence.mjs";
import {toyBindingCounter} from "../src/models.mjs";
const root="baseline/v0.21.3/",file="content/chapter-09/results.json";
const original=readFileSync(root+file),manifest=JSON.parse(readFileSync(root+"PUBLIC-MANIFEST.json","utf8"));
const project=()=>projectFrozenChapter9Evidence(original,manifest);
const altered=()=>JSON.parse(original.toString("utf8"));
const reject=(raw,pattern)=>{
 const bytes=Buffer.from(JSON.stringify(raw));
 const trust={...manifest,files:{...manifest.files,[file]:createHash("sha256").update(bytes).digest("hex")}};
 assert.throws(()=>projectFrozenChapter9Evidence(bytes,trust),pattern);
};
test("Chapter 9 pins the source and the distinct confirmation/final splits",()=>{
 const p=project();
 assert.equal(p.schema,"wakeel-v022-frozen-compositional-binding/v1");
 assert.equal(p.chapter_id,"CH-9");
 assert.equal(p.source_sha256,manifest.files[file]);
 assert.deepEqual(p.seeds,[42,43,44]);
 assert.equal(p.rows.length,7);
 assert.deepEqual(p.rows.map(x=>x.params),[561,961,753,945,753,945,1017]);
 assert.deepEqual(p.rows.map(x=>x.final_in.first_query_binary_accuracy.mean),[.5,.75,.8125,.8646,.6667,.9375,.75]);
 assert.deepEqual(p.rows.map(x=>x.final_in.first_query_paired_both_correct.mean),[0,0,.3333,.5,0,.75,.0417]);
 assert.deepEqual(p.rows.map(x=>x.confirm_in.first_query_binary_accuracy.mean),[.5,.7448,.8125,.8594,.6667,.9375,.75]);
 assert.deepEqual(p.rows.map(x=>x.final_ood16.first_query_binary_accuracy.mean),[.5,.75,.8125,.8333,.6667,.9375,.75]);
});
test("Count ceilings, seed variation, and descriptive interaction remain distinct",()=>{
 const p=project(),m=p.rows[5];
 assert.deepEqual(p.references,{first_query_count_ceiling:.75,paired_count_ceiling:0,second_query_count_ceiling:1,oracle:1});
 assert.deepEqual(m.final_in.first_query_binary_accuracy.per_seed,[.9375,.875,1]);
 assert.deepEqual(m.final_in.first_query_paired_both_correct.per_seed,[.75,.5,1]);
 assert.deepEqual(p.factorial,{one_layer_with_pe:.8125,one_layer_no_pe:.6667,two_layers_with_pe:.8646,two_layers_no_pe:.9375,descriptive_interaction:.2187});
 assert.equal(p.rows[1].final_in.second_query_binary_accuracy.mean,1);
 assert.equal(p.rows[1].final_in.first_query_paired_both_correct.mean,0);
});
test("Free rollout and single-seed interventions are not teacher-forced aggregates",()=>{
 const p=project(),m=p.interventions_final_in_seed42.find(x=>x.model==="no_position_2h");
 assert.equal(m.baseline,1);
 assert.deepEqual(m.modes,{memory_only:{accuracy:.6875,flip_rate:.3125},prior_answer_only:{accuracy:.6875,flip_rate:.3125},both:{accuracy:1,flip_rate:0}});
 assert.ok(p.rows.every(x=>Object.keys(x.free_rollout_final_in).sort().join(",")==="both,q1,q2"));
});
test("Public projection excludes raw records, traces, and private review metadata",()=>{
 const p=project(),s=JSON.stringify(p);
 for(const k of ["curve","first_query_breakdowns","attention_maps","provenance","dataset_sha256","run_log","protocol","gates","training_runs","spec","budget"])assert.ok(!s.includes('"'+k+'"'),k);
 assert.deepEqual(Object.keys(p).sort(),["schema","chapter_id","status","source_path","source_sha256","seeds","rows","references","factorial","interventions_final_in_seed42"].sort());
});
test("Reject source corruption and version mismatch",()=>{
 const bad=Buffer.from(original);bad[0]=0x5b;
 assert.throws(()=>projectFrozenChapter9Evidence(bad,manifest),/SHA-256 mismatch/);
 assert.throws(()=>projectFrozenChapter9Evidence(original,{...manifest,source_version:"v0.21.2"}),/source\/version mismatch/);
});
test("Reject relabeling of quarantined data, overlap and altered reference ceilings",()=>{
 let x=altered();x.protocol.quarantine.stage_order.reverse();reject(x,/quarantine order mismatch/);
 x=altered();x.gates["2_split_contract"].family_overlap_counts["train|val"]=1;reject(x,/family overlap/);
 x=altered();x.gates["4_references"].blind_post_freeze.segments.final_in.bag_ceiling_first_query=1;reject(x,/reference mismatch/);
});
test("Reject inconsistent metrics, seed means, or descriptive interaction",()=>{
 let x=altered();x.experiments.final_test_post_freeze.no_position_2h.seed_42.final_in.matched_pairs=15;reject(x,/metric mismatch/);
 x=altered();x.experiments.final_test_post_freeze.no_position_2h.mean_across_seeds.final_in_first_query_binary_accuracy.mean=.5;reject(x,/mean mismatch/);
 x=altered();x.experiments.factorial_reading_descriptive.descriptive_interaction=0;reject(x,/descriptive factorial mismatch/);
});
test("Enumerated count-only classifier hits first-query ceiling but cannot solve both matched counterparts",()=>{
 for(const query of ["A","B"]){
  const x=toyBindingCounter(query,"true");
  assert.equal(x.rows.length,4);assert.equal(x.first_accuracy,.75);
  assert.equal(x.pair_both_correct,0);assert.equal(x.second_accuracy,1);assert.equal(x.invalid_second,0);
  const pair=x.rows.filter(r=>r.a!==r.b);
  assert.equal(pair[0].sum,pair[1].sum);assert.equal(pair[0].counter_first,pair[1].counter_first);
  assert.notEqual(pair[0].first,pair[1].first);
 }
});
test("Flipped displayed prior breaks the arithmetic shortcut without changing first-query accuracy",()=>{
 for(const query of ["A","B"]){
  const x=toyBindingCounter(query,"flipped");
  assert.equal(x.first_accuracy,.75);assert.equal(x.pair_both_correct,0);
  assert.equal(x.second_accuracy,0);assert.equal(x.invalid_second,2);
 }
 assert.throws(()=>toyBindingCounter("C","true"),RangeError);
 assert.throws(()=>toyBindingCounter("A",true),RangeError);
});
