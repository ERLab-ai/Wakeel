import test from "node:test";
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {createHash} from "node:crypto";
import {projectFrozenChapter6Evidence} from "../scripts/frozen-evidence.mjs";
import {toyMemoryRetention} from "../src/models.mjs";

const root="baseline/v0.21.3/",sourcePath="content/chapter-06/results.json";
const original=readFileSync(root+sourcePath);
const manifest=JSON.parse(readFileSync(root+"PUBLIC-MANIFEST.json","utf8"));
const changed=()=>JSON.parse(original.toString("utf8"));
function fixture(raw){
 const bytes=Buffer.from(JSON.stringify(raw));
 return {bytes,trust:{...manifest,files:{...manifest.files,
  [sourcePath]:createHash("sha256").update(bytes).digest("hex")}}};
}
const project=()=>projectFrozenChapter6Evidence(original,manifest);
const at=(out,delay)=>out.rows.find(x=>x.delay===delay);

test("Chapter 6 evidence originates in the pinned source and retains corpus metrics",()=>{
 const p=project();
 assert.equal(p.schema,"wakeel-v022-frozen-recurrent-memory/v1");
 assert.equal(p.chapter_id,"CH-6");
 assert.equal(p.status,"approved-source-metrics-not-independent-review");
 assert.equal(p.source_sha256,manifest.files[sourcePath]);
 assert.equal(p.source_path,sourcePath);
 assert.deepEqual(p.languages,{
  ar:{parameters:1358,validation_loss:3.1235,test_loss:3.1899},
  en:{parameters:1308,validation_loss:2.9291,test_loss:2.9962}
 });
});

test("Chapter 6 preserves all seven delays and the distinct training protocols",()=>{
 const p=project();
 assert.deepEqual(p.seeds,[42,43,44]);
 assert.deepEqual(p.train_delays,[2,4,5,6,7,8]);
 assert.deepEqual(p.holdout_delays,[16]);
 assert.deepEqual(p.rows.map(x=>x.delay),[2,4,5,6,7,8,16]);
 assert.deepEqual(p.rows.map(x=>x.fixed_accuracy),[1,1,0.667,0.5,0.5,0.5,0.5]);
 assert.deepEqual(p.rows.map(x=>x.mixed_accuracy),Array(7).fill(0.833));
 assert.ok(p.rows.slice(0,-1).every(x=>x.mixed_in_domain));
 assert.equal(at(p,16).mixed_in_domain,false);
 assert.ok(p.rows.every(x=>x.test_per_class===40&&x.oracle_accuracy===1));
});

test("Chapter 6 reports failed seeds rather than hiding them in the mean",()=>{
 const p=project(),r=at(p,16);
 assert.deepEqual(r.seeds.map(x=>x.seed),[42,43,44]);
 assert.deepEqual(r.seeds.map(x=>x.fixed_accuracy),[0.5,0.5,0.5]);
 assert.deepEqual(r.seeds.map(x=>x.mixed_accuracy),[1,0.5,1]);
 assert.equal(r.fixed_nll,0.7024);
 assert.equal(r.mixed_nll,0.3201);
 assert.equal(r.untrained_accuracy,0.517);
 assert.deepEqual(at(p,5).seeds.map(x=>x.fixed_accuracy),[0.5,0.5,1]);
});

test("Chapter 6 discloses only scoped aggregates, not raw samples or provenance internals",()=>{
 const p=project(),str=JSON.stringify(p);
 assert.deepEqual(Object.keys(p).sort(),[
  "schema","chapter_id","status","source_path","source_sha256","seeds",
  "train_delays","holdout_delays","languages","rows"
 ].sort());
 for(const forbidden of ["generation","text","prompt","curve","train_curve_mean",
  "memory_probe","mixed_length_model","provenance","input_hash","test_draws_rejected_as_collisions",
  "filler_alphabet_size","classes","task","total_seconds"]){
  assert.equal(str.includes('"'+forbidden+'"'),false,forbidden);
 }
});

test("Chapter 6 rejects byte tampering and invalid provenance",()=>{
 const bad=Buffer.from(original);bad[0]=0x5b;
 assert.throws(()=>projectFrozenChapter6Evidence(bad,manifest),/SHA-256 mismatch/);
 assert.throws(()=>projectFrozenChapter6Evidence(original,{...manifest,source_version:"v0.21.2"}),/source version/);
 let raw=changed();raw.simulator_version="wrong";let f=fixture(raw);
 assert.throws(()=>projectFrozenChapter6Evidence(f.bytes,f.trust),/simulator\/input provenance/);
 raw=changed();raw.provenance.input_hash="0".repeat(64);f=fixture(raw);
 assert.throws(()=>projectFrozenChapter6Evidence(f.bytes,f.trust),/simulator\/input provenance/);
});

test("Chapter 6 refuses protocol or holdout relabeling even with rehashed fixtures",()=>{
 let raw=changed();raw.mixed_length_model.train_delays.push(16);let f=fixture(raw);
 assert.throws(()=>projectFrozenChapter6Evidence(f.bytes,f.trust),/mixed protocol mismatch/);
 raw=changed();raw.mixed_length_model.means.delay_16.in_domain=true;f=fixture(raw);
 assert.throws(()=>projectFrozenChapter6Evidence(f.bytes,f.trust),/mixed metrics mismatch delay_16/);
 raw=changed();raw.memory_probe.oracle_label_swap_control=1;f=fixture(raw);
 assert.throws(()=>projectFrozenChapter6Evidence(f.bytes,f.trust),/fixed protocol mismatch/);
 raw=changed();raw.memory_probe.delays_data.delay_2.oracle_binary_through_eval_path=0;f=fixture(raw);
 assert.throws(()=>projectFrozenChapter6Evidence(f.bytes,f.trust),/fixed metrics mismatch delay_2/);
});

test("Chapter 6 rejects inconsistent per-seed scores and aggregate losses",()=>{
 let raw=changed();raw.mixed_length_model.per_seed["43"].delay_16.binary=1;let f=fixture(raw);
 assert.throws(()=>projectFrozenChapter6Evidence(f.bytes,f.trust),/mixed per-seed mismatch/);
 raw=changed();raw.memory_probe.delays_data.delay_16.accuracy_binary=0.8;f=fixture(raw);
 assert.throws(()=>projectFrozenChapter6Evidence(f.bytes,f.trust),/fixed metrics mismatch/);
 raw=changed();raw.mixed_length_model.means.delay_16.test_nll_mean=1;f=fixture(raw);
 assert.throws(()=>projectFrozenChapter6Evidence(f.bytes,f.trust),/mixed NLL mismatch/);
 raw=changed();raw.languages.ar.rnn.mean_test=4;f=fixture(raw);
 assert.throws(()=>projectFrozenChapter6Evidence(f.bytes,f.trust),/corpus means mismatch ar/);
});

test("Scalar recurrence has a full deterministic trace and no classifier",()=>{
 const p=toyMemoryRetention(1,0.9,4);
 assert.equal(p.states.length,5);
 assert.ok(Math.abs(p.final_state-0.6561)<1e-12);
 assert.ok(Math.abs(p.magnitude-0.6561)<1e-12);
 const neg=toyMemoryRetention(-1,0.9,16);
 assert.ok(neg.final_state<0);
 assert.ok(Math.abs(neg.magnitude-0.9**16)<1e-12);
 assert.deepEqual(Object.keys(p).sort(),["states","final_state","magnitude"].sort());
});

test("Scalar recurrence distinguishes no steps, complete decay and ideal retention",()=>{
 assert.deepEqual(toyMemoryRetention(1,0,0).states,[1]);
 assert.deepEqual(toyMemoryRetention(1,0,2).states,[1,0,0]);
 assert.deepEqual(toyMemoryRetention(-1,1,4).states,[-1,-1,-1,-1,-1]);
 assert.equal(toyMemoryRetention(1,0.5,2).final_state,0.25);
});

test("Scalar recurrence fails closed on undeclared, unbounded or nonnumeric inputs",()=>{
 for(const args of [[0,0.9,4],[2,0.9,4],["1",0.9,4],[1,-1,4],
  [1,0.8,4],[1,NaN,4],[1,1,-1],[1,1,17],[1,1,2.5],[1,1,"4"]]){
  assert.throws(()=>toyMemoryRetention(...args),RangeError,String(args));
 }
});
