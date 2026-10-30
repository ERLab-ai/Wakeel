import test from "node:test";
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {compileLearning} from "../scripts/build-learning.mjs";
const read=(file)=>JSON.parse(readFileSync("baseline/v0.21.3/"+file,"utf8"));
const readInputs=()=>[
  read("data/WAKEEL-2-JOURNEYS-v0.3.json"),
  read("data/WAKEEL-2-KNOWLEDGE-REGISTRY-v0.3.json"),
  read("PUBLIC-MANIFEST.json")
];
test("derive 44 ordered stations, nine chapters and 389 independent topic entries",()=>{
  const result=compileLearning(...readInputs());
  assert.deepEqual(result.counts,{chapters:9,stations:44,topics:389,relations:125});
  assert.deepEqual(result.stations.map(s=>s.order),Array.from({length:44},(_,i)=>i+1));
  assert.equal(result.chapters.flatMap(x=>x.station_orders).length,44);
  assert.ok(result.topics.some(t=>!result.stations.some(s=>s.topic_id===t.id)));
});
test("Arabic/English authored activity mirror has no missing outcomes",()=>{
  const result=compileLearning(...readInputs());
  for(const s of result.stations){
    assert.equal(s.activity.possible_outcomes_ar.length,s.activity.possible_outcomes_en.length);
    assert.ok(s.why_here_ar && s.why_here_en);
    assert.equal(Boolean(s.deliverable_ar),Boolean(s.deliverable_en));
  }
});
test("refuse missing English and mismatched alternatives",()=>{
  const inputs=readInputs();
  inputs[0].journeys[0].stations[0].activity.tries_en="";
  assert.throws(()=>compileLearning(...inputs),/activity parity/);
  const second=readInputs();
  second[0].journeys[0].stations[0].activity.possible_outcomes_en.pop();
  assert.throws(()=>compileLearning(...second),/outcome parity/);
});
test("refuse chapter holes, missing scientific topic identities and edited source manifests",()=>{
  let a=readInputs();a[0].journeys[0].chapters[0].station_orders.pop();
  assert.throws(()=>compileLearning(...a),/cover each station/);
  a=readInputs();a[1].topics[0].title_en="";
  assert.throws(()=>compileLearning(...a),/incomplete bilingual topic/);
  a=readInputs();a[2].source_version="v0.21.2";
  assert.throws(()=>compileLearning(...a),/wrong frozen source/);
});
test("public catalog contains only curated learner-facing fields, not experiment results/reviews",()=>{
  const out=compileLearning(...readInputs());
  const raw=JSON.stringify(out);
  assert.ok(!raw.includes("runs.log"));
  assert.ok(!raw.includes("review-v0.21"));
  assert.ok(!("results" in out));
  assert.ok(!("sources" in out.topics[0]));
});
