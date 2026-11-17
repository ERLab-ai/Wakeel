/**
 * A narrow, read-only projection of the approved Chapter 1 reference results.
 *
 * This script runs only during the build; never ship the source results JSON.
 * It is not a new simulation, a quality estimate or independent validation.
 */
import {createHash} from "node:crypto";

const SOURCE_PATH="content/chapter-01/results.json";
const MODES=["seed","tests"];
const LANGUAGES=["ar","en"];

function requireEvidence(ok,reason){
 if(!ok)throw new Error("Frozen Chapter 1 evidence rejected: "+reason);
}
function countCorrect(rows,total,expected,context){
 requireEvidence(Array.isArray(rows)&&rows.length===total,context+" row count");
 requireEvidence(rows.every(row=>row && typeof row.correct==="boolean"),context+" row correctness types");
 requireEvidence(rows.filter(row=>row.correct).length===expected,context+" aggregate/row mismatch");
}

/**
 * @param {Buffer} bytes Original, unmodified source bytes.
 * @param {object} manifest Frozen v0.21.3 public-source manifest.
 * @returns {object} Only explicitly permitted aggregate counts and provenance.
 */
export function projectFrozenChapter1Evidence(bytes,manifest){
 requireEvidence(Buffer.isBuffer(bytes),"source must be raw bytes");
 requireEvidence(manifest?.source_version==="v0.21.3","wrong source version");
 const expected=manifest?.files?.[SOURCE_PATH];
 requireEvidence(typeof expected==="string"&&/^[a-f0-9]{64}$/.test(expected),"missing source SHA-256");
 const digest=createHash("sha256").update(bytes).digest("hex");
 requireEvidence(digest===expected,"SHA-256 mismatch");
 let raw;
 try{raw=JSON.parse(bytes.toString("utf8"));}
 catch{throw new Error("Frozen Chapter 1 evidence rejected: invalid JSON");}
 requireEvidence(raw?.generated_from==="content/chapter-01/examples.json","unexpected evidence origin");
 const datasets={};
 for(const lang of LANGUAGES){
  const original=raw.datasets?.[lang];
  requireEvidence(original && typeof original==="object","missing language "+lang);
  const projected={};
  for(const split of MODES){
   const row=original[split],context=lang+"/"+split;
   requireEvidence(row && Number.isSafeInteger(row.total) && row.total>0,context+" invalid total");
   for(const metric of ["rule_correct","example_correct"]){
    requireEvidence(Number.isSafeInteger(row[metric])&&row[metric]>=0&&row[metric]<=row.total,
      context+" invalid "+metric);
   }
   countCorrect(row.rule_rows,row.total,row.rule_correct,context+"/rules");
   countCorrect(row.example_rows,row.total,row.example_correct,context+"/examples");
   projected[split]={
    total:row.total,
    rules_correct:row.rule_correct,
    nearest_example_correct:row.example_correct
   };
  }
  datasets[lang]=projected;
 }
 return {
  schema:"wakeel-v022-frozen-aggregate/v1",
  chapter_id:"CH-1",
  status:"approved-source-aggregate-not-independent-review",
  source_path:SOURCE_PATH,
  source_sha256:digest,
  datasets
 };
}
