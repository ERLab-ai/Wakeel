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

/**
 * Project only the token-count comparison for the Chapter 2 reference sentences.
 * Arabic and English use related but non-identical sentences; the numbers do
 * not support language efficiency, pricing or model-level performance claims.
 */
export function projectFrozenChapter2Evidence(bytes,manifest){
 const sourcePath="content/chapter-02/results.json";
 requireEvidence(Buffer.isBuffer(bytes),"Chapter 2 source must be raw bytes");
 requireEvidence(manifest?.source_version==="v0.21.3","Chapter 2 wrong source version");
 const expected=manifest?.files?.[sourcePath];
 requireEvidence(typeof expected==="string"&&/^[a-f0-9]{64}$/.test(expected),
   "Chapter 2 missing source SHA-256");
 const digest=createHash("sha256").update(bytes).digest("hex");
 requireEvidence(digest===expected,"Chapter 2 SHA-256 mismatch");
 let raw;
 try{raw=JSON.parse(bytes.toString("utf8"));}
 catch{throw new Error("Frozen Chapter 2 evidence rejected: invalid JSON");}
 requireEvidence(raw?.generated_from==="content/chapter-02/examples.json",
   "Chapter 2 unexpected evidence origin");
 requireEvidence(raw?.pinned_tokenizer==="edu-tok-0.1","Chapter 2 tokenizer version mismatch");
 const samples={};
 for(const lang of LANGUAGES){
  const entry=raw.sections?.tokenization?.[lang];
  requireEvidence(entry && typeof entry.text==="string" && entry.text.trim().length>0,
    "Chapter 2 missing reference sentence "+lang);
  for(const [countKey,tokensKey] of [["ws_keep_count","ws_keep_tokens"],["edu_count","edu_tokens"]]){
   requireEvidence(Number.isSafeInteger(entry[countKey])&&entry[countKey]>0,
     "Chapter 2 invalid "+lang+"/"+countKey);
   requireEvidence(Array.isArray(entry[tokensKey])&&
     entry[tokensKey].every(x=>typeof x==="string"&&x.length>0)&&
     entry[tokensKey].length===entry[countKey],
     "Chapter 2 count/token array mismatch "+lang+"/"+countKey);
  }
  samples[lang]={
   whitespace_units:entry.ws_keep_count,
   educational_tokenizer_units:entry.edu_count
  };
 }
 return {
  schema:"wakeel-v022-frozen-tokenization/v1",
  chapter_id:"CH-2",
  status:"approved-source-counts-not-independent-review",
  source_path:sourcePath,
  source_sha256:digest,
  tokenizer:"edu-tok-0.1",
  samples
 };
}

/**
 * Frozen Chapter 3: allowlist only validation/test losses, selected smoothing,
 * and a single pinned mini-example probability. Never publish the corpus,
 * generated text, training curves, full predictions or raw results JSON.
 */
export function projectFrozenChapter3Evidence(bytes,manifest){
 const sourcePath="content/chapter-03/results.json";
 requireEvidence(Buffer.isBuffer(bytes),"Chapter 3 source must be raw bytes");
 requireEvidence(manifest?.source_version==="v0.21.3","Chapter 3 wrong source version");
 const expected=manifest?.files?.[sourcePath];
 requireEvidence(typeof expected==="string"&&/^[a-f0-9]{64}$/.test(expected),
  "Chapter 3 missing source SHA-256");
 const digest=createHash("sha256").update(bytes).digest("hex");
 requireEvidence(digest===expected,"Chapter 3 SHA-256 mismatch");
 let raw;
 try{raw=JSON.parse(bytes.toString("utf8"));}
 catch{throw new Error("Frozen Chapter 3 evidence rejected: invalid JSON");}
 requireEvidence(raw?.generated_from==="content/chapter-03/examples.json",
  "Chapter 3 unexpected evidence origin");
 const mini=raw.worked_example;
 requireEvidence(mini && mini.alpha===1 && mini.output_vocab_size===6,
  "Chapter 3 unexpected worked-example settings");
 const reference=mini.probabilities?.["<s>→س"];
 requireEvidence(typeof reference==="number"&&Number.isFinite(reference)&&
  Math.abs(reference-(2/7))<=0.0005,
  "Chapter 3 worked-example probability mismatch");
 const datasets={};
 const grid=[0.1,0.5,1];
 for(const lang of LANGUAGES){
  const data=raw.languages?.[lang], models=data?.models;
  requireEvidence(models && typeof models==="object","Chapter 3 missing models "+lang);
  const projected={};
  for(const model of ["unigram","count_bigram","neural_bigram"]){
   const row=models[model];
   requireEvidence(row && typeof row==="object","Chapter 3 missing "+lang+"/"+model);
   for(const key of ["val_loss","test_loss"]){
    requireEvidence(typeof row[key]==="number"&&Number.isFinite(row[key])&&
     row[key]>0&&row[key]<30,
     "Chapter 3 invalid "+lang+"/"+model+"/"+key);
   }
   projected[model]={validation_loss:row.val_loss,test_loss:row.test_loss};
  }
  const bigram=models.count_bigram;
  requireEvidence(Array.isArray(bigram.alpha_grid)&&
    bigram.alpha_grid.length===grid.length&&
    bigram.alpha_grid.every((x,i)=>x===grid[i]),
    "Chapter 3 validation alpha grid drift "+lang);
  requireEvidence(grid.includes(bigram.chosen_alpha),
    "Chapter 3 invalid selected alpha "+lang);
  const chosenKey=bigram.chosen_alpha===1?"1.0":String(bigram.chosen_alpha);
  requireEvidence(bigram.val_losses&&
    Object.keys(bigram.val_losses).sort().join(",")==="0.1,0.5,1.0"&&
    Object.values(bigram.val_losses).every(x=>typeof x==="number"&&Number.isFinite(x)&&x>0),
    "Chapter 3 incomplete validation loss grid "+lang);
  requireEvidence(Math.abs(bigram.val_losses[chosenKey]-bigram.val_loss)<1e-9,
    "Chapter 3 selected alpha/validation loss mismatch "+lang);
  projected.count_bigram.chosen_alpha=bigram.chosen_alpha;
  datasets[lang]=projected;
 }
 return {
  schema:"wakeel-v022-frozen-bigram-metrics/v1",
  chapter_id:"CH-3",
  status:"approved-source-metrics-not-independent-review",
  source_path:sourcePath,
  source_sha256:digest,
  mini_example:{alpha:1,total_context_count:1,observed_next_count:1,output_vocabulary:6,
    reported_probability:reference},
  datasets
 };
}

/**
 * A strict, build-only projection of frozen Chapter 4 N-gram reference metrics.
 * Backoff results are deliberately omitted because source Backoff scores
 * are not normalized probabilities and their loss is not comparable here.
 */
export function projectFrozenChapter4Evidence(bytes,manifest){
 const sourcePath="content/chapter-04/results.json";
 requireEvidence(Buffer.isBuffer(bytes),"Chapter 4 source must be raw bytes");
 requireEvidence(manifest?.source_version==="v0.21.3","Chapter 4 wrong source version");
 const expected=manifest?.files?.[sourcePath];
 requireEvidence(typeof expected==="string"&&/^[a-f0-9]{64}$/.test(expected),
   "Chapter 4 missing source SHA-256");
 const digest=createHash("sha256").update(bytes).digest("hex");
 requireEvidence(digest===expected,"Chapter 4 SHA-256 mismatch");
 let raw;
 try{raw=JSON.parse(bytes.toString("utf8"));}
 catch{throw new Error("Frozen Chapter 4 evidence rejected: invalid JSON");}
 const inputPath="content/chapter-03/examples.json";
 requireEvidence(raw?.simulator_version==="simulate_chapter4-1.0" &&
   raw.generated_from===inputPath &&
   raw.input_hash===manifest.files?.[inputPath],
   "Chapter 4 simulator/input provenance mismatch");
 const datasets={};
 for(const lang of LANGUAGES){
  const data=raw.languages?.[lang];
  const contextVocabulary=data?.vocab_size_incl_specials;
  const outputVocabulary=data?.output_space;
  requireEvidence(Number.isSafeInteger(contextVocabulary)&&contextVocabulary>=2&&
    contextVocabulary<=64&&outputVocabulary===contextVocabulary-1,
    "Chapter 4 invalid vocabularies "+lang);
  const orders=[];
  let evaluationTokens=null;
  for(let k=0;k<=3;k++){
   const source=data.models?.["k"+k];
   const context=lang+"/k"+k;
   requireEvidence(source&&typeof source==="object",
     "Chapter 4 missing model "+context);
   for(const field of ["train_nll","val_nll","test_nll"]){
    requireEvidence(typeof source[field]==="number"&&
      Number.isFinite(source[field])&&source[field]>0&&source[field]<30,
      "Chapter 4 invalid loss "+context+"/"+field);
   }
   const grid=[0.1,0.5,1];
   requireEvidence(grid.includes(source.alpha_from_validation) &&
     source.val_losses_grid &&
     Object.keys(source.val_losses_grid).sort().join(",")==="0.1,0.5,1.0",
     "Chapter 4 invalid alpha grid "+context);
   const alphaKey=source.alpha_from_validation===1?"1.0":String(source.alpha_from_validation);
   const scores=Object.values(source.val_losses_grid);
   requireEvidence(scores.every(v=>typeof v==="number"&&Number.isFinite(v)&&v>0) &&
     Math.abs(source.val_losses_grid[alphaKey]-source.val_nll)<1e-8 &&
     source.val_nll<=Math.min(...scores)+1e-8,
     "Chapter 4 invalid validation selection "+context);
   const expectedDense=contextVocabulary**k*outputVocabulary;
   requireEvidence(Number.isSafeInteger(source.theoretical_dense_cells)&&
     source.theoretical_dense_cells===expectedDense,
     "Chapter 4 dense-table mismatch "+context);
   requireEvidence(Number.isSafeInteger(source.sparse_storage_records)&&
     source.sparse_storage_records>0&&source.sparse_storage_records<=expectedDense,
     "Chapter 4 invalid sparse records "+context);
   for(const name of ["unseen_context_rate_val","unseen_context_rate_test"]){
    const n=source[name];
    requireEvidence(typeof n==="number"&&Number.isFinite(n)&&n>=0&&n<=1,
      "Chapter 4 invalid unseen rate "+context+"/"+name);
   }
   requireEvidence(Number.isSafeInteger(source.eval_tokens)&&source.eval_tokens>0,
     "Chapter 4 invalid evaluation token count "+context);
   if(evaluationTokens===null)evaluationTokens=source.eval_tokens;
   requireEvidence(source.eval_tokens===evaluationTokens,
     "Chapter 4 unequal test token counts "+lang);
   orders.push({
    k,
    selected_alpha:source.alpha_from_validation,
    training_loss:source.train_nll,
    validation_loss:source.val_nll,
    test_loss:source.test_nll,
    unseen_validation_rate:source.unseen_context_rate_val,
    unseen_test_rate:source.unseen_context_rate_test,
    dense_cells:source.theoretical_dense_cells,
    sparse_records:source.sparse_storage_records
   });
  }
  datasets[lang]={context_vocabulary:contextVocabulary,
    output_vocabulary:outputVocabulary,orders};
 }
 return {
  schema:"wakeel-v022-frozen-context-orders/v1",
  chapter_id:"CH-4",
  status:"approved-source-metrics-not-independent-review",
  source_path:sourcePath,
  source_sha256:digest,
  datasets
 };
}
