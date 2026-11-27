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

/**
 * Chapter 5 read-only metrics: a strict, SHA-pinned projection of the MLP
 * representation experiment and the frozen-random-embedding ablation.
 * No embedding vectors, weights, corpora or generated samples are exported.
 */
export function projectFrozenChapter5Evidence(bytes,manifest){
 const sourcePath="content/chapter-05/results.json";
 const sourceInput="content/chapter-03/examples.json";
 requireEvidence(Buffer.isBuffer(bytes),"Chapter 5 source must be raw bytes");
 requireEvidence(manifest?.source_version==="v0.21.3","Chapter 5 wrong source version");
 const expected=manifest?.files?.[sourcePath];
 requireEvidence(typeof expected==="string"&&/^[a-f0-9]{64}$/.test(expected),
  "Chapter 5 missing source SHA-256");
 const digest=createHash("sha256").update(bytes).digest("hex");
 requireEvidence(digest===expected,"Chapter 5 SHA-256 mismatch");
 let raw;
 try{raw=JSON.parse(bytes.toString("utf8"));}
 catch{throw new Error("Frozen Chapter 5 evidence rejected: invalid JSON");}
 requireEvidence(raw?.simulator_version==="simulate_chapter5-1.0" &&
  raw.input_hash===manifest.files?.[sourceInput],
  "Chapter 5 simulator/input provenance mismatch");
 const datasets={};
 const checkpoints=[1,10,50,100,200,300];
 const lossOk=(v)=>typeof v==="number"&&Number.isFinite(v)&&v>0&&v<30;
 for(const lang of LANGUAGES){
  const data=raw.languages?.[lang];
  requireEvidence(data&&typeof data==="object","Chapter 5 missing language "+lang);
  const {architecture:arch,vocab,params,losses,training_curve:curve,
   frozen_embedding_ablation:frozen,val_seen_unseen_split:split,units}=data;
  requireEvidence(arch?.context_k===2 && arch.embedding_dim===8 &&
   arch.hidden===32 && arch.activation==="tanh" &&
   arch.output_excludes_start===true &&
   Number.isSafeInteger(vocab?.context)&&vocab.context>=2&&vocab.context<=64&&
   vocab.output===vocab.context-1,
   "Chapter 5 unexpected architecture "+lang);
  const embedded=vocab.context*arch.embedding_dim;
  const hidden=(arch.context_k*arch.embedding_dim+1)*arch.hidden;
  const output=(arch.hidden+1)*vocab.output;
  const total=embedded+hidden+output;
  requireEvidence(params?.embedding===embedded&&
   params.hidden_layer===hidden&&params.output_layer===output&&
   params.total_declared===total&&frozen?.trainable===false&&
   frozen.trainable_params===total-embedded,
   "Chapter 5 parameter accounting mismatch "+lang);
  requireEvidence(lossOk(losses?.train_final)&&lossOk(losses?.val_nll)&&lossOk(losses?.test_nll)&&
   lossOk(frozen.train_nll)&&lossOk(frozen.val_nll)&&lossOk(frozen.test_nll),
   "Chapter 5 invalid learned/frozen loss "+lang);
  requireEvidence(Array.isArray(curve)&&curve.length===checkpoints.length &&
   curve.every((p,i)=>p?.epoch===checkpoints[i]&&lossOk(p.train_nll))&&
   Math.abs(curve.at(-1).train_nll-losses.train_final)<1e-8,
   "Chapter 5 training curve/final loss mismatch "+lang);
  requireEvidence(Number.isSafeInteger(units?.val)&&units.val>0 &&
   Number.isSafeInteger(split?.seen_count)&&split.seen_count>0 &&
   Number.isSafeInteger(split?.unseen_count)&&split.unseen_count>0 &&
   split.seen_count+split.unseen_count===units.val &&
   lossOk(split.seen_nll)&&lossOk(split.unseen_nll),
   "Chapter 5 seen/unseen validation split mismatch "+lang);
  const weighted=(split.seen_nll*split.seen_count+
    split.unseen_nll*split.unseen_count)/units.val;
  requireEvidence(Math.abs(weighted-losses.val_nll)<=0.0002,
   "Chapter 5 weighted validation loss mismatch "+lang);
  datasets[lang]={
   architecture:{context_vocabulary:vocab.context,output_vocabulary:vocab.output,
    context_order:arch.context_k,embedding_dim:arch.embedding_dim,hidden:arch.hidden},
   parameters:{total,embedding:embedded,hidden,output,
    trainable_frozen_embeddings:total-embedded},
   learned:{train_loss:losses.train_final,validation_loss:losses.val_nll,
    test_loss:losses.test_nll},
   frozen:{train_loss:frozen.train_nll,validation_loss:frozen.val_nll,
    test_loss:frozen.test_nll},
   validation_contexts:{
    seen:{count:split.seen_count,loss:split.seen_nll},
    unseen:{count:split.unseen_count,loss:split.unseen_nll}
   },
   training_curve:curve.map(x=>({epoch:x.epoch,loss:x.train_nll}))
  };
 }
 return {
  schema:"wakeel-v022-frozen-embedding-ablation/v1",
  chapter_id:"CH-5",
  status:"approved-source-metrics-not-independent-review",
  source_path:sourcePath,
  source_sha256:digest,
  datasets
 };
}

/**
 * Chapter 6: distinguish separately trained fixed-delay models from mixed-delay
 * training and genuinely held-out delay 16. Project aggregate measurements only.
 */
export function projectFrozenChapter6Evidence(bytes,manifest){
 const sourcePath="content/chapter-06/results.json";
 const inputPath="content/chapter-03/examples.json";
 requireEvidence(Buffer.isBuffer(bytes),"Chapter 6 source must be raw bytes");
 requireEvidence(manifest?.source_version==="v0.21.3","Chapter 6 wrong source version");
 const expected=manifest?.files?.[sourcePath];
 requireEvidence(typeof expected==="string"&&/^[a-f0-9]{64}$/.test(expected),"Chapter 6 missing source SHA-256");
 const digest=createHash("sha256").update(bytes).digest("hex");
 requireEvidence(digest===expected,"Chapter 6 SHA-256 mismatch");
 let raw;
 try{raw=JSON.parse(bytes.toString("utf8"));}
 catch{throw new Error("Frozen Chapter 6 evidence rejected: invalid JSON");}
 requireEvidence(raw?.simulator_version==="simulate_chapter6-3.2" &&
   /^[a-f0-9]{64}$/.test(manifest.files?.[inputPath]||"") &&
   raw.input_hash===manifest.files[inputPath] &&
   raw.provenance?.input_hash===raw.input_hash &&
   raw.provenance?.output_excludes_start===true &&
   raw.provenance?.vocab_source==="train_only","Chapter 6 simulator/input provenance mismatch");
 const delays=[2,4,5,6,7,8,16],seeds=[42,43,44];
 const fixed=raw.memory_probe,mixed=raw.mixed_length_model;
 const same=(a,b)=>Array.isArray(a)&&a.length===b.length&&a.every((v,i)=>v===b[i]);
 const mean=a=>a.reduce((x,y)=>x+y,0)/a.length;
 const acc=x=>typeof x==="number"&&Number.isFinite(x)&&x>=0&&x<=1;
 const nll=x=>typeof x==="number"&&Number.isFinite(x)&&x>=0&&x<30;
 const near=(a,b,tolerance)=>Number.isFinite(a)&&Math.abs(a-b)<=tolerance;
 requireEvidence(same(fixed?.delays,delays)&&same(fixed?.seeds,seeds)&&
   fixed.train_per_class===120&&fixed.test_per_class===40&&fixed.epochs===300&&fixed.lr===0.3&&
   fixed.oracle_label_swap_control===0,"Chapter 6 fixed protocol mismatch");
 requireEvidence(same(mixed?.train_delays,delays.slice(0,-1))&&same(mixed?.holdout_delays,[16])&&
   same(mixed?.seeds,seeds)&&mixed.train_per_class_per_delay===60&&
   mixed.test_per_class===40&&mixed.epochs===300&&mixed.lr===0.3,
   "Chapter 6 mixed protocol mismatch");
 const rows=delays.map(delay=>{
  const key="delay_"+delay,f=fixed.delays_data?.[key],m=mixed.means?.[key];
  requireEvidence(f?.n_test===80&&f.oracle_binary_through_eval_path===1&&
    Array.isArray(f.per_seed_binary)&&f.per_seed_binary.length===3&&f.per_seed_binary.every(acc)&&
    acc(f.accuracy_binary)&&near(f.accuracy_binary,mean(f.per_seed_binary),0.00051)&&
    acc(f.untrained_baseline_binary)&&nll(f.test_answer_nll_mean),
    "Chapter 6 fixed metrics mismatch "+key);
  requireEvidence(m?.in_domain===(delay!==16)&&
    Array.isArray(m.per_seed_binary)&&m.per_seed_binary.length===3&&m.per_seed_binary.every(acc)&&
    acc(m.binary_mean)&&near(m.binary_mean,mean(m.per_seed_binary),0.00051)&&nll(m.test_nll_mean),
    "Chapter 6 mixed metrics mismatch "+key);
  const seedRows=seeds.map((seed,i)=>{
   const value=mixed.per_seed?.[String(seed)]?.[key];
   requireEvidence(value?.in_domain===m.in_domain&&value.binary===m.per_seed_binary[i]&&
     nll(value.test_nll),"Chapter 6 mixed per-seed mismatch "+seed+"/"+key);
   return {seed,fixed_accuracy:f.per_seed_binary[i],mixed_accuracy:value.binary,mixed_nll:value.test_nll};
  });
  requireEvidence(near(m.test_nll_mean,mean(seedRows.map(x=>x.mixed_nll)),0.00011),
    "Chapter 6 mixed NLL mismatch "+key);
  requireEvidence(Array.isArray(f.per_seed_test_nll)&&f.per_seed_test_nll.length===3&&
    f.per_seed_test_nll.every(nll)&&near(f.test_answer_nll_mean,mean(f.per_seed_test_nll),0.00011),
    "Chapter 6 fixed NLL mismatch "+key);
  return {delay,mixed_in_domain:m.in_domain,test_per_class:40,
    fixed_accuracy:f.accuracy_binary,mixed_accuracy:m.binary_mean,
    fixed_nll:f.test_answer_nll_mean,mixed_nll:m.test_nll_mean,
    untrained_accuracy:f.untrained_baseline_binary,oracle_accuracy:1,
    seeds:seedRows.map((x,i)=>({...x,fixed_nll:f.per_seed_test_nll[i]}))};
 });
 const languages={};
 for(const lang of LANGUAGES){
  const data=raw.languages?.[lang]?.rnn;
  requireEvidence(data&&Number.isSafeInteger(data.params)&&data.params>0,
    "Chapter 6 missing corpus model "+lang);
  const runs=seeds.map(seed=>{
   const s=data.seeds?.[String(seed)];
   requireEvidence(nll(s?.val_nll)&&nll(s?.test_nll),"Chapter 6 missing corpus seed "+lang+"/"+seed);
   return {seed,validation_loss:s.val_nll,test_loss:s.test_nll};
  });
  requireEvidence(nll(data.mean_val)&&nll(data.mean_test)&&
    near(data.mean_val,mean(runs.map(x=>x.validation_loss)),0.00011)&&
    near(data.mean_test,mean(runs.map(x=>x.test_loss)),0.00011),
    "Chapter 6 corpus means mismatch "+lang);
  languages[lang]={parameters:data.params,validation_loss:data.mean_val,test_loss:data.mean_test};
 }
 return {schema:"wakeel-v022-frozen-recurrent-memory/v1",chapter_id:"CH-6",
   status:"approved-source-metrics-not-independent-review",source_path:sourcePath,source_sha256:digest,
   seeds,train_delays:delays.slice(0,-1),holdout_delays:[16],languages,rows};
}

/** Chapter 7: holdout retrieval and order evidence; validation sensitivity stays separate. */
export function projectFrozenChapter7Evidence(bytes,manifest){
 const sourcePath="content/chapter-07/results.json",inputPath="content/chapter-03/examples.json";
 requireEvidence(Buffer.isBuffer(bytes)&&manifest?.source_version==="v0.21.3","Chapter 7 source/version mismatch");
 const expected=manifest.files?.[sourcePath],digest=createHash("sha256").update(bytes).digest("hex");
 requireEvidence(/^[a-f0-9]{64}$/.test(expected||"")&&digest===expected,"Chapter 7 SHA-256 mismatch");
 let raw;try{raw=JSON.parse(bytes.toString("utf8"));}catch{throw new Error("Chapter 7 invalid JSON");}
 const seeds=[42,43,44],delays=[2,4,5,6,7,8,16],orderDelays=[4,8,16];
 const same=(a,b)=>Array.isArray(a)&&a.length===b.length&&a.every((x,i)=>x===b[i]);
 const mean=a=>a.reduce((x,y)=>x+y,0)/a.length;
 const acc=x=>typeof x==="number"&&Number.isFinite(x)&&x>=0&&x<=1;
 const loss=x=>typeof x==="number"&&Number.isFinite(x)&&x>=0&&x<30;
 const close=(a,b,tol)=>Number.isFinite(a)&&Math.abs(a-b)<=tol;
 const p=raw.protocol,sp=p?.splits;
 requireEvidence(raw.simulator_version==="simulate_chapter7-2.1"&&
  /^[a-f0-9]{64}$/.test(manifest.files?.[inputPath]||"")&&raw.input_hash===manifest.files[inputPath]&&
  p?.epochs===300&&p.lr===0.3&&p.emb_dim===8&&p.dk===8&&same(p.seeds,seeds)&&
  sp?.train_pairs===100&&sp.val_pairs===30&&sp.holdout_pairs===40&&
  sp.retrieval?.train_per_class===100&&sp.retrieval.val_per_class===30&&sp.retrieval.holdout_per_class===40,
  "Chapter 7 protocol mismatch");
 const models={rnn:1183,attn_content:671,attn_pos:991,attn_nopos:671,mean:535,pos0:535,pos_mid_shift:535};
 const projectRow=(data,key,model,shifted)=>{
  const runs=seeds.map(seed=>{
   const record=data?.["seed_"+seed],m=record?.[model],s=record?.sizes;
   requireEvidence(/^[a-f0-9]{16}$/.test(record?.dataset_fingerprint||"")&&
    (shifted?s?.train_pairs===100&&s.val_pairs===30&&s.holdout_pairs===40:
     s?.train_per_class===100&&s.val_per_class===30&&s.holdout_per_class===40)&&
    m?.params===models[model]&&acc(m.holdout_binary)&&loss(m.holdout_test_nll)&&
    (!shifted||acc(m.shifted_pair_binary)),"Chapter 7 seed/model mismatch "+key+"/"+model+"/"+seed);
   if(shifted){
    const c=record.controls,i=record.paired_invariance_maxdiff;
    requireEvidence(c?.oracle_standard===1&&c.oracle_shifted===1&&c.oracle_label_swap===0&&
     c.positive_control_pos0_standard===1&&c.positive_control_pos_mid_shifted===1&&
     i?.mean===0&&i.attn_nopos===0,"Chapter 7 order controls mismatch "+key);
    if(["mean","attn_nopos"].includes(model))requireEvidence(m.holdout_binary===0.5&&m.shifted_pair_binary===0.5,
     "Chapter 7 permutation invariance mismatch "+key);
    if(model==="pos0"||model==="pos_mid_shift")requireEvidence(
     m.holdout_binary===(model==="pos0"?1:0.5)&&m.shifted_pair_binary===(model==="pos0"?0.5:1),
     "Chapter 7 positive control mismatch "+key);
   }else requireEvidence(record.oracle_ceiling===1,"Chapter 7 retrieval oracle mismatch "+key);
   return {seed,accuracy:m.holdout_binary,nll:m.holdout_test_nll,
    ...(shifted?{shifted_accuracy:m.shifted_pair_binary}:{})};
  });
  const avg=data.mean_across_seeds?.[model];
  requireEvidence(acc(avg?.holdout_binary)&&loss(avg.holdout_test_nll)&&
   close(avg.holdout_binary,mean(runs.map(x=>x.accuracy)),0.001)&&
   close(avg.holdout_test_nll,mean(runs.map(x=>x.nll)),0.00011)&&
   (!shifted||(acc(avg.shifted_pair_binary)&&close(avg.shifted_pair_binary,mean(runs.map(x=>x.shifted_accuracy)),0.001))),
   "Chapter 7 mean mismatch "+key+"/"+model);
  return {parameters:models[model],accuracy:avg.holdout_binary,nll:avg.holdout_test_nll,
    ...(shifted?{shifted_accuracy:avg.shifted_pair_binary}:{}),runs};
 };
 const a=raw.experiments?.retrieval,b=raw.experiments?.order;
 const retrieval=delays.map(delay=>{
  const key="delay_"+delay,data=a?.delays?.[key];
  return {delay,rnn:projectRow(data,key,"rnn",false),
   attn_content:projectRow(data,key,"attn_content",false),oracle_accuracy:1};
 });
 const orderModels=["rnn","attn_pos","attn_nopos","mean","pos0","pos_mid_shift"];
 const order=orderDelays.map(delay=>{
  const key="delay_"+delay,data=b?.delays?.[key];
  return {delay,models:Object.fromEntries(orderModels.map(m=>[m,projectRow(data,key,m,true)]))};
 });
 const sensitivity=b?.init_sensitivity;
 requireEvidence(typeof sensitivity?.eval_set==="string"&&sensitivity.eval_set.startsWith("validation")&&
  same(sensitivity.amplify_values,[1,3]),"Chapter 7 sensitivity split mismatch");
 const variants=["amp1","all3","kq_only3","v_only3","pos_only3"];
 const validation=orderDelays.map(delay=>{
  const data=sensitivity.delays?.["delay_"+delay];
  return {delay,variants:Object.fromEntries(variants.map(variant=>{
   const runs=seeds.map(seed=>data?.["seed_"+seed]?.[variant]?.val_binary);
   const avg=data?.mean_across_seeds?.[variant];
   requireEvidence(runs.every(acc)&&acc(avg)&&close(avg,mean(runs),0.001),
    "Chapter 7 sensitivity mean mismatch "+delay+"/"+variant);
   return [variant,avg];
  }))};
 });
 return {schema:"wakeel-v022-frozen-attention-order/v1",chapter_id:"CH-7",
  status:"approved-source-metrics-not-independent-review",source_path:sourcePath,source_sha256:digest,
  seeds,retrieval,order,validation_sensitivity:validation};
}
