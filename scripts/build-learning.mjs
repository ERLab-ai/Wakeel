import { readFileSync, writeFileSync, mkdirSync, cpSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { spawnSync } from "node:child_process";

const source = "baseline/v0.21.3";
const JOURNEY = source + "/data/WAKEEL-2-JOURNEYS-v0.3.json";
const REGISTRY = source + "/data/WAKEEL-2-KNOWLEDGE-REGISTRY-v0.3.json";
const MANIFEST = source + "/PUBLIC-MANIFEST.json";
const FIELDS = ["sees", "tries", "hypothesis", "possible_outcomes", "interpretation", "discovers"];

function demand(ok, message) {
  if (!ok) throw new Error("Learning catalog rejected: " + message);
}
function text(value) {
  return typeof value === "string" && value.trim().length > 0;
}
export function compileLearning(journeyData, registry, manifest) {
  demand(Array.isArray(journeyData.journeys) && journeyData.journeys.length === 1, "exactly one canonical journey");
  const j = journeyData.journeys[0];
  demand(j.id === "JRN-LLM-001", "unexpected journey identity");
  demand(j.chapters?.length === 9 && j.stations?.length === 44, "approved journey station/chapter counts");
  demand(registry.topics?.length === 389 && registry.relations?.length === 125, "approved registry counts");
  demand(manifest.source_version === "v0.21.3" && manifest.count_public_source_files === 75, "wrong frozen source");
  demand(manifest.files && manifest.files["data/WAKEEL-2-JOURNEYS-v0.3.json"] && manifest.files["data/WAKEEL-2-KNOWLEDGE-REGISTRY-v0.3.json"], "source hashes absent");
  const topicMap = new Map(registry.topics.map(t => [t.id, t]));
  demand(topicMap.size === 389, "duplicate topic ids");
  const domains = registry.domains.map(d => ({ id: d.id, group_id: d.group_id, title_ar: d.title_ar, title_en: d.title_en }));
  const domainIds = new Set(domains.map(d => d.id));
  const topics = registry.topics.map(t => {
    demand(text(t.id) && text(t.title_ar) && text(t.title_en) && domainIds.has(t.domain_id), "incomplete bilingual topic " + t.id);
    return { id:t.id, domain_id:t.domain_id, title_ar:t.title_ar, title_en:t.title_en,
      coverage:t.coverage || "", status:t.status || "", independent_entry:t.independent_entry === true };
  });
  const stationByOrder = new Map();
  const stations = j.stations.map(s => {
    demand(Number.isInteger(s.order) && s.order >= 1 && s.order <= 44 && !stationByOrder.has(s.order), "duplicate/out-of-range station order");
    demand(topicMap.has(s.topic_id), "station topic not in registry");
    demand(text(s.why_here_ar) && text(s.why_here_en), "station bilingual field " + s.order + "/why_here");
    // G7 requires bilingual presence parity, not mandatory deliverables for legacy optional stations.
    demand(Boolean(text(s.deliverable_ar)) === Boolean(text(s.deliverable_en)),
      "station bilingual presence parity " + s.order + "/deliverable");
    const activity = {};
    for (const key of FIELDS) {
      const ar = s.activity?.[key + "_ar"], en = s.activity?.[key + "_en"];
      if (key === "possible_outcomes") {
        demand(Array.isArray(ar) && Array.isArray(en) && ar.length > 0 && ar.length === en.length && ar.every(text) && en.every(text), "outcome parity station " + s.order);
      } else {
        demand(text(ar) && text(en), "activity parity station " + s.order + "/" + key);
      }
      activity[key + "_ar"] = ar;
      activity[key + "_en"] = en;
    }
    stationByOrder.set(s.order, s.topic_id);
    return { order:s.order, topic_id:s.topic_id, role:s.role, why_here_ar:s.why_here_ar, why_here_en:s.why_here_en,
      deliverable_ar:s.deliverable_ar, deliverable_en:s.deliverable_en, activity };
  }).sort((a,b)=>a.order-b.order);
  demand(stations.every((s,i)=>s.order===i+1), "station order must be continuous");
  const encountered = [];
  const chapters = j.chapters.map(c => {
    for (const k of ["title", "goal"]) demand(text(c[k + "_ar"]) && text(c[k + "_en"]), "chapter language " + c.id + "/" + k);
    demand(Array.isArray(c.station_orders) && c.station_orders.length > 0, "chapter must have stations");
    for (const order of c.station_orders) {
      demand(stationByOrder.has(order), "chapter references unknown station");
      encountered.push(order);
    }
    return { id:c.id, title_ar:c.title_ar, title_en:c.title_en, goal_ar:c.goal_ar, goal_en:c.goal_en,
      station_orders:c.station_orders.slice(), content_status:c.content?.status || "unknown" };
  });
  demand(encountered.length===44 && new Set(encountered).size===44, "chapters must cover each station once");
  const relations = registry.relations.map(r => {
    demand(topicMap.has(r.source) && topicMap.has(r.target), "relation references missing topic");
    return { source:r.source, target:r.target, type:r.type };
  });
  return {
    schema:"wakeel-v022-learning-catalog/v1",
    source:{version:"v0.21.3",journey_id:j.id,
      journey_sha256:manifest.files["data/WAKEEL-2-JOURNEYS-v0.3.json"],
      registry_sha256:manifest.files["data/WAKEEL-2-KNOWLEDGE-REGISTRY-v0.3.json"],
      note:"Navigation catalog generated from approved source; not scientific experiment output."},
    journey:{title_ar:j.title_ar,title_en:j.title_en, goal_ar:j.goal_ar, goal_en:j.goal_en},
    counts:{chapters:9,stations:44,topics:389,relations:125},
    chapters,stations,topics,domains,relations
  };
}
export function renderApprovedChapters(root=process.cwd()){
  const isWindows=process.platform==="win32";
  const executable=process.env.PYTHON || (isWindows ? "py" : "python3");
  const args=[...(isWindows && executable==="py"?["-3"]:[]),"tools/render_chapters.py"];
  // Windows: the Python child inherits a legacy console code page (cp1252) unless
  // UTF-8 mode is forced, which corrupts Arabic chapter output through the pipe.
  // PYTHONUTF8/PYTHONIOENCODING make the renderer locale-independent by design.
  const processResult=spawnSync(executable,args,{
    cwd:root,encoding:"utf8",maxBuffer:6*1024*1024,windowsHide:true,
    env:{...process.env,PYTHONUTF8:"1",PYTHONIOENCODING:"utf-8"}
  });
  demand(!processResult.error && processResult.status===0,
    "approved chapter renderer failed: "+(processResult.error?.message || processResult.stderr || "unknown failure"));
  let result;
  try{ result=JSON.parse(processResult.stdout); }
  catch{ throw new Error("Approved chapter renderer produced invalid JSON"); }
  demand(result.schema==="wakeel-v022-bilingual-chapter-render/v1" &&
    result.count===18 && Object.keys(result.chapters||{}).length===9,
    "unexpected approved chapter render count");
  return result;
}

export function build(root=process.cwd()){
  const read=(relative)=>JSON.parse(readFileSync(path.join(root,relative),"utf8"));
  const sourceManifest=read(MANIFEST);
  const out=compileLearning(read(JOURNEY), read(REGISTRY), sourceManifest);
  const rendered=renderApprovedChapters(root);
  for(const chapter of out.chapters){
    const article=rendered.chapters[chapter.id];
    demand(article && article.ar && article.en, "missing bilingual authored chapter " + chapter.id);
    for(const lang of ["ar","en"]){
      const file=article[lang].source_path;
      demand(sourceManifest.files[file]===article[lang].source_sha256,
        "chapter source digest differs from frozen public source: " + file);
    }
  }
  out.chapter_articles=rendered.chapters;
  const dir=path.join(root,"dist");
  mkdirSync(path.join(dir,"src"),{recursive:true});
  mkdirSync(path.join(dir,"data"),{recursive:true});
  for(const file of ["index.html","styles.css","src/app.mjs","src/models.mjs","src/learning-view.mjs"]){
    cpSync(path.join(root,file),path.join(dir,file));
  }
  writeFileSync(path.join(dir,"data/learning.json"),JSON.stringify(out)+"\n","utf8");
  const files=[
    "index.html","styles.css","src/app.mjs","src/models.mjs","src/learning-view.mjs","data/learning.json"
  ];
  demand(files.length===6,"release asset allowlist changed");
  console.log(JSON.stringify({status:"PASS",assets:files,counts:out.counts,bytes:Buffer.byteLength(JSON.stringify(out))}));
  return out;
}
if(process.argv[1] && path.resolve(process.argv[1])===fileURLToPath(import.meta.url)){
  build();
}