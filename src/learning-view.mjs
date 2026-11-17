/**
 * Bilingual learning journey and open atlas built from authentic approved catalog.
 * Station cards are authored; other registry entries are NOT represented as lessons.
 * This module never changes scientific evidence, model state or archived results.
 */
const labels={
 ar:{
  journeyTag:"المسار التعليمي المعتمد · 9 فصول · 44 محطة",
  journeyTitle:"رحلة بناء نموذج لغوي",
  journeyIntro:"اختر محطة أو اقرأ الفصل العلمي كاملاً. يمكن دخول أي محطة مباشرة دون شرط مسبق.",
  fullChapter:"اقرأ النص العلمي الكامل لهذا الفصل",
  chapterSource:"النص الأصلي المعتمد · بصمة SHA-256",
  atlasTag:"سجل معرفي مفتوح · 389 موضوعاً",
  atlasTitle:"الأطلس المعرفي",
  atlasIntro:"ادخل أي موضوع مباشرة. الموضوع في السجل ليس درساً مؤلفاً بالضرورة؛ ستجد رابط محطته إن كان جزءاً من الرحلة.",
  atlasSearch:"ابحث بالعربية أو الإنجليزية أو معرّف الموضوع",
  placeholder:"ابحث باسم الموضوع أو رمزه…",
  loading:"جار تحميل الفهرس التعليمي…", loadError:"تعذّر تحميل فهرس التعلم. شغّل الموقع عبر خادم محلي باستخدام npm run serve.",
  chapter:"الفصل",station:"المحطة",goal:"هدف الفصل",why:"لماذا هذه المحطة؟",deliverable:"المخرج المطلوب",
  sees:"ما الذي يراه المتعلم؟",tries:"ما الذي يجربه؟",hypothesis:"الفرضية",possible_outcomes:"النتائج المحتملة",
  interpretation:"كيف نفسر المخرجات؟",discovers:"ما الذي يكتشفه؟",roles:{core:"أساسية",optional:"اختيارية"},
  topic:"موضوع",topicNotLesson:"هذا مدخل مستقل في السجل المعرفي، وليس درساً مؤلفاً بعد.",
  topicWithStation:"له محطة في الرحلة",goStation:"انتقل إلى المحطة",
  related:"موضوعات ذات علاقة",domain:"المجال",noRelation:"لا توجد علاقة مسجلة لهذه الصفحة.",
  atlasCount:"نتيجة من أصل",viewMore:"عرض المزيد",noResults:"لم يُعثر على موضوع مطابق.",
  evidenceTitle:"نتائج الفصل الأول المجمدة — عرض القراءات الأصلية",
  evidenceDescription:"مقارنة على رسائل محددة فقط. الأرقام من نتائج المصدر المجمد؛ لم تُعَد التجربة هنا ولم تُراجع نتائجها علمياً مراجعة مستقلة.",
  evidenceSplit:"مجموعة الرسائل",evidenceTotal:"عدد الرسائل",evidenceRules:"إجابات صحيحة بالقواعد",evidenceNearest:"إجابات صحيحة بأقرب مثال",
  evidenceSeed:"الرسائل الست المعروفة عند كتابة القواعد",evidenceTests:"تسع صياغات جديدة للاختبار",
  evidenceSource:"ملف النتائج الأصلي:",
  evidenceLimit:"هذه أعداد نجاح ضمن مجموعة صغيرة ومحددة؛ لا تُثبت جودة عامة أو دلالة إحصائية أو تفوقاً على بيانات أخرى.",
  source:"المصدر",sourceInfo:"v0.21.3 — بيانات مؤلفة ومجمّدة، وليست تجربة تدريب جديدة",
  topicOpen:"استكشف موضوعاً مستقلاً",allTopics:"جميع الموضوعات",notFound:"المحطة أو الموضوع غير موجود في المصدر المعتمد."
 },
 en:{
  journeyTag:"Approved learning journey · 9 chapters · 44 stations",
  journeyTitle:"Build a language model: the journey",
  journeyIntro:"Open any station or read its entire authored scientific chapter. Prior completion is never required.",
  fullChapter:"Read the complete authored chapter",
  chapterSource:"Approved source text · SHA-256",
  atlasTag:"Open knowledge registry · 389 topics",
  atlasTitle:"Knowledge atlas",
  atlasIntro:"Open any topic directly. A registry topic is not necessarily an authored lesson; a journey link is provided where one exists.",
  atlasSearch:"Search by Arabic, English, or topic ID",placeholder:"Search topic title or ID…",
  loading:"Loading approved learning catalog…",loadError:"Unable to load learning catalog. Serve the site locally with npm run serve.",
  chapter:"Chapter",station:"Station",goal:"Chapter objective",why:"Why now?",deliverable:"Expected deliverable",
  sees:"What the learner sees",tries:"What the learner tries",hypothesis:"Hypothesis",possible_outcomes:"Possible outcomes",
  interpretation:"Interpreting outcomes",discovers:"What the learner discovers",roles:{core:"Core",optional:"Optional"},
  topic:"Topic",topicNotLesson:"An independently accessible registry topic; a full authored lesson is not yet provided.",
  topicWithStation:"Linked journey station",goStation:"Open station",
  related:"Related topics",domain:"Domain",noRelation:"No recorded topic relationships for this entry.",
  atlasCount:"results out of",viewMore:"Show more",noResults:"No matching topic.",
  evidenceTitle:"Frozen Chapter 1 results — read-only replay",
  evidenceDescription:"A comparison on specific messages only. These numbers come from the frozen source; no experiment has been rerun here and no independent scientific approval is claimed.",
  evidenceSplit:"Message set",evidenceTotal:"Messages",evidenceRules:"Correct with rules",evidenceNearest:"Correct with nearest example",
  evidenceSeed:"Original six used to write the rules",evidenceTests:"Nine alternate test phrasings",
  evidenceSource:"Original results file:",
  evidenceLimit:"These are success counts on a small, defined sample. They do not establish general performance, statistical significance, or superiority on other data.",
  source:"Source",sourceInfo:"v0.21.3 — authored, frozen educational data; not a new training run",
  topicOpen:"Open independent topic",allTopics:"All topics",notFound:"Station or topic does not exist in the approved source."
 }
};
const q=id=>document.getElementById(id);
const el=(tag,cls,text)=>{const n=document.createElement(tag);if(cls)n.className=cls;if(text!==undefined)n.textContent=String(text);return n;};
const append=(p,...children)=>{p.append(...children);return p;};
const local=(object,l,field)=>object?.[field+"_"+l]||"";
const link=(hash,txt,cls)=>{const a=el("a",cls,txt);a.href=hash;return a;};
let catalog=null,language="ar",currentStation=1,currentTopic="FND-001",shown=40,search="",expandedChapter="",expandedEvidenceChapter="";
const getDict=()=>labels[language];
function setStaticLanguage(){
 const d=getDict();
 for(const n of document.querySelectorAll("[data-learn]")){const translation=d[n.dataset.learn];if(typeof translation==="string")n.textContent=translation;}
 q("topic-search").placeholder=d.placeholder;
}
function selectChapter(){
 return catalog.chapters.find(ch=>ch.station_orders.includes(currentStation))||catalog.chapters[0];
}
function renderJourney(){
 if(!catalog)return;
 const d=getDict(),chapter=selectChapter(),station=catalog.stations.find(s=>s.order===currentStation);
 const topics=new Map(catalog.topics.map(t=>[t.id,t]));
 const chapterList=q("chapter-nav"),stationList=q("station-nav"),area=q("station-content");
 chapterList.replaceChildren();stationList.replaceChildren();area.replaceChildren();
 for(const ch of catalog.chapters){
  const index=catalog.chapters.indexOf(ch)+1;
  const a=link("#station/"+ch.station_orders[0],(language==="ar"?"الفصل ":"Chapter ")+index+" · "+local(ch,language,"title"),"chapter-choice"+(ch.id===chapter.id?" active":""));
  if(ch.id===chapter.id)a.setAttribute("aria-current","true");
  chapterList.append(a);
 }
 const chapterHead=el("h3","station-list-label",d.chapter+" "+(catalog.chapters.indexOf(chapter)+1));
 stationList.append(chapterHead);
 for(const order of chapter.station_orders){
   const found=catalog.stations.find(s=>s.order===order),topic=topics.get(found.topic_id);
   const a=link("#station/"+order,order+" · "+local(topic,language,"title"),"station-choice"+(order===currentStation?" active":""));
   if(order===currentStation)a.setAttribute("aria-current","step");
   stationList.append(a);
 }
 const topic=topics.get(station.topic_id);
 const status=el("p","eyebrow",d.station+" "+station.order+" / 44 · "+station.topic_id+" · "+(d.roles[station.role]||station.role));
 const title=el("h3","station-main-title",local(topic,language,"title"));
 const subtitle=el("p","chapter-subtitle",d.chapter+" "+(catalog.chapters.indexOf(chapter)+1)+" · "+local(chapter,language,"title"));
 area.append(status,title,subtitle);
 const objective=el("div","learning-note");
 append(objective,el("strong","",d.goal),el("p","",local(chapter,language,"goal")));
 area.append(objective);
 for(const [titleKey,val] of [["why",local(station,language,"why_here")],["deliverable",local(station,language,"deliverable")]]){
  if(val)area.append(append(el("section","lesson-field"),el("h4","",d[titleKey]),el("p","",val)));
 }
 const act=station.activity;
 for(const key of ["sees","tries","hypothesis","possible_outcomes","interpretation","discovers"]){
   const section=el("section","lesson-field"+(key==="hypothesis"?" hypothesis-field":""));
   section.append(el("h4","",d[key]));
   if(key==="possible_outcomes"){
     const list=el("ol","outcomes");
     for(const outcome of local(act,language,key)){list.append(el("li","",outcome));}
     section.append(list);
   }else{section.append(el("p","",local(act,language,key)));}
   area.append(section);
 }
 // The HTML is rendered at build time from exactly 18 SHA-checked, approved
 // Markdown chapters, with raw HTML disabled. Never insert user input here.
 const approved=catalog.chapter_articles?.[chapter.id]?.[language];
 if(approved && typeof approved.html==="string"){
   const details=el("details","chapter-reader");
   details.open=expandedChapter===chapter.id;
   const summary=el("summary","chapter-reader-summary",d.fullChapter);
   const sourceInfo=el("p","chapter-provenance",d.chapterSource+": "+approved.source_sha256);
   const prose=el("div","chapter-prose");
   prose.lang=language;
   prose.dir=language==="ar"?"rtl":"ltr";
   prose.innerHTML=approved.html;
   details.append(summary,sourceInfo,prose);
   details.addEventListener("toggle",()=>{
      expandedChapter=details.open?chapter.id:"";
   });
   area.append(details);
 }
 // Aggregates are projected only at build time from SHA-verified frozen Chapter 1 results.
 // This is a read-only, scoped evidence view, not a rerun of the original experiment.
 const evidence=catalog.evidence_replays?.[chapter.id];
 if(evidence?.schema==="wakeel-v022-frozen-aggregate/v1" &&
    evidence.chapter_id===chapter.id && evidence.status==="approved-source-aggregate-not-independent-review"){
  const details=el("details","evidence-replay");
  details.open=expandedEvidenceChapter===chapter.id;
  details.append(el("summary","evidence-summary",d.evidenceTitle));
  details.append(el("p","evidence-note",d.evidenceDescription));
  const provenance=el("p","evidence-provenance",d.evidenceSource+" ");
  const origin=el("code","",evidence.source_path+" · SHA-256: "+evidence.source_sha256);
  origin.dir="ltr";provenance.append(origin);details.append(provenance);
  const scroll=el("div","evidence-scroll");
  const table=el("table","evidence-table");
  table.append(el("caption","",d.evidenceTitle));
  const head=el("thead"),header=el("tr");
  for(const label of [d.evidenceSplit,d.evidenceTotal,d.evidenceRules,d.evidenceNearest]){
   const th=el("th","",label);th.scope="col";header.append(th);
  }
  head.append(header);table.append(head);
  const body=el("tbody");
  const dataset=evidence.datasets?.[language];
  for(const [key,label] of [["seed",d.evidenceSeed],["tests",d.evidenceTests]]){
   const values=dataset?.[key];
   if(!values)continue;
   const row=el("tr"),name=el("th","",label);name.scope="row";row.append(name);
   const counts=[String(values.total),
     String(values.rules_correct)+" / "+String(values.total),
     String(values.nearest_example_correct)+" / "+String(values.total)];
   for(const count of counts){
    const cell=el("td","evidence-number",count);
    cell.dir="ltr";row.append(cell);
   }
   body.append(row);
  }
  table.append(body);scroll.append(table);details.append(scroll);
  details.append(el("p","evidence-note",d.evidenceLimit));
  details.addEventListener("toggle",()=>{
   expandedEvidenceChapter=details.open?chapter.id:"";
  });
  area.append(details);
 }
 const foot=append(el("div","learning-source"),el("span","",d.source+": "),el("span","",d.sourceInfo));
 area.append(foot);
 const nav=el("div","learning-next");
 if(station.order>1) nav.append(link("#station/"+(station.order-1),language==="ar"?"المحطة السابقة":"Previous station","secondary"));
 nav.append(link("#topic/"+station.topic_id,d.topicOpen,"secondary"));
 if(station.order<44) nav.append(link("#station/"+(station.order+1),language==="ar"?"المحطة التالية":"Next station","primary"));
 area.append(nav);
}
function matches(topic){
 const term=search.trim().toLocaleLowerCase();
 return !term||[topic.id,topic.title_ar,topic.title_en].some(v=>String(v||"").toLocaleLowerCase().includes(term));
}
function renderTopicList(){
 if(!catalog)return;
 const d=getDict(),all=catalog.topics.filter(matches),items=all.slice(0,shown),list=q("topic-results");
 list.replaceChildren();
 const count=el("p","topic-count",all.length+" "+d.atlasCount+" "+catalog.topics.length);
 list.append(count);
 for(const t of items){
  const a=link("#topic/"+encodeURIComponent(t.id),local(t,language,"title"),"topic-result"+(t.id===currentTopic?" active":""));
  a.append(el("small","",t.id));
  if(t.id===currentTopic)a.setAttribute("aria-current","true");
  list.append(a);
 }
 if(!all.length)list.append(el("p","hint",d.noResults));
 if(shown<all.length){
   const more=el("button","secondary show-more",d.viewMore);
   more.type="button";more.addEventListener("click",()=>{shown+=40;renderTopicList();});
   list.append(more);
 }
}
function renderTopic(){
 if(!catalog)return;
 const d=getDict(),t=catalog.topics.find(x=>x.id===currentTopic);
 const pane=q("topic-detail");pane.replaceChildren();
 if(!t){pane.append(el("p","",d.notFound));return;}
 const domain=catalog.domains.find(x=>x.id===t.domain_id),station=catalog.stations.find(s=>s.topic_id===t.id);
 pane.append(el("p","eyebrow",t.id+" · "+d.domain+" · "+(domain?local(domain,language,"title"):t.domain_id)));
 pane.append(el("h3","station-main-title",local(t,language,"title")));
 if(station){
  pane.append(append(el("div","learning-note"),el("strong","",d.topicWithStation),link("#station/"+station.order,d.goStation+" "+station.order,"secondary inline-link")));
 }else{
  pane.append(el("p","topic-limit",d.topicNotLesson));
 }
 const rel=catalog.relations.filter(r=>r.source===t.id||r.target===t.id);
 pane.append(el("h4","relations-title",d.related+" ("+rel.length+")"));
 const box=el("div","relations");
 for(const r of rel.slice(0,20)){
  const counterpart=r.source===t.id?r.target:r.source;
  const other=catalog.topics.find(x=>x.id===counterpart);
  if(other)box.append(link("#topic/"+other.id,local(other,language,"title")+" · "+r.type,"relation-link"));
 }
 if(!rel.length)box.append(el("p","hint",d.noRelation));
 pane.append(box);
}
function renderAll(){
 setStaticLanguage();renderJourney();renderTopic();renderTopicList();
}
/**
 * Parse a catalog deep link without allowing a malformed URI escape to throw
 * during a hashchange event. Catalog membership is checked by the caller.
 */
export function parseLearningRoute(hash){
 if(typeof hash!=="string")return null;
 let value;
 try{value=decodeURIComponent(hash.startsWith("#")?hash.slice(1):hash);}
 catch{return null;}
 const station=/^station\/([1-9]\d?)$/.exec(value);
 if(station){
  const order=Number(station[1]);
  return order<=44?{kind:"station",order}:null;
 }
 const chapter=/^chapter\/([1-9])$/.exec(value);
 if(chapter)return {kind:"chapter",index:Number(chapter[1])};
 const topic=/^topic\/([A-Za-z0-9-]+)$/.exec(value);
 return topic?{kind:"topic",id:topic[1]}:null;
}
function navigateHash({moveFocus=false}={}){
 if(!catalog)return;
 const route=parseLearningRoute(location.hash);
 if(!route)return;
 const reveal=(sectionId,headingSelector)=>{
  const section=q(sectionId);
  section.scrollIntoView({block:"start"});
  if(moveFocus){
   const heading=section.querySelector(headingSelector);
   if(heading){
    heading.setAttribute("tabindex","-1");
    heading.focus({preventScroll:true});
   }
  }
 };
 if(route.kind==="station"){
  currentStation=route.order;
  renderJourney();
  reveal("journey","#station-content .station-main-title");
 }else if(route.kind==="chapter"){
  const chapter=catalog.chapters[route.index-1];
  if(chapter){
   currentStation=chapter.station_orders[0];
   expandedChapter=chapter.id;
   renderJourney();
   reveal("journey","#station-content .station-main-title");
  }
 }else if(route.kind==="topic"){
  if(catalog.topics.some(t=>t.id===route.id)){
   currentTopic=route.id;
   renderTopic();
   renderTopicList();
   reveal("atlas","#topic-detail .station-main-title");
  }
 }
}
export async function initLearning(){
 const area=q("station-content");
 if(!area)return;
 area.textContent=getDict().loading;
 q("topic-search").addEventListener("input",e=>{search=e.target.value;shown=40;renderTopicList();});
 window.addEventListener("hashchange",()=>navigateHash({moveFocus:true}));
 document.addEventListener("wakeel:locale",e=>{
  language=e.detail?.locale==="en"?"en":"ar";renderAll();
 });
 try{
  const response=await fetch("./data/learning.json",{cache:"no-store"});
  if(!response.ok)throw new Error("HTTP "+response.status);
  const data=await response.json();
  if(data.schema!=="wakeel-v022-learning-catalog/v1" || data.counts.stations!==44 ||
     data.counts.topics!==389 || data.counts.chapters!==9 || data.counts.relations!==125 ||
     !data.chapter_articles || Object.keys(data.chapter_articles).length!==9)throw new Error("Invalid approved catalog");
  catalog=data;renderAll();navigateHash();
 }catch(err){
  console.error("Wakeel catalog loading failed",err);
  area.textContent=getDict().loadError;
  q("topic-detail").textContent=getDict().loadError;
 }
}