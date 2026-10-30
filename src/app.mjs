import { START, END, fitBigram, distribution, initialAssociations, swapAssociations, lookupAssociation, assessAssociation } from "./models.mjs";

const samples = {
  ar: "الذكاء يتعلم من البيانات. الذكاء يتغير بالتجربة.",
  en: "models learn from data. models learn by testing."
};
const t = {
  ar: {
    skip:"تجاوز إلى المحتوى",brandSubtitle:"تعلم الذكاء الاصطناعي بالتجربة",
    navBigram:"النموذج الأول",navAssociations:"الارتباطات",langLabel:"اللغة",
    eyebrow:"نسخة تطوير تجريبية · v0.22",
    headline:"لا تكتفِ بمعرفة الإجابة. غيّر التجربة واكتشف لماذا.",
    intro:"ابدأ بنموذج يحسب احتمالات الحرف التالي، ثم اختبر ارتباط المفتاح بالقيمة. كل نتيجة هنا توضيحية، وليست ادعاءً عن قدرات نموذج لغوي مدرّب.",
    start:"ابدأ المختبر",explore:"استكشف الارتباطات",
    heroLimit:"هذه محاكيات تعليمية محلية، لا تمثل أدلة التجارب العلمية المجمدة في وكيل v1.58.",
    bigramTag:"تجربة قابلة لإعادة التشغيل",bigramTitle:"من عدّ الحروف إلى توقع الحرف التالي",
    bigramDesc:"نموذج Bigram حرفي بسيط: يرى فقط الحرف السابق، ويقدر احتمال الحرف الذي يليه من عدد مرات ظهور كل انتقال.",
    corpusLabel:"النص المستخدم لبناء جدول العدّ",corpusHint:"عدّل النص لتكتشف كيف تتغير الاحتمالات. حتى 320 حرف Unicode.",
    contextLabel:"الحرف السابق",smoothingLabel:"التنعيم",smoothOn:"إضافة واحد",smoothOff:"بدون",
    bigramLimit:"التنعيم خيار تعليمي. الاحتمالات ليست نتيجة تدريب شبكة عصبية.",
    distributionTitle:"توزيع الحرف التالي",computed:"محسوب مباشرة",
    assocTag:"اختبار مضاد بسيط",assocTitle:"هل تتبع الإجابة المفتاح أم موقع القيمة؟",
    assocDesc:"ابدأ بجدول مفتاح/قيمة، توقع الإجابة، ثم بدّل القيم مع الإبقاء على المفاتيح واختبر توقعك مجدداً.",
    keyA:"المفتاح A",keyB:"المفتاح B",swap:"بدّل القيم",question:"ما قيمة المفتاح؟",
    yourAnswer:"توقعك",check:"افحص التوقع",observationTitle:"الملاحظة",notEvidence:"تمثيل توضيحي",
    assocLimit:"هذه ذاكرة ارتباطات مصغرة؛ ليست آلية Attention ولا اختباراً تجريبياً لقدرة Transformer.",
    footer:"وكيل — المعرفة تبدأ بسؤال يمكن اختباره.",
    footerLimit:"لم يُنقل بعد مصدر v1.58 إلى هذا المستودع؛ هذه تجربة واجهة مستقلة للمراجعة.",
    startLabel:"بداية النص",endLabel:"نهاية النص",spaceLabel:"مسافة",
    corpusError:"أدخل نصاً من حرف واحد على الأقل، وبحد أقصى 320 حرفاً.",
    noSamples:"لم يُرصد انتقال لهذه الحالة. أضف بيانات أو اختر التنعيم.",
    nextExplain:"كل نسبة = (مرات الانتقال + التنعيم) ÷ (مجموع الانتقالات + التنعيم × حجم المفردات).",
    needAnswer:"اختر توقعاً قبل الفحص.",preCheck:"توقع القيمة ثم افحصها؛ تبديل القيم يعيد التقييم إلى حالة لم تُختبر.",
    correct:"توقع مطابق للجدول الحالي.",incorrect:"التوقع لا يطابق الجدول الحالي.",
    actual:"القيمة الحالية للمفتاح",changed:"تبدّلت القيم؛ اختر توقعاً جديداً."
  },
  en: {
    skip:"Skip to content",brandSubtitle:"Learning AI through experimentation",
    navBigram:"First model",navAssociations:"Associations",langLabel:"Language",
    eyebrow:"Experimental development · v0.22",
    headline:"Don't just read the answer. Change the experiment and discover why.",
    intro:"Start with a next-character probability model, then test key/value associations. Results here are illustrative, not claims about the abilities of a trained language model.",
    start:"Start the lab",explore:"Explore associations",
    heroLimit:"These are local teaching simulations, not the frozen scientific experiment evidence in Wakeel v1.58.",
    bigramTag:"Reproducible activity",bigramTitle:"From character counts to next-character predictions",
    bigramDesc:"A simple character-level Bigram: it sees only the previous character and estimates the following character's probability from transition counts.",
    corpusLabel:"Corpus used for the count table",corpusHint:"Edit the text and observe the probabilities. Up to 320 Unicode characters.",
    contextLabel:"Previous character",smoothingLabel:"Smoothing",smoothOn:"Add one",smoothOff:"None",
    bigramLimit:"Smoothing is a teaching option. These probabilities do not come from neural-network training.",
    distributionTitle:"Next-character distribution",computed:"Computed live",
    assocTag:"Simple counterfactual",assocTitle:"Does the answer follow the key or the value's position?",
    assocDesc:"Start with a key/value table, predict a lookup, then swap the values while preserving the keys and test again.",
    keyA:"Key A",keyB:"Key B",swap:"Swap values",question:"Which key?",yourAnswer:"Your prediction",
    check:"Check prediction",observationTitle:"Observation",notEvidence:"Illustrative analogy",
    assocLimit:"This is a tiny association memory, not the Attention mechanism or an empirical test of a Transformer.",
    footer:"Wakeel — knowledge starts with a testable question.",
    footerLimit:"The v1.58 source has not yet been imported into this repository; this is a separate UI experiment for review.",
    startLabel:"Start of text",endLabel:"End of text",spaceLabel:"Space",
    corpusError:"Enter between 1 and 320 Unicode characters.",
    noSamples:"No transition was observed in this state. Add data or enable smoothing.",
    nextExplain:"Each probability = (transition count + smoothing) ÷ (total transitions + smoothing × vocabulary size).",
    needAnswer:"Select a prediction before checking.",preCheck:"Predict a value, then check; swapping the values resets the untested prediction.",
    correct:"Prediction matches the current table.",incorrect:"Prediction does not match the current table.",
    actual:"Current value for key",changed:"Values swapped. Choose a new prediction."
  }
};
const $ = (id) => document.getElementById(id);
let locale = "ar";
let customized = false;
let pairs = initialAssociations();
let associationMessage = "preCheck";
let associationState = "";
function translate() {
  const dict = t[locale];
  document.documentElement.lang = locale;
  document.documentElement.dir = locale === "ar" ? "rtl" : "ltr";
  document.title = locale === "ar" ? "وكيل | المختبرات التفاعلية" : "Wakeel | Interactive labs";
  for (const node of document.querySelectorAll("[data-i18n]")) {
    node.textContent = dict[node.dataset.i18n];
  }
}
function readable(token) {
  if (token === START) return t[locale].startLabel;
  if (token === END) return t[locale].endLabel;
  if (token === " ") return t[locale].spaceLabel;
  return token;
}
function renderBigram() {
  const area = $("bigram-output");
  area.replaceChildren();
  let model;
  try {
    model = fitBigram($("corpus").value);
  } catch {
    const error = document.createElement("p");
    error.textContent = t[locale].corpusError;
    area.append(error);
    $("context").replaceChildren();
    return;
  }
  const select = $("context");
  const previous = select.value;
  select.replaceChildren();
  for (const token of model.contexts) {
    const option = document.createElement("option");
    option.value = token;
    option.textContent = readable(token);
    select.append(option);
  }
  select.value = model.contexts.includes(previous) ? previous : START;
  const rows = distribution(model, select.value, Number($("smoothing").value));
  if (!rows.some((row) => row.probability > 0)) {
    const warning = document.createElement("p");
    warning.textContent = t[locale].noSamples;
    area.append(warning);
    return;
  }
  for (const row of rows) {
    const outer = document.createElement("div");
    outer.className = "bar-row";
    const token = document.createElement("span");
    token.className = "bar-token";
    token.textContent = readable(row.token);
    const track = document.createElement("div");
    track.className = "bar-track";
    const fill = document.createElement("div");
    fill.className = "bar-fill";
    fill.style.width = (row.probability * 100).toFixed(2) + "%";
    track.append(fill);
    const pct = document.createElement("span");
    pct.className = "bar-percent";
    pct.textContent = (row.probability * 100).toFixed(1) + "%";
    pct.title = String(row.count);
    outer.append(token, track, pct);
    area.append(outer);
  }
  const note = document.createElement("p");
  note.className = "explanation";
  note.textContent = t[locale].nextExplain;
  area.append(note);
}
function renderAssociations() {
  $("value-a").textContent = String(pairs.A);
  $("value-b").textContent = String(pairs.B);
  const output = $("association-output");
  output.className = "observation" + (associationState ? " " + associationState : "");
  output.textContent = t[locale][associationMessage];
  if (associationMessage === "correct" || associationMessage === "incorrect") {
    output.textContent += " " + t[locale].actual + " " + $("query-key").value + " = " + lookupAssociation(pairs, $("query-key").value) + ".";
  }
}
function resetPrediction(message) {
  $("answer").value = "";
  associationMessage = message;
  associationState = "";
  renderAssociations();
}
$("locale").addEventListener("change", () => {
  locale = $("locale").value === "en" ? "en" : "ar";
  if (!customized) $("corpus").value = samples[locale];
  translate();
  renderBigram();
  renderAssociations();
});
$("corpus").addEventListener("input", () => { customized = true; renderBigram(); });
$("context").addEventListener("change", renderBigram);
$("smoothing").addEventListener("change", renderBigram);
$("query-key").addEventListener("change", () => resetPrediction("preCheck"));
$("answer").addEventListener("change", () => {
  associationMessage = "preCheck";
  associationState = "";
  renderAssociations();
});
$("swap").addEventListener("click", () => {
  pairs = swapAssociations(pairs);
  resetPrediction("changed");
});
$("check").addEventListener("click", () => {
  const answer = $("answer").value;
  if (answer !== "1" && answer !== "2") {
    associationMessage = "needAnswer";
    associationState = "bad";
  } else {
    const ok = assessAssociation(pairs, $("query-key").value, answer);
    associationMessage = ok ? "correct" : "incorrect";
    associationState = ok ? "good" : "bad";
  }
  renderAssociations();
});
$("corpus").value = samples.ar;
translate();
renderBigram();
renderAssociations();