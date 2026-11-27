import {smoothedTransition,denseContextTableCells,toyScalarEmbeddingUpdate,toyMemoryRetention,toyAttentionOrder,toyCausalPrefix} from "./models.mjs";
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
  causalTitle:"الفصل الثامن — الانتباه الذاتي السببي والاسترجاع المزدوج",
  causalIntro:"استرجاع قيمتين مستقلتين A وB من سجلات ذرية. هذه مهمة اختيار سجل، لا إثبات ربط مفتاح وقيمة منفصلين. النموذج رأس واحد وطبقة واحدة، وليس Transformer كاملاً. التدريب 512 حلقة، التحقق والتطويري 128 لكل منهما، والاختبار 64 لكل مقطع. طول الحشو 16 خارج أطوال التدريب 2،4،8.",
  causalMain:"نتائج المصدر المجمد — اختر مقطع التقييم",
  causalConfirm:"التجربة العاملية 2×2 — مجموعة تأكيدية جديدة",
  causalSplit:"مقطع التقييم",causalVal:"التحقق (اختيار المعدل)",causalDev:"التطويري (للتقرير فقط)",
  causalIn:"داخل التوزيع",causalOod:"طول الحشو المحجوز 16",causalModel:"النموذج",causalParams:"المعلمات",
  causalBinary:"الدقة الثنائية",causalFull:"دقة كامل المفردة",causalNll:"NLL كامل المفردة",causalBinNll:"NLL ثنائي منمّط",
  causalLimit:"الأرقام متوسط ثلاث بذور 42،43،44. القارئ الثابت والاستعلام المشتق يختلفان أيضاً في سلك الإخراج. أُعيد تشغيل الاختبار النهائي بعد تصحيح تجميع التدرجات وفق الميزانية المجمدة؛ ليس اختباراً لم يسبق الاطلاع عليه.",
  causalConfirmLimit:"المجموعة التأكيدية الجديدة: 64 حلقة لكل مقطع، بذرة بيانات 9204، دون اختيار إعدادات جديد. أعداد المعلمات 680/736/936/992 غير متساوية؛ المقارنة العاملية اتجاهية ولا تعزل الأثر السببي لمصدر الاستعلام.",
  causalLab:"مختبر مستقل — ثبات البادئة عند تغيير المستقبل",
  causalLabIntro:"أربع قيم عددية، الاستعلام ثابت Q=1 عند الموضع 1 (الترقيم من صفر)، والمفتاح والقيمة يساويان العدد نفسه. غيّر القيمتين المستقبليتين وقارن القراءة بالقناع وبدونه؛ ليس تدريباً ولا إعادة تنفيذ للنموذج الأصلي.",
  causalFuture1:"قيمة المستقبل في الموضع 2",causalFuture2:"قيمة المستقبل في الموضع 3",causalMask:"القناع السببي",causalOn:"مفعّل",causalOff:"غير مفعّل",
  causalOutput:"مخرج القراءة",causalPrefix:"مخرج البادئة وحدها",causalFutureWeight:"مجموع أوزان المستقبل",
  causalPosition:"الموضع",causalValue:"القيمة",causalWeight:"وزن الانتباه",
  causalLabLimit:"عند تفعيل القناع تكون أوزان المستقبل صفراً ويتساوى المخرج مع البادئة. تعطيل القناع ضابط تعليمي مضاد؛ هذا المثال العددي لا يثبت صحة جميع تطبيقات الانتباه ولا يقيس دقة النموذج.",
  attentionTitle:"الفصل السابع — الاسترجاع وتمييز الترتيب",
  attentionIntro:"نتائج المصدر المجمد لقارئ أحادي الاستعلام، لا Transformer كامل. في الاسترجاع 100 مثال تدريب و30 تحقق و40 اختبار محجوز لكل فئة؛ في الترتيب 100/30/40 زوجاً متقابلاً. ثلاث بذور، 300 عصر، معدل تعلم 0.3. الاختبار المحجوز لا يستخدم للضبط. النتائج متوسط ثلاث بذور لنماذج مدربة عند كل تأخير، وليست اختبار أطوال جديدة لنموذج واحد. خسارة جدول الترتيب تخص الترتيب القياسي.",
  attentionRetrieval:"الاسترجاع — الاختبار المحجوز",attentionOrder:"تمييز الترتيب — الاختبار المحجوز",
  attentionDelay:"طول الحشو",attentionAccuracy:"الدقة الثنائية",attentionNll:"خسارة الاختبار NLL",
  attentionRetrievalLimit:"رمز الفئة فريد وغير موجود في الحشو؛ قد ينجح قارئ المحتوى دون تعلم الموضع. عند 16، الدقة 79.2% لكن NLL=0.7113 أعلى من ln(2)≈0.6931، وهو مرجع احتمالي ثنائي متوازن، لا توزيعاً منتظماً على المفردات الكاملة. لا تعني الدقة الأعلى احتمالات أفضل.",
  attentionStandard:"الترتيب القياسي",attentionShifted:"الترتيب المزاح",
  attentionModel:"النموذج",attentionParams:"المعلمات",attentionOracle:"مرجع يعرف الموضع مسبقاً",
  attentionOrderLimit:"الأزواج لها المحتوى نفسه بترتيبين. قارئ المتوسط وقارئ المحتوى بلا موضع متطابقان في مخرجات كل زوج؛ أما إخفاق RNN والانتباه الموضعي في هذه الإعدادات فلا يثبت عجزاً بنيوياً. pos0 وpos_mid_shift تحكمان يقرآن موضعاً معلوماً بالبناء، والثاني مدرّب على التوزيع المزاح.",
  attentionSensitivity:"حساسية التهيئة — تحقق استكشافي فقط",
  attentionSensitivityLimit:"التهيئة ×3 لكل المجموعات نجحت على التحقق؛ لا نتيجة اختبار محجوز لهذه الإعدادات هنا. لم تعزل التجربة السبب، وزيادة معدل التعلم وحده لم تحقق النجاح نفسه في التجارب المسجلة.",
  attentionAmp:"التهيئة",attentionOriginal:"الأصل ×1",attentionAll:"كل المجموعات ×3",
  attentionKQ:"المفاتيح والاستعلام فقط ×3",attentionV:"القيم فقط ×3",attentionP:"الموضع فقط ×3",
  attentionLab:"تمرين مستقل: قارئ موزون لزوج متقابل",
  attentionLabIntro:"A=1 وB=−1 والحشو=0. الدرجة = معامل المحتوى × القيمة + انحياز موضع محدد، ثم softmax ومجموع موزون. لا أوزان متعلمة ولا مصنّف. اختر إزاحة الرمزين وانحياز الموضع، وقارن الزوجين؛ الموضع الصحيح معلوم للمرجع وحده.",
  attentionScale:"معامل المحتوى",attentionBias:"انحياز الموضع",attentionIndex:"موضع الانحياز (يبدأ من 0)",
  attentionPosition:"موضع",attentionValue:"القيمة",attentionWeight:"وزن softmax",
  attentionPairA:"الزوج الأول",attentionPairB:"الزوج الثاني",attentionSum:"المجموع الموزون",
  attentionLabLimit:"دون انحياز موضعي، تبديل الرمزين لا يغير المجموع الموزون. إعطاء القارئ موضعاً معروفاً لا يثبت تعلم العثور عليه. المجموع ليس احتمالاً أو دقة استرجاع.",
  memoryTitle:"الفصل السادس — الذاكرة المتكررة والتعميم على الأطوال",
  memoryIntro:"مهمة استرجاع الحرف الأول بعد حشو عشوائي ثم علامة السؤال. القياس بعد استهلاك علامة السؤال؛ الخسارة على موضع الإجابة فقط.",
  memoryProtocol:"طول ثابت: تدريب مستقل لكل تأخير ولكل بذرة (120 تسلسلاً لكل فئة). مزيج الأطوال: نموذج واحد لكل بذرة (60 لكل فئة لكل طول من 2،4،5،6،7،8). الاختبار عند كل تأخير: 40 مثالاً لكل فئة؛ البذور 42،43،44. كلاهما 300 عصر ومعدل تعلم 0.3. التأخير 16 محجوز في تجربة المزيج فقط. لكل نموذج ثابت 240 تسلسلاً تدريبياً، ولكل نموذج مزيج 720؛ لذلك لا تعزل المقارنة أثر تنوع الأطوال عن حجم بيانات التدريب.",
  memoryDelay:"طول الحشو",memoryFixed:"تدريب بطول ثابت",memoryMixed:"تدريب بمزيج أطوال",
  memoryDomain:"مجال تجربة المزيج",memoryInside:"ضمن أطوال التدريب",memoryOutside:"طول محجوز",
  memoryAccuracy:"الدقة الثنائية (متوسط 3 بذور)",memorySeedAccuracy:"الدقة الثنائية",memoryNll:"خسارة الإجابة على الاختبار (NLL)",
  memoryUntrained:"خط الأساس غير المدرّب (ثنائي)",memoryOracle:"مرجع Oracle",
  memorySeeds:"تفصيل النتائج حسب البذرة",memorySeed:"البذرة",memoryCorpus:"RNN على رسائل الاجتماع (متوسط 3 بذور)",
  memoryValidation:"خسارة التحقق",memoryTest:"خسارة الاختبار",memoryParameters:"المعلمات",
  memoryCaveat:"المقارنة بين بروتوكولي تدريب مختلفين، لا قياس ذاكرة نموذج واحد عبر الأطوال الثابتة. نجحت بذرتان من ثلاث في تجربة المزيج عند 16، وفشلت الثالثة. هذا لا يثبت حد ذاكرة عاماً أو تفوقاً معمارياً، ولم تُختبر أطوال أكبر من 16.",
  memoryLabTitle:"تمرين مستقل: أثر اضمحلال حالة عددية",
  memoryLabIntro:"نموذج حسابي مبسط بلا تدريب أو حشو مؤثر أو مصنّف: h(t+1)=ρh(t). غيّر الحالة الأولية ومعامل الاحتفاظ وعدد الخطوات.",
  memoryInitial:"الحالة الأولية",memoryRetention:"معامل الاحتفاظ ρ",memorySteps:"عدد الخطوات",
  memoryFinal:"الحالة النهائية",memoryMagnitude:"مقدار الحالة المطلق",
  memoryTrace:"مسار الحالة",memoryStep:"الخطوة",memoryState:"الحالة",
  memoryLabLimit:"مقدار الحالة ليس احتمال استرجاع ولا دقة RNN. حتى مقدار صغير غير صفري قد يحتفظ بإشارته في الحساب المثالي. لا تمثل هذه المعادلة الشبكة المتعلمة في التجربة الأصلية.",
  embeddingTitle:"الفصل الخامس — التمثيلات المتعلمة وعزل التضمينات",
  embeddingIntro:"نتائج أصلية من المصدر المجمد: نموذج MLP بتضمينات قابلة للتعلم، ونفس التجربة بتضمينات عشوائية مجمدة. لا تدريب جديد في المتصفح.",
  embeddingModel:"النموذج",embeddingValidation:"خسارة التحقق",embeddingTest:"خسارة الاختبار",
  embeddingTrainable:"MLP — تضمينات قابلة للتعلم",embeddingFrozen:"MLP — تضمينات مجمدة عشوائياً",
  embeddingTrigram:"Trigram بالعدّ (مرجع الفصل الرابع)",
  embeddingTrigramSource:"مرجع Trigram المستقل من الفصل الرابع:",
  embeddingTrainableParams:"المعلمات القابلة للتدريب",
  embeddingParams:"تفصيل المعلمات: الإجمالي / التضمينات / المخفية / المخرجات",
  embeddingContexts:"سياقات التحقق",embeddingSeen:"مرصودة في التدريب",embeddingUnseen:"غير مرصودة في التدريب",
  embeddingCount:"عدد المواضع",embeddingLoss:"متوسط الخسارة",
  embeddingCurve:"منحنى خسارة تدريب MLP بالتضمينات المتعلمة",
  embeddingEpoch:"العصر",embeddingTrainLoss:"خسارة التدريب",
  embeddingCaveat:"في هذه التجربة فقط كانت خسارة التحقق والاختبار في Trigram أقل من MLP. فرق النسختين المتعلمة والمجمدة لا يثبت أن الأبعاد اكتسبت معاني لغوية؛ ومجموعة الاختبار ليست اختبار تعميم مستقلاً.",
  embeddingLabTitle:"جرّب فرق معامل قابل للتعلم ومعامل مجمد",
  embeddingLabIntro:"تمرين مستقل على معامل عددي واحد وخسارة تربيعية. يقارن تحديث المعامل بالتدرج مقابل تثبيته؛ ليس تدريب MLP ولا تضمينات الفصل الأصلية.",
  embeddingInitial:"قيمة المعامل الأولية",embeddingTarget:"القيمة المستهدفة",
  embeddingRate:"معدل التعلم",embeddingSteps:"خطوات التحديث",
  embeddingValue:"قيمة المعامل",embeddingSingleLoss:"خسارة المثال",
  embeddingInvalid:"أدخل قيماً عددية بين −2 و2.",
  embeddingDemoLimit:"هذا تمرين على خسارة مثال واحد. لا يجوز مقارنة أرقامه بخسائر التحقق والاختبار المرجعية أو استنتاج تعميم النموذج.",
  contextEvidenceTitle:"الفصل الرابع — طول السياق وندرة البيانات",
  contextEvidenceIntro:"نتائج النماذج من المصدر المجمد: خسائر التحقق والاختبار، والسياقات غير المرصودة، وحجم الجدول النظري مقارنة بعدد سجلات السياق–المحرف المخزنة.",
  contextOrder:"طول السياق (k)",contextValidation:"خسارة التحقق",
  contextTest:"خسارة الاختبار",contextUnseen:"سياقات تحقق غير مرصودة",
  contextDense:"خلايا الجدول الكثيف (نظرياً)",contextSparse:"سجلات السياق–المحرف المخزنة",
  contextAlpha:"التنعيم المختار على مجموعة التحقق",
  contextCaveat:"هذه نتائج مجموعة صغيرة اطلعنا عليها سابقاً، وليست إثبات تعميم مستقل. أرقام Backoff مستبعدة لأن درجاتها غير مطبّعة ولا تصلح للمقارنة المباشرة كاحتمالات.",
  contextLabTitle:"جرّب نمو حجم جدول السياق",
  contextLabIntro:"غيّر طول السياق وحجمي المفردات. الناتج عدد خلايا نظري فقط؛ لا يخصص ذاكرة ولا يشغّل نموذجاً. أعداد السجلات الفعلية في الجدول أعلاه تبقى ثابتة.",
  contextVocabulary:"مفردات السياق (تتضمن رمز البداية)",
  contextOutputVocabulary:"مفردات المخرجات",contextCells:"الخلايا النظرية",
  contextInvalid:"القيم غير صالحة: 2–64 لكل حجم مفردات و0–3 لطول السياق.",
  contextFormula:"المعادلة: V_context^k × V_output. حجم الجدول ليس قياساً مباشراً لاستهلاك الذاكرة أو زمن التشغيل.",
  bigramEvidenceTitle:"الفصل الثالث — قياسات النماذج الأصلية",
  bigramEvidenceIntro:"قيم خسارة التحقق والاختبار مسجلة في المصدر المجمد. تُقارن النماذج على بيانات الفصل نفسه وبطريقة التمثيل نفسها؛ الأرقام ليست نتائج تشغيل جديد.",
  bigramModel:"النموذج",bigramValidation:"خسارة التحقق",bigramTest:"خسارة الاختبار",
  bigramUnigram:"أحادي التكرار",bigramCount:"Bigram بالعدّ",bigramNeural:"Bigram عصبي",
  bigramChosen:"قيمة التنعيم المختارة على مجموعة التحقق:",
  bigramLabTitle:"جرّب حساب احتمال انتقال واحد",
  bigramLabIntro:"غيّر الأعداد ضمن الحدود المحددة. هذا حساب جديد بمعادلة التنعيم، وليس إعادة تدريب للنموذج أو إعادة تشغيل اختبار الفصل.",
  bigramTotal:"مرات ظهور السياق (n)",bigramObserved:"مرات الانتقال الصحيح (c)",
  bigramVocabulary:"حجم مفردات المخرجات (V)",bigramAlpha:"قوة التنعيم (α)",
  bigramOutput:"احتمال الانتقال وخسارته",bigramProbability:"احتمال الانتقال",
  bigramLoss:"خسارة الانتقال (nats)",bigramInvalid:"أدخل أعداداً صحيحة ضمن الحدود، مع 0 ≤ c ≤ n.",
  bigramReference:"مثال المصدر المجمد: n=1، c=1، V=6، α=1؛ الاحتمال المسجل:",
  bigramCaveat:"خسارة انتقال واحد (−ln p) لا تُقارن مباشرةً بمتوسط خسارة التحقق أو الاختبار في الجدول. لا يمكن استنتاج تعميم النموذج من هذا الحساب.",
  tokensTitle:"الفصل الثاني — مقارنة وحدات النص المجمدة",
  tokensDescription:"مثالان مرجعيان مختلفان في الصياغة؛ يتغير عدد الوحدات عند تغيير طريقة التقسيم. النتائج الأصلية معروضة دون إعادة تشغيل المُقسّم.",
  tokensMethod:"طريقة التقسيم",tokensCount:"عدد الوحدات",
  tokensWhitespace:"الكلمات المفصولة بالمسافات (مع إبقاء علامات الترقيم)",
  tokensEducational:"المُقسّم التعليمي المحدد بالإصدار",
  tokensLimit:"الجملتان ليستا ترجمة متطابقة. لا تثبت الأرقام أن لغة ما أعلى كلفة من الأخرى؛ كما أنها لا تقيس نموذجاً إنتاجياً.",
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
  causalTitle:"Chapter 8 — causal self-attention and dual retrieval",
  causalIntro:"Retrieve two independent values A and B from atomic records. This tests record selection, not separate key/value binding. One head, one layer, not a full Transformer. Training has 512 loops; validation and development each have 128; each test split has 64. Filler length 16 is outside training lengths 2,4,8.",
  causalMain:"Frozen source results — select evaluation split",
  causalConfirm:"2×2 factorial experiment — new confirmation set",
  causalSplit:"Evaluation split",causalVal:"Validation (rate selection)",causalDev:"Development (report only)",
  causalIn:"In-distribution",causalOod:"Reserved filler length 16",causalModel:"Model",causalParams:"Parameters",
  causalBinary:"Binary accuracy",causalFull:"Full-vocabulary accuracy",causalNll:"Full-vocabulary NLL",causalBinNll:"Binary-normalized NLL",
  causalLimit:"Means of seeds 42,43,44. Fixed and state-derived query readers also differ in output wiring. The final test was reread after the gradient-aggregation fix under the frozen budget; it is not a never-before-read test.",
  causalConfirmLimit:"New confirmation sets: 64 loops per split, data seed 9204, no new settings selection. Parameter counts 680/736/936/992 differ; the factorial comparison is directional and does not isolate the causal effect of query source.",
  causalLab:"Separate lab — prefix invariance under future changes",
  causalLabIntro:"Four scalar values, fixed Q=1 at position 1 (zero-based), with each key and value equal to its scalar. Change the two future values and compare masked and unmasked reads. No training or reproduction of the original model.",
  causalFuture1:"Future value at position 2",causalFuture2:"Future value at position 3",causalMask:"Causal mask",causalOn:"Enabled",causalOff:"Disabled",
  causalOutput:"Reader output",causalPrefix:"Prefix-only output",causalFutureWeight:"Total future weight",
  causalPosition:"Position",causalValue:"Value",causalWeight:"Attention weight",
  causalLabLimit:"With masking enabled, future weights are zero and the output equals the prefix-only read. Disabling masking is a teaching counter-control; this scalar example does not establish correctness of all attention implementations or model accuracy.",
  attentionTitle:"Chapter 7 — retrieval and order discrimination",
  attentionIntro:"Frozen single-query reader results, not a full Transformer. Retrieval: 100 train, 30 validation, 40 holdout per class; order: 100/30/40 matched pairs. Three seeds, 300 epochs, learning rate 0.3. Holdout is not used for tuning. Results are means of three seeds trained at each delay, not unseen-length evaluation of one model. Order-table NLL refers to standard-order holdout.",
  attentionRetrieval:"Retrieval — holdout",attentionOrder:"Order discrimination — holdout",
  attentionDelay:"Filler length",attentionAccuracy:"Binary accuracy",attentionNll:"Holdout NLL",
  attentionRetrievalLimit:"The class token is unique and absent from filler, so content retrieval can succeed without learning position. At 16, accuracy is 79.2% but NLL=0.7113 exceeds ln(2)≈0.6931, a balanced binary probabilistic reference, not uniform over the full vocabulary. Higher accuracy need not mean better probabilities.",
  attentionStandard:"Standard order",attentionShifted:"Shifted order",
  attentionModel:"Model",attentionParams:"Parameters",attentionOracle:"Known-position oracle",
  attentionOrderLimit:"Matched pairs have identical content in different orders. Mean and no-position readers have identical within-pair logits. Failure of RNN and positional attention in these settings does not establish structural incapacity. pos0 and pos_mid_shift are known-position positive controls; the latter is trained on the shifted distribution.",
  attentionSensitivity:"Initialization sensitivity — exploratory validation only",
  attentionSensitivityLimit:"Scaling all groups ×3 succeeds on validation; no holdout result for these settings is reported here. The mechanism is not isolated; increasing learning rate alone did not reproduce the success in the recorded experiments.",
  attentionAmp:"Initialization",attentionOriginal:"Original ×1",attentionAll:"All groups ×3",
  attentionKQ:"Keys and query only ×3",attentionV:"Values only ×3",attentionP:"Positions only ×3",
  attentionLab:"Separate exercise: weighted reader on a matched pair",
  attentionLabIntro:"A=1, B=−1, filler=0. Score = content scale × value + selected-position bias, then softmax and weighted sum. No learned weights or classifier. Shift the symbols and select a position bias; only the reference is given the correct position.",
  attentionScale:"Content scale",attentionBias:"Position bias",attentionIndex:"Bias index (zero-based)",
  attentionPosition:"Position",attentionValue:"Value",attentionWeight:"Softmax weight",
  attentionPairA:"First counterpart",attentionPairB:"Second counterpart",attentionSum:"Weighted sum",
  attentionLabLimit:"Without positional bias, swapping the symbols leaves the weighted sum unchanged. Supplying a known position does not demonstrate learned retrieval. The sum is neither a probability nor recall accuracy.",
  memoryTitle:"Chapter 6 — recurrent memory and length generalization",
  memoryIntro:"Retrieve the first character after random filler and a question mark. Evaluation consumes the question mark; loss is computed at the answer position only.",
  memoryProtocol:"Fixed length: a separately trained model for each delay and seed (120 sequences per class). Mixed lengths: one model per seed (60 per class per training delay 2,4,5,6,7,8). Test at each delay: 40 examples per class; seeds 42,43,44. Both use 300 epochs and learning rate 0.3. Delay 16 is held out only in mixed-length training. Each fixed-delay model sees 240 training sequences, each mixed model 720; the comparison does not isolate length diversity from training-set size.",
  memoryDelay:"Filler length",memoryFixed:"Fixed-length training",memoryMixed:"Mixed-length training",
  memoryDomain:"Mixed-length domain",memoryInside:"Training length",memoryOutside:"Held-out length",
  memoryAccuracy:"Binary accuracy (mean of 3 seeds)",memorySeedAccuracy:"Binary accuracy",memoryNll:"Test answer NLL",
  memoryUntrained:"Untrained binary baseline",memoryOracle:"Oracle reference",
  memorySeeds:"Per-seed results",memorySeed:"Seed",memoryCorpus:"Meeting-message RNN (mean of 3 seeds)",
  memoryValidation:"Validation loss",memoryTest:"Test loss",memoryParameters:"Parameters",
  memoryCaveat:"These are different training protocols, not a single model measured across fixed lengths. Two of three mixed-length seeds succeed at delay 16; the third fails. This is not a universal memory limit or architectural ranking. Delays beyond 16 were not tested.",
  memoryLabTitle:"Separate exercise: decay of a scalar state",
  memoryLabIntro:"A simplified calculation without training, interfering filler inputs or a classifier: h(t+1)=ρh(t). Vary the initial state, retention factor and number of steps.",
  memoryInitial:"Initial state",memoryRetention:"Retention factor ρ",memorySteps:"Steps",
  memoryFinal:"Final state",memoryMagnitude:"Absolute state magnitude",
  memoryTrace:"State trace",memoryStep:"Step",memoryState:"State",
  memoryLabLimit:"State magnitude is not a recall probability or RNN accuracy. Even a small nonzero state can retain its sign in ideal arithmetic. This recurrence is not the trained network in the frozen experiment.",
  embeddingTitle:"Chapter 5 — learned representations and frozen-embedding ablation",
  embeddingIntro:"Original frozen-source results: an MLP with trainable embeddings versus the same setup with frozen random embeddings. No browser training run.",
  embeddingModel:"Model",embeddingValidation:"Validation loss",embeddingTest:"Test loss",
  embeddingTrainable:"MLP — trainable embeddings",embeddingFrozen:"MLP — frozen random embeddings",
  embeddingTrigram:"Count Trigram (Chapter 4 reference)",
  embeddingTrigramSource:"Separate Trigram source from Chapter 4:",
  embeddingTrainableParams:"Trainable parameters",
  embeddingParams:"Parameter accounting: total / embedding / hidden / output",
  embeddingContexts:"Validation contexts",embeddingSeen:"Seen in training",embeddingUnseen:"Unseen in training",
  embeddingCount:"Positions",embeddingLoss:"Mean loss",
  embeddingCurve:"Learned-embedding MLP training loss curve",
  embeddingEpoch:"Epoch",embeddingTrainLoss:"Training loss",
  embeddingCaveat:"On this specific dataset, Count Trigram had lower validation/test loss than the MLP. The trainable–frozen difference does not prove linguistic meaning in embedding dimensions, and these test sets are not an independent generalization evaluation.",
  embeddingLabTitle:"Try one trainable versus one frozen parameter",
  embeddingLabIntro:"An independent scalar quadratic-loss exercise: a parameter changes through gradient updates or remains fixed. This is not the chapter's MLP or its embedding training.",
  embeddingInitial:"Initial parameter",embeddingTarget:"Target value",
  embeddingRate:"Learning rate",embeddingSteps:"Update steps",
  embeddingValue:"Parameter value",embeddingSingleLoss:"Example loss",
  embeddingInvalid:"Enter finite numbers between −2 and 2.",
  embeddingDemoLimit:"This is the loss of one toy example. It is not comparable with frozen validation/test corpus losses and establishes no generalization result.",
  contextEvidenceTitle:"Chapter 4 — context length and data sparsity",
  contextEvidenceIntro:"Frozen source results for validation/test losses, unseen contexts, and theoretical dense-table size versus actually stored context–character records.",
  contextOrder:"Context order (k)",contextValidation:"Validation loss",
  contextTest:"Test loss",contextUnseen:"Unseen validation contexts",
  contextDense:"Dense-table cells (theoretical)",contextSparse:"Stored context–character records",
  contextAlpha:"Smoothing selected on validation",
  contextCaveat:"These small reference test sets have been seen before; this is not independent generalization proof. Backoff numbers are excluded because their scores are unnormalized and cannot be directly compared as probabilities.",
  contextLabTitle:"Explore theoretical context-table growth",
  contextLabIntro:"Change context length and vocabulary sizes. The output is only a theoretical cell count: no memory allocation or model run. Actual stored records in the frozen table above remain unchanged.",
  contextVocabulary:"Context vocabulary (includes start symbol)",
  contextOutputVocabulary:"Output vocabulary",contextCells:"Theoretical cells",
  contextInvalid:"Invalid values: each vocabulary size must be 2–64 and order must be 0–3.",
  contextFormula:"Formula: V_context^k × V_output. Cell count is not a direct memory-use or runtime measurement.",
  bigramEvidenceTitle:"Chapter 3 — frozen model metrics",
  bigramEvidenceIntro:"Validation and test losses are recorded in the frozen source. Models use the chapter's same evaluation data and representation; these are not new run results.",
  bigramModel:"Model",bigramValidation:"Validation loss",bigramTest:"Test loss",
  bigramUnigram:"Unigram",bigramCount:"Count Bigram",bigramNeural:"Neural Bigram",
  bigramChosen:"Smoothing selected using validation:",
  bigramLabTitle:"Calculate one smoothed transition",
  bigramLabIntro:"Vary the bounded counts below. This is a new calculation using the smoothing equation, not retraining or rerunning Chapter 3's test.",
  bigramTotal:"Context observations (n)",bigramObserved:"Observed next-character count (c)",
  bigramVocabulary:"Output vocabulary size (V)",bigramAlpha:"Smoothing strength (α)",
  bigramOutput:"Transition probability and loss",bigramProbability:"Transition probability",
  bigramLoss:"Single-transition loss (nats)",bigramInvalid:"Enter in-range integers with 0 ≤ c ≤ n.",
  bigramReference:"Frozen source example: n=1, c=1, V=6, α=1; recorded probability:",
  bigramCaveat:"A single-transition loss (−ln p) cannot be compared directly to the mean validation or test losses in the table. This calculation makes no generalization claim.",
  tokensTitle:"Chapter 2 — frozen text-unit comparison",
  tokensDescription:"Two reference sentences with different wording; changing the segmentation method changes the unit count. Original outputs are displayed without rerunning the tokenizer.",
  tokensMethod:"Segmentation method",tokensCount:"Number of units",
  tokensWhitespace:"Whitespace-separated units (punctuation retained)",
  tokensEducational:"Pinned educational tokenizer",
  tokensLimit:"These sentences are not exact translations. The counts do not show that one language costs more, nor do they measure a production model.",
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
 // A separate local, bounded calculation accompanies frozen Chapter 3 metrics.
 // Neither the learner's counts nor computed outputs modify the reference evidence.
 if(evidence?.schema==="wakeel-v022-frozen-bigram-metrics/v1" &&
    evidence.chapter_id===chapter.id && evidence.status==="approved-source-metrics-not-independent-review"){
  const details=el("details","evidence-replay bigram-evidence");
  details.open=expandedEvidenceChapter===chapter.id;
  details.append(el("summary","evidence-summary",d.bigramEvidenceTitle));
  details.append(el("p","evidence-note",d.bigramEvidenceIntro));
  const provenance=el("p","evidence-provenance",d.evidenceSource+" ");
  const code=el("code","",evidence.source_path+" · SHA-256: "+evidence.source_sha256);
  code.dir="ltr";provenance.append(code);details.append(provenance);
  const scroll=el("div","evidence-scroll"),table=el("table","evidence-table");
  table.append(el("caption","",d.bigramEvidenceTitle));
  const thead=el("thead"),header=el("tr");
  for(const label of [d.bigramModel,d.bigramValidation,d.bigramTest]){
   const th=el("th","",label);th.scope="col";header.append(th);
  }
  thead.append(header);table.append(thead);
  const rows=el("tbody"),reference=evidence.datasets?.[language];
  for(const [key,label] of [["unigram",d.bigramUnigram],
   ["count_bigram",d.bigramCount],["neural_bigram",d.bigramNeural]]){
   const values=reference?.[key];
   if(!values)continue;
   const row=el("tr"),head=el("th","",label);head.scope="row";row.append(head);
   for(const metric of ["validation_loss","test_loss"]){
    const cell=el("td","evidence-number",values[metric].toFixed(4));
    cell.dir="ltr";row.append(cell);
   }
   rows.append(row);
  }
  table.append(rows);scroll.append(table);details.append(scroll);
  details.append(el("p","evidence-note",
   d.bigramChosen+" "+reference?.count_bigram?.chosen_alpha));
  const lab=el("section","smoothing-lab");
  lab.append(el("h4","",d.bigramLabTitle),el("p","evidence-note",d.bigramLabIntro));
  const controls=el("div","smoothing-controls");
  const inputs={};
  const fields=[
   ["total",d.bigramTotal,"0","100","1"],
   ["observed",d.bigramObserved,"0","100","1"],
   ["vocabulary",d.bigramVocabulary,"2","32","1"]
  ];
  for(const [key,label,min,max,step] of fields){
   const field=el("label","smoothing-field");
   const caption=el("span","",label);
   const input=el("input");
   input.type="number";input.min=min;input.max=max;input.step=step;
   const defaults={total:evidence.mini_example.total_context_count,
    observed:evidence.mini_example.observed_next_count,
    vocabulary:evidence.mini_example.output_vocabulary};
   input.value=String(defaults[key]);
   field.append(caption,input);controls.append(field);inputs[key]=input;
  }
  const alphaField=el("label","smoothing-field");
  alphaField.append(el("span","",d.bigramAlpha));
  const alpha=el("select");
  for(const value of [0.1,0.5,1]){
   const option=el("option","",String(value));option.value=String(value);
   alpha.append(option);
  }
  alpha.value=String(evidence.mini_example.alpha);
  alphaField.append(alpha);controls.append(alphaField);
  lab.append(controls);
  const output=el("p","smoothing-output");output.setAttribute("role","status");
  output.setAttribute("aria-live","polite");
  const update=()=>{
   try{
    const values=fields.map(([key])=>{
     const input=inputs[key];
     if(input.value.trim()===""||!input.checkValidity())throw new RangeError();
     return Number(input.value);
    });
    const result=smoothedTransition(...values,Number(alpha.value));
    output.textContent=d.bigramProbability+": "+result.probability.toFixed(4)+
      " · "+d.bigramLoss+": "+result.loss.toFixed(4);
   }catch{
    output.textContent=d.bigramInvalid;
   }
  };
  for(const input of [...Object.values(inputs),alpha]){
   input.addEventListener("input",update);input.addEventListener("change",update);
  }
  update();
  lab.append(el("p","evidence-note",d.bigramReference+" "+
   evidence.mini_example.reported_probability.toFixed(3)));
  lab.append(output,el("p","evidence-note",d.bigramCaveat));
  details.append(lab);
  details.addEventListener("toggle",()=>{
   expandedEvidenceChapter=details.open?chapter.id:"";
  });
  area.append(details);
 }
 // Display only frozen, SHA-verified Chapter 2 reference counts, never raw tokens.
 if(evidence?.schema==="wakeel-v022-frozen-tokenization/v1" &&
    evidence.chapter_id===chapter.id && evidence.status==="approved-source-counts-not-independent-review"){
  const details=el("details","evidence-replay tokenization-evidence");
  details.open=expandedEvidenceChapter===chapter.id;
  details.append(el("summary","evidence-summary",d.tokensTitle));
  details.append(el("p","evidence-note",d.tokensDescription));
  const provenance=el("p","evidence-provenance",d.evidenceSource+" ");
  const origin=el("code","",evidence.source_path+" · SHA-256: "+evidence.source_sha256+
     " · "+evidence.tokenizer);
  origin.dir="ltr";provenance.append(origin);details.append(provenance);
  const scroll=el("div","evidence-scroll"),table=el("table","evidence-table");
  table.append(el("caption","",d.tokensTitle));
  const head=el("thead"),header=el("tr");
  for(const label of [d.tokensMethod,d.tokensCount]){
   const th=el("th","",label);th.scope="col";header.append(th);
  }
  head.append(header);table.append(head);
  const values=evidence.samples?.[language];
  const body=el("tbody");
  for(const [label,value] of [[d.tokensWhitespace,values?.whitespace_units],
    [d.tokensEducational,values?.educational_tokenizer_units]]){
   if(!Number.isSafeInteger(value))continue;
   const row=el("tr"),name=el("th","",label);
   name.scope="row";
   const cell=el("td","evidence-number",String(value));cell.dir="ltr";
   row.append(name,cell);body.append(row);
  }
  table.append(body);scroll.append(table);details.append(scroll);
  details.append(el("p","evidence-note",d.tokensLimit));
  details.addEventListener("toggle",()=>{
   expandedEvidenceChapter=details.open?chapter.id:"";
  });
  area.append(details);
 }
 // Build-time SHA verification keeps frozen Chapter 4 results read-only.
 // This panel excludes unnormalized Backoff scores, raw contexts and samples.
 if(evidence?.schema==="wakeel-v022-frozen-context-orders/v1" &&
    evidence.chapter_id===chapter.id &&
    evidence.status==="approved-source-metrics-not-independent-review"){
  const details=el("details","evidence-replay context-evidence");
  details.open=expandedEvidenceChapter===chapter.id;
  details.append(el("summary","evidence-summary",d.contextEvidenceTitle));
  details.append(el("p","evidence-note",d.contextEvidenceIntro));
  const provenance=el("p","evidence-provenance",d.evidenceSource+" ");
  const source=el("code","",evidence.source_path+" · SHA-256: "+evidence.source_sha256);
  source.dir="ltr";provenance.append(source);details.append(provenance);
  const dataset=evidence.datasets?.[language];
  const scroll=el("div","evidence-scroll"),table=el("table","evidence-table");
  table.append(el("caption","",d.contextEvidenceTitle));
  const head=el("thead"),rowHead=el("tr");
  for(const heading of [d.contextOrder,d.contextValidation,d.contextTest,
   d.contextUnseen,d.contextDense,d.contextSparse,d.contextAlpha]){
   const th=el("th","",heading);th.scope="col";rowHead.append(th);
  }
  head.append(rowHead);table.append(head);
  const body=el("tbody");
  for(const record of dataset?.orders||[]){
   const row=el("tr");
   const label=el("th","",String(record.k));label.scope="row";label.dir="ltr";
   row.append(label);
   for(const value of [
    record.validation_loss.toFixed(4),
    record.test_loss.toFixed(4),
    (record.unseen_validation_rate*100).toFixed(2)+"%",
    record.dense_cells.toLocaleString("en-US"),
    record.sparse_records.toLocaleString("en-US"),
    String(record.selected_alpha)
   ]){
    const cell=el("td","evidence-number",value);cell.dir="ltr";row.append(cell);
   }
   body.append(row);
  }
  table.append(body);scroll.append(table);details.append(scroll);
  details.append(el("p","evidence-note",d.contextCaveat));
  const lab=el("section","context-lab");
  lab.append(el("h4","",d.contextLabTitle),el("p","evidence-note",d.contextLabIntro));
  const controls=el("div","smoothing-controls");
  const inputs={};
  for(const [key,label,value] of [
   ["contextVocabulary",d.contextVocabulary,dataset?.context_vocabulary],
   ["outputVocabulary",d.contextOutputVocabulary,dataset?.output_vocabulary]
  ]){
   const field=el("label","smoothing-field");
   const input=el("input","context-"+key);
   input.type="number";input.min="2";input.max="64";input.step="1";
   input.value=String(value);
   field.append(el("span","",label),input);controls.append(field);inputs[key]=input;
  }
  const orderField=el("label","smoothing-field");
  orderField.append(el("span","",d.contextOrder));
  const select=el("select","context-order");
  for(const k of [0,1,2,3]){
   const option=el("option","",String(k));option.value=String(k);select.append(option);
  }
  select.value="2";orderField.append(select);controls.append(orderField);
  lab.append(controls);
  const output=el("p","smoothing-output context-output");
  output.setAttribute("role","status");output.setAttribute("aria-live","polite");
  const update=()=>{
   try{
    const values=[inputs.contextVocabulary,inputs.outputVocabulary].map(input=>{
      if(input.value.trim()===""||!input.checkValidity())throw new RangeError();
      return Number(input.value);
    });
    const count=denseContextTableCells(...values,Number(select.value));
    output.textContent=d.contextCells+": "+count.toLocaleString("en-US");
   }catch{
    output.textContent=d.contextInvalid;
   }
  };
  for(const input of [...Object.values(inputs),select]){
   input.addEventListener("input",update);input.addEventListener("change",update);
  }
  update();
  lab.append(output,el("p","evidence-note",d.contextFormula));
  details.append(lab);
  details.addEventListener("toggle",()=>{
   expandedEvidenceChapter=details.open?chapter.id:"";
  });
  area.append(details);
 }
 // Chapter 5 source-backed losses and a separate scalar training illustration.
 if(evidence?.schema==="wakeel-v022-frozen-embedding-ablation/v1" &&
    evidence.chapter_id===chapter.id &&
    evidence.status==="approved-source-metrics-not-independent-review"){
  const details=el("details","evidence-replay embedding-evidence");
  details.open=expandedEvidenceChapter===chapter.id;
  details.append(el("summary","evidence-summary",d.embeddingTitle));
  details.append(el("p","evidence-note",d.embeddingIntro));
  const provenance=el("p","evidence-provenance",d.evidenceSource+" ");
  const source=el("code","",evidence.source_path+" · SHA-256: "+evidence.source_sha256);
  source.dir="ltr";provenance.append(source);details.append(provenance);
  const current=evidence.datasets?.[language];
  const baselineSource=catalog.evidence_replays?.["CH-4"];
  const trigram=baselineSource?.schema==="wakeel-v022-frozen-context-orders/v1"?
    baselineSource.datasets?.[language]?.orders?.find(x=>x.k===2):null;
  if(trigram){
   const prior=el("p","evidence-provenance",d.embeddingTrigramSource+" ");
   const pinned=el("code","",baselineSource.source_path+" · SHA-256: "+baselineSource.source_sha256);
   pinned.dir="ltr";prior.append(pinned);details.append(prior);
  }
  const table=el("table","evidence-table");
  table.append(el("caption","",d.embeddingTitle));
  const header=el("tr");
  for(const heading of [d.embeddingModel,d.embeddingValidation,d.embeddingTest,
   d.embeddingTrainableParams]){
   const th=el("th","",heading);th.scope="col";header.append(th);
  }
  const thead=el("thead");thead.append(header);table.append(thead);
  const tbody=el("tbody");
  for(const [name,values,params] of [
   [d.embeddingTrainable,current.learned,current.parameters.total],
   [d.embeddingFrozen,current.frozen,current.parameters.trainable_frozen_embeddings],
   ...(trigram?[[d.embeddingTrigram,{
     validation_loss:trigram.validation_loss,test_loss:trigram.test_loss},null]]:[])
  ]){
   const row=el("tr"),head=el("th","",name);head.scope="row";row.append(head);
   for(const value of [values.validation_loss.toFixed(4),values.test_loss.toFixed(4),
    params===null?"—":params.toLocaleString("en-US")]){
    const td=el("td","evidence-number",value);td.dir="ltr";row.append(td);
   }
   tbody.append(row);
  }
  table.append(tbody);
  const scroll=el("div","evidence-scroll");scroll.append(table);details.append(scroll);
  const p=current.parameters;
  details.append(el("p","evidence-note",
   d.embeddingParams+": "+[p.total,p.embedding,p.hidden,p.output].join(" / ")));
  const splitTable=el("table","evidence-table");
  splitTable.append(el("caption","",d.embeddingContexts));
  const splitHead=el("tr");
  for(const heading of [d.embeddingContexts,d.embeddingCount,d.embeddingLoss]){
   const th=el("th","",heading);th.scope="col";splitHead.append(th);
  }
  const splitThead=el("thead");splitThead.append(splitHead);splitTable.append(splitThead);
  const splitBody=el("tbody");
  for(const [name,values] of [[d.embeddingSeen,current.validation_contexts.seen],
   [d.embeddingUnseen,current.validation_contexts.unseen]]){
   const row=el("tr"),head=el("th","",name);head.scope="row";row.append(head);
   for(const value of [String(values.count),values.loss.toFixed(4)]){
    const td=el("td","evidence-number",value);td.dir="ltr";row.append(td);
   }
   splitBody.append(row);
  }
  splitTable.append(splitBody);
  const splitScroll=el("div","evidence-scroll");splitScroll.append(splitTable);
  details.append(splitScroll);
  const curveTable=el("table","evidence-table");
  curveTable.append(el("caption","",d.embeddingCurve));
  const curveHead=el("tr");
  for(const heading of [d.embeddingEpoch,d.embeddingTrainLoss]){
   const th=el("th","",heading);th.scope="col";curveHead.append(th);
  }
  const curveThead=el("thead");curveThead.append(curveHead);curveTable.append(curveThead);
  const curveBody=el("tbody");
  for(const point of current.training_curve){
   const row=el("tr");
   for(const value of [String(point.epoch),point.loss.toFixed(4)]){
    const td=el("td","evidence-number",value);td.dir="ltr";row.append(td);
   }
   curveBody.append(row);
  }
  curveTable.append(curveBody);
  const curveScroll=el("div","evidence-scroll");curveScroll.append(curveTable);
  details.append(curveScroll,el("p","evidence-note",d.embeddingCaveat));
  const lab=el("section","embedding-lab");
  lab.append(el("h4","",d.embeddingLabTitle),el("p","evidence-note",d.embeddingLabIntro));
  const controls=el("div","smoothing-controls");
  const scalarInputs={};
  for(const [key,label,defaultValue] of [
   ["initial",d.embeddingInitial,"0"],["target",d.embeddingTarget,"1"]
  ]){
   const field=el("label","smoothing-field");
   const input=el("input","embedding-"+key);
   input.type="number";input.min="-2";input.max="2";input.step="0.1";
   input.value=defaultValue;
   field.append(el("span","",label),input);controls.append(field);scalarInputs[key]=input;
  }
  for(const [key,label,values,defaultValue] of [
   ["rate",d.embeddingRate,[0,0.25,0.5,1],"0.5"],
   ["steps",d.embeddingSteps,[0,1,2,5,10],"1"]
  ]){
   const field=el("label","smoothing-field"),select=el("select","embedding-"+key);
   field.append(el("span","",label));
   for(const value of values){
    const option=el("option","",String(value));option.value=String(value);select.append(option);
   }
   select.value=defaultValue;field.append(select);controls.append(field);
   scalarInputs[key]=select;
  }
  lab.append(controls);
  const output=el("p","smoothing-output embedding-output");
  output.setAttribute("role","status");output.setAttribute("aria-live","polite");
  const update=()=>{
   try{
    for(const input of [scalarInputs.initial,scalarInputs.target]){
     if(input.value.trim()===""||!input.checkValidity())throw new RangeError();
    }
    const x=toyScalarEmbeddingUpdate(
     Number(scalarInputs.initial.value),Number(scalarInputs.target.value),
     Number(scalarInputs.rate.value),Number(scalarInputs.steps.value));
    output.textContent=
     d.embeddingTrainable+": "+d.embeddingValue+" "+x.learned_value.toFixed(4)+
     " · "+d.embeddingSingleLoss+" "+x.learned_loss.toFixed(4)+
     " | "+d.embeddingFrozen+": "+d.embeddingValue+" "+x.frozen_value.toFixed(4)+
     " · "+d.embeddingSingleLoss+" "+x.frozen_loss.toFixed(4);
   }catch{
    output.textContent=d.embeddingInvalid;
   }
  };
  for(const input of Object.values(scalarInputs)){
   input.addEventListener("input",update);input.addEventListener("change",update);
  }
  update();lab.append(output,el("p","evidence-note",d.embeddingDemoLimit));
  details.append(lab);
  details.addEventListener("toggle",()=>{
   expandedEvidenceChapter=details.open?chapter.id:"";
  });
  area.append(details);
 }
 // Chapter 6: compare distinct frozen protocols, then isolate a scalar recurrence.
 if(evidence?.schema==="wakeel-v022-frozen-recurrent-memory/v1"&&
    evidence.chapter_id===chapter.id&&evidence.status==="approved-source-metrics-not-independent-review"){
  const details=el("details","evidence-replay memory-evidence");
  details.open=expandedEvidenceChapter===chapter.id;
  details.append(el("summary","evidence-summary",d.memoryTitle),
   el("p","evidence-note",d.memoryIntro),el("p","evidence-note",d.memoryProtocol));
  const provenance=el("p","evidence-provenance",d.evidenceSource+" ");
  const code=el("code","",evidence.source_path+" · SHA-256: "+evidence.source_sha256);
  code.dir="ltr";provenance.append(code);details.append(provenance);
  const tableFor=(caption,headings,parent,className="")=>{
   const scroll=el("div","evidence-scroll"),table=el("table","evidence-table "+className);
   table.append(el("caption","",caption));
   const head=el("thead"),row=el("tr");
   for(const label of headings){const th=el("th","",label);th.scope="col";row.append(th);}
   head.append(row);table.append(head);
   const body=el("tbody");table.append(body);scroll.append(table);parent.append(scroll);
   return body;
  };
  const addRow=(body,first,values)=>{
   const row=el("tr"),head=el("th","",String(first));head.scope="row";row.append(head);
   for(const value of values){const numeric=/^[0-9,%.−-]+$/.test(String(value));const td=el("td",numeric?"evidence-number":"",String(value));td.dir=numeric?"ltr":(language==="ar"?"rtl":"ltr");row.append(td);}
   body.append(row);
  };
  const percent=v=>(v*100).toFixed(1)+"%";
  const corpus=evidence.languages[language];
  const corpusTable=tableFor(d.memoryCorpus,[d.memoryParameters,d.memoryValidation,d.memoryTest],details,"memory-corpus");
  addRow(corpusTable,corpus.parameters.toLocaleString("en-US"),
   [corpus.validation_loss.toFixed(4),corpus.test_loss.toFixed(4)]);
  const scores=tableFor(d.memoryAccuracy,
   [d.memoryDelay,d.memoryFixed,d.memoryMixed,d.memoryDomain],details,"memory-accuracy");
  for(const row of evidence.rows){
   addRow(scores,row.delay,[percent(row.fixed_accuracy),percent(row.mixed_accuracy),
    row.mixed_in_domain?d.memoryInside:d.memoryOutside]);
  }
  const selectField=(parent,cls,label,values,initial)=>{
   const field=el("label","smoothing-field"),select=el("select",cls);
   field.append(el("span","",label));
   for(const value of values){const option=el("option","",String(value));option.value=String(value);select.append(option);}
   select.value=String(initial);field.append(select);parent.append(field);return select;
  };
  const drill=el("section","memory-drill");
  drill.append(el("h4","",d.memorySeeds));
  const delaySelect=selectField(drill,"memory-delay",d.memoryDelay,evidence.rows.map(x=>x.delay),16);
  const seedBody=tableFor(d.memorySeeds,[d.memorySeed,d.memoryFixed+" · "+d.memorySeedAccuracy,
   d.memoryMixed+" · "+d.memorySeedAccuracy,d.memoryFixed+" · "+d.memoryNll,
   d.memoryMixed+" · "+d.memoryNll],drill,"memory-seeds");
  const stats=el("p","evidence-note memory-stats");stats.setAttribute("role","status");
  const updateDetail=()=>{
   const row=evidence.rows.find(x=>x.delay===Number(delaySelect.value));
   if(!row)return;
   seedBody.replaceChildren();
   for(const seed of row.seeds)addRow(seedBody,seed.seed,[percent(seed.fixed_accuracy),
    percent(seed.mixed_accuracy),seed.fixed_nll.toFixed(4),seed.mixed_nll.toFixed(4)]);
   stats.textContent=(row.mixed_in_domain?d.memoryInside:d.memoryOutside)+" · "+
    d.memoryNll+": "+d.memoryFixed+" "+row.fixed_nll.toFixed(4)+" / "+
    d.memoryMixed+" "+row.mixed_nll.toFixed(4)+" · "+
    d.memoryUntrained+" "+percent(row.untrained_accuracy)+" · "+d.memoryOracle+" "+percent(row.oracle_accuracy);
  };
  delaySelect.addEventListener("change",updateDetail);updateDetail();
  drill.append(stats);details.append(drill,el("p","evidence-note",d.memoryCaveat));
  const lab=el("section","memory-lab");
  lab.append(el("h4","",d.memoryLabTitle),el("p","evidence-note",d.memoryLabIntro));
  const controls=el("div","smoothing-controls");
  const initial=selectField(controls,"memory-initial",d.memoryInitial,[-1,1],1);
  const retention=selectField(controls,"memory-retention",d.memoryRetention,[0,0.5,0.9,1],0.9);
  const steps=selectField(controls,"memory-steps",d.memorySteps,[0,2,4,5,6,7,8,16],4);
  lab.append(controls);
  const output=el("p","smoothing-output memory-output");output.setAttribute("role","status");
  lab.append(output);
  const trace=tableFor(d.memoryTrace,[d.memoryStep,d.memoryState],lab,"memory-trace");
  const updateLab=()=>{
   const result=toyMemoryRetention(Number(initial.value),Number(retention.value),Number(steps.value));
   output.textContent=d.memoryFinal+": "+result.final_state.toFixed(6)+" · "+d.memoryMagnitude+": "+result.magnitude.toFixed(6);
   trace.replaceChildren();
   result.states.forEach((v,i)=>addRow(trace,i,[v.toFixed(6)]));
  };
  for(const select of [initial,retention,steps])select.addEventListener("change",updateLab);
  updateLab();lab.append(el("p","evidence-note",d.memoryLabLimit));details.append(lab);
  details.addEventListener("toggle",()=>{expandedEvidenceChapter=details.open?chapter.id:"";});
  area.append(details);
 }
 // Chapter 7 keeps holdout, exploratory validation, and toy arithmetic distinct.
 if(evidence?.schema==="wakeel-v022-frozen-attention-order/v1"&&
    evidence.chapter_id===chapter.id&&evidence.status==="approved-source-metrics-not-independent-review"){
  const details=el("details","evidence-replay attention-evidence");
  details.open=expandedEvidenceChapter===chapter.id;
  details.append(el("summary","evidence-summary",d.attentionTitle),el("p","evidence-note",d.attentionIntro));
  const provenance=el("p","evidence-provenance",d.evidenceSource+" ");
  const source=el("code","",evidence.source_path+" · SHA-256: "+evidence.source_sha256);
  source.dir="ltr";provenance.append(source);details.append(provenance);
  const tableFor=(parent,caption,labels,cls)=>{
   const scroll=el("div","evidence-scroll"),table=el("table","evidence-table "+cls);
   table.append(el("caption","",caption));
   const head=el("thead"),row=el("tr");
   for(const label of labels){const th=el("th","",label);th.scope="col";row.append(th);}
   head.append(row);table.append(head);
   const body=el("tbody");table.append(body);scroll.append(table);parent.append(scroll);return body;
  };
  const addRow=(body,label,values)=>{
   const row=el("tr"),th=el("th","",String(label));th.scope="row";row.append(th);
   for(const v of values){const td=el("td","evidence-number",String(v));td.dir="ltr";row.append(td);}
   body.append(row);
  };
  const pct=x=>(x*100).toFixed(1)+"%";
  const a=tableFor(details,d.attentionRetrieval,[d.attentionDelay,"RNN · "+d.attentionAccuracy,
   "RNN · "+d.attentionNll,"attn_content · "+d.attentionAccuracy,
   "attn_content · "+d.attentionNll,d.attentionOracle],"attention-retrieval");
  for(const row of evidence.retrieval)addRow(a,row.delay,[pct(row.rnn.accuracy),row.rnn.nll.toFixed(4),
   pct(row.attn_content.accuracy),row.attn_content.nll.toFixed(4),pct(row.oracle_accuracy)]);
  details.append(el("p","evidence-note",d.attentionRetrievalLimit));
  const selectField=(parent,cls,label,choices,value)=>{
   const field=el("label","smoothing-field"),select=el("select",cls);field.append(el("span","",label));
   for(const [v,label] of choices){const option=el("option","",String(label));option.value=String(v);select.append(option);}
   select.value=String(value);field.append(select);parent.append(field);return select;
  };
  const drill=el("section","attention-drill");drill.append(el("h4","",d.attentionOrder));
  const delay=selectField(drill,"attention-delay",d.attentionDelay,[4,8,16].map(x=>[x,x]),4);
  const order=tableFor(drill,d.attentionOrder,[d.attentionModel,d.attentionParams,d.attentionStandard,
   d.attentionShifted,d.attentionNll],"attention-order");
  const updateOrder=()=>{
   const row=evidence.order.find(x=>x.delay===Number(delay.value));order.replaceChildren();
   for(const [name,m] of Object.entries(row.models))addRow(order,name,[m.parameters,pct(m.accuracy),pct(m.shifted_accuracy),m.nll.toFixed(4)]);
  };
  delay.addEventListener("change",updateOrder);updateOrder();
  drill.append(el("p","evidence-note",d.attentionOrderLimit));details.append(drill);
  const validation=tableFor(details,d.attentionSensitivity,[d.attentionAmp,4,8,16],"attention-validation");
  for(const [key,label] of [["amp1",d.attentionOriginal],["all3",d.attentionAll],
   ["kq_only3",d.attentionKQ],["v_only3",d.attentionV],["pos_only3",d.attentionP]])
   addRow(validation,label,evidence.validation_sensitivity.map(row=>pct(row.variants[key])));
  details.append(el("p","evidence-note",d.attentionSensitivityLimit));
  const lab=el("section","attention-lab");lab.append(el("h4","",d.attentionLab),el("p","evidence-note",d.attentionLabIntro));
  const controls=el("div","smoothing-controls"),choices=[0,1,2,4].map(x=>[x,x]);
  const scale=selectField(controls,"attention-scale",d.attentionScale,choices,1);
  const bias=selectField(controls,"attention-bias",d.attentionBias,choices,0);
  const index=selectField(controls,"attention-index",d.attentionIndex,[0,1,2,3].map(x=>[x,x]),0);
  const shift=selectField(controls,"attention-shift",d.attentionOrder,[[0,d.attentionStandard],[1,d.attentionShifted]],0);
  lab.append(controls);
  const output=el("p","smoothing-output attention-output");output.setAttribute("role","status");lab.append(output);
  const trace=tableFor(lab,d.attentionLab,[d.attentionPosition,d.attentionPairA+" · "+d.attentionValue,
   d.attentionPairA+" · "+d.attentionWeight,d.attentionPairB+" · "+d.attentionValue,
   d.attentionPairB+" · "+d.attentionWeight],"attention-trace");
  const update=()=>{
   const p=toyAttentionOrder(Number(scale.value),Number(bias.value),Number(index.value),shift.value==="1");
   output.textContent=d.attentionSum+": "+p.first.weighted_sum.toFixed(4)+" / "+p.second.weighted_sum.toFixed(4)+
    " · "+d.attentionOracle+": "+p.first.oracle_value+" / "+p.second.oracle_value;
   trace.replaceChildren();
   p.first.values.forEach((v,i)=>addRow(trace,i,[v,p.first.weights[i].toFixed(4),p.second.values[i],p.second.weights[i].toFixed(4)]));
  };
  for(const field of [scale,bias,index,shift])field.addEventListener("change",update);
  update();lab.append(el("p","evidence-note",d.attentionLabLimit));details.append(lab);
  details.addEventListener("toggle",()=>{expandedEvidenceChapter=details.open?chapter.id:"";});
  area.append(details);
 }
 // Chapter 8: main and independent confirmation data stay visibly separate.
 if(evidence?.schema==="wakeel-v022-frozen-causal-dual-lookup/v1"&&
    evidence.chapter_id===chapter.id&&evidence.status==="approved-source-metrics-not-independent-review"){
  const details=el("details","evidence-replay causal-evidence");
  details.open=expandedEvidenceChapter===chapter.id;
  details.append(el("summary","evidence-summary",d.causalTitle),el("p","evidence-note",d.causalIntro));
  const provenance=(path,hash)=>{
   const p=el("p","evidence-provenance",d.evidenceSource+" ");
   const code=el("code","",path+" · SHA-256: "+hash);code.dir="ltr";p.append(code);details.append(p);
  };
  const tableFor=(parent,caption,labels,cls)=>{
   const scroll=el("div","evidence-scroll"),table=el("table","evidence-table "+cls);
   table.append(el("caption","",caption));const head=el("thead"),row=el("tr");
   for(const label of labels){const th=el("th","",label);th.scope="col";row.append(th);}
   head.append(row);table.append(head);const body=el("tbody");table.append(body);scroll.append(table);parent.append(scroll);return body;
  };
  const addRow=(body,label,values)=>{
   const row=el("tr"),th=el("th","",String(label));th.scope="row";row.append(th);
   for(const v of values){const td=el("td","evidence-number",String(v));td.dir="ltr";row.append(td);}body.append(row);
  };
  const selectField=(parent,cls,label,choices,value)=>{
   const field=el("label","smoothing-field"),select=el("select",cls);field.append(el("span","",label));
   for(const [v,label] of choices){const option=el("option","",String(label));option.value=String(v);select.append(option);}
   select.value=String(value);field.append(select);parent.append(field);return select;
  };
  const headings=[d.causalModel,d.causalParams,d.causalBinary,d.causalFull,d.causalNll,d.causalBinNll];
  const pct=x=>(x*100).toFixed(2)+"%";
  const render=(body,records,split)=>{
   body.replaceChildren();
   for(const record of records){
    const v=record[split];
    addRow(body,record.model,[record.params,pct(v.binary_accuracy.mean),
     pct(v.full_vocab_accuracy.mean),v.nll_full_vocab.mean.toFixed(4),v.nll_binary_normalized.mean.toFixed(4)]);
   }
  };
  provenance(evidence.source_path,evidence.source_sha256);
  const main=el("section","causal-drill");main.append(el("h4","",d.causalMain));
  const split=selectField(main,"causal-split",d.causalSplit,[["validation",d.causalVal],["development",d.causalDev],
   ["test_in",d.causalIn],["test_ood",d.causalOod]],"test_in");
  const mainBody=tableFor(main,d.causalMain,headings,"causal-main");
  const updateMain=()=>render(mainBody,evidence.main,split.value);
  split.addEventListener("change",updateMain);updateMain();main.append(el("p","evidence-note",d.causalLimit));details.append(main);
  provenance(evidence.confirmation_source_path,evidence.confirmation_source_sha256);
  const confirm=el("section","causal-drill");confirm.append(el("h4","",d.causalConfirm));
  const cs=selectField(confirm,"causal-confirm-split",d.causalSplit,[["confirm_in",d.causalIn],["confirm_ood",d.causalOod]],"confirm_in");
  const cb=tableFor(confirm,d.causalConfirm,headings,"causal-confirm");
  const updateConfirm=()=>render(cb,evidence.confirmation,cs.value);
  cs.addEventListener("change",updateConfirm);updateConfirm();confirm.append(el("p","evidence-note",d.causalConfirmLimit));details.append(confirm);
  const lab=el("section","causal-lab");lab.append(el("h4","",d.causalLab),el("p","evidence-note",d.causalLabIntro));
  const controls=el("div","smoothing-controls"),choices=[-2,-1,0,1,2].map(x=>[x,x]);
  const f1=selectField(controls,"causal-future1",d.causalFuture1,choices,2);
  const f2=selectField(controls,"causal-future2",d.causalFuture2,choices,-2);
  const mask=selectField(controls,"causal-mask",d.causalMask,[[1,d.causalOn],[0,d.causalOff]],1);
  lab.append(controls);const output=el("p","smoothing-output causal-output");output.setAttribute("role","status");lab.append(output);
  const trace=tableFor(lab,d.causalLab,[d.causalPosition,d.causalValue,d.causalWeight],"causal-trace");
  const update=()=>{
   const v=toyCausalPrefix(Number(f1.value),Number(f2.value),mask.value==="1");
   output.textContent=d.causalOutput+": "+v.output.toFixed(6)+" · "+d.causalPrefix+": "+v.prefix_output.toFixed(6)+
    " · "+d.causalFutureWeight+": "+v.future_weight.toFixed(6);
   trace.replaceChildren();v.values.forEach((value,i)=>addRow(trace,i,[value,v.weights[i].toFixed(6)]));
  };
  for(const control of [f1,f2,mask])control.addEventListener("change",update);
  update();lab.append(el("p","evidence-note",d.causalLabLimit));details.append(lab);
  details.addEventListener("toggle",()=>{expandedEvidenceChapter=details.open?chapter.id:"";});
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