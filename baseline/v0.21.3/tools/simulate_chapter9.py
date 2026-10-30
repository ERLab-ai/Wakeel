#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
simulate_chapter9.py — الفصل التاسع P1: الارتباط التركيبي بين المفتاح والقيمة

ينفّذ مرحلة P1 من عقد WAKEEL-2 CH9 (CH9-EXPERIMENT-CONTRACT، بوابة الانتقال §8):
تجميد تجارب التدريب وعزل التبعية، ثم تنفيذها وإظهار جميع النتائج ولو كانت سلبية.
تأليف النصين العربي/الإنجليزي يأتي بعد هذه الجولة (بوابة 11 مؤجلة صراحة).

المهمة (من P0): سجلان بنيويان `KEY VALUE SEP` لكل من A/B (لا حقيقة ذرّية)،
حشو مستقل، ثم سؤالان Q_A/Q_B مع إجابتيهما داخل الحلقة نفسها؛ القيمتان
مستقلتان (00/01/10/11 متوازنة). السؤال الأول هو مقياس الربط الأساسي: سقف
Bag-of-Tokens المحسوب فيه 0.750 وأزواج 01/10 المتطابقة الـBag سقفها المزدوج 0.

البروتوكول المُجمَّد قبل أي نتيجة (لا ضبط ولا اختيار بعد القراءة):
* البيانات: مقاطع P0 الخمسة ببذرة 9102 مطابقةً حرفياً لمخرجات P0 المرجعية
  (golden_ch9_p0.json)، ثم مجموعتان محجوزتان جديدتان مستقلتان عن عائلات P0:
  تأكيدية ببذرة 9204 (confirm_in/confirm_ood16) واختبار نهائي ببذرة 9203
  (final_in/final_ood16) — لا يُحسب عليها أي مقياس نموذجي ولا فحص تسميات قبل
  التجميد؛ الفحوص البنيوية قبل التجميد تقتصر على مقاطع غير المحجوز، وحدود
  الوصول المبكر موثقة نصاً في النتائج (quarantine).
* النماذج المتعلمة السبعة: local_only (يُقاس ولا يُفترض فشله)،
  fixed_query_prefix_reader (منفذ الإخراج: ملخص بادئة باستعلام ثابت ‖ رمز
  السؤال الحالي)، causal_self_attention_1h (جيبي الموضع، residual)،
  causal_self_attention_2h (طبقتا انتباه سببي متراكبتان)، no_position
  (بلا ترميز موضعي — انتباه أحادي الطبقة دالةً في تعداد الرموز لا ترتيبها)،
  no_position_2h (بلا ترميز موضعي بطبقتين — القناع السببي المتراكم قد يحمل
  معلومة ترتيب؛ يُقاس)، csa1h_concat (تبديل سلك الإخراج concat — تعداد
  معلمات مختلف معلن، لا ادعاء عاملياً).
* الميزانية: Adam، 1200 عصراً (نقطة بدء معلنة لا نتيجة مضمونة)، تحديث واحد
  لكل عصر، تجميع موزون عبر كتل الأطوال L = Σ nᵢ·Lᵢ/Σ nᵢ وتدرجها الموزون
  (إرث تصحيح v1.1 من CH-8 وبوابة 5b). مسبح معدل التعلم {0.003, 0.01, 0.03}
  يُجمَّد على التحقق وحده قبل أي قراءة dev/تأكيدي/نهائي، بقاعدة متشائماتية
  معلنة: مرشح كل lr هو أسوأ مرساةٍ له (min)، الأعلى يفوز، التعادل → الأصغر؛
  المرسيات الثلاث: fixed_reader وcsa_1h وcsa_2h؛ مقياس الاختيار دقة السؤال
  الأول الثنائية على val. البذور 42/43/44 لكل نموذج.
* البوابات 1–10 من §7 قبل قبول أي نتيجة تدريب (و11 مؤجلة للتأليف، و12
  بنيوية: لا مسّ لملفات الرحلة/السجل/الأطلس — الكتابة في content/chapter-09 فقط).

تصحيحات v1.1 (بعد التدقيق العلمي والبرمجي المستقل لـ v0.18 — قرار GO مشروط):
وصف القارئ الثابت صُحح (ليس محايد الترتيب بنيوياً)، ترتيب قراءة المحجوز صار
حرفياً عالمياً (كل التأكيدية ثم كل النهائية)، عدّ التدريبات موثق (30/27)،
وأُضيفت قراءة عاملية وصفية من المتوسطات المسجلة فقط. البروتوكول العلمي لم
يتغير ولا رقم واحد من نتائج v1.0 المؤرشفة.

لا افتراض تفوق لأي معمارية ولا افتراض فشل لأي ضابط؛ كل رقم مقيس هنا
ويُكتب في results.json، ولا يُعلن شيء لم يُحسب.
"""
import hashlib
import json
import os
import platform
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_OUTDIR = Path(os.environ.get("WAKEEL_CH9_OUT_DIR",
                              str(ROOT / "content" / "chapter-09")))
OUT = _OUTDIR / "results.json"
RUN_LOG = _OUTDIR / "runs.log"
VERSION = "simulate_chapter9-1.1"

# ─────────────────────────── بروتوكول معلن قبل أي نتيجة ───────────────────────────
K_A, K_B, V_0, V_1, SEP, Q_A, Q_B, ANS_0, ANS_1 = range(9)
FILLER_START, N_FILLER = 9, 24
V = 9 + N_FILLER                                   # 33 رمزاً
KEYS = {"A": K_A, "B": K_B}
QUERIES = {"A": Q_A, "B": Q_B}
ANS_IDS = (ANS_0, ANS_1)
QUERY_IDS = (Q_A, Q_B)

D_MODEL, DK = 8, 8
PE_LEN = 26                                        # أطول حلقة: 6 + 16 + 4 = 26
P0_SEED = 9102
CONFIRM_SEED = 9204                                # مجموعة تأكيدية مستقلة (قبل النهائي)
FINAL_SEED = 9203                                  # الاختبار النهائي المحجوز
MODEL_SEEDS = (42, 43, 44)
EPOCHS = 1200
CURVE_AT = (1, 100, 300, 600, 900, 1200)
LR_GRID = (0.003, 0.01, 0.03)
SWEEP_SEED = 42
SWEEP_ANCHORS = ("fixed_query_prefix_reader",
                 "causal_self_attention_1h",
                 "causal_self_attention_2h")
FD_EPS = 1e-5
GATE_GRAD_TOL = 1e-6
GATE_LOGIT_TOL = 1e-9
GATE_ZERO_TOL = 1e-12

# مقاطع P0 (بذرة 9102 — بنفس ترتيب توليد P0 حرفياً لضمان التطابق المرجعي)
P0_LENGTHS = {
    "train": {2: 160, 4: 176, 8: 176},
    "val": {2: 32, 4: 48, 8: 48},
    "dev_holdout": {2: 32, 4: 48, 8: 48},
    "p0_probe_in": {2: 16, 4: 16, 8: 32},
    "p0_probe_ood_length16": {16: 64},
}
# المجموعات المحجوزة الجديدة (عائلات منفصلة عن P0 بالكامل)
CONFIRM_LENGTHS = {"confirm_in": {2: 16, 4: 16, 8: 32},
                   "confirm_ood16": {16: 64}}
FINAL_LENGTHS = {"final_in": {2: 16, 4: 16, 8: 32},
                 "final_ood16": {16: 64}}

SEGMENTS = tuple(P0_LENGTHS) + tuple(CONFIRM_LENGTHS) + tuple(FINAL_LENGTHS)
NON_BLIND = tuple(P0_LENGTHS)                       # train/val/dev/probes (probes اطّلع عليها P0)
BLIND = tuple(CONFIRM_LENGTHS) + tuple(FINAL_LENGTHS)
P0_GOLDEN = ROOT / "_ch9_p0" / "golden_ch9_p0.json"

STRUCT_CLASSES = tuple((va, vb, ro, qo) for va in (0, 1) for vb in (0, 1)
                       for ro in (0, 1) for qo in (0, 1))
INTERVENTION_MODES = ("memory_only", "prior_answer_only", "both")


def sinusoidal_pe(max_len, d):
    pe = np.zeros((max_len, d))
    pos = np.arange(max_len)[:, None]
    div = np.exp(np.arange(0, d, 2) * (-np.log(10000.0) / d))
    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div)
    return pe


PE_TABLE = sinusoidal_pe(PE_LEN, D_MODEL)


# ═══════════════════════ شبكة انتشار عكسي مصغّرة (float64 — إرث CH-8) ═══════════════════════
class Node:
    __slots__ = ("v", "parents", "bw")

    def __init__(self, v, parents=(), bw=None):
        self.v = v
        self.parents = parents
        self.bw = bw


def unbroadcast(g, shape):
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g


def backward(root):
    grads = {}
    topo, seen, stack = [], set(), [(root, False)]
    while stack:
        node, done = stack.pop()
        if done:
            topo.append(node)
            continue
        if id(node) in seen:
            continue
        seen.add(id(node))
        stack.append((node, True))
        for p in node.parents:
            if id(p) not in seen:
                stack.append((p, False))
    grads[id(root)] = np.ones_like(root.v)
    for node in reversed(topo):
        g = grads.get(id(node))
        if g is None or not node.bw:
            continue
        grads.pop(id(node))
        for pi, f in node.bw:
            gg = unbroadcast(f(g), node.parents[pi].v.shape)
            k = id(node.parents[pi])
            grads[k] = gg if k not in grads else grads[k] + gg
    return grads


def leaf(v):
    return Node(np.asarray(v, dtype=np.float64))


def add(a, b):
    return Node(a.v + b.v, (a, b), [(0, lambda g: g), (1, lambda g: g)])


def scale(a, s):
    return Node(a.v * s, (a,), [(0, lambda g: g * s)])


def mm2v(a, b):
    """(N,T,d)@(d,) → (N,T) — درجات الاستعلام الثابت."""
    A, q = a.v, b.v
    return Node(A @ q, (a, b),
                [(0, lambda g: np.einsum("nt,d->ntd", g, q)),
                 (1, lambda g: np.einsum("nt,ntd->d", g, A))])


def mm3_2(a, b):
    """(N,T,d)@(d,m) → (N,T,m)"""
    A = a.v
    return Node(A @ b.v, (a, b),
                [(0, lambda g: g @ b.v.T),
                 (1, lambda g: A.reshape(-1, A.shape[-1]).T @ g.reshape(-1, g.shape[-1]))])


def einsum_nts(a, b):
    """(N,T,dk)×(N,S,dk) → (N,T,S)"""
    A, B = a.v, b.v
    return Node(np.einsum("ntd,nsd->nts", A, B), (a, b),
                [(0, lambda g: np.einsum("nts,nsd->ntd", g, B)),
                 (1, lambda g: np.einsum("nts,ntd->nsd", g, A))])


def einsum_av(a, b):
    """(N,T,S)×(N,S,d) → (N,T,d)"""
    A, B = a.v, b.v
    return Node(np.einsum("nts,nsd->ntd", A, B), (a, b),
                [(0, lambda g: np.einsum("ntd,nsd->nts", g, B)),
                 (1, lambda g: np.einsum("nts,ntd->nsd", A, g))])


def expand_t(a, n, t, s):
    """(N,S) → (N,T,S) بثّاً على محور الاستعلام."""
    return Node(np.broadcast_to(a.v[:, None, :], (n, t, s)),
                (a,), [(0, lambda g: g.sum(axis=1))])


def emb(tablenode, ids):
    T = tablenode.v

    def f(g):
        gt = np.zeros_like(T)
        np.add.at(gt, ids.ravel(), g.reshape(-1, T.shape[1]))
        return gt
    return Node(T[ids], (tablenode,), [(0, f)])


def causal_mask(T):
    m = np.zeros((T, T))
    m[np.triu_indices(T, 1)] = -np.inf
    return m


_MASK_CACHE = {}


def get_mask(T):
    if T not in _MASK_CACHE:
        _MASK_CACHE[T] = causal_mask(T)
    return _MASK_CACHE[T]


def mask_softmax(x, T):
    """softmax آخر محور بعد إخفاء محكم (−inf) — وزن المستقبل = 0 بالآلة."""
    z = x.v + get_mask(T)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(axis=-1, keepdims=True)

    def f(g):
        return p * (g - np.sum(g * p, axis=-1, keepdims=True))
    return Node(p, (x,), [(0, f)])


def concat1(a, b):
    n = a.v.shape[-1]
    return Node(np.concatenate([a.v, b.v], axis=-1), (a, b),
                [(0, lambda g: g[..., :n]), (1, lambda g: g[..., n:])])


def nll_pick(logits, rows, flat_pos, targets):
    """−mean log P[target] على مواضع الإشراف فقط."""
    L = logits.v
    logp = L - L.max(axis=-1, keepdims=True)
    logp = logp - np.log(np.exp(logp).sum(axis=-1, keepdims=True))
    M = len(flat_pos)
    picked = logp[rows, flat_pos, targets]
    loss = float(-picked.mean())

    def f(g):
        d = np.zeros_like(L)
        sel = L[rows, flat_pos]
        sm = np.exp(sel - sel.max(axis=-1, keepdims=True))
        sm = sm / sm.sum(axis=-1, keepdims=True)
        d[rows, flat_pos] = sm
        d[rows, flat_pos, targets] -= 1.0
        return d * (g.item() / M)
    return Node(np.asarray(loss), (logits,), [(0, f)])


# ═══════════════════════════════ النماذج الموحّدة ═══════════════════════════════
# مواصفات معلنة قبل النتائج — تعداد المعلمات يُقاس ويُنشر لكل نموذج (H5):
SPECS = {
    "local_only": {
        "arch": "local",
        "desc": "ضابط غياب الذاكرة: logits من تضمين رمز السؤال الحالي فقط — "
                "يُقاس أداؤه ولا يُفترض فشله العددي"},
    "fixed_query_prefix_reader": {
        "arch": "attention", "layers": 1, "pos_mode": "sin",
        "query_mode": "fixed", "out_mode": "concat",
        "desc": "قارئ CH-8 بإنتاجية المفردة الجديدة: استعلام متعلم ثابت لكل "
                "المواضع؛ منفذ المعلومات الذي يراه رأس الإخراج = concat(ملخص "
                "البادئة، تضمين رمز السؤال الحالي+الموضع). ليس محايد الترتيب "
                "بنيوياً (تصحيح v1.1 — التدقيق المستقل): المفاتيح والقيم تحمل "
                "الترميز الموضعي فدرجات الانتباه تعتمد على المواضع؛ استقراره "
                "عند 0.750 واقعة مقيسة لا سقف بنيوي"},
    "causal_self_attention_1h": {
        "arch": "attention", "layers": 1, "pos_mode": "sin",
        "query_mode": "state", "out_mode": "residual",
        "desc": "رأس واحد/طبقة واحدة، استعلام من تمثيل الموضع، ترميز جيبي حتمي، "
                "logits = WO·(h_t + c_t) + bO — residual معلن"},
    "causal_self_attention_2h": {
        "arch": "attention", "layers": 2, "pos_mode": "sin",
        "query_mode": "state", "out_mode": "residual",
        "desc": "طبقتا انتباه سببي متراكبتان (لكل طبقة WQ/WK/WV خاصة، residual "
                "لكل طبقة، PE مرة واحدة في المدخل): إمكان معماري معلن — الطبقة "
                "الأولى قد تكوّن تمثيلاً سياقياً لرمز القيمة من مفتاحه السابق "
                "والثانية قد تستعلم عنه عند السؤال؛ لا ضمان تجريبي"},
    "no_position": {
        "arch": "attention", "layers": 1, "pos_mode": "none",
        "query_mode": "state", "out_mode": "residual",
        "desc": "عزل الموضع أحادي الطبقة: بلا ترميز موضعي؛ انتباه الطبقة الواحدة "
                "دالة في هوية الرموز فقط، فخرجها عند السؤال دالة في تعداد بادئة "
                "لا ترتيبها (خاصية بنيوية تُقاس نتائجها لا تُفترض)"},
    "no_position_2h": {
        "arch": "attention", "layers": 2, "pos_mode": "none",
        "query_mode": "state", "out_mode": "residual",
        "desc": "عزل الموضع بطبقتين: القناع السببي المتراكم عبر طبقتين قد ينقل "
                "معلومة ترتيب (تحذير العقد) — يُقاس"},
    "csa1h_concat": {
        "arch": "attention", "layers": 1, "pos_mode": "sin",
        "query_mode": "state", "out_mode": "concat",
        "desc": "تبديل سلك الإخراج إلى concat(c_t, h_t): بُعد رأس الإخراج يتضاعف "
                "وتعداد المعلمات يزيد — معلن، ولا ادعاء عزلياً عاملياً منفصلاً عن "
                "تعداد المعلمات"},
}
LEARNED_MODELS = tuple(SPECS)


def init_params(rng, spec):
    p = {"E": rng.normal(0, 0.1, (V, D_MODEL))}
    if spec.get("arch") == "attention":
        for li in range(spec.get("layers", 1)):
            tag = f"{li + 1}" if spec.get("layers", 1) > 1 else ""
            if spec["query_mode"] == "fixed" and li == 0:
                p["q"] = rng.normal(0, 0.1, DK)
            else:
                p[f"WQ{tag}"] = rng.normal(0, 0.1, (D_MODEL, DK))
            p[f"WK{tag}"] = rng.normal(0, 0.1, (D_MODEL, DK))
            p[f"WV{tag}"] = rng.normal(0, 0.1, (D_MODEL, DK))
        out_dim = 2 * D_MODEL if spec["out_mode"] == "concat" else D_MODEL
        p["WO"] = rng.normal(0, 0.1, (out_dim, V))
    else:
        p["WO"] = rng.normal(0, 0.1, (D_MODEL, V))
    p["bO"] = np.zeros(V)
    return p


def build_graph(P, X, spec, leaves=None):
    """يرجع (logits Node، قائمة انتباه الطبقات أو None). القناع السببي في التنعيم."""
    L = leaves or {}
    nd = lambda k: L[k] if L else leaf(P[k])
    ids = np.asarray(X)
    N, T = ids.shape
    if spec.get("arch") == "local":
        H = emb(nd("E"), ids)
        logits = add(mm3_2(H, nd("WO")), nd("bO"))
        return logits, None
    H = emb(nd("E"), ids)
    if spec["pos_mode"] == "sin":
        H = add(H, Node(PE_TABLE[None, :T, :].copy()))
    atts = []
    n_layers = spec.get("layers", 1)
    for li in range(n_layers):
        tag = f"{li + 1}" if n_layers > 1 else ""
        K = mm3_2(H, nd(f"WK{tag}"))
        Vv = mm3_2(H, nd(f"WV{tag}"))
        if spec["query_mode"] == "fixed" and li == 0:
            s = scale(mm2v(H, nd("q")), 1.0 / np.sqrt(DK))
            scores = expand_t(s, N, T, T)
        else:
            Q = mm3_2(H, nd(f"WQ{tag}"))
            scores = scale(einsum_nts(Q, K), 1.0 / np.sqrt(DK))
        A = mask_softmax(scores, T)
        atts.append(A)
        C = einsum_av(A, Vv)
        H = add(H, C) if spec["out_mode"] == "residual" else concat1(C, H)
    logits = add(mm3_2(H, nd("WO")), nd("bO"))
    return logits, atts


class WakeelModel:
    """كل النماذج المتعلمة — نفس مسار التدريب والتقييم والفحص."""

    def __init__(self, name, seed):
        self.name = name
        self.seed = seed
        self.spec = SPECS[name]
        rng = np.random.default_rng(seed)
        self.params = init_params(rng, self.spec)

    def params_count(self):
        return int(sum(a.size for a in self.params.values()))

    def loss_and_grads(self, X, qpos, ans):
        leaves = {k: leaf(v) for k, v in self.params.items()}
        logits, _ = build_graph(None, X, self.spec, leaves=leaves)
        rows, flat_pos, targets = _flat_supervision(qpos, ans)
        loss_node = nll_pick(logits, rows, flat_pos, targets)
        g = backward(loss_node)
        grads = {k: g.get(id(n), np.zeros_like(self.params[k]))
                 for k, n in leaves.items()}
        return float(loss_node.v), grads

    def aggregate_grads(self, groups, total):
        """الخسارة المجمعة الموزونة وتدرجها المُرسل فعلاً إلى المحسّن (إرث 5b):
        L = Σ nᵢ·Lᵢ/Σ nᵢ و ∇L = Σ nᵢ·∇Lᵢ/Σ nᵢ."""
        acc = {k: np.zeros_like(v) for k, v in self.params.items()}
        leaves = {k: leaf(v) for k, v in self.params.items()}
        loss_sum = 0.0
        for X, qpos, ans, cnt in groups:
            logits, _ = build_graph(None, X, self.spec, leaves=leaves)
            rows, flat_pos, targets = _flat_supervision(qpos, ans)
            loss_node = nll_pick(logits, rows, flat_pos, targets)
            loss_sum += float(loss_node.v) * cnt
            g = backward(loss_node)
            for k, n in leaves.items():
                gk = g.get(id(n))
                if gk is not None:
                    acc[k] += gk * cnt
        loss = loss_sum / total
        for k in acc:
            acc[k] /= total
        return loss, acc

    def train_full_batch(self, groups, total, opt):
        loss, grads = self.aggregate_grads(groups, total)
        if not np.isfinite(loss):
            return float("nan")
        opt.step(self.params, grads)
        return loss

    def forward_logits(self, X):
        logits, _ = build_graph(self.params, X, self.spec)
        return logits.v

    def forward_attention(self, X):
        logits, atts = build_graph(self.params, X, self.spec)
        return logits.v, ([a.v for a in atts] if atts else None)

    def next_logits(self, prefix_ids):
        return self.forward_logits([list(prefix_ids)])[0, -1]


def _flat_supervision(qpos, ans):
    rows, flat_pos, targets = [], [], []
    for n in range(len(qpos)):
        for k, t in enumerate(qpos[n]):
            rows.append(n)
            flat_pos.append(t)
            targets.append(ANS_0 + int(ans[n][k]))
    return (np.array(rows), np.array(flat_pos), np.array(targets))


class Adam:
    def __init__(self, keys, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: 0.0 for k in keys}
        self.v = {k: 0.0 for k in keys}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k, g in grads.items():
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


# ── المراجع غير المتعلمة (سقوف/ضوابط — عبر مسار التقييم نفسه) ──
def parse_memory(prefix):
    memory = {}
    i = 0
    while i < len(prefix):
        tok = prefix[i]
        if tok in (K_A, K_B):
            if i + 2 >= len(prefix) or prefix[i + 1] not in (V_0, V_1) \
                    or prefix[i + 2] != SEP:
                raise ValueError("Malformed key-value record")
            if tok in memory:
                raise ValueError("Duplicate key")
            memory[tok] = prefix[i + 1] - V_0
            i += 3
        else:
            i += 1
    return memory


class OracleStructural:
    """مرجع CH9-P0: يربط كل مفتاح بقيمته المتجاورة ويتحقق من SEP — سقف منفصل
    لا يُقارن بزمن تعلم الشبكات."""

    def __init__(self, flip_labels=False):
        self.flip_labels = flip_labels

    def params_count(self):
        return 0

    def next_logits(self, prefix_ids):
        if not prefix_ids or prefix_ids[-1] not in (Q_A, Q_B):
            raise ValueError("Oracle must be called at a query position")
        key = K_A if prefix_ids[-1] == Q_A else K_B
        memory = parse_memory(prefix_ids)
        if set(memory) != {K_A, K_B}:
            raise ValueError("Missing required memory records")
        v = memory[key]
        if self.flip_labels:
            v = 1 - v
        logits = np.full(V, -30.0)
        logits[ANS_0 + v] = 30.0
        return logits

    def forward_logits(self, X):
        out = np.full((X.shape[0], X.shape[1], V), -30.0)
        out[:, :, ANS_0] = 15.0
        out[:, :, ANS_1] = 15.0
        for i in range(X.shape[0]):
            for t in range(X.shape[1]):
                if X[i, t] in QUERY_IDS:
                    out[i, t] = self.next_logits(list(X[i, :t + 1]))
        return out

    def forward_attention(self, X):
        return self.forward_logits(X), None


class OracleNoMemory:
    """ضابط بلا ذاكرة عبر المسار نفسه: احتمالان ثنائيان متساويان."""

    def params_count(self):
        return 0

    def next_logits(self, prefix_ids):
        logits = np.full(V, -30.0)
        logits[ANS_0] = 0.0
        logits[ANS_1] = 0.0
        return logits

    def forward_logits(self, X):
        base = np.full(V, -30.0)
        base[ANS_0] = 0.0
        base[ANS_1] = 0.0
        return np.tile(base, (X.shape[0], X.shape[1], 1))

    def forward_attention(self, X):
        return self.forward_logits(X), None


def new_model(name, seed):
    return WakeelModel(name, seed)


# ═══════════════════════════════ توليد البيانات (إرث P0 الحرفي + المحجوز) ═══════════════════════════════
def make_loop(filler_path, values, record_order, query_order):
    """كل سجل ثلاثة رموز KEY VALUE SEP — لا حقيقة ذرّية."""
    va, vb = values
    records = [("A", va), ("B", vb)]
    if record_order:
        records.reverse()
    queries = [("A", va), ("B", vb)]
    if query_order:
        queries.reverse()
    tokens = []
    for k, v in records:
        tokens.extend((KEYS[k], V_0 + v, SEP))
    tokens += [FILLER_START + int(f) for f in filler_path]
    qpos, qkeys, answers = [], [], []
    for k, v in queries:
        qpos.append(len(tokens))
        qkeys.append(k)
        answers.append(int(v))
        tokens.extend((QUERIES[k], ANS_0 + int(v)))
    return {"tokens": tokens, "length": len(filler_path),
            "family": [len(filler_path), list(filler_path)],
            "values": [int(va), int(vb)], "record_order": int(record_order),
            "query_order": int(query_order), "query_positions": qpos,
            "query_keys": qkeys, "answers": answers}


def _gen_block(rng, lengths, split, owner, seq_seen, prefix_seen):
    """كتلة توليد واحدة — نفس تتابع استدعاءات العشوائية مثل P0 حرفياً."""
    out = []
    for L, count in sorted(lengths.items()):
        if count % len(STRUCT_CLASSES):
            raise ValueError("Split length count must be divisible by 16")
        for _ in range(count // 16):
            for _attempt in range(1000000):
                path = tuple(rng.randrange(N_FILLER) for _ in range(L))
                fam = (L, path)
                if fam not in owner:
                    break
            else:
                raise RuntimeError(f"Filler-family pool exhausted at length {L}")
            owner[fam] = split
            classes = list(STRUCT_CLASSES)
            rng.shuffle(classes)
            for va, vb, ro, qo in classes:
                lp = make_loop(path, (va, vb), ro, qo)
                whole = tuple(lp["tokens"])
                if whole in seq_seen:
                    raise RuntimeError("Duplicate full sequence")
                seq_seen[whole] = split
                for qp in lp["query_positions"]:
                    pre = tuple(lp["tokens"][:qp + 1])
                    if pre in prefix_seen and prefix_seen[pre] != split:
                        raise RuntimeError("Cross-split exact query-prefix overlap")
                    prefix_seen[pre] = split
                out.append(lp)
    return out


def generate_all():
    """مقاطع P0 الخمسة ببذرة 9102 (تطابق مرجع P0)، ثم التأكيدية 9204 ثم النهائية
    9203 — دفاتر العائلات/الحلقات/البادئات مشتركة فتنفصل العائلات عبر المقاطع كلها."""
    rng = random.Random(P0_SEED)
    owner, seq_seen, prefix_seen = {}, {}, {}
    splits = {}
    for sp, lens in P0_LENGTHS.items():                    # بترتيب الإدراج نفسه كما في P0
        splits[sp] = _gen_block(rng, lens, sp, owner, seq_seen, prefix_seen)
    rng_confirm = random.Random(CONFIRM_SEED)
    for sp, lens in CONFIRM_LENGTHS.items():
        splits[sp] = _gen_block(rng_confirm, lens, sp, owner, seq_seen, prefix_seen)
    rng_final = random.Random(FINAL_SEED)
    for sp, lens in FINAL_LENGTHS.items():
        splits[sp] = _gen_block(rng_final, lens, sp, owner, seq_seen, prefix_seen)
    return splits, owner


def seg_digest(loops):
    payload = [[lp["tokens"], lp["query_positions"], lp["values"]] for lp in loops]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":"),
                                      ensure_ascii=False).encode()).hexdigest()


# ═══════════════════════════════ سقوف Bag وأزواج 01/10 (إرث P0) ═══════════════════════════════
def bag_signature(lp, qidx):
    prefix = lp["tokens"][:lp["query_positions"][qidx] + 1]
    return tuple(sorted(Counter(prefix).items()))


def bag_ceiling(loops, qidx):
    """الحد الأمثل تجريبياً لمصنف Order-free عند السؤال qidx (0 أول/1 ثانٍ)."""
    classes = defaultdict(Counter)
    for lp in loops:
        classes[bag_signature(lp, qidx)][lp["answers"][qidx]] += 1
    return sum(max(n.values()) for n in classes.values()) / len(loops)


def first_query_paired_bags(loops):
    """أزواج 01/10 متطابقة الـBag بإجابة أولى معكوسة — لكل (عائلة، ترتيب سجل، ترتيب سؤال)."""
    by_structure = defaultdict(dict)
    for idx, lp in enumerate(loops):
        if tuple(lp["values"]) not in ((0, 1), (1, 0)):
            continue
        key = ((lp["length"], tuple(lp["family"][1])),
               lp["record_order"], lp["query_order"])
        by_structure[key][tuple(lp["values"])] = idx
    pairs = []
    for group in by_structure.values():
        if set(group) != {(0, 1), (1, 0)}:
            raise RuntimeError("Missing matched opposite-binding pair")
        ia, ib = group[(0, 1)], group[(1, 0)]
        a, b = loops[ia], loops[ib]
        if bag_signature(a, 0) != bag_signature(b, 0) or a["answers"][0] == b["answers"][0]:
            raise RuntimeError("Matched bag changed or answer failed to flip")
        pairs.append((ia, ib))
    return pairs


# ═══════════════════════════════ التدخلات المزدوجة عند السؤال الثاني ═══════════════════════════════
def intervene(lp, mode):
    """تلاعب بمدخلات السؤال الثاني (تدخلات مدخلات لا عوالم بديلة مكتملة):
    memory_only: قلب قيمة سجل مفتاح السؤال الأول (تصبح البادئة غير متسقة عمداً)؛
    prior_answer_only: قلب رمز الجواب السابق وحده؛ both: الاثنان معاً (متسق من جديد)."""
    if mode not in INTERVENTION_MODES:
        raise ValueError(mode)
    x = list(lp["tokens"])
    first_key = KEYS[lp["query_keys"][0]]
    rec_at = x.index(first_key)
    if x[rec_at + 1] not in (V_0, V_1) or x[rec_at + 2] != SEP:
        raise ValueError("Malformed first-query key record")
    prev_answer_at = lp["query_positions"][0] + 1
    if mode in ("memory_only", "both"):
        x[rec_at + 1] = V_1 if x[rec_at + 1] == V_0 else V_0
    if mode in ("prior_answer_only", "both"):
        x[prev_answer_at] = ANS_1 if x[prev_answer_at] == ANS_0 else ANS_0
    return x


def intervention_eval(model, loops):
    """استجابة النموذج المقاسة عند السؤال الثاني للتدخلات الثلاثة (تُقاس ولا تستنتج).
    سؤال المفتاح الآخر لا تتغير تسميته الصحيحة بأي تدخل (بوابة 8)."""
    out = {"loops": len(loops), "modes": {}}
    base_correct = 0
    stats = {m: {"q2_binary_correct": 0, "pred_flip_vs_base": 0.0,
                 "mean_max_binary_prob_shift": 0.0} for m in INTERVENTION_MODES}
    for lp in loops:
        t2 = lp["query_positions"][1]
        pb = _binary_pair(model.next_logits(lp["tokens"][:t2 + 1]))
        base_pred = int(np.argmax(pb))
        base_correct += int(base_pred == lp["answers"][1])
        for m in INTERVENTION_MODES:
            x = intervene(lp, m)
            pm = _binary_pair(model.next_logits(x[:t2 + 1]))
            pred = int(np.argmax(pm))
            stats[m]["q2_binary_correct"] += int(pred == lp["answers"][1])
            stats[m]["pred_flip_vs_base"] += int(pred != base_pred)
            stats[m]["mean_max_binary_prob_shift"] += float(np.max(np.abs(pm - pb)))
    out["baseline_q2_binary_correct"] = round(base_correct / len(loops), 4)
    for m in INTERVENTION_MODES:
        s = stats[m]
        out["modes"][m] = {"q2_binary_accuracy": round(s["q2_binary_correct"] / len(loops), 4),
                           "prediction_flip_rate_vs_base": round(s["pred_flip_vs_base"] / len(loops), 4),
                           "mean_max_binary_prob_shift": round(
                               s["mean_max_binary_prob_shift"] / len(loops), 4)}
    return out


# ═══════════════════════════════ مسار التقييم الواحد ═══════════════════════════════
def _binary_pair(logits):
    two = logits[[ANS_0, ANS_1]]
    z = two - two.max()
    e = np.exp(z)
    return e / e.sum()


def _zero_att_groups():
    return {f"{key}|{'first' if pos == 0 else 'second'}":
            {"correct_key": 0.0, "correct_value": 0.0, "other_key": 0.0,
             "other_value": 0.0, "sep": 0.0, "filler": 0.0,
             "prior_query": 0.0, "prior_answer": 0.0, "self_query": 0.0}
            for key in ("A", "B") for pos in (0, 1)}


def _att_add(group, arow, lp, t, pos_idx):
    """تصنيف انتباه موضع سؤال — قرينة لا إسناد سببي؛ يُجمع لكل (مفتاح، موضع)."""
    L = lp["length"]
    recs = {0: (lp["tokens"][0], lp["tokens"][1]),
            3: (lp["tokens"][3], lp["tokens"][4])}
    asked_key = lp["query_keys"][pos_idx]
    correct_start = 0 if recs[0][0] == KEYS[asked_key] else 3
    other_start = 3 - correct_start
    for j in range(t + 1):
        w = float(arow[j])
        if j in (2, 5):
            group["sep"] += w
        elif j == correct_start:
            group["correct_key"] += w
        elif j == correct_start + 1:
            group["correct_value"] += w
        elif j == other_start:
            group["other_key"] += w
        elif j == other_start + 1:
            group["other_value"] += w
        elif 6 <= j < 6 + L:
            group["filler"] += w
        elif j == 6 + L:
            group["prior_query" if pos_idx == 1 else "self_query"] += w
        elif j == 6 + L + 1:
            group["prior_answer"] += w
        elif j == 8 + L:
            group["self_query"] += w
        else:  # غير ممكن بنيوياً
            raise RuntimeError(f"attention category fall-through at j={j}")


def _att_sum_check(counts, sums, what):
    """بوابة التطبيع: مجموع فئات كل مجموعة = عددها ضمن دقة الآلة — يرفض دائماً
    بغضّ النظر عن python -O؛ يُستخدم أيضاً للاختبار السلبي (بوابة 9)."""
    for gk, n in counts.items():
        if n <= 0:
            raise RuntimeError(f"{what}: no positions for group {gk}")
        total = sum(sums[gk].values())
        if abs(total - n) > 1e-9 * max(n, 1):
            raise RuntimeError(
                f"{what}: normalization broken for {gk}: sum={total} n={n}")


def _norm_att(att):
    """تطبيع مستقل لكل (طبقة، مفتاح، موضع) بعدده هو — قبل التقريب وبعده فحص المجموع."""
    out = {}
    for layer_key, layers in att.items():
        counts, sums = layers["n"], layers["sum"]
        _att_sum_check(counts, sums, f"attention[{layer_key}]")
        out[layer_key] = {gk: {kk: round(vv / counts[gk], 4)
                               for kk, vv in sv.items()} for gk, sv in sums.items()}
        for gk in out[layer_key]:
            s = round(sum(out[layer_key][gk].values()), 4)
            if abs(s - 1.0) > 5e-4:
                raise RuntimeError(f"rounded attention off for {layer_key}/{gk}: {s}")
    return out


def attention_accumulate(model, loops):
    """تجميع خام لانتباه مواضع الأسئلة لكل (طبقة، مفتاح، موضع) — يُرجع None
    للنماذج بلا انتباه (local/المراجع) فلا تُنشأ مجموعات فارغة زائفة."""
    if getattr(model, "spec", {}).get("arch") != "attention":
        return None
    n_layers = model.spec.get("layers", 1)
    att = {f"layer{i + 1}": {"sum": _zero_att_groups(),
                             "n": {k: 0 for k in _zero_att_groups()}}
           for i in range(n_layers)}
    by_len = {}
    for i, lp in enumerate(loops):
        by_len.setdefault(lp["length"], []).append(i)
    for Ln, idxs in sorted(by_len.items()):
        batch = [loops[i] for i in idxs]
        X = np.array([lp["tokens"] for lp in batch])
        _, A = model.forward_attention(X)
        if A is None:
            return None
        for bi, lp in enumerate(batch):
            for pos_idx, t in enumerate(lp["query_positions"]):
                gk = f"{lp['query_keys'][pos_idx]}|{'first' if pos_idx == 0 else 'second'}"
                for li in range(len(A)):
                    _att_add(att[f"layer{li + 1}"]["sum"][gk],
                             A[li][bi, t], lp, t, pos_idx)
                    att[f"layer{li + 1}"]["n"][gk] += 1
    return att


def binding_metrics(model, loops, capture_attention=False):
    """مسار تقييم واحد لكل النماذج (المراجع ضمناً) — teacher forcing على الحلقات
    الكاملة (تكافؤ البادئة مُثبت بالبوابة 5-ج)."""
    n = len(loops)
    per_loop = [None] * n
    by_len = {}
    for i, lp in enumerate(loops):
        by_len.setdefault(lp["length"], []).append(i)
    for Ln, idxs in sorted(by_len.items()):
        batch = [loops[i] for i in idxs]
        X = np.array([lp["tokens"] for lp in batch])
        logits, _ = model.forward_attention(X)
        for bi, lp in enumerate(batch):
            rec = {"cls": f"{lp['values'][0]}{lp['values'][1]}",
                   "record_order": lp["record_order"],
                   "query_order": lp["query_order"], "length": lp["length"],
                   "first_key": lp["query_keys"][0]}
            for pos_idx, t in enumerate(lp["query_positions"]):
                lv = logits[bi, t]
                z = lv - lv.max()
                p_full = np.exp(z)
                p_full = p_full / p_full.sum()
                ans_tok = ANS_0 + lp["answers"][pos_idx]
                nlf = -np.log(max(p_full[ans_tok], 1e-300))
                pb = _binary_pair(lv)
                nlb = -np.log(max(pb[lp["answers"][pos_idx]], 1e-300))
                rec[f"q{pos_idx + 1}_bin_ok"] = int(int(np.argmax(pb)) == lp["answers"][pos_idx])
                rec[f"q{pos_idx + 1}_full_ok"] = int(int(np.argmax(lv)) == ans_tok)
                rec[f"q{pos_idx + 1}_nll_full"] = nlf
                rec[f"q{pos_idx + 1}_nll_bin"] = nlb
            per_loop[idxs[bi]] = rec

    def acc(field):
        return round(sum(r[field] for r in per_loop) / n, 4)

    def nll(field):
        return round(float(np.mean([r[field] for r in per_loop])), 4)

    out = {
        "n_loops": n,
        "first_query_binary_accuracy": acc("q1_bin_ok"),
        "first_query_full_vocab_accuracy": acc("q1_full_ok"),
        "first_query_nll_binary": nll("q1_nll_bin"),
        "second_query_binary_accuracy": acc("q2_bin_ok"),
        "second_query_full_vocab_accuracy": acc("q2_full_ok"),
        "second_query_nll_binary": nll("q2_nll_bin"),
        "both_answers_correct": round(sum(r["q1_bin_ok"] and r["q2_bin_ok"]
                                          for r in per_loop) / n, 4),
        "full_vocab_accuracy": round(sum(r["q1_full_ok"] + r["q2_full_ok"]
                                         for r in per_loop) / (2 * n), 4),
        "nll_full_vocab": round(float(np.mean([r["q1_nll_full"] + r["q2_nll_full"]
                                               for r in per_loop])) / 2, 4),
        "nll_binary_normalized": round(float(np.mean([r["q1_nll_bin"] + r["q2_nll_bin"]
                                                      for r in per_loop])) / 2, 4),
    }
    pairs = first_query_paired_bags(loops)
    ok_pairs = sum(per_loop[ia]["q1_bin_ok"] and per_loop[ib]["q1_bin_ok"]
                   for ia, ib in pairs)
    out["first_query_paired_both_correct"] = round(ok_pairs / len(pairs), 4)
    out["matched_pairs"] = len(pairs)

    def breakdown(field):
        groups = defaultdict(list)
        for r in per_loop:
            groups[str(r[field])].append(r["q1_bin_ok"])
        return {k: {"n": len(v),
                    "first_query_binary_accuracy": round(float(np.mean(v)), 4)}
                for k, v in sorted(groups.items())}

    out["first_query_breakdowns"] = {
        "value_class": breakdown("cls"),
        "record_order": breakdown("record_order"),
        "query_order": breakdown("query_order"),
        "filler_length": breakdown("length"),
        "first_asked_key": breakdown("first_key"),
    }
    if capture_attention:
        att = attention_accumulate(model, loops)
        if att is not None:
            out["attention_at_query_positions"] = _norm_att(att)
    return out


def free_rollout(model, loops):
    """توليد حر جشع: جواب السؤال الأول المولَّد يدخل سياق السؤال الثاني."""
    n = pos1 = pos2 = both = gen1_ans = 0
    for lp in loops:
        t1, t2 = lp["query_positions"]
        a1, a2 = lp["answers"]
        ctx = list(lp["tokens"][:t1 + 1])
        g1 = int(np.argmax(model.next_logits(ctx)))
        gen1_ans += int(g1 in ANS_IDS)
        ctx2 = ctx + [g1] + list(lp["tokens"][t1 + 2:t2 + 1])
        g2 = int(np.argmax(model.next_logits(ctx2)))
        c1, c2 = g1 == ANS_0 + a1, g2 == ANS_0 + a2
        pos1 += c1
        pos2 += c2
        both += c1 and c2
        n += 1
    return {"n_loops": n,
            "q1_accuracy": round(pos1 / n, 4),
            "q2_free_accuracy": round(pos2 / n, 4),
            "both_correct_free": round(both / n, 4),
            "gen1_is_answer_token_rate": round(gen1_ans / n, 4)}


# ═══════════════════════ عينات بوابات الآلة (تركيبية ببذور خاصة — خارج المقاطع) ═══════════════════════
def _gate_batch(lengths_counts, seed):
    rng = np.random.default_rng(seed)
    X, qpos, ans = [], [], []
    for Ln, cnt in sorted(lengths_counts.items()):
        for _ in range(cnt):
            va, vb, ro, qo = [int(x) for x in rng.integers(0, 2, 4)]
            path = tuple(int(f) for f in rng.integers(0, N_FILLER, size=Ln))
            lp = make_loop(path, (va, vb), ro, qo)
            X.append(lp["tokens"])
            qpos.append(lp["query_positions"])
            ans.append(lp["answers"])
    return np.array(X), qpos, ans


def _gate_mixed():
    return [_gate_batch({2: 3}, seed=555), _gate_batch({8: 3}, seed=556)]


def _suffix_gate_loops():
    rng = np.random.default_rng(77)
    out = []
    for Ln in (2, 8):
        for _ in range(3):
            va, vb, ro, qo = [int(x) for x in rng.integers(0, 2, 4)]
            path = tuple(int(f) for f in rng.integers(0, N_FILLER, size=Ln))
            out.append(make_loop(path, (va, vb), ro, qo))
    return out


_FD_CACHE = None


def fd_batches():
    global _FD_CACHE
    if _FD_CACHE is None:
        _FD_CACHE = [_gate_batch({2: 4}, seed=901), _gate_batch({8: 4}, seed=902)]
    return _FD_CACHE


# ═══════════════════════════════ بوابات §7 (1–10؛ 11 مؤجلة؛ 12 بنيوية) ═══════════════════════════════
def gate1_input_contract(segments, seg_names):
    checked = 0
    for sp in seg_names:
        for lp in segments[sp]:
            tk = lp["tokens"]
            L = lp["length"]
            if len(tk) != 10 + L:
                raise RuntimeError(f"gate1: loop length {len(tk)} != {10 + L}")
            if tk[0] not in (K_A, K_B) or tk[3] not in (K_A, K_B) or tk[0] == tk[3]:
                raise RuntimeError("gate1: record keys malformed/duplicated")
            if tk[1] not in (V_0, V_1) or tk[4] not in (V_0, V_1):
                raise RuntimeError("gate1: record values malformed")
            if tk[2] != SEP or tk[5] != SEP:
                raise RuntimeError("gate1: SEP delimiter missing")
            if lp["query_positions"] != [6 + L, 8 + L]:
                raise RuntimeError("gate1: query positions malformed")
            if tk[6 + L] not in QUERY_IDS or tk[8 + L] not in QUERY_IDS \
                    or tk[6 + L] == tk[8 + L]:
                raise RuntimeError("gate1: query tokens malformed")
            rec = {tk[0]: tk[1] - V_0, tk[3]: tk[4] - V_0}
            for pos_idx in (0, 1):
                key = lp["query_keys"][pos_idx]
                if tk[6 + L + 2 * pos_idx] != QUERIES[key]:
                    raise RuntimeError("gate1: query key mismatch")
                if tk[7 + L + 2 * pos_idx] != ANS_0 + rec[KEYS[key]]:
                    raise RuntimeError("gate1: shown answer contradicts record")
            checked += 1
    return {"loops_checked": checked,
            "atomic_fact_tokens": "لا يوجد — كل مفتاح وقيمته رمزان منفصلان وSEP متحقق",
            "verified_segments": list(seg_names)}


def _balance_check(segments, seg_names):
    detail = {}
    for sp in seg_names:
        by_len = defaultdict(Counter)
        for lp in segments[sp]:
            by_len[lp["length"]][(lp["values"][0], lp["values"][1],
                                  lp["record_order"], lp["query_order"])] += 1
        for Ln, counts in sorted(by_len.items()):
            n_block = sum(counts.values()) // 16
            if not all(c == n_block for c in counts.values()) or len(counts) != 16:
                raise RuntimeError(f"gate2: balance broken {sp} L={Ln}")
        detail[sp] = "16 صنفاً متوازناً في كل كتلة (مقطع×طول)"
    return detail


def gate2_split_contract(segments, owner):
    sizes = {s: len(segments[s]) for s in SEGMENTS}
    want = {"train": 512, "val": 128, "dev_holdout": 128,
            "p0_probe_in": 64, "p0_probe_ood_length16": 64,
            "confirm_in": 64, "confirm_ood16": 64,
            "final_in": 64, "final_ood16": 64}
    if sizes != want:
        raise RuntimeError(f"gate2: split sizes {sizes}")
    fams = defaultdict(set)
    for fam, sp in owner.items():
        fams[sp].add(fam)
    overlaps = {f"{a}|{b}": len(fams[a] & fams[b])
                for i, a in enumerate(SEGMENTS) for b in SEGMENTS[i + 1:]}
    if any(v != 0 for v in overlaps.values()):
        raise RuntimeError(f"gate2: family overlap {overlaps}")
    golden = json.loads(P0_GOLDEN.read_text(encoding="utf-8"))
    p0_match = {sp: seg_digest(segments[sp]) == golden["split_sha256"][sp]
                for sp in P0_LENGTHS}
    if not all(p0_match.values()):
        raise RuntimeError(f"gate2: P0 golden digests not reproduced: {p0_match}")
    balance_non_blind = _balance_check(segments, NON_BLIND)
    return {"segment_sizes": sizes,
            "family_counts": {s: len(fams[s]) for s in SEGMENTS},
            "family_overlap_counts": overlaps,
            "p0_golden_digests_match": p0_match,
            "balance_non_blind": balance_non_blind,
            "ledger_note": "دفاتر التوليد (عائلات/حلقات/بادئات) مشتركة عبر المقاطع "
                           "التسعة: لا عائلة ولا حلقة ولا بادئة كاملة مشتركة",
            "data_generation_seeds": {"p0": P0_SEED, "confirm": CONFIRM_SEED,
                                      "final": FINAL_SEED},
            "generation_order": ["P0 splits (9102)", "confirm (9204)", "final (9203)"]}


def gate3_pair_builders(segments, seg_names):
    counts = {}
    for sp in seg_names:
        pairs = first_query_paired_bags(segments[sp])   # raises on any breakage
        counts[sp] = len(pairs)
    # Mutation monitor (contract 7.3): every mutation must be DETECTED.
    # Structural note: permuting positions inside one prefix can never change a
    # bag (it is order-free) -- the honest negative controls break the pair set
    # itself, which is what the builder must catch.
    sample_pairs = first_query_paired_bags(segments["train"])
    ia, ib = sample_pairs[0]
    a, b = segments["train"][ia], segments["train"][ib]
    sample = segments["train"][0]
    flagged = {}
    # (a) duplicated pair member (opposite member missing) -> builder rejects
    try:
        first_query_paired_bags([a, dict(a)])
        flagged["duplicate_pair_member_detected"] = False
    except RuntimeError:
        flagged["duplicate_pair_member_detected"] = True
    # (b) tampered filler in one member under the same family (bag changed) -> rejects
    b2 = dict(b)
    b2["tokens"] = list(b["tokens"])
    b2["tokens"][6] = (b2["tokens"][6] - FILLER_START + 1) % N_FILLER + FILLER_START
    try:
        first_query_paired_bags([a, b2])
        flagged["tampered_filler_breaks_matched_bag"] = False
    except RuntimeError:
        flagged["tampered_filler_breaks_matched_bag"] = True
    # (c) SEP replaced by filler -> input contract rejects
    mut2 = dict(sample)
    mut2["tokens"] = list(sample["tokens"])
    mut2["tokens"][2] = FILLER_START
    try:
        gate1_input_contract({"m": [mut2]}, ("m",))
        flagged["sep_filler_swap_detected_by_input_contract"] = False
    except RuntimeError:
        flagged["sep_filler_swap_detected_by_input_contract"] = True
    # (d) shown answer contradicting the record -> input contract rejects
    mut3 = dict(sample)
    mut3["tokens"] = list(sample["tokens"])
    t1 = sample["query_positions"][0]
    mut3["tokens"][t1 + 1] = ANS_1 if mut3["tokens"][t1 + 1] == ANS_0 else ANS_0
    try:
        gate1_input_contract({"m": [mut3]}, ("m",))
        flagged["answer_record_mismatch_detected"] = False
    except RuntimeError:
        flagged["answer_record_mismatch_detected"] = True
    if not all(flagged.values()):
        raise RuntimeError(f"gate3: mutation monitor failed: {flagged}")
    return {"opposite_pairs_per_segment": counts,
            "mutation_monitor": flagged,
            "note": "per (family, record_order, query_order) a 01/10 pair with "
                    "identical first-query bag and flipped first answer -- the "
                    "P0 invariant, re-verified per segment; the mutation monitor "
                    "proves the checkers fire on broken inputs"}


def gate4_references(segments, seg_names):
    out = {"segments": {}}
    o = OracleStructural()
    for sp in seg_names:
        ev = binding_metrics(o, segments[sp])
        if ev["first_query_binary_accuracy"] != 1.0 or \
                ev["second_query_binary_accuracy"] != 1.0 or \
                ev["first_query_paired_both_correct"] != 1.0:
            raise RuntimeError(f"gate4: oracle failed on {sp}")
        out["segments"][sp] = {
            "oracle_binary_both_queries": 1.0,
            "bag_ceiling_first_query": round(bag_ceiling(segments[sp], 0), 8),
            "bag_ceiling_second_query": round(bag_ceiling(segments[sp], 1), 8),
            "bag_matched_pair_both_correct_ceiling_first_query": 0.0,
            "oracle_matched_pair_both_correct": 1.0}
        if out["segments"][sp]["bag_ceiling_first_query"] != 0.75:
            raise RuntimeError(f"gate4: bag ceiling != 0.75 on {sp}")
    flip = binding_metrics(OracleStructural(flip_labels=True), segments["val"])
    if flip["first_query_binary_accuracy"] != 0.0:
        raise RuntimeError("gate4: label flip did not collapse to 0")
    out["val_label_flip_binary_accuracy"] = 0.0
    out["no_memory_control"] = binding_metrics(OracleNoMemory(), segments["val"])
    out["local_only_note"] = "يُقاس أداء local_only بعد التدريب ولا يُفترض فشله العددي"
    return out


def gate5_causal_mask_exact(models):
    per = {}
    for Xb, _, _ in _gate_mixed():
        for name, m in models.items():
            _, A = m.forward_attention(Xb)
            if A is None:
                continue
            for li, Al in enumerate(A):
                T = Al.shape[-1]
                future = max(float(Al[i, t, j])
                             for i in range(len(Xb)) for t in range(T)
                             for j in range(t + 1, T))
                row_err = float(np.max(np.abs(Al.sum(axis=-1) - 1.0)))
                key = f"{name}_layer{li + 1}"
                cur = per.get(key, {"future_weight_max": 0.0,
                                    "row_sum_max_abs_error": 0.0})
                cur["future_weight_max"] = max(cur["future_weight_max"], future)
                cur["row_sum_max_abs_error"] = max(cur["row_sum_max_abs_error"], row_err)
                per[key] = cur
                if future != 0.0 or row_err > GATE_ZERO_TOL:
                    print(f"FAIL gate5 {key}: {cur}")
                    return None
    return per


def _alt_suffix(lp, t, mode):
    toks = list(lp["tokens"][:t + 1])
    rest = list(lp["tokens"][t + 1:])
    if mode == "flip_answers":
        rest = [ANS_1 if x == ANS_0 else (ANS_0 if x == ANS_1 else x) for x in rest]
    else:
        rest = [((x - FILLER_START + 1) % N_FILLER) + FILLER_START
                if FILLER_START <= x < FILLER_START + N_FILLER else x for x in rest]
    return toks + rest


def gate5b_future_suffix_invariance(model):
    worst = 0.0
    for lp in _suffix_gate_loops():
        for t in (lp["query_positions"][0], 6 + lp["length"] // 2):
            L0 = model.forward_logits([lp["tokens"]])[0, t]
            for mode in ("flip_answers", "different_filler"):
                L1 = model.forward_logits([_alt_suffix(lp, t, mode)])[0, t]
                worst = max(worst, float(np.max(np.abs(L0 - L1))))
    return worst


def gate5c_prefix_vs_full(model):
    worst = 0.0
    for lp in _suffix_gate_loops():
        full, _ = model.forward_attention(np.array([lp["tokens"]]))
        for t in lp["query_positions"]:
            pre = model.next_logits(lp["tokens"][:t + 1])
            worst = max(worst, float(np.max(np.abs(full[0, t] - pre))))
    return worst


def gate6_finite_difference(model, batches):
    worst = 0.0
    for X, qpos, ans in batches:
        _, analytic = model.loss_and_grads(X, qpos, ans)
        for pname, arr in model.params.items():
            garr = analytic[pname]
            for idx in np.ndindex(arr.shape):
                old = arr[idx]
                arr[idx] = old + FD_EPS
                lp_, _ = model.loss_and_grads(X, qpos, ans)
                arr[idx] = old - FD_EPS
                lm_, _ = model.loss_and_grads(X, qpos, ans)
                arr[idx] = old
                num = (lp_ - lm_) / (2 * FD_EPS)
                worst = max(worst, abs(float(garr[idx]) - num))
    return worst


def gate6b_aggregated_gradient(model):
    b1 = _gate_batch({2: 3}, seed=903)   # 6 مواضع إشراف
    b2 = _gate_batch({8: 5}, seed=904)   # 10 مواضع إشراف
    groups = [(b1[0], b1[1], b1[2], 6), (b2[0], b2[1], b2[2], 10)]
    total = 16
    _, grads = model.aggregate_grads(groups, total)

    def agg_loss(weighted=True):
        s = 0.0
        for X, qpos, ans, cnt in groups:
            leaves = {k: leaf(v) for k, v in model.params.items()}
            logits, _ = build_graph(None, X, model.spec, leaves=leaves)
            rows, flat_pos, targets = _flat_supervision(qpos, ans)
            v = float(nll_pick(logits, rows, flat_pos, targets).v)
            s += v * (cnt if weighted else 1)
        return s / (total if weighted else len(groups))

    worst = 0.0
    for pname, arr in model.params.items():
        garr = grads[pname]
        for idx in np.ndindex(arr.shape):
            old = arr[idx]
            arr[idx] = old + FD_EPS
            lp_ = agg_loss()
            arr[idx] = old - FD_EPS
            lm_ = agg_loss()
            arr[idx] = old
            num = (lp_ - lm_) / (2 * FD_EPS)
            worst = max(worst, abs(float(garr[idx]) - num))
    # اختبار سلبي إلزامي: التجميع غير الموزون يجب أن يُكشف (عيب v0.16-CH8)
    _, grads_unweighted = None, None
    acc = {k: np.zeros_like(v) for k, v in model.params.items()}
    leaves = {k: leaf(v) for k, v in model.params.items()}
    for X, qpos, ans, cnt in groups:
        logits, _ = build_graph(None, X, model.spec, leaves=leaves)
        rows, flat_pos, targets = _flat_supervision(qpos, ans)
        loss_node = nll_pick(logits, rows, flat_pos, targets)
        g = backward(loss_node)
        for k, nd_ in leaves.items():
            gk = g.get(id(nd_))
            if gk is not None:
                acc[k] += gk
    for k in acc:
        acc[k] /= len(groups)
    neg_worst = 0.0
    for pname, arr in model.params.items():
        garr = acc[pname]
        for idx in np.ndindex(arr.shape):
            old = arr[idx]
            arr[idx] = old + FD_EPS
            lp_ = agg_loss()
            arr[idx] = old - FD_EPS
            lm_ = agg_loss()
            arr[idx] = old
            num = (lp_ - lm_) / (2 * FD_EPS)
            neg_worst = max(neg_worst, abs(float(garr[idx]) - num))
    if not (neg_worst > 10 * GATE_GRAD_TOL):
        raise RuntimeError("gate6b: negative test failed to expose unweighted aggregation")
    return {"buckets_positions": [6, 10], "total": total,
            "worst_abs_error": round(worst, 12),
            "negative_unweighted_worst_abs_error": round(neg_worst, 12),
            "eps": FD_EPS,
            "objective": "L = Σ nᵢ·Lᵢ/Σ nᵢ وتدرجها الموزون (بوابة إرث CH-8 5b)"}


def gate7_loss_contract(train_loops):
    checked = 0
    q2_overlap = 0
    for lp in train_loops:
        L = lp["length"]
        if lp["query_positions"] != [6 + L, 8 + L]:
            raise RuntimeError("gate7: supervised positions malformed")
        ctx1 = lp["tokens"][:6 + L + 1]
        if ctx1[-1] not in QUERY_IDS:
            raise RuntimeError("gate7: context does not end at a query")
        if any(tok in ANS_IDS for tok in ctx1):
            raise RuntimeError("gate7: an answer token appears before the first query")
        t2 = 8 + L
        ctx2 = lp["tokens"][:t2 + 1]
        if ctx2[-1] not in QUERY_IDS:
            raise RuntimeError("gate7: second context malformed")
        tgt2 = ANS_0 + lp["answers"][1]
        if tgt2 in ANS_IDS and tgt2 in ctx2[:-1]:
            q2_overlap += 1
        checked += 2
    by_len = Counter(lp["length"] for lp in train_loops)
    return {"supervised_positions": checked,
            "positions_per_loop": 2,
            "q1_context_free_of_answer_tokens": True,
            "q2_target_token_already_visible_rate": round(q2_overlap / len(train_loops), 4),
            "q2_visibility_note": "في الأصناف 00/11 (نصف العينات) يظهر رمز جواب "
                                  "السؤال الثاني كسابقه — هذه قناة الاختصار البنيوية "
                                  "التي تفحصها التدخلات المزدوجة، وليست خطأ loss",
            "train_buckets_positions": {str(k): v * 2 for k, v in sorted(by_len.items())}}


def gate8_double_intervention(loops):
    n = 0
    for lp in loops:
        base = lp["tokens"]
        first_key = KEYS[lp["query_keys"][0]]
        rec_at = base.index(first_key)
        x_m = intervene(lp, "memory_only")
        x_a = intervene(lp, "prior_answer_only")
        x_b = intervene(lp, "both")
        dm = [i for i, (a, b) in enumerate(zip(base, x_m)) if a != b]
        da = [i for i, (a, b) in enumerate(zip(base, x_a)) if a != b]
        db = [i for i, (a, b) in enumerate(zip(base, x_b)) if a != b]
        if dm != [rec_at + 1] or x_m[rec_at + 1] not in (V_0, V_1):
            raise RuntimeError("gate8: memory_only changed more/other than one value token")
        if da != [lp["query_positions"][0] + 1] or x_a[da[0]] not in ANS_IDS:
            raise RuntimeError("gate8: prior_answer_only changed more/other than one answer token")
        if sorted(db) != sorted([rec_at + 1, lp["query_positions"][0] + 1]):
            raise RuntimeError("gate8: both changed positions other than the declared two")
        other_key = K_B if first_key == K_A else K_A
        o_at = base.index(other_key)
        for x in (x_m, x_a, x_b):
            if x[o_at + 1] != base[o_at + 1]:
                raise RuntimeError("gate8: other-key record was touched")
            if x[lp["query_positions"][1] + 1] != base[lp["query_positions"][1] + 1]:
                raise RuntimeError("gate8: second answer token was touched")
        n += 1
    return {"loops_checked": n,
            "memory_only_changes": "رمز قيمة واحد (سجل مفتاح السؤال الأول)",
            "prior_answer_only_changes": "رمز جواب سابق واحد",
            "both_changes": "الرمزان معاً (بادئة متسقة من جديد)",
            "other_key_record_and_q2_label_untouched": True}


def gate9_measurement_integrity(trained_models, val_loops):
    # (أ) تطبيع مستقل لكل مجموعة على مجمع خام حقيقي — على val كاملة (أعداد
    # متساوية عبر المجموعات) وعلى مجموعة جزئية (أعداد غير متساوية بين المجموعات)
    m1 = trained_models["causal_self_attention_1h"]
    raw_full = attention_accumulate(m1, val_loops)
    for lk, layers in raw_full.items():
        _att_sum_check(layers["n"], layers["sum"], f"gate9-full[{lk}]")
    raw_part = attention_accumulate(m1, val_loops[:33])
    unequal = {lk: len(set(layers["n"].values())) > 1
               for lk, layers in raw_part.items()}
    for lk, layers in raw_part.items():
        _att_sum_check(layers["n"], layers["sum"], f"gate9-part[{lk}]")
    # اختبار سلبي: إنقاص عدد مجموعة واحدة وحدة واحدة يجب أن ترفضه البوابة
    counts, sums = raw_full["layer1"]["n"], raw_full["layer1"]["sum"]
    k0 = sorted(counts)[0]
    tampered = dict(counts)
    tampered[k0] = counts[k0] - 1
    try:
        _att_sum_check(tampered, sums, "gate9-negative")
        raise RuntimeError("gate9: normalization negative test did not fire")
    except RuntimeError as e:
        if "did not fire" in str(e):
            raise
    maps = binding_metrics(m1, val_loops, capture_attention=True)[
        "attention_at_query_positions"]
    # (ب) حساب العرض لا يغير نتائج الدقة: التقاط الانتباه مقابل عدمه
    for name, m in trained_models.items():
        with_cap = binding_metrics(m, val_loops, capture_attention=True)
        without = binding_metrics(m, val_loops, capture_attention=False)
        for k in ("first_query_binary_accuracy", "second_query_binary_accuracy",
                  "both_answers_correct", "nll_full_vocab"):
            if with_cap[k] != without[k]:
                raise RuntimeError(f"gate9: capture changed {k} for {name}")
    return {"normalization_groups": sorted(maps["layer1"].keys()),
            "sum_check_tolerance": 1e-9,
            "partial_subset_unequal_group_counts": unequal,
            "negative_off_by_one_count_rejected": True,
            "capture_invariance_models": list(trained_models),
            "rounded_sum_max_abs_error": max(
                abs(round(sum(v.values()), 4) - 1.0)
                for lg in maps.values() for v in lg.values())}


# ═══════════════════════════════ التدريب ═══════════════════════════════
def make_groups(loops):
    by_len = {}
    for lp in loops:
        by_len.setdefault(lp["length"], []).append(lp)
    groups, total = [], 0
    for Ln, lps in sorted(by_len.items()):
        X = np.array([lp["tokens"] for lp in lps])
        qpos = [lp["query_positions"] for lp in lps]
        ans = [lp["answers"] for lp in lps]
        cnt = sum(len(q) for q in qpos)
        total += cnt
        groups.append((X, qpos, ans, cnt))
    return groups, total


def train_model(model, train_loops, lr, epochs=EPOCHS):
    groups, total = make_groups(train_loops)
    opt = Adam(list(model.params), lr)
    curve, t0 = [], []
    final = float("nan")
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        final = model.train_full_batch(groups, total, opt)
        if not np.isfinite(final):
            final = float("nan")
            break
        if ep in CURVE_AT:
            curve.append({"epoch": ep,
                          "train_nll": round(final, 4) if np.isfinite(final) else None})
    updates = opt.t
    return {"final_train_nll": round(final, 4) if np.isfinite(final) else None,
            "stable": bool(np.isfinite(final)),
            "curve": curve,
            "updates": updates,
            "train_seconds": round(time.perf_counter() - t0, 2)}


def train_batch_steps(model, batch, lr, epochs):
    X, qpos, ans = batch
    total = sum(len(q) for q in qpos)
    opt = Adam(list(model.params), lr)
    for _ in range(epochs):
        model.train_full_batch([(X, qpos, ans, total)], total, opt)


def validation_sweep(train_loops, val_loops):
    """مسبح الاختيار على التحقق وحده — قاعدة معلنة قبل النتائج:
    مرشح كل lr = min عبر المرسيات الثلاث (دقة السؤال الأول الثنائية على val)؛
    الأعلى يفوز؛ التعادل → الأصغر. لا مساس بdev/التأكيدي/النهائي."""
    trials = []
    for name in SWEEP_ANCHORS:
        for lr in LR_GRID:
            m = new_model(name, SWEEP_SEED)
            train_model(m, train_loops, lr)
            ev = binding_metrics(m, val_loops)
            trials.append({"model": name, "lr": lr, "epochs": EPOCHS,
                           "seed": SWEEP_SEED,
                           "val_first_query_binary_accuracy":
                               ev["first_query_binary_accuracy"],
                           "val_first_query_paired_both_correct":
                               ev["first_query_paired_both_correct"]})
            print(f"[مسبح] {name} lr={lr}: first_q={ev['first_query_binary_accuracy']} "
                  f"paired={ev['first_query_paired_both_correct']}")
    frozen_lr = max(LR_GRID, key=lambda lr: (
        min(t["val_first_query_binary_accuracy"] for t in trials if t["lr"] == lr), -lr))
    return trials, frozen_lr


# ═════════════════════════════════ التشغيل ═════════════════════════════════
def main():
    t_start = time.perf_counter()
    print(f"== {VERSION} ==")

    # 1) توليد البيانات (كل المقاطع؛ المحجوز لا يُلمس إلا بحدود العقد الموثقة)
    segments, owner = generate_all()
    print(f"[بيانات] {{{', '.join(f'{s}: {len(segments[s])}' for s in SEGMENTS)}}}")

    # 2) بوابات ما قبل التجميد على غير المحجوز: 1، 2، 3، 4، 7، 8
    g1 = gate1_input_contract(segments, NON_BLIND)
    g2 = gate2_split_contract(segments, owner)
    g3 = gate3_pair_builders(segments, NON_BLIND)
    g4 = gate4_references(segments, NON_BLIND)
    g7 = gate7_loss_contract(segments["train"])
    g8 = gate8_double_intervention(segments["train"][:256] + segments["val"])
    print(f"[بوابات] 1/2/3/4/7/8 ناجحة — مرجع P0 مستعاد حرفياً "
          f"({g2['p0_golden_digests_match'] and 'digests OK'})")

    # 3) بوابات الآلة على نماذج جديدة: 5 (قناع/لاحقة/بادئة)، 6 (فروق مركزية + 6b)
    fresh = {name: new_model(name, MODEL_SEEDS[0]) for name in LEARNED_MODELS}
    g5 = gate5_causal_mask_exact(fresh)
    if g5 is None:
        return 1
    g5b_fresh = {n: gate5b_future_suffix_invariance(m) for n, m in fresh.items()}
    g5c_fresh = {n: gate5c_prefix_vs_full(m) for n, m in fresh.items()}
    for n in fresh:
        if g5b_fresh[n] > GATE_LOGIT_TOL or g5c_fresh[n] > GATE_LOGIT_TOL:
            print(f"FAIL gate5 fresh {n}: {g5b_fresh[n]}/{g5c_fresh[n]}")
            return 1
    g6 = {}
    for n in LEARNED_MODELS:
        m = fresh[n]
        err_fresh = gate6_finite_difference(m, fd_batches())
        train_batch_steps(m, fd_batches()[1], lr=0.05, epochs=150)
        err_tr = gate6_finite_difference(m, fd_batches())
        g6[n] = {"untrained_worst_abs_error": round(err_fresh, 12),
                 "briefly_trained_worst_abs_error": round(err_tr, 12),
                 "eps": FD_EPS}
        if err_fresh > GATE_GRAD_TOL or err_tr > GATE_GRAD_TOL:
            print(f"FAIL gate6 {n}: {err_fresh:.3e}/{err_tr:.3e}")
            return 1
        print(f"[بوابة 6] {n}: fresh={err_fresh:.2e} trained={err_tr:.2e}")
    g6b = {n: gate6b_aggregated_gradient(fresh[n]) for n in LEARNED_MODELS}
    for n, rec in g6b.items():
        if rec["worst_abs_error"] > GATE_GRAD_TOL:
            print(f"FAIL gate6b {n}: {rec['worst_abs_error']:.3e}")
            return 1
    print("[بوابات] 5/5b/5c ناجحة (جديدة) — و6/6b لكل بنية مع الاختبار السلبي")

    # 4) مسبح الاختيار على التحقق وحده → تجميد الميزانية
    trials, frozen_lr = validation_sweep(segments["train"], segments["val"])
    print(f"[تجميد] Adam lr={frozen_lr}، epochs={EPOCHS}، بذور={MODEL_SEEDS}")

    # 5) التدريب النهائي (كل نموذج × كل بذرة) + تقييم غير المحجوز
    trained_store = {}
    models_out = {}
    key_metrics = ("first_query_binary_accuracy",
                   "first_query_paired_both_correct",
                   "second_query_binary_accuracy",
                   "both_answers_correct", "nll_full_vocab",
                   "nll_binary_normalized")
    for name in LEARNED_MODELS:
        per_seed = {}
        for seed in MODEL_SEEDS:
            m = new_model(name, seed)
            info = train_model(m, segments["train"], frozen_lr)
            trained_store[(name, seed)] = m
            block = {"params": m.params_count(),
                     "budget": {"updates": info["updates"],
                                "train_seconds": info["train_seconds"],
                                "final_train_nll": info["final_train_nll"],
                                "stable": info["stable"],
                                "curve": info["curve"]},
                     "val": binding_metrics(m, segments["val"]),
                     "dev_holdout": binding_metrics(m, segments["dev_holdout"]),
                     "p0_probe_in": binding_metrics(m, segments["p0_probe_in"]),
                     "p0_probe_ood_length16": binding_metrics(
                         m, segments["p0_probe_ood_length16"]),
                     "free_rollout_val": free_rollout(m, segments["val"]),
                     "free_rollout_p0_probe_in": free_rollout(
                         m, segments["p0_probe_in"])}
            per_seed[f"seed_{seed}"] = block
            print(f"[تدريب] {name} seed={seed}: "
                  f"val.first_q={block['val']['first_query_binary_accuracy']} "
                  f"paired={block['val']['first_query_paired_both_correct']} "
                  f"({info['train_seconds']}s, {m.params_count()} معلمة)")
        # متوسطات عبر البذور للمقاييس الرئيسية على val/dev
        mean_block = {}
        for seg in ("val", "dev_holdout"):
            for met in key_metrics:
                vals = [per_seed[f"seed_{s}"][seg][met] for s in MODEL_SEEDS]
                mean_block[f"{seg}_{met}"] = {
                    "mean": round(float(np.mean(vals)), 4),
                    "std": round(float(np.std(vals)), 4),
                    "per_seed": vals}
        per_seed["mean_across_seeds"] = mean_block
        models_out[name] = {"spec": SPECS[name], "seeds": per_seed,
                            "params_same_all_seeds": len({
                                per_seed[f"seed_{s}"]["params"]
                                for s in MODEL_SEEDS}) == 1}

    # 6) بوابات على النماذج المدربة (بذرة 42) + بوابة 9
    g5b_trained = {n: gate5b_future_suffix_invariance(trained_store[(n, MODEL_SEEDS[0])])
                   for n in LEARNED_MODELS}
    g5c_trained = {n: gate5c_prefix_vs_full(trained_store[(n, MODEL_SEEDS[0])])
                   for n in LEARNED_MODELS}
    for n in LEARNED_MODELS:
        if g5b_trained[n] > GATE_LOGIT_TOL or g5c_trained[n] > GATE_LOGIT_TOL:
            print(f"FAIL gate5 trained {n}: {g5b_trained[n]}/{g5c_trained[n]}")
            return 1
    print("[بوابات] 5b/5c ناجحة على النماذج المدربة (بذرة 42)")
    seed42_store = {n: trained_store[(n, MODEL_SEEDS[0])] for n in LEARNED_MODELS}
    g9 = gate9_measurement_integrity(seed42_store, segments["val"])
    print("[بوابة 9] التطبيع مستقل لكل مجموعة والاختبار السلبي يشتعل عند اللزوم")

    # 7) التدخلات المقاسة (بذرة 42، val — قبل التجميد؛ النهائي بعد التجميد)
    interventions = {n: intervention_eval(seed42_store[n], segments["val"])
                     for n in LEARNED_MODELS}
    for n in LEARNED_MODELS:
        r = interventions[n]
        print(f"[تدخل] {n}: base_q2={r['baseline_q2_binary_correct']} "
              f"mem={r['modes']['memory_only']['q2_binary_accuracy']} "
              f"ans={r['modes']['prior_answer_only']['q2_binary_accuracy']} "
              f"both={r['modes']['both']['q2_binary_accuracy']}")
    attention_maps = {n: binding_metrics(seed42_store[n], segments["val"],
                                         capture_attention=True)
                      .get("attention_at_query_positions", {}) for n in LEARNED_MODELS}

    # 8) ▼▼ التجميد ▼▼ — كل ما سبق جرى قبل أي لمسة مقيسة للمحجوز؛
    #    فحوص التسميات البنيوية للمحجوز تُنفَّذ الآن فقط، ثم المقياس.
    print("[▼ التجميد ▼] فتح المقاطع المحجوزة: التأكيدية أولاً ثم النهائية")
    g1_blind = gate1_input_contract(segments, BLIND)
    g2_blind_balance = _balance_check(segments, BLIND)
    g3_blind = gate3_pair_builders(segments, BLIND)
    g4_blind = gate4_references(segments, BLIND)

    # (تصحيح v1.1 — التدقيق المستقل): الترتيب الحرفي العالمي — تكتمل كل قراءات
    # التأكيدية (كل النماذج × البذور) قبل أي قراءة نهائية. في v1.0 كانت القراءتان
    # تتداخلان داخل كل (نموذج، بذرة) بلا أي أثر عددي (التقييم لاخزيّ) — موثق في
    # archive/v1.0 وسجل الجولة v0.19.
    confirm_out, final_out = {}, {}
    for name in LEARNED_MODELS:
        c_ps = {}
        for seed in MODEL_SEEDS:
            m = trained_store[(name, seed)]
            c_ps[f"seed_{seed}"] = {
                "confirm_in": binding_metrics(m, segments["confirm_in"]),
                "confirm_ood16": binding_metrics(m, segments["confirm_ood16"]),
                "free_rollout_confirm_in": free_rollout(m, segments["confirm_in"])}
        mean_block = {}
        for seg in ("confirm_in", "confirm_ood16"):
            for met in key_metrics:
                vals = [c_ps[f"seed_{s}"][seg][met] for s in MODEL_SEEDS]
                mean_block[f"{seg}_{met}"] = {
                    "mean": round(float(np.mean(vals)), 4),
                    "std": round(float(np.std(vals)), 4),
                    "per_seed": vals}
        c_ps["mean_across_seeds"] = mean_block
        confirm_out[name] = c_ps
        fv = c_ps["mean_across_seeds"]
        print(f"[تأكيدي] {name}: confirm_in.first_q="
              f"{fv['confirm_in_first_query_binary_accuracy']['mean']} "
              f"paired={fv['confirm_in_first_query_paired_both_correct']['mean']} "
              f"confirm_ood16.first_q="
              f"{fv['confirm_ood16_first_query_binary_accuracy']['mean']}")
    print("[▼ التأكيدية اكتملت لكل النماذج والبذور ▼] تبدأ قراءة النهائية الآن")
    for name in LEARNED_MODELS:
        f_ps = {}
        for seed in MODEL_SEEDS:
            m = trained_store[(name, seed)]
            f_ps[f"seed_{seed}"] = {
                "final_in": binding_metrics(m, segments["final_in"]),
                "final_ood16": binding_metrics(m, segments["final_ood16"]),
                "free_rollout_final_in": free_rollout(m, segments["final_in"])}
        mean_block = {}
        for seg in ("final_in", "final_ood16"):
            for met in key_metrics:
                vals = [f_ps[f"seed_{s}"][seg][met] for s in MODEL_SEEDS]
                mean_block[f"{seg}_{met}"] = {
                    "mean": round(float(np.mean(vals)), 4),
                    "std": round(float(np.std(vals)), 4),
                    "per_seed": vals}
        f_ps["mean_across_seeds"] = mean_block
        final_out[name] = f_ps
        ff = f_ps["mean_across_seeds"]
        print(f"[نهائي] {name}: final_in.first_q="
              f"{ff['final_in_first_query_binary_accuracy']['mean']} "
              f"paired={ff['final_in_first_query_paired_both_correct']['mean']} "
              f"final_ood16.first_q="
              f"{ff['final_ood16_first_query_binary_accuracy']['mean']}")
    # قراءة عاملية وصفية (عمق × ترميز موضعي) من المتوسطات المسجلة فقط — بلا تجربة جديدة
    def _fmean(model):
        return final_out[model]["mean_across_seeds"][
            "final_in_first_query_binary_accuracy"]["mean"]
    fr_cells = {"one_layer_with_pe": _fmean("causal_self_attention_1h"),
                "one_layer_no_pe": _fmean("no_position"),
                "two_layers_with_pe": _fmean("causal_self_attention_2h"),
                "two_layers_no_pe": _fmean("no_position_2h")}
    factorial_reading = {
        "basis": "متوسطات دقة السؤال الأول على final_in عبر 3 بذور — وصفية لا "
                 "استدلالية، مشتقة حسابياً من النتائج المسجلة بلا تشغيل جديد",
        "cells": fr_cells,
        "depth_effect_with_pe": round(
            fr_cells["two_layers_with_pe"] - fr_cells["one_layer_with_pe"], 4),
        "depth_effect_without_pe": round(
            fr_cells["two_layers_no_pe"] - fr_cells["one_layer_no_pe"], 4),
        "pe_removal_effect_one_layer": round(
            fr_cells["one_layer_no_pe"] - fr_cells["one_layer_with_pe"], 4),
        "pe_removal_effect_two_layers": round(
            fr_cells["two_layers_no_pe"] - fr_cells["two_layers_with_pe"], 4),
        "descriptive_interaction": round(
            (fr_cells["two_layers_no_pe"] - fr_cells["one_layer_no_pe"])
            - (fr_cells["two_layers_with_pe"] - fr_cells["one_layer_with_pe"]), 4),
        "confound_note": "إضافة طبقة ترفع المعلمات (753→945) فلا تعزل أثر العمق "
                         "سببياً عن السعة؛ إزالة PE داخل العمق نفسه تحافظ على عدد "
                         "المعلمات — مقارنة أنظف لعامل الترميز الموضعي",
        "variance_note": "3 بذور على مجموعة نهائية مشتركة (لا أمثلة مستقلة بعد "
                         "الجمع عبر البذور)؛ التشتت كبير في الخلايا ثنائية الطبقة "
                         "ولا تُستخدم اختبارات دلالة",
    }
    interventions_final = {n: intervention_eval(seed42_store[n], segments["final_in"])
                           for n in LEARNED_MODELS}
    attention_maps_final = {
        n: binding_metrics(seed42_store[n], segments["final_in"],
                           capture_attention=True)
        .get("attention_at_query_positions", {}) for n in LEARNED_MODELS}

    # 9) التجميع والكتابة
    src_path = Path(__file__).resolve()
    results = {
        "simulator_version": VERSION,
        "stage": "P1 — تنفيذ تجارب التدريب وعزل التبعية وإظهار كل النتائج "
                 "(بوابة الانتقال §8 من عقد CH9). v1.1: تصحيحات توثيقية بعد "
                 "التدقيق المستقل (وصف القارئ الثابت، ترتيب القراءة، عدّ "
                 "التدريبات، قراءة عاملية وصفية) بلا أي تغيير عددي",
        "protocol": {
            "task": "ربط تركيبي: سجلان KEY VALUE SEP + حشو + سؤالان بإجابتيهما؛ "
                    "القيمتان مستقلتان (16 صنفاً بنيوياً متوازناً لكل عائلة حشو)",
            "vocabulary": {"size": V,
                           "atomic": ["K_A", "K_B", "V0", "V1", "SEP", "Q_A",
                                      "Q_B", "ANS0", "ANS1"],
                           "fillers": f"F0..F{N_FILLER - 1} (مجموعة منفصلة بلا إشارات فئات)",
                           "note": "لا رمز حقيقة ذرّية — كل مفتاح وقيمته رمزان منفصلان"},
            "loop_structure": "سجلان (بترتيب متبدَّل متوازن) + حشو {2,4,8,16} + "
                              "[سؤال، جواب، سؤال، جواب] بترتيب متبدَّل متوازن",
            "supervised_positions_per_loop": 2,
            "loss_mask": "NLL عند موضعي السؤال فقط (تنبؤ الجواب التالي)؛ بوابة 7 "
                         "تتحقق أن سياق السؤال الأول خالٍ من رموز الأجوبة، وأن ظهور "
                         "جواب سابق مطابق لهدف السؤال الثاني (أصناف 00/11) قناة "
                         "معلنة تفحصها التدخلات",
            "primary_metrics": "دقة السؤال الأول + paired_both_correct على أزواج "
                               "01/10 متطابقة الـBag — مؤشرا الربط الأساسيان (§6)",
            "output_policy": "softmax على المفردة كاملة (33)؛ تُعرض دقة/احتمالات "
                             "ثنائية منمّطة على {ANS0,ANS1} بجوار كاملة المفردة "
                             "وNLL بنوعيه — لا يُساوَى المقياسان",
            "optimizer": "Adam (full-batch موزون عبر كتل الأطوال {2,4,8}: "
                         "L = Σ nᵢ·Lᵢ/Σ nᵢ وتدرج Σ nᵢ·∇Lᵢ/Σ nᵢ — إرث تصحيح CH-8 "
                         "5b)؛ تحديث واحد لكل عصر",
            "frozen_budget": {"lr": frozen_lr, "epochs": EPOCHS,
                              "selection_rule": "مرشح كل lr = min عبر المرسيات "
                                                "الثلاث على دقة السؤال الأول بval؛ "
                                                "الأعلى يفوز؛ التعادل → الأصغر",
                              "grid": list(LR_GRID),
                              "anchors": list(SWEEP_ANCHORS),
                              "sweep_seed": SWEEP_SEED,
                              "selected_on": "validation وحده"},
            "seeds": {"model_init": list(MODEL_SEEDS),
                      "data": {"p0": P0_SEED, "confirm": CONFIRM_SEED,
                               "final": FINAL_SEED}},
            "splits": {s: len(segments[s]) for s in SEGMENTS},
            "quarantine": {
                "policy": "p0_probe_* اطّلع عليها P0 فلا تُعد اختباراً أعمى؛ "
                          "confirm_* (بذرة 9204) وfinal_* (بذرة 9203) محجوزتان: "
                          "لا مقياس نموذج ولا فحص تسميات قبل التجميد",
                "pre_freeze_touch_boundary": "قبل التجميد لم تُلمس المقاطع "
                                             "المحجوزة إلا بحدود التوليد الإنشائية "
                                             "(دفاتر تفرد العائلات/الحلقات/البادئات "
                                             "المشتركة) — بلا أي إحصاء تسميات ولا "
                                             "تقييم نموذج؛ بوابات 1/2(توازن)/3/4 "
                                             "عليها نُفِّذت بعد التجميد وقُرِئت "
                                             "التأكيدية قبل النهائية",
                "stage_order": ["generate_all", "gates_1_2_3_4_7_8_non_blind",
                                "gates_5_6_fresh", "lr_sweep_on_val", "FREEZE_BUDGET",
                                "final_training_val_dev_probes", "gates_trained_5_9",
                                "interventions_val", "FREEZE_MODELS",
                                "blind_structural_gates",
                                "confirmatory_read_all_models_all_seeds",
                                "final_read_all_models_all_seeds",
                                "post_freeze_interventions_attention_final_in",
                                "write_results"],
                "v1_0_read_order_note": "في تشغيل v1.0 (run c5f901ca7f8218fc، "
                                        "المؤرشف في archive/v1.0) كانت قراءة "
                                        "المحجوز تتداخل: confirm ثم final داخل كل "
                                        "(نموذج، بذرة) — بلا أي اختيار أو تحديث "
                                        "بعد التجميد؛ أُعيد هيكلتها في v1.1 إلى "
                                        "الترتيب العالمي المعلن والأرقام متطابقة",
                "families_disjoint_from_p0": True},
            "training_runs": {
                "full_training_calls": len(SWEEP_ANCHORS) * len(LR_GRID)
                                       + len(LEARNED_MODELS) * len(MODEL_SEEDS),
                "sweep_calls": len(SWEEP_ANCHORS) * len(LR_GRID),
                "final_calls": len(LEARNED_MODELS) * len(MODEL_SEEDS),
                "duplicated_between_sweep_and_final": sum(
                    1 for a in SWEEP_ANCHORS if SWEEP_SEED in MODEL_SEEDS),
                "unique_configurations": len(SWEEP_ANCHORS) * len(LR_GRID)
                                         + len(LEARNED_MODELS) * len(MODEL_SEEDS)
                                         - sum(1 for a in SWEEP_ANCHORS
                                               if SWEEP_SEED in MODEL_SEEDS),
                "note": "استدعاءات train_model الكاملة (1200 عصراً)؛ لا تشمل "
                        "تدريبات البوابات القصيرة (150 خطوة). التكوينات المكررة: "
                        "المرسيات الثلاث ببذرة المسبح عند lr المجمد تُدرَّب مرتين "
                        "(مسبح ثم نهائي) بحتمية متطابقة — تصحيح عدّ v1.1"},
            "corrections_v1_1": {
                "trigger": "التدقيق العلمي والبرمجي المستقل لـ v0.18 "
                           "(content/chapter-09/review-v1/) — قرار GO مشروط",
                "items": [
                    "وصف fixed_query_prefix_reader: ليس محايد الترتيب بنيوياً؛ PE "
                    "يدخل المفاتيح والقيم والدرجات، ونتيجة 0.750 مقيسة لا سقف بنيوي "
                    "(اختبار المراجع المضاد: فرق لوغيتات 6.8e-05 على زوج متطابق "
                    "الـBag، و3.5e-18 عند تصفير PE)",
                    "ترتيب قراءة المحجوز صار حرفياً عالمياً: كل التأكيدية ثم كل "
                    "النهائية (v1.0: تداخل لكل نموذج/بذرة بلا أثر عددي)",
                    "توثيق عدد التدريبات: 30 استدعاء كاملاً / 27 تكويناً فريداً",
                    "قراءة عاملية وصفية (عمق × ترميز موضعي) من المتوسطات المسجلة "
                    "فقط — بلا تشغيل جديد وبلا ادعاء سببي",
                ],
                "numerical_identity": "أعيد التشغيل تحت البروتوكول المجمد نفسه؛ "
                                      "تطابق الأرقام العلمية مع v1.0 المؤرشفة "
                                      "يُتحقق منه بمقارنة الأرشيف بعد التشغيل",
            },
            "model_configs": SPECS,
            "parameter_counts": {n: trained_store[(n, MODEL_SEEDS[0])].params_count()
                                 for n in LEARNED_MODELS},
            "references": "oracle_structural (سقف=1 عبر مسار التقييم نفسه)؛ "
                          "bag ceilings محسوبة (Q1=0.750، مزدوج الأزواج=0)؛ "
                          "Q2 ceiling يُسجَّل كما حُسب؛ local_only مقيس لا مفترض",
            "attention_maps": "قرينة لا إسناد سببي؛ تُجمَّع لكل (طبقة، مفتاح، موضع) "
                              "بنسب مجموعها 1.0 قبل التقريب (بوابة 9) مع اختبار سلبي",
            "stance": "لا افتراض تفوق ولا افتراض فشل؛ النتائج كما تقيس",
        },
        "gates": {
            "1_input_contract": {"non_blind": g1, "blind_post_freeze": g1_blind},
            "2_split_contract": {**g2, "balance_blind_post_freeze": g2_blind_balance},
            "3_pair_builders": {"non_blind": g3, "blind_post_freeze": g3_blind},
            "4_references": {"non_blind": g4, "blind_post_freeze": g4_blind},
            "5_causal_mask": {"mask_exact": g5,
                              "future_suffix_invariance": {
                                  "fresh": g5b_fresh, "trained_seed42": g5b_trained,
                                  "tolerance": GATE_LOGIT_TOL},
                              "prefix_vs_full": {
                                  "fresh": g5c_fresh, "trained_seed42": g5c_trained,
                                  "tolerance": GATE_LOGIT_TOL}},
            "6_finite_difference_gradient": g6,
            "6b_aggregated_gradient": g6b,
            "7_loss_contract": g7,
            "8_double_intervention": g8,
            "9_measurement_integrity": g9,
            "10_provenance": "بصمات المصدر/البيانات/البذور/البيئة أدناه؛ "
                             "test_chapter9.py يصنّف الفحوص environment/gate-diagnostic/"
                             "science مع تسامحات معلنة وبوابة Mutation",
            "11_prose_discipline": {
                "mechanism": "tools/test_chapter9.py — بوابة النثر على ar.md/en.md: "
                             "كل أرقام الجداول مشتقة من results.json، مصطلحات "
                             "إلزامية وممنوعة، وتكافؤ أرقام الجداول ثنائي اللغة",
                "scope": "تُنفَّذ في جولة التأليف (P2) بعد اعتماد المراجعة "
                         "المستقلة للاتجاه العلمي"},
            "12_no_journey_merge": "لم يُعدَّل شيء خارج content/chapter-09؛ لا محطات "
                                   "ولا سجل ولا أطلس (الرحلة 36 محطة/7 فصول كما هي)",
        },
        "experiments": {
            "selection_sweep": {"trials": trials, "frozen_lr": frozen_lr},
            "models_non_blind": models_out,
            "interventions_val_seed42": interventions,
            "interventions_final_in_seed42_post_freeze": interventions_final,
            "attention_maps_val_seed42": attention_maps,
            "attention_maps_final_in_seed42_post_freeze": attention_maps_final,
            "factorial_reading_descriptive": factorial_reading,
            "confirmatory_post_freeze": confirm_out,
            "final_test_post_freeze": final_out,
        },
        "provenance": {
            "dataset_sha256": {s: seg_digest(segments[s]) for s in SEGMENTS},
            "p0_golden_file": str(P0_GOLDEN.relative_to(ROOT)),
            "p0_golden_match": g2["p0_golden_digests_match"],
            "source_sha256": hashlib.sha256(src_path.read_bytes()).hexdigest(),
            "runtime_versions": {"python": platform.python_version(),
                                 "numpy": np.__version__},
            "config_fingerprint": _fp({
                "version": VERSION, "dims": [D_MODEL, DK], "pe_len": PE_LEN,
                "seeds": sorted(MODEL_SEEDS), "p0_seed": P0_SEED,
                "confirm_seed": CONFIRM_SEED, "final_seed": FINAL_SEED,
                "epochs": EPOCHS, "sweep": list(LR_GRID), "vocab": V,
                "splits": {s: len(segments[s]) for s in SEGMENTS}}),
        },
        "total_seconds": round(time.perf_counter() - t_start, 2),
    }
    results_text = json.dumps(results, ensure_ascii=False, indent=1) + "\n"
    entry = {
        "simulator_version": VERSION,
        "source_sha256": results["provenance"]["source_sha256"],
        "config_fingerprint": results["provenance"]["config_fingerprint"],
        "dataset_sha256": results["provenance"]["dataset_sha256"],
        "frozen_budget": results["protocol"]["frozen_budget"],
        "python": platform.python_version(),
        "numpy": np.__version__,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_seconds": results["total_seconds"],
        "note": "سجل تشغيل ملحق (append-only): يوثق تسلسل التشغيلات وسلاسل البصمات؛ "
                "ولا يثبت بذاته انعدام وصول سابق — حد معلن",
    }
    run_id = hashlib.sha256((results_text +
                             json.dumps(entry, sort_keys=True, ensure_ascii=False)
                             ).encode("utf-8")).hexdigest()[:16]
    results["run_log"] = {"path": "content/chapter-09/runs.log", "run_id": run_id}
    final_text = json.dumps(results, ensure_ascii=False, indent=1) + "\n"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(final_text, encoding="utf-8", newline="\n")
    entry["run_id"] = run_id
    entry["results_sha256"] = hashlib.sha256(final_text.encode("utf-8")).hexdigest()
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(RUN_LOG, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"run_log: {run_id}")
    try:
        shown = OUT.relative_to(ROOT)
    except ValueError:
        shown = OUT
    print(f"written: {shown} | {results['total_seconds']}s")
    return 0


def _fp(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False)
                          .encode("utf-8")).hexdigest()[:16]


if __name__ == "__main__":
    sys.exit(main())
