#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
simulate_chapter8.py — الفصل الثامن: من قارئ استعلام واحد إلى Causal Self-Attention (v1.0)

ينفّذ عقد WAKEEL-2-CH8-CAUSAL-SELF-ATTENTION حرفياً:

* مهمة الاسترجاع المزدوجة: مفتاحان A/B وقيمتان مستقلتان ثنائيتان 0/1
  (لا يُفترض تعاكس القيمتين)، سجلات ذاكرة ذرّية M_A0/M_A1/M_B0/M_B1،
  سؤالان Q_A/Q_B وإجابتان ضمن الحلقة نفسها، حشو من مجموعة منفصلة (24 رمزاً)
  بلا إشارات فئات. أطوال الحشو في التدريب {2,4,8} بمزيج أطوال؛ الطول 16
  محجوز خارج كل أطوال التطوير ويُقرأ في الاختبار النهائي فقط.
* الفصل على مستوى «عائلة الحلقة» = مسار الحشو (كل تحويلاتها: تبديل ترتيب
  الأسئلة، تبديل ترتيب السجلات، قلب قيمة — تبقى في المقطع نفسه)، مع منع
  تكرار الحلقات الكاملة ومنع تطابق البادئات الكاملة (حتى موضع أي سؤال)
  بين المقاطع، وموازنة تامة 00/01/10/11 وترتيبي السؤال وترتيب السجلات
  داخل كل كتلة (مقطع × طول).
* بوابات العقد الاثنتا عشرة تُنفَّذ قبل تفسير أي رقم نموذجي؛ أي فشل يوقف
  المحاكي بلا كتابة نتائج (البوابة 12 = test_chapter8.py على النصين).
* النماذج: local_only، fixed_query_prefix_reader (قارئ الاستعلام الواحد
  المُعاد تطبيقه لكل موضع — ليس مستحيلاً نظرياً)، causal_self_attention_1h
  (رأس واحد/طبقة واحدة، تمثيل موضعي جيبي حتمي، Residual معلن قبل النتائج)،
  oracle_memory_lookup سقفاً منفصلاً عبر مسار التقييم نفسه، وعزلتا
  no_position وno_query_dependency، ومقارنة منفصلة معلنة لتضمين موضعي
  متعلم (بتحفّظ مواضع غير مدربة).
* الانضباط: ميزانية تحسين واحدة مجمّدة لكل النماذج تُختار آلياً بقاعدة
  معلنة مسبقاً من مسبح مسجل على التحقق وحده (بوابة validation_selection)؛
  تقييم تطويري (dev holdout) يُبلَّغ ولا يُستخدم للاختيار؛ الاختبار النهائي
  (داخل-التوزيع + طول-16 المحجوز) لا يُقرأ إلا آخر شيء بعد نجاح كل البوابات.
* المشتقات تحليلية عبر شبكة انتشار عكسي مصغّرة (float64)، ويُلزَم فحص
  فروق مركزية لكل عنصر في كل مجموعة معلمات لكل صنف نموذج، على دفعات
  بأقنعة مختلفة، قبل التدريب وبعده جزئياً.

ملاحظة: عينات بوابات التدرج/القناع تركيبية ببذور خاصة ولا تنتمي إلى أي
مقطع تقييم؛ مجموعات التقييم الخمسة هي المولَّدة في generate_datasets فقط.

لا افتراض تفوق لأي معمارية؛ كل رقم في هذا الملف مقيس هنا ويُكتب في
results.json، ولا يُعلن شيء لم يُحسب.
"""
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_OUTDIR = Path(os.environ.get("WAKEEL_CH8_OUT_DIR",
                              str(ROOT / "content" / "chapter-08")))
OUT = _OUTDIR / "results.json"
RUN_LOG = _OUTDIR / "runs.log"
VERSION = "simulate_chapter8-1.2"

# ────────────────────────────── بروتوكول معلن قبل أي نتيجة ──────────────────────────────
EMB_DIM, DK = 8, 8
PE_MAX_LEN = 22            # أطول حلقة: 2 + 16 + 4 = 22
VOCAB_N_FILLER = 24
V = 4 + 2 + 2 + VOCAB_N_FILLER               # 32 رمزاً
ID_MA0, ID_MA1, ID_MB0, ID_MB1 = 0, 1, 2, 3
ID_QA, ID_QB = 4, 5
ID_ANS0, ID_ANS1 = 6, 7
ID_FILLER0 = 8
ANS_IDS = (ID_ANS0, ID_ANS1)
QUERY_IDS = (ID_QA, ID_QB)

DATA_SEED = 8102           # بذرة توليد البيانات — مستقلة عن بذور النموذج
MODEL_SEEDS = (42, 43, 44)
TRAIN_LENGTHS = {2: 160, 4: 176, 8: 176}     # 512 حلقة تدريب
VAL_LENGTHS = {2: 32, 4: 48, 8: 48}          # 128 تحقق
DEV_LENGTHS = {2: 32, 4: 48, 8: 48}          # 128 تقييم تطويري
TEST_IN_LENGTHS = {2: 16, 4: 16, 8: 32}      # 64 اختبار نهائي داخل-التوزيع
TEST_OOD_LENGTHS = {16: 64}                  # 64 اختبار نهائي خارج-التوزيع (طول محجوز)
EPOCHS = 1200
CURVE_AT = (1, 100, 300, 600, 900, 1200)
SWEEP_GRID = (0.01, 0.03, 0.1)
SWEEP_SEED = 42
FD_EPS = 1e-5
GATE_GRAD_TOL = 1e-6
GATE_LOGIT_TOL = 1e-9
GATE_ZERO_TOL = 1e-12

SEGMENTS = ("train", "val", "dev_holdout", "test_in", "test_ood")


# ═════════════════════════════ شبكة انتشار عكسي مصغّرة (float64) ═════════════════════════════
class Node:
    __slots__ = ("v", "parents", "bw")

    def __init__(self, v, parents=(), bw=None):
        self.v = v
        self.parents = parents
        self.bw = bw            # قائمة (فهرس_الوالد، دالة(g) → تدرج) أو None


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
        if g is None:
            continue
        if not node.bw:
            continue            # ورقة/معلمة: يبقى تدرجها في grads ليجلبها المستدعي
        grads.pop(id(node))     # وسيط: استُهلك تدرجه
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


def mm2(a, b):
    """(n,d)@(d,m) → (n,m)"""
    A, B = a.v, b.v
    return Node(A @ B, (a, b),
                [(0, lambda g: g @ B.T), (1, lambda g: A.T @ g)])


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

    def bw():
        return [(0, lambda g: np.einsum("nts,nsd->ntd", g, B)),
                (1, lambda g: np.einsum("nts,ntd->nsd", g, A))]
    return Node(np.einsum("ntd,nsd->nts", A, B), (a, b), bw())


def einsum_av(a, b):
    """(N,T,S)×(N,S,d) → (N,T,d)"""
    A, B = a.v, b.v

    def bw():
        # dA[t,s] = Σ_d g[t,d] B[s,d] ؛ dB[s,d] = Σ_t A[t,s] g[t,d]
        return [(0, lambda g: np.einsum("ntd,nsd->nts", g, B)),
                (1, lambda g: np.einsum("nts,ntd->nsd", A, g))]
    return Node(np.einsum("nts,nsd->ntd", A, B), (a, b), bw())


def expand_t(a, n, t, s):
    """(N,S) → (N,T,S) بثّاً على محور الاستعلام (المحور الأوسط) —
    عكس البث = الجمع على محور الإدراج."""
    return Node(np.broadcast_to(a.v[:, None, :], (n, t, s)),
                (a,), [(0, lambda g: g.sum(axis=1))])


def emb(tablenode, ids):
    T = tablenode.v

    def bw():
        def f(g):
            gt = np.zeros_like(T)
            np.add.at(gt, ids.ravel(), g.reshape(-1, T.shape[1]))
            return gt
        return [(0, f)]
    return Node(T[ids], (tablenode,), bw())


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
    """softmax آخر محور بعد إخفاء محكم (−inf) قبل التنعيم — وزن المستقبل = 0 بالآلة."""
    z = x.v + get_mask(T)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(axis=-1, keepdims=True)

    def bw():
        def f(g):
            return p * (g - np.sum(g * p, axis=-1, keepdims=True))
        return [(0, f)]
    return Node(p, (x,), bw())


def concat1(a, b):
    n = a.v.shape[-1]
    return Node(np.concatenate([a.v, b.v], axis=-1), (a, b),
                [(0, lambda g: g[..., :n]), (1, lambda g: g[..., n:])])


def nll_pick(logits, rows, flat_pos, targets):
    """−mean log P[target] على المواضع المُشرفة فقط (loss_mask)."""
    L = logits.v
    logp = L - L.max(axis=-1, keepdims=True)
    logp = logp - np.log(np.exp(logp).sum(axis=-1, keepdims=True))
    M = len(flat_pos)
    picked = logp[rows, flat_pos, targets]
    loss = float(-picked.mean())

    def bw():
        def f(g):
            d = np.zeros_like(L)
            sel = L[rows, flat_pos]
            sm = np.exp(sel - sel.max(axis=-1, keepdims=True))
            sm = sm / sm.sum(axis=-1, keepdims=True)
            d[rows, flat_pos] = sm
            d[rows, flat_pos, targets] -= 1.0
            return d * (g.item() / M)
        return [(0, f)]
    return Node(np.asarray(loss), (logits,), bw())


def sinusoidal_pe(max_len, d):
    pe = np.zeros((max_len, d))
    pos = np.arange(max_len)[:, None]
    div = np.exp(np.arange(0, d, 2) * (-np.log(10000.0) / d))
    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div)
    return pe


PE_TABLE = sinusoidal_pe(PE_MAX_LEN, EMB_DIM)


# ═════════════════════════════════ النماذج الموحّدة ═════════════════════════════════
# مواصفات معلنة قبل النتائج:
SPECS = {
    "fixed_query_prefix_reader": {
        "arch": "attention", "pos_mode": "sin", "query_mode": "fixed",
        "out_mode": "concat", "out_dim": 2 * EMB_DIM,
        "desc": "قارئ الاستعلام الواحد (CH-7) مُعاد تطبيقه لكل موضع؛ رأس الإخراج "
                "يرى (ملخص البادئة، رمز السؤال الحالي)؛ استعلام متعلم ثابت"},
    "causal_self_attention_1h": {
        "arch": "attention", "pos_mode": "sin", "query_mode": "state",
        "out_mode": "residual", "out_dim": EMB_DIM,
        "desc": "استعلام مشتق من تمثيل كل موضع؛ رأس واحد طبقة واحدة؛ "
                "logits = WO·(h_t + c_t) + bO — residual معلن"},
    "no_position": {
        "arch": "attention", "pos_mode": "none", "query_mode": "state",
        "out_mode": "residual", "out_dim": EMB_DIM,
        "desc": "عزل الموضع: نفس بنية الأساسية بلا تمثيل موضعي"},
    "no_query_dependency": {
        "arch": "attention", "pos_mode": "sin", "query_mode": "fixed",
        "out_mode": "residual", "out_dim": EMB_DIM,
        "desc": "عزل تبعية الاستعلام للرمز الحالي: نفس بنية الأساسية والاستعلام "
                "مجمّد إلى متجه متعلم واحد لكل المواضع"},
    "csa_learned_absolute_pe": {
        "arch": "attention", "pos_mode": "learned", "query_mode": "state",
        "out_mode": "residual", "out_dim": EMB_DIM,
        "desc": "مقارنة منفصلة معلنة: تضمين موضعي مطلق متعلم — مواضع الأطوال "
                "المحجوزة (16) غير مدربة، فشل الطول 16 فيها متوقع التفسير"},
}
LEARNED_MODELS = tuple(SPECS) + ("local_only",)


def init_params(rng, spec):
    p = {"E": rng.normal(0, 0.1, (V, EMB_DIM))}
    if spec.get("arch") == "attention":
        if spec["pos_mode"] == "learned":
            p["P"] = rng.normal(0, 0.1, (PE_MAX_LEN, EMB_DIM))
        if spec["query_mode"] == "fixed":
            p["q"] = rng.normal(0, 0.1, DK)
        else:
            p["WQ"] = rng.normal(0, 0.1, (EMB_DIM, DK))
        p["WK"] = rng.normal(0, 0.1, (EMB_DIM, DK))
        p["WV"] = rng.normal(0, 0.1, (EMB_DIM, DK))
        p["WO"] = rng.normal(0, 0.1, (spec["out_dim"], V))
    else:  # local
        p["WO"] = rng.normal(0, 0.1, (EMB_DIM, V))
    p["bO"] = np.zeros(V)
    return p


def build_graph(P, X, spec, leaves=None):
    """يرجع (logits Node، A Node أو None). القناع السببي مدمج في التنعيم."""
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
    elif spec["pos_mode"] == "learned":
        H = add(H, emb(nd("P"), np.tile(np.arange(T), (N, 1))))
    K = mm3_2(H, nd("WK"))
    Vv = mm3_2(H, nd("WV"))
    if spec["query_mode"] == "fixed":
        s = scale(mm2v(H, nd("q")), 1.0 / np.sqrt(DK))       # (N,T) درجات مشتركة
        scores = expand_t(s, N, T, T)                        # (N,T,T) ثم تنعيم بادئي
    else:
        Q = mm3_2(H, nd("WQ"))
        scores = scale(einsum_nts(Q, K), 1.0 / np.sqrt(DK))
    A = mask_softmax(scores, T)
    C = einsum_av(A, Vv)
    out = add(H, C) if spec["out_mode"] == "residual" else concat1(C, H)
    logits = add(mm3_2(out, nd("WO")), nd("bO"))
    return logits, A


class WakeelModel:
    """كل النماذج المتعلمة — نفس مسار التدريب والتقييم والفحص."""

    def __init__(self, name, seed):
        self.name = name
        self.seed = seed
        self.spec = SPECS[name] if name in SPECS else {"arch": "local"}
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
        """الخسارة المجمعة الموزونة وتدرجها المُرسل فعلاً إلى المحسّن:
        L = Σ_i n_i·L_i / Σ_i n_i  و  ∇L = Σ_i n_i·∇L_i / Σ_i n_i.
        (تصحيح v1.1 — التدقيق المستقل v0.16: كانت التدرجات تُجمع دون وزن n_i،
        وبوابة الفروق المنتهية داخل الدفعة الواحدة لا تكشف ذلك منطقياً.)"""
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
                    acc[k] += gk * cnt          # وزن تدرج الكتلة بعدد مواضعها
        loss = loss_sum / total
        for k in acc:
            acc[k] /= total
        return loss, acc

    def train_full_batch(self, groups, total, opt):
        """full-batch موزون عبر كتل الأطوال {2,4,8}: تدرجات الكتل تُوزن بأعداد
        مواضعها ثم تحديث Adam واحد لكل عصر (تصحيح v1.1)."""
        loss, grads = self.aggregate_grads(groups, total)
        if not np.isfinite(loss):
            return float("nan")
        opt.step(self.params, grads)
        return loss

    def forward_logits(self, X):
        logits, _ = build_graph(self.params, X, self.spec)
        return logits.v

    def forward_attention(self, X):
        logits, A = build_graph(self.params, X, self.spec)
        return logits.v, (A.v if A is not None else None)

    def next_logits(self, prefix_ids):
        return self.forward_logits([list(prefix_ids)])[0, -1]


def _flat_supervision(qpos, ans):
    rows, flat_pos, targets = [], [], []
    for n in range(len(qpos)):
        for k, t in enumerate(qpos[n]):
            rows.append(n)
            flat_pos.append(t)
            targets.append(ans[n][k])
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


# ── المراجع (سقوف/ضوابط — بلا تعلم) ──
class OracleMemoryLookup:
    """سقف مرجعي منفصل: يعرف مفتاح السؤال ويقرأ سجل الذاكرة الصحيح عبر المسار نفسه."""

    def __init__(self, flip_labels=False):
        self.flip_labels = flip_labels

    def params_count(self):
        return 0

    def next_logits(self, prefix_ids):
        last = prefix_ids[-1]
        rec_ids = (ID_MA0, ID_MA1) if last == ID_QA else \
                  ((ID_MB0, ID_MB1) if last == ID_QB else None)
        if rec_ids is None:
            raise ValueError("المرجع يُستدعى عند موضع سؤال فقط")
        rec = next((t for t in prefix_ids if t in rec_ids), None)
        logits = np.full(V, -30.0)
        if rec is None:
            logits[ID_ANS0] = 0.0
            logits[ID_ANS1] = 0.0          # المهمة غير قابلة للتحديد بلا السجل
            return logits
        ans = ID_ANS1 if rec == rec_ids[1] else ID_ANS0
        if self.flip_labels:
            ans = ID_ANS1 if ans == ID_ANS0 else ID_ANS0
        logits[ans] = 30.0
        return logits

    def forward_logits(self, X):
        out = np.zeros((X.shape[0], X.shape[1], V))
        for i, row in enumerate(X):
            for t in range(X.shape[1]):
                if row[t] in QUERY_IDS:
                    out[i, t] = self.next_logits(list(row[:t + 1]))
        return out

    def forward_attention(self, X):
        return self.forward_logits(X), None


class OracleNoMemory:
    """ضابط بلا ذاكرة: المهمة غير قابلة للتحديد — احتمالات ثنائية متساوية."""

    def params_count(self):
        return 0

    def next_logits(self, prefix_ids):
        logits = np.full(V, -30.0)
        logits[ID_ANS0] = 0.0
        logits[ID_ANS1] = 0.0
        return logits

    def forward_logits(self, X):
        base = np.full(V, -30.0)
        base[ID_ANS0] = 0.0
        base[ID_ANS1] = 0.0
        return np.tile(base, (X.shape[0], X.shape[1], 1))

    def forward_attention(self, X):
        return self.forward_logits(X), None


def new_model(name, seed):
    return WakeelModel(name, seed)


# ═════════════════════════════════ توليد البيانات ═════════════════════════════════
# 16 صنفاً بنيوياً: (قيمة A، قيمة B، ترتيب السؤالين، ترتيب السجلين)
STRUCT_CLASSES = [(va, vb, qo, ro) for va in (0, 1) for vb in (0, 1)
                  for qo in (0, 1) for ro in (0, 1)]


def build_loop(length, cls, filler_path):
    va, vb, qo, ro = cls
    recs = [(ID_MA0 + va, ID_MB0 + vb)]
    if ro == 1:
        recs = [recs[0][::-1]]
    recs = recs[0]
    queries = [(ID_QA, va), (ID_QB, vb)]
    if qo == 1:
        queries = queries[::-1]
    tokens = list(recs) + [ID_FILLER0 + int(f) for f in filler_path]
    qpos, qmeta = [], []
    for qid, val in queries:
        qpos.append(len(tokens))
        tokens.append(int(qid))
        qmeta.append({"qid": int(qid), "key": "A" if qid == ID_QA else "B",
                      "answer_token": ID_ANS0 + int(val), "answer_binary": int(val)})
        tokens.append(ID_ANS0 + int(val))
    return {"tokens": tokens, "length": length, "cls": [int(c) for c in cls],
            "family": (length, tuple(int(f) for f in filler_path)),
            "qpos": qpos, "qmeta": qmeta}


def flip_va_loop(loop):
    cls = list(loop["cls"])
    cls[0] ^= 1
    return build_loop(loop["length"], cls, loop["family"][1])


def swap_qorder_loop(loop):
    cls = list(loop["cls"])
    cls[2] ^= 1
    return build_loop(loop["length"], cls, loop["family"][1])


def strip_memory_loop(loop):
    L = loop["length"]
    return {"tokens": loop["tokens"][2:], "length": L, "cls": loop["cls"],
            "family": loop["family"],
            "qpos": [p - 2 for p in loop["qpos"]],
            "qmeta": [dict(m) for m in loop["qmeta"]]}


def generate_datasets():
    """توليد المقاطع الخمسة بدفاتر العائلات/الحلقات/البادئات وبوابة التوازن."""
    rng = np.random.default_rng(DATA_SEED)
    segments = {s: [] for s in SEGMENTS}
    fam_ledger = {}
    loop_ledger = {}
    prefix_ledger = {}
    plan = [("train", TRAIN_LENGTHS), ("val", VAL_LENGTHS), ("dev_holdout", DEV_LENGTHS),
            ("test_in", TEST_IN_LENGTHS), ("test_ood", TEST_OOD_LENGTHS)]
    for seg, lengths in plan:
        for Ln, count in sorted(lengths.items()):
            if count % 16 != 0:
                raise RuntimeError("كل كتلة (مقطع×طول) من مضاعفات 16 للتوازن التام")
            for _ in range(count // 16):
                for idx in rng.permutation(16):
                    cls = STRUCT_CLASSES[int(idx)]
                    for _try in range(1_000_000):
                        path = tuple(int(x) for x in rng.integers(0, VOCAB_N_FILLER, size=Ln))
                        fam = (Ln, path)
                        if fam in fam_ledger:
                            continue
                        fam_ledger[fam] = seg
                        break
                    else:
                        raise RuntimeError(f"فضاء العائلات مستنزف L={Ln}")
                    loop = build_loop(Ln, tuple(int(c) for c in cls), path)
                    key = tuple(loop["tokens"])
                    if key in loop_ledger:
                        raise RuntimeError("حلقة مكررة عبر المقاطع")
                    loop_ledger[key] = seg
                    for p in loop["qpos"]:
                        pk = tuple(loop["tokens"][:p + 1])
                        if prefix_ledger.get(pk, seg) != seg:
                            raise RuntimeError("بادئة كاملة متطابقة بين مقطعين")
                        prefix_ledger[pk] = seg
                    segments[seg].append(loop)
    # توازن تام داخل كل كتلة (مقطع × طول)
    for seg, loops in segments.items():
        by_len = {}
        for lp in loops:
            by_len.setdefault(lp["length"], []).append(lp)
        for Ln, lps in sorted(by_len.items()):
            counts = {c: 0 for c in STRUCT_CLASSES}
            for lp in lps:
                counts[tuple(lp["cls"])] += 1
            if not all(v == len(lps) // 16 for v in counts.values()):
                raise RuntimeError(f"توازن مكسور {seg} L={Ln}")
            if not all(lp["length"] == Ln and lp["family"][0] == Ln for lp in lps):
                raise RuntimeError(f"طول حلقة لا يطابق كتلته {seg} L={Ln}")
    fams = {s: set() for s in SEGMENTS}
    for fam, seg in fam_ledger.items():
        fams[seg].add(fam)
    for i, a in enumerate(SEGMENTS):
        for b in SEGMENTS[i + 1:]:
            if fams[a] & fams[b]:
                raise RuntimeError(f"تداخل عائلات {a}|{b}")
    return segments, fam_ledger


def seg_hash(loops):
    payload = json.dumps([[lp["tokens"], lp["cls"], lp["qpos"]] for lp in loops],
                         sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ═════════════════════════════════ مسار التقييم الواحد ═════════════════════════════════
def _binary_pair(logits):
    two = logits[[ID_ANS0, ID_ANS1]]
    z = two - two.max()
    e = np.exp(z)
    return e / e.sum()


def evaluate(model, loops, capture_attention=False):
    """مسار تقييم واحد لكل النماذج (المراجع ضمناً) — teacher forcing على الحلقات الكاملة
    (تكافؤ البادئة مُثبت بالبوابة 3)."""
    n = bin_ok = full_ok = 0
    nll_full_sum = nll_bin_sum = 0.0
    per_q = {k: {"n": 0, "bin_ok": 0, "full_ok": 0, "nll_full": 0.0, "nll_bin": 0.0}
             for k in ("A", "B")}
    att = {"A": _zero_att(), "B": _zero_att(), "n": {"A": 0, "B": 0}} \
        if capture_attention else None
    by_len = {}
    for lp in loops:
        by_len.setdefault(lp["length"], []).append(lp)
    for Ln, lps in sorted(by_len.items()):
        X = np.array([lp["tokens"] for lp in lps])
        logits, A = model.forward_attention(X)
        for i, lp in enumerate(lps):
            for k, t in enumerate(lp["qpos"]):
                meta = lp["qmeta"][k]
                lv = logits[i, t]
                z = lv - lv.max()
                p_full = np.exp(z)
                p_full = p_full / p_full.sum()
                ans_tok = meta["answer_token"]
                ans_bin = meta["answer_binary"]
                nlf = -np.log(max(p_full[ans_tok], 1e-300))
                pb = _binary_pair(lv)
                nlb = -np.log(max(pb[ans_bin], 1e-300))
                nll_full_sum += nlf
                nll_bin_sum += nlb
                full_ok += int(int(np.argmax(lv)) == ans_tok)
                bin_ok += int(int(np.argmax(pb)) == ans_bin)
                q = per_q[meta["key"]]
                q["n"] += 1
                q["bin_ok"] += int(int(np.argmax(pb)) == ans_bin)
                q["full_ok"] += int(int(np.argmax(lv)) == ans_tok)
                q["nll_full"] += nlf
                q["nll_bin"] += nlb
                n += 1
                if att is not None and A is not None:
                    _acc_att(att, A[i, t], lp, t, meta)
    out = {"n_loops": len(loops), "n_positions": n,
           "binary_accuracy": round(bin_ok / n, 4),
           "full_vocab_accuracy": round(full_ok / n, 4),
           "nll_full_vocab": round(float(nll_full_sum) / n, 4),
           "nll_binary_normalized": round(float(nll_bin_sum) / n, 4),
           "per_query": {k: {"binary_accuracy": round(v["bin_ok"] / v["n"], 4),
                             "full_vocab_accuracy": round(v["full_ok"] / v["n"], 4),
                             "nll_full_vocab": round(float(v["nll_full"]) / v["n"], 4),
                             "nll_binary_normalized": round(float(v["nll_bin"]) / v["n"], 4)}
                         for k, v in per_q.items()}}
    if att is not None and (att["n"]["A"] + att["n"]["B"]) > 0:
        out["attention_at_query_positions"] = _norm_att(att)
    return out


def _zero_att():
    return {"correct_record": 0.0, "other_record": 0.0, "filler": 0.0,
            "prior_qa": 0.0, "self_query": 0.0}


def _acc_att(att, arow, lp, t, meta):
    """كتلة الانتباه عند موضع سؤال — قرينة فقط؛ الإسناد السببي عبر التدخلات.
    (تصحيح v1.2 — التدقيق المستقل v0.17: عدّاد مستقل لكل نوع سؤال؛ كان العدد
    المشترك يقسم مجموع كل نوع على 2 فيتكون 0.5 بدل 1.0.)"""
    correct = None
    for p in (0, 1):
        tok = lp["tokens"][p]
        if (meta["key"] == "A" and tok in (ID_MA0, ID_MA1)) or \
           (meta["key"] == "B" and tok in (ID_MB0, ID_MB1)):
            correct = p
    z = _zero_att()
    for j in range(t + 1):
        if j in (0, 1):
            z["correct_record" if j == correct else "other_record"] += float(arow[j])
        elif 2 <= j < 2 + lp["length"]:
            z["filler"] += float(arow[j])
        elif j == t:
            z["self_query"] += float(arow[j])
        else:
            z["prior_qa"] += float(arow[j])
    a = att[meta["key"]]
    for k in z:
        a[k] += z[k]
    att["n"][meta["key"]] += 1


def _norm_att(att):
    """تطبيع مستقل لكل نوع سؤال بعدده هو، مع بوابة صريحة على القيم غير المقرَّبة:
    مجموع فئات كل نوع = 1.0 (ضمن دقة الآلة) — استثناء صريح لا يُعطَّل بـ python -O."""
    out = {}
    for k in ("A", "B"):
        n = att["n"][k]
        if n <= 0:
            raise RuntimeError(f"تطبيع الانتباه: لا مواضع للاستعلام {k}")
        total = sum(att[k].values())
        if abs(total - n) > 1e-9 * max(n, 1):
            raise RuntimeError(
                f"تطبيع الانتباه مكسور للنوع {k}: مجموع الفئات {total} ≠ "
                f"عدد المواضع {n} — يرفض بغضّ النظر عن وضع python -O")
        out[k] = {kk: round(vv / n, 4) for kk, vv in att[k].items()}
    return out


def free_rollout(model, loops):
    """توليد حُرّ جشع: الجواب الأول المولَّد يدخل سياق السؤال الثاني (قد يختلف عن teacher forcing)."""
    n = pos1 = pos2 = both = gen1_ans = 0
    for lp in loops:
        t1, t2 = lp["qpos"]
        a1 = lp["qmeta"][0]["answer_token"]
        a2 = lp["qmeta"][1]["answer_token"]
        ctx = list(lp["tokens"][:t1 + 1])
        g1 = int(np.argmax(model.next_logits(ctx)))
        gen1_ans += int(g1 in ANS_IDS)
        ctx2 = ctx + [g1] + list(lp["tokens"][t1 + 2:t2 + 1])
        g2 = int(np.argmax(model.next_logits(ctx2)))
        c1, c2 = g1 == a1, g2 == a2
        pos1 += c1
        pos2 += c2
        both += c1 and c2
        n += 1
    return {"n_loops": n,
            "position1_accuracy": round(pos1 / n, 4),
            "position2_accuracy": round(pos2 / n, 4),
            "both_correct": round(both / n, 4),
            "gen1_is_answer_token_rate": round(gen1_ans / n, 4)}


# ═════════════════════════════ عينات بوابات الآلة (تركيبية ببذور خاصة) ═════════════════════════════
def _gate_batch(lengths_counts, seed):
    rng = np.random.default_rng(seed)
    X, qpos, ans = [], [], []
    for Ln, cnt in sorted(lengths_counts.items()):
        for _ in range(cnt):
            cls = STRUCT_CLASSES[int(rng.integers(0, 16))]
            path = rng.integers(0, VOCAB_N_FILLER, size=Ln)
            lp = build_loop(Ln, cls, tuple(int(f) for f in path))
            X.append(lp["tokens"])
            qpos.append(lp["qpos"])
            ans.append([m["answer_token"] for m in lp["qmeta"]])
    return np.array(X), qpos, ans


def _gate_mixed():
    """دفعتان منتظمتا الشكل (طول 2 و8) لبوابات القناع واللوغيتات."""
    return [_gate_batch({2: 3}, seed=555), _gate_batch({8: 3}, seed=556)]


def _suffix_gate_loops():
    rng = np.random.default_rng(77)
    out = []
    for Ln in (2, 8):
        for _ in range(3):
            cls = STRUCT_CLASSES[int(rng.integers(0, 16))]
            path = rng.integers(0, VOCAB_N_FILLER, size=Ln)
            out.append(build_loop(Ln, cls, tuple(int(f) for f in path)))
    return out


_FD_CACHE = None


def fd_batches():
    """دفعتان بأقنعة سببية مختلفتان (أطوال 2 و8) لفحص الفروق المركزية."""
    global _FD_CACHE
    if _FD_CACHE is None:
        _FD_CACHE = [_gate_batch({2: 4}, seed=901), _gate_batch({8: 4}, seed=902)]
    return _FD_CACHE


# ═════════════════════════════ البوابات 1–11 (12 في test_chapter8.py) ═════════════════════════════
def gate1_causal_mask_exact(models):
    per = {}
    for Xb, _, _ in _gate_mixed():
        for name, m in models.items():
            _, A = m.forward_attention(Xb)
            if A is None:
                continue
            T = A.shape[-1]
            future = max(float(A[i, t, j])
                         for i in range(len(Xb)) for t in range(T)
                         for j in range(t + 1, T))
            row_err = float(np.max(np.abs(A.sum(axis=-1) - 1.0)))
            cur = per.get(name, {"future_weight_max": 0.0,
                                 "row_sum_max_abs_error": 0.0})
            cur["future_weight_max"] = max(cur["future_weight_max"], future)
            cur["row_sum_max_abs_error"] = max(cur["row_sum_max_abs_error"], row_err)
            per[name] = cur
            if future != 0.0 or row_err > GATE_ZERO_TOL:
                print(f"FAIL gate1 {name}: {cur}")
                return None
    return per


def _alt_suffix(lp, t, mode):
    toks = list(lp["tokens"][:t + 1])
    rest = list(lp["tokens"][t + 1:])
    if mode == "flip_answers":
        rest = [ID_ANS1 if x == ID_ANS0 else (ID_ANS0 if x == ID_ANS1 else x)
                for x in rest]
    else:  # حشو بديل بعد t
        rest = [((x - ID_FILLER0 + 1) % VOCAB_N_FILLER) + ID_FILLER0
                if ID_FILLER0 <= x < ID_FILLER0 + VOCAB_N_FILLER else x for x in rest]
    return toks + rest


def gate2_future_suffix_invariance(model):
    """لوغيتات الموضع t متطابقة عند اختلاف ما بعده — بقلب إجابات المستقبل وبحشو بديل."""
    worst = 0.0
    for lp in _suffix_gate_loops():
        for t in (lp["qpos"][0], 2 + lp["length"] // 2):
            L0 = model.forward_logits([lp["tokens"]])[0, t]
            for mode in ("flip_answers", "different_filler"):
                L1 = model.forward_logits([_alt_suffix(lp, t, mode)])[0, t]
                worst = max(worst, float(np.max(np.abs(L0 - L1))))
    return worst


def gate3_prefix_vs_full(model):
    """تنبؤ الموضع t على البادئة وحدها = تنبؤه في الحلقة الكاملة بالقناع السببي."""
    worst = 0.0
    for lp in _suffix_gate_loops():
        full, _ = model.forward_attention(np.array([lp["tokens"]]))
        for t in lp["qpos"]:
            pre = model.next_logits(lp["tokens"][:t + 1])
            worst = max(worst, float(np.max(np.abs(full[0, t] - pre))))
    return worst


def gate4_loss_target_alignment(train_loops):
    checked = 0
    for lp in train_loops:
        if lp["qpos"] != [2 + lp["length"], 4 + lp["length"]]:
            raise RuntimeError(f"بوابة 4: مواضع السؤال لا تطابق البنية: {lp['qpos']}")
        for k, t in enumerate(lp["qpos"]):
            ctx = lp["tokens"][:t + 1]
            tgt = lp["qmeta"][k]["answer_token"]
            if not (ctx[-1] in QUERY_IDS and tgt in ANS_IDS and tgt != ctx[-1]):
                raise RuntimeError("بوابة 4: سياق لا ينتهي بسؤال أو هدف خارج {0,1} "
                                   "أو هدف يساوي رمز السؤال")
            checked += 1
    return {"supervised_positions": checked,
            "contexts_end_at": "Q_A/Q_B فقط",
            "targets_in": "ANS0/ANS1 فقط",
            "answer_never_input_at_query_position": True}


def gate5_finite_difference(model, batches):
    """فروق مركزية لكل عنصر في كل مجموعة معلمات — عينات غير تافهة بأقنعة مختلفة."""
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


def gate5b_aggregated_gradient(model):
    """بوابة التدرج المُجمَّع عبر كتل بأعداد مواضع مُشرفة مختلفة: التدرج الذي
    يصل إلى المحسّن يجب أن يساوي مشتقة الخسارة الموزونة المجمعة
    L = Σ n_i·L_i / Σ n_i — الفروق المنتهية داخل دفعة منفردة لا تكشف عيب
    التجميع (عيب v0.16 الذي أثبته التدقيق المستقل عددياً)."""
    b1 = _gate_batch({2: 3}, seed=903)   # 6 مواضع إشراف
    b2 = _gate_batch({8: 5}, seed=904)   # 10 مواضع إشراف
    groups = [(b1[0], b1[1], b1[2], 6), (b2[0], b2[1], b2[2], 10)]
    total = 16
    _, grads = model.aggregate_grads(groups, total)

    def agg_loss():
        s = 0.0
        for X, qpos, ans, cnt in groups:
            leaves = {k: leaf(v) for k, v in model.params.items()}
            logits, _ = build_graph(None, X, model.spec, leaves=leaves)
            rows, flat_pos, targets = _flat_supervision(qpos, ans)
            s += float(nll_pick(logits, rows, flat_pos, targets).v) * cnt
        return s / total

    worst = 0.0
    for pname, arr in model.params.items():
        garr = grads[pname]
        for idx in np.ndindex(arr.shape):
            old = arr[idx]
            arr[idx] = old + FD_EPS
            lp = agg_loss()
            arr[idx] = old - FD_EPS
            lm = agg_loss()
            arr[idx] = old
            num = (lp - lm) / (2 * FD_EPS)
            worst = max(worst, abs(float(garr[idx]) - num))
    return {"buckets_positions": [6, 10], "total": total,
            "worst_abs_error": round(worst, 12), "eps": FD_EPS,
            "objective": "L = Σ n_i·L_i / Σ n_i وتدرجها الموزون"}


def gate6_counterfactual_pairs(segments):
    """(بيانات) قلب قيمة A يغيّر إجابة A فقط؛ تبديل ترتيب الأسئلة يبدّل المواضع لا السجلات."""
    n = 0
    for seg in SEGMENTS:
        for lp in segments[seg]:
            fv = flip_va_loop(lp)
            if fv["family"] != lp["family"] or \
               [m["key"] for m in fv["qmeta"]] != [m["key"] for m in lp["qmeta"]]:
                raise RuntimeError("بوابة 6: نظير قلب القيمة غيّر العائلة أو مفاتيح الأسئلة")
            for k, m in enumerate(lp["qmeta"]):
                a_o = m["answer_token"]
                a_f = fv["qmeta"][k]["answer_token"]
                if m["key"] == "A":
                    if a_o == a_f:
                        raise RuntimeError("بوابة 6: قلب قيمة A لم يغيّر جواب A")
                else:
                    if a_o != a_f:
                        raise RuntimeError("بوابة 6: قلب قيمة A غيّر جواب B")
            sq = swap_qorder_loop(lp)
            if sq["tokens"][:2 + lp["length"]] != lp["tokens"][:2 + lp["length"]]:
                raise RuntimeError("بوابة 6: تبديل ترتيب الأسئلة غيّر السجلات أو الحشو")
            for k in (0, 1):
                if sq["qmeta"][k]["key"] == lp["qmeta"][k]["key"]:
                    raise RuntimeError("بوابة 6: تبديل الترتيب لم يبدّل مفاتيح المواضع")
            # التبديل يغيّر أيّ موضع يطلب أيّ قيمة — لا قيم السجلات نفسها:
            if {(m["key"], m["answer_binary"]) for m in sq["qmeta"]} != \
               {(m["key"], m["answer_binary"]) for m in lp["qmeta"]}:
                raise RuntimeError("بوابة 6: تبديل الترتيب غيّر أزواج (مفتاح، قيمة)")
            n += 1
    return {"loops_checked": n,
            "value_flip_changes_only_A_answer": True,
            "query_order_swap_changes_positions_not_records": True}


def gate7_dataset_exclusivity(segments, fam_ledger):
    seg_hashes = {s: seg_hash(segments[s]) for s in SEGMENTS}
    sizes = {s: len(segments[s]) for s in SEGMENTS}
    if sizes != {"train": 512, "val": 128, "dev_holdout": 128,
                 "test_in": 64, "test_ood": 64}:
        raise RuntimeError(f"بوابة 7: أحجام المقاطع مخالفة: {sizes}")
    fams = {s: set() for s in SEGMENTS}
    for fam, seg in fam_ledger.items():
        fams[seg].add(fam)
    overlaps = {f"{a}|{b}": len(fams[a] & fams[b])
                for i, a in enumerate(SEGMENTS) for b in SEGMENTS[i + 1:]}
    if any(v != 0 for v in overlaps.values()):
        raise RuntimeError(f"بوابة 7: تداخل عائلات بين المقاطع: {overlaps}")
    return {"segment_sizes": sizes, "segment_sha256": seg_hashes,
            "family_overlap_counts": overlaps,
            "split_policy": "الفصل على مستوى عائلة الحلقة (مسار الحشو) وكل تحويلاتها؛ "
                            "لا حلقة مكررة ولا بادئة كاملة مشتركة بين المقاطع",
            "data_generation_seed": DATA_SEED,
            "balance": "داخل كل كتلة (مقطع×طول): توازن تام للأصناف الستة عشر "
                       "(قيمتا A/B × ترتيب السؤالين × ترتيب السجلين)"}


def gate8_positive_oracle(val, dev, test_in):
    o = OracleMemoryLookup()
    out = {}
    for name, seg in (("val", val), ("dev_holdout", dev), ("test_in", test_in)):
        ev = evaluate(o, seg)
        if ev["binary_accuracy"] != 1.0:
            raise RuntimeError(f"بوابة 8 فشلت على {name}: {ev['binary_accuracy']}")
        out[f"{name}_binary_accuracy"] = ev["binary_accuracy"]
    evf = evaluate(OracleMemoryLookup(flip_labels=True), val)
    if evf["binary_accuracy"] != 0.0:
        raise RuntimeError("بوابة 8: قلب التسميات لم يسقط إلى 0.000")
    out["val_label_flip_binary_accuracy"] = 0.0
    return out


def gate9_no_memory_control(trained_models, val):
    stripped = [strip_memory_loop(lp) for lp in val]
    out = {"oracle_no_memory_binary_accuracy":
           evaluate(OracleNoMemory(), stripped)["binary_accuracy"],
           "n_loops": len(stripped),
           "note": "بحذف سجلي الذاكرة تصبح القيمتان غير محددتين من السياق المرئي "
                   "(مثال متوازن) — المرجع نفسه 0.500، والنماذج تُقاس فعلاً"}
    for name, m in trained_models.items():
        out[name] = evaluate(m, stripped)
    return out


# ═════════════════════════════════ التدريب ═════════════════════════════════
def make_groups(loops):
    by_len = {}
    for lp in loops:
        by_len.setdefault(lp["length"], []).append(lp)
    groups, total = [], 0
    for Ln, lps in sorted(by_len.items()):
        X = np.array([lp["tokens"] for lp in lps])
        qpos = [lp["qpos"] for lp in lps]
        ans = [[m["answer_token"] for m in lp["qmeta"]] for lp in lps]
        cnt = sum(len(q) for q in qpos)
        total += cnt
        groups.append((X, qpos, ans, cnt))
    return groups, total


def train_model(model, train_loops, lr, epochs=EPOCHS):
    groups, total = make_groups(train_loops)
    opt = Adam(list(model.params), lr)
    curve, t0 = [], time.perf_counter()
    final = float("nan")
    for ep in range(1, epochs + 1):
        final = model.train_full_batch(groups, total, opt)
        if not np.isfinite(final):
            final = float("nan")
            break
        if ep in CURVE_AT:
            curve.append({"epoch": ep,
                          "train_nll": round(final, 4) if np.isfinite(final) else None})
    return {"final_train_nll": round(final, 4) if np.isfinite(final) else None,
            "stable": bool(np.isfinite(final)),
            "curve": curve,
            "train_seconds": round(time.perf_counter() - t0, 2)}


def train_batch_steps(model, batch, lr, epochs):
    """خطوات تدريب full-batch على دفعة بوابة تركيبية واحدة (لتغيير المعلمات عن التهيئة)."""
    X, qpos, ans = batch
    total = sum(len(q) for q in qpos)
    opt = Adam(list(model.params), lr)
    for _ in range(epochs):
        model.train_full_batch([(X, qpos, ans, total)], total, opt)


# ═════════════════════════════ التداخلات السببية المقاسة ═════════════════════════════
def counterfactual_value_swap(model, loops):
    """قلب قيمة A: هل يتبع النموذج الإجابة الجديدة عند Q_A؟ وهل يثبت تنبؤ Q_B؟
    القياس بمفتاح السؤال (Q_A/Q_B) لا بموضعه في الحلقة."""
    follow = stable = nA = 0
    dA = dB = 0.0
    n = len(loops)
    for lp in loops:
        fv = flip_va_loop(lp)
        kA = next(k for k, m in enumerate(lp["qmeta"]) if m["key"] == "A")
        kB = next(k for k, m in enumerate(lp["qmeta"]) if m["key"] == "B")
        # Q_A قبل/بعد القلب
        pa = _binary_pair(model.next_logits(lp["tokens"][:lp["qpos"][kA] + 1]))
        pb = _binary_pair(model.next_logits(fv["tokens"][:fv["qpos"][kA] + 1]))
        if int(np.argmax(pa)) == lp["qmeta"][kA]["answer_binary"]:
            nA += 1
            follow += int(int(np.argmax(pb)) == fv["qmeta"][kA]["answer_binary"])
        dA += float(np.max(np.abs(pa - pb)))
        # Q_B قبل/بعد القلب (يجب ألا يتغير — قيمة B ثابتة)
        ra = _binary_pair(model.next_logits(lp["tokens"][:lp["qpos"][kB] + 1]))
        rb = _binary_pair(model.next_logits(fv["tokens"][:fv["qpos"][kB] + 1]))
        stable += int(int(np.argmax(ra)) == int(np.argmax(rb)))
        dB += float(np.max(np.abs(ra - rb)))
    return {"loops": n,
            "A_initially_correct_n": nA,
            "A_follow_rate_when_initially_correct": round(follow / nA, 4) if nA else None,
            "B_prediction_stability_rate": round(stable / n, 4),
            "mean_max_binary_prob_shift_A": round(dA / n, 4),
            "mean_max_binary_prob_shift_B": round(dB / n, 4)}


def query_order_swap_eval(model, loops):
    sw = [swap_qorder_loop(lp) for lp in loops]
    return {"val_original_binary": evaluate(model, loops)["binary_accuracy"],
            "val_swapped_binary": evaluate(model, sw)["binary_accuracy"]}


# ═════════════════════════════ التجميد والتقارير ═════════════════════════════
def _mean_eval_block(per_seed, section):
    metrics = ("binary_accuracy", "full_vocab_accuracy",
               "nll_full_vocab", "nll_binary_normalized")
    out = {}
    for met in metrics:
        vals = [ps[section][met] for ps in per_seed]
        out[f"{section}_{met}"] = {"mean": round(float(np.mean(vals)), 4),
                                   "std": round(float(np.std(vals)), 4),
                                   "per_seed": vals}
    return out


def validation_sweep(train_loops, val_loops):
    """بوابة 11: مسبح الاختيار على التحقق وحده، بقاعدة معلنة، بلا لمس holdout/الاختبار."""
    trials = []
    for name in ("fixed_query_prefix_reader", "causal_self_attention_1h"):
        for lr in SWEEP_GRID:
            m = new_model(name, SWEEP_SEED)
            train_model(m, train_loops, lr)
            ev = evaluate(m, val_loops)
            trials.append({"model": name, "lr": lr, "epochs": EPOCHS,
                           "seed": SWEEP_SEED,
                           "val_binary_accuracy": ev["binary_accuracy"],
                           "val_nll_full_vocab": ev["nll_full_vocab"]})
            print(f"[مسبح] {name} lr={lr}: binary={ev['binary_accuracy']} "
                  f"nll={ev['nll_full_vocab']}")
    # القاعدة: أسوأ نموذج عند كل lr هو مرشحه؛ الأعلى يفوز؛ التعادل → أقل lr.
    frozen_lr = max(SWEEP_GRID, key=lambda lr: (
        min(t["val_binary_accuracy"] for t in trials if t["lr"] == lr), -lr))
    return trials, frozen_lr


# ═════════════════════════════════ التشغيل ═════════════════════════════════
def main():
    t_start = time.perf_counter()
    print(f"== {VERSION} ==")

    # 1) البيانات + بوابات البيانات (6، 7، 4)
    segments, fam_ledger = generate_datasets()
    g6 = gate6_counterfactual_pairs(segments)
    g7 = gate7_dataset_exclusivity(segments, fam_ledger)
    g4 = gate4_loss_target_alignment(segments["train"])
    print(f"[بيانات] {{{', '.join(f'{s}: {len(segments[s])}' for s in SEGMENTS)}}} "
          f"— بوابات 6/7/4 ناجحة")

    # 2) بوابات الآلية على نماذج جديدة: 1، 2، 3، 5
    fresh = {name: new_model(name, MODEL_SEEDS[0]) for name in SPECS}
    fresh["local_only"] = new_model("local_only", MODEL_SEEDS[0])
    g1 = gate1_causal_mask_exact(fresh)
    g2_fresh = {n: gate2_future_suffix_invariance(m) for n, m in fresh.items()}
    g3_fresh = {n: gate3_prefix_vs_full(m) for n, m in fresh.items()}
    for n in fresh:
        if g2_fresh[n] > GATE_LOGIT_TOL:
            print(f"FAIL gate2 fresh {n}: {g2_fresh[n]}")
            return 1
        if g3_fresh[n] > GATE_LOGIT_TOL:
            print(f"FAIL gate3 fresh {n}: {g3_fresh[n]}")
            return 1
    g5 = {}
    for n in LEARNED_MODELS:
        m = fresh[n]
        err_fresh = gate5_finite_difference(m, fd_batches())
        # نموذج مُدرَّب جزئياً (معلمات غير بسيطة) ثم فحص ثانٍ:
        train_batch_steps(m, fd_batches()[1], lr=0.05, epochs=150)
        err_tr = gate5_finite_difference(m, fd_batches())
        g5[n] = {"untrained_worst_abs_error": round(err_fresh, 12),
                 "briefly_trained_worst_abs_error": round(err_tr, 12),
                 "eps": FD_EPS,
                 "note": "عينات فحص تركيبية ببذور خاصة — لا تنتمي لمقاطع التقييم"}
        if err_fresh > GATE_GRAD_TOL or err_tr > GATE_GRAD_TOL:
            print(f"FAIL gate5 {n}: fresh={err_fresh:.3e} trained={err_tr:.3e}")
            return 1
    # بوابة 5b (v1.1): التدرج المُجمَّع عبر كتل غير متساوية — عيب v0.16
    g5b = {n: gate5b_aggregated_gradient(fresh[n]) for n in LEARNED_MODELS}
    for n, rec in g5b.items():
        if rec["worst_abs_error"] > GATE_GRAD_TOL:
            print(f"FAIL gate5b {n}: {rec['worst_abs_error']:.3e}")
            return 1
    print("[بوابات] 1/2/3/5 ناجحة (جديدة + مدرَّبة جزئياً) — و5b للتجميع الموزون")

    # 3) مسبح الاختيار على التحقق وحده (بوابة 11) → تجميد الميزانية
    trials, frozen_lr = validation_sweep(segments["train"], segments["val"])
    print(f"[بوابة 11] الميزانية المجمدة: Adam lr={frozen_lr}، epochs={EPOCHS}")

    # 4) بوابة المراجع (8) قبل أي قراءة نهائية
    g8 = gate8_positive_oracle(segments["val"], segments["dev_holdout"],
                               segments["test_in"])
    print("[بوابة 8] المرجع 1.000 عبر المسار نفسه وقلب التسميات 0.000")

    # 5) التدريب النهائي (كل نموذج × كل بذور) + تقييم تحقق/تطويري — بلا قراءة نهائية
    trained_store, dual = {}, {}
    for name in LEARNED_MODELS:
        per_seed = {}
        for seed in MODEL_SEEDS:
            m = new_model(name, seed)
            info = train_model(m, segments["train"], frozen_lr)
            trained_store[(name, seed)] = m
            ev_val = evaluate(m, segments["val"])
            ev_dev = evaluate(m, segments["dev_holdout"])
            fr_val = free_rollout(m, segments["val"])
            att = evaluate(m, segments["val"], capture_attention=True) \
                .get("attention_at_query_positions", {})
            att_err = max((abs(sum(att[k].values()) - 1.0) for k in ("A", "B")),
                          default=0.0) if att else 0.0
            if att_err > 1e-3:
                print(f"FAIL attention normalization {name}: {att_err}")
                return 1
            per_seed[f"seed_{seed}"] = {
                "params": m.params_count(),
                "train_nll": info["final_train_nll"],
                "stable": info["stable"],
                "train_seconds": info["train_seconds"],
                "curve": info["curve"],
                "val": ev_val, "dev_holdout": ev_dev,
                "free_rollout_val": fr_val,
                "attention_maps_val": att,
                "attention_sum_max_abs_error": round(att_err, 6)}
            print(f"[تدريب] {name} seed={seed}: val={ev_val['binary_accuracy']} "
                  f"dev={ev_dev['binary_accuracy']} "
                  f"params={m.params_count()} ({info['train_seconds']}s)")
        per_seed["mean_across_seeds"] = _mean_eval_block(
            [per_seed[f"seed_{s}"] for s in MODEL_SEEDS], "val")
        per_seed["mean_across_seeds"].update(_mean_eval_block(
            [per_seed[f"seed_{s}"] for s in MODEL_SEEDS], "dev_holdout"))
        dual[name] = per_seed

    # 6) إعادة البوابات 2/3 على النماذج المدربة (بذرة 42)
    g2_trained = {n: gate2_future_suffix_invariance(trained_store[(n, MODEL_SEEDS[0])])
                  for n in LEARNED_MODELS}
    g3_trained = {n: gate3_prefix_vs_full(trained_store[(n, MODEL_SEEDS[0])])
                  for n in LEARNED_MODELS}
    for n in LEARNED_MODELS:
        if g2_trained[n] > GATE_LOGIT_TOL or g3_trained[n] > GATE_LOGIT_TOL:
            print(f"FAIL gate2/3 trained {n}: {g2_trained[n]}/{g3_trained[n]}")
            return 1
    print("[بوابات] 2/3 ناجحة على النماذج المدربة (بذرة 42)")

    # 7) ▼ التجميد ▼ الاختبار النهائي: داخل-التوزيع + طول-16 المحجوز (آخر قراءة)
    final_test = {}
    for name in LEARNED_MODELS:
        per_seed = {}
        for seed in MODEL_SEEDS:
            m = trained_store[(name, seed)]
            ev_in = evaluate(m, segments["test_in"])
            ev_ood = evaluate(m, segments["test_ood"])
            per_seed[f"seed_{seed}"] = {
                "test_in": ev_in, "test_ood_length16": ev_ood,
                "free_rollout_test_in": free_rollout(m, segments["test_in"])}
            print(f"[اختبار] {name} seed={seed}: in={ev_in['binary_accuracy']} "
                  f"ood16={ev_ood['binary_accuracy']}")
        per_seed["mean_across_seeds"] = _mean_eval_block(
            [per_seed[f"seed_{s}"] for s in MODEL_SEEDS], "test_in")
        per_seed["mean_across_seeds"].update(_mean_eval_block(
            [per_seed[f"seed_{s}"] for s in MODEL_SEEDS], "test_ood_length16"))
        final_test[name] = per_seed

    # 8) العزلات والتدخلات المقاسة (بوابة 9 ضمناً)
    ablations = {}
    ablations["no_memory"] = gate9_no_memory_control(
        {n: trained_store[(n, MODEL_SEEDS[0])] for n in LEARNED_MODELS},
        segments["val"])
    csw, qsw = {}, {}
    for n in ("causal_self_attention_1h", "fixed_query_prefix_reader",
              "no_position", "no_query_dependency"):
        m = trained_store[(n, MODEL_SEEDS[0])]
        csw[n] = {"val": counterfactual_value_swap(m, segments["val"]),
                  "test_in": counterfactual_value_swap(m, segments["test_in"])}
        qsw[n] = query_order_swap_eval(m, segments["val"])
    ablations["counterfactual_value_swap"] = csw
    ablations["query_order_swap"] = qsw
    lg = {}
    for n in LEARNED_MODELS:
        per_seed = {}
        for s in MODEL_SEEDS:
            m = trained_store[(n, s)]
            per_seed[f"seed_{s}"] = {
                "test_in_by_length": {str(Ln): evaluate(
                    m, [lp for lp in segments["test_in"] if lp["length"] == Ln])["binary_accuracy"]
                    for Ln in (2, 4, 8)},
                "test_ood16_binary": evaluate(m, segments["test_ood"])["binary_accuracy"],
                "test_ood16_nll_full": evaluate(m, segments["test_ood"])["nll_full_vocab"]}
        per_seed["mean_ood16_binary"] = round(float(np.mean(
            [per_seed[f"seed_{s}"]["test_ood16_binary"] for s in MODEL_SEEDS])), 4)
        lg[n] = per_seed
    ablations["length_generalization"] = lg
    ablations["length_generalization_note_learned_pe"] = (
        "تضمين موضعي مطلق متعلم: مواضع أطوال ما بعد التدريب (طول الحشو 16 → مواضع "
        "14–21) لم تُرَ أبداً في التدريب؛ فشل الطول 16 في هذه المقارنة محتمل التفسير "
        "بمواضع غير مدربة — لهذا استُخدم الترميز الجيبي الحتمي في الاختبار الأساسي وفق العقد.")
    ablations["isolation_summaries"] = {
        "no_position": {"val": dual["no_position"]["mean_across_seeds"],
                        "baseline": dual["causal_self_attention_1h"]["mean_across_seeds"]},
        "no_query_dependency": {"val": dual["no_query_dependency"]["mean_across_seeds"],
                                "baseline": dual["causal_self_attention_1h"]["mean_across_seeds"]}}

    # 9) التجميع والكتابة
    src_path = Path(__file__).resolve()
    results = {
        "simulator_version": VERSION,
        "protocol": {
            "optimizer": "Adam (full-batch موزون عبر كتل الأطوال {2,4,8}: "
                         "L = Σ n_i·L_i / Σ n_i وتدرج Σ n_i·∇L_i / Σ n_i — "
                         "تحديث واحد لكل عصر)",
            "training_aggregation": {
                "v1.1_fix": "التدقيق المستقل v0.16 أثبت أن تجميع التدرجات كان بلا "
                            "وزن أعداد المواضع؛ صُحح إلى الوزن الكامل وأُعيد تشغيل "
                            "كل النماذج والبذور تحت البروتوكول نفسه، وأُرشفت نتائج "
                            "v0.16 في content/chapter-08/archive/v0.16",
                "gate": "5b_aggregated_gradient: فروق مركزية للتدرج المجمع عبر "
                        "كتلتين بـ6 و10 مواضع",
                "run_kind": "v1.1 versioned replication تحت البروتوكول المجمد نفسه — "
                            "لا اختيار ولا ضبط في أي مقطع"},
            "frozen_budget": {"lr": frozen_lr, "epochs": EPOCHS,
                              "selection_rule": "عند كل lr يُرشَّح أسوأ النموذجين "
                                                "(binary على التحقق)؛ الأعلى يفوز؛ "
                                                "التعادل → أقل lr",
                              "selected_on": "validation فقط — dev_holdout والاختبار "
                                             "النهائي لم يُستخدما للاختيار"},
            "seeds": {"model_init": list(MODEL_SEEDS), "data_generation": DATA_SEED,
                      "note": "بذرة البيانات مستقلة عن بذور النموذج؛ البيانات مشتركة "
                              "بين كل النماذج"},
            "tokenizer": "رموز ذرّية (بلا تجزئة) — قيد علمي معلن: تختبر اختيار السجل "
                         "الصحيح، ولا تُثبت ربط مفتاح/قيمة مكتوبين برمزين منفصلين "
                         "(امتداد معلن مستقبلي)",
            "vocabulary": {"size": V,
                           "memory": ["M_A0", "M_A1", "M_B0", "M_B1"],
                           "queries": ["Q_A", "Q_B"],
                           "answers": ["ANS0", "ANS1"],
                           "fillers": f"F0..F{VOCAB_N_FILLER - 1} (مجموعة منفصلة بلا إشارات فئات)"},
            "vocabulary_counts": {"memory": 4, "queries": 2, "answers": 2,
                                  "fillers": VOCAB_N_FILLER, "total": V},
            "loop_lengths": {"train_val_dev": [2, 4, 8], "reserved_ood": 16},
            "structural_balance_classes": 16,
            "supervised_positions_per_loop": 2,
            "loop_structure": "سجلان (بترتيب متبدَّل متوازن) + حشو بطول {2,4,8} + "
                              "[سؤال، جواب، سؤال، جواب] بترتيب متبدَّل متوازن؛ "
                              "القيمتان مستقلتان موزعتان بالتساوي (لا تعاكس مفروض)",
            "output_policy": "softmax على المفردة الكاملة (32)؛ تُعرض: دقة ثنائية "
                             "منمّطة على {ANS0,ANS1}، دقة كامل-المفردة، NLL كامل-المفردة، "
                             "NLL ثنائي منمّط — لا يُساوَى المقياسان؛ العشوائي الثنائي "
                             "0.5/NLL=ln2≈0.6931 والعشوائي الكامل NLL=ln32≈3.4657",
            "loss_mask": "NLL عند موضعي الإجابة بعد Q_A/Q_B فقط (موضعان لكل حلقة)؛ "
                         "باقي المواضع غير مُشرفة؛ موضع السؤال لا يتلقى رمز الإجابة مدخلاً",
            "prediction_task": "تنبؤ بالرمز التالي عند موضعي السؤال؛ القناع السببي في "
                               "التدريب والتقييم والتوليد؛ teacher forcing والتوليد "
                               "الحر يُسجلان منفصلين",
            "architecture_notes": {
                "positional_encoding": "جيبية حتمية (لا تُتعلم) في الاختبار الأساسي "
                                       "لأطوال {2,4,8,16} جميعها؛ التضمين المطلق "
                                       "المتعلم مقارنة منفصلة معلنة",
                "residual": "causal_self_attention_1h: logits = WO·(h_t + c_t) + bO؛ "
                            "القارئ الثابت: logits = WO·concat(c_t, h_t) + bO",
                "heads_layers": "رأس واحد، طبقة واحدة — لا ادعاء بناء Transformer كامل",
                "complexity": "استعلام ثابت لتنبؤ واحد O(T)؛ انتباه ذاتي كثيف لكل "
                              "المواضع O(T²) لحساب الدرجات — المقارنة الزمنية تعتمد "
                              "عدد الاستعلامات والتنفيذ والتخزين والعتاد"},
            "model_configs": {n: SPECS[n] for n in SPECS},
            "splits": {"train": 512, "val": 128, "dev_holdout": 128,
                       "test_in_distribution": 64, "test_ood_length16": 64,
                       "dev_holdout_policy": "تقييم تطويري يُبلَّغ فقط — لا اختيار عليه",
                       "test_policy": "لا يُقرأ ولا يُختار عليه إلا بعد نجاح كل البوابات "
                                      "وتجميد الميزانية (الخطوة الأخيرة في المحاكي)"},
            "eval_path": "مسار تقييم واحد لكل النماذج (المرجع ضمناً)؛ تقييم الحلقة الكاملة "
                         "بالقناع السببي يكافئ تقييم البادئة (بوابة 3)؛ خريطة الانتباه "
                         "قرينة فقط — الإسناد السببي عبر قلب القيمة والإخفاء",
            "stance": "لا افتراض تفوق لأي معمارية؛ النتائج كما تقيس — نجاحاً أو فشلاً",
        },
        "gates": {
            "1_causal_mask_exact": g1,
            "2_future_suffix_invariance": {"fresh": g2_fresh,
                                           "trained_seed42": g2_trained,
                                           "tolerance": GATE_LOGIT_TOL},
            "3_prefix_vs_full": {"fresh": g3_fresh, "trained_seed42": g3_trained,
                                 "tolerance": GATE_LOGIT_TOL},
            "4_loss_target_alignment": g4,
            "5_finite_difference_gradient": g5,
            "5b_aggregated_gradient": g5b,
            "6_counterfactual_pairs": g6,
            "7_dataset_exclusivity": g7,
            "8_positive_oracle": g8,
            "9_no_memory_control": {"oracle_binary": ablations["no_memory"]["oracle_no_memory_binary_accuracy"],
                                    "measured_models": list(LEARNED_MODELS)},
            "10_seed_scope": "كل مقياس مسجل لكل بذرة مع المتوسط والتشتت (std ثلاث بذور)",
            "11_validation_selection": {"sweep_trials": trials, "frozen_lr": frozen_lr,
                                        "grid": list(SWEEP_GRID),
                                        "holdout_used_for_selection": False},
            "12_prose_regression": {"mechanism": "tools/test_chapter8.py",
                                    "scope": "كل رقم في النصين مستمد من results.json "
                                             "وفحص المصطلحات (استعلام ثابت/لكل موضع، "
                                             "سببي، معلمات، علم بالتوقيت)"},
        },
        "experiments": {
            "dual_lookup": dual,
            "final_test": final_test,
            "ablations": ablations,
        },
        "provenance": {
            "dataset_sha256": g7["segment_sha256"],
            "source_sha256": hashlib.sha256(src_path.read_bytes()).hexdigest(),
            "runtime_versions": {"python": platform.python_version(),
                                 "numpy": np.__version__},
            "config_fingerprint": _fp({
                "version": VERSION, "dims": [EMB_DIM, DK],
                "seeds": sorted(MODEL_SEEDS), "data_seed": DATA_SEED,
                "epochs": EPOCHS, "sweep": list(SWEEP_GRID),
                "splits": g7["segment_sizes"], "vocab": V}),
        },
        "total_seconds": round(time.perf_counter() - t_start, 2),
    }
    # سجل تشغيل ملحق (append-only): run_id يُحسب من محتوى النتائج قبل إضافة حقل السجل
    # (لا دورية)؛ ويحمل السطر في الملف بصمة النتائج النهائية وسلاسل البصمات.
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
    results["run_log"] = {"path": "content/chapter-08/runs.log", "run_id": run_id}
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
