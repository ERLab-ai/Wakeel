#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
simulate_chapter7.py — الفصل السابع: الانتباه والوصول إلى معلومة سابقة. (v2.0)

تصحيحات تدقيق v0.12 (بوابة قبل الاعتماد):
  P0-1 مشتقة softmax المصححة: dS = A·(g − Σ A·g) — كانت Σ g بلا أوزان.
       فصل _loss_and_grads عن التحديث ليفحصه التدرج العددي إلزامياً في test_chapter7.
  P0-2 أزواج متقابلة بالحشو نفسه: مسار حشو واحد يُصدر منه الترتيبان معاً،
       والفصل بين المجموعات على مستوى الزوج. bag والانتباه-بلا-موضع يجب أن
       يعطيا لوغيتات متطابقة للترتيبين (تُقاس وتُسجل) ودقة 0.500 بالضبط.
  P1-3 أحجام صريحة: تحقق 30/فئة وholdout 40/فئة في التجربتين، مع دفتر استبعاد
       مشترك (بادئات كاملة) عبر train→val→holdout وفحص انفصال صارم.
  P1-4 الإزاحة: تبديل موضعي حتمي على أمثلة holdout نفسها (نفس الحشو والترتيب
       الداخلي للحشو والبطاقة والتسمية والطول) — لا سحب جديد.
  P1-5 تسمية صريحة: قارئ الاسترجاع محتوى-بلا-موضع (671 معلمة)؛ والتجربة
       «نفس عائلة المهمة ببروتوكول مقارنة منقح» لا «إعادة حرفية» لـCH-6.
  P2 تحكم إيجابي متعلم: Pos0Reader يقرأ الموضع 0 بنيوياً — يجب أن يبلغ الكمال
       على المعيار القياسي (بوابة صارمة)، ويسقط على المزاح (فlique معلنة).

ضوابط أخرى كما في v1.0: بيانات مشتركة ببصمة، full-batch GD، مسار تقييم واحد
من CH-6، مرجع OracleAt سقفاً منفصلاً، O(n) مقابل O(n²) معلن، لا افتراض تفوق.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import simulate_chapter6 as ch6  # noqa: E402

SRC = ch6.SRC
OUT = ROOT / "content" / "chapter-07" / "results.json"
VERSION = "simulate_chapter7-2.1"

EMB_DIM = 8
DK = 8
EPOCHS, LR = 300, 0.3
SEEDS = [42, 43, 44]
DELAYS_RETRIEVAL = (2, 4, 5, 6, 7, 8, 16)
DELAYS_ORDER = (4, 8, 16)
TRAIN_PAIRS, VAL_PAIRS, HOLDOUT_PAIRS = 100, 30, 40
TRAIN_PER_CLASS, VAL_PER_CLASS, HOLDOUT_PER_CLASS = 100, 30, 40
CURVE_AT = (1, 100, 200, 300)


def softmax_last(x):
    z = x - x.max(axis=-1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=-1, keepdims=True)


class AttnReader:
    """قارئ انتباه أحادي الاستعلام. t_i = E[x_i] (+ P[i] إذا use_pos).
    التجميع فوق كل رموز البادئة (بما فيها ؟)."""

    def __init__(self, V, V_out, seed, use_pos, max_len=40, amplify=1.0, amplify_map=None):
        """amplify_map: تضخيم انتقائي لتهيئة مجموعات محددة (عزل سببي استكشافي) —
        مثال {"Wk": 3.0, "wq": 3.0} يضخم المفاتيح والاستعلام وحدهما."""
        amap = {"E": amplify, "Wk": amplify, "Wv": amplify,
                "wq": amplify, "Wo": amplify, "P": amplify}
        if amplify_map:
            amap.update(amplify_map)
        rng = np.random.default_rng(seed)
        self.E = rng.normal(0, 0.1, (V, EMB_DIM)) * amap["E"]
        self.P = rng.normal(0, 0.1, (max_len, EMB_DIM)) * amap["P"] if use_pos else None
        self.Wk = rng.normal(0, 0.1, (DK, EMB_DIM)) * amap["Wk"]
        self.Wv = rng.normal(0, 0.1, (DK, EMB_DIM)) * amap["Wv"]
        self.wq = rng.normal(0, 0.1, DK) * amap["wq"]
        self.Wo = rng.normal(0, 0.1, (EMB_DIM, V_out)) * amap["Wo"]
        self.bo = np.zeros(V_out)
        self.use_pos = use_pos

    def params_count(self):
        n = (self.E.size + self.Wk.size + self.Wv.size
             + self.wq.size + self.Wo.size + self.bo.size)
        return int(n + (self.P.size if self.P is not None else 0))

    def _pool(self, X):
        T = self.E[X]
        if self.P is not None:
            T = T + self.P[None, :X.shape[1], :]
        K = T @ self.Wk.T
        Vv = T @ self.Wv.T
        scores = (K @ self.wq) / np.sqrt(DK)
        A = softmax_last(scores)
        C = np.einsum("nl,nld->nd", A, Vv)
        return C, (T, K, Vv, A)

    def _loss_and_grads(self, seqs, answers):
        """loss وكل التدرجات التحليلية — بلا تحديث (للفحص العددي والتدريب)."""
        X = np.array([s[:-1] for s in seqs])
        Y = np.array(answers)
        N = len(X)
        C, (T, K, Vv, A) = self._pool(X)
        Z = C @ self.Wo + self.bo
        Pp = softmax_last(Z)
        loss = float(-np.mean(np.log(np.maximum(Pp[np.arange(N), Y], 1e-12))))
        dZ = Pp.copy()
        dZ[np.arange(N), Y] -= 1.0
        dZ /= N
        g = {
            "Wo": C.T @ dZ,
            "bo": dZ.sum(axis=0),
        }
        dC = dZ @ self.Wo.T
        dV = A[:, :, None] * dC[:, None, :]
        g["Wv"] = np.einsum("nld,nle->de", dV, T)
        dT = dV @ self.Wv
        gdot = np.einsum("nd,nld->nl", dC, Vv)
        # مشتقة softmax الصحيحة (تصحيح P0-1 لـv1.0 التي جمعت gdot بلا أوزان):
        dS = A * (gdot - np.sum(A * gdot, axis=1, keepdims=True))
        dK = (dS / np.sqrt(DK))[:, :, None] * self.wq[None, None, :]
        g["Wk"] = np.einsum("nld,nle->de", dK, T)
        g["wq"] = np.einsum("nl,nld->d", dS, K) / np.sqrt(DK)
        dT += dK @ self.Wk
        gE = np.zeros_like(self.E)
        np.add.at(gE, X.ravel(), dT.reshape(-1, EMB_DIM))
        g["E"] = gE
        if self.P is not None:
            gP = np.zeros_like(self.P)
            gP[:X.shape[1]] = dT.sum(axis=0)
            g["P"] = gP
        return loss, g

    def train_epoch(self, seqs, answers, lr):
        loss, g = self._loss_and_grads(seqs, answers)
        self.E -= lr * g["E"]
        self.Wk -= lr * g["Wk"]
        self.Wv -= lr * g["Wv"]
        self.wq -= lr * g["wq"]
        self.Wo -= lr * g["Wo"]
        self.bo -= lr * g["bo"]
        if self.P is not None:
            self.P -= lr * g["P"]
        return loss

    def predict_next_logits(self, prefix_ids):
        X = np.array(prefix_ids)[None, :]
        C, _ = self._pool(X)
        return (C @ self.Wo + self.bo)[0]


class MeanReader:
    """خط أساس bag-of-tokens: متوسط التضمينات — أعمى الترتيب بنيوياً."""

    def __init__(self, V, V_out, seed, amplify=1.0):
        rng = np.random.default_rng(seed)
        self.E = rng.normal(0, 0.1, (V, EMB_DIM)) * amplify
        self.Wo = rng.normal(0, 0.1, (EMB_DIM, V_out)) * amplify
        self.bo = np.zeros(V_out)

    def params_count(self):
        return int(self.E.size + self.Wo.size + self.bo.size)

    def _loss_and_grads(self, seqs, answers):
        X = np.array([s[:-1] for s in seqs])
        Y = np.array(answers)
        N, L = X.shape
        C = self.E[X].mean(axis=1)
        Z = C @ self.Wo + self.bo
        Pp = softmax_last(Z)
        loss = float(-np.mean(np.log(np.maximum(Pp[np.arange(N), Y], 1e-12))))
        dZ = Pp.copy()
        dZ[np.arange(N), Y] -= 1.0
        dZ /= N
        g = {"Wo": C.T @ dZ, "bo": dZ.sum(axis=0)}
        dC = dZ @ self.Wo.T
        gE = np.zeros_like(self.E)
        np.add.at(gE, X.ravel(), np.repeat(dC / L, L, axis=0))
        g["E"] = gE
        return loss, g

    def train_epoch(self, seqs, answers, lr):
        loss, g = self._loss_and_grads(seqs, answers)
        self.E -= lr * g["E"]
        self.Wo -= lr * g["Wo"]
        self.bo -= lr * g["bo"]
        return loss

    def predict_next_logits(self, prefix_ids):
        C = self.E[np.array(prefix_ids)].mean(axis=0)
        return C @ self.Wo + self.bo


class PosReader:
    """تحكم إيجابي متعلم: يقرأ موضعاً معلوماً بالبناء (0 قياسياً، delay//2 مزاحاً)
    — c = E[x_pos] ثم خط إخراج. يثبت أن الأنبوب قادر على التعبير عن الحل."""

    def __init__(self, V, V_out, seed, position):
        rng = np.random.default_rng(seed)
        self.E = rng.normal(0, 0.1, (V, EMB_DIM))
        self.Wo = rng.normal(0, 0.1, (EMB_DIM, V_out))
        self.bo = np.zeros(V_out)
        self.position = position

    def params_count(self):
        return int(self.E.size + self.Wo.size + self.bo.size)

    def _loss_and_grads(self, seqs, answers):
        X = np.array([s[:-1] for s in seqs])
        Y = np.array(answers)
        N = len(X)
        C = self.E[X[:, self.position]]
        Z = C @ self.Wo + self.bo
        Pp = softmax_last(Z)
        loss = float(-np.mean(np.log(np.maximum(Pp[np.arange(N), Y], 1e-12))))
        dZ = Pp.copy()
        dZ[np.arange(N), Y] -= 1.0
        dZ /= N
        g = {"Wo": C.T @ dZ, "bo": dZ.sum(axis=0)}
        dC = dZ @ self.Wo.T
        gE = np.zeros_like(self.E)
        np.add.at(gE, X[:, self.position], dC)
        g["E"] = gE
        return loss, g

    def train_epoch(self, seqs, answers, lr):
        loss, g = self._loss_and_grads(seqs, answers)
        self.E -= lr * g["E"]
        self.Wo -= lr * g["Wo"]
        self.bo -= lr * g["bo"]
        return loss

    def predict_next_logits(self, prefix_ids):
        C = self.E[prefix_ids[self.position]]
        return C @ self.Wo + self.bo


class OracleAt:
    """مرجع يعرف موضع المعلومة مسبقاً — سقف منفصل عبر مسار التقييم نفسه."""

    def __init__(self, position, out_pos, n_out):
        self.position = position
        self.out_pos = out_pos
        self.n_out = n_out

    def predict_next_logits(self, prefix_ids):
        logits = np.full(self.n_out, -30.0)
        logits[self.out_pos[prefix_ids[self.position]]] = 30.0
        return logits


def fp(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False)
                          .encode("utf-8")).hexdigest()[:16]


def rnn_params(model):
    return int(sum(x.size for x in [model.E, model.Wx, model.Wh, model.bh,
                                    model.Wy, model.by]))


def train_rnn(model, seqs):
    t = time.perf_counter()
    out_pos_s = ch6_out_pos
    pairs = [(s, [out_pos_s[i] for i in s[1:]], [0.0] * (len(s) - 2) + [1.0])
             for s in seqs]
    curve = []
    for ep in range(1, EPOCHS + 1):
        tl = model.train_epoch(pairs, LR)
        if ep in CURVE_AT:
            curve.append({"epoch": ep, "train_nll": round(tl, 4)})
    return {"params": rnn_params(model),
            "train_seconds": round(time.perf_counter() - t, 2), "curve": curve}


def train_reader(model, seqs, out_pos_s):
    t = time.perf_counter()
    answers = [out_pos_s[s[-1]] for s in seqs]
    curve = []
    for ep in range(1, EPOCHS + 1):
        tl = model.train_epoch(seqs, answers, LR)
        if ep in CURVE_AT:
            curve.append({"epoch": ep, "train_nll": round(tl, 4)})
    return {"params": model.params_count(),
            "train_seconds": round(time.perf_counter() - t, 2), "curve": curve}


def draw_matched_pairs(delay, n, rng, synth_vocab, ledger):
    """أزواج متقابلة: مسار حشو واحد يُصدر الترتيبين معاً؛ ledger يحجب التكرار."""
    pairs = []
    while len(pairs) < n:
        fill = [synth_vocab[c] for c in rng.choice(list(ch6.FILLER), size=delay, replace=True)]
        a = [synth_vocab[ch6.CLASS_A], synth_vocab[ch6.CLASS_B]] + fill + [synth_vocab[ch6.QUERY]]
        b = [synth_vocab[ch6.CLASS_B], synth_vocab[ch6.CLASS_A]] + fill + [synth_vocab[ch6.QUERY]]
        key = tuple(fill)
        if key in ledger:
            continue
        ledger.add(key)
        pairs.append((a, synth_vocab[ch6.CLASS_A], b, synth_vocab[ch6.CLASS_B]))
    return pairs


def main():
    global ch6_out_pos
    raw = SRC.read_bytes()
    input_hash = hashlib.sha256(raw).hexdigest()
    src = json.loads(SRC.read_text(encoding="utf-8"))
    t0 = time.perf_counter()

    synth_vocab = ch6.build_vocab(ch6.CLASS_A + ch6.CLASS_B + ch6.FILLER + ch6.QUERY)
    out_ids_s, out_pos_s = ch6.out_space(synth_vocab)
    ch6_out_pos = out_pos_s
    V_s, V_out_s = len(synth_vocab), len(out_ids_s)
    id_a, id_b = synth_vocab[ch6.CLASS_A], synth_vocab[ch6.CLASS_B]

    results = {"simulator_version": VERSION, "input_hash": input_hash,
               "protocol": {"epochs": EPOCHS, "lr": LR, "seeds": SEEDS,
                            "optimizer": "full-batch gradient descent",
                            "emb_dim": EMB_DIM, "dk": DK,
                            "splits": {"train_pairs": TRAIN_PAIRS,
                                       "val_pairs": VAL_PAIRS,
                                       "holdout_pairs": HOLDOUT_PAIRS,
                                       "retrieval": {"train_per_class": TRAIN_PER_CLASS,
                                                     "val_per_class": VAL_PER_CLASS,
                                                     "holdout_per_class": HOLDOUT_PER_CLASS},
                                       "holdout_policy": "لا يُقرأ إلا للتقرير — لا ضبط فائق إطلاقاً",
                                       "exclusivity": "دفتر استبعاد مشترك بالبادئات الكاملة عبر train→val→holdout"},
                            "paired_design": "مسار حشو واحد يُصدر منه الترتيبان معاً؛ الفصل على مستوى الزوج",
                            "complexity_note": "استعلام واحد O(n)؛ Self-Attention الكاملة O(n²) غير منفذة",
                            "softmax_fix": "dS = A·(g − Σ A·g) — تصحيح P0-1 (تدقيق v0.12)"},
               "experiments": {}}

    # ═══════════ تجربة أ: الاسترجاع (نفس عائلة مهمة CH-6، بروتوكول مقارنة منقح) ═══════════
    exp_a = {"task": "استرجاع الحرف الأول بعد حشو ثم ؟ — نفس عائلة مهمة CH-6 "
                     "(100/فئة تدريباً بدل 120، وقسم تحقق مضاف): ليست إعادة حرفية",
             "models": {"rnn": "RNN شيفرة CH-6",
                        "attn_content": "قارئ انتباه محتوى بلا ترميز موضعي (671 معلمة)"},
             "oracle": "OracleAt(0)", "delays": {}}
    for delay in DELAYS_RETRIEVAL:
        per_delay = {}
        for seed in SEEDS:
            rng_tr = np.random.default_rng(1000 + seed)
            rng_val = np.random.default_rng(5000 + seed)
            rng_ho = np.random.default_rng(6000 + seed)
            ledger = set()   # بادئات كاملة — انفصال صارم عبر المقاطع الثلاثة

            def draw(first, n, rng):
                out = []
                while len(out) < n:
                    s = ch6.synth_seq(first, delay, rng, synth_vocab)[:-1]
                    key = tuple(s)
                    if key in ledger:
                        continue
                    ledger.add(key)
                    out.append(s)
                return out

            tr_a = [ch6.synth_seq(id_a, delay, rng_tr, synth_vocab)
                    for _ in range(TRAIN_PER_CLASS)]
            tr_b = [ch6.synth_seq(id_b, delay, rng_tr, synth_vocab)
                    for _ in range(TRAIN_PER_CLASS)]
            for s in tr_a + tr_b:
                ledger.add(tuple(s[:-1]))
            val_a, val_b = draw(id_a, VAL_PER_CLASS, rng_val), draw(id_b, VAL_PER_CLASS, rng_val)
            ho_a, ho_b = draw(id_a, HOLDOUT_PER_CLASS, rng_ho), draw(id_b, HOLDOUT_PER_CLASS, rng_ho)
            sizes = {"train_per_class": TRAIN_PER_CLASS,
                     "val_per_class": len(val_a),
                     "holdout_per_class": len(ho_a)}
            if sizes != {"train_per_class": TRAIN_PER_CLASS, "val_per_class": VAL_PER_CLASS,
                         "holdout_per_class": HOLDOUT_PER_CLASS}:
                print(f"FAIL sizes delay={delay} seed={seed}: {sizes}")
                return 1

            all_seqs = tr_a + tr_b
            oracle = OracleAt(0, out_pos_s, V_out_s)
            entry = {"dataset_fingerprint": fp({"train": all_seqs, "val": val_a + val_b,
                                                "holdout": ho_a + ho_b}),
                     "sizes": sizes}
            for name in ("rnn", "attn_content"):
                if name == "rnn":
                    model = ch6.SimpleRNN(V_s, V_out_s, seed)
                    info = train_rnn(model, all_seqs)
                else:
                    model = AttnReader(V_s, V_out_s, seed, use_pos=False)
                    info = train_reader(model, all_seqs, out_pos_s)
                ha = ch6.evaluate(model, ho_a, id_a, out_pos_s, id_a, id_b)
                hb = ch6.evaluate(model, ho_b, id_b, out_pos_s, id_a, id_b)
                va = ch6.evaluate(model, val_a, id_a, out_pos_s, id_a, id_b)
                vb = ch6.evaluate(model, val_b, id_b, out_pos_s, id_a, id_b)
                entry[name] = {"params": info["params"],
                               "train_seconds": info["train_seconds"],
                               "holdout_binary": round((ha[1] + hb[1]) / 2, 3),
                               "holdout_argmax": round((ha[0] + hb[0]) / 2, 3),
                               "holdout_test_nll": round((ha[2] + hb[2]) / 2, 4),
                               "per_class_binary": [round(ha[1], 3), round(hb[1], 3)],
                               "val_binary": round((va[1] + vb[1]) / 2, 3),
                               "curve": info["curve"]}
            oa = ch6.evaluate(oracle, ho_a, id_a, out_pos_s, id_a, id_b)[1]
            ob = ch6.evaluate(oracle, ho_b, id_b, out_pos_s, id_a, id_b)[1]
            if oa != 1.0 or ob != 1.0:
                print(f"FAIL oracle retrieval delay={delay} seed={seed}: {oa}/{ob}")
                return 1
            entry["oracle_ceiling"] = 1.0
            per_delay[f"seed_{seed}"] = entry
            print(f"[أ] delay={delay} seed={seed}: " +
                  " ".join(f"{n}={entry[n]['holdout_binary']}" for n in ("rnn", "attn_content")))
        per_delay["mean_across_seeds"] = {
            name: {"holdout_binary": round(float(np.mean(
                [per_delay[f"seed_{s}"][name]["holdout_binary"] for s in SEEDS])), 3),
                "holdout_test_nll": round(float(np.mean(
                    [per_delay[f"seed_{s}"][name]["holdout_test_nll"] for s in SEEDS])), 4)}
            for name in ("rnn", "attn_content")}
        exp_a["delays"][f"delay_{delay}"] = per_delay
    results["experiments"]["retrieval"] = exp_a

    # ═══════════ تجربة ب: تمييز الترتيب — أزواج متقابلة بالحشو نفسه ═══════════
    exp_b = {"task": "تمييز الترتيب: لكل مسار حشو مثالان متقابلان — "
                     "[أ م + f + ؟]←أ و [م أ + f نفسه + ؟]←م",
             "shortcut_removed": "رمزا الفئة حاضران في كل مثال؛ الإجابة يحددها الموضع",
             "splits_at": "مستوى الزوج (الترتيبان معاً في المقطع نفسه)",
             "models": ["rnn", "attn_pos", "attn_nopos", "mean",
                        "pos0_positive_control", "pos_mid_shifted_control"],
             "delays": {}}
    for delay in DELAYS_ORDER:
        per_delay = {}
        for seed in SEEDS:
            rng_tr = np.random.default_rng(7000 + seed)
            rng_val = np.random.default_rng(8000 + seed)
            rng_ho = np.random.default_rng(9000 + seed)
            ledger = set()

            def draw_pairs(n, rng):
                pairs = []
                while len(pairs) < n:
                    fill = [synth_vocab[c] for c in
                            rng.choice(list(ch6.FILLER), size=delay, replace=True)]
                    a = [id_a, id_b] + fill + [synth_vocab[ch6.QUERY]]
                    b = [id_b, id_a] + fill + [synth_vocab[ch6.QUERY]]
                    key = tuple(fill)
                    if key in ledger:
                        continue
                    ledger.add(key)
                    assert sorted(a) == sorted(b)          # بوابة التقابل
                    pairs.append((a, id_a, b, id_b))
                return pairs

            tr_pairs = draw_pairs(TRAIN_PAIRS, rng_tr)
            va_pairs = draw_pairs(VAL_PAIRS, rng_val)
            ho_pairs = draw_pairs(HOLDOUT_PAIRS, rng_ho)

            # تسلسلات التدريب تتضمن رمز الجواب (الهدف عند موضع الإجابة)؛
            # أمثلة التقييم بادئات حتى ؟ فقط
            tr_seqs = [p[0] + [p[1]] for p in tr_pairs] \
                + [p[2] + [p[3]] for p in tr_pairs]
            ho_std = [p[0] for p in ho_pairs] + [p[2] for p in ho_pairs]
            # الإزاحة: تبديل موضعي حتمي على أمثلة holdout نفسها (نفس الحشو بترتيبه،
            # وبنفس البادئة كاملة حتى ؟) — بناء كتليّ: كتلة أ ثم كتلة ب
            pre = delay // 2

            def make_shifted(s):
                f = s[2:2 + delay]
                sh = f[:pre] + [s[0], s[1]] + f[pre:] + [s[-1]]
                assert sorted(sh) == sorted(s) and len(sh) == len(s) \
                    and sh[-1] == s[-1] and sh[2:2 + delay] != s[2:2 + delay]
                return sh

            ho_shift = [make_shifted(p[0]) for p in ho_pairs] \
                + [make_shifted(p[2]) for p in ho_pairs]

            oracle_std = OracleAt(0, out_pos_s, V_out_s)
            oracle_shift = OracleAt(pre, out_pos_s, V_out_s)
            # التحكم الإيجابي المزاح يُدرَّب على النسخ المزاحة من أزواج التدريب
            # نفسها (تحويل حتمي متسق مع توزيع التقييم)
            tr_shift_seqs = [make_shifted(p[0]) + [p[1]] for p in tr_pairs] \
                + [make_shifted(p[2]) + [p[3]] for p in tr_pairs]
            entry = {"dataset_fingerprint": fp(
                {"train_pairs": tr_pairs, "val_pairs": va_pairs, "holdout_pairs": ho_pairs}),
                "sizes": {"train_pairs": len(tr_pairs), "val_pairs": len(va_pairs),
                          "holdout_pairs": len(ho_pairs)}}
            models = {
                "rnn": lambda: ch6.SimpleRNN(V_s, V_out_s, seed),
                "attn_pos": lambda: AttnReader(V_s, V_out_s, seed, use_pos=True),
                "attn_nopos": lambda: AttnReader(V_s, V_out_s, seed, use_pos=False),
                "mean": lambda: MeanReader(V_s, V_out_s, seed),
                "pos0": lambda: PosReader(V_s, V_out_s, seed, position=0),
                "pos_mid_shift": lambda: PosReader(V_s, V_out_s, seed, position=pre),
            }
            trained = {}
            for name, factory in models.items():
                model = factory()
                if name == "rnn":
                    info = train_rnn(model, tr_seqs)
                elif name == "pos_mid_shift":
                    info = train_reader(model, tr_shift_seqs, out_pos_s)
                else:
                    info = train_reader(model, tr_seqs, out_pos_s)
                trained[name] = model
                h_a = ch6.evaluate(model, ho_std[:len(ho_pairs)], id_a, out_pos_s, id_a, id_b)
                h_b = ch6.evaluate(model, ho_std[len(ho_pairs):], id_b, out_pos_s, id_a, id_b)
                v_a = ch6.evaluate(model, [p[0] for p in va_pairs], id_a, out_pos_s, id_a, id_b)
                v_b = ch6.evaluate(model, [p[2] for p in va_pairs], id_b, out_pos_s, id_a, id_b)
                s_a = ch6.evaluate(model, ho_shift[:len(ho_pairs)], id_a, out_pos_s, id_a, id_b)
                s_b = ch6.evaluate(model, ho_shift[len(ho_pairs):], id_b, out_pos_s, id_a, id_b)
                entry[name] = {
                    "params": info["params"],
                    "train_seconds": info["train_seconds"],
                    "holdout_binary": round((h_a[1] + h_b[1]) / 2, 3),
                    "holdout_argmax": round((h_a[0] + h_b[0]) / 2, 3),
                    "holdout_test_nll": round((h_a[2] + h_b[2]) / 2, 4),
                    "per_class_binary": [round(h_a[1], 3), round(h_b[1], 3)],
                    "val_binary": round((v_a[1] + v_b[1]) / 2, 3),
                    "shifted_pair_binary": round((s_a[1] + s_b[1]) / 2, 3),
                    "shifted_pair_nll": round((s_a[2] + s_b[2]) / 2, 4),
                    "curve": info["curve"],
                }
            # بوابة عدم-التبادلية: bag وattn_nopos لوغيتاتهما متطابقة داخل الزوج
            # قبل التدريب (نماذج جديدة) وبعده (المدرَّبة) — ودقتهما 0.500 بالضبط
            inv = {}
            for name in ("mean", "attn_nopos"):
                diffs = []
                for m_fresh, m_trained in ((models[name](), trained[name]),):
                    for a, lab_a, b, lab_b in ho_pairs[:10]:
                        la = m_fresh.predict_next_logits(a)
                        lb = m_fresh.predict_next_logits(b)
                        diffs.append(float(np.max(np.abs(la - lb))))
                    for a, lab_a, b, lab_b in ho_pairs[:10]:
                        la = m_trained.predict_next_logits(a)
                        lb = m_trained.predict_next_logits(b)
                        diffs.append(float(np.max(np.abs(la - lb))))
                inv[name] = round(max(diffs), 12)
                if inv[name] > 1e-8:
                    print(f"FAIL invariance {name} delay={delay} seed={seed}: {inv[name]}")
                    return 1
            entry["paired_invariance_maxdiff"] = inv
            for name in ("mean", "attn_nopos"):
                if entry[name]["holdout_binary"] != 0.5:
                    print(f"FAIL order-blind exact 0.5: {name} delay={delay} "
                          f"seed={seed} = {entry[name]['holdout_binary']}")
                    return 1
            o_std = (ch6.evaluate(oracle_std, ho_std[:len(ho_pairs)], id_a, out_pos_s, id_a, id_b)[1]
                     + ch6.evaluate(oracle_std, ho_std[len(ho_pairs):], id_b, out_pos_s, id_a, id_b)[1]) / 2
            o_sh = (ch6.evaluate(oracle_shift, ho_shift[:len(ho_pairs)], id_a, out_pos_s, id_a, id_b)[1]
                    + ch6.evaluate(oracle_shift, ho_shift[len(ho_pairs):], id_b, out_pos_s, id_a, id_b)[1]) / 2
            swap = ch6.evaluate(oracle_std, ho_std[:len(ho_pairs)], id_b, out_pos_s, id_a, id_b)[1]
            if o_std != 1.0 or o_sh != 1.0:
                print(f"FAIL oracle order delay={delay} seed={seed}: {o_std}/{o_sh}")
                return 1
            # بوابة التحكم الإيجابي المتعلم: pos0 = 1.000 قياسياً
            if entry["pos0"]["holdout_binary"] != 1.0:
                print(f"FAIL positive control pos0 delay={delay} seed={seed}: "
                      f"{entry['pos0']['holdout_binary']}")
                return 1
            entry["controls"] = {
                "oracle_standard": round(o_std, 3),
                "oracle_shifted": round(o_sh, 3),
                "oracle_label_swap": round(swap, 3),
                "positive_control_pos0_standard": entry["pos0"]["holdout_binary"],
                "positive_control_pos_mid_shifted": entry["pos_mid_shift"]["shifted_pair_binary"],
            }
            per_delay[f"seed_{seed}"] = entry
            print(f"[ب] delay={delay} seed={seed}: " +
                  " ".join(f"{n}={entry[n]['holdout_binary']}/{entry[n]['shifted_pair_binary']}"
                           for n in ("rnn", "attn_pos", "attn_nopos", "mean", "pos0")))
        per_delay["mean_across_seeds"] = {
            name: {"holdout_binary": round(float(np.mean(
                [per_delay[f"seed_{s}"][name]["holdout_binary"] for s in SEEDS])), 3),
                "shifted_pair_binary": round(float(np.mean(
                    [per_delay[f"seed_{s}"][name]["shifted_pair_binary"] for s in SEEDS])), 3),
                "holdout_test_nll": round(float(np.mean(
                    [per_delay[f"seed_{s}"][name]["holdout_test_nll"] for s in SEEDS])), 4)}
            for name in models}
        exp_b["delays"][f"delay_{delay}"] = per_delay
    results["experiments"]["order"] = exp_b

    # ═══════════ حساسية التهيئة وعزل مكوناتها (استكشافية — على التحقق) ═══════════
    # تدقيق v0.13: التهيئة ×3 مكّنت attn_pos من تعلم الترتيب (تحقق 1.000)؛
    # هنا نعيد إنتاجها ونعزل المكون المضخم (kq/v/pos) — ولا يُلمس holdout.
    sens = {"purpose": "حساسية تعلم الترتيب لمقياس تهيئة attn_pos وعزل المكون",
            "eval_set": "validation (استكشافية — holdout غير مستخدم للاختيار)",
            "amplify_values": [1, 3],
            "isolation_variants": {"all3": "كل المجموعات ×3",
                                   "kq_only3": "المفاتيح والاستعلام فقط ×3",
                                   "v_only3": "القيم فقط ×3",
                                   "pos_only3": "الترميز الموضعي فقط ×3"},
            "delays": {}}
    for delay in DELAYS_ORDER:
        per_delay = {"grad_norms_at_init": {}}
        for seed in SEEDS:
            rng_tr = np.random.default_rng(7000 + seed)
            rng_val = np.random.default_rng(8000 + seed)
            seen = set()
            tr_pairs = draw_matched_pairs(delay, TRAIN_PAIRS, rng_tr, synth_vocab, seen)
            va_pairs = draw_matched_pairs(delay, VAL_PAIRS, rng_val, synth_vocab, seen)
            val_prompt_sets = ([p[0] for p in va_pairs], [p[2] for p in va_pairs])
            tr_seqs = [p[0] + [p[1]] for p in tr_pairs] + [p[2] + [p[3]] for p in tr_pairs]
            answers = [out_pos_s[s[-1]] for s in tr_seqs]
            # معايير تدرج التهيئة (فرضية ضعف إشارة توجيه الانتباه)
            m0 = AttnReader(V_s, V_out_s, seed, use_pos=True)
            _, g0 = m0._loss_and_grads(tr_seqs, answers)
            per_delay["grad_norms_at_init"][f"seed_{seed}"] = {
                k: round(float(np.linalg.norm(v)), 6)
                for k, v in g0.items() if k in ("Wk", "wq", "Wv", "P", "Wo")}
            variants = {
                "amp1": dict(amplify=1.0),
                "all3": dict(amplify=3.0),
                "kq_only3": dict(amplify=1.0, amplify_map={"Wk": 3.0, "wq": 3.0}),
                "v_only3": dict(amplify=1.0, amplify_map={"Wv": 3.0}),
                "pos_only3": dict(amplify=1.0, amplify_map={"P": 3.0}),
            }
            if delay == 4:  # عزلات موسعة (تدقيق v0.13.1)
                variants.update({
                    "E_only3": dict(amplify=1.0, amplify_map={"E": 3.0}),
                    "Wo_only3": dict(amplify=1.0, amplify_map={"Wo": 3.0}),
                    "E_Wo3": dict(amplify=1.0, amplify_map={"E": 3.0, "Wo": 3.0}),
                    "E_P3": dict(amplify=1.0, amplify_map={"E": 3.0, "P": 3.0}),
                    "kqv3": dict(amplify=1.0, amplify_map={"Wk": 3.0, "wq": 3.0, "Wv": 3.0}),
                    "kqp3": dict(amplify=1.0, amplify_map={"Wk": 3.0, "wq": 3.0, "P": 3.0}),
                })
            block = {}
            for vname, kw in variants.items():
                model = AttnReader(V_s, V_out_s, seed, use_pos=True, **kw)
                tl = 0.0
                for _ in range(EPOCHS):
                    tl = model.train_epoch(tr_seqs, answers, LR)
                va = ch6.evaluate(model, val_prompt_sets[0], id_a, out_pos_s, id_a, id_b)
                vb = ch6.evaluate(model, val_prompt_sets[1], id_b, out_pos_s, id_a, id_b)
                stable = bool(np.isfinite(tl))
                block[vname] = {"val_binary": round((va[1] + vb[1]) / 2, 3) if stable else None,
                                "final_train_nll": round(tl, 4) if stable else "NaN(unstable)"}
            # مسبح معدل التعلم (تهيئة أصلية ×1) — تأخير 4 فقط: هل ×3 مجرد
            # رفع فعلي للمعدل؟ (تدقيق v0.13.1: الفرضية المرشحة للنفي)
            if delay == 4:
                lr_sweep = {}
                for lr_v in (0.3, 0.9, 3.0, 9.0, 27.0):
                    m2 = AttnReader(V_s, V_out_s, seed, use_pos=True)
                    t2 = 0.0
                    for _ in range(EPOCHS):
                        t2 = m2.train_epoch(tr_seqs, answers, lr_v)
                    va2 = ch6.evaluate(m2, val_prompt_sets[0], id_a, out_pos_s, id_a, id_b)
                    vb2 = ch6.evaluate(m2, val_prompt_sets[1], id_b, out_pos_s, id_a, id_b)
                    stable = bool(np.isfinite(t2))
                    lr_sweep[str(lr_v)] = {
                        "val_binary": round((va2[1] + vb2[1]) / 2, 3) if stable else None,
                        "final_train_nll": round(t2, 4) if stable else "NaN(unstable)",
                        "stable": stable}
                per_delay[f"lr_sweep_seed_{seed}"] = lr_sweep
            per_delay[f"seed_{seed}"] = block
            print(f"[س] delay={delay} seed={seed}: " +
                  " ".join(f"{v}={block[v]['val_binary']}" for v in variants))
        per_delay["mean_across_seeds"] = {
            v: round(float(np.mean([per_delay[f"seed_{s}"][v]["val_binary"]
                                    for s in SEEDS])), 3)
            for v in variants}
        sens["delays"][f"delay_{delay}"] = per_delay
    results["experiments"]["order"]["init_sensitivity"] = sens
    results["total_seconds"] = round(time.perf_counter() - t0, 2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1) + "\n",
                   encoding="utf-8", newline="\n")
    print(f"written: {OUT.relative_to(ROOT)} | {results['total_seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
