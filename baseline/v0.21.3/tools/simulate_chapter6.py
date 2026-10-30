#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
simulate_chapter6.py — الفصل السادس: الذاكرة المتكررة (RNN) وحدودها.

جزء أ — نمذجة لغوية بالمحارف على نفس المتن (نفس المفردة عبر CH-3/4/5/6):
  h_t = tanh(W_x @ e(x_t) + W_h @ h_{t-1} + b_h)،  h_0 = 0 لكل رسالة (إعادة تعيين الحالة)
  softmax على مساحة الإخراج المضغوطة (تستبعد <s>) عبر out_pos — نفس اصطلاح CH-5.
  المفردة من التدريب فقط؛ val/test تُرمَّز عبر <unk> (بلا تسرب مفردة).

جزء ب — مهمة ذاكرة اصطناعية مضبوطة:
  تسلسل = [فئة (أ أو م)] + حشو × delay + [؟] + [الجواب = الحرف الأول]
  الخسارة على موضع الإجابة فقط (بعد ؟) — الحشو عشوائي فلا تعلّم منه.

v3.0 — تصحيحات تدقيق v0.10 المستقل:
  1) predict_next_logits(): يستهلك كل رموز البادئة بما فيها ؟ قبل قراءة التنبؤ
     (v2.1 كان يقرأ التوقع بعد آخر حشو وقبل ؟ — off-by-one).
  2) مسار تقييم واحد evaluate() للدقة (argmax/binary) وخسارة الاختبار عند موضع
     الإجابة — منفصلة عن منحنى التدريب.
  3) مرجع Oracle صريح يمر عبر نفس دالة التقييم (يشترط 1.000) + فحص مضاد
     بتبديل التسميات (يشترط 0.000).
  4) أمثلة اختبار بلا تطابق حرفي لمسار الحشو مع أي تسلسل تدريب (مع عدّاد
     الرفض المفلتر).
  5) تأخيرات 2/4/5/6/7/8/16 — لرسم شكل الانتقال لا افتراضه.
  فحوص السلامة صارمة: أي إخفاق = خروج برمز غير صفري.

v3.2 — ملاحظتا تدقيق v0.11:
  1) التوليد: نموذج البذرة 42 صراحة + فصل model_init_seed عن sampling_seed
     + بادئة `<s>` واحدة كالتدريب (كانت اثنتين).
  2) الجزء ج — نموذج واحد يُدرَّب على مزيج أطوال (2..8) ويُقاس داخل مجاله
     وعند تأخير محجوز (16): تمييز «التعلم عند طول» عن «التعميم على أطوال» —
     الجزء ب يبقى منحنى تعلم عند طول ثابت (نموذج جديد لكل تأخير).
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "content" / "chapter-03" / "examples.json"
OUT = ROOT / "content" / "chapter-06" / "results.json"
VERSION = "simulate_chapter6-3.2"

START, UNK = "<s>", "<unk>"
EMB_DIM, HID_DIM = 8, 16
EPOCHS_REAL, LR_REAL = 100, 0.1
EPOCHS_SYNTH, LR_SYNTH = 300, 0.3
SYNTH_TRAIN_PER_CLASS, SYNTH_TEST_PER_CLASS = 120, 40
SEEDS = [42, 43, 44]
DELAYS = (2, 4, 5, 6, 7, 8, 16)
CURVE_EPOCHS = (1, 25, 50, 75, 100)
SYNTH_CURVE_EPOCHS = (1, 50, 100, 200, 300)
# حروف الحشو: 27 حرفاً — بلا أ ولا م (حرفا الفئة) ولا ؟
FILLER = "ابتثجحخدذرزسشصضطظعغفقكلنهوي"
CLASS_A, CLASS_B, QUERY = "أ", "م", "؟"


def build_vocab(msgs):
    """مفردة من الرسائل المعطاة فقط + <s> + <unk> (مرتبة حتمياً)."""
    chars = set()
    for m in msgs:
        chars.update(m)
        chars.add(START)
    v = sorted(chars)
    if UNK not in v:
        v.append(UNK)
    return {c: i for i, c in enumerate(v)}


def out_space(vocab):
    """مساحة الإخراج المضغوطة: كل فهارس المفردة ما عدا <s> (سياق فقط)."""
    out_ids = sorted(i for c, i in vocab.items() if c != START)
    out_pos = {full: pos for pos, full in enumerate(out_ids)}
    return out_ids, out_pos


def predict_next_logits(model, prefix_ids):
    """لوغيتات التوقع **بعد استهلاك كل رموز البادئة** (بما فيها ؟) —
    موضع التقييم مطابق لموضع هدف التدريب. (تصحيح off-by-one في v2.1)."""
    h = np.zeros(HID_DIM)
    for tok in prefix_ids:
        h = np.tanh(model.Wx @ model.E[tok] + model.Wh @ h + model.bh)
    return model.Wy @ h + model.by


class SimpleRNN:
    """h_t = tanh(Wx e_t + Wh h_{t-1} + bh)؛ softmax عبر Wy [V_out, HID].
    كل زوج تدريب = (ids، أهداف مضغوطة، أوزان مواضع) — الوزن 0 يتخطى الخسارة
    والتدرج المباشر d مع بقاء انتشار التدرج عبر Wh (تدفق زمني كامل)."""

    def __init__(self, V_ctx, V_out, seed):
        rng = np.random.default_rng(seed)
        self.E = rng.normal(0, 0.1, (V_ctx, EMB_DIM))
        self.Wx = rng.normal(0, 0.1, (HID_DIM, EMB_DIM))
        self.Wh = np.eye(HID_DIM) * 0.9 + rng.normal(0, 0.05, (HID_DIM, HID_DIM))
        self.bh = np.zeros(HID_DIM)
        self.Wy = rng.normal(0, 0.1, (V_out, HID_DIM))
        self.by = np.zeros(V_out)
        self.V_out = V_out

    def predict_next_logits(self, prefix_ids):
        return predict_next_logits(self, prefix_ids)

    def forward(self, ids):
        """ids: فهارس كاملة. يعيد H (طول n) و Zs (طول n-1؛ Zs[t] يتنبأ بالهدف t
        بعد استهلاك ids[0..t])."""
        h = np.zeros(HID_DIM)
        H, Zs = [h], []
        for t in range(len(ids) - 1):
            e = self.E[ids[t]]
            h = np.tanh(self.Wx @ e + self.Wh @ h + self.bh)
            H.append(h)
            Zs.append(self.Wy @ h + self.by)
        return H, Zs

    def loss(self, pairs):
        """متوسط NLL عبر المواضع الموزونة فقط (أوزان 1 افتراضياً)."""
        total, n = 0.0, 0
        for ids, ys, ws in pairs:
            _, Zs = self.forward(ids)
            for t, z in enumerate(Zs):
                if ws[t] == 0:
                    continue
                zz = z - z.max()
                p = np.exp(zz)
                p /= p.sum()
                total -= np.log(max(p[ys[t]], 1e-12))
                n += 1
        return total / max(n, 1)

    def train_epoch(self, pairs, lr):
        g = {k: np.zeros_like(v) for k, v in
             [("E", self.E), ("Wx", self.Wx), ("Wh", self.Wh),
              ("bh", self.bh), ("Wy", self.Wy), ("by", self.by)]}
        total_loss, total_n = 0.0, 0
        for ids, ys, ws in pairs:
            H, Zs = self.forward(ids)
            dh_next = np.zeros(HID_DIM)
            for t in reversed(range(len(ys))):
                if ws[t] > 0:
                    z = Zs[t]
                    zz = z - z.max()
                    p = np.exp(zz)
                    p /= p.sum()
                    total_loss -= np.log(max(p[ys[t]], 1e-12))
                    total_n += 1
                    d = p.copy()
                    d[ys[t]] -= 1.0
                    d *= ws[t]
                    g["Wy"] += np.outer(d, H[t + 1])
                    g["by"] += d
                    dh = (self.Wy.T @ d + self.Wh.T @ dh_next) * (1 - H[t + 1] ** 2)
                else:
                    dh = (self.Wh.T @ dh_next) * (1 - H[t + 1] ** 2)
                g["Wx"] += np.outer(dh, self.E[ids[t]])
                g["Wh"] += np.outer(dh, H[t])
                g["bh"] += dh
                g["E"][ids[t]] += self.Wx.T @ dh
                dh_next = dh
        n = max(total_n, 1)
        for k in g:
            g[k] /= n
        self.E -= lr * g["E"]
        self.Wx -= lr * g["Wx"]
        self.Wh -= lr * g["Wh"]
        self.bh -= lr * g["bh"]
        self.Wy -= lr * g["Wy"]
        self.by -= lr * g["by"]
        return total_loss / n

    def generate(self, vocab, out_ids, prompt, max_chars, seed):
        """عينة توليد: بادئة واحدة `<s>` (كالتدريب — لا رمزان)، فكّ فهرس الإخراج
        عبر out_ids (لا خلط بين الفهرسين)، وتمرير السياق بالفهارس الكاملة."""
        rng = np.random.default_rng(seed)
        inv_full = {i: c for c, i in vocab.items()}
        ctx = [vocab[START]] + [vocab.get(c, vocab[UNK]) for c in prompt]
        h = np.zeros(HID_DIM)
        for c in ctx[:-1]:
            h = np.tanh(self.Wx @ self.E[c] + self.Wh @ h + self.bh)
        cur = ctx[-1]
        out = []
        for _ in range(max_chars):
            h = np.tanh(self.Wx @ self.E[cur] + self.Wh @ h + self.bh)
            z = self.Wy @ h + self.by
            z = z - z.max()
            p = np.exp(z)
            p /= p.sum()
            nxt = int(rng.choice(self.V_out, p=p))
            ch = inv_full[out_ids[nxt]]
            out.append("?" if ch == UNK else ch)
            cur = out_ids[nxt]
        return "".join(out)


class OracleMemory:
    """مرجع ذاكرة صريحة: يقرأ الحرف الأول من البادئة نفسها ويركز كل كتلته عليه.
    لا يحاكي RNN — لكنه يمارس **نفس** دالة التقييم عبر predict_next_logits."""

    def __init__(self, out_pos, n_out):
        self.out_pos = out_pos
        self.n_out = n_out

    def predict_next_logits(self, prefix_ids):
        logits = np.full(self.n_out, -30.0)
        logits[self.out_pos[prefix_ids[0]]] = 30.0
        return logits


def evaluate(model, prompts, label_full, out_pos, id_a, id_b):
    """مسار التقييم الوحيد: (دقة argmax، دقة ثنائية، NLL اختبار عند موضع الإجابة)."""
    hit_argmax = hit_binary = 0
    nll_sum = 0.0
    tgt = out_pos[label_full]
    for s in prompts:
        z = model.predict_next_logits(s)
        zz = z - z.max()
        p = np.exp(zz)
        p /= p.sum()
        nll_sum -= np.log(max(p[tgt], 1e-12))
        if int(np.argmax(p)) == tgt:
            hit_argmax += 1
        guess = id_a if p[out_pos[id_a]] >= p[out_pos[id_b]] else id_b
        if guess == label_full:
            hit_binary += 1
    n = len(prompts)
    return hit_argmax / n, hit_binary / n, nll_sum / n


def make_pairs(msgs, vocab, out_pos):
    """لكل رسالة: (فهارس كاملة، أهداف مضغوطة، أوزان 1) — نمذجة لغوية كاملة."""
    pairs = []
    for m in msgs:
        ids = [vocab[START]] + [vocab.get(c, vocab[UNK]) for c in m]
        ys = [out_pos[i] for i in ids[1:]]
        pairs.append((ids, ys, [1.0] * len(ys)))
    return pairs


def synth_seq(first_id, delay, rng, synth_vocab):
    fill = [synth_vocab[c] for c in rng.choice(list(FILLER), size=delay, replace=True)]
    return [first_id] + fill + [synth_vocab[QUERY], first_id]


def synth_pairs(seqs, out_pos):
    """أوزان: 0 لكل المواضع عدا موضع الإجابة الأخير (التنبؤ بعد ؟)."""
    pairs = []
    for s in seqs:
        ys = [out_pos[i] for i in s[1:]]
        ws = [0.0] * len(ys)
        ws[-1] = 1.0
        pairs.append((s, ys, ws))
    return pairs


def draw_test_prompts(first_id, delay, rng_te, synth_vocab, train_filler):
    """سحبات اختبار حصرية: يُرفض أي مسار حشو ورد حرفياً في التدريب (الفئتان)."""
    prompts, rejected = [], 0
    while len(prompts) < SYNTH_TEST_PER_CLASS:
        s = synth_seq(first_id, delay, rng_te, synth_vocab)
        if tuple(s[1:delay + 1]) in train_filler:
            rejected += 1
            continue
        prompts.append(s[:-1])          # البادئة حتى ؟ — الجواب لا يُدخل
    return prompts, rejected


def main():
    raw = SRC.read_bytes()
    input_hash = hashlib.sha256(raw).hexdigest()
    src = json.loads(SRC.read_text(encoding="utf-8"))
    t0 = time.perf_counter()
    results = {"simulator_version": VERSION, "input_hash": input_hash, "languages": {}}

    # ───────────────────────── الجزء أ: RNN على المتن الحقيقي ─────────────────────────
    for lang in ("ar", "en"):
        cfg = src["corpus"][lang]
        vocab = build_vocab(cfg["train"])          # التدريب فقط — لا تسرب مفردة
        out_ids, out_pos = out_space(vocab)
        V_ctx, V_out = len(vocab), len(out_ids)
        n_unseen_chars = len({c for split in ("val", "test") for m in cfg[split]
                              for c in m if c not in vocab})
        tr_pairs = make_pairs(cfg["train"], vocab, out_pos)
        va_pairs = make_pairs(cfg["val"], vocab, out_pos)
        te_pairs = make_pairs(cfg["test"], vocab, out_pos)

        rnn_seeds = {}
        models_by_seed = {}
        for seed in SEEDS:
            m = SimpleRNN(V_ctx, V_out, seed)
            models_by_seed[seed] = m
            curve = []
            for ep in range(1, EPOCHS_REAL + 1):
                tl = m.train_epoch(tr_pairs, LR_REAL)
                if ep in CURVE_EPOCHS:
                    curve.append({"epoch": ep, "train_nll": round(tl, 4)})
            rnn_seeds[str(seed)] = {
                "val_nll": round(m.loss(va_pairs), 4),
                "test_nll": round(m.loss(te_pairs), 4),
                "curve": curve}

        vals = [v["val_nll"] for v in rnn_seeds.values()]
        tests = [v["test_nll"] for v in rnn_seeds.values()]
        params = int(sum(x.size for x in
                         [m.E, m.Wx, m.Wh, m.bh, m.Wy, m.by]))
        # التوليد من نموذج البذرة 42 صراحة؛ 42 بذرة السحب أثناء التوليد — بذرتان منفصلتان
        gen = models_by_seed[42].generate(vocab, out_ids, "أو" if lang == "ar" else "I'", 60, 42)

        results["languages"][lang] = {
            "rnn": {"seeds": rnn_seeds,
                    "mean_val": round(float(np.mean(vals)), 4),
                    "mean_test": round(float(np.mean(tests)), 4),
                    "params": params,
                    "generation": {"text": gen, "model_init_seed": 42,
                                   "sampling_seed": 42,
                                   "prompt": "أو" if lang == "ar" else "I'",
                                   "start_symbols": 1}},
            "units": {"train_msgs": len(tr_pairs), "val_msgs": len(va_pairs),
                      "test_msgs": len(te_pairs)},
            "vocab": {"context": V_ctx, "output": V_out},
            "vocab_leak_check": {"vocab_source": "train_only",
                                 "unseen_chars_in_val_test": n_unseen_chars,
                                 "encoded_as": UNK},
        }
        print(f"== {lang} == V_ctx={V_ctx} V_out={V_out} "
              f"val={rnn_seeds['42']['val_nll']} test={rnn_seeds['42']['test_nll']} "
              f"params={params} unseen={n_unseen_chars} gen={gen[:24]!r}")

    # ───────────────────────── الجزء ب: مهمة الذاكرة الاصطناعية ─────────────────────────
    synth_vocab = build_vocab([CLASS_A + CLASS_B + FILLER + QUERY])
    out_ids_s, out_pos_s = out_space(synth_vocab)
    V_s, V_out_s = len(synth_vocab), len(out_ids_s)
    id_a, id_b = synth_vocab[CLASS_A], synth_vocab[CLASS_B]
    oracle = OracleMemory(out_pos_s, V_out_s)

    memory_probe = {"task": "استرجاع الحرف الأول بعد حشو بطول delay ثم ؟",
                    "eval_position": "التنبؤ بعد استهلاك كل رموز البادئة بما فيها ؟ "
                                     "(predict_next_logits) — مطابق لموضع هدف التدريب",
                    "loss_masking": "الخسارة على موضع الإجابة فقط (بعد ؟) — الحشو عشوائي",
                    "test_dedup": "أمثلة الاختبار بلا تطابق حرفي لمسار الحشو مع أي تسلسل "
                                  "تدريب (الفئتان معاً)؛ عدّاد الرفض مسجل",
                    "classes": [CLASS_A, CLASS_B],
                    "filler_alphabet_size": len(FILLER),
                    "train_per_class": SYNTH_TRAIN_PER_CLASS,
                    "test_per_class": SYNTH_TEST_PER_CLASS,
                    "epochs": EPOCHS_SYNTH, "lr": LR_SYNTH, "seeds": SEEDS,
                    "delays": list(DELAYS),
                    "random_answer_nll": round(float(np.log(V_out_s)), 3),
                    "delays_data": {}}

    # فحص مضاد بتبديل التسميات: أمثلة فئة (أ) تُقيَّم بتوقع (م).
    # الأوراكل يقرأ (أ) من المدخل فيجيب بها؛ مسار تقييم لا يقارن بالتسمية
    # فعلاً (أ يعيد «صحيح» دائماً) سيسقط هنا إلى 0.000.
    rng_probe = np.random.default_rng(999)
    swap_prompts = [synth_seq(id_a, 4, rng_probe, synth_vocab)[:-1] for _ in range(10)]
    swap_binary = evaluate(oracle, swap_prompts, id_b, out_pos_s, id_a, id_b)[1]
    memory_probe["oracle_label_swap_control"] = round(swap_binary, 3)

    hard_failures = []
    for delay in DELAYS:
        per_seed = {"argmax": [], "binary": [], "acc_a": [], "acc_b": [],
                    "test_nll": [], "test_nll_a": [], "test_nll_b": []}
        untrained = {"argmax": [], "binary": [], "test_nll": []}
        rejected_total = 0
        curves = []
        for seed in SEEDS:
            rng_tr = np.random.default_rng(1000 + seed)
            rng_te = np.random.default_rng(2000 + seed)
            tr_seqs = ([synth_seq(id_a, delay, rng_tr, synth_vocab)
                        for _ in range(SYNTH_TRAIN_PER_CLASS)] +
                       [synth_seq(id_b, delay, rng_tr, synth_vocab)
                        for _ in range(SYNTH_TRAIN_PER_CLASS)])
            train_filler = {tuple(s[1:delay + 1]) for s in tr_seqs}
            prompts_a, rej_a = draw_test_prompts(id_a, delay, rng_te, synth_vocab, train_filler)
            prompts_b, rej_b = draw_test_prompts(id_b, delay, rng_te, synth_vocab, train_filler)
            rejected_total += rej_a + rej_b
            tr_pairs = synth_pairs(tr_seqs, out_pos_s)

            model = SimpleRNN(V_s, V_out_s, seed)
            ua = evaluate(model, prompts_a, id_a, out_pos_s, id_a, id_b)
            ub = evaluate(model, prompts_b, id_b, out_pos_s, id_a, id_b)
            untrained["argmax"].append((ua[0] + ub[0]) / 2)
            untrained["binary"].append((ua[1] + ub[1]) / 2)
            untrained["test_nll"].append((ua[2] + ub[2]) / 2)
            curve = []
            for ep in range(1, EPOCHS_SYNTH + 1):
                tl = model.train_epoch(tr_pairs, LR_SYNTH)
                if ep in SYNTH_CURVE_EPOCHS:
                    curve.append({"epoch": ep, "train_answer_nll": round(tl, 4)})
            curves.append(curve)
            # التقييم الصحيح: بعد استهلاك ؟ — عبر نفس مسار التقييم
            ta = evaluate(model, prompts_a, id_a, out_pos_s, id_a, id_b)
            tb = evaluate(model, prompts_b, id_b, out_pos_s, id_a, id_b)
            per_seed["argmax"].append((ta[0] + tb[0]) / 2)
            per_seed["binary"].append((ta[1] + tb[1]) / 2)
            per_seed["acc_a"].append(ta[1])
            per_seed["acc_b"].append(tb[1])
            per_seed["test_nll"].append((ta[2] + tb[2]) / 2)
            per_seed["test_nll_a"].append(ta[2])
            per_seed["test_nll_b"].append(tb[2])
            # الأوراكل عبر نفس المسار: يجب 1.000
            oa = evaluate(oracle, prompts_a, id_a, out_pos_s, id_a, id_b)[1]
            ob = evaluate(oracle, prompts_b, id_b, out_pos_s, id_a, id_b)[1]
            if oa != 1.0 or ob != 1.0:
                hard_failures.append(f"delay_{delay}/seed_{seed}: oracle={oa}/{ob} != 1.0")

        mean_binary = float(np.mean(per_seed["binary"]))
        memory_probe["delays_data"][f"delay_{delay}"] = {
            "n_test": 2 * SYNTH_TEST_PER_CLASS,
            "accuracy_argmax": round(float(np.mean(per_seed["argmax"])), 3),
            "accuracy_binary": round(mean_binary, 3),
            "accuracy_class_A_binary": round(float(np.mean(per_seed["acc_a"])), 3),
            "accuracy_class_B_binary": round(float(np.mean(per_seed["acc_b"])), 3),
            "per_seed_binary": [round(x, 3) for x in per_seed["binary"]],
            "per_seed_argmax": [round(x, 3) for x in per_seed["argmax"]],
            "test_answer_nll_mean": round(float(np.mean(per_seed["test_nll"])), 4),
            "test_answer_nll_std": round(float(np.std(per_seed["test_nll"])), 4),
            "per_seed_test_nll": [round(x, 4) for x in per_seed["test_nll"]],
            "per_seed_test_nll_a": [round(x, 4) for x in per_seed["test_nll_a"]],
            "per_seed_test_nll_b": [round(x, 4) for x in per_seed["test_nll_b"]],
            "untrained_baseline_binary": round(float(np.mean(untrained["binary"])), 3),
            "untrained_baseline_argmax": round(float(np.mean(untrained["argmax"])), 3),
            "untrained_baseline_test_nll": round(float(np.mean(untrained["test_nll"])), 3),
            "oracle_binary_through_eval_path": 1.0,
            "test_draws_rejected_as_collisions": rejected_total,
            "train_curve_mean": [{"epoch": c["epoch"],
                                  "train_answer_nll": round(float(np.mean(
                                      [cv[i]["train_answer_nll"] for cv in curves])), 4)}
                                 for i, c in enumerate(curves[0])],
        }
        d = memory_probe["delays_data"][f"delay_{delay}"]
        print(f"  delay={delay}: binary={d['accuracy_binary']} argmax={d['accuracy_argmax']} "
              f"test_nll={d['test_answer_nll_mean']}±{d['test_answer_nll_std']} "
              f"rejected={rejected_total}")

    # ──────────────── الجزء ج: نموذج واحد على مزيج أطوال (تدقيق v0.11) ────────────────
    # الجزء ب يدرّب نموذجاً جديداً لكل تأخير — سبع مهام مستقلة، لا ذاكرة نموذج واحد.
    # هنا يُدرَّب نموذج واحد لكل بذرة على مزيج أطوال، ثم يُقاس داخل مجال تدريبه
    # وعند تأخير محجوز خارجه — تمييز «التعلم عند طول» عن «التعميم على أطوال».
    MIXED_TRAIN_DELAYS = (2, 4, 5, 6, 7, 8)
    MIXED_HOLDOUT_DELAYS = (16,)
    MIXED_TRAIN_PER_CLASS_PER_DELAY = 60
    mixed = {"purpose": "نموذج واحد يُدرَّب على مزيج أطوال ويُقاس داخل مجاله وخارجه",
             "train_delays": list(MIXED_TRAIN_DELAYS),
             "holdout_delays": list(MIXED_HOLDOUT_DELAYS),
             "train_per_class_per_delay": MIXED_TRAIN_PER_CLASS_PER_DELAY,
             "test_per_class": SYNTH_TEST_PER_CLASS,
             "epochs": EPOCHS_SYNTH, "lr": LR_SYNTH, "seeds": SEEDS,
             "per_seed": {}, "means": {}}
    for seed in SEEDS:
        rng_tr = np.random.default_rng(3000 + seed)
        rng_te = np.random.default_rng(4000 + seed)
        tr_pairs = []
        seen_filler = set()
        for d in MIXED_TRAIN_DELAYS:
            seqs = []
            while len(seqs) < 2 * MIXED_TRAIN_PER_CLASS_PER_DELAY:
                first = id_a if len(seqs) % 2 == 0 else id_b
                s = synth_seq(first, d, rng_tr, synth_vocab)
                ft = tuple(s[1:d + 1])
                if ft in seen_filler:
                    continue
                seen_filler.add(ft)
                seqs.append(s)
            tr_pairs.extend(synth_pairs(seqs, out_pos_s))
        test_sets = {}
        for d in MIXED_TRAIN_DELAYS + MIXED_HOLDOUT_DELAYS:
            # التداخل يُستبعد ضد حشوات التدريب داخل المجال فقط؛ تأخير 16 محجوز بلا تدريب
            pool = seen_filler if d in MIXED_TRAIN_DELAYS else set()
            pa, _ = draw_test_prompts(id_a, d, rng_te, synth_vocab, pool)
            pb, _ = draw_test_prompts(id_b, d, rng_te, synth_vocab, pool)
            test_sets[d] = (pa, pb)
        model = SimpleRNN(V_s, V_out_s, seed)
        for _ in range(EPOCHS_SYNTH):
            model.train_epoch(tr_pairs, LR_SYNTH)
        block = {}
        for d, (pa, pb) in test_sets.items():
            ta = evaluate(model, pa, id_a, out_pos_s, id_a, id_b)
            tb = evaluate(model, pb, id_b, out_pos_s, id_a, id_b)
            block[f"delay_{d}"] = {
                "in_domain": d in MIXED_TRAIN_DELAYS,
                "binary": round((ta[1] + tb[1]) / 2, 3),
                "argmax": round((ta[0] + tb[0]) / 2, 3),
                "test_nll": round((ta[2] + tb[2]) / 2, 4)}
        mixed["per_seed"][str(seed)] = block
    for d in MIXED_TRAIN_DELAYS + MIXED_HOLDOUT_DELAYS:
        vb = [mixed["per_seed"][str(s)][f"delay_{d}"]["binary"] for s in SEEDS]
        vn = [mixed["per_seed"][str(s)][f"delay_{d}"]["test_nll"] for s in SEEDS]
        mixed["means"][f"delay_{d}"] = {
            "in_domain": d in MIXED_TRAIN_DELAYS,
            "binary_mean": round(float(np.mean(vb)), 3),
            "per_seed_binary": [round(x, 3) for x in vb],
            "test_nll_mean": round(float(np.mean(vn)), 4)}
    results["mixed_length_model"] = mixed
    print("mixed-length binary:", {k: v["binary_mean"] for k, v in mixed["means"].items()})

    # ── فحوص سلامة صارمة: الإخفاق = خروج برمز غير صفري ──
    for delay in DELAYS:
        blk = memory_probe["delays_data"][f"delay_{delay}"]
        if blk["oracle_binary_through_eval_path"] != 1.0:
            hard_failures.append(f"delay_{delay}: oracle عبر مسار التقييم != 1.0")
        if not 0.25 <= blk["untrained_baseline_binary"] <= 0.75:
            hard_failures.append(f"delay_{delay}: خط الأساس غير المدرّب خارج النطاق العشوائي")
        if blk["untrained_baseline_argmax"] > 0.15:
            hard_failures.append(f"delay_{delay}: argmax غير المدرّب أعلى من العشوائي")
    if memory_probe["oracle_label_swap_control"] != 0.0:
        hard_failures.append("فحص تبديل التسميات: الأوراكل لم يسقط إلى 0.0")
    d2 = memory_probe["delays_data"]["delay_2"]["accuracy_binary"]
    d4 = memory_probe["delays_data"]["delay_4"]["accuracy_binary"]
    if d2 < 0.9 or d4 < 0.9:
        hard_failures.append(f"التدريب ضعيف: delay_2={d2}, delay_4={d4} (< 0.9)")

    memory_probe["sanity_checks"] = (["FAIL: " + f for f in hard_failures]
                                     or ["PASS: كل الفحوص سليمة"])
    memory_probe["note"] = ("الدقة والخسارة عند موضع الإجابة (بعد استهلاك ؟) عبر مسار "
                            "تقييم واحد؛ binary إجبار بين حرفي الفئة (خط الأساس ≈ 0.5)، "
                            "argmax على كامل مساحة الإخراج (خط الأساس ≈ 1/V_out). "
                            "test_answer_nll خسارة اختبار — منحنى train_curve_mean تدريب فقط.")
    results["memory_probe"] = memory_probe
    results["provenance"] = {
        "input_hash": input_hash,
        "state_reset": "كل forward يبدأ h=0 — لا تسرب بين الرسائل",
        "output_excludes_start": True,
        "vocab_source": "train_only",
        "synth_train_test": "سحبات منفصلة (بذور 1000+seed للتدريب و2000+seed للاختبار) "
                            "مع استبعاد تطابقات مسار الحشو الحرفية",
        "wh_init": "هوية 0.9 + ضجيج — يسهل تدفق التدرج عبر التأخير (وثّق بشفافية)",
        "eval_fix": "v3.0: تصحيح off-by-one — التقييم بعد استهلاك ؟ (تدقيق مستقل v0.10)",
    }
    results["total_seconds"] = round(time.perf_counter() - t0, 2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1) + "\n",
                   encoding="utf-8", newline="\n")
    print(f"written: {OUT.relative_to(ROOT)} | {results['total_seconds']}s")
    print("sanity:", memory_probe["sanity_checks"])
    if hard_failures:
        print("فشل فحوص سلامة صارمة — exit 1")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
