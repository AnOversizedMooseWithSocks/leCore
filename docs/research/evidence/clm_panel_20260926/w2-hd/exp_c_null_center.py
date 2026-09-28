"""exp_c_null_center.py -- REAL WORDING (CLINC150), in the engine's HYPERVECTOR text space: the hashed
char 3-5-gram bundle (holographic_systemone.hashed_ngram_encode, dim 2048) -- the reflex's 'ngram' key and
RecordCodec's text filler. NOT the MeaningIndex (that is a sparse IDF lexical index, not hypervectors).

Questions: (1) how anisotropic is this space (the 'narrow cone')? (2) does centering (subtract the running
mean hypervector) help ranking and/or abstention? (3) does a NULL prototype (centroid of out-of-scope TRAIN
questions) competing in the argmax cut out-of-scope serves at equal in-scope coverage?
Prototypes: centroid of K=20 train wordings per intent (the E1.1 acceptance setting). Thresholds are chosen
on VAL, reported on TEST.
"""
import sys, json, time
import numpy as np
sys.path.insert(0, "/home/claude/leCore")
from holographic.agents_and_reasoning.holographic_systemone import hashed_ngram_encode

DATA = "/tmp/claude-0/-home-claude/9ee53278-d85f-54b0-979e-17c930fe54e2/scratchpad/data/clinc150_full.json"
K = 20
T0 = time.perf_counter()
d = json.load(open(DATA))
enc = hashed_ngram_encode(dim=2048)


def E(texts):
    X = np.stack([enc(t) for t in texts])
    return X / np.linalg.norm(X, axis=1, keepdims=True)


intents = sorted({i for _, i in d["train"]})
by = {}
for t, i in d["train"]:
    by.setdefault(i, []).append(t)
tr_txt, tr_lab = [], []
for c, i in enumerate(intents):
    for t in by[i][:K]:
        tr_txt.append(t); tr_lab.append(c)
Xtr = E(tr_txt); ytr = np.array(tr_lab)
Xoos_tr = E([t for t, _ in d["oos_train"]])
Xva = E([t for t, _ in d["val"]]); yva = np.array([intents.index(i) for _, i in d["val"]])
Xva_o = E([t for t, _ in d["oos_val"]])
Xte = E([t for t, _ in d["test"]]); yte = np.array([intents.index(i) for _, i in d["test"]])
Xte_o = E([t for t, _ in d["oos_test"]])
t_enc = time.perf_counter() - T0


def unit(X):
    return X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), 1e-12)


def protos(X, y):
    return unit(np.stack([X[y == c].mean(0) for c in range(len(intents))]))


def auroc(pos, neg):
    s = np.concatenate([pos, neg]); lab = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    o = np.argsort(s, kind="stable"); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    return float((r[lab == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def arm(name, center=False, null=False):
    mu = Xtr.mean(0) if center else np.zeros(Xtr.shape[1])
    f = (lambda X: unit(X - mu))
    P = protos(f(Xtr), ytr)
    N = unit(f(Xoos_tr).mean(0)) if null else None

    def score(X):
        S = f(X) @ P.T
        top = S.max(1); pred = S.argmax(1)
        if null:
            nul = f(X) @ N
            abst = nul >= top                      # NULL wins the argmax -> "none of these"
        else:
            abst = np.zeros(len(X), bool)
        srt = np.sort(S, 1)
        gap = srt[:, -1] - srt[:, -2]
        return pred, top, gap, abst
    pv, sv, gv, av = score(Xva); pvo, svo, gvo, avo = score(Xva_o)
    pt, st, gt, at = score(Xte); pto, sto, gto, ato = score(Xte_o)
    # threshold on VAL: the score floor at which val OOS served == 2% (NULL abstentions count as refused)
    cand = np.sort(np.concatenate([sv, svo]))
    th = None
    for x in cand:
        if ((svo >= x) & ~avo).mean() <= 0.02:
            th = x; break
    served_in = (st >= th) & ~at; served_o = (sto >= th) & ~ato
    return {"arm": name, "test_top1": round(float((pt == yte).mean()), 4),
            "auroc_inscope_vs_oos_topscore": round(auroc(st, sto), 4),
            "auroc_correct_vs_wrong_topscore": round(auroc(st[pt == yte], st[pt != yte]), 4),
            "null_abstains_inscope": round(float(at.mean()), 4), "null_abstains_oos": round(float(ato.mean()), 4),
            "at_val_oos2pct_threshold": {"test_inscope_served_correct": round(float((served_in & (pt == yte)).mean()), 4),
                                         "test_inscope_served_wrong": round(float((served_in & (pt != yte)).mean()), 4),
                                         "test_oos_served": round(float(served_o.mean()), 4)}}


def aniso(X, y, center):
    rng = np.random.default_rng(0)
    Z = unit(X - Xtr.mean(0)) if center else X
    i = rng.integers(len(Z), size=4000); j = rng.integers(len(Z), size=4000)
    m = y[i] != y[j]
    return round(float(np.einsum("ij,ij->i", Z[i[m]], Z[j[m]]).mean()), 4)


res = {"encode_s": round(t_enc, 1), "K_per_intent": K, "dim": 2048,
       "anisotropy_mean_cos_between_different_intents": {"raw": aniso(Xte, yte, False), "centered": aniso(Xte, yte, True)},
       "arms": [arm("raw"), arm("centered", center=True), arm("raw+NULL", null=True), arm("centered+NULL", center=True, null=True)]}
res["cpu_s"] = round(time.perf_counter() - T0, 1)
print(json.dumps(res, indent=1))
json.dump(res, open(__file__.replace(".py", ".json"), "w"), indent=1)
