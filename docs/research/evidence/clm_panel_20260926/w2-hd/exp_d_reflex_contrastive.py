"""exp_d_reflex_contrastive.py -- how should the reflex experience trace (lever7.DisplacementTrace) store a
CORRECTION -- the served answer a was wrong, the truth is y?

Today (p27 reflex_learn): write(k, y) only. The delta rule corrects ALONG y ONLY (by design: the off-v part
of a prediction is crosstalk), so the wrong atom a written earlier keeps its full strength; and a second
correction write(k, y) is SKIPPED as low-surprise once y is stored. Hypothesis: the correction is a coin flip
that can never be repaired.

Arms, all on the REAL DisplacementTrace class (dim 2048), keys = hashed char-n-gram vectors of real CLINC150
questions (the reflex's 'ngram' key), labels = derived atoms, 40 background (question, intent) pairs per trace
(load ~0.02, under the live 0.03 tile advisory):
  current   write(k, y)                                               (what reflex_learn does)
  signed    write(k, y) then add bind(k, -s_a * a): the along-atom negative, s_a = <pred, a>/(rho |a|^2)
            -- the InfoNCE 'onehot - p' step restricted to {truth, served} atoms; never writes crosstalk back
  outcome   write(k, y) then add bind(k, NEG (*) a): the rejection kept as DATA under an OUTCOME role;
            read vetoes a label whose NEG evidence beats its positive evidence
Case A: the wrong lesson was written at the SAME key (a noisy verdict later corrected).
Case B: the wrong answer leaked from a NEIGHBOUR key k' (cos 0.8, synthetic mix) whose own truth IS a;
        after the correction at k, does k still read y and k' still read a?
"""
import sys, json, time
import numpy as np
sys.path.insert(0, "/home/claude/leCore")
from holographic.agents_and_reasoning.holographic_lever7 import DisplacementTrace
from holographic.agents_and_reasoning.holographic_ai import bind, unbind, derived_atom
from holographic.agents_and_reasoning.holographic_systemone import hashed_ngram_encode

DATA = "/tmp/claude-0/-home-claude/9ee53278-d85f-54b0-979e-17c930fe54e2/scratchpad/data/clinc150_full.json"
D = 2048
T0 = time.perf_counter()
d = json.load(open(DATA))
intents = sorted({i for _, i in d["train"]})
L = np.stack([derived_atom(0, "answer:" + i, D) for i in intents])
NEG = derived_atom(0, "role:outcome_neg", D, unitary=True)
enc = hashed_ngram_encode(dim=D)
rng = np.random.default_rng(0)
pool = [d["train"][i] for i in rng.choice(len(d["train"]), 1200, replace=False)]
cache = {}


def key(t):
    if t not in cache:
        v = enc(t); cache[t] = v / np.linalg.norm(v)
    return cache[t]


def read(tr, k, arm):
    r = tr.read(k)
    s = L @ (r / (np.linalg.norm(r) + 1e-12))
    if arm == "outcome":
        rn = unbind(r, NEG)
        neg = L @ (rn / (np.linalg.norm(rn) + 1e-12))
        s = np.where(neg > s, -1.0, s)                 # a label rejected here more strongly than it is supported
    return int(np.argmax(s)), float(np.sort(s)[-1] - np.sort(s)[-2])


def correct(tr, k, y, a, arm):
    tr.write(k, L[y])
    if arm == "signed":
        pred = tr.read(k)
        s_a = float(np.dot(pred, L[a])) / (tr._rho * float(np.dot(L[a], L[a])))
        s_a = min(max(s_a, 0.0), 1.0)
        if s_a > 0:
            tr._trace = tr._trace + bind(k, -s_a * L[a])
            tr._audit.append((k.tolist(), (-s_a * L[a]).tolist()))   # the negative must be IN the audit or replay/retile erases it
    elif arm == "outcome":
        tr._trace = tr._trace + bind(k, bind(NEG, L[a]))
        tr._audit.append((k.tolist(), bind(NEG, L[a]).tolist()))


def trial(t, arm, case, n_corrections=1):
    r = np.random.default_rng(100 + t)
    idx = r.choice(len(pool), 42, replace=False)
    bg = [pool[i] for i in idx[:40]]
    q_text, q_int = pool[idx[40]]
    y = intents.index(q_int)
    a = int(r.integers(len(intents)))
    while a == y:
        a = int(r.integers(len(intents)))
    tr = DisplacementTrace(dim=D, seed=0)
    for bt, bi in bg:
        tr.write(key(bt), L[intents.index(bi)])
    k = key(q_text)
    if case == "A":
        tr.write(k, L[a])                                   # the wrong lesson at the same key
        kp = None
    else:
        noise = r.standard_normal(D); noise -= noise.dot(k) * k; noise /= np.linalg.norm(noise)
        kp = 0.8 * k + 0.6 * noise                          # cos(k, k') = 0.8
        tr.write(kp, L[a])                                  # the neighbour's own (correct) lesson
    before = read(tr, k, arm)[0] == y
    for _ in range(n_corrections):
        correct(tr, k, y, a, arm)
    got, margin = read(tr, k, arm)
    out = {"k_reads_truth": got == y, "k_reads_wrong": got == a, "k_before": before, "margin": margin}
    if kp is not None:
        out["kprime_keeps_its_answer"] = read(tr, kp, arm)[0] == a
    # background damage: fraction of the 40 background pairs still read back right
    out["bg_ok"] = np.mean([read(tr, key(bt), arm)[0] == intents.index(bi) for bt, bi in bg[:20]])
    # replay faithfulness of the audit (the class's own replay) -- does the negative survive a retile?
    rep = tr.replay()
    out["replay_agrees"] = read(rep, k, arm)[0] == got
    return out


res = {"dim": D, "trials": 150, "background_pairs": 40}
for case in ("A", "B"):
    for arm in ("current", "signed", "outcome"):
        for nc in ((1, 3) if case == "A" else (1,)):
            rows = [trial(t, arm, case, nc) for t in range(150)]
            agg = {k: round(float(np.mean([r[k] for r in rows])), 3) for k in rows[0]}
            res["case%s_%s_corr%d" % (case, arm, nc)] = agg
res["cpu_s"] = round(time.perf_counter() - T0, 1)
print(json.dumps(res, indent=1))
json.dump(res, open(__file__.replace(".py", ".json"), "w"), indent=1)
