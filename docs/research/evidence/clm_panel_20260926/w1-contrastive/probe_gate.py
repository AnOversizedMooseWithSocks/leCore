"""probe_gate.py -- w1-contrastive scratch probe (NOT repo code).

Question (sweep 181's open negative): the teacher-relative serve bar could not tell a NOISY TEACHER from
CONFUSABLE ROWS (Banking77 perfect teacher: 6.8% wrong at floor 0.90). Can the teacher's SELF-CONSISTENCY
(re-ask a small sample; a perfect teacher on confusable rows is consistent, a noisy one is not) estimate the
noise rate eps, and does a noise-corrected calibration P(correct|g) = (P(agree|g) - eps/m)/(1 - eps - eps/m)
open the gate for the noisy teacher WITHOUT opening it for the confusable-but-perfect case?

STATIC INDEX (isolates the gate, not the learning loop): MeaningIndex (repo code, unmodified) holds K wordings per
intent; a calibration stream is ranked once (answers(k=8)); teachers are simulated on the recorded candidates
exactly like tools/bench_meaning.py's ScriptedModel (perfect: the right candidate or 'new'; noisy r: with prob r
a uniformly random WRONG candidate). The gate is then applied to held-out test + out-of-scope questions.
"""
import csv, json, os, random, sys, time
import numpy as np

sys.path.insert(0, "/home/claude/leCore")
from holographic.agents_and_reasoning.holographic_meaning import MeaningIndex
from holographic.agents_and_reasoning.holographic_systemone import IsotonicCalibrator

DATA = "/tmp/claude-0/-home-claude/9ee53278-d85f-54b0-979e-17c930fe54e2/scratchpad/data"


def load(ds):
    cl = json.load(open(os.path.join(DATA, "clinc150_full.json")))
    if ds == "clinc":
        tr = {}
        for t, i in cl["train"]:
            tr.setdefault(i, []).append(t)
        return tr, [tuple(x) for x in cl["test"]], [t for t, _ in cl["oos_train"]], [t for t, _ in cl["oos_test"]]
    def rows(fn):
        with open(os.path.join(DATA, fn), newline="") as f:
            return [(r["text"], r["category"]) for r in csv.DictReader(f)]
    tr = {}
    for t, i in rows("banking77_train.csv"):
        tr.setdefault(i, []).append(t)
    return tr, rows("banking77_test.csv"), [t for t, _ in cl["oos_train"]], [t for t, _ in cl["oos_test"]]


def build(ds, K, n_cal, n_test, n_oos, seed=0):
    tr, test, oos_tr, oos_te = load(ds)
    mi = MeaningIndex()
    rint = {}
    for intent in sorted(tr):
        rid = mi.add_row(tr[intent][0], akey=intent)
        rint[rid] = intent
        for t in tr[intent][1:K]:
            mi.link(rid, t, learn=True)
    rng = random.Random(seed)
    cal = [(t, i) for i in sorted(tr) for t in tr[i][K:]]
    rng.shuffle(cal)
    cal = cal[:n_cal] + [(t, "oos") for t in oos_tr[: n_cal // 10]]
    rng.shuffle(cal)
    te = list(test); rng.shuffle(te); te = te[:n_test] + [(t, "oos") for t in oos_te[:n_oos]]

    def rec(q, want):
        ranked = mi.answers(q, k=8)
        g = MeaningIndex.confidence(ranked)
        cands = [rint.get(r) for r, _, _ in ranked]
        scores = [s for _, s, _ in ranked]
        return {"g": g, "want": want, "cands": cands, "scores": scores}
    t0 = time.time()
    C = [rec(q, w) for q, w in cal]
    T = [rec(q, w) for q, w in te]
    return C, T, time.time() - t0


def teacher(r, eps, rng):
    """-> picked rank (1-based) or 0 for 'new'. Mirrors bench_meaning.ScriptedModel."""
    right = [k for k, c in enumerate(r["cands"]) if c == r["want"]]
    if rng.random() < eps:
        wrong = [k for k in range(len(r["cands"])) if k not in right]
        return (rng.choice(wrong) + 1) if wrong else 0, True
    return (right[0] + 1) if right else 0, False


def evaluate(C, T, eps, seed=0, n_reask=100, p_serve=0.95):
    rng = random.Random(seed + 7)
    picks, noisy = [], []
    for r in C:
        pk, nz = teacher(r, eps, rng)
        picks.append(pk); noisy.append(nz)
    g = np.array([r["g"] for r in C])
    agree = np.array([1.0 if p == 1 else 0.0 for p in picks])
    cal = IsotonicCalibrator(list(g), list(agree))
    # ceiling exactly as MeaningIndex.ceiling(): agreement on the most confident fifth (>= 20)
    order = np.argsort(-g, kind="stable")[:max(20, len(g) // 5)]
    ceiling = float(agree[order].mean())
    # eps from SELF-CONSISTENCY: re-ask n_reask verdicts (deterministic sample) with an independent draw
    idx = list(range(len(C))); random.Random(seed + 11).shuffle(idx); idx = idx[:n_reask]
    rng2 = random.Random(seed + 13)
    same = 0
    for i in idx:
        a, _ = teacher(C[i], eps, rng2); b, _ = teacher(C[i], eps, rng2)
        same += (a == b)
    a2 = same / float(len(idx))
    m_bar = float(np.mean([max(1, len(r["cands"]) - 1) for r in C]))
    # P(two independent asks agree) = (1-e)^2 + e^2/m  ->  solve the quadratic for e in [0, 0.5]
    A_, B_, C_ = 1 + 1 / m_bar, -2.0, 1 - a2
    disc = max(0.0, B_ * B_ - 4 * A_ * C_)
    eps_hat = max(0.0, min(0.5, (-B_ - np.sqrt(disc)) / (2 * A_)))
    # the genuine pick-rank distribution vs the noise one (for the per-verdict NCE posterior)
    dis = [(p, nz) for p, nz in zip(picks, noisy) if p != 1]
    res = {"eps": eps, "n_cal": len(C), "ceiling": round(ceiling, 4), "reask_agree": round(a2, 3),
           "eps_hat_selfconsistency": round(eps_hat, 4), "m_bar": round(m_bar, 2),
           "disagreements": len(dis),
           "disagree_rank_hist_genuine": np.bincount([p for p, nz in dis if not nz], minlength=9).tolist(),
           "disagree_rank_hist_noise": np.bincount([p for p, nz in dis if nz], minlength=9).tolist()}
    # per-verdict quarantine rule: a disagreement that picks rank >= 3 (NCE: noise is uniform over ranks, genuine
    # corrections concentrate on the runner-up) -- recall of planted errors, false quarantine of genuine ones
    q_noise = sum(1 for p, nz in dis if nz and p >= 3); n_noise = sum(1 for p, nz in dis if nz)
    q_gen = sum(1 for p, nz in dis if (not nz) and p >= 3); n_gen = sum(1 for p, nz in dis if not nz)
    res["quarantine_rank>=3"] = {"planted_recall": round(q_noise / max(1, n_noise), 3),
                                 "genuine_quarantined": round(q_gen / max(1, n_gen), 3),
                                 "n_noise_disagree": n_noise, "n_genuine_disagree": n_gen}
    # gates on the test set
    tg = np.array([r["g"] for r in T])
    pa = np.array([cal.predict(x) for x in tg])
    top_ok = np.array([bool(r["cands"]) and r["cands"][0] == r["want"] for r in T])
    is_oos = np.array([r["want"] == "oos" for r in T])
    has = np.array([bool(r["cands"]) for r in T])

    def gate(serve):
        serve = serve & has
        ins = ~is_oos
        return {"in_correct": round(float((serve & ins & top_ok).sum() / ins.sum()), 4),
                "in_WRONG": round(float((serve & ins & ~top_ok).sum() / ins.sum()), 4),
                "oos_served": round(float((serve & is_oos).sum() / max(1, is_oos.sum())), 4)}
    res["gate_abs_0.95"] = gate(pa >= p_serve)
    for fl in (0.90, 0.75):
        bar = max(fl, p_serve * ceiling)
        res["gate_ceiling_floor%.2f(bar %.3f)" % (fl, bar)] = gate(pa >= bar)
    for e_name, e in (("eps_hat", eps_hat), ("eps_true", eps)):
        pc = (pa - e / m_bar) / max(1e-9, 1 - e - e / m_bar)
        res["gate_noisecorrected_%s" % e_name] = gate(pc >= p_serve)
    return res


if __name__ == "__main__":
    ds = sys.argv[1]
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    C, T, secs = build(ds, K, n_cal=1500, n_test=1500, n_oos=300)
    out = {"dataset": ds, "K_wordings_per_intent": K, "rank_seconds": round(secs, 1), "runs": []}
    for eps in (0.0, 0.1, 0.2):
        out["runs"].append(evaluate(C, T, eps))
    print(json.dumps(out, indent=1))
