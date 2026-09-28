"""exp_e_projection_cost.py -- REAL WORDING (CLINC150). What does it COST to put the free-text tower into
hypervector space? Same features (holographic_meaning.meaning_features: stems, stem bigrams, char 4-grams),
same IDF over intents, same centroid scorer (K=20 train wordings per intent) -- the ONLY difference is the
representation: the exact sparse vector vs its hypervector (sum of one random atom per feature, dim D).
This isolates the superposition (random-projection) noise of 'states and candidates in ONE HV space'.
No association expansion, no best-wording blend (both arms equally), so the absolute number is below the
MeaningIndex's own; the DIFFERENCE is the measurement."""
import sys, json, time, math
import numpy as np
sys.path.insert(0, "/home/claude/leCore")
from holographic.agents_and_reasoning.holographic_meaning import meaning_features

DATA = "/tmp/claude-0/-home-claude/9ee53278-d85f-54b0-979e-17c930fe54e2/scratchpad/data/clinc150_full.json"
K = 20
T0 = time.perf_counter()
d = json.load(open(DATA))
intents = sorted({i for _, i in d["train"]})
by = {}
for t, i in d["train"]:
    by.setdefault(i, []).append(t)
train = [(t, c) for c, i in enumerate(intents) for t in by[i][:K]]
test = [(t, intents.index(i)) for t, i in d["test"]]
vocab, df_sets = {}, {}
F_tr = [meaning_features(t) for t, _ in train]
F_te = [meaning_features(t) for t, _ in test]
for f, (_, c) in zip(F_tr, train):
    for k in f:
        vocab.setdefault(k, len(vocab)); df_sets.setdefault(k, set()).add(c)
nF = len(vocab)
idf = np.zeros(nF)
for k, j in vocab.items():
    idf[j] = max(0.0, math.log((len(intents) + 1.0) / (len(df_sets[k]) + 0.5)))


def sparse(f):
    cols = np.array([vocab[k] for k in f if k in vocab], np.int64)
    vals = np.array([f[k] for k in f if k in vocab], float) * (idf[cols] if len(cols) else 1.0)
    n = np.linalg.norm(vals)
    return cols, (vals / n if n > 0 else vals)


S_tr = [sparse(f) for f in F_tr]; S_te = [sparse(f) for f in F_te]
y_tr = np.array([c for _, c in train]); y_te = np.array([c for _, c in test])

# exact sparse arm: dense rows over the feature vocabulary (150 x nF is small)
C = np.zeros((len(intents), nF))
for (cols, vals), c in zip(S_tr, y_tr):
    C[c, cols] += vals
C /= np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)
pred = np.array([int(np.argmax(C[:, cols] @ vals)) if len(cols) else -1 for cols, vals in S_te])
res = {"features": nF, "K": K, "sparse_exact_top1": round(float((pred == y_te).mean()), 4), "hv": []}

for D in (1024, 2048, 8192):
    A = np.random.default_rng(12345).standard_normal((nF, D)).astype(np.float32) / np.float32(math.sqrt(D))

    def hv(s):
        cols, vals = s
        return (vals.astype(np.float32) @ A[cols]) if len(cols) else np.zeros(D, np.float32)
    Htr = np.stack([hv(s) for s in S_tr]); Hte = np.stack([hv(s) for s in S_te])
    Htr /= np.maximum(np.linalg.norm(Htr, axis=1, keepdims=True), 1e-12)
    P = np.stack([Htr[y_tr == c].mean(0) for c in range(len(intents))])
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    ph = np.argmax(Hte @ P.T, 1)
    res["hv"].append({"D": D, "top1": round(float((ph == y_te).mean()), 4),
                      "agree_with_sparse": round(float((ph == pred).mean()), 4)})
    del A
res["cpu_s"] = round(time.perf_counter() - T0, 1)
print(json.dumps(res, indent=1))
json.dump(res, open(__file__.replace(".py", ".json"), "w"), indent=1)
