"""probe_proto.py -- w1-contrastive scratch probe (NOT repo code).

Question: on REAL human wording (Banking77, CLINC150), does a softmax-weighted (InfoNCE-gradient) prototype
update beat SystemOne's miss-only AdaptHD update, and the positive-only "add every confirmed wording" update
(what MeaningIndex.link effectively does to a row centroid)?  Clean labels and 10% wrong labels.

Encoder: SystemOne's hashed_ngram_encode (char 3..5-grams, sha256-seeded gaussian atoms), reproduced exactly
but computed n-gram-major so the atom cache fits in memory. dim 2048.

Protocol (prequential, one pass, no epochs -- learning from use):
  init: prototypes = unit mean of the first K=5 training wordings per intent (SystemOne.fit's accumulator)
  stream: N other training wordings, shuffled (seed 0); decide first, then update from the (possibly noisy) label
  noise: with prob r the label is a uniformly random WRONG intent among the top-8 candidates (the bench's stand-in)
  test: top-1 on the held-out test split with the final prototypes; AUROC of three confidence signals.
Every arm sees the identical stream and noise draws.
"""
import csv, hashlib, json, os, random, sys, time
import numpy as np

DATA = "/tmp/claude-0/-home-claude/9ee53278-d85f-54b0-979e-17c930fe54e2/scratchpad/data"
DIM = 2048


def load(ds):
    if ds == "clinc":
        d = json.load(open(os.path.join(DATA, "clinc150_full.json")))
        tr = {}
        for t, i in d["train"]:
            tr.setdefault(i, []).append(t)
        return tr, [tuple(x) for x in d["test"]]
    def rows(fn):
        with open(os.path.join(DATA, fn), newline="") as f:
            return [(r["text"], r["category"]) for r in csv.DictReader(f)]
    tr = {}
    for t, i in rows("banking77_train.csv"):
        tr.setdefault(i, []).append(t)
    return tr, rows("banking77_test.csv")


def encode_all(texts, dim=DIM, lo=3, hi=5):
    """Bit-for-bit hashed_ngram_encode, n-gram-major: each atom is generated once and scattered."""
    post = {}
    for j, text in enumerate(texts):
        t = " " + text.lower() + " "
        for n in range(lo, hi + 1):
            for i in range(len(t) - n + 1):
                g = t[i:i + n]
                d = post.setdefault(g, {})
                d[j] = d.get(j, 0) + 1
    H = np.zeros((len(texts), dim), np.float64)
    for g in sorted(post):
        seed = int.from_bytes(hashlib.sha256(g.encode()).digest()[:8], "big") % (2 ** 32)
        v = np.random.default_rng(seed).standard_normal(dim)
        idx = np.fromiter(post[g].keys(), np.int64)
        c = np.fromiter(post[g].values(), np.float64)
        H[idx] += c[:, None] * v[None, :]
    H /= np.linalg.norm(H, axis=1, keepdims=True) + 1e-12
    return H


def auroc(score, ok):
    score, ok = np.asarray(score, float), np.asarray(ok, bool)
    pos, neg = score[ok], score[~ok]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    r = np.argsort(np.argsort(allv, kind="stable"), kind="stable") + 1.0
    # average ranks for ties
    order = np.argsort(allv, kind="stable")
    sv = allv[order]
    ranks = np.empty_like(r)
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


_ENC = {}


def run(ds, n_stream, noise, arms, seed=int(os.environ.get("PSEED", "0")), K=int(os.environ.get("PK", "5"))):
    tr, test = load(ds)
    labels = sorted(tr)
    L = {l: i for i, l in enumerate(labels)}
    init = [(t, L[l]) for l in labels for t in tr[l][:K]]
    stream = [(t, L[l]) for l in labels for t in tr[l][K:]]
    random.Random(seed).shuffle(stream)
    stream = stream[:n_stream]
    texts = [t for t, _ in init] + [t for t, _ in stream] + [t for t, _ in test]
    t0 = time.time()
    key = (ds, n_stream, seed, K)
    if key not in _ENC:
        _ENC[key] = encode_all(texts)
    H = _ENC[key]
    enc_s = time.time() - t0
    Hi, Hs, Ht = H[:len(init)], H[len(init):len(init) + len(stream)], H[len(init) + len(stream):]
    yi = np.array([y for _, y in init]); ys = np.array([y for _, y in stream])
    yt = np.array([L[l] for _, l in test])
    C = len(labels)
    A0 = np.zeros((C, DIM))
    for c in range(C):
        mu = Hi[yi == c].mean(0)
        A0[c] = mu / (np.linalg.norm(mu) + 1e-12)
    # identical noise draws for every arm: decided against the FROZEN top-8 so the draw does not depend on the arm
    rng = np.random.default_rng(seed + 99)
    flip = rng.random(len(stream)) < noise
    pick = rng.random(len(stream))
    out = {"dataset": ds, "n_stream": len(stream), "noise": noise, "encode_s": round(enc_s, 1), "arms": {}}
    for name, cfg in arms.items():
        A = A0.copy()
        P = A / np.linalg.norm(A, axis=1, keepdims=True)
        preq, flips_bad = 0, 0
        nrow = np.bincount(yi, minlength=C).astype(float)   # confirmed wordings per row (maturity)
        for n in range(len(stream)):
            q = Hs[n]
            s = P @ q
            pred = int(np.argmax(s))
            y = int(ys[n])
            preq += (pred == y)
            lab = y
            if flip[n]:
                top8 = np.argsort(-s, kind="stable")[:8]
                wrong = [int(k) for k in top8 if k != y]
                lab = wrong[int(pick[n] * len(wrong))]
            kind = cfg["rule"]
            lr = cfg.get("lr", 1.0)
            if kind == "frozen":
                continue
            if kind == "adapthd":                      # SystemOne.observe: miss-only pull/push
                if pred != lab:
                    A[lab] += lr * q; A[pred] -= lr * q
                    for i in (lab, pred):
                        P[i] = A[i] / (np.linalg.norm(A[i]) + 1e-12)
                continue
            if kind == "positive":                     # MeaningIndex.link-like: every confirmed wording joins
                A[lab] += lr * q
                P[lab] = A[lab] / (np.linalg.norm(A[lab]) + 1e-12)
                continue
            if kind in ("staged", "annealed"):
                k0 = cfg.get("k0", 3)
                if kind == "staged" and nrow[lab] < k0:
                    # EASY STAGE: a young row only pulls (the positive-only rule), and is never pushed
                    A[lab] += lr * q
                    P[lab] = A[lab] / (np.linalg.norm(A[lab]) + 1e-12)
                    nrow[lab] += 1
                    continue
                kind = "infonce"
            nrow[lab] += 1
            if kind == "infonce":                      # A_k += lr * (1[k=lab] - p_k) q  over the candidate set
                tau = cfg["tau"]
                if cfg["rule"] == "annealed":           # per-row temperature: diffuse (easy) young, sharp (hard) mature
                    tau = cfg["tau"] * (1.0 + cfg.get("k0", 3) / max(nrow[lab], 1.0))
                if cfg["rule"] == "staged":             # hard stage: never push a row that is itself still young
                    pass
                cand = np.arange(C) if cfg.get("topk") is None else np.argsort(-s, kind="stable")[:cfg["topk"]]
                if cfg.get("topk") is not None and lab not in cand:
                    cand = np.append(cand, lab)
                z = s[cand] / tau
                z -= z.max()
                p = np.exp(z); p /= p.sum()
                li = np.flatnonzero(cand == lab)[0]
                eps = cfg.get("eps", 0.0)
                if eps > 0:
                    # NOISE-CORRECTED TARGET (forward correction / E-step with a known flip model): the observed
                    # label came from the true class k with T(k->lab) = 1-eps if k==lab else eps/m (m = the wrong
                    # candidates the teacher could have picked). posterior t_k ~ p_k * T(k->lab). eps=0 -> one-hot.
                    m = max(1, min(len(cand), 8) - 1)
                    t = p * (eps / m)
                    t[li] = p[li] * (1.0 - eps)
                    t /= t.sum()
                else:
                    t = np.zeros_like(p); t[li] = 1.0
                g = t - p
                sel = np.abs(g) > 1e-4
                if cfg["rule"] == "staged":
                    sel &= (nrow[cand] >= cfg.get("k0", 3)) | (cand == lab)
                touch = cand[sel]
                gg = g[sel]
                A[touch] += lr * gg[:, None] * q[None, :]
                P[touch] = A[touch] / (np.linalg.norm(A[touch], axis=1, keepdims=True) + 1e-12)
                continue
            raise ValueError(kind)
        St = Ht @ P.T
        pr = np.argmax(St, axis=1)
        ok = pr == yt
        srt = -np.sort(-St, axis=1)
        gap = srt[:, 0] - srt[:, 1]
        res = {"prequential_acc": round(preq / len(stream), 4), "test_top1": round(float(ok.mean()), 4),
               "auroc_gap": round(auroc(gap, ok), 4)}
        for tau in (0.02, 0.05):
            z = (St - srt[:, :1]) / tau
            p = np.exp(z); p /= p.sum(1, keepdims=True)
            ptop = p.max(1)
            clm = ptop - (1 - ptop) / (C - 1)          # CLM's "top prob minus mean of the others"
            res["auroc_ptop_tau%.2f" % tau] = round(auroc(ptop, ok), 4)
            res["auroc_clm_tau%.2f" % tau] = round(auroc(clm, ok), 4)
        # reverse direction (row->state) as a per-row log-partition over a bank of the last 256 stream questions
        # (querybank normalization / inverted softmax): s'_k = s_k - tau*log mean_j exp(s_jk/tau)
        bank = Hs[-256:] @ P.T
        for tau in (0.05,):
            bk = tau * np.log(np.mean(np.exp((bank - bank.max(0, keepdims=True)) / tau), axis=0)) + bank.max(0)
            prb = np.argmax(St - bk[None, :], axis=1)
            res["test_top1_qbnorm_tau%.2f" % tau] = round(float((prb == yt).mean()), 4)
            # the same, with the FALSE-NEGATIVE RULE: bank questions labelled k are not negatives for row k
            yb = ys[-256:]
            E = np.exp((bank - bank.max()) / tau)
            E[np.arange(len(yb)), yb] = 0.0
            cnt = len(yb) - np.bincount(yb, minlength=C)
            bk2 = tau * np.log(E.sum(0) / np.maximum(cnt, 1) + 1e-300) + bank.max()
            prb2 = np.argmax(St - bk2[None, :], axis=1)
            res["test_top1_qbnorm_excl_tau%.2f" % tau] = round(float((prb2 == yt).mean()), 4)
            for lam in (0.25, 0.5):
                prb3 = np.argmax(St - lam * bk2[None, :], axis=1)
                res["test_top1_qbnorm_excl_lam%.2f" % lam] = round(float((prb3 == yt).mean()), 4)
        out["arms"][name] = res
    return out


if __name__ == "__main__":
    ds = sys.argv[1]; n = int(sys.argv[2])
    arms = {
        "frozen": {"rule": "frozen"},
        "adapthd_miss_only": {"rule": "adapthd"},
        "positive_only": {"rule": "positive"},
        "infonce_t0.02": {"rule": "infonce", "tau": 0.02},
        "infonce_t0.05": {"rule": "infonce", "tau": 0.05},
        "infonce_t0.10": {"rule": "infonce", "tau": 0.10},
        "infonce_t0.05_top8": {"rule": "infonce", "tau": 0.05, "topk": 8},
        "adapthd_lr0.3": {"rule": "adapthd", "lr": 0.3},
        "infonce_t0.05_lr0.3": {"rule": "infonce", "tau": 0.05, "lr": 0.3},
        "infonce_nc0.1_t0.05_lr0.3": {"rule": "infonce", "tau": 0.05, "lr": 0.3, "eps": 0.1},
        "staged_k3_t0.05_lr0.3": {"rule": "staged", "tau": 0.05, "lr": 0.3, "k0": 3},
        "annealed_k3_t0.05_lr0.3": {"rule": "annealed", "tau": 0.05, "lr": 0.3, "k0": 3},
        "infonce_nc0.1_t0.05_lr0.3_top8": {"rule": "infonce", "tau": 0.05, "lr": 0.3, "eps": 0.1, "topk": 8},
    }
    if len(sys.argv) > 4:
        arms = {k: v for k, v in arms.items() if k in sys.argv[4].split(",")}
    t0 = time.time()
    rs = []
    for nz in str(sys.argv[3]).split(","):
        r = run(ds, n, float(nz), arms)
        r["total_s"] = round(time.time() - t0, 1)
        rs.append(r)
    print(json.dumps(rs, indent=1))
