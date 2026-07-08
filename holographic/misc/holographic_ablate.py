"""The ablation table (G2): where is VSA actually load-bearing?

The README repeatedly, honestly finds the simple baseline ties the holographic one. Taken
together that poses a question never answered system-wide: which subsystems genuinely need
superposition / binding / corruption-robustness, and which are a VSA showcase where VSA
isn't the reason they work? This module answers it the only honest way -- for each
subsystem, run the DUMBEST honest non-holographic baseline on the SAME task, data, and
metric, measure both across seeds with the variance harness, and let the confidence
intervals decide the verdict:

    holo lower CI  > baseline upper CI -> VSA load-bearing  (a real win, invest here)
    baseline lower > holo upper CI     -> baseline wins     (VSA decorative here)
    intervals overlap                  -> uniformity        (within the noise)

USE REAL DATA -- the verdicts below run on Reuters, UDHR, and Brown; the algebra-level
rows (key->value, recall index) are hermetic but at real scale. Run `python
holographic_ablate.py` to print the live table, or import `ablation_table()`.
"""
import math
import re
from collections import defaultdict, Counter

import numpy as np

from holographic.misc.holographic_measure import measure
from holographic.agents_and_reasoning.holographic_honesty import bh_fdr


def verdict(holo, base):
    """Honest verdict from two measure() stat dicts, judged by their 95% CIs."""
    hl, hh = holo["ci"]
    bl, bh = base["ci"]
    if hl > bh:
        return "VSA load-bearing"
    if bl > hh:
        return "baseline wins"
    return "uniformity"


def _paired_perm_p(holo_scores, base_scores):
    """One-sided p-value for 'holo > base' by permutation -- the engine's own threshold-free
    instrument. When the two arms share seeds (equal length) the seeds pair, so it is a
    SIGN-FLIP test on the paired differences (how often a random re-signing reaches the
    observed mean difference). When the seed counts differ (an arm ran fewer seeds) the
    scores cannot pair, so it falls back to a TWO-SAMPLE label-permutation test (how often
    relabelling the pooled scores reaches the observed gap of means). Small families are
    enumerated exactly; larger ones are sampled. No-difference returns 1.0."""
    h = np.asarray(holo_scores, float)
    b = np.asarray(base_scores, float)
    if len(h) == len(b):                                  # paired: sign-flip the differences
        d = h - b
        n = len(d)
        if n == 0 or not np.any(d):
            return 1.0
        obs = d.mean()
        if n <= 20:
            signs = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1) * 2 - 1
            return float(((signs * np.abs(d)).mean(axis=1) >= obs - 1e-12).mean())
        rng = np.random.default_rng(0)
        flips = rng.choice([-1.0, 1.0], size=(20000, n))
        return float(((flips * np.abs(d)).mean(axis=1) >= obs - 1e-12).mean())
    # unpaired: permute which pooled scores are labelled 'holo'
    obs = h.mean() - b.mean()
    if obs == 0:
        return 1.0
    pool = np.concatenate([h, b])
    nh = len(h)
    rng = np.random.default_rng(0)
    perms = np.array([rng.permutation(pool) for _ in range(20000)])
    diffs = perms[:, :nh].mean(axis=1) - perms[:, nh:].mean(axis=1)
    return float((diffs >= obs - 1e-12).mean())


def fdr_verdicts(rows, alpha=0.1):
    """Apply false-discovery control across the WHOLE ablation family at once.

    The per-subsystem verdict() decides on that subsystem's own 95% CIs -- but the table is a
    SCAN over many subsystems, and scanning enough of them means one can clear a per-test bar
    by luck. This is the exposure bh_fdr exists for. Each subsystem gets a paired permutation
    p-value (holo > baseline); bh_fdr (Benjamini-Yekutieli, dependent=True -- the subsystems
    share data and methodology) then holds the false-discovery rate among the surviving calls
    at alpha across the family. Returns (augmented_rows, n_load_bearing, n_survive) where each
    augmented row is (name, holo, base, base_name, verdict, p, survives_fdr)."""
    testable = [(i, r) for i, r in enumerate(rows)
                if r[1] is not None and "scores" in r[1] and "scores" in r[2]]
    pvals = [_paired_perm_p(r[1]["scores"], r[2]["scores"]) for _, r in testable]
    reject, _ = bh_fdr(np.array(pvals), alpha=alpha, dependent=True) if pvals \
        else (np.zeros(0, bool), 0)
    survive = {}
    for (idx, _), p, rj in zip(testable, pvals, reject):
        survive[idx] = (float(p), bool(rj))
    out, n_lb, n_surv = [], 0, 0
    for i, r in enumerate(rows):
        p, rj = survive.get(i, (float("nan"), False))
        out.append((*r, p, rj))
        if r[4] == "VSA load-bearing":
            n_lb += 1
            if rj:
                n_surv += 1
    return out, n_lb, n_surv


# ---------------------------------------------------------------------------
# Each subsystem: a holographic score fn and the dumbest honest baseline, on real data.
# ---------------------------------------------------------------------------

def _reuters_docs():
    from nltk.corpus import reuters
    cats = ["earn", "acq", "crude", "trade", "money-fx"]
    docs = {c: [] for c in cats}
    for f in reuters.fileids():
        cs = reuters.categories(f)
        if len(cs) == 1 and cs[0] in cats and len(docs[cs[0]]) < 60:
            toks = [w.lower() for w in reuters.words(f) if w.isalpha()][:120]
            if len(toks) > 20:
                docs[cs[0]].append(toks)
    return cats, [(t, c) for c in cats for t in docs[c]]


def _reuters_split(alldocs, seed):
    rng = np.random.default_rng(seed)
    items = list(alldocs); rng.shuffle(items)
    by = {}
    for t, c in items:
        by.setdefault(c, []).append(t)
    tr, te = [], []
    for c, ts in by.items():
        k = int(len(ts) * 0.7)
        tr += [(t, c) for t in ts[:k]]
        te += [(t, c) for t in ts[k:]]
    return tr, te


def topic_classify(seeds=range(6)):
    """Holographic UnifiedMind classify vs bag-of-words nearest centroid (Reuters)."""
    cats, alldocs = _reuters_docs()
    vocab = sorted({w for t, _ in alldocs for w in t})
    wi = {w: i for i, w in enumerate(vocab)}

    def holo(seed):
        from holographic.misc.holographic_unified import UnifiedMind
        tr, te = _reuters_split(alldocs, seed)
        m = UnifiedMind(dim=1024, seed=seed)
        m.absorb([(" ".join(t), c) for t, c in tr])
        return np.mean([m.classify(" ".join(t))[0] == c for t, c in te])

    def bow(seed):
        tr, te = _reuters_split(alldocs, seed)

        def vec(t):
            v = np.zeros(len(vocab))
            for w in t:
                if w in wi:
                    v[wi[w]] += 1
            n = np.linalg.norm(v)
            return v / n if n else v
        cents = {c: np.mean([vec(t) for t, cc in tr if cc == c], axis=0) for c in cats}
        return np.mean([max(cents, key=lambda C: np.dot(vec(t), cents[C])) == c for t, c in te])

    return measure(holo, seeds), measure(bow, seeds), "bag-of-words centroid"


def language_id(seeds=range(6)):
    """Holographic trigram-binding profiles vs bag-of-trigrams centroid (UDHR)."""
    from nltk.corpus import udhr
    files = {"en": "English-Latin1", "fr": "French_Francais-Latin1",
             "de": "German_Deutsch-Latin1", "es": "Spanish_Espanol-Latin1",
             "it": "Italian_Italiano-Latin1", "nl": "Dutch_Nederlands-Latin1"}
    texts = {k: re.sub(r"[^a-z ]+", " ", udhr.raw(f).lower()) for k, f in files.items()}

    def split(seed):
        rng = np.random.default_rng(seed); tr, te = {}, []
        for k, full in texts.items():
            ch = [full[i:i + 200] for i in range(0, len(full) - 200, 200)]
            rng.shuffle(ch); c = int(len(ch) * 0.6)
            tr[k] = ch[:c]; te += [(x, k) for x in ch[c:]]
        return tr, te

    def holo(seed):
        from holographic.misc.holographic_text import LanguageID
        tr, te = split(seed)
        lid = LanguageID(dim=512, seed=seed).fit(tr)
        return np.mean([lid.identify(x) == k for x, k in te])

    def bag(seed):
        tr, te = split(seed)
        tris = sorted({s[i:i + 3] for k in tr for s in tr[k] for i in range(len(s) - 2)})
        ti = {t: i for i, t in enumerate(tris)}

        def vec(s):
            v = np.zeros(len(tris))
            for i in range(len(s) - 2):
                if s[i:i + 3] in ti:
                    v[ti[s[i:i + 3]]] += 1
            n = np.linalg.norm(v)
            return v / n if n else v
        cents = {k: np.mean([vec(s) for s in tr[k]], axis=0) for k in tr}
        return np.mean([max(cents, key=lambda K: np.dot(vec(x), cents[K])) == k for x, k in te])

    return measure(holo, seeds), measure(bag, seeds), "bag-of-trigrams centroid"


def segmentation(seeds=range(6)):
    """Holographic branching entropy vs EXACT count-based branching entropy (Brown)."""
    from nltk.corpus import brown
    from holographic.misc.holographic_segment import Segmenter, boundary_f1
    words = [w.lower() for w in brown.words(categories="news") if w.isalpha()][:1500]
    spaceless = "".join(words); truth, pos = set(), -1
    for w in words:
        pos += len(w); truth.add(pos)

    def holo(seed):
        seg = Segmenter(dim=512, order=3, seed=seed).fit(spaceless)
        return boundary_f1(seg.boundaries(spaceless, 70), truth)["f1"]

    def exact(seed):
        K = 3; nxt = defaultdict(Counter)
        for i in range(1, len(spaceless)):
            nxt[spaceless[max(0, i - K):i]][spaceless[i]] += 1
        H = []
        for i in range(1, len(spaceless)):
            c = nxt[spaceless[max(0, i - K):i]]; tot = sum(c.values())
            H.append(-sum((v / tot) * math.log2(v / tot) for v in c.values()) if tot else 0.0)
        H = np.array(H); thr = np.percentile(H, 70)
        pred = set(i for i in range(1, len(H) - 1)
                   if H[i] >= thr and H[i] >= H[i - 1] and H[i] >= H[i + 1])
        return boundary_f1(pred, truth)["f1"]

    return measure(holo, seeds), measure(exact, range(3)), "exact count entropy"


def key_value_noisy(seeds=range(6), noise=0.5):
    """VSA bind/bundle store with cosine cleanup vs an exact Python dict, on NOISY keys
    (the realistic case -- keys are encoded percepts, never bit-exact). The dict can't
    match a perturbed key at all; cosine cleanup still recovers the value."""
    from holographic.agents_and_reasoning.holographic_ai import bind, unbind, cosine

    def setup(seed):
        rng = np.random.default_rng(seed); dim, n = 512, 30

        def rv():
            v = rng.standard_normal(dim); return v / np.linalg.norm(v)
        keys = [rv() for _ in range(n)]
        vals = [rv() for _ in range(n)]
        return rng, dim, n, keys, vals

    def holo(seed):
        rng, dim, n, keys, vals = setup(seed)
        trace = np.sum([bind(k, v) for k, v in zip(keys, vals)], axis=0)
        ok = 0
        for i in range(n):
            q = keys[i] + noise * (lambda v: v / np.linalg.norm(v))(rng.standard_normal(dim))
            q /= np.linalg.norm(q)
            est = unbind(trace, q)
            ok += (int(np.argmax([cosine(est, v) for v in vals])) == i)
        return ok / n

    def dct(seed):
        rng, dim, n, keys, vals = setup(seed)
        D = {k.tobytes(): i for i, k in enumerate(keys)}
        ok = 0
        for i in range(n):
            q = keys[i] + noise * (lambda v: v / np.linalg.norm(v))(rng.standard_normal(dim))
            q /= np.linalg.norm(q)
            ok += (D.get(q.tobytes(), -1) == i)        # exact hash never matches a noisy key
        return ok / n

    return measure(holo, seeds), measure(dct, seeds), "exact dict (noisy keys)"


def recall_index(seeds=range(5)):
    """HoloForest approximate recall vs exact brute-force scan. The forest LOSES on raw
    recall (exact scan is trivially 1.0) but reaches its recall at a FRACTION of the
    comparisons -- so this row's honest reading is 'scale, not accuracy', reported with
    the comparison fraction alongside."""
    from holographic.misc.holographic_tree import HoloForest

    def make(seed, N=2000, dim=128):
        rng = np.random.default_rng(seed)
        items = rng.standard_normal((N, dim)); items /= np.linalg.norm(items, axis=1, keepdims=True)
        queries = []
        for _ in range(120):
            i = int(rng.integers(N))
            q = items[i] + 0.15 * rng.standard_normal(dim); q /= np.linalg.norm(q)
            queries.append(q)
        return items, queries

    fracs = []

    def holo(seed):
        items, queries = make(seed)
        F = HoloForest(items.shape[1], n_trees=4, leaf_size=64, seed=seed).build(items)
        ok, comps = 0, []
        for q in queries:
            true = int((items @ q).argmax())
            ok += (F.recall(q, beam=4) == true)
            comps.append(F.last_comparisons)
        fracs.append(np.mean(comps) / len(items))
        return ok / len(queries)

    def scan(seed):
        return 1.0                                       # exact scan: trivially perfect, 100% cost

    h = measure(holo, seeds)
    b = measure(scan, range(3))
    h["comparison_fraction"] = float(np.mean(fracs))
    return h, b, "exact scan (100% comparisons)"


SUBSYSTEMS = [
    ("topic classify (Reuters)", topic_classify),
    ("language ID (UDHR)", language_id),
    ("segmentation (Brown)", segmentation),
    ("key->value, noisy keys", key_value_noisy),
    ("recall index (forest)", recall_index),
]


def ablation_table(seeds=range(6)):
    """Run every subsystem's ablation and return a list of rows:
    (name, holo_stats, base_stats, baseline_name, verdict)."""
    rows = []
    for name, fn in SUBSYSTEMS:
        try:
            h, b, base_name = fn()
            rows.append((name, h, b, base_name, verdict(h, b)))
        except Exception as e:
            rows.append((name, None, None, str(e), "skipped"))
    return rows


def _demo():
    rows = ablation_table()
    aug, n_lb, n_surv = fdr_verdicts(rows, alpha=0.1)
    print("ABLATION TABLE -- is VSA load-bearing? (holographic vs dumbest honest baseline,")
    print("real data, judged by 95% CIs from the variance harness)\n")
    print(f"  {'subsystem':28} {'holo':>14} {'baseline':>14}  {'verdict':18} {'p':>7} {'FDR':>9}")
    for name, h, b, base_name, v, p, rj in aug:
        if h is None:
            print(f"  {name:28} (skipped: {base_name})")
            continue
        extra = ""
        if "comparison_fraction" in h:
            extra = f"  [forest @ {h['comparison_fraction']*100:.0f}% comparisons -> the win is SCALE]"
        tag = "survives" if rj else ("LUCK?" if v == "VSA load-bearing" else "--")
        print(f"  {name:28} {h['mean']:.3f}+/-{h['std']:.3f} {b['mean']:.3f}+/-{b['std']:.3f}  "
              f"{v:18} {p:7.4f} {tag:>9}{extra}")
    print(f"\n{n_surv}/{n_lb} 'load-bearing' verdicts survive family-wise false-discovery control "
          f"(BH-Yekutieli, alpha=0.1)")
    print("Reading: 'VSA load-bearing' = superposition/binding/cleanup is the reason it works;")
    print("'uniformity' = the simple baseline ties it (the idea works, not the VSA encoding);")
    print("'baseline wins' = VSA is decorative here (but the forest row buys sublinear SCALE).")
    print("FDR: the per-test CI decides one subsystem; the p-column + FDR judge the WHOLE scan,")
    print("so a verdict that cleared its own CI by luck across many subsystems is caught.")


if __name__ == "__main__":
    _demo()
