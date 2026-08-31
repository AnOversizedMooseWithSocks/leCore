"""Stress the f32 shader path on REAL data, then walk the six levers in cost order.

WHY REAL DATA: my earlier margin measurement used freshly minted unitary atoms, which are
near-orthogonal BY CONSTRUCTION -- the friendliest possible corpus. Real documents share
vocabulary, so their vectors are CORRELATED and the cleanup margin is the thing that should
collapse. A safety factor of 10^7 on synthetic atoms is a hypothesis about the instrument,
not a result about the browser.

Corpus = this engine's own module docstrings. Encoding = bag-of-atoms (derive_atom per token,
bundled, normalised) -- a real VSA text encoding, not a toy.
"""
import glob, re, sys
import numpy as np
import lecore
from holographic.agents_and_reasoning.holographic_ai import derived_atom as derive_atom, unitary_vector

STOP = set("the a an of to and or is are was in on for with that this it as by be from at "
           "not but if then than so its it's which what when how you your we our".split())

def tokens(text):
    return [w for w in re.findall(r"[a-z][a-z0-9_]{2,}", text.lower()) if w not in STOP]

def load_corpus(limit=400):
    """Real docstrings from the live tree -- name + first paragraph."""
    docs = []
    for path in sorted(glob.glob("holographic/*/holographic_*.py"))[:limit]:
        try:
            src = open(path, encoding="utf-8", errors="ignore").read(4000)
        except OSError:
            continue
        m = re.search(r'"""(.{60,1200}?)"""', src, re.S)
        if not m:
            continue
        toks = tokens(m.group(1))
        if len(toks) >= 25:
            docs.append((path.split("/")[-1][:-3], toks))
    return docs

def encode(toks, dim, seed=0):
    """Bag-of-atoms: bundle a derived atom per token, then normalise."""
    v = np.zeros(dim)
    for t in set(toks):                       # set(): term presence, the BM25-ish shape
        v += derive_atom(seed, t, dim)   # derived_atom(seed, name, dim)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def f32_scores(V, q):
    """What a fragment shader computes: f32 matvec."""
    return (V.astype(np.float32) @ q.astype(np.float32)).astype(np.float64)

def trial(docs, dim, K, frac=0.4, seed=0, rng=None):
    """Encode K docs, query each with a random FRACTION of its own tokens, measure margins."""
    rng = rng or np.random.default_rng(0)
    sub = docs[:K]
    V = np.stack([encode(t, dim, seed) for _, t in sub])
    hits64 = hits32 = gate_answered = gate_correct = 0
    margins, ratios = [], []
    for i, (_, toks) in enumerate(sub):
        u = sorted(set(toks))
        pick = rng.choice(len(u), max(3, int(len(u) * frac)), replace=False)
        q = encode([u[j] for j in pick], dim, seed)
        s64 = V @ q
        s32 = f32_scores(V, q)
        a64, a32 = int(np.argmax(s64)), int(np.argmax(s32))
        hits64 += (a64 == i); hits32 += (a32 == i)
        srt = np.sort(s64)[::-1]
        margin = float(srt[0] - srt[1])
        eps = float(np.max(np.abs(s32 - s64)))
        margins.append(margin); ratios.append(margin / max(2 * eps, 1e-30))
        if margin > 2 * eps:                  # T1's gate
            gate_answered += 1
            gate_correct += (a32 == a64)
    n = len(sub)
    return dict(K=n, dim=dim, acc64=hits64 / n, acc32=hits32 / n,
                margin_med=float(np.median(margins)), margin_min=float(np.min(margins)),
                ratio_med=float(np.median(ratios)), ratio_min=float(np.min(ratios)),
                gate_rate=gate_answered / n,
                gate_precision=(gate_correct / gate_answered) if gate_answered else float("nan"))

if __name__ == "__main__":
    docs = load_corpus()
    print("REAL CORPUS: %d module docstrings, median %d distinct tokens"
          % (len(docs), int(np.median([len(set(t)) for _, t in docs]))))
    print("\n--- BASELINE: f32 shader path on real, CORRELATED text (dim=512)")
    print("  K     acc_f64  acc_f32  margin_med  margin_min  ratio_med  ratio_min  gate%  gate_prec")
    for K in (32, 128, 256, len(docs)):
        r = trial(docs, 512, K)
        print("  %-5d %-8.3f %-8.3f %-11.5f %-11.5f %-10.1f %-10.1f %-6.2f %.3f" %
              (r["K"], r["acc64"], r["acc32"], r["margin_med"], r["margin_min"],
               r["ratio_med"], r["ratio_min"], r["gate_rate"], r["gate_precision"]))

    print("\n--- LEVER 4 (more dimensions): does raising D recover the margin?")
    print("  dim   acc_f64  acc_f32  margin_med  ratio_min  gate%")
    for dim in (128, 256, 512, 1024, 2048):
        r = trial(docs, dim, min(256, len(docs)))
        print("  %-5d %-8.3f %-8.3f %-11.5f %-10.1f %.2f" %
              (dim, r["acc64"], r["acc32"], r["margin_med"], r["ratio_min"], r["gate_rate"]))
