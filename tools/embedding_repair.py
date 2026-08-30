"""embedding_repair.py -- FIX AN EMBEDDING TABLE IN PLACE (cp116).

This is the first operation in this codebase that changes a model's weights to make it
BETTER rather than smaller. Everything before it either measured the model or removed
something from it; six separate compression attempts returned negatives.

THE OPERATION. A vocabulary of a quarter of a million tokens has a long tail whose
embeddings were estimated from very few occurrences. Those rows are noise-dominated. But a
token's SPELLING predicts its meaning well enough to identify it (measured: 255x chance for
held-out words, 131x for words below the frequency floor), so a badly-estimated row can be
rebuilt from the rows of tokens that share its form.

Crucially the rebuilt value is NOT the form prediction. Measured in cp114:

    raw form vector used as the embedding      variance explained  -0.112
    Hewitt mean-of-all baseline                                     0.000
    form-SELECTED donors, convex combination, k=64                 +0.063

Form selects; real in-vocabulary vectors supply the value, because they already lie on the
manifold. Random donors score -0.015 and shuffled-letter selection -0.084, so the selection
is doing the work.

HONEST EXPECTATIONS. The literature is consistent that initialisation advantages are large
at step 0 and shrink to roughly 0.5-1.5 downstream points after adaptation (FOCUS, ZeTT,
SambaLingo). This repairs rows; it does not make a model smarter.

WHEN TO REPAIR A ROW -- the decision rule, and what it is NOT. A rebuilt row lands at a
FIXED quality regardless of how bad the row it replaces was, so rebuilding wins only below
that quality. Repair rows you believe are worse than the rebuild; keep the rest.

    row's cosine to truth   0.97  0.90  0.80  0.71  0.56 | 0.45  0.31
    rebuild instead?         no    no    no    no    no  |  YES   YES

CORRECTION (cp123). This was originally recorded as a CONSTANT, cosine 0.507, and treated
as if it were a property of the method. Kanerva's challenge -- if it is a critical distance
it must move with dimension -- showed it is neither a constant nor a law:

    dim    rebuilt quality    measured crossover
     25         0.720              0.713
     50         0.674              0.671
    100         0.603              0.606
    200         0.507              0.508

The crossover TRACKS THE REBUILD QUALITY EXACTLY at every dimension. It is the tautology
"rebuild when your row is worse than the rebuild would be" -- which is a correct and useful
rule, and not a discovered threshold. 0.507 was simply the rebuild quality at D=200.

So DO NOT carry 0.507 to another model. Call `rebuild_quality()` on the artifact in hand
and use its result as the threshold; a 1024-dimensional table will have a different one.
"""


def rebuild_quality(words, vectors, k=64, holdout=400, seed=0):
    """The crossover, MEASURED on the artifact in hand rather than assumed.

    Returns the mean cosine a rebuilt row achieves against the true row on held-out
    entries. Rows believed worse than this are worth repairing; better ones are not.
    """
    import numpy as _np
    from tools.form_prior import FormPrior as _FP
    rng = _np.random.default_rng(seed)
    n = len(words)
    holdout = max(10, min(int(holdout), n // 4))
    o = rng.permutation(n)
    te, tr = o[:holdout], o[holdout:]
    V = _np.asarray(vectors, _np.float64)
    fp = _FP().fit([words[i] for i in tr], V[tr])
    P = fp.predict_via_donors([words[i] for i in te], [words[i] for i in tr], V[tr], k=k)
    T = V[te]
    P = P / (_np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
    T = T / (_np.linalg.norm(T, axis=1, keepdims=True) + 1e-12)
    return float(_np.mean(_np.sum(P * T, 1)))

import os
import re
import sys
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.form_prior import FormPrior  # noqa: E402


def load_table(model_dir):
    """Return (embedding matrix, token strings, row indices) for alphabetic tokens."""
    import holographic.io_and_interop.holographic_gdnruntime as g
    w = g.load_weights_dir(model_dir)
    if isinstance(w, tuple):
        w = w[0]
    key = [k for k in w if k.endswith("embed_tokens.weight")]
    if not key:
        raise ValueError("no embed_tokens.weight in %s" % model_dir)
    E = np.asarray(w[key[0]], np.float32)

    tok_path = os.path.join(model_dir, "tokenizer.json")
    words, rows = [], []
    if os.path.exists(tok_path):
        with open(tok_path, "r", encoding="utf-8") as fh:
            tk = json.load(fh)
        vocab = (tk.get("model") or {}).get("vocab") or {}
        for s, i in vocab.items():
            clean = s.lstrip("\u0120\u2581").lower()
            if len(clean) >= 4 and re.fullmatch(r"[a-z]+", clean) and int(i) < E.shape[0]:
                words.append(clean)
                rows.append(int(i))
    return E, words, rows


def holdout_test(E, words, rows, holdout=400, k=64, seed=0):
    """Rebuild rows the table already has, and report how close the rebuild lands.

    This is self-referential: it needs no external ground truth and works on any artifact.
    """
    rng = np.random.default_rng(seed)
    n = len(words)
    if n < holdout * 2:
        holdout = max(10, n // 4)
    order = rng.permutation(n)
    te, tr = order[:holdout], order[holdout:]
    Wtr = [words[i] for i in tr]
    Wte = [words[i] for i in te]
    Etr = E[[rows[i] for i in tr]].astype(np.float64)
    Ete = E[[rows[i] for i in te]].astype(np.float64)

    fp = FormPrior().fit(Wtr, Etr)
    rebuilt = fp.predict_via_donors(Wte, Wtr, Etr, k=k)
    mu = Etr.mean(0)

    def cos(P, Y):
        Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
        Yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12)
        return float(np.mean(np.sum(Pn * Yn, 1)))

    def r2(P, Y):
        a = float(np.sum(P * Y) / max(np.sum(P * P), 1e-12))
        return 1.0 - float(np.sum((a * P - Y) ** 2) / max(np.sum((Y - mu) ** 2), 1e-12))

    base = np.tile(mu, (len(Ete), 1))
    rand = Etr[rng.integers(0, len(Etr), size=(len(Ete), k))].mean(1)
    return {
        "n_alpha_tokens": n,
        "held_out": int(holdout),
        "k": int(k),
        "rebuilt_cos": cos(rebuilt, Ete),
        "rebuilt_r2": r2(rebuilt, Ete),
        "mean_baseline_cos": cos(base, Ete),
        "mean_baseline_r2": 0.0,
        "random_donor_cos": cos(rand, Ete),
        "random_donor_r2": r2(rand, Ete),
    }


def repair(E, words, rows, targets, k=64, blend=1.0):
    """Rebuild `targets` (indices into words/rows) from the remaining rows. Returns a copy.

    blend=1.0 replaces the row outright; blend<1.0 mixes with the existing row, which is the
    safer choice when the existing estimate is merely noisy rather than absent.
    """
    tset = set(int(t) for t in targets)
    donors = [i for i in range(len(words)) if i not in tset]
    Wd = [words[i] for i in donors]
    Ed = E[[rows[i] for i in donors]].astype(np.float64)
    fp = FormPrior().fit(Wd, Ed)
    Wt = [words[i] for i in targets]
    new = fp.predict_via_donors(Wt, Wd, Ed, k=k)
    out = E.copy()
    for j, t in enumerate(targets):
        r = rows[t]
        old = out[r].astype(np.float64)
        v = blend * new[j] + (1.0 - blend) * old
        scale = np.linalg.norm(old) / (np.linalg.norm(v) + 1e-12)
        out[r] = (v * scale).astype(E.dtype)  # preserve the row's original norm
    return out


def main(argv):
    if not argv:
        print(__doc__)
        return 3
    model_dir = argv[0]

    def opt(f, d, c):
        return c(argv[argv.index(f) + 1]) if f in argv else d

    E, words, rows = load_table(model_dir)
    print("embedding repair: %s" % model_dir)
    print("  table %s, alphabetic tokens usable: %d" % (E.shape, len(words)))
    if len(words) < 40:
        print("  too few alphabetic tokens to fit a form prior -- BLOCKED")
        return 2
    res = holdout_test(E, words, rows, holdout=opt("--holdout", 400, int), k=opt("--k", 64, int))
    print("  held out %d rows, rebuilt from the rest with k=%d donors\n" % (res["held_out"], res["k"]))
    print("  method                    cosine to true row    variance explained")
    print("  mean-of-all baseline      %.4f                %+.4f" % (res["mean_baseline_cos"], 0.0))
    print("  random donors             %.4f                %+.4f" % (res["random_donor_cos"], res["random_donor_r2"]))
    print("  form-selected donors      %.4f                %+.4f" % (res["rebuilt_cos"], res["rebuilt_r2"]))
    gain = res["rebuilt_cos"] - res["mean_baseline_cos"]
    print()
    if res["rebuilt_r2"] > 0.01 and gain > 0.01:
        print("  VERDICT: this table HAS form-meaning structure -- repair is worth running.")
    else:
        print("  VERDICT: no exploitable form-meaning structure in this table.")
        print("  On an untrained or at-chance artifact this is the expected result: the rows")
        print("  encode no relationship between spelling and meaning, so there is nothing to")
        print("  rebuild from. Run tools/fixture_gate.py to confirm which case this is.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
