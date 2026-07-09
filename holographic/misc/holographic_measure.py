"""The variance harness: every headline number gets a mean, a spread, and a confidence
interval across seeds -- so a lucky-seed point estimate can't pass as a real result.

WHY this exists
---------------
This whole engine is built on RANDOM vectors. Atoms are random, the RP-tree's
hyperplanes are random, the reservoir is random, train/test splits are shuffled. A
single-seed score is therefore a sample from a distribution, and reporting it alone hides
how wide that distribution is. For an engine whose entire pitch is "measured, not
promised," reporting a number without its noise is the sharpest blind spot -- so this
points that same discipline at the numbers themselves.

measure(run_once, seeds) runs a scored experiment once per seed and returns the mean, the
sample standard deviation, and a 95% percentile-bootstrap confidence interval (no
distributional assumptions). assert_robust(stats, floor) passes only if the LOWER CI
bound clears the floor -- which is what stops a single fortunate seed from passing a
test the typical seed would fail. report() formats "mean +/- std (95% CI [lo, hi], n)".

USE REAL DATA. The point of the harness is to characterise the real distribution of a
real claim; running it on a toy makes the spread meaningless. The measurements wired
through it here all run on real corpora (Gutenberg Alice, UDHR, Reuters, Brown).
"""
import numpy as np


def measure(run_once, seeds=range(10), n_boot=2000, boot_seed=0):
    """Run a scored experiment across seeds; return mean, std, and a 95% bootstrap CI.

    run_once(seed) -> a single scalar score (accuracy, F1, recall@1, stars, ...). The
    seed should control the RANDOM part of the system (atom mint, projection, split), so
    the spread across seeds is the genuine seed-sensitivity of the claim.
    """
    xs = np.array([float(run_once(s)) for s in seeds], dtype=float)
    rng = np.random.default_rng(boot_seed)
    boots = np.array([rng.choice(xs, len(xs), replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    std = float(xs.std(ddof=1)) if len(xs) > 1 else 0.0
    return {"mean": float(xs.mean()), "std": std, "ci": (float(lo), float(hi)),
            "n": len(xs), "scores": xs}


def assert_robust(stats, floor):
    """Pass only if the LOWER CI bound clears the floor -- not just the mean. This is
    what stops a lucky-seed point estimate from passing as a real result."""
    lo = stats["ci"][0]
    assert lo >= floor, (f"lower 95% CI {lo:.4f} < floor {floor} "
                         f"(mean {stats['mean']:.4f} +/- {stats['std']:.4f}, n={stats['n']})")


def is_fragile(stats, margin_floor):
    """A claim is FRAGILE if its spread is large relative to how far its mean sits above
    the floor it needs to clear. std >= half the margin means a couple of unlucky seeds
    could sink it -- flag it rather than report it as solid."""
    margin = stats["mean"] - margin_floor
    return margin <= 0 or stats["std"] >= 0.5 * margin


def report(name, stats, floor=None):
    """Format a stats dict as 'name: mean +/- std (95% CI [lo, hi], n)', with a
    fragile/solid tag if a floor is given."""
    lo, hi = stats["ci"]
    s = (f"{name}: {stats['mean']:.3f} +/- {stats['std']:.3f} "
         f"(95% CI [{lo:.3f}, {hi:.3f}], n={stats['n']})")
    if floor is not None:
        s += "  FRAGILE" if is_fragile(stats, floor) else "  solid"
    return s


def _demo():
    """Print the variance table for the load-bearing claims, on REAL corpora. This is
    the credibility table the design note asks for: every headline number with its
    spread and a solid/fragile verdict."""
    import re
    rows = []
    try:
        from nltk.corpus import gutenberg, udhr, brown, reuters
        gutenberg.fileids()
    except Exception:
        print("NLTK corpora unavailable; cannot run the real-data variance table.")
        return

    from holographic.misc.holographic_text import HolographicNGram, LanguageID
    from holographic.misc.holographic_segment import Segmenter, boundary_f1
    from holographic.misc.holographic_unified import UnifiedMind

    # n-gram next-char on Alice
    alice = re.sub(r"\s+", " ", re.sub(r"[^a-z ]+", " ",
                   gutenberg.raw("carroll-alice.txt").lower()))
    cut = int(len(alice) * 0.85); a_tr, a_te = alice[:cut], alice[cut:cut + 3500]
    rows.append(("ngram next-char (Alice)",
                 measure(lambda s: HolographicNGram(dim=1024, n=6, seed=s).fit(a_tr).predict_accuracy(a_te),
                         seeds=range(6)), 0.55))

    # language ID on UDHR
    files = {"en": "English-Latin1", "fr": "French_Francais-Latin1",
             "de": "German_Deutsch-Latin1", "es": "Spanish_Espanol-Latin1",
             "it": "Italian_Italiano-Latin1", "nl": "Dutch_Nederlands-Latin1"}
    texts = {k: re.sub(r"[^a-z ]+", " ", udhr.raw(f).lower()) for k, f in files.items()}

    def langid(seed):
        rng = np.random.default_rng(seed); tr, te = {}, []
        for k, full in texts.items():
            ch = [full[i:i + 200] for i in range(0, len(full) - 200, 200)]
            rng.shuffle(ch); c = int(len(ch) * 0.6)
            tr[k] = ch[:c]; te += [(x, k) for x in ch[c:]]
        lid = LanguageID(dim=512, seed=seed).fit(tr)
        return float(np.mean([lid.identify(x) == k for x, k in te]))
    rows.append(("language ID (UDHR 6-lang)", measure(langid, seeds=range(6)), 0.9))

    # segmentation F1 on Brown
    words = [w.lower() for w in brown.words(categories="news") if w.isalpha()][:1500]
    spaceless = "".join(words); truth, pos = set(), -1
    for w in words:
        pos += len(w); truth.add(pos)
    rows.append(("segmentation F1 (Brown)",
                 measure(lambda s: boundary_f1(Segmenter(dim=512, order=3, seed=s).fit(spaceless)
                                               .boundaries(spaceless, 70), truth)["f1"],
                         seeds=range(6)), 0.4))

    # topic classification on Reuters
    cats = ["earn", "acq", "crude", "trade", "money-fx"]
    docs = {c: [] for c in cats}
    for f in reuters.fileids():
        cs = reuters.categories(f)
        if len(cs) == 1 and cs[0] in cats and len(docs[cs[0]]) < 60:
            toks = [w.lower() for w in reuters.words(f) if w.isalpha()][:120]
            if len(toks) > 20:
                docs[cs[0]].append(" ".join(toks))
    alldocs = [(t, c) for c in cats for t in docs[c]]

    def reuters_acc(seed):
        rng = np.random.default_rng(seed); items = list(alldocs); rng.shuffle(items)
        by = {}
        for t, c in items:
            by.setdefault(c, []).append(t)
        tr, te = [], []
        for c, ts in by.items():
            k = int(len(ts) * 0.7); tr += [(t, c) for t in ts[:k]]; te += [(t, c) for t in ts[k:]]
        m = UnifiedMind(dim=1024, seed=seed); m.absorb(tr)
        return float(np.mean([m.classify(t)[0] == c for t, c in te]))
    rows.append(("topic classify (Reuters 5-cat)", measure(reuters_acc, seeds=range(6)), 0.72))

    print("VARIANCE TABLE -- load-bearing claims on real corpora (mean +/- std, 95% CI):")
    for name, stats, floor in rows:
        print("  " + report(name, stats, floor))


if __name__ == "__main__":
    _demo()
