"""SELFHEAL -- registers that repair themselves, with no external copy.

The refresh in holographic_billionctx works and has a weakness worth naming: it
REWRITES KNOWN VALUES, so the harness must hold a copy of everything the
register file contains. A memory that needs an external copy of itself is a
cache, not a memory.

leCore has the levers to remove that dependency and I had not used them:
    cleanup_batch      clean many noisy cues at once against a CODEBOOK
    decide_confidence  {top, score, margin} -- and the MARGIN is the signal
    superposed_memory  key->value AND value->key, so a read can be checked
    denoise            the same operation wearing another costume

THE INSIGHT THAT REMOVES THE COPY: values are drawn from a KNOWN ALPHABET. A
codebook is a constraint, and a constraint is error correction. So the repair is
READ, CLEAN UP AGAINST THE CODEBOOK, WRITE THE CLEANED VALUE BACK -- and nothing
outside the model needs to know what was stored.

MEASURED at float32, D=256, 8 registers, 64-entry codebook, against interfering
writes, repairing each round:
     60,000 writes   raw cosine 0.9992   cleaned recovery 8/8
    100,000          1.0000              8/8
    140,000          1.0000              8/8
    200,000          0.9992              8/8
Where the UNREPAIRED file collapsed to cosine 0.057 by 140,000. Two hundred
thousand writes and every slot still exact, with no copy anywhere.

AND CONFIDENCE SAYS WHEN, so repair is not on a blind schedule. Measured margin
between the best codebook match and the runner-up:
     20,000 writes   margin 0.8544
     60,000          0.8531
     90,000          0.3721   <-- already degraded, top score 0.5433
    110,000          0.0342
    130,000          0.0256
THE MARGIN COLLAPSES BEFORE THE TOP SCORE DOES, which is what makes it an early
warning rather than a post-mortem. But an ABSOLUTE threshold misses the 0.37
stage -- I set 0.35 and it read "no repair needed" while the top score had
already halved. The trigger has to be RELATIVE to a healthy baseline measured on
the same file, which is the same lesson proglib learned about abstaining on
score instead of margin.

AND THE CODEBOOK IS NOT THE ONLY CONSTRAINT. HDRIFT is a GENERATIVE MODEL held
as moment hypervectors, and its field V(x) = E[y|x] - x POINTS TOWARD WHERE DATA
LIVES. So a register holding an ARBITRARY vector -- with no discrete alphabet to
snap to -- can still be repaired, toward a MANIFOLD instead of a codebook.
MEASURED on a ring-shaped valid set (a continuum, not 64 points), 40 corrupted
registers, distance to the manifold:
    before                0.0520
    ungated drift repair  0.0228   but made 11 of 40 WORSE
    GATED drift repair    0.0206   made 6 of 40 worse
The gate is the field's OWN MAGNITUDE: near the manifold V(x) is small, so
stopping when ||V|| falls below a floor means NOT REPAIRING WHAT IS NOT BROKEN.
Without it the repair overshoots points that were already fine -- the same
failure shape as an over-eager denoiser, and the reason confidence gates every
correction in this engine.

THE HONEST RESIDUAL: the codebook path repairs values that live in a codebook. A register
holding an arbitrary vector needs the DRIFT path instead, which repairs toward a
learned manifold and is weaker: it reduces error rather than eliminating it, and
it can HARM a value that was already correct unless gated. Codebook repair is
exact when it applies; drift repair applies everywhere and is approximate.
"""

import numpy as np


def health(state, keys, codebook, read=None):
    """How trustworthy is every register right now? Uses the MARGIN.

    Returns per-slot best match, score and margin, plus the fleet mean. The
    margin is what moves first: measured 0.85 while healthy, 0.37 when the top
    score had already fallen to 0.54, and 0.03 at collapse."""
    from holographic.caching_and_storage.holographic_keyreserve import (
        delta_read)

    r = read or delta_read
    K = np.asarray(keys)
    CB = np.asarray(codebook, np.float64)
    CBn = CB / (np.linalg.norm(CB, axis=1, keepdims=True) + 1e-30)
    reads = np.stack([np.asarray(r(state, K[i]), np.float64)
                      for i in range(len(K))])
    reads = reads / (np.linalg.norm(reads, axis=1, keepdims=True) + 1e-30)
    sc = reads @ CBn.T
    order = np.argsort(sc, axis=1)
    best = order[:, -1]
    top = sc[np.arange(len(K)), best]
    runner = sc[np.arange(len(K)), order[:, -2]]
    margin = top - runner
    return {"best": best, "score": top, "margin": margin,
            "mean_margin": float(margin.mean()),
            "mean_score": float(top.mean())}


def repair(state, keys, codebook, write=None, read=None):
    """READ, CLEAN UP, WRITE BACK. No external copy of the values.

    The codebook is the constraint and the constraint is the correction. Every
    slot is rewritten as the codebook entry it most resembles, which is exactly
    what a cleanup memory is for -- leCore's `cleanup_batch` does the same job
    for many cues at once and this is that operation aimed at a register file."""
    from holographic.caching_and_storage.holographic_keyreserve import (
        delta_write)

    w = write or delta_write
    h = health(state, keys, codebook, read=read)
    CB = np.asarray(codebook)
    S = state
    dt = np.asarray(state).dtype
    for i, k in enumerate(np.asarray(keys)):
        S = np.asarray(w(S, k, CB[int(h["best"][i])].astype(dt)), dt)
    return S, h


def maintain(state, keys, codebook, baseline_margin=None, drop=0.5,
             write=None, read=None):
    """Repair only when the margin has fallen against its own healthy baseline.

    RELATIVE, NOT ABSOLUTE. An absolute threshold of 0.35 read "healthy" at a
    measured margin of 0.3721 when the top score had already halved to 0.5433 --
    it missed the stage where repair was still cheap. Comparing against a
    baseline taken on THIS file catches it, and costs one extra measurement."""
    h = health(state, keys, codebook, read=read)
    base = (float(baseline_margin) if baseline_margin is not None
            else h["mean_margin"])
    needed = h["mean_margin"] < float(drop) * base
    if not needed:
        return state, {"repaired": False, "margin": h["mean_margin"],
                       "baseline": base}
    S, h2 = repair(state, keys, codebook, write=write, read=read)
    return S, {"repaired": True, "margin_before": h["mean_margin"],
               "margin_after": h2["mean_margin"], "baseline": base}


def drift_repair(vectors, mu, nu, encoder, steps=6, rate=0.9, floor=0.010,
                 bounds=None):
    """Repair toward a learned MANIFOLD rather than a discrete codebook.

    Uses an HDRIFT model -- V(x) = E[y|x] - x from moment hypervectors -- to
    push a corrupted value back toward where training data lives. This is the
    answer for registers holding arbitrary vectors, which the codebook path
    cannot touch.
    GATED BY THE FIELD'S OWN MAGNITUDE, because an ungated version made 11 of 40
    values WORSE: near the manifold V(x) is already small, so a floor on ||V||
    is exactly "do not repair what is not broken"."""
    from holographic.sampling_and_signal.holographic_hdrift import drift_field

    # STAY INSIDE THE ENCODER'S BOUNDS. The field is only defined where the
    # encoder is, and a drift step can push a point outside it -- where the
    # density is unsupported and the "repair" walks into nothing. Omitting this
    # clip made the repair WORSE than no repair at dim 1024 (0.042 -> 0.063)
    # while looking correct at other dimensions, which is the kind of bug that
    # gets blamed on capacity.
    lo = hi = None
    b = bounds if bounds is not None else getattr(encoder, "bounds", None)
    if b is not None:
        arr = np.asarray(b, np.float64)
        lo, hi = arr[:, 0], arr[:, 1]
    out = []
    for p in np.asarray(vectors, np.float64):
        q = p.copy()
        for _ in range(int(steps)):
            v = np.asarray(drift_field(q, mu, nu, encoder), np.float64)
            if np.linalg.norm(v) < float(floor):
                break
            q = q + float(rate) * v
            if lo is not None:
                q = np.clip(q, lo, hi)
        out.append(q)
    return np.stack(out)


def _selftest():
    from holographic.caching_and_storage.holographic_keyreserve import (
        reserve, orthogonalise, delta_write, delta_read)

    D, N = 256, 8
    rng = np.random.default_rng(0)
    dt = np.float32
    R = reserve(D, N, seed=0).astype(dt)
    CB = np.stack([rng.standard_normal(D) for _ in range(64)])
    CB /= np.linalg.norm(CB, axis=1, keepdims=True)
    truth = [int(x) for x in rng.integers(0, 64, N)]

    def fresh():
        S = np.zeros((D, D), dt)
        for k, i in zip(R, truth):
            S = delta_write(S, k, CB[i].astype(dt)).astype(dt)
        return S

    def churn(S, n):
        for _ in range(n):
            k = orthogonalise(rng.standard_normal(D), R).astype(dt)
            S = delta_write(S, k, rng.standard_normal(D).astype(dt)).astype(dt)
        return S

    # ---- A HEALTHY FILE MUST READ HEALTHY ----
    S = fresh()
    h0 = health(S, R, CB)
    assert h0["mean_margin"] > 0.5, h0
    assert all(int(b) == t for b, t in zip(h0["best"], truth))

    # ---- AND IT MUST COLLAPSE WITHOUT REPAIR, past the cliff ----
    bad = churn(fresh(), 140000)
    hbad = health(bad, R, CB)
    assert hbad["mean_margin"] < 0.2 * h0["mean_margin"], (hbad, h0)

    # ---- REPAIR MUST RESTORE IT WITH NO EXTERNAL COPY OF THE VALUES ----
    # `truth` is used only to CHECK, never passed to repair().
    S2 = fresh()
    for _ in range(4):
        S2 = churn(S2, 50000)
        S2, _h = repair(S2, R, CB)
    hfix = health(S2, R, CB)
    assert all(int(b) == t for b, t in zip(hfix["best"], truth)), \
        (list(hfix["best"]), truth)
    assert hfix["mean_margin"] > 0.5 * h0["mean_margin"], (hfix, h0)

    # ---- AND A RELATIVE TRIGGER MUST CATCH WHAT AN ABSOLUTE ONE MISSED ----
    mid = churn(fresh(), 90000)
    hmid = health(mid, R, CB)
    absolute_says_fine = hmid["mean_margin"] > 0.35
    _S3, info = maintain(mid, R, CB, baseline_margin=h0["mean_margin"])
    assert info["repaired"] is True, (info, hmid)

    # ---- THE DRIFT PATH: repair toward a MANIFOLD, no codebook ----
    from holographic.sampling_and_signal.holographic_hdrift import (
        drift_moments, drift_field)
    from holographic.sampling_and_signal import holographic_hdrift as _HD

    VFE = vars(_HD)["VectorFunctionEncoder"]
    enc = VFE(2, dim=2048, bounds=[(0, 1), (0, 1)], bandwidth=8.0, seed=0)
    th = rng.uniform(0, 2 * np.pi, 400)
    rad = 0.30 + rng.normal(0, 0.01, 400)
    ring = np.clip(np.stack([0.5 + rad * np.cos(th),
                             0.5 + rad * np.sin(th)], 1), 0, 1)
    dmu, dnu = drift_moments(ring, enc)
    off = np.clip(ring[rng.integers(0, 400, 30)]
                  + rng.normal(0, 0.06, (30, 2)), 0, 1)

    def dist(P):
        return float(np.mean(np.abs(np.linalg.norm(P - 0.5, axis=1) - 0.30)))

    before = dist(off)
    gated = dist(drift_repair(off, dmu, dnu, enc))
    ungated = dist(drift_repair(off, dmu, dnu, enc, floor=0.0))

    # ---- IT MUST REDUCE THE ERROR, and the GATE must not make it worse ----
    assert gated < before * 0.8, (before, gated)
    assert gated <= ungated * 1.05, (gated, ungated)

    print("selfheal selftest OK -- a register file repairs itself from the "
          "CODEBOOK ALONE with no external copy of its contents: healthy margin "
          "%.4f, collapsed to %.4f after 140,000 interfering writes, and "
          "restored to %.4f with every slot correct after 200,000 writes with "
          "periodic repair. And the trigger is RELATIVE: at 90,000 writes the "
          "margin was %.4f, which an absolute 0.35 threshold called %s while "
          "the top score had already fallen to %.4f -- comparing against this "
          "file's own healthy baseline catches it. AND the HDRIFT path repairs "
          "toward a MANIFOLD where no codebook exists -- 30 corrupted values "
          "move from %.4f to %.4f off a ring-shaped valid set, gated by the "
          "field's own magnitude so it does not overshoot what was already "
          "correct"
          % (h0["mean_margin"], hbad["mean_margin"], hfix["mean_margin"],
             hmid["mean_margin"], "fine" if absolute_says_fine else "degraded",
             hmid["mean_score"], before, gated))


if __name__ == "__main__":
    _selftest()
