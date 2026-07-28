"""AMP-1 -- Approximate Message Passing recovery (holographic_amp).

WHAT THIS IS
------------
The FIFTH member of the engine's bundle-recovery family, recovering the components of
`cue = sum_i w_i * codebook[i]`. The others:
  * LINEAR      -- one-shot correlations, keep the top m (order-free, washes out at load);
  * OCCLUSION   -- greedy matching pursuit, one atom per step, never revisited;
  * IHT         -- projected gradient descent: gradient step then keep the K largest, iterated;
  * CoSaMP      -- batch selection with a least-squares solve each round (the strongest, so far).

AMP (Donoho, Maleki & Montanari 2009) is IHT WITH ONE EXTRA TERM, and that term is the whole idea. The
plain gradient residual `z = y - A w` carries correlations between the current estimate and the design
matrix, so the effective noise seen by the denoiser is NOT Gaussian and the threshold cannot be set from
theory. AMP adds the ONSAGER CORRECTION -- a memory term proportional to the fraction of currently active
coefficients -- which cancels exactly those correlations. With it, the residual behaves like AWGN, so the
threshold can be read off the residual norm itself (STATE EVOLUTION) instead of being a tuned constant, and
the sparsity K need not be known in advance.

WHY IT WAS BUILT HERE
  RESEARCH_CONSOLIDATED.md ranks SPARCs + AMP as the #1 transplant, on the strength of a structural
  isomorphism that is real: a SPARC design matrix has L sections of M columns with one nonzero per section,
  which is exactly L codebooks of M codewords; a VSA bundle IS a noisy SPARC codeword, and unbundling IS
  section decoding. Adjacent work (Kleyko et al., Neural Computation 2023) reports information rate rising
  0.60 -> 1.26 bits/dim from decoder changes alone.

THE MEASURED RESULT -- A CROSSOVER, NOT A WIN, AND IT REFUTED THE AUTHOR'S OWN PREDICTION
-----------------------------------------------------------------------------------------
Five predictions were recorded BEFORE measuring. The load-bearing one was "AMP does NOT lift the capacity
ceiling; the likely honest result is possible-but-doesn't-pay". THAT PREDICTION WAS WRONG, and this
docstring's first draft asserted it as fact. The selftest assertion written to pin it FIRED, which is the
only reason it was caught -- the negative was written before the measurement existed. Record the sequence,
not just the conclusion: a claim written ahead of its measurement is a claim you will have to retract.

Measured against the HONEST baseline -- CoSaMP, already shipped and wired -- at D=512, N=2048, 8 seeds:

    M      M/D    AMP F1            CoSaMP F1         AMP ms   CoSaMP ms
    16     0.03   1.000 +/- 0.000   1.000 +/- 0.000     21.5         3.3
    64     0.12   1.000 +/- 0.000   1.000 +/- 0.000     21.5        25.4
    86     0.17   1.000 +/- 0.000   1.000 +/- 0.000     21.5        52.6
    128    0.25   0.896 +/- 0.078   0.709 +/- 0.224     22.1       499.7
    171    0.33   0.558 +/- 0.053   0.167 +/- 0.037     20.6       792.4
    200    0.39   0.466 +/- 0.024   0.290 +/- 0.019     21.1      1026.6

So AMP DOES push past the wall where CoSaMP falls apart, with tighter variance, at constant ~21 ms while
CoSaMP's per-round least-squares grows to a full second -- 48x faster at M=200.

THE BASELINE IS NOT A STRAWMAN, checked explicitly before claiming any of this: CoSaMP at 120 iterations
(6.5 SECONDS) still scores 0.154 at M=171 against AMP's 0.559 at 30 iterations. Both have converged; more
iterations help neither. The gap is the algorithm, not the budget.

AND NEITHER METHOD DOMINATES -- the crossover is the real finding:

  * COHERENT DICTIONARIES DESTROY AMP, exactly as the theory says they must (state evolution assumes a
    roughly i.i.d. design matrix). Measured at M=32, D=512, atoms pulled toward 8 shared directions:
        coherence   0.0     0.5     1.0     2.0     4.0
        AMP       1.000   0.052   0.026   0.021   0.026
        CoSaMP    1.000   1.000   1.000   0.562   0.135
        IHT       0.984   0.693   0.411   0.224   0.099
    AMP is the WORST member of the family the moment the dictionary stops being incoherent. CoSaMP remains
    the method there and must not be replaced by this.
  * AMP IS SLOWER AT LIGHT LOAD (21 ms vs CoSaMP's 3 ms at M=16) because its cost is flat in M.
  * AMP DOES NOT NEED K. State evolution sets its own threshold, so it can recover without being told the
    sparsity -- which occlusion, IHT and CoSaMP all require. K-free operation only works at LIGHT load
    (see the alpha note in amp_recall); at heavy load you must supply K.

USE AMP for incoherent dictionaries at heavy load, or when K is unknown and the load is light. USE CoSaMP
for anything coherent, or at light load where it is both exact and cheaper. This is the same shape as the
IHT-versus-occlusion crossover the family already documents: a fifth member, not an upgrade.

A NOTE ON WHY THE DOC'S #1 RANKING IS RIGHT FOR THE WRONG REASON. RESEARCH_CONSOLIDATED.md ranks AMP first
on the strength of beating Bottleneck 2's "20-32 instruction" ceiling. That ceiling was measured with the
LINEAR readout and never existed -- CoSaMP already held exact support to M=86. AMP is worth having anyway,
but for the band M/D 0.25-0.39, which the doc does not mention. Benchmarked against the linear figure it
would have looked like a 5x win; the honest figure is a win over CoSaMP in one regime and a catastrophic
loss in another.

DETERMINISM (per ISA.md)
  No RNG anywhere: the gradient step, the soft threshold, the Onsager term and the final support extraction
  are all deterministic given the cue and codebook. Fixed iteration count. Same inputs, same recovery.
"""

import time

import numpy as np


def soft_threshold(x, tau):
    """The soft-threshold denoiser eta(x, tau) = sign(x) * max(|x| - tau, 0) -- shrink toward zero by tau and
    clamp at zero. This is AMP's nonlinearity, and the difference from IHT's hard top-K projection is the
    reason AMP does not need to be told K: the threshold decides how many coefficients survive, and state
    evolution sets the threshold from the residual."""
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)


def amp_recall(cue, codebook, K=None, iters=30, alpha=None, tol=1e-12, stats=None):
    """Recover the active atoms of `cue` (a bundle of `codebook` rows) by Approximate Message Passing.

    Each iteration is a gradient step, a soft threshold, and a residual update carrying the ONSAGER
    CORRECTION -- a memory term proportional to the fraction of currently non-zero coefficients, which
    cancels the estimate/design-matrix correlations that make a plain gradient residual non-Gaussian. Because
    the residual then behaves like AWGN, the threshold is read from its norm (`alpha` * ||z|| / sqrt(D))
    rather than tuned, so `K` is OPTIONAL: pass it to force exactly K survivors for a head-to-head against
    the rest of the family, leave it None to let the thresholding decide the support size.

    Returns (index, weight) pairs descending by |weight| -- the same shape as occlusion_recall / iht_recall /
    cosamp_recall. Pass stats={} to read stats['iters'] and stats['tau'].

    Kept negative: this does NOT beat CoSaMP on capacity (see the module docstring and measure_vs_cosamp);
    its advantages are cost and not needing K, and it degrades on coherent dictionaries where CoSaMP does
    not."""
    A = np.asarray(codebook, float)
    if A.ndim != 2:
        raise ValueError("codebook must be 2-D (N atoms x D dims), got shape %r" % (A.shape,))
    y = np.asarray(cue, float)
    if y.ndim != 1 or y.shape[0] != A.shape[1]:
        raise ValueError("cue must be 1-D of length D=%d, got %r" % (A.shape[1], y.shape))
    n_atoms, dim = A.shape

    # THE THRESHOLD CONSTANT, AND WHY IT DEPENDS ON WHETHER K IS KNOWN. `alpha` scales the state-evolution
    # threshold, and the two modes want OPPOSITE values -- measured, D=512, N=2048, 5 seeds:
    #
    #   with K supplied (F1 of the top-K):   alpha 1.5 -> 1.000 to M=86, 0.908 at M=128
    #                                        alpha 3.0 -> 0.594 at M=64  (over-pruned)
    #   K-free (support size / capture):     alpha 1.5 -> ~150 atoms returned for a true M=8 (useless)
    #                                        alpha 3.0 -> 8 atoms, 100% capture at M<=16
    #
    # The reason is structural, not a fudge: when K is given, PRECISION IS ENFORCED DOWNSTREAM by the top-K
    # cut, so the threshold should be permissive and maximise recall. When K is absent the threshold IS the
    # support decision and must be conservative. Chosen once from the sweep above and recorded here -- a
    # fixed documented constant, not a fitted weight (core forbids learned weights).
    if alpha is None:
        alpha = 1.5 if K is not None else 3.0

    w = np.zeros(n_atoms)                 # coefficient estimate
    z = y.copy()                          # residual
    last = None
    it = 0
    tau = 0.0
    for it in range(1, iters + 1):
        # STATE EVOLUTION: the residual norm IS the effective noise level once the Onsager term is present,
        # so the threshold follows from it. This is what removes the tuned-constant and the known-K.
        tau = alpha * np.linalg.norm(z) / np.sqrt(dim)
        w_new = soft_threshold(w + A @ z, tau)
        # THE ONSAGER CORRECTION. Without this term the residual is not AWGN and the whole scheme reduces to
        # a worse-tuned IHT -- it is the single line that separates AMP from projected gradient descent.
        active = float(np.count_nonzero(w_new)) / dim
        z = y - A.T @ w_new + active * z
        if last is not None and np.linalg.norm(w_new - last) <= tol * max(1.0, np.linalg.norm(last)):
            w = w_new
            break
        last, w = w_new, w_new

    if stats is not None:
        stats["iters"] = it
        stats["tau"] = float(tau)

    # Support extraction. With K given, take the K largest so the result is directly comparable with the
    # rest of the family; without it, report every survivor of the threshold.
    nz = np.flatnonzero(w) if K is None else np.argsort(-np.abs(w))[:K]
    order = nz[np.argsort(-np.abs(w[nz]))]
    return [(int(i), float(w[i])) for i in order]


def measure_vs_cosamp(dim=512, n_atoms=2048, loads=(16, 32, 64, 86, 128, 171), seeds=6, seed0=0):
    """Head-to-head against the HONEST baseline -- CoSaMP, already shipped -- at matched load, over several
    seeds, reporting exact-support F1 mean and spread plus wall time for each.

    WHY THIS FUNCTION EXISTS: the research consolidation ranks AMP #1 by scoring it against Bottleneck 2's
    "20-32 instruction" ceiling, which was measured with the LINEAR readout. CoSaMP already holds exact
    support to M=86 at D=512. Benchmarking AMP against the linear figure would manufacture a win, so the
    comparison is pinned here against the strongest thing in the engine and kept runnable."""
    from holographic.sampling_and_signal.holographic_cosamp import cosamp_recall

    def _f1(got, want):
        got, want = set(got), set(want)
        if not got or not want:
            return 0.0
        tp = len(got & want)
        p, r = tp / len(got), tp / len(want)
        return 0.0 if p + r == 0 else 2 * p * r / (p + r)

    rows = []
    for M in loads:
        fa, fc, ta, tc = [], [], [], []
        for s in range(seeds):
            rng = np.random.default_rng(seed0 + s)
            cb = rng.standard_normal((n_atoms, dim))
            cb /= np.linalg.norm(cb, axis=1, keepdims=True)
            true = rng.choice(n_atoms, size=M, replace=False)
            cue = cb[true].sum(axis=0)
            t0 = time.perf_counter()
            ga = [i for i, _ in amp_recall(cue, cb, K=M)]
            ta.append((time.perf_counter() - t0) * 1e3)
            t0 = time.perf_counter()
            gc = [i for i, _ in cosamp_recall(cue, cb, M)]
            tc.append((time.perf_counter() - t0) * 1e3)
            fa.append(_f1(ga, true))
            fc.append(_f1(gc, true))
        rows.append({"M": M, "M_over_D": M / dim,
                     "amp_f1": float(np.mean(fa)), "amp_sd": float(np.std(fa)),
                     "cosamp_f1": float(np.mean(fc)), "cosamp_sd": float(np.std(fc)),
                     "amp_ms": float(np.median(ta)), "cosamp_ms": float(np.median(tc))})
    return rows


def _selftest():
    rng = np.random.default_rng(0)
    dim, n_atoms = 256, 1024
    cb = rng.standard_normal((n_atoms, dim))
    cb /= np.linalg.norm(cb, axis=1, keepdims=True)

    # 1. EXACT SUPPORT at light load, with K supplied -- the basic contract.
    for M in (4, 8, 16):
        true = rng.choice(n_atoms, size=M, replace=False)
        got = [i for i, _ in amp_recall(cb[true].sum(axis=0), cb, K=M)]
        assert set(got) == set(true.tolist()), "AMP missed the support at M=%d" % M

    # 2. THE CAPABILITY THAT IS ACTUALLY NEW: recovery WITHOUT being told K. Every other member of the
    #    family requires the sparsity up front; state evolution means AMP does not.
    true = rng.choice(n_atoms, size=8, replace=False)
    got = [i for i, _ in amp_recall(cb[true].sum(axis=0), cb, K=None)]
    assert set(true.tolist()) <= set(got), "K-free AMP lost part of the true support"
    # The K-free support must be TIGHT, not merely inclusive. Measured at the recorded default it returns
    # ~10 atoms for a true 8; 3x is the loose bar that still catches a regression to the ~150 the
    # recall-oriented alpha produces.
    assert len(got) <= 3 * 8, "K-free AMP returned an oversized support (%d) -- check the alpha default" % len(got)

    # 3. THE ONSAGER TERM MUST MATTER. Removing it is what turns AMP back into a badly-tuned IHT, so a
    #    numeric contract pins that the term is load-bearing rather than decorative.
    def _no_onsager(cue, A, K, iters=30, alpha=1.5):
        w, z = np.zeros(A.shape[0]), cue.copy()
        for _ in range(iters):
            tau = alpha * np.linalg.norm(z) / np.sqrt(A.shape[1])
            w = soft_threshold(w + A @ z, tau)
            z = cue - A.T @ w                      # <-- no memory term
        idx = np.argsort(-np.abs(w))[:K]
        return [int(i) for i in idx]

    M = 32
    true = rng.choice(n_atoms, size=M, replace=False)
    cue = cb[true].sum(axis=0)
    with_on = len(set(i for i, _ in amp_recall(cue, cb, K=M)) & set(true.tolist()))
    without = len(set(_no_onsager(cue, cb, M)) & set(true.tolist()))
    assert with_on >= without, "the Onsager term made recovery WORSE (%d vs %d) -- re-derive it" % (with_on, without)

    # 4. SOFT THRESHOLD is the exact shrinkage, asserted numerically (it is the denoiser; an off-by-one
    #    here silently changes every recovery).
    x = np.array([-3.0, -0.5, 0.0, 0.5, 3.0])
    assert np.allclose(soft_threshold(x, 1.0), [-2.0, 0.0, 0.0, 0.0, 2.0])

    # 5. DETERMINISM: no RNG in the solver, so a repeat must be bit-identical.
    a = amp_recall(cue, cb, K=M)
    b = amp_recall(cue, cb, K=M)
    assert a == b, "AMP is not deterministic"

    # 6. SHAPE GUARDS fail loudly rather than broadcasting into a confident wrong answer.
    for bad_cue, bad_cb in ((np.zeros(5), cb), (np.zeros(dim), np.zeros(dim))):
        try:
            amp_recall(bad_cue, bad_cb, K=2)
            raise AssertionError("AMP accepted a bad shape")
        except ValueError:
            pass

    # 7. THE CROSSOVER, PINNED IN BOTH DIRECTIONS. An earlier version of this assertion claimed AMP does
    #    NOT beat CoSaMP at high load; it FIRED, and the docstring was rewritten from the measurement
    #    rather than the test being relaxed. Both halves are now asserted, because a one-sided claim is
    #    what produced the wrong docstring in the first place.
    rows = measure_vs_cosamp(dim=256, n_atoms=1024, loads=(86,), seeds=3)
    assert rows[0]["amp_f1"] >= rows[0]["cosamp_f1"] - 1e-9, \
        "AMP no longer holds at M/D~0.33 -- re-measure and rewrite the crossover"

    # ... and the OTHER half: a coherent dictionary must still destroy AMP relative to CoSaMP. If this
    # stops being true, AMP's documented failure regime is wrong and the guidance must change.
    from holographic.sampling_and_signal.holographic_cosamp import cosamp_recall
    rng2 = np.random.default_rng(11)
    base = rng2.standard_normal((8, 256))
    coh = rng2.standard_normal((1024, 256)) + 1.0 * base[rng2.integers(0, 8, size=1024)]
    coh /= np.linalg.norm(coh, axis=1, keepdims=True)
    t = rng2.choice(1024, size=32, replace=False)
    cue2 = coh[t].sum(axis=0)
    amp_hit = len(set(i for i, _ in amp_recall(cue2, coh, K=32)) & set(t.tolist()))
    cos_hit = len(set(i for i, _ in cosamp_recall(cue2, coh, 32)) & set(t.tolist()))
    assert amp_hit < cos_hit, \
        "AMP no longer loses on a coherent dictionary (%d vs %d) -- rewrite the failure regime" % (amp_hit, cos_hit)

    print("holographic_amp: all selftests passed (support, K-free recall, Onsager, determinism, kept negative)")


if __name__ == "__main__":
    _selftest()
