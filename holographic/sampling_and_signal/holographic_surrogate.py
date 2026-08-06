"""holographic_surrogate.py -- the phase-randomized null for CONTINUOUS signals (the honest baseline the panel kept
asking for).

WHY THIS MODULE EXISTS
----------------------
The shuffle-null (a permutation) is the right null for DISCRETE corpora: it destroys order while preserving the
symbol histogram, so structure counts only above chance co-occurrence. But for a CONTINUOUS, autocorrelated
signal a permutation is TOO STRONG a null -- it destroys the autocorrelation that even a trivial forecaster
(persistence, a smoother) exploits, so any such method looks brilliant against it. Both the SETI seat (forecast
abstain gate) and the adaptive pipeline flagged this: the honest continuous baseline must preserve the
autocorrelation and ask whether there is structure BEYOND it.

THE METHOD (Theiler et al. 1992, surrogate data)
  A phase-randomized surrogate has the SAME power spectrum (hence the same autocorrelation, by Wiener-Khinchin) as
  the original, but random phases -- so any DETERMINISTIC / nonlinear structure is destroyed while the linear
  second-order statistics are preserved exactly. Recipe: FFT the signal, keep the magnitudes, replace the phases
  with random ones (kept antisymmetric so the inverse transform is real), inverse-FFT. A statistic that is large
  on the real signal but not on an ensemble of its surrogates reflects structure the power spectrum ALONE does not
  explain -- the honest thing to credit.

WHAT IT IS FOR
  `surrogate_zscore(x, statistic)` measures any structure `statistic` on `x` against an ensemble of phase-
  randomized surrogates and reports a z-score -- the continuous-signal analogue of the shuffle-null z the discrete
  path already uses. Wire this behind a forecast/structure gate for continuous data.

KEPT NEGATIVE: a phase-randomized surrogate assumes the signal's non-Gaussianity is not itself the structure of
interest -- for strongly non-Gaussian amplitudes the amplitude-adjusted variant (AAFT) is stricter. We ship the
basic phase-randomization (the common case) and NAME AAFT as the follow-on, rather than silently over-claiming.

NumPy only. Deterministic given the seed.
"""

import numpy as np


def phase_randomize(x, seed=0, rng=None):
    """Return a PHASE-RANDOMIZED surrogate of a 1-D real signal `x`: same power spectrum (same autocorrelation) as
    `x`, but random phases -- so any deterministic/nonlinear structure is destroyed while the linear second-order
    statistics are preserved EXACTLY (Theiler et al. 1992). The phases are kept antisymmetric so the inverse
    transform is real, and the DC / Nyquist bins are left real. Deterministic given `seed`. This is the honest
    null for a continuous, autocorrelated signal -- unlike a permutation, it does NOT destroy the autocorrelation
    a trivial forecaster would exploit.

    `rng` accepts an ALREADY-CONSTRUCTED generator, for callers drawing many surrogates in a loop who must
    not restart the stream each time. Added when four modules (hrnn, statedemand, triage, unified_p14) were
    found to have each inlined this function verbatim -- they took a live rng, which is the only reason they
    could not call this one. Passing `rng` bypasses `seed`; the seed path is untouched and bit-identical.

    KEPT NEGATIVE, and the reason unifying them was a CORRECTNESS fix rather than tidying: all four copies
    set the DC phase to literal 0.0. DC is real, so its true phase is 0 OR pi -- and for a signal with a
    NEGATIVE mean it is pi. Forcing 0.0 silently FLIPS THE SIGN OF THE MEAN in the surrogate. This function
    preserves np.angle(F[0]) instead, which is why the copies had to go rather than be budgeted."""
    x = np.asarray(x, float).ravel()
    n = len(x)
    F = np.fft.rfft(x)
    mag = np.abs(F)
    rng = np.random.default_rng(seed) if rng is None else rng
    # random phases for the interior bins; DC (0) and, for even n, Nyquist stay at their original (real) phase.
    phases = rng.uniform(0, 2 * np.pi, size=len(F))
    phases[0] = np.angle(F[0])                                 # DC must stay real
    if n % 2 == 0:
        phases[-1] = np.angle(F[-1])                           # Nyquist bin must stay real for even-length signals
    surrogate = np.fft.irfft(mag * np.exp(1j * phases), n=n)
    return surrogate


def surrogate_zscore(x, statistic, n_surrogates=64, seed=0):
    """Measure a structure `statistic(x)` against an ensemble of PHASE-RANDOMIZED surrogates and report how far
    the real value exceeds the surrogate null, in null standard deviations (a z-score). `statistic` is any
    callable signal -> float that is LARGE when there is structure the power spectrum alone does not explain (e.g.
    a forecaster's skill, a nonlinearity measure). Because the surrogates share the real signal's autocorrelation,
    a high z means structure BEYOND linear autocorrelation -- the honest continuous-signal null. Returns a dict:
    `value` (statistic on x), `null_mean`, `null_std`, `z`. Deterministic given `seed`."""
    x = np.asarray(x, float).ravel()
    value = float(statistic(x))
    null = np.empty(n_surrogates)
    for i in range(n_surrogates):
        null[i] = float(statistic(phase_randomize(x, seed=seed + i + 1)))
    null_mean = float(null.mean())
    null_std = float(null.std()) + 1e-12
    return {"value": value, "null_mean": null_mean, "null_std": null_std,
            "z": (value - null_mean) / null_std}


def amplitude_adjusted_surrogate(x, seed=0):
    """AAFT (amplitude-adjusted Fourier transform) surrogate -- the STRICTER null for NON-GAUSSIAN signals
    (Theiler et al. 1992). Basic phase_randomize preserves the power spectrum but GAUSSIANIZES the marginal (it is
    a sum of sinusoids, so the central limit theorem pulls its histogram toward a bell curve) -- which destroys the
    fat tails of, e.g., price returns and makes any tail-sensitive statistic falsely flag the surrogate as
    'different'. AAFT preserves BOTH the exact amplitude DISTRIBUTION and (approximately) the power spectrum.

    Recipe: (1) build a Gaussian signal with the SAME rank-order as x; (2) phase-randomize it (Gaussian marginal,
    so phase-randomization does not distort it); (3) map x's sorted amplitudes onto that surrogate's rank-order --
    so the result is a PERMUTATION of x's own values (exact same histogram) arranged with randomized phase.
    Deterministic given `seed`.

    KEPT NEGATIVE: AAFT's spectrum match is APPROXIMATE (the rank-remapping perturbs it slightly), unlike basic
    phase_randomize which is exact -- a known bias for strongly-coloured non-Gaussian signals (the iterated variant
    IAAFT tightens it, named as the follow-on). Use basic phase_randomize when the signal is ~Gaussian and the
    spectrum must match exactly; use AAFT when the amplitude distribution matters (fat tails)."""
    x = np.asarray(x, float).ravel()
    n = len(x)
    rng = np.random.default_rng(seed)
    # (1) a Gaussian signal with the same rank-order as x: place sorted Gaussian draws at x's ranks.
    ranks = np.argsort(np.argsort(x))                          # rank of each sample of x (0..n-1)
    gauss = np.sort(rng.normal(size=n))
    g = gauss[ranks]                                           # Gaussian marginal, x's ordering
    # (2) phase-randomize the Gaussian version (phase-randomization is faithful for a Gaussian marginal).
    g_surr = phase_randomize(g, seed=seed + 104729)           # a distinct sub-seed for the phase draw
    # (3) give the surrogate x's EXACT amplitude distribution by mapping x's sorted values onto g_surr's ranks.
    x_sorted = np.sort(x)
    surr_ranks = np.argsort(np.argsort(g_surr))
    return x_sorted[surr_ranks]


def iaaft_surrogate(x, n_iter=100, tol=1e-8, seed=0):
    """IAAFT (iterated amplitude-adjusted Fourier transform) surrogate -- the gold-standard null that matches BOTH
    the exact amplitude distribution AND (to convergence) the exact power spectrum (Schreiber & Schmitz 1996).
    AAFT only APPROXIMATES the spectrum; IAAFT fixes that by ITERATING two projections until they agree -- the
    same 'iterate a projection' move guide_structure/IK/PBD/the resonator all share, here alternating between the
    spectrum constraint and the amplitude-distribution constraint:

      (a) SPECTRUM step: FFT the current surrogate, replace its magnitudes with the TARGET (original) magnitudes,
          keep its phases, inverse-FFT -> exact spectrum, but amplitudes drift.
      (b) AMPLITUDE step: rank-order the result and map the original's SORTED amplitudes onto those ranks ->
          exact amplitude distribution, but the spectrum drifts slightly.

    Iterating (a),(b) converges: each step is a projection onto a constraint set, and the fixed point satisfies
    both. Stops when the ranks stop changing (a true fixed point) or `n_iter` runs out. Returns the surrogate.
    Deterministic given `seed`.

    WHY prefer this over AAFT: for strongly-coloured non-Gaussian signals (e.g. fat-tailed price returns with real
    autocorrelation) AAFT's approximate spectrum biases the null; IAAFT removes that bias. It costs the iterations.
    KEPT NEGATIVE: IAAFT can only satisfy BOTH constraints exactly if they are compatible; when they conflict it
    settles at a compromise (the ranks oscillate) -- we return the last amplitude-exact state and report via the
    caller's own spectrum check, never pretending both are perfect."""
    x = np.asarray(x, float).ravel()
    n = len(x)
    target_mag = np.abs(np.fft.rfft(x))
    x_sorted = np.sort(x)
    rng = np.random.default_rng(seed)
    # start from a random permutation of x (exact amplitude distribution, random spectrum/phase).
    surr = x[rng.permutation(n)]
    prev_ranks = None
    for _ in range(n_iter):
        # (a) spectrum step: impose the target magnitudes, keep current phases.
        F = np.fft.rfft(surr)
        phases = np.angle(F)
        surr = np.fft.irfft(target_mag * np.exp(1j * phases), n=n)
        # (b) amplitude step: impose the exact amplitude distribution via rank mapping.
        ranks = np.argsort(np.argsort(surr))
        surr = x_sorted[ranks]
        # converged when the ordering stops changing (a fixed point of the two projections).
        if prev_ranks is not None and np.array_equal(ranks, prev_ranks):
            break
        prev_ranks = ranks
    return surr


def sign_flip(x, seed=0):
    """SIGN-FLIP surrogate -- randomise the DIRECTION of every sample while keeping its MAGNITUDE, and hence
    keeping magnitude clustering (volatility clustering) EXACTLY intact. |surrogate| == |x| elementwise.

    This is the workhorse null for a DIRECTIONAL claim ("this signal predicts which way the next move goes").
    A plain shuffle would also destroy the magnitude clustering, so a directional statistic measured against it
    is credited for structure that lives in the magnitudes -- which was never the claim. Sign-flipping destroys
    exactly the thing under test and nothing else, which is what a procedure-matched null is supposed to do.

    KEPT NEGATIVE (pinned in _selftest): this is the WRONG null for any statistic that is a function of |x|
    alone -- variance, energy, absolute-value autocorrelation, drawdown magnitude. Such a statistic is IDENTICAL
    on every sign-flip surrogate, so the null has ZERO spread and the z-score is meaningless (a divide-by-nothing
    that a naive harness will report as an enormous z). If your statistic does not change sign when the data
    does, use `iid_shuffle` or `phase_randomize` instead. Deterministic given `seed`."""
    x = np.asarray(x, float).ravel()
    rng = np.random.default_rng(seed)
    # +-1 per sample, drawn independently: the sign channel is destroyed, the magnitude channel untouched.
    signs = rng.integers(0, 2, size=len(x)) * 2 - 1
    return x * signs


def iid_shuffle(x, seed=0):
    """IID-SHUFFLE surrogate -- a plain random permutation. Preserves the exact value histogram and destroys ALL
    ordering (short and long range alike). The strongest, bluntest null: use it when the claim is "there is ANY
    temporal structure here at all".

    KEPT NEGATIVE: too strong a null for most continuous signals. Because it destroys the autocorrelation that
    even a trivial forecaster (persistence, a smoother) exploits, ANY such method looks brilliant against it --
    the module docstring's founding complaint. Reach for `phase_randomize` (keeps the spectrum) or
    `block_shuffle` (keeps short-range structure) unless you really mean to test against total disorder.
    Deterministic given `seed`."""
    x = np.asarray(x, float).ravel()
    rng = np.random.default_rng(seed)
    return x[rng.permutation(len(x))]


def block_shuffle(x, block, seed=0):
    """BLOCK-SHUFFLE surrogate (the moving-block bootstrap's null) -- cut `x` into contiguous blocks of length
    `block` and shuffle the BLOCK ORDER. Structure SHORTER than `block` survives intact inside each block;
    structure LONGER than `block` is destroyed. The block length is therefore a dial that says which scale the
    claim is about: "is there structure beyond `block` samples?"

    Use it to separate scales -- e.g. keep intraday shape but destroy day-to-day ordering. A trailing partial
    block is kept whole and shuffled with the rest, so no samples are dropped and the value histogram is exact.

    KEPT NEGATIVE: the JOINS between reordered blocks are discontinuities that did not exist in `x`. Any
    statistic sensitive to jumps (a range/gap detector, a high-order difference, an event counter keyed on large
    moves) sees roughly `len(x)/block` fake events per surrogate and will read as significant in the WRONG
    direction. With block=1 this degenerates to `iid_shuffle` (every sample is a join). Deterministic given
    `seed`."""
    x = np.asarray(x, float).ravel()
    block = int(block)
    if block < 1:
        raise ValueError("block must be >= 1, got %r (use block=1 for the iid_shuffle degenerate case)" % (block,))
    rng = np.random.default_rng(seed)
    # split into contiguous chunks; the last one may be short and is shuffled along with the rest (no drops).
    edges = list(range(0, len(x), block))
    chunks = [x[i:i + block] for i in edges]
    order = rng.permutation(len(chunks))
    return np.concatenate([chunks[i] for i in order]) if chunks else x.copy()


# The surrogate kinds a caller can name by string, so a pipeline/null harness can take `surrogate="sign_flip"`
# from a config or an HTTP request instead of a callable. Each entry is fn(x, seed) -> surrogate.
_SURROGATE_KINDS = {
    "phase": lambda x, seed: phase_randomize(x, seed=seed),
    "aaft": lambda x, seed: amplitude_adjusted_surrogate(x, seed=seed),
    "iaaft": lambda x, seed: iaaft_surrogate(x, seed=seed),
    "sign_flip": lambda x, seed: sign_flip(x, seed=seed),
    "iid_shuffle": lambda x, seed: iid_shuffle(x, seed=seed),
}


def make_surrogate(kind, **kwargs):
    """Resolve a surrogate NAME to a callable fn(x, seed) -> surrogate, so harnesses can take the null as a
    string ("phase", "aaft", "iaaft", "sign_flip", "iid_shuffle", "block_shuffle") from config or over HTTP
    instead of requiring a Python callable. `block_shuffle` needs its scale: make_surrogate("block_shuffle",
    block=24). Extra kwargs are bound into the returned callable. Raises ValueError naming the valid kinds.

    Passing an already-callable `kind` returns it unchanged, so every harness in the engine can accept either
    form on the same parameter."""
    if callable(kind):
        return kind
    if kind == "block_shuffle":
        block = kwargs.get("block")
        if block is None:
            raise ValueError("surrogate 'block_shuffle' requires block=<int> (the scale below which structure "
                             "survives); e.g. make_surrogate('block_shuffle', block=24)")
        return lambda x, seed: block_shuffle(x, block, seed=seed)
    if kind not in _SURROGATE_KINDS:
        raise ValueError("unknown surrogate %r; valid kinds are: %s, block_shuffle (or pass a callable "
                         "fn(x, seed))" % (kind, ", ".join(sorted(_SURROGATE_KINDS))))
    return _SURROGATE_KINDS[kind]


def surrogate_ensemble(x, kind="phase", n=200, seed=0, **kwargs):
    """Yield `n` surrogates of `x` one at a time as a GENERATOR -- the memory-light form for long series, where
    materialising n x len(x) floats is the difference between fitting in cache and swapping. Each surrogate gets
    its own sub-seed (seed + i + 1), so the ensemble is reproducible and no two members share a draw.

    `kind` is any name `make_surrogate` accepts, or a callable fn(x, seed). Consume it in a loop:

        null = np.array([stat(s) for s in surrogate_ensemble(x, "sign_flip", n=500)])

    The existing single-surrogate functions are unchanged and still return arrays -- this is an additive,
    opt-in streaming form, not a replacement."""
    fn = make_surrogate(kind, **kwargs)
    x = np.asarray(x, float).ravel()
    for i in range(int(n)):
        yield fn(x, seed + i + 1)


def surrogate_batch(x, kind="phase", n=200, seed=0, **kwargs):
    """The MATERIALISED form of surrogate_ensemble: an (n, len(x)) array instead of a generator. Same members,
    same sub-seeds, same order -- surrogate_batch(...)[i] is bit-identical to the i-th yield of
    surrogate_ensemble(...) at the same seed.

    Exists because a generator cannot cross a process boundary: over the HTTP service a generator degrades to a
    repr stub, which is a dead end for an agent. The in-process caller keeps the memory-light generator; the
    remote caller asks for the array. Same split as proc_texture (a callable) vs texture_image (its JSON
    sibling). Cost is explicit: n * len(x) floats resident at once, so keep  modest over the wire."""
    return np.array(list(surrogate_ensemble(x, kind=kind, n=n, seed=seed, **kwargs)), dtype=float)


def trev(x, lag=1):
    """TIME-REVERSAL ASYMMETRY statistic (Ramsey & Rothman 1996; Theiler's `trev`) -- the normalised third moment
    of the lagged difference:

        trev = mean((x[t+lag] - x[t])**3) / mean((x[t+lag] - x[t])**2)**1.5

    A series played BACKWARDS has its differences negated, so the odd moment flips sign while the even one does
    not: trev is exactly zero for any process whose statistics are invariant under time reversal, and non-zero
    when the rises and the falls have different SHAPES (slow grind up / fast crash down being the archetype).
    Normalising by the variance term makes it scale-free and comparable across series.

    A non-zero value on its own means nothing -- finite samples give non-zero odd moments by luck. Pair it with
    `time_arrow_test`, which measures it against a surrogate ensemble that shares the signal's spectrum.

    Time-irreversibility implies the generating process is NONLINEAR (a linear Gaussian process is time-
    reversible), which is why this is a standard first-pass triage flag on an unfamiliar signal."""
    v = np.asarray(x, float).ravel()
    lag = int(lag)
    if lag < 1 or lag >= len(v):
        raise ValueError("lag must satisfy 1 <= lag < len(x) (got lag=%r, len=%d)" % (lag, len(v)))
    d = v[lag:] - v[:-lag]
    denom = float(np.mean(d ** 2)) ** 1.5
    if denom < 1e-300:
        return 0.0                                             # a constant series has no arrow to measure
    return float(np.mean(d ** 3) / denom)


def time_arrow_test(x, lag=1, n_surrogates=200, seed=0, kind="iaaft"):
    """Does this series have an ARROW OF TIME? Measures `trev` on `x` against an ensemble of surrogates and
    reports {value, null_mean, null_std, z, p, n_surrogates, kind}. `p` is the two-sided fraction of the null at
    least as extreme, with the +1 plug so it is never exactly 0. A large |z| says the rises and falls have
    genuinely different shapes -- the signal was produced by a NONLINEAR process (linear Gaussian processes are
    time-reversible), which is a triage flag for "look harder here", not a detection of anything in particular.

    Default `kind="iaaft"` on purpose. Basic `phase_randomize` also GAUSSIANISES the marginal, and a skewed
    marginal alone produces a non-zero trev -- so against a phase-randomised null a merely SKEWED series scores
    a large z that has nothing to do with its dynamics. IAAFT keeps the exact amplitude distribution, so what is
    left to explain is the ORDERING. Pass kind="phase" only when the marginal is known to be symmetric.

    KEPT NEGATIVE, measured and expensive: a significant global arrow can be entirely DIFFUSE. In the campaign
    this statistic reached z=+6.4 (daily) and z=+4.0 (5-minute) on a real instrument, and all three attempts to
    LOCALISE that asymmetry -- to find windows where it concentrated and condition on them -- came back null. An
    arrow in aggregate does NOT imply per-window predictability. Report it as a property of the process, never
    as a signal."""
    x = np.asarray(x, float).ravel()
    value = trev(x, lag=lag)
    null = np.array([trev(s, lag=lag) for s in surrogate_ensemble(x, kind, n=n_surrogates, seed=seed)], float)
    null_mean = float(null.mean())
    null_std = float(null.std())
    z = (value - null_mean) / (null_std + 1e-300)
    # two-sided p about the NULL's own centre (the +1 plug, North et al. 2002 -- never exactly zero).
    extreme = int(np.sum(np.abs(null - null_mean) >= abs(value - null_mean)))
    p = (extreme + 1) / (len(null) + 1)
    return {"value": value, "null_mean": null_mean, "null_std": null_std, "z": float(z),
            "p": float(p), "n_surrogates": int(len(null)), "kind": kind if isinstance(kind, str) else "callable"}

def _selftest():
    """Contracts:

    1. A phase-randomized surrogate has (near-)identical power spectrum to the original -- the autocorrelation is
       PRESERVED (this is the whole point; a permutation would fail this).
    2. A permutation, by contrast, DESTROYS the autocorrelation -- demonstrating why the surrogate is the honest
       null for continuous data.
    3. surrogate_zscore: a DETERMINISTIC nonlinear signal (whose structure is not captured by its spectrum) scores
       high against the surrogate null; a linear-Gaussian process (structure fully in its spectrum) scores near 0.
    4. Determinism.
    """
    rng = np.random.default_rng(0)
    n = 1024
    # an autocorrelated signal: a smooth random walk (strong autocorrelation, structure IS its spectrum).
    walk = np.cumsum(rng.normal(size=n))
    walk -= walk.mean()

    # (1) surrogate preserves the power spectrum.
    surr = phase_randomize(walk, seed=1)
    ps_x = np.abs(np.fft.rfft(walk))
    ps_s = np.abs(np.fft.rfft(surr))
    assert np.allclose(ps_x, ps_s, atol=1e-6), "surrogate must preserve the power spectrum"

    # (2) a permutation destroys the autocorrelation (lag-1), the surrogate preserves it.
    def ac1(v):
        v = v - v.mean()
        return float(np.dot(v[:-1], v[1:]) / (np.dot(v, v) + 1e-12))
    perm = walk[rng.permutation(n)]
    assert ac1(walk) > 0.9                                     # the walk is strongly autocorrelated
    assert abs(ac1(perm)) < 0.2                                # permutation kills it
    assert ac1(surr) > 0.7                                     # surrogate keeps most of it

    # (3) a statistic sensitive to DETERMINISTIC structure beyond the spectrum: nonlinear predictability. A
    #     deterministic map (e.g. the logistic/tent map) is highly predictable from its own past even though its
    #     BROADBAND spectrum looks like noise -- so a nearest-neighbour predictor's skill is high on the real
    #     signal but collapses on phase-randomized surrogates (which share the spectrum but destroy the determinism).
    def predictability(v, k=3):
        """One-step nearest-neighbour prediction skill: for each point, find the past delay-vector most similar to
        the current one and predict the next value from it; return correlation of predictions to truth. High for a
        deterministic map, ~0 for linear-stochastic (whose future is not determined by its past pattern)."""
        v = np.asarray(v, float)
        m = len(v)
        emb = np.array([v[i:i + k] for i in range(m - k - 1)])
        nxt = v[k:m - 1]
        if len(emb) < 20:
            return 0.0
        preds, truth = [], []
        # split: build the library on the first half, predict the second (no peeking).
        half = len(emb) // 2
        lib, lib_next = emb[:half], nxt[:half]
        for i in range(half, len(emb)):
            d = np.sum((lib - emb[i]) ** 2, axis=1)
            j = int(np.argmin(d))
            preds.append(lib_next[j]); truth.append(nxt[i])
        preds, truth = np.array(preds), np.array(truth)
        if preds.std() < 1e-9 or truth.std() < 1e-9:
            return 0.0
        return float(np.corrcoef(preds, truth)[0, 1])

    # the logistic map at r=3.9: deterministic chaos -- broadband spectrum but perfectly determined by its past.
    logistic = np.zeros(n)
    logistic[0] = 0.4
    for i in range(1, n):
        logistic[i] = 3.9 * logistic[i - 1] * (1 - logistic[i - 1])
    logistic -= logistic.mean()
    znl = surrogate_zscore(logistic, predictability, n_surrogates=40, seed=2)
    # a linear-Gaussian AR(1) process: predictable only as far as its autocorrelation, which the surrogate KEEPS,
    # so its predictability does NOT stand out above the surrogate null.
    ar = np.zeros(n)
    for i in range(1, n):
        ar[i] = 0.8 * ar[i - 1] + rng.normal()
    zlin = surrogate_zscore(ar, predictability, n_surrogates=40, seed=3)
    assert znl["z"] > 3.0, znl                                 # deterministic chaos beats its phase-random null
    assert znl["z"] > zlin["z"], (znl["z"], zlin["z"])         # and beats it by more than a linear process does

    # (4) determinism.
    assert np.array_equal(phase_randomize(walk, seed=5), phase_randomize(walk, seed=5))

    # (5) AAFT preserves the EXACT amplitude distribution (fat tails) where basic phase_randomize Gaussianizes it.
    #     Build a fat-tailed signal (Student-t-like: cubed Gaussian) and compare marginals.
    heavy = rng.standard_normal(n) ** 3                       # a fat-tailed signal
    aaft = amplitude_adjusted_surrogate(heavy, seed=6)
    basic = phase_randomize(heavy, seed=6)
    # AAFT's values are a PERMUTATION of the original -> identical sorted amplitudes (exact histogram match).
    assert np.allclose(np.sort(aaft), np.sort(heavy)), "AAFT must preserve the exact amplitude distribution"
    # basic phase-randomization pulls the kurtosis toward Gaussian (3); AAFT keeps the original's high kurtosis.
    def kurt(v):
        v = (v - v.mean()) / (v.std() + 1e-12)
        return float(np.mean(v ** 4))
    k_orig, k_aaft, k_basic = kurt(heavy), kurt(aaft), kurt(basic)
    assert abs(k_aaft - k_orig) < abs(k_basic - k_orig), (k_orig, k_aaft, k_basic)   # AAFT closer to the truth
    # AAFT still approximately preserves the autocorrelation (the point of a surrogate, vs a plain shuffle).
    assert ac1(amplitude_adjusted_surrogate(walk, seed=7)) > 0.4

    # (6) IAAFT matches the spectrum BETTER than AAFT while keeping the EXACT amplitude distribution. Use a
    #     coloured fat-tailed signal (fat tails + autocorrelation), where AAFT's approximate spectrum shows.
    coloured_heavy = np.cumsum(rng.standard_normal(n) ** 3)   # fat-tailed increments + strong autocorrelation
    coloured_heavy -= coloured_heavy.mean()
    ia = iaaft_surrogate(coloured_heavy, n_iter=100, seed=8)
    aa = amplitude_adjusted_surrogate(coloured_heavy, seed=8)
    tgt_mag = np.abs(np.fft.rfft(coloured_heavy))
    err_ia = np.linalg.norm(np.abs(np.fft.rfft(ia)) - tgt_mag) / (np.linalg.norm(tgt_mag) + 1e-12)
    err_aa = np.linalg.norm(np.abs(np.fft.rfft(aa)) - tgt_mag) / (np.linalg.norm(tgt_mag) + 1e-12)
    assert err_ia < err_aa, (err_ia, err_aa)                  # IAAFT's spectrum is closer to the target
    assert np.allclose(np.sort(ia), np.sort(coloured_heavy))  # and it still keeps the EXACT amplitude distribution
    assert np.array_equal(iaaft_surrogate(coloured_heavy, seed=9), iaaft_surrogate(coloured_heavy, seed=9))  # deterministic

    # (7) SIGN-FLIP: the directional null. |surrogate| == |x| EXACTLY, so magnitude clustering (volatility
    #     clustering) is untouched while the direction channel is destroyed -- exactly what a claim about
    #     DIRECTION should be measured against.
    sf = sign_flip(walk, seed=1)
    assert np.array_equal(np.abs(sf), np.abs(walk)), "sign_flip must preserve every magnitude exactly"
    assert ac1(walk) > 0.9 and abs(ac1(sf)) < 0.2                      # signed structure destroyed...
    assert abs(ac1(np.abs(sf)) - ac1(np.abs(walk))) < 1e-12            # ...magnitude structure identical
    # KEPT NEGATIVE, pinned: a MAGNITUDE-ONLY statistic is constant across sign-flip surrogates, so its null has
    # zero spread and any z computed from it is a divide-by-nothing. A naive harness reports that as a huge z.
    mag_null = np.array([float(np.mean(np.abs(s))) for s in surrogate_ensemble(walk, "sign_flip", n=16, seed=0)])
    assert mag_null.std() < 1e-12, ("sign_flip is the WRONG null for a magnitude-only statistic -- expected a "
                                    "degenerate (zero-spread) null, got std=%g" % mag_null.std())

    # (8) IID-SHUFFLE: exact histogram, all ordering gone.
    ii = iid_shuffle(walk, seed=1)
    assert np.allclose(np.sort(ii), np.sort(walk)) and abs(ac1(ii)) < 0.2

    # (9) BLOCK-SHUFFLE: structure shorter than `block` survives, longer is destroyed; histogram exact.
    bs = block_shuffle(walk, 64, seed=1)
    assert np.allclose(np.sort(bs), np.sort(walk))
    assert ac1(bs) > 0.9, ac1(bs)                                      # within-block smoothness survives (0.98)
    # block=1 IS iid_shuffle, bit-for-bit at the same seed -- the documented degenerate case, pinned so the
    # two paths can never drift apart.
    assert np.array_equal(block_shuffle(walk, 1, seed=3), iid_shuffle(walk, seed=3))
    # KEPT NEGATIVE, pinned: the block JOINS are discontinuities the real signal never had. A jump detector sees
    # roughly len(x)/block fake events per surrogate (0 -> 10 here) and reads significant in the WRONG direction.
    thr = 5.0 * float(np.std(np.diff(walk)))
    jumps_real = int(np.sum(np.abs(np.diff(walk)) > thr))
    jumps_bs = int(np.sum(np.abs(np.diff(bs)) > thr))
    assert jumps_real == 0 and jumps_bs >= 5, (jumps_real, jumps_bs)

    # (10) make_surrogate: names resolve, callables pass through, and every refusal NAMES the valid options.
    assert np.array_equal(make_surrogate("sign_flip")(walk, 1), sf)
    assert make_surrogate(len) is len                                   # a callable is returned unchanged
    for bad, needle in ((("no_such_kind",), "unknown surrogate"), (("block_shuffle",), "requires block")):
        try:
            make_surrogate(*bad)
            raise AssertionError("expected ValueError for %r" % (bad,))
        except ValueError as e:
            assert needle in str(e) and "block_shuffle" in str(e), str(e)

    # (11) surrogate_ensemble: a generator of n DISTINCT, reproducible members (memory-light for long series).
    ens = list(surrogate_ensemble(walk, "phase", n=5, seed=7))
    assert len(ens) == 5 and not np.array_equal(ens[0], ens[1])
    assert all(np.array_equal(a, b) for a, b in zip(ens, surrogate_ensemble(walk, "phase", n=5, seed=7)))
    # surrogate_batch is the SAME members materialised -- the JSON-boundary sibling must never drift from the
    # generator it wraps, so the identity is pinned member-for-member rather than merely by shape.
    batch = surrogate_batch(walk, "phase", n=5, seed=7)
    assert batch.shape == (5, len(walk)) and all(np.array_equal(batch[i], ens[i]) for i in range(5))

    # (12) TREV / time_arrow_test: a slow-rise/instant-fall sawtooth is blatantly time-IRREVERSIBLE (its falls
    #      have a different shape from its rises, so the odd moment of the lagged difference is large and
    #      negative); a linear-Gaussian AR(1) is time-REVERSIBLE and must not flag.
    t_idx = np.arange(n)
    saw = (t_idx % 50) / 50.0
    saw = saw - saw.mean()
    arrow_saw = time_arrow_test(saw, n_surrogates=60, seed=1)
    arrow_ar = time_arrow_test(ar, n_surrogates=60, seed=1)
    assert trev(saw) < -1.0, trev(saw)                                  # measured -6.93
    assert arrow_saw["z"] < -10.0, arrow_saw                            # measured z = -26.9
    # The AR(1) control lands at z=+2.28 (p=0.049) in THIS realisation -- a genuinely time-reversible process
    # throwing a 2-sigma reading on one draw. Kept loud rather than reseeded away: it is the in-file reminder
    # that a single z is not a result, which is why split_half and bh_fdr exist next door.
    assert abs(arrow_ar["z"]) < 3.0, arrow_ar
    assert arrow_saw["p"] < arrow_ar["p"]
    assert abs(trev(np.ones(n))) == 0.0                                 # a constant series has no arrow
    try:
        trev(saw, lag=0)
        raise AssertionError("expected ValueError for lag=0")
    except ValueError as e:
        assert "lag" in str(e)

    print("holographic_surrogate selftest OK (surrogate preserves the power spectrum exactly; permutation kills "
          "the lag-1 autocorrelation (%.2f->%.2f) while the surrogate keeps it (%.2f); the logistic map's "
          "predictability scores z=%.1f against its phase-randomized null vs z=%.1f for a linear AR(1); AAFT keeps "
          "fat tails (kurtosis %.1f vs basic %.1f, truth %.1f); IAAFT matches the spectrum better than AAFT "
          "(rel err %.3f vs %.3f) with the exact distribution; deterministic)"
          % (ac1(walk), ac1(perm), ac1(surr), znl["z"], zlin["z"], k_aaft, k_basic, k_orig, err_ia, err_aa))
    print("  + directional/scale nulls: sign_flip keeps every magnitude exactly (|abs| ac1 %.3f both sides) while "
          "killing signed ac1 (%.3f->%.3f) -- and its null is DEGENERATE for a magnitude-only statistic "
          "(std %.1e, kept negative); block_shuffle(64) keeps within-block structure (ac1 %.3f) but injects "
          "%d fake jumps at the joins (kept negative), and block=1 is bit-identical to iid_shuffle; "
          "time arrow: sawtooth trev=%.2f z=%.1f vs AR(1) z=%.2f (p %.3f vs %.3f)"
          % (ac1(np.abs(walk)), ac1(walk), ac1(sf), mag_null.std(), ac1(bs), jumps_bs,
             trev(saw), arrow_saw["z"], arrow_ar["z"], arrow_saw["p"], arrow_ar["p"]))


if __name__ == "__main__":
    _selftest()
